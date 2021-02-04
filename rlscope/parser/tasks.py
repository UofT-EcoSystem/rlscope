"""
Define ``--task`` arguments for the ``rls-run`` command.

RL-Scope uses the luigi DAG execution framework internally for
running scripts for processing trace files and creating plots.
This allows us to have a single command (``rls-run``) for various tasks,
and allows specifying task specific command line arguments
(e.g., ``--OverlapStackedBarTask-rotation 15``)
Every class name ending in "Task" can be used as a ``--task`` argument.

For luigi documetation (e.g., to add your own task), see:
https://luigi.readthedocs.io/en/stable/index.html
"""
import luigi

from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler import rlscope_logging
import subprocess
import multiprocessing
import re
import pwd
import textwrap
import datetime
import pprint
import sys
import os

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.util import pprint_msg
from rlscope.parser.tfprof import TraceEventsParser
from rlscope.parser.pyprof import PythonProfileParser, PythonFlameGraphParser, PythonProfileTotalParser
from rlscope.parser.plot import TimeBreakdownPlot, PlotSummary, CombinedProfileParser, CategoryOverlapPlot, UtilizationPlot, HeatScalePlot, ConvertResourceOverlapToResourceSubplot, VennJsPlotter, SlidingWindowUtilizationPlot, CUDAEventCSVReader
from rlscope.parser.db import SQLParser, sql_input_path, GetConnectionPool
from rlscope.parser import db
from rlscope.parser.stacked_bar_plots import OverlapStackedBarPlot, CategoryTransitionPlot, TexMetrics
from rlscope.profiler.util import print_cmd
from rlscope.parser.cpu_gpu_util import UtilParser, UtilPlot, GPUUtilOverTimePlot, NvprofKernelHistogram, CrossProcessOverlapHistogram, NvprofTraces
from rlscope.parser.training_progress import TrainingProgressParser, ProfilingOverheadPlot
from rlscope.parser.extrapolated_training_time import ExtrapolatedTrainingTimeParser
from rlscope.parser.profiling_overhead import CallInterceptionOverheadParser, CUPTIOverheadParser, CUPTIScalingOverheadParser, CorrectedTrainingTimeParser, PyprofOverheadParser, TotalTrainingTimeParser, SQLOverheadEventsParser
from rlscope.parser.one_off_plot import GpuUtilExperiment
from rlscope import py_config

from rlscope.parser.common import *

PARSER_KLASSES = [PythonProfileParser, PythonFlameGraphParser, PlotSummary, TimeBreakdownPlot, CategoryOverlapPlot, UtilizationPlot, HeatScalePlot, TraceEventsParser, SQLParser]
PARSER_NAME_TO_KLASS = dict((ParserKlass.__name__, ParserKlass) \
                            for ParserKlass in PARSER_KLASSES)

# Make this pipeline:
# - rls-run --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules SQLParser
# |-> rls-run --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type CategoryOverlap
# |-> rls-run --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type ResourceOverlap
# |-> rls-run --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type OperationOverlap
# |-> rls-run --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type ResourceSubPlot
#
# Wrap running all the existing classes with a class that outputs a "done" marker file.
# That way, we don't need to modify the classes to use luigi's "with self.output().open('w') as outfile:".

# List of runnable tasks for rls-run.
# Used for generating command-line usage help.

# RLSCOPE_TASKS = ...
# NOT_RUNNABLE_TASKS = ...
def get_NOT_RUNNABLE_TASKS():
    return [IMLTask, IMLTaskDB, _UtilizationPlotTask]

def get_RLSCOPE_TASKS():
    global NOT_RUNNABLE_TASKS
    rlscope_tasks = set()
    for name, cls in globals().items():
        if isinstance(cls, type) and issubclass(cls, IMLTask) and cls not in NOT_RUNNABLE_TASKS:
            rlscope_tasks.add(cls)
    return rlscope_tasks

def get_username():
    return pwd.getpwuid(os.getuid())[0]

param_visible_overhead = luigi.BoolParameter(
    description=textwrap.dedent("""\
        If true, make profiling overhead visible during rlscope-drill.  
        If false (and calibration files are given), then subtract overhead 
        making it 'invisible' in rlscope-drill"""),
    default=False,
    parsing=luigi.BoolParameter.EXPLICIT_PARSING)

param_postgres_password = luigi.Parameter(description="Postgres password; default: env.PGPASSWORD", default=None)
param_postgres_user = luigi.Parameter(description="Postgres user", default=None)
param_postgres_host = luigi.Parameter(description="Postgres host", default=None)

param_debug = luigi.BoolParameter(description="debug")
param_debug_single_thread = luigi.BoolParameter(description=textwrap.dedent("""
        Run any multiprocessing stuff using a single thread for debugging.
        """))
param_debug_perf = luigi.BoolParameter(description=textwrap.dedent("""
        Collect some coarse-grained timing info to help debug performance issues during trace analysis.
        """))
param_debug_memoize = luigi.BoolParameter(description=textwrap.dedent("""
        Memoize reading/generation of files to accelerate develop/test code iteration.
        """))
param_line_numbers = luigi.BoolParameter(description="Show line numbers and timestamps in RL-Scope logging messages")

# Luigi's approach of discerning None from
class NoValueType:
    def __init__(self):
        pass
no_value = NoValueType()

def _get_param(desc, default=no_value):
    if default == no_value:
        param = luigi.Parameter(description=desc)
    else:
        param = luigi.Parameter(description=desc, default=default)
    return param

CALIBRATION_OPTS = [
    'cupti_overhead_json',
    'LD_PRELOAD_overhead_json',
    'python_annotation_json',
    'python_clib_interception_tensorflow_json',
    'python_clib_interception_simulator_json',
]

def get_param_cupti_overhead_json(default=no_value):
    desc = "Calibration: mean per-CUDA API CUPTI overhead when GPU activities are recorded (see: CUPTIOverheadTask)"
    return _get_param(desc=desc, default=default)
def get_param_LD_PRELOAD_overhead_json(default=no_value):
    desc = "Calibration: mean overhead for intercepting CUDA API calls with LD_PRELOAD  (see: CallInterceptionOverheadTask)"
    return _get_param(desc=desc, default=default)
def get_param_python_clib_interception_overhead_json(clib, default=no_value):
    desc = "Calibration: mean for {clib} Python->C++ interception overhead (see: PyprofOverheadTask)".format(
        clib=clib)
    return _get_param(desc=desc, default=default)

def get_param_python_annotation_overhead_json(default=no_value):
    desc = "Calibration: means for operation annotation overhead (see: PyprofOverheadTask)"
    return _get_param(desc=desc, default=default)

def calibration_files_present(task):
    return all(getattr(task, opt) is not None for opt in CALIBRATION_OPTS)

# NOTE: this params REQUIRE a value (since no default is present)
param_cupti_overhead_json = get_param_cupti_overhead_json()
param_cupti_overhead_json_optional = get_param_cupti_overhead_json(default=None)
param_LD_PRELOAD_overhead_json = get_param_LD_PRELOAD_overhead_json()
param_LD_PRELOAD_overhead_json_optional = get_param_LD_PRELOAD_overhead_json(default=None)

param_python_annotation_json = get_param_python_annotation_overhead_json()
param_python_annotation_json_optional = get_param_python_annotation_overhead_json(default=None)
param_python_clib_interception_tensorflow_json = get_param_python_clib_interception_overhead_json(clib='TensorFlow')
param_python_clib_interception_tensorflow_json_optional = get_param_python_clib_interception_overhead_json(clib='TensorFlow', default=None)
param_python_clib_interception_simulator_json = get_param_python_clib_interception_overhead_json(clib='Simulator')
param_python_clib_interception_simulator_json_optional = get_param_python_clib_interception_overhead_json(clib='Simulator', default=None)

class IMLTask(luigi.Task):
    rlscope_directory = luigi.Parameter(description="Location of trace-files")
    debug_memoize = luigi.BoolParameter(description="If true, memoize partial results for quicker runs", default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    def output(self):
        return luigi.LocalTarget(self._done_file)

    @property
    def _done_file(self):
        """
        e.g.

        <--rlscope-directory>/SQLParserTask.task
        """
        return "{dir}/{name}.task".format(
            dir=self.rlscope_directory, name=self._task_name)

    @property
    def _task_name(self):
        return self.__class__.__name__

    def mark_done(self, start_t, end_t):
        task_file = self.output()
        return self._mark_done(task_file, start_t, end_t)

    def _mark_done(self, task_file, start_t, end_t):
        # logger.info("MARK DONE @ {path}".format(path=self.output()))
        if self.skip_output:
            logger.info("> Skipping output={path} for task {name}".format(
                path=self._done_file,
                name=self._task_name))
            return
        with task_file.open('w') as f:
            print_cmd(cmd=sys.argv, files=[f])
            delta = end_t - start_t
            minutes, seconds = divmod(delta.total_seconds(), 60)
            print(textwrap.dedent("""\
            > Started running at {start}
            > Ended running at {end}
            > Took {min} minutes and {sec} seconds.
            """.format(
                start=start_t,
                end=end_t,
                min=minutes,
                sec=seconds,
            )), file=f)

    def rlscope_run(self):
        raise NotImplementedError("{klass} must override rlscope_run()".format(
            klass=self.__class__.__name__))

    def _run_with_timer(self):
        start_t = datetime.datetime.now()
        self.rlscope_run()
        end_t = datetime.datetime.now()

        self.mark_done(start_t, end_t)

    def run(self):
        setup_logging_from_self(self)
        self._run_with_timer()


class IMLTaskDB(IMLTask):
    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    @property
    def db_path(self):
        return sql_input_path(self.rlscope_directory)

    @property
    def maxconn(self):
        """
        Maximum number of concurrent postgres connection.

        By default (and to detect connection leaks), only allow at most 1 postgres connection.
        If a task/parser wants more, just override this with the exact amount.
        """
        return 1

    def run(self):
        setup_logging_from_self(self)
        if self.maxconn > 0 and db.USE_CONNECTION_POOLING:
            # Create a postgres connection pool that allows at most maxconn connections
            # in the entire python process.
            with GetConnectionPool(conn_kwargs=dict(
                db_path=self.db_path,
                host=self.postgres_host,
                user=self.postgres_user,
                password=self.postgres_password,
            ), maxconn=self.maxconn, new_process=True) as pool:
                self._run_with_timer()
        else:
            # DON'T create a postgres connection pool.
            self._run_with_timer()


class SQLParserTask(IMLTaskDB):
    def requires(self):
        return []

    def rlscope_run(self):
        self.sql_parser = SQLParser(
            directory=self.rlscope_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )
        self.sql_parser.run()

class SQLOverheadEventsTask(IMLTaskDB):
    """
    How RL-Scope handles subtracting CPU-overhead:

    High-level idea:

        - If we want CPU-overhead to be invisible, REMOVE CPU-overhead:
          (GPU, CPU, CPU-overhead) -> (GPU)
          (GPU, CPU-overhead)      -> (GPU)
          (CPU-overhead)           -> IGNORE
          (CPU, CPU-overhead)      -> IGNORE

        - If we want CPU-overhead to be visible, KEEP CPU-overhead category:
          (GPU, CPU-overhead) -> (GPU, CPU-overhead)
          (CPU-overhead)      -> (CPU-overhead)

    Low-level implementation:

        - During ResourceOverlap / ResourceSubplot / OperationOverlap:
          - If making CPU-overhead visible:
            Pre-reduce:
            - Count constants.CATEGORIES_PROF as constants.CATEGORY_CPU
          - If making CPU-overhead invisible:
            Pre-reduce:
            - Keep constants.CATEGORIES_PROF as-is in CategoryKey.non_ops
              ( we cannot remove yet, since we need to know how to subtract overlap )
            Post-reduce:
            - Remove constants.CATEGORIES_PROF from CategoryKey.non_ops, delete "empty" keys

        - During CategoryOverlap:
          - If making CPU-overhead visible:
            Pre-reduce:
            - Keep category (could be in constants.CATEGORIES_PROF, constants.CATEGORIES_CPU, constants.CATEGORY_GPU)
            Post-reduce:
            - Keep category
          - If making CPU-overhead invisible:
            Pre-reduce:
            - Keep constants.CATEGORIES_PROF as-is in CategoryKey.non_ops
              ( we cannot remove yet, since we need to know how to subtract overlap )
            Post-reduce:
            - Remove constants.CATEGORIES_PROF from CategoryKey.non_ops, delete "empty" keys

         NOTE: only difference of CategoryOverlap with the rest is that we "keep" low-level category.
         CPU-overhead should show up at the CategoryOverlap level (i.e. Python, CUDA API C, ...).
         CPU-overhead here is one of the constants.CATEGORIES_PROF (e.g. constants.CATEGORY_PROF_CUPTI).
    """
    # NOTE: accept calibration JSON files as parameters instead of as DAG dependencies
    # to allow using calibration files generated from a separate workload.
    # TODO: we still need to investigate whether calibration using a separate workload generalizes properly.
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    def requires(self):
        return [
            mk_SQLParserTask(self),
        ]

    def rlscope_run(self):
        # TODO: implement SQLOverheadEventsParser to insert overhead events.
        raise NotImplementedError("Use cpp code, not old python implementation... (not maintained anymore)")
        self.sql_overhead_events_parser = SQLOverheadEventsParser(
            directory=self.rlscope_directory,
            cupti_overhead_json=self.cupti_overhead_json,
            LD_PRELOAD_overhead_json=self.LD_PRELOAD_overhead_json,
            # pyprof_overhead_json=self.pyprof_overhead_json,

            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,

            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )
        self.sql_overhead_events_parser.run()

class _UtilizationPlotTask(IMLTaskDB):
    visible_overhead = param_visible_overhead
    # Q: Is there a good way to choose this automatically...?
    # PROBLEM: n_workers should be the size of the pool...but how should we choose the number of splits to make...?
    # If we choose n_splits == n_workers, then the ENTIRE event trace will STILL get swallowed into memory....
    # Ideally, we should choose n_split such that n_workers*( [total event trace size]/n_splits ) << [total memory size]
    # However I don't see an obvious way of calculating [total event trace size]...
    # I guess we could "configure/calibrate" it, but that's a pain in the butt.
    # IDEAL: specify a time interval like 10 seconds, such that a chunk is chosen such that it takes ~ 10 seconds to process;
    # that would ensure that we don't waste ALL our time on synchronization/serialization.
    # SOLUTION: just manually add a reasonable default value on our system, and make it configurable for future uses.
    n_workers = luigi.IntParameter(
        description="How many threads to simultaneously run overlap-computation; if --debug-single-thread, uses 1",
        default=multiprocessing.cpu_count())
    # n_splits = luigi.IntParameter(description=textwrap.dedent("""
    # Overlap-computation is parallelized by \"splitting\" into n-splits, and processing using a pool of --n-workers thread;
    # when processing a trace, how many splits should we make?
    #
    # Default if not provided: --n-workers
    # NOTE: this reads the whole event-trace into memory.
    # """), default=None)
    # This is easier to tune than --n-splits
    # (the effect of --n-splits is workload dependent)
    events_per_split = luigi.IntParameter(
        description=textwrap.dedent("""
        Approximately how many events per split should there be? 
        Assuming events_per_split is > 10x the number of events per iteration, 
        events_per_split should roughly linearly correlate with memory usage and processing time of the split.
        
        Minimum: 1000
        """),

        # metric                       per_sec           raw
        # events  15816.74161747038 events/sec  17844 events")
        # default=10000,

        # NOTE: at events_per_split=50000, we still maximize events/sec.
        # We want to choose the smallest events_per_split that still maximizes events/sec,
        # since smaller events per split means a larger number of splits
        # => greater opportunity for processing splits in parallel.

        # metric                       per_sec           raw
        # events  25679.08856381489 events/sec  94230 events")
        default=50000,

        # metric                        per_sec            raw
        # events  28956.472301404858 events/sec  199384 events")
        # default=100000,

        # metric                       per_sec             raw
        # events  28452.14408169205 events/sec  1246530 events")
        # default=1000000,
    )

    # NOT optional (to ensure we remember to do overhead correction).
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    def requires(self):
        return [
            mk_SQL_tasks(self),
        ]

    def rlscope_run(self):
        # If calibration files aren't provided, then overhead will be visible.
        # If calibration files are provided, we can subtract it:
        #   Use whatever they want based on --visible-overhead
        if not calibration_files_present(task=self):
            logger.warning(
                ("Calibration files aren't all present; we cannot subtract overhead without "
                 "all of these options present: {msg}").format(
                    msg=pprint_msg(CALIBRATION_OPTS),
                ))
            visible_overhead = True
        else:
            visible_overhead = self.visible_overhead

        self.sql_parser = UtilizationPlot(
            overlap_type=self.overlap_type,
            directory=self.rlscope_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            visible_overhead=visible_overhead,
            n_workers=self.n_workers,
            events_per_split=self.events_per_split,
            password=self.postgres_password,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
            debug_perf=self.debug_perf,
            debug_memoize=self.debug_memoize,
        )
        self.sql_parser.run()

class ResourceOverlapTask(_UtilizationPlotTask):
    overlap_type = 'ResourceOverlap'

class CategoryOverlapTask(_UtilizationPlotTask):
    overlap_type = 'CategoryOverlap'

class ResourceSubplotTask(_UtilizationPlotTask):
    overlap_type = 'ResourceSubplot'

class OperationOverlapTask(_UtilizationPlotTask):
    overlap_type = 'OperationOverlap'

def _mk(kwargs, TaskKlass):
    task_kwargs = keep_task_kwargs(kwargs, TaskKlass)
    if kwargs.get('debug', False):
        logger.info("{klass} kwargs: {msg}".format(
            klass=TaskKlass.__name__,
            msg=pprint_msg(task_kwargs)))
    return TaskKlass(**task_kwargs)

# ASSUME: rls-analyze is on $PATH;
# i.e. they have run:
# $ cd $RLSCOPE_DIR
# $ source source_me.sh
CPP_ANALYZE_BIN = 'rls-analyze'
class RLSAnalyze(IMLTask):

    # NOT optional (to ensure we remember to do overhead correction).
    # rlscope_directory = luigi.Parameter(description="Location of trace-files")
    mode = luigi.ChoiceParameter(
        choices=[
            'overlap',
            'gpu_hw',
        ],
        default='overlap',
        description=textwrap.dedent("""\
        overlap:
          Calculate the time spent in different parts of the RL software stack, scoped to high-level user operations.
          i.e.
          CPU - TensorFlow C++, Simulator, Python
          GPU - kernel executions
          
        gpu_hw:
          Calculate GPU hardware counters scoped to high-level user operations.
        """).rstrip())
    output_directory = luigi.Parameter(description="Directory to output analysis files; default = --rlscope-directory", default=None)

    # optional.
    cupti_overhead_json = param_cupti_overhead_json_optional
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json_optional
    python_annotation_json = param_python_annotation_json_optional
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json_optional
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json_optional

    # visible_overhead = param_visible_overhead

    @property
    def _done_file(self):
        if self.output_directory is not None:
            out_dir = self.output_directory
        else:
            out_dir = self.rlscope_directory
        return "{dir}/{name}.task".format(
            dir=out_dir, name=self._task_name)

    def overhead_files(self):
        overhead_files = dict()
        overhead_files['cupti_overhead_json'] = self.cupti_overhead_json
        overhead_files['LD_PRELOAD_overhead_json'] = self.LD_PRELOAD_overhead_json
        overhead_files['python_annotation_json'] = self.python_annotation_json
        overhead_files['python_clib_interception_tensorflow_json'] = self.python_clib_interception_tensorflow_json
        overhead_files['python_clib_interception_simulator_json'] = self.python_clib_interception_simulator_json
        return overhead_files

    def check_overhead_files(self):
        overhead_files = self.overhead_files()
        if any(json_path is None for json_path in overhead_files.values()):
            if not all(json_path is None for json_path in overhead_files.values()):
                def get_optname(attr):
                    opt = attr
                    opt = re.sub('_', '-', opt)
                    return f"--{opt}"
                # missing_overhead_opts = dict((get_optname(attr), json_path) for attr, json_path in overhead_files.items() if json_path is None)
                missing_overhead_opts = sorted(list(get_optname(attr) for attr, json_path in overhead_files.items() if json_path is None))
                raise RuntimeError(textwrap.dedent("""\
                You must either provide ALL overhead correction files or NO overhead correction files.
                Missing overhead correction files:
                {msg}
                """.format(msg=textwrap.indent(pprint.pformat(missing_overhead_opts), prefix='  '))))
            return False
        return True

    def rlscope_run(self):
        cmd = [CPP_ANALYZE_BIN]
        cmd.extend([
            '--rlscope_directory', self.rlscope_directory,
            '--mode', self.mode,
        ])
        if self.output_directory is not None:
            cmd.extend([
                '--output_directory', self.output_directory,
            ])
        has_overhead_files = self.check_overhead_files()
        if has_overhead_files:
            cmd.extend([
                '--cupti_overhead_json', self.cupti_overhead_json,
                '--LD_PRELOAD_overhead_json', self.LD_PRELOAD_overhead_json,
                '--python_annotation_json', self.python_annotation_json,
                '--python_clib_interception_tensorflow_json', self.python_clib_interception_tensorflow_json,
                '--python_clib_interception_simulator_json', self.python_clib_interception_simulator_json,
            ])
        if self.debug:
            cmd.extend(['--debug'])
        print_cmd(cmd)
        subprocess.check_call(cmd)

# class All(IMLTaskDB):
class All(IMLTask):
    # Don't output All.task, so that if (for example)
    # ResourceOverlapTask.task is deleted, we will still re-run it.
    skip_output = True

    # NOT optional (to ensure we remember to do overhead correction).
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    visible_overhead = param_visible_overhead

    def requires(self):
        kwargs = self.param_kwargs

        return [
            _mk(kwargs, RLSAnalyze),
            _mk(kwargs, HeatScaleTask),
            _mk(kwargs, VennJsPlotTask),
        ]

    def rlscope_run(self):
        pass


class HeatScaleTask(IMLTask):
    # step_sec=1.,
    # pixels_per_square=10,
    # decay=0.99,
    # def requires(self):
    #     return [
    #         # NOTE: we DON'T need overhead events.
    #         mk_SQLParserTask(self),
    #     ]

    # NOT optional (to ensure we remember to do overhead correction).
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    def requires(self):
        kwargs = self.param_kwargs

        return [
            _mk(kwargs, RLSAnalyze),
        ]

    def rlscope_run(self):
        self.heat_scale = HeatScalePlot(
            directory=self.rlscope_directory,
            # host=self.postgres_host,
            # user=self.postgres_user,
            # password=self.postgres_password,
            debug=self.debug,
        )
        self.heat_scale.run()

class ConvertResourceOverlapToResourceSubplotTask(IMLTask):
    # step_sec=1.,
    # pixels_per_square=10,
    # decay=0.99,
    # def requires(self):
    #     return [
    #         # NOTE: we DON'T need overhead events.
    #         mk_SQLParserTask(self),
    #     ]

    # NOT optional (to ensure we remember to do overhead correction).
    # cupti_overhead_json = param_cupti_overhead_json
    # LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    # python_annotation_json = param_python_annotation_json
    # python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    # python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    # def requires(self):
    #     kwargs = self.param_kwargs
    #
    #     return [
    #         _mk(kwargs, RLSAnalyze),
    #     ]

    def rlscope_run(self):
        self.converter = ConvertResourceOverlapToResourceSubplot(
            directory=self.rlscope_directory,
            debug=self.debug,
        )
        self.converter.run()

class TraceEventsTask(luigi.Task):
    rlscope_directory = luigi.Parameter(description="Location of trace-files")
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    filter_op = luigi.BoolParameter(description="If true, JUST show --op-name events not other operations", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    overlaps_event_id = luigi.IntParameter(description="show events that overlap with this event (identified by its event_id)", default=None)
    op_name = luigi.Parameter(description="operation name (e.g. q_forward)", default=None)
    process_name = luigi.Parameter(description="show events belonging to this process", default=None)
    start_usec = luigi.FloatParameter(description="Show events whose start-time is >= start_usec", default=None)
    end_usec = luigi.FloatParameter(description="Show events whose end-time is <= end_usec", default=None)

    # Needed by mk_SQL_tasks
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    skip_output = False

    def requires(self):
        return [
            mk_SQL_tasks(self),
        ]

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        self.dumper = TraceEventsParser(
            directory=self.rlscope_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            filter_op=self.filter_op,
            overlaps_event_id=self.overlaps_event_id,
            op_name=self.op_name,
            process_name=self.process_name,
            # process_op_nest=False,
            start_usec=self.start_usec,
            end_usec=self.end_usec,
        )
        self.dumper.run()

param_x_type = luigi.ChoiceParameter(choices=OverlapStackedBarPlot.SUPPORTED_X_TYPES,
                                     default='rl-comparison',
                                     description=textwrap.dedent("""\
                               What should we show on the x-axis and title of the stacked bar-plot?
                               
                               algo-comparison:
                                 You want to make a plot that supports a statement across all RL algorithms, 
                                 when training a specific environment.
                                 
                                                 Comparing algorithms 
                                              when training {environment}
                                            |
                                    Time    |
                                  breakdown |
                                            |
                                            |---------------------------
                                             algo_1,     ...,     algo_n
                                                     RL algorithm
                                          
                                 
                                 i.e. your experiments involve:
                                 - SAME environment
                                 - DIFFERENT algorithms
                                 
                               env-comparison:
                                 You want to make a plot that supports a statement across all environments, 
                                 when training a specific RL algorithm.
                                 
                                               Comparing environments
                                              when training {algorithm}
                                            |
                                    Time    |
                                  breakdown |
                                            |
                                            |---------------------------
                                             env_1,      ...,      env_n
                                                     Environment 
                                          
                                 
                                 i.e. your experiments involve:
                                 - DIFFERENT environment
                                 - SAME algorithm
                               
                               rl-comparison:
                                 You want to make a plot that supports a statement across ALL RL workloads;
                                 i.e. across all (algorithm, environment) combinations.
                                 
                                                      Comparing RL workloads
                                            |
                                    Time    |
                                  breakdown |
                                            |
                                            |---------------------------------------
                                             (algo_1, env_1),  ...,  (algo_n, env_n)
                                                   (RL algorithm, Environment)
                                 
                                 i.e. your experiments involve:
                                 - DIFFERENT environments
                                 - DIFFERENT algorithms
                               """))


class UtilTask(luigi.Task):
    rlscope_directories = luigi.ListParameter(description="Multiple --rlscope-directory entries for finding overlap_type files: *.venn_js.js")
    directory = luigi.Parameter(description="Output directory", default=".")
    suffix = luigi.Parameter(description="Add suffix to output files: MachineGPUUtil.{suffix}.{ext}", default=None)
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    # optional.
    cupti_overhead_json = param_cupti_overhead_json_optional
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json_optional
    python_annotation_json = param_python_annotation_json_optional
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json_optional
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json_optional

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = UtilParser(**kwargs)
        self.dumper.run()

class UtilPlotTask(luigi.Task):
    csv = luigi.Parameter(description="Path to overall_machine_util.raw.csv [output from UtilTask]")
    directory = luigi.Parameter(description="Output directory", default=".")
    x_type = param_x_type
    y_title = luigi.Parameter(description="y-axis title", default=None)
    suffix = luigi.Parameter(description="Add suffix to output files: MachineGPUUtil.{suffix}.{ext}", default=None)

    # Plot attrs
    rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    # optional.
    cupti_overhead_json = param_cupti_overhead_json_optional
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json_optional
    python_annotation_json = param_python_annotation_json_optional
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json_optional
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json_optional

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = UtilPlot(**kwargs)
        self.dumper.run()

class TrainingProgressTask(luigi.Task):
    rlscope_directories = luigi.ListParameter(description="Multiple --rlscope-directory entries for finding overlap_type files: *.venn_js.js")
    directory = luigi.Parameter(description="Output directory", default=".")
    # suffix = luigi.Parameter(description="Add suffix to output files: TrainingProgress.{suffix}.{ext}", default=None)
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    baseline_config = luigi.Parameter(description="The baseline configuration to compare all others against; default: config_uninstrumented", default=None)
    ignore_phase = luigi.BoolParameter(description="Bug workaround: for training progress files that didn't record phase, just ignore it.", default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = TrainingProgressParser(**kwargs)
        self.dumper.run()

class ProfilingOverheadPlotTask(luigi.Task):
    csv = luigi.Parameter(description="Path to overall_machine_util.raw.csv [output from UtilTask]")
    directory = luigi.Parameter(description="Output directory", default=".")
    x_type = param_x_type
    y_title = luigi.Parameter(description="y-axis title", default='Total training time (seconds)')
    suffix = luigi.Parameter(description="Add suffix to output files: MachineGPUUtil.{suffix}.{ext}", default=None)
    stacked = luigi.BoolParameter(description="Make stacked bar-plot", default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    preset = luigi.Parameter(description="preset configuration for plot bar order and plot labels", default=None)

    # Plot attrs
    rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = ProfilingOverheadPlot(**kwargs)
        self.dumper.run()

class ExtrapolatedTrainingTimeTask(IMLTaskDB):
    dependency = luigi.Parameter(description="JSON file containing Hard-coded computational dependencies A.phase -> B.phase", default=None)
    algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    # x_type = param_x_type
    # y_title = luigi.Parameter(description="y-axis title", default='Total training time (seconds)')
    # suffix = luigi.Parameter(description="Add suffix to output files: MachineGPUUtil.{suffix}.{ext}", default=None)
    # stacked = luigi.BoolParameter(description="Make stacked bar-plot", default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # preset = luigi.Parameter(description="preset configuration for plot bar order and plot labels", default=None)
    #
    # # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    # width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    # height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    # Needed by mk_SQL_tasks
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    def requires(self):
        return [
            mk_SQL_tasks(self),
        ]

    def rlscope_run(self):
        kwargs = kwargs_from_task(self)
        assert 'directory' not in kwargs
        kwargs['directory'] = kwargs['rlscope_directory']
        del kwargs['rlscope_directory']
        # logger.info(pprint_msg({'kwargs': kwargs}))
        self.dumper = ExtrapolatedTrainingTimeParser(**kwargs)
        self.dumper.run()

class GeneratePlotIndexTask(luigi.Task):
    rlscope_directory = luigi.Parameter(description="Location of trace-files")
    # out_dir = luigi.Parameter(description="Location of trace-files", default=None)
    # replace = luigi.BoolParameter(description="debug", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    # # Needed by mk_SQL_tasks
    # cupti_overhead_json = param_cupti_overhead_json
    # LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    # python_annotation_json = param_python_annotation_json
    # python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    # python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    skip_output = False

    def requires(self):
        # Requires that traces files have been collected...
        # So, lets just depend on the SQL parser to have loaded everything.
        return [
            # mk_SQL_tasks(self),
        ]

    def output(self):
        # Q: What about --replace?  Conditionally include this output...?
        return [
            luigi.LocalTarget(_j(self.rlscope_directory, 'rlscope_plot_index_data.py')),
        ]

    def run(self):
        setup_logging_from_self(self)
        cmd = ['rls-generate-plot-index']
        cmd.extend(['--rlscope-directory', self.rlscope_directory])
        if self.debug:
            cmd.extend(['--debug'])
        print_cmd(cmd)
        subprocess.check_call(cmd)

param_hack_upper_right_legend_bbox_x = luigi.FloatParameter(description="matplotlib hack: add to x-position of upper right legend so it's outside the plot area", default=None)
param_xtick_expression = luigi.Parameter(description="Python expression to generate xtick labels for plot.  Expression has access to individual 'row' and entire dataframe 'df'", default=None)
class OverlapStackedBarTask(luigi.Task):
    rlscope_directories = luigi.ListParameter(description="Multiple --rlscope-directory entries for finding overlap_type files: *.venn_js.js")
    rlscope_config_directories = luigi.ListParameter(description="Multiple --rlscope-directory entries for finding rlscope_config.json files (needed for making uncorrected plots)", default=None)
    unins_rlscope_directories = luigi.ListParameter(description="Multiple --rlscope-directory entries for finding total uninstrumented training time (NOTE: every rlscope_directory should have an unins_rlscope_directory)")
    directory = luigi.Parameter(description="Output directory", default=".")
    xtick_expression = param_xtick_expression
    title = luigi.Parameter(description="Plot title", default=None)
    x_title = luigi.Parameter(description="x-axis title", default=None)
    x_order_by = luigi.Parameter(description="order x-field by this dataframe field", default=None)
    rotation = luigi.FloatParameter(description="x-axis title rotation", default=15.)
    hack_upper_right_legend_bbox_x = param_hack_upper_right_legend_bbox_x

    overlap_type = luigi.ChoiceParameter(choices=OverlapStackedBarPlot.SUPPORTED_OVERLAP_TYPES, description="What type of <overlap_type>*.venn_js.js files should we read from?")
    resource_overlap = luigi.ListParameter(description="What resources are we looking at for things like --overlap-type=OperationOverlap? e.g. ['CPU'], ['CPU', 'GPU']", default=None)
    operation = luigi.Parameter(description="What operation are we looking at for things like --overlap-type=CategoryOverlap? e.g. ['step'], ['sample_action']", default=None)
    training_time = luigi.BoolParameter(description="Plot a second y-axis with total training time", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    extrapolated_training_time = luigi.BoolParameter(description="Extrapolate total training time if full uninstrumented run is not present", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    detailed = luigi.BoolParameter(description="Provide detailed operation/category breakdown in a single view", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    remap_df = luigi.Parameter(description="Transform df pandas.DataFrame object; useful for remapping regions to new ones", default=None)
    y2_logscale = luigi.BoolParameter(description="Show training time y-axis in logscale", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    # For some reason, (ppo2, MinitaurBulletEnv-v0) only has:
    # - regions: [('sample_action',)]
    # Whereas, for ppo2 we expect:
    # - regions: [('compute_advantage_estimates',), ('optimize_surrogate',), ('sample_action',)]
    # TODO: re-run MinitaurBulletEnv-v0
    ignore_inconsistent_overlap_regions = luigi.BoolParameter(description="If *.venn_js.json overlap data have inconsistent overlap regions, just use files that agree the most and ignore the rest", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    skip_plot = luigi.BoolParameter(description="Don't plot *.png file; just output the *.csv file we WOULD plot", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    y_type = luigi.ChoiceParameter(choices=OverlapStackedBarPlot.SUPPORTED_Y_TYPES, default='percent',
                                   description=textwrap.dedent("""\
                                   What should we show on the y-axis of the stacked bar-plot?
                                   
                                   percent:
                                     Don't show total training time.
                                     Useful if you just want to show a percent breakdown using a partial trace of training.
                                     
                                   seconds:
                                     Show absolute training time on the y-axis.
                                     TODO: we should extrapolate total training time based on number of timestamps ran, 
                                     and number of timesteps that will be run.
                                   """))
    x_type = param_x_type
    # postgres_host = param_postgres_host
    # postgres_user = param_postgres_user
    # postgres_password = param_postgres_password
    show_title = luigi.BoolParameter(description="Whether to add a title to the plot", default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    show_legend = luigi.BoolParameter(description="Whether show the legend", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)
    long_env = luigi.BoolParameter(description="full environment name: Humanoid -> HumanoidBulletEnv-v0", default=None)
    keep_zero = luigi.BoolParameter(description="If a stacked-bar element is zero in all the bar-charts, still show it in the legend.", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    y_lim_scale_factor = luigi.FloatParameter(description="scale ylim.max by scale-factor (hack to make room for bar-labels)", default=None)
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    suffix = luigi.Parameter(description="Add suffix to output files: OverlapStackedBarPlot.overlap_type_*.{suffix}.{ext}", default=None)

    # Needed by mk_SQL_tasks (for GeneratePlotIndexTask)
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    skip_output = False

    def requires(self):
        # TODO: we require (exactly 1) <overlap_type>.venn_js.js in each rlscope_dir.
        # TODO: we need to sub-select if there are multiple venn_js.js files...need selector arguments
        requires = []
        for rlscope_dir in self.rlscope_directories:
            kwargs = forward_kwargs(from_task=self, ToTaskKlass=GeneratePlotIndexTask)
            requires.append(GeneratePlotIndexTask(
                rlscope_directory=rlscope_dir,
                **kwargs))
        return requires

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = OverlapStackedBarPlot(**kwargs)
        self.dumper.run()

class GpuHwPlotTask(IMLTask):
    gpu_hw_directories = luigi.ListParameter(description="Multiple --rlscope-directory containing GPUHwCounterSampler.csv from running \"rls-prof --config gpu-hw\"")
    time_breakdown_directories = luigi.ListParameter(description="Multiple --rlscope-directory containing GPUHwCounterSampler.csv from running \"rls-prof --config gpu-hw\"")
    directory = luigi.Parameter(description="Output directory", default=".")
    xtick_expression = param_xtick_expression
    x_title = luigi.Parameter(description="x-axis title", default=None)
    title = luigi.Parameter(description="title", default=None)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    op_mapping = luigi.Parameter(description="Python expression defining a function mapping(algo) that returns a mapping that defines composite operations from rlscope.prof.operation annotations in the profiled code", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)
    rotation = luigi.FloatParameter(description="x-axis title rotation", default=None)

    # IGNORED.
    y2_logscale = luigi.BoolParameter(description="Show training time y-axis in logscale", parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    def requires(self):
        # TODO: we require (exactly 1) <overlap_type>.venn_js.js in each rlscope_dir.
        # TODO: we need to sub-select if there are multiple venn_js.js files...need selector arguments
        requires = []
        for rlscope_dir in self.time_breakdown_directories:
            kwargs = forward_kwargs(from_task=self, ToTaskKlass=GeneratePlotIndexTask)
            del kwargs['rlscope_directory']
            requires.append(GeneratePlotIndexTask(
                rlscope_directory=rlscope_dir,
                **kwargs))
        return requires

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        obj_args = dict()
        obj_args['rlscope_dir'] = self.gpu_hw_directories
        obj_args['time_breakdown_dir'] = self.time_breakdown_directories
        obj_args['output_directory'] = self.rlscope_directory
        obj_args['xtick_expression'] = self.xtick_expression
        obj_args['op_mapping'] = self.op_mapping
        obj_args['x_title'] = self.x_title
        obj_args['title'] = self.title
        obj_args['debug'] = self.debug
        obj_args['width'] = self.width
        logger.debug(f'GpuHwPlotTask.width = {self.width}')
        obj_args['height'] = self.height
        self.dumper = GpuUtilExperiment(obj_args)
        self.dumper.run()

class CategoryTransitionPlotTask(luigi.Task):
    time_breakdown_directories = luigi.ListParameter(description="RL-Scope directories containing uncorrected processed output of RLSAnalyze")
    rlscope_directories = luigi.ListParameter(description="RL-Scope directories containing raw RL-Scope trace files")
    category = luigi.Parameter(description="Category", default=None)
    directory = luigi.Parameter(description="Output directory", default=".")
    hack_upper_right_legend_bbox_x = param_hack_upper_right_legend_bbox_x

    xtick_expression = param_xtick_expression
    remap_df = luigi.Parameter(description="Transform df pandas.DataFrame object; useful for remapping regions to new ones", default=None)
    title = luigi.Parameter(description="Plot title", default=None)
    x_title = luigi.Parameter(description="x-axis title", default=None)
    rotation = luigi.FloatParameter(description="x-axis title rotation", default=15.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)
    include_gpu = luigi.BoolParameter(description="Include GPU transitions", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    include_python = luigi.BoolParameter(description="Include Python transitions", parsing=luigi.BoolParameter.EXPLICIT_PARSING)
    include_simulator = luigi.BoolParameter(description="Include Simulation during non-Simulator operation", parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    line_numbers = param_line_numbers

    skip_output = False

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = CategoryTransitionPlot(**kwargs)
        self.dumper.run()

class TexMetricsTask(luigi.Task):
    """
    Generate latex variable definitions for "Quantified intuitive findings" and
    "Suprising findings" regarding Framework choice RLScope paper.
    """
    algo_choice_csv = luigi.Parameter(description="stable_baselines_fig_10_algo_choice/OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.csv", default=None)

    framework_choice_csv = luigi.Parameter(description="TD3: OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.csv", default=None)
    framework_choice_ddpg_csv = luigi.Parameter(description="DDPG: OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.csv", default=None)

    framework_choice_uncorrected_csv = luigi.Parameter(description="TD3: OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.csv", default=None)
    framework_choice_ddpg_uncorrected_csv = luigi.Parameter(description="DDPG: OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.csv", default=None)

    framework_choice_trans_csv = luigi.Parameter(description="TD3: CategoryTransitionPlot.combined.csv", default=None)
    framework_choice_ddpg_trans_csv = luigi.Parameter(description="DDPG: CategoryTransitionPlot.combined.csv", default=None)

    directory = luigi.Parameter(description="Output directory", default=".")
    file_suffix = luigi.Parameter(description="Generated LaTeX file will contain *.<file_suffix>.tex", default=None)
    tex_variable_prefix = luigi.Parameter(description=r"Prefix \newcommand LaTeX variables with --tex-variable-prefix (NOTE: prefix will be camel cased)", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    line_numbers = param_line_numbers

    skip_output = False

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = TexMetrics(**kwargs)
        self.dumper.run()


def forward_kwargs(from_task, ToTaskKlass, ignore_argnames=None):
    kwargs = kwargs_from_task(from_task)
    fwd_kwargs = dict()
    for attr, value in kwargs.items():
        if (ignore_argnames is None or attr not in ignore_argnames) and \
                hasattr(ToTaskKlass, attr) and isinstance(getattr(ToTaskKlass, attr), luigi.Parameter):
            fwd_kwargs[attr] = value
    return fwd_kwargs

def kwargs_from_task(task):
    kwargs = dict()
    for key, value in task.__dict__.items():
        name = key

        m = re.search(r'^_', key)
        if m:
            continue

        m = re.search(r'^postgres_(?P<name>.*)', key)
        if m:
            name = m.group('name')

        kwargs[name] = value
    return kwargs

def keep_task_kwargs(kwargs, TaskKlass):
    keep_kwargs = dict()
    for opt, value in kwargs.items():
        if hasattr(TaskKlass, opt):
            keep_kwargs[opt] = value
    return keep_kwargs

def mk_SQLParserTask(task):
    return SQLParserTask(rlscope_directory=task.rlscope_directory, debug=task.debug, debug_single_thread=task.debug_single_thread,
                         postgres_host=task.postgres_host, postgres_user=task.postgres_user, postgres_password=task.postgres_password)

def mk_SQL_tasks(task):
    """
    Create Task dependencies for creating SQL database.

    If calibration files are provided, return dependency that ALSO "injects" overhead events
    so it can be subtracted downstream.

    :param task:
        Task that depends on having a SQL database, and that has arguments needed for SQL database
        (e.g. host, user, etc).
    """
    overhead_opts = [getattr(task, opt) for opt in CALIBRATION_OPTS]

    # Either ALL overhead calibration files are provided (subtract ALL overheads),
    # or NONE are provided (subtract NO overheads).
    if any(optval is not None for optval in overhead_opts):
        # assert all(getattr(task, opt) is not None for opt in overhead_opts)
        for optval in overhead_opts:
            if optval is None:
                raise RuntimeError(textwrap.dedent("""\
                RL-Scope ERROR: If you provide the {optval} calibration file, you must provide all these calibration files:
                {files}
                """).format(
                    files=textwrap.indent(pprint.pformat(CALIBRATION_OPTS), prefix='  '),
                    optval=optval,
                ).rstrip())

    has_calibration_files = all(opt is not None for opt in overhead_opts)

    # Add SQLParser args.
    sql_kwargs = dict(
        rlscope_directory=task.rlscope_directory,
        debug=task.debug,
        debug_single_thread=task.debug_single_thread,
        postgres_host=task.postgres_host,
        postgres_user=task.postgres_user,
        postgres_password=task.postgres_password
    )
    kwargs = dict()
    kwargs.update(sql_kwargs)

    if not has_calibration_files:
        return SQLParserTask(**kwargs)

    calibration_kwargs = dict()
    for opt in CALIBRATION_OPTS:
        calibration_kwargs[opt] = getattr(task, opt)
    # Add calibration files.
    kwargs.update(calibration_kwargs)

    # NOTE: SQLOverheadEventsParser depends on SQLParserTask.
    # logger.info("mk SQLOverheadEventsParser: {msg}".format(
    #     msg=pprint_msg(kwargs)))
    return SQLOverheadEventsTask(**kwargs)

from rlscope.profiler.rlscope_logging import logger
def main(argv=None, should_exit=True):
    if argv is None:
        argv = list(sys.argv[1:])
    ret = luigi.run(cmdline_args=argv, detailed_summary=True)
    retcode = 0
    if ret.status not in [luigi.LuigiStatusCode.SUCCESS, luigi.LuigiStatusCode.SUCCESS_WITH_RETRY]:
        retcode = 1
        logger.error(textwrap.dedent("""\
        RL-Scope analysis of trace files failed.
          > Debugging recommendation:
            Look for the last "> CMD:" that was run, and re-run it manually 
            with "--pdb --debug" flags added to break when it fails.
        """).rstrip())
    sys.exit(retcode)

class CallInterceptionOverheadTask(luigi.Task):
    # csv = luigi.Parameter(description="Path to overall_machine_util.raw.csv [output from UtilTask]")
    interception_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config interception'")
    uninstrumented_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config interception'")
    directory = luigi.Parameter(description="Output directory", default=".")

    # interception_directories = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config interception'", default=".")
    # no_interception_directories = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config interception'", default=".")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = CallInterceptionOverheadParser(**kwargs)
        self.dumper.run()

class CUPTIOverheadTask(luigi.Task):
    gpu_activities_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config gpu-activities'")
    no_gpu_activities_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config no-gpu-activities'")
    directory = luigi.Parameter(description="Output directory", default=".")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = CUPTIOverheadParser(**kwargs)
        self.dumper.run()

class CUPTIScalingOverheadTask(luigi.Task):
    gpu_activities_api_time_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config gpu-activities-api-time'")
    interception_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config interception'")

    directory = luigi.Parameter(description="Output directory", default=".")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_memoize = param_debug_memoize
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = CUPTIScalingOverheadParser(**kwargs)
        self.dumper.run()

class CorrectedTrainingTimeTask(luigi.Task):
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json
    rlscope_directories = luigi.ListParameter(description="RL-Scope directory that ran with full tracing enabled")
    uninstrumented_directories = luigi.ListParameter(description="RL-Scope directories for uninstrumented runs (rls-prof --config uninstrumented)")
    directory = luigi.Parameter(description="Output directory", default=".")
    # rlscope_prof_config = luigi.ChoiceParameter(description=textwrap.dedent("""
    rlscope_prof_config = luigi.Parameter(description=textwrap.dedent("""
    What option did you pass to \"rls-prof --config\"? 
    We use this to determine what overheads to subtract:
    
    instrumented: 
        We JUST subtract LD_PRELOAD overhead.
        (i.e. CUPTI is NOT enabled here).
    # gpu-activity: 
    #     We subtract:
    #     - LD_PRELOAD overhead
    #     - CUPTI overhead
    full:
        (default)
        We subtract:
        - LD_PRELOAD overhead
        - CUPTI overhead
        - Python pyprof overhead (if we detect pyprof events, otherwise this is zero)
    """),
                                            )

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_memoize = param_debug_memoize
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = CorrectedTrainingTimeParser(**kwargs)
        self.dumper.run()

class PyprofOverheadTask(luigi.Task):
    uninstrumented_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config uninstrumented train.py --rlscope-disable'")
    # pyprof_annotations_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config uninstrumented train.py --rlscope-disable-tfprof --rlscope-disable-pyprof-interceptions --rlscope-training-progress'")
    # pyprof_interceptions_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config uninstrumented train.py --rlscope-disable-tfprof --rlscope-disable-pyprof-annotations --rlscope-training-progress'")

    pyprof_annotations_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config uninstrumented train.py --rlscope-disable-tfprof --rlscope-disable-pyprof-interceptions'", default=None)
    pyprof_interceptions_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config uninstrumented train.py --rlscope-disable-tfprof --rlscope-disable-pyprof-annotations'", default=None)

    directory = luigi.Parameter(description="Output directory", default=".")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers
    # algo_env_from_dir = luigi.BoolParameter(description="Add algo/env columns based on directory structure of --rlscope-directories <algo>/<env>/rlscope_dir", default=True, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = PyprofOverheadParser(**kwargs)
        self.dumper.run()

class TotalTrainingTimeTask(luigi.Task):
    uninstrumented_directory = luigi.ListParameter(description="RL-Scope directory that ran with 'rls-prof --config uninstrumented train.py --rlscope-disable'")
    directory = luigi.Parameter(description="Output directory", default=".")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        self.dumper = TotalTrainingTimeParser(**kwargs)
        self.dumper.run()

class VennJsPlotTask(IMLTask):

    # NOT optional (to ensure we remember to do overhead correction).
    cupti_overhead_json = param_cupti_overhead_json
    LD_PRELOAD_overhead_json = param_LD_PRELOAD_overhead_json
    python_annotation_json = param_python_annotation_json
    python_clib_interception_tensorflow_json = param_python_clib_interception_tensorflow_json
    python_clib_interception_simulator_json = param_python_clib_interception_simulator_json

    # venn_js = luigi.ListParameter(description="venn_js.json path")
    # directory = luigi.Parameter(description="Output directory", default=".")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    algo = luigi.Parameter(description="RL algorithm", default=None)
    env = luigi.Parameter(description="Simulator", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    def requires(self):
        kwargs = self.param_kwargs

        return [
            _mk(kwargs, RLSAnalyze),
        ]

    # def output(self):
    #     return []

    def rlscope_run(self):
        venn_js_paths = []
        for path in each_file_recursive(self.rlscope_directory):
            if not is_venn_js_file(path):
                continue
            venn_js_paths.append(path)
        logger.info("Plot venn_js files:\n{msg}".format(
            msg=pprint_msg(venn_js_paths),
        ))
        for venn_js in venn_js_paths:
            kwargs = kwargs_from_task(self)
            plotter = VennJsPlotter(venn_js=venn_js, **kwargs)
            plotter.run()

class VennJsPlotOneTask(luigi.Task):
    venn_js = luigi.Parameter(description="venn_js path")

    # Plot attrs
    # rotation = luigi.FloatParameter(description="x-axis title rotation", default=45.)
    width = luigi.FloatParameter(description="Width of plot in inches", default=None)
    height = luigi.FloatParameter(description="Height of plot in inches", default=None)

    algo = luigi.Parameter(description="RL algorithm", default=None)
    env = luigi.Parameter(description="Simulator", default=None)

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        plotter = VennJsPlotter(**kwargs)
        plotter.run()


class SlidingWindowUtilizationPlotTask(IMLTask):
    polling_util_json = luigi.Parameter(description="Output from: rls-analyze --mode=polling_util")
    window_size_us = luigi.FloatParameter(description=textwrap.dedent("""\
    The sampling period (in nvidia-smi documentation lingo); nvidia-smi's GPU utilization metric looks 
    at a sliding window of time to determine the (%) of bins that had a kernel executing in them.  
    --window_size_us must be evenly divisible by the polling interval used in --sample_periods_json.
    """))

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    # TODO: require running "rls-analyze --mode=polling_util", and lookup polling_util_json from that task.
    def requires(self):
        return []

    def rlscope_run(self):
        kwargs = kwargs_from_task(self)
        assert 'directory' not in kwargs
        kwargs['directory'] = kwargs['rlscope_directory']
        del kwargs['rlscope_directory']
        plotter = SlidingWindowUtilizationPlot(**kwargs)
        plotter.run()


class CUDAEventCSVTask(IMLTask):

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    skip_output = False

    def requires(self):
        return []

    def rlscope_run(self):
        kwargs = kwargs_from_task(self)
        assert 'directory' not in kwargs
        kwargs['directory'] = kwargs['rlscope_directory']
        del kwargs['rlscope_directory']
        dumper = CUDAEventCSVReader(**kwargs)
        dumper.run()

class GPUUtilOverTimePlotTask(IMLTask):
    """
    Compare GPU utilization over time using a synthetic experiment that varies two parameters:
    - kernel_delay_us: time between launching consecutive kernels.
    - kernel_duration_us: time that kernels run for.
    """

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    rlscope_directories = luigi.ListParameter(description="Multiple --rlscope-directory entries for finding overlap_type files: *.venn_js.js")
    show_std = luigi.BoolParameter(description="If true, show stdev for kernel delay and duration", default=False, parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    skip_output = False

    def requires(self):
        return []

    def rlscope_run(self):
        kwargs = kwargs_from_task(self)
        assert 'directory' not in kwargs
        kwargs['directory'] = kwargs['rlscope_directory']
        del kwargs['rlscope_directory']
        dumper = GPUUtilOverTimePlot(**kwargs)
        dumper.run()

class NvprofKernelHistogramTask(luigi.Task):
    """
    Compare GPU utilization over time using a synthetic experiment that varies two parameters:
    - kernel_delay_us: time between launching consecutive kernels.
    - kernel_duration_us: time that kernels run for.
    """

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    nvprof_file = luigi.Parameter(description="$ nvprof -o <nvprof_file.nvprof>")

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        dumper = NvprofKernelHistogram(**kwargs)
        dumper.run()

class CrossProcessOverlapHistogramTask(luigi.Task):
    """
    Plot a histogram of inter-worker (cross process) overlap between GPU kernels.
    """

    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    cross_process_overlap = luigi.Parameter(description="Output file from analysis: OverlapResult.cross_process.json")

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        dumper = CrossProcessOverlapHistogram(**kwargs)
        dumper.run()

class NvprofTracesTask(luigi.Task):
    """
    Generate nvprof csv files of traces:
    $ nvprof -i profile.nvprof --print-gpu-trace 2>&1 > profile.nvprof.gpu_trace.csv
    $ nvprof -i profile.nvprof --print-api-trace 2>&1 > profile.nvprof.api_trace.csv
    Plot a histogram of inter-worker (cross process) overlap between GPU kernels.
    """
    debug = param_debug
    debug_single_thread = param_debug_single_thread
    debug_perf = param_debug_perf
    line_numbers = param_line_numbers

    rlscope_directory = luigi.Parameter(description="Location of trace-files")

    n_workers = luigi.IntParameter(
        description="How many threads to simultaneously run",
        default=multiprocessing.cpu_count())

    force = luigi.BoolParameter(description="re-run nvprof even if csv file exists")

    def requires(self):
        return []

    def output(self):
        return []

    def run(self):
        setup_logging_from_self(self)
        kwargs = kwargs_from_task(self)
        assert 'directory' not in kwargs
        kwargs['directory'] = kwargs['rlscope_directory']
        del kwargs['rlscope_directory']
        dumper = NvprofTraces(**kwargs)
        dumper.run()


def setup_logging_from_self(self):
    setup_logging(
        debug=getattr(self, 'debug', False),
        line_numbers=getattr(self, 'line_numbers', None),
    )
def setup_logging_from_kwargs(kwargs):
    setup_logging(
        debug=kwargs.get('debug', False),
        line_numbers=kwargs.get('line_numbers', None),
    )
def setup_logging(debug, line_numbers=None):
    if debug:
        line_numbers = True
    rlscope_logging.setup_logger(
        debug=debug,
        line_numbers=line_numbers,
    )

NOT_RUNNABLE_TASKS = get_NOT_RUNNABLE_TASKS()
RLSCOPE_TASKS = get_RLSCOPE_TASKS()
RLSCOPE_TASKS.add(TraceEventsTask)
RLSCOPE_TASKS.add(OverlapStackedBarTask)
RLSCOPE_TASKS.add(UtilTask)
RLSCOPE_TASKS.add(UtilPlotTask)
RLSCOPE_TASKS.add(TrainingProgressTask)
RLSCOPE_TASKS.add(ProfilingOverheadPlotTask)
RLSCOPE_TASKS.add(ExtrapolatedTrainingTimeTask)
RLSCOPE_TASKS.add(CallInterceptionOverheadTask)
RLSCOPE_TASKS.add(CUPTIOverheadTask)
RLSCOPE_TASKS.add(CorrectedTrainingTimeTask)
RLSCOPE_TASKS.add(PyprofOverheadTask)
RLSCOPE_TASKS.add(CUPTIScalingOverheadTask)
RLSCOPE_TASKS.add(TotalTrainingTimeTask)
RLSCOPE_TASKS.add(ConvertResourceOverlapToResourceSubplotTask)
RLSCOPE_TASKS.add(VennJsPlotTask)
RLSCOPE_TASKS.add(VennJsPlotOneTask)
RLSCOPE_TASKS.add(SlidingWindowUtilizationPlotTask)
RLSCOPE_TASKS.add(CUDAEventCSVTask)
RLSCOPE_TASKS.add(GPUUtilOverTimePlotTask)
RLSCOPE_TASKS.add(NvprofKernelHistogramTask)
RLSCOPE_TASKS.add(CrossProcessOverlapHistogramTask)
RLSCOPE_TASKS.add(CategoryTransitionPlotTask)
RLSCOPE_TASKS.add(NvprofTracesTask)
RLSCOPE_TASKS.add(TexMetricsTask)

if __name__ == "__main__":
    main()
