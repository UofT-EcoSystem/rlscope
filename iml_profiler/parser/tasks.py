"""
Define IML DAG execution graph using "luigi" DAG execution framework.

For luigi documetation, see:
https://luigi.readthedocs.io/en/stable/index.html
"""
import logging
import luigi

import re
import pwd
import textwrap
import datetime
import pprint
import sys
import os

from iml_profiler.parser.tfprof import TotalTimeParser, TraceEventsParser
from iml_profiler.parser.pyprof import PythonProfileParser, PythonFlameGraphParser, PythonProfileTotalParser
from iml_profiler.parser.plot import TimeBreakdownPlot, PlotSummary, CombinedProfileParser, CategoryOverlapPlot, UtilizationPlot, HeatScalePlot
from iml_profiler.parser.db import SQLParser
from iml_profiler.parser.stacked_bar_plots import OverlapStackedBarPlot

PARSER_KLASSES = [PythonProfileParser, PythonFlameGraphParser, PlotSummary, TimeBreakdownPlot, CategoryOverlapPlot, UtilizationPlot, HeatScalePlot, TotalTimeParser, TraceEventsParser, SQLParser]
PARSER_NAME_TO_KLASS = dict((ParserKlass.__name__, ParserKlass) \
                            for ParserKlass in PARSER_KLASSES)

# Make this pipeline:
# - iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules SQLParser
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type CategoryOverlap
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type ResourceOverlap
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type OperationOverlap
# |-> iml-analyze --directories ~/clone/baselines/output/PongNoFrameskip-v4/docker --rules UtilizationPlot --overlap-type ResourceSubPlot
#
# Wrap running all the existing classes with a class that outputs a "done" marker file.
# That way, we don't need to modify the classes to use luigi's "with self.output().open('w') as outfile:".

# List of runnable tasks for iml-analyze.
# Used for generating command-line usage help.

# IML_TASKS = ...
# NOT_RUNNABLE_TASKS = ...
def get_NOT_RUNNABLE_TASKS():
    return [IMLTask, _UtilizationPlotTask]

def get_IML_TASKS():
    global NOT_RUNNABLE_TASKS
    iml_tasks = set()
    for name, cls in globals().items():
        if isinstance(cls, type) and issubclass(cls, IMLTask) and cls not in NOT_RUNNABLE_TASKS:
            iml_tasks.add(cls)
    return iml_tasks

def get_username():
    return pwd.getpwuid(os.getuid())[0]

param_postgres_password = luigi.Parameter(description="Postgres password; default: env.PGPASSWORD", default=None)
param_postgres_user = luigi.Parameter(description="Postgres user", default=None)
param_postgres_host = luigi.Parameter(description="Postgres host", default=None)

param_debug_single_thread = luigi.BoolParameter(description=textwrap.dedent("""
        Run any multiprocessing stuff using a single thread for debugging.
        """))

class IMLTask(luigi.Task):
    iml_directory = luigi.Parameter(description="Location of trace-files")
    debug = luigi.BoolParameter(description="debug")
    debug_single_thread = param_debug_single_thread

    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    skip_output = False

    def output(self):
        return luigi.LocalTarget(self._done_file)

    @property
    def _done_file(self):
        """
        e.g.

        <--iml-directory>/SQLParserTask.task
        """
        return "{dir}/{name}.task".format(
            dir=self.iml_directory, name=self._task_name)

    @property
    def _task_name(self):
        return self.__class__.__name__

    def mark_done(self, start_t, end_t):
        if self.skip_output:
            logging.info("> Skipping output={path} for task {name}".format(
                path=self._done_file,
                name=self._task_name))
            return
        with self.output().open('w') as f:
            print_cmd(cmd=sys.argv, file=f)
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

    def iml_run(self):
        raise NotImplementedError("{klass} must override iml_run()".format(
            klass=self.__class__.__name__))

    def run(self):
        start_t = datetime.datetime.now()
        self.iml_run()
        end_t = datetime.datetime.now()

        self.mark_done(start_t, end_t)

class SQLParserTask(IMLTask):
    def requires(self):
        return []

    def iml_run(self):
        self.sql_parser = SQLParser(
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )
        self.sql_parser.run()

class _UtilizationPlotTask(IMLTask):
    def requires(self):
        return [
            mk_SQLParserTask(self),
        ]

    def iml_run(self):
        self.sql_parser = UtilizationPlot(
            overlap_type=self.overlap_type,
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
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

class All(IMLTask):
    # Don't output All.task, so that if (for example)
    # ResourceOverlapTask.task is deleted, we will still re-run it.
    skip_output = True

    def requires(self):
        kwargs = self.param_kwargs
        return [
            ResourceOverlapTask(**kwargs),
            CategoryOverlapTask(**kwargs),
            ResourceSubplotTask(**kwargs),
            OperationOverlapTask(**kwargs),
            HeatScaleTask(**kwargs),
        ]

    def iml_run(self):
        pass

class HeatScaleTask(IMLTask):
    # step_sec=1.,
    # pixels_per_square=10,
    # decay=0.99,
    def requires(self):
        return [
            mk_SQLParserTask(self),
        ]

    def iml_run(self):
        self.heat_scale = HeatScalePlot(
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
        )
        self.heat_scale.run()

class TraceEventsTask(luigi.Task):
    iml_directory = luigi.Parameter(description="Location of trace-files")
    debug = luigi.BoolParameter(description="debug")
    debug_single_thread = param_debug_single_thread

    postgres_password = param_postgres_password
    postgres_user = param_postgres_user
    postgres_host = param_postgres_host

    overlaps_event_id = luigi.IntParameter(description="show events that overlap with this event (identified by its event_id)", default=None)
    op_name = luigi.Parameter(description="operation name (e.g. q_forward)", default=None)
    process_name = luigi.Parameter(description="show events belonging to this process", default=None)
    start_usec = luigi.FloatParameter(description="Show events whose start-time is >= start_usec", default=None)
    end_usec = luigi.FloatParameter(description="Show events whose end-time is <= end_usec", default=None)

    skip_output = False

    def requires(self):
        return [
            mk_SQLParserTask(self),
        ]

    def output(self):
        return []

    def run(self):
        self.dumper = TraceEventsParser(
            directory=self.iml_directory,
            host=self.postgres_host,
            user=self.postgres_user,
            password=self.postgres_password,
            debug=self.debug,
            overlaps_event_id=self.overlaps_event_id,
            op_name=self.op_name,
            process_name=self.process_name,
            # process_op_nest=False,
            start_usec=self.start_usec,
            end_usec=self.end_usec,
        )
        self.dumper.run()

class OverlapStackedBarTask(luigi.Task):
    iml_directories = luigi.ListParameter(description="Multiple --iml-directory entries for finding overlap_type files: *.venn_js.js")
    directory = luigi.Parameter(description="Output directory", default=".")
    title = luigi.Parameter(description="Plot title", default=None)
    rotation = luigi.FloatParameter(description="x-axis title rotation", default=10.)
    overlap_type = luigi.ChoiceParameter(choices=OverlapStackedBarPlot.SUPPORTED_OVERLAP_TYPES, description="What type of <overlap_type>*.venn_js.js files should we read from?")
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
    x_type = luigi.ChoiceParameter(choices=OverlapStackedBarPlot.SUPPORTED_X_TYPES,
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
    # postgres_host = param_postgres_host
    # postgres_user = param_postgres_user
    # postgres_password = param_postgres_password
    show_title = luigi.BoolParameter(description="Whether to add a title to the plot", default=True)
    show_legend = luigi.BoolParameter(description="Whether show the legend", default=True)
    debug = luigi.BoolParameter(description="debug")
    debug_single_thread = param_debug_single_thread
    suffix = luigi.Parameter(description="Add suffix to output files: OverlapStackedBarPlot.overlap_type_*.{suffix}.{ext}", default=None)

    skip_output = False

    def requires(self):
        # TODO: we require (exactly 1) <overlap_type>.venn_js.js in each iml_dir.
        # TODO: we need to sub-select if there are multiple venn_js.js files...need selector arguments
        return [
            # mk_SQLParserTask(self),
        ]

    def output(self):
        return []

    def run(self):
        kwargs = kwargs_from_task(self)
        self.dumper = OverlapStackedBarPlot(**kwargs)
        self.dumper.run()

def kwargs_from_task(task):
    kwargs = dict()
    for key, value in task.__dict__.items():
        name = key

        m = re.search('^_', key)
        if m:
            continue

        m = re.search('^postgres_(?P<name>.*)', key)
        if m:
            name = m.group('name')

        kwargs[name] = value
    return kwargs

def mk_SQLParserTask(task):
    return SQLParserTask(iml_directory=task.iml_directory, debug=task.debug, debug_single_thread=task.debug_single_thread,
                         postgres_host=task.postgres_host, postgres_user=task.postgres_user, postgres_password=task.postgres_password),

def print_cmd(cmd, file=sys.stdout):
    if type(cmd) == list:
        cmd_str = " ".join([str(x) for x in cmd])
    else:
        cmd_str = cmd
    print(("> CMD:\n"
           "  $ {cmd}\n"
           "  PWD={pwd}"
           ).format(
        cmd=cmd_str,
        pwd=os.getcwd(),
    ), file=file)

from iml_profiler.profiler import glbl
def main(argv=None):
    glbl.setup_logging()
    if argv is None:
        argv = list(sys.argv[1:])
    luigi.run(cmdline_args=argv)

NOT_RUNNABLE_TASKS = get_NOT_RUNNABLE_TASKS()
IML_TASKS = get_IML_TASKS()
IML_TASKS.add(TraceEventsTask)
IML_TASKS.add(OverlapStackedBarTask)

if __name__ == "__main__":
    main()
