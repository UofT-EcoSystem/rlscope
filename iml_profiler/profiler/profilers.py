import argparse
import inspect
import cProfile, pstats, io
import codecs
import sys
import json
import argparse
import pprint
import traceback
import subprocess
import textwrap
import os
import time
import re
import math
import contextlib
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

from iml_profiler.profiler.concurrent import ForkedProcessPool

import iml_profiler

from iml_profiler.profiler.util import pprint_msg
from iml_profiler.scripts.utilization_sampler import util_sampler
from iml_profiler.profiler.util import args_to_cmdline, get_available_gpus, get_available_cpus
from iml_profiler.profiler.util import get_stacktrace

from iml_profiler.profiler.iml_logging import logger

ORIG_EXCEPT_HOOK = sys.excepthook
def cleanup_profiler_excepthook(exctype, value, traceback):
    # Stop utilization sampler if it is running.
    #
    # NOTE: If we crash unexpectedly, make sure to terminate the utilization_sampler.py process.
    # This is important when running unit-tests; otherwise the "train" portion of the unit-test will hang!
    # It's also important to prevent zombie utilization_sampler.py from accumulating.
    if iml_profiler.api.prof is not None:
        iml_profiler.api.prof.maybe_terminate_utilization_sampler(warn_terminated=True)
    return ORIG_EXCEPT_HOOK(exctype, value, traceback)


from iml_profiler.profiler import unit_test_util
from iml_profiler.profiler.util import print_cmd

# from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from iml_profiler.protobuf.pyprof_pb2 import ProcessMetadata, TrainingProgress, IncrementalTrainingProgress, TP_NO_PROGRESS, TP_HAS_PROGRESS

# pip install py-cpuinfo
import cpuinfo
import psutil

from os import environ as ENV

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler.parser.common import *
from iml_profiler.parser import constants
from iml_profiler.profiler import timer as iml_timer
# from iml_profiler.profiler import cudaprofile
from iml_profiler.clib import rlscope_api
from iml_profiler.profiler import clib_wrap
# from iml_profiler.profiler.clib_wrap import MICROSECONDS_IN_SECOND

from iml_profiler.profiler import proto_util

from iml_profiler import py_config

# DEBUG_OP_STACK = True
DEBUG_OP_STACK = False

# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

TF_PRINT_TIMESTAMP = ENV.get('TF_PRINT_TIMESTAMP', 'no') == 'yes'

UTILIZATION_SAMPLER_PY = _j(py_config.ROOT, 'python', 'scripts', 'utilization_sampler.py')
PYTHON_BIN = 'python3'

# Warn about forgetting to call iml.prof.report_progress(...) every 30 seconds.
REPORT_PROGRESS_WARN_EVERY_SEC = 10.

# Dump training_progress.*.proto files every 10 seconds.
DUMP_TRAINING_PROGRESS_EVERY_SEC = 10.

# If we exceed 1000 session.run(...) calls without dumping a trace, only print a warning every 100 calls.
WARN_EVERY_CALL_MODULO = 100

# For warnings that go off all the time, how frequently should we inform the user about them?
WARN_EVERY_SEC = 10

# Number of steps before we dump a trace file
# STEPS_PER_TRACE = 1000
# STEPS_PER_TRACE = 10

_TF_MODIFIED = False
def modify_tensorflow(tfprof_enabled, pyprof_enabled, allow_missing_librlscope=False):
    global _TF_MODIFIED
    if _TF_MODIFIED:
        return

    import iml_profiler.profiler.session
    import iml_profiler.profiler.estimator

    uninstrumented_run = not tfprof_enabled and not pyprof_enabled

    setup(tfprof_enabled, pyprof_enabled,
          allow_missing_librlscope=allow_missing_librlscope)
    if not uninstrumented_run:
        iml_profiler.profiler.session.setup()
        iml_profiler.profiler.estimator.setup()

    if pyprof_enabled:
        clib_wrap.setup()

    # from tensorflow.python.profiler import profile_context
    # profile_context.MAX_TRACED_STEPS = 99999999

    _TF_MODIFIED = True

# All currently active Profiler objects (there should really only be one).
# Used for hooking into sess.as_default()
PROFILERS = []


SETUP_DONE = False
def setup(tfprof_enabled, pyprof_enabled, allow_skip=False, allow_missing_librlscope=False):
    global SETUP_DONE
    if allow_skip and SETUP_DONE:
        return
    assert not SETUP_DONE

    rlscope_api.find_librlscope()

    uninstrumented_run = not tfprof_enabled and not pyprof_enabled

    # setup_wrap_BaseSession_as_default()

    if rlscope_api.is_used():
        rlscope_api.load_library()
    else:
        if not allow_missing_librlscope:
            # if tfprof_enabled:
            logger.error(textwrap.dedent("""\
            To profile using RLScope, you must re-run your command-line as:
              $ iml-prof {cmd}
            If you want to run without RLScope, add --iml-disable to the above command.
            """).format(
                cmd=' '.join(shlex.quote(opt) for opt in [sys.executable] + sys.argv),
            ).rstrip())
            sys.exit(1)

    if not uninstrumented_run:
        # iml_profiler.profiler.session.register_session_active_hook(AddProfileContextHook)
        # iml_profiler.profiler.session.register_session_inactive_hook(RemoveProfileContextHook)
        # iml_profiler.profiler.session.register_session_run_hook(MaybeDumperTfprofContextHook)
        # clib_wrap.register_record_event_hook(DumpPyprofTraceHook)

        sys.excepthook = cleanup_profiler_excepthook

    SETUP_DONE = True

def get_tfprof_path(directory, bench_name, session_id, trace_id):
    tfprof_path = _j(
        directory,
        "profile{bench}{trace}{sess}.proto".format(
            bench=bench_suffix(bench_name),
            trace=trace_suffix(trace_id),
            sess=sess_suffix(session_id),
        ))
    return tfprof_path

class ProtoDumper:
    def __init__(self,
                 name,
                 trace_id,
                 proto,
                 proto_path,
                 debug=False):
        self.name = name
        self.trace_id = trace_id
        self.proto = proto
        self.proto_path = proto_path
        self.debug = debug

    def dump(self):
        """
        Dump protobuf to:
        - {proto}.trace_<next_trace_id>.proto
        """
        if self.debug:
            logger.info("{name}.dump start, path={path}".format(
                name=self.name,
                path=self.proto_path))

        with open(self.proto_path, 'wb') as f:
            f.write(self.proto.SerializeToString())

        if self.debug:
            logger.info("{name}.dump done, path={path}".format(
                name=self.name,
                path=self.proto_path))

class ProcessMetadataDumper:
    def __init__(self,
                 trace_id,
                 process_metadata,
                 process_metadata_proto_path,
                 debug=False):
        assert type(process_metadata) == ProcessMetadata
        self.trace_id = trace_id
        self.process_metadata = process_metadata
        self.process_metadata_proto_path = process_metadata_proto_path
        self.debug = debug

    def dump(self):
        """
        Dump ProcessMetadata to:
        - process_metadata.trace_<next_trace_id>.proto
        """
        if self.debug:
            logger.info("{klass}.dump start, path={path}".format(
                klass=self.__class__.__name__,
                path=self.process_metadata_proto_path))

        with open(self.process_metadata_proto_path, 'wb') as f:
            f.write(self.process_metadata.SerializeToString())

        if self.debug:
            logger.info("{klass}.dump done, path={path}".format(
                klass=self.__class__.__name__,
                path=self.process_metadata_proto_path))

"""
Add after-inactive hooks to call remove_profile_context, and after-active hooks to add_profile_context.
"""

_prof_singleton = None

class Profiler:
    """
    Generic profiler that uses BOTH CUDAProfiler and PythonProfiler.

    Intended use case:

    profiler = Profiler(...)

    for epoch in range(epochs):
        for i in range(steps_per_epoch):

            #
            # Only collect profiling information during inner training loop operations.
            # Don't both collecting "noise" for one-time initialization.
            #

            profiler.enable_profiling()
            # Some part of your inner-training loop.
            # For e.g. if its MNIST, this will be both the Forward/Backward passes.
            sess.run(train_op, ...)
            profiler.disable_profiling()


    Members:

    self.profile_time_sec:
        Total time spent profiling, in seconds.
        i.e. the time spent in between enable/disable profiling calls:

        self.enable_profiling()
        ... # This time.
        self.disable_profiling()

        This time can be compared/validated against the total time reported
        by the profiler (nvprof/pyprof).

    :param reports_progress
        Whether or not the current process will call:

            iml.prof.report_progress(percent_complete)

        If this is true, then if --iml-trace-time-sec expires, we will WAIT
        until iml.prof.report_progress is called prior to exiting, and we will warn
        if it hasn't been called.

        Q: Can't we imply this from whether it ever calls report_progress?
        A: No, since it may just take a really long time before report_progress ever
           gets called.

    :param just_sample_util
        Just collect machine utilization samples;
        don't collect any profiling information.
        This can be run with vanilla tensorflow.

    :param exit_early
        if True, exit ML script immediately after benchmarking bench_name.
        Othwerwise, continue executing ML script until it finishes.
    """
    def __init__(self, directory=None,
                 bench_name=NO_BENCH_NAME,
                 num_calls=None,
                 trace_time_sec=None,
                 max_timesteps=None,
                 num_traces=None,
                 keep_traces=None,
                 tfprof=True,
                 reports_progress=False,
                 just_sample_util=None,
                 training_progress=None,
                 c_lib_func_pyprof_pattern=None,
                 # tfprof=True,
                 repetition_time_limit_sec=10.,
                 debug=None,
                 exit_early=True,
                 require_end_operation=False,
                 python=None,
                 # WARNING: MAKE SURE to use None as the default for boolean flags!
                 # Otherwise, if you set this to False by default, get_argval will ALWAYS return
                 # False (even if --iml-calibration is set!)
                 calibration=None,
                 disable=None,
                 disable_pyprof_annotations=None,
                 disable_pyprof_interceptions=None,
                 disable_pyprof=None,
                 disable_tfprof=None,
                 disable_pyprof_trace=None,
                 disable_gpu_hw=None,
                 env=None,
                 algo=None,
                 delay=None,
                 delay_training_loop_iters=None,
                 max_training_loop_iters=None,
                 delay_passes=None,
                 max_passes=None,
                 unit_test=None,
                 unit_test_name=None,
                 skip_rm_traces=None,
                 args=None):

        def get_iml_argname(argname, internal=False):
            name = argname
            # name = re.sub('_', '-', name)
            if internal:
                name = "iml_internal_{name}".format(name=name)
            else:
                name = "iml_{name}".format(name=name)
            return name

        def get_argval(argname, klass_arg, default_arg, allow_none=True, internal=False):
            """
            Extract --iml-* args added by add_iml_arguments, unless provided with arguments to the constructor.

            :param argname:
                Name of argument (without iml prefix).
            :param klass_arg:
                Value provided to constructor.
            :param default_arg:
                Default value to use if klass_arg is not provided to constructor and/or args is not provided.
            :return:
            """
            if args is None or klass_arg is not None:
                return klass_arg

            iml_argname = get_iml_argname(argname, internal=internal)
            
            if hasattr(args, iml_argname):
                if getattr(args, iml_argname) is not None:
                    # assert isinstance(args, argparse.Namespace)
                    argval = getattr(args, iml_argname)
                    return argval
            elif iml_argname in args and args[iml_argname] is not None:
                # assert type(args) == dict
                argval = args[iml_argname]
                return argval

            if not allow_none and default_arg is None:
                self._failing = True
                raise RuntimeError("IML: you must provide a value for --{arg}".format(
                    arg=re.sub('_', '-', iml_argname)))

            return default_arg

        def get_internal_argval(argname, default_arg=None, allow_none=True):
            """
            Extract --iml-internal-* args added by add_iml_arguments, unless provided with arguments to the constructor.
            """
            klass_arg = None
            argval = get_argval(argname, klass_arg, default_arg,
                                allow_none=allow_none, internal=True)
            return argval

        self.algo = algo
        self.env = env

        self.metadata = dict()
        add_metadata = dict()
        if self.algo is not None:
            add_metadata['algo'] = self.algo
        if self.env is not None:
            add_metadata['env'] = self.env
        if len(add_metadata) > 0:
            self.set_metadata(add_metadata)

        self._failing = False
        self._has_called_enable_tracing = False
        self.num_training_loop_iters = 0
        self.num_passes = 0
        self.pass_idx = 0
        self.has_next_pass = False
        self.debug = get_argval('debug', debug, False)
        self.calibration = get_argval('calibration', calibration, False)
        if self.debug:
            py_config.DEBUG = self.debug
        self.directory = get_argval('directory', directory, None, allow_none=False)
        self.disable = get_argval('disable', disable, False)
        if 'IML_CONFIG' in os.environ:
            self.iml_config = os.environ['IML_CONFIG']
            if not self.calibration and self.iml_config == 'uninstrumented':
                # WARNING: We do NOT do this for --iml-calibration runs, as it will cause a BUG!
                # In order for calibration to work properly, we need to be able to enable each book-keeping feature in isolation!
                # Q: What code depends on "--config uninstrumented" (without --iml-calibration) implying --iml-disable?
                # A: minigo code.
                logger.warning("DISABLE ALL IML FEATURES for --config={config} run".format(
                    config=self.iml_config))
                self.disable = True
        self.disable_pyprof_annotations = get_argval('disable_pyprof_annotations', disable_pyprof_annotations, False)
        self.disable_pyprof_interceptions = get_argval('disable_pyprof_interceptions', disable_pyprof_interceptions, False)
        self.disable_pyprof = get_argval('disable_pyprof', disable_pyprof, False)
        # NOTE: currently has no effect since tfprof is entirely implemented in LD_PRELOAD librlscope.so library.
        self.disable_tfprof = get_argval('disable_tfprof', disable_tfprof, False)
        # Disable OLD tfprof tracing code.  We now use iml-prof to trace stuff.
        self.disable_pyprof_trace = get_argval('disable_pyprof_trace', disable_pyprof_trace, False)
        self.disable_gpu_hw = get_argval('disable_gpu_hw', disable_gpu_hw, False)
        self.delay = get_argval('delay', delay, None)

        self.delay_training_loop_iters = get_argval('delay_training_loop_iters', delay_training_loop_iters, None)
        self.max_training_loop_iters = get_argval('max_training_loop_iters', max_training_loop_iters, None)

        self.delay_passes = get_argval('delay_passes', delay_passes, None)
        self.max_passes = get_argval('max_passes', max_passes, None)

        self.just_sample_util = get_argval('just_sample_util', just_sample_util, False)
        self.training_progress = get_argval('training_progress', training_progress, False)
        self._loaded_libcupti = False

        tfprof_enabled = not self.disable and not self.disable_tfprof
        # pyprof_enabled = Do we want to enable Python->C++ interception for collecting pyprof events?
        pyprof_enabled = not self.disable and not self.disable_pyprof and not self.disable_pyprof_interceptions
        modify_tensorflow(
            tfprof_enabled=tfprof_enabled,
            pyprof_enabled=pyprof_enabled,
            allow_missing_librlscope=self.disable,
        )
        if self.disable:
            logger.info("IML: note that profiling is disabled for this run")
        if ( self.disable or self.disable_gpu_hw ) and rlscope_api.is_used():
            logger.info("(--iml-disable-gpu-hw) Disable GPU HW sampling")
            rlscope_api.disable_gpu_hw()

        # self.manager = multiprocessing.Manager()
        # self.pyprof_dump_manager = clib_wrap.PyprofDumpManager(self.manager)

        # if self.disable_pyprof_trace:
        #     clib_wrap.disable_pyprof_trace()

        global _prof_singleton
        if _prof_singleton is not None:
            self._failing = True
            raise RuntimeError("IML: Only a single profiler.Profiler object can be created; use iml.handle_iml_args + iml.prof instead.")
        _prof_singleton = self

        self.machine_name = get_machine_name()

        self.percent_complete = None
        self._tracing_enabled = False
        self._hw_pass_running = False
        self._incremental_training_progress = dict()
        self._last_dumped_training_progress = None
        self._start_percent_complete = 0.
        self._start_num_timesteps = 0
        # self._delayed_disable = False
        self.num_timesteps = None
        self.total_timesteps = None
        # self.bg_dumper = ForkedProcessPool(name="bg_dumper", debug=self.debug,
        #                                    # cpu_affinity=dump_cpus,
        #                                    )

        self._op_stack = []
        self._start_us = None
        self._stop_us = None

        self.util_sampler = None
        if self.just_sample_util:
            self.util_sampler = util_sampler(
                iml_directory=self.directory,
                debug=self.debug,
            )

        """
        If set, require the user to call prof.end_operation
        (don't allow a call to prof.set_operation to also count as a call to prof.end_operation)
        """
        self.require_end_operation = require_end_operation
        self.start_call_us = dict()
        self.end_call_us = dict()

        self._is_finishing = False
        self.next_trace_id = None
        self.process_name = None
        # self.init_trace_id()
        self._tfprof_enabled = False
        self._pyprof_enabled = False
        self._iml_prof_enabled = False
        self.total_profile_time_sec = 0

        self.python = get_argval('python', python, False)
        self.skip_rm_traces = get_argval('skip_rm_traces', skip_rm_traces, False)
        self.exit_early = exit_early
        self.tfprof = tfprof
        self.reports_progress = reports_progress
        self.c_lib_func_pyprof_pattern = c_lib_func_pyprof_pattern
        self.repetition_time_limit_sec = repetition_time_limit_sec
        self.num_calls = get_argval('num_calls', num_calls, None)
        self.trace_time_sec = get_argval('trace_time_sec', trace_time_sec, None)
        self.max_timesteps = get_argval('max_timesteps', max_timesteps, None)
        self._last_warned_trace_time_sec = None
        self._last_warned_report_progress_idx = None
        self._should_finish_idx = 0
        self.start_trace_time_sec = None
        self.num_traces = get_argval('num_traces', num_traces, None)
        self.keep_traces = get_argval('keep_traces', keep_traces, False)
        self.bench_name = get_argval('bench_name', bench_name, None)

        self.util_sampler_pid = get_internal_argval('util_sampler_pid')
        self.handle_utilization_sampler = False

        self.start_trace_time_sec = get_internal_argval('start_trace_time_sec')
        self.phase = get_internal_argval('phase', constants.DEFAULT_PHASE)

        self.parent_process_name = get_internal_argval('parent_process_name')

        self.unit_test = get_argval('unit_test', unit_test, False)
        self.unit_test_name = get_argval('unit_test_name', unit_test_name, None)
        if self.unit_test:
            self.ut = unit_test_util.UnitTestDataDumper(debug=self.debug)

        # if not self.tfprof:
        #     self.cuda_profiler = CUDAProfiler()
        self.start_t = dict()
        self.end_t = dict()
        self.time_sec = dict()
        self.code_count = dict()
        self.steps = 0
        # clib_wrap.set_step(self._pyprof_step, ignore_disable=True)
        self.average_time_per_call_sec = None
        self.average_time_per_call_no_profile_sec = None

        if self.python:
            self.pyprof = PythonProfiler(self.directory)

        if self.debug:
            logger.info(pprint_msg({'Profiler.attrs': self.__dict__}))

        # self.sess = None
        # self.pctx = None

        # Total times collected from running profiled operations.
        # self.profile_sec = []
        # Total times collected from running un-profiled operations.
        # self.no_profile_sec = []

        # clib_wrap.wrap_libs()

    def get_start_trace_time_sec(self):
        # NOTE: ML script may fork new python scripts before tracing with pyprof/tfprof even begins
        # ( i.e. before prof.start(), prof.set_operation() )
        # So, in that case, start the timer immediately prior to fork.
        self._init_trace_time()
        return self.start_trace_time_sec

    def _delete_traces(self):
        """
        Delete ALL traces in self.out_dir.

        Generally used to delete traces from a PREVIOUS run before re-running the profiler.
        """
        os.makedirs(self.directory, exist_ok=True)
        if self.skip_rm_traces:
            logger.info("IML: SKIP deleting trace-files rooted at {dir} (--iml-skip-rm-traces)")
            return
        recursive_delete_trace_files(self.directory)

    def _init_trace_id_from_traces(self):
        """
        Keep traces from previous runs of the profiler/ML-script.

        Start recording traces using the next available trace id.
        """
        self.next_trace_id = 0
        # NOTE: We DON'T want to keep existing trace files across runs...
        # we really should add code for deleting existing trace files...
        for dirpath, dirnames, filenames in os.walk(self.out_dir):
            for base in filenames:
                path = _j(dirpath, base)
                m = is_pyprof_file(path)
                if m:
                    trace_id = int(m.group('trace_id'))
                    self.next_trace_id = max(self.next_trace_id, trace_id + 1)

    def init_trace_id(self):
        if self.process_name is None or self.phase is None:
            return

        assert self.next_trace_id is None

        # See --iml-keep-traces
        if self.keep_traces:
            self._init_trace_id_from_traces()
        else:
            self._delete_traces()
            self.next_trace_id = 0

        if py_config.DEBUG:
            logger.info("> Using next_trace_id = {id}".format(id=self.next_trace_id))

    def _init_num_calls(self, bench_name, func, *args, **kwargs):
        """
        PSEUDOCODE:

        We may want to run additional iterations because:

        - We want (stdev / mean) to be below a percentage
          NOTE: it may be the case that what we are measuring is highly variable,
          and this threshold is never crossed!

        - We only want to run the benchmark for a certain time limit (e.g. 1 minute).

        - We need the # of iterations of what we are measuring to be large enough so that:
            time(# of iterations) < clock_precision

        Q: Is there something (statisically) wrong with running some functions MORE than others,
        then combining their results to get a standard deviation result?

          If something that's VERY short has high (relatively speaking) stdev, it will
          contribute LITTLE when adding it to a LONG function.

          This is fine, since we care about the function that dominates the runtime.


        total_time_sec = None
        iterations = 100
        # Total time to run a single repetition
        time_limit_sec = 1 minute
        do:
          start_t = time.time()
          for i in iterations:
            func(...)
          end_t = time.time()
          total_time_sec = end_t - start_t
        while (stdev / mean) > 1% and total_time_sec < time_limit_sec:
        """
        if self.num_calls is not None:
            return

        # Dynamically calculate number of iterations to run for; start with 10 at least.
        iterations = 10
        # max_stdev_percent = 1.
        # time_limit_sec = 60
        # time_limit_sec = 10
        # time_limit_sec = 100

        min_guess_sec = 1.

        def report_decision(total_time_sec, iterations):
            logger.info("> Dynamic iterations for bench_name={b} decided on: {it} iterations".format(
                it=iterations,
                b=bench_name))
            logger.info("  1 repetition takes ~ {sec} seconds".format(sec=total_time_sec))

        # repetition_time_sec = np.zeros(self.repetitions, dtype=np.float32)
        total_time_sec = None
        while True:
            # for r in range(self.repetitions):
            start_t = time.time()
            for i in range(iterations):
                # self._iter(r=-1, i=-1)
                func(*args, **kwargs)
            end_t = time.time()
            total_time_sec = end_t - start_t
            # repetition_time_sec[r] = total_time_sec
            if total_time_sec > self.repetition_time_limit_sec:
                # Took longer than 1 minute to run a single repetition;
                # just run it for that time and the stdev we get is what we get.
                report_decision(total_time_sec, iterations)
                # self.iterations = iterations
                self.num_calls = iterations
                # self.average_time_per_call_no_profile_sec = total_time_sec/float(iterations)
                return

            # stdev = np.std(repetition_time_sec)
            # mean = np.mean(repetition_time_sec)
            # if stdev/mean <= max_stdev_percent/100.:
            #     # After 3 repetitions, the standard-deviation as a percentage of the mean was <= 1%;
            #     # good enough!
            #     break

            if total_time_sec > min_guess_sec:
                # Use current iteration time to guess the minimum number of iterations needed to be >= time_limit_sec.
                #
                # ( 2^p ) * total_time_sec >= time_limit_sec
                # p >= log_2 [ time_limit_sec / total_time_sec ]
                # p = ceil [ log_2 [ time_limit_sec / total_time_sec ] ]
                next_iterations = iterations * 2**math.ceil(np.log2(self.repetition_time_limit_sec / total_time_sec))
                assert iterations < next_iterations
                iterations = next_iterations
            else:
                iterations *= 2

        # report_decision(total_time_sec, iterations)
        # self.iterations = iterations

    @property
    def out_dir(self):
        """
        Output directory is process and phase specific.
        NOTE: this is important so we don't "overwrite" the logs of a concurrent process!

        e.g. for minigo:
                process_name    phase
                ----------      ---------
        process/loop_init/phase/bootstrap
        ├── config.trace_1.json
        ├── dump_event.trace_1.proto
        ├── profile.trace_1.session_0.proto
        └── pyprof.trace_1.proto
        :return:
        """
        assert self.process_name is not None
        assert self.phase is not None
        direc = phase_directory(self.directory, self.process_name, self.phase)
        os.makedirs(direc, exist_ok=True)
        return direc

    def pyprof_proto_path(self, trace_id):
        ret = _j(self.out_dir, "pyprof{bench}{trace}.proto".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(trace_id),
        ))
        return ret

    def pyprof_call_times_path(self, trace_id):
        ret = _j(self.out_dir, "pyprof_call_times{bench}{trace}.pickle".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(trace_id),
        ))
        return ret

    # def unit_test_path(self, trace_id):
    #     ret = _j(self.out_dir, "unit_test{bench}{trace}.pickle".format(
    #         bench=bench_suffix(self.bench_name),
    #         trace=trace_suffix(trace_id),
    #     ))
    #     return ret

    def dump_event_proto_path(self, trace_id):
        ret = _j(self.out_dir, "dump_event{bench}{trace}.proto".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(trace_id),
        ))
        return ret

    def config_path(self, trace_id):
        config_path = _j(self.out_dir, "config{bench}{trace}.json".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(trace_id),
        ))
        return config_path

    def tfprof_path(self, session_id, trace_id):
        return get_tfprof_path(self.out_dir, self.bench_name, session_id, trace_id)

    def profile_time_sec(self, bench_name):
        return self.time_sec[bench_name]

    def _check_no_annotations(self, caller_name):
        if len(self._op_stack) > 0:
            self._failing = True
            raise RuntimeError(self._iml_err_msg(
                "You cannot call {caller} while annotations are active since you'll end up losing tfprof/pyprof event data.".format(
                    caller=caller_name,
                ),
                stack=get_stacktrace(), msg_type="ERROR"))

    def _enable_tracing(self):
        logger.info("IML: enable tracing")

        self._check_no_annotations(caller_name='iml.prof.enable_tracing()')

        self._init_trace_time()

        if not self.disable:
            self._start_pyprof()
            self._start_tfprof()

        if not self.disable or self.training_progress:
            # Q: is setting --iml-training-progress going to result in us recording events during uninstrumented runs...?
            # A: No, the --config option we give to iml-prof ensures various events aren't recorded.
            # NOTE: We want to collect CUDA API call stats for uninstrumented runs also!
            self._start_iml_prof()

        if self.just_sample_util:
            self._init_trace_time()
            self.util_sampler.start()

        self._tracing_enabled = True

    @property
    def tracing_enabled(self):
        return self._tracing_enabled

    def _should_enable_tracing(self):
        if py_config.DEBUG and py_config.DEBUG_GPU_HW:
            if not self._tracing_enabled:
                logger.info(rls_log_msg('GPU_HW',
                        textwrap.dedent(f"""\
                            tracing_enabled = {self._tracing_enabled}
                            has_called_enable_tracing = {self._has_called_enable_tracing}
                            delay_training_loop_iters = {self.delay_training_loop_iters}
                               num_training_loop_iters =  {self.num_training_loop_iters}
                               delay_training_loop_iters =  {self.delay_training_loop_iters}
                            delay_passes = {self.delay_passes}
                               num_passes =  {self.num_passes}
                               delay_passes =  {self.delay_passes}
                        """)))
        if self._tracing_enabled:
            return False
        if not self._has_called_enable_tracing:
            return False
        if self.delay_training_loop_iters is not None:
            return self.num_training_loop_iters >= self.delay_training_loop_iters
        if self.delay_passes is not None:
            return self.num_passes >= self.delay_passes
        return True
        # return not self._tracing_enabled and self._has_called_enable_tracing and (
        #     (
        #         self.delay_training_loop_iters is None or
        #         self.num_training_loop_iters >= self.delay_training_loop_iters
        #     )
        # )

    def enable_tracing(self):
        """
        Turn on IML tracing.

        :return:
        """
        if self.disable and not self.training_progress:
            return

        if self.reports_progress:
            # Wait for iml.prof.report_progress() to get called until we enable tracing.
            # This ensures that we measure the delta in percent_complete'd over the
            # same interval of time we do tracing for.
            self._has_called_enable_tracing = True

        # if self.disable:
        #     return

        if not self.reports_progress:
            self._enable_tracing()

    def _disable_tracing(self):
        logger.info("IML: disable tracing")
        self._stop_pyprof()
        self._stop_tfprof()
        self._stop_iml_prof()
        self._tracing_enabled = False

    # # Calling iml.prof.enable()/disable() repeatedly wouldn't currently support percent_complete tracking...
    # # So let's just not allow disable() for now.
    # def disable_tracing(self):
    #     """
    #     Turn off IML tracing.
    #
    #     :return:
    #     """
    #     if self.reports_progress:
    #         self._delayed_disable = True
    #     else:
    #         self._disable_tracing()

    def start(self, start_utilization_sampler=False, handle_utilization_sampler=False):
        PROFILERS.append(self)

        # Collect GPU utilization info, even for uninstrumented runs.
        self.handle_utilization_sampler = handle_utilization_sampler
        if not self.just_sample_util and ( start_utilization_sampler or handle_utilization_sampler ):
            self._launch_utilization_sampler()

        if self.disable:
            return

        self._start_us = iml_timer.now_us()

        if self.unit_test:
            self.ut.start()

        # If --iml-delay, delay collecting traces until they explicitly call iml.prof.enable().
        if not self.delay:
            self.enable_tracing()

    def _maybe_end_operations(self):
        while len(self._op_stack) != 0:
            self.end_operation(self._cur_operation, skip_finish=True)

    def stop(self):
        PROFILERS.remove(self)

        if self.just_sample_util:
            self.util_sampler.stop()

        # Q: Any way to avoid forgetting to terminate utilization sampler?
        # A: Not really...can add a cleanup() script/python-API to call in the code when the programmer expects to terminate...
        # harder to forget than a call to stop() that's missing a parameter.
        self.maybe_terminate_utilization_sampler(warn_terminated=False)

        if self.disable:
            return

        self._maybe_end_operations()
        self._maybe_finish(finish_now=True, should_exit=False)

    def _start_iml_prof(self):
        if self._iml_prof_enabled or not rlscope_api.is_used():
            return

        if self.debug:
            logger.info('Start iml-prof libcupti tracing')
        if rlscope_api.is_used():
            rlscope_api.enable_tracing()

        self._iml_prof_enabled = True

    def _stop_iml_prof(self):
        if not self._iml_prof_enabled:
            return

        if self.debug:
            logger.info('Stop iml-prof libcupti tracing')

        if rlscope_api.is_used():
            rlscope_api.disable_tracing()

        self._iml_prof_enabled = False

    def _dump_iml_prof(self):
        raise NotImplementedError("TODO: call C++ function that dumps CUDA API stats to protobuf file (rlscope_api.collect())")

    def _start_tfprof(self, skip_init_trace_time=False):
        """
        Meant to be called right before we start measuring individual operations.

        Does setup needed for profiling:
        - Wrap TF library to measure python API time
        - Enable tfprof to hook into session.run(...) for measuring TF-side GPU/C++ API time

        :return:
        """
        if self.process_name is None:
            self._failing = True
            raise RuntimeError("You need to call profiler.set_process_name(...) before profiling.")
        assert self.phase is not None

        if self._tfprof_enabled or self.disable_tfprof:
            if self.disable_tfprof:
                logger.info("Skipping tfprof profiling (--iml-disable-tfprof)")
            return

        # if self.step_start_tracing is None:
        #     self.step_start_tracing = self.steps

        if not skip_init_trace_time:
            self._init_trace_time()

        # sess = self._cur_session(allow_none=True)
        # if sess is None:
        #     if allow_skip:
        #         # They called set_operation before calling set_session/sess.as_default().
        #         # Delay preallocation of tracer until session is set.
        #         return
        #     raise RuntimeError(
        #         "Couldn't find current session; you either need to call Profiler.set_session(sess), "
        #         "or do \"with sess.as_default():\"")

        # self._tfprof_enable_tracing()
        # self.pctx.enable_tracing()

        self._tfprof_enabled = True

    # def _tfprof_enable_tracing(self):
    #     for session in iml_profiler.profiler.session.ACTIVE_SESSIONS:
    #         pctx = ProfileContextManager.get_profile_context(session)
    #         pctx.enable_tracing()

    # def _tfprof_disable_tracing(self):
    #     for session in iml_profiler.profiler.session.ACTIVE_SESSIONS:
    #         pctx = ProfileContextManager.get_profile_context(session)
    #         pctx.disable_tracing()

    def _stop_tfprof(self):
        """
        Stop profiling:
        - Collect trace data, and dump it to file(s)

        :return:
        """
        assert self.process_name is not None
        assert self.phase is not None

        if not self._tfprof_enabled:
            return
        # NOTE: this is when we collect the trace data... keep that in mind when we implement DUMP.

        # self.pctx.disable_tracing()
        # self._tfprof_disable_tracing()

        self._tfprof_enabled = False

    @property
    def is_tfprof_enabled(self):
      return self._tfprof_enabled

    def _check_profiling_started(self):
        global PROFILERS
        started = self in PROFILERS
        if not started:
            self._failing = True
            raise RuntimeError("IML: You need to call profiler.start() before profiling.")

    def _push_operation(self, bench_name):
        # Currently we don't bother to support the following:
        # prof.set_operation('op1')
        # prof.set_operation('op1') <-- probably a bug.
        assert bench_name not in self._op_stack
        assert bench_name != NO_BENCH_NAME
        self._op_stack.append(bench_name)
        if rlscope_api.is_used():
            rlscope_api.push_operation(bench_name)

    def set_max_operations(self, operation, num_pushes):
        if self.disable or self.disable_gpu_hw:
            return
        if rlscope_api.is_used():
            rlscope_api.set_max_operations(operation, num_pushes)

    def _pop_operation(self, bench_name):
        assert self._op_stack[-1] == bench_name
        self._op_stack.pop()
        if rlscope_api.is_used():
            rlscope_api.pop_operation()

    def _start_pass(self):
        assert not self._hw_pass_running
        if py_config.DEBUG and py_config.DEBUG_GPU_HW:
            logger.info(rls_log_msg('GPU_HW', f"start_pass"))
        if rlscope_api.is_used():
            self._hw_pass_running = True
            rlscope_api.start_pass()

    def _end_pass(self):
        assert self._hw_pass_running
        if py_config.DEBUG and py_config.DEBUG_GPU_HW:
            logger.info(rls_log_msg('GPU_HW', f"end_pass"))
        if rlscope_api.is_used():
            self._hw_pass_running = False
            rlscope_api.end_pass()
            self.has_next_pass = self._has_next_pass()
        self.pass_idx += 1

    def _has_next_pass(self):
        assert not self._hw_pass_running
        if py_config.DEBUG and py_config.DEBUG_GPU_HW:
            logger.info(rls_log_msg('GPU_HW', f"has_next_pass"))
        if rlscope_api.is_used():
            return rlscope_api.has_next_pass()
        else:
            return False

    @property
    def _cur_operation(self):
        if len(self._op_stack) == 0:
            return NO_BENCH_NAME
        return self._op_stack[-1]

    def set_operation(self, bench_name):

        should_skip = self.disable or self.disable_pyprof or self.disable_pyprof_annotations or not self._pyprof_enabled

        if should_skip:
            return

        self._check_profiling_started()

        if py_config.DEBUG and py_config.DEBUG_OPERATIONS:
            logger.info("> set_operation(op={op})".format(op=bench_name))

        self._push_operation(bench_name)

        self.start_call_us[bench_name] = iml_timer.now_us()

    def operation(self, operation, skip=False):
        return Operation(operation, prof=self, skip=skip)

    def profile(self, process_name, phase_name=constants.DEFAULT_PHASE, handle_utilization_sampler=True):
        """
        with iml.prof.profile('loop_train_eval', phase_name='sgd_updates'):
            ... code to profile ...

        :param process_name:
        :param phase_name:
        :param handle_utilization_sampler:
            If True, handle start/stopping iml-util-sampler.
            i.e.
            - when profilng start, launch iml-util-sampler.
            - when profilng stops, send SIGTERM to iml-util-sampler.

            handle_utilization_sampler=True makes sense if your training code is contained
            within a single python script and process.

            handle_utilization_sampler=False makes sense for minigo,
            since there are multiple scripts, we make an outer bash script handle
            starting/stopping iml-util-sampler.
        :return:
        """
        return Profile(
            prof=self,
            process_name=process_name,
            phase_name=phase_name,
            handle_utilization_sampler=handle_utilization_sampler,
        )

    def _dump_iml_config(self):
        path = self._iml_config_path
        attrs = dict(self.__dict__)
        def should_keep(attr):
            return type(attrs[attr]) in {dict, list, str, int, float, bool, type(None)}
        for k in list(attrs.keys()):
            if not should_keep(k):
                del attrs[k]
        tensorflow_config = get_tensorflow_config()
        attrs['tensorflow_config'] = tensorflow_config
        attrs['env'] = dict(os.environ)
        attrs['metadata'] = dict(self.metadata)
        if self.debug:
            logger.info("Dump IML configuration information to {path}".format(path=path))
        dump_json(attrs, path)

    def _init_trace_time(self):
        """
        Record the start time-since-epoch of tracing information being collected.

        (i.e. the time should proceed start_time_us of all recorded events)
        """
        if self.start_trace_time_sec is None:
            self.start_trace_time_sec = time.time()

    def _start_pyprof(self):
        if self._pyprof_enabled or self.disable_pyprof:
            if self.disable_pyprof:
                logger.info("Skipping pyprof profiling (--iml-disable-pyprof)")
            return
        if self.debug:
            logger.info("Start pyprof\n{stack}".format(stack=get_stacktrace()))
        self._init_trace_time()
        if self.python:
            # Using cProfile python profiler to collect python events.
            logger.info("IML: Enabling python cProfile profiler")
            self.pyprof.enable()
        else:
            # Use custom function-wrappers around TensorFlow/Simulator C++ libraries to record
            # events marking "Python" time and "TensorFlow C++" / "Simulator" time.
            clib_wrap.enable_tracing()
            # clib_wrap.set_step(self._pyprof_step, expect_traced=True)
            # if (py_config.DEBUG or TF_PRINT_TIMESTAMP) and clib_wrap.is_recording():
            #     logger.info("> RECORDING pyprof_step = {step}".format(step=self._pyprof_step))
        self._pyprof_enabled = True

    def _stop_pyprof(self):
        if not self._pyprof_enabled:
            return
        if self.python:
            self.pyprof.disable()
        else:
            clib_wrap.disable_tracing()
        self._pyprof_enabled = False

    @property
    def _pyprof_step(self):
        """
        Some operations just won't call session.run(...).
        In that case, we cannot depend on using the tfprof step number since
        it won't increment between next_step calls.
        So, just use our internal next_step counter.
        """
        return self.steps

    def end_operation(self, bench_name, skip_finish=False):
        assert bench_name != NO_BENCH_NAME

        should_skip = self.disable or self.disable_pyprof or self.disable_pyprof_annotations or not self._pyprof_enabled

        if not should_skip or ( should_skip and self.training_progress ):
            self._dump_training_progress(debug=self.debug)

        if not should_skip:

            if py_config.DEBUG and py_config.DEBUG_OPERATIONS:
                should_finish = self.should_finish()
                logger.info('> end_operation({op}), tracing time = {sec}, should_finish = {should_finish}'.format(
                    sec=self._total_trace_time_sec(),
                    should_finish=should_finish,
                    op=bench_name))


            # if self.calls_traced > self.num_calls and len(self._op_stack) > 0 and self.calls_traced % WARN_EVERY_CALL_MODULO == 0:
            #     logger.info("> IML: WARNING, use more fine grained operations so we can free memory by dumping traces more frequently")
            #     logger.info("  - calls traced = {calls_traced}, number of calls per-trace = {num_calls}".format(
            #         calls_traced=self.calls_traced,
            #         num_calls=self.num_calls,
            #     ))
            #     logger.info("  - currently active operations: {ops} <-- make these more fine-grained!".format(
            #         ops=self._op_stack))

            if self._cur_operation == NO_BENCH_NAME and bench_name != self._cur_operation:
                """
                start_operation was called, but was skipped since _should_measure_call 
                returned false.
                """
                assert len(self._op_stack) == 0
                return

            if self._cur_operation != bench_name:
                self._failing = True
                raise RuntimeError(textwrap.dedent("""
                Detected non stack-oriented nesting of profiling statements:
                    prof.set_operation({b1})
                    ...
                    prof.end_operation({b2})
                """.format(
                    b1=self._cur_operation,
                    b2=bench_name,
                )))

            self.end_call_us[bench_name] = iml_timer.now_us()
            # self.disable_profiling(bench_name, num_calls=1)

            # Record the last amount of time in between returning
            # from a call to q_forward, and finishing benchmarking.
            # This will include time spent in the tensorflow python API
            if self._pyprof_enabled:
                clib_wrap.record_python_event('Finish python benchmark', self.end_call_us[bench_name])
            # time_sec = (self.end_call_us[bench_name] - self.start_call_us[bench_name])/constants.MICROSECONDS_IN_SECOND
            # if clib_wrap.is_recording():
            #     self.profile_sec.append(time_sec)
            # else:
            #     self.no_profile_sec.append(time_sec)
            op_start_us = self.start_call_us[bench_name]
            op_end_us = self.end_call_us[bench_name]
            rlscope_api.record_event(
                category=constants.CATEGORY_OPERATION,
                start_us=op_start_us,
                duration_us=op_end_us - op_start_us,
                name=bench_name,
            )
            del self.start_call_us[bench_name]
            del self.end_call_us[bench_name]

            self._pop_operation(bench_name)
            # Attribute overhead event to the parent operation (most operations are occuring)
            # NOTE: if no operation is still running, where should we attribute it to?
            # rlscope_api.record_overhead_event(overhead_type='pyprof_annotation', num_events=1)
            if len(self._op_stack) == 0:
                # Attribute annotation overhead to the operation we just finished
                rlscope_api.record_overhead_event_for_operation(overhead_type='pyprof_annotation', operation=bench_name, num_events=1)
            else:
                # Attribute annotation overhead to the parent operation (NOT the one that just finished)
                rlscope_api.record_overhead_event(overhead_type='pyprof_annotation', num_events=1)

            # We terminate annotations if they're been going for too long.
            # if not self.reports_progress:
            #     self._maybe_warn_live_annotations()

            if len(self._op_stack) == 0:
                self.steps += 1
                # self._disable_tracing()

        if self.reports_progress:
            self._maybe_warn_report_progress()

        if not skip_finish and not self.reports_progress:
            # Regarding self.reports_progress:
            # - If this process reports training progress (self.reports_progress),
            #   exit AFTER --iml-trace-time-sec is up AND iml.prof.report_progress has JUST been called.
            # - If this process does NOT report training progress, just exit once --iml-trace-time-sec is up.
            self._maybe_finish(debug=self.debug)

    def set_metadata(self, variables):
        """
        e.g.

        iml.prof.set_metadata({
            'algo': algo,
            'env': env,
        })

        :param variables:
            Dictionary of key/value pairs to record in iml_config.json file.
            This is for convenience when running experiments, call this API isn't neccessary.
        :return:
        """
        self.metadata.update(variables)

        self._maybe_dump_iml_config()

    def _maybe_dump_iml_config(self):
        if self.process_name is not None and self.phase is not None:
            self._dump_iml_config()

    def _is_term_opt_set(self):
        return self.max_timesteps is not None or \
               self.max_training_loop_iters is not None or \
               self.max_passes is not None or \
               self.trace_time_sec is not None

    def _term_opts(self):
        return dict(
            max_timesteps=self.max_timesteps,
            max_training_loop_iters=self.max_training_loop_iters,
            max_passes=self.max_passes,
            trace_time_sec=self.trace_time_sec,
        )

    def _set_term_opt(self, func, opt, value, skip_if_set):
        """
        Only set self.opt if another termination opt isn't already set.
        """
        term_opts = self._term_opts()
        # Should be one of --iml-trace-time-sec, --iml-max-timesteps, --iml-max-training-loop-iters
        assert opt in term_opts
        if skip_if_set and self._is_term_opt_set():
            logger.info(("IML: SKIP {func}({opt}={value}) "
                          "since trace-termination-options are already set: {opts}").format(
                func=func,
                opt=opt,
                value=value,
                opts=pprint_msg(self._term_opts()),
            ))
            return
        logger.info("IML: Setting iml.prof.{var} = {val}".format(
            var=opt,
            val=value,
        ))
        setattr(self, opt, value)

    def _maybe_set_opt(self, opt, value, funcname, skip_if_set):
        """
        Only set self.opt if self.opt is currently None
        """
        if skip_if_set and getattr(self, opt) is not None:
            logger.info(("IML: SKIP {func}({opt}={value}) "
                          "since {opt} is already set: {opts}").format(
                # e.g., 'iml.prof.set_delay_passes',
                func=funcname,
                opt=opt,
                value=value,
                opts={
                    opt: getattr(self, opt),
                },
            ))
            return
        logger.info("IML: Setting iml.prof.{var} = {val}".format(
            var=opt,
            val=value,
        ))
        setattr(self, opt, value)

    def set_max_training_loop_iters(self, max_training_loop_iters, skip_if_set):
        """
        Set the maximum training loop iterations to collect traces for before exiting the training script early.

        :param: skip_if_set : bool
            If True, then ONLY set max_training_loop_iters if the following trace-termination-options have not been
            provided already via cmdline:
                --iml-max-timesteps
                --iml-max-training-loop-iters
                --iml-trace-time-sec

            If False, set max_training_loop_iters (possibly overriding  --iml-max-training-loop-iters)

        :return:
        """
        assert not self._tracing_enabled
        assert not self._has_called_enable_tracing
        self._set_term_opt('iml.prof.set_max_training_loop_iters',
                           'max_training_loop_iters', max_training_loop_iters,
                           skip_if_set)
        self._maybe_dump_iml_config()

    def set_max_passes(self, max_passes, skip_if_set):
        assert not self._tracing_enabled
        assert not self._has_called_enable_tracing
        self._set_term_opt('iml.prof.set_max_passes',
                           'max_passes', max_passes,
                           skip_if_set)
        self._maybe_dump_iml_config()

    def set_delay_passes(self, delay_passes, skip_if_set):
        """
        Set the delay in "passes" (calls to iml.prof.report_progress) before trace collection begins.
        """
        assert not self._tracing_enabled
        assert not self._has_called_enable_tracing
        self._maybe_set_opt('delay_passes', delay_passes, 'iml.prof.set_delay_passes', skip_if_set)
        self._maybe_set_opt('delay', True, 'iml.prof.set_delay', skip_if_set)
        self._maybe_dump_iml_config()

    def set_delay_training_loop_iters(self, delay_training_loop_iters, skip_if_set):
        """
        Set the delay in training loop iterations before trace collection begins.

        :param: skip_if_set : bool
            If True, then ONLY set delay_training_loop_iters if the following have not been
            provided already via cmdline:
                --iml-delay-training-loop-iters

            If False, set delay_training_loop_iters (possibly overriding --iml-delay-training-loop-iters)

        :return:
        """
        assert not self._tracing_enabled
        assert not self._has_called_enable_tracing
        self._maybe_set_opt('delay_training_loop_iters', delay_training_loop_iters, 'iml.prof.set_delay_training_loop_iters', skip_if_set)
        self._maybe_dump_iml_config()

    def set_process_name(self, process_name):
        if process_name == '':
            raise ValueError("IML ERROR: You cannot use an empty-string for process_name")
        self.process_name = process_name
        # clib_wrap.set_process_name(process_name)
        self.init_trace_id()
        # self._maybe_init_profile_context()

        self._maybe_set_metadata()

        # if self.process_name is not None and self.phase is not None:
        #     self._force_load_libcupti()

        self._maybe_dump_iml_config()

    def _maybe_set_metadata(self):
        if rlscope_api.is_used() and \
            ( self.directory is not None and \
              self.process_name is not None and \
              self.machine_name is not None and \
              self.phase is not None ):
            rlscope_api.set_metadata(
                self.directory,
                self.process_name,
                self.machine_name,
                self.phase,
            )

    @property
    def is_root_process(self):
        return self.process_name is None

    def _launch_utilization_sampler(self):
        assert not self.just_sample_util

        if not self.is_root_process:
            logger.info("IML: Warning; you are starting the utilization sampler later than expected (this is not the root process of your training script")

        if self.util_sampler_pid is not None:
            logger.info("IML: Warning; you're already running utilization sampler @ pid={pid}".format(pid=self.util_sampler_pid))
            return

        util_cmdline = ['iml-util-sampler']
        util_cmdline.extend(['--iml-directory', _a(self.directory)])
        # Sample memory-usage of the entire process tree rooted at ths process.
        util_cmdline.extend(['--iml-root-pid', str(os.getpid())])
        if py_config.DEBUG_UTIL_SAMPLER and self.debug:
            util_cmdline.append('--iml-debug')
        # We make sure nvidia-smi runs fast at the VERY START of training
        # (to avoid false alarms when training is busy with the CPU/GPU).
        util_cmdline.append('--skip-smi-check')
        # if self.debug:
        logger.info("> CMDLINE: {cmd}".format(cmd=' '.join(util_cmdline)))
        # self.util_sampler_proc = subprocess.Popen(util_cmdline, creationflags=subprocess.DETACHED_PROCESS)
        self.util_sampler_proc = subprocess.Popen(util_cmdline)
        self.util_sampler_pid = self.util_sampler_proc.pid
        logger.info("IML: CPU/GPU utilization sampler running @ pid={pid}".format(pid=self.util_sampler_pid))

    def _terminate_utilization_sampler(self, warn_terminated=True):
        assert not self.just_sample_util

        assert self.util_sampler_pid is not None
        logger.info("IML: terminating CPU/GPU utilization sampler @ pid={pid}".format(pid=self.util_sampler_pid))

        try:
            proc = psutil.Process(self.util_sampler_pid)
        except psutil.NoSuchProcess as e:
            if warn_terminated:
                logger.info("IML: Warning; tried to terminate utilization sampler @ pid={pid} but it wasn't running".format(pid=self.util_sampler_pid))
            return

        proc.terminate()

    def set_phase(self, phase):
        assert type(phase) == str

        self.phase = phase

        self._maybe_set_metadata()

        # if self.process_name is not None and self.phase is not None:
        #     self._force_load_libcupti()

        self._maybe_dump_iml_config()

        if self.disable:
            return

        if self.unit_test:
            self.ut.set_phase(phase)

        if len(self._op_stack) != 0:
            self._failing = True
            raise RuntimeError("IML: ERROR, you cannot change phases while operations are in-progress: ops = {ops}".format(
                ops=self._op_stack))

        # ProfileContextManager.recreate_sessions_profile_contexts(phase, self.machine_name)

    def maybe_terminate_utilization_sampler(self, warn_terminated):
        if self.handle_utilization_sampler and self.util_sampler_pid is not None:
            self._terminate_utilization_sampler(warn_terminated)

    def finish(self, should_exit=True):
        timer = SimpleTimer("Profiler.finish")
        timer.reset_start_time()
        if self._is_finishing:
            # We've already called this function to terminate tracing.
            #
            # Multiple calls happen to this function since users write their code like this:
            #
            #   with iml.prof.profiler(...):          # -> This registers a iml.prof.stop() handler to be called on exit
            #     iml.prof.report_progress(...)       # -> This calls sys.exit(0) during Profiler.finish()
            #                                         # -> We exit the with block; iml.prof.stop() is called
            #                                         #    and calls into iml.prof.finish()
            return
        if should_exit:
            self._is_finishing = True

        if self._hw_pass_running:
            # Q: Any way to "discard" an incomplete pass?
            self._end_pass()

        self._disable_tracing()
        timer.end_operation('disable_tracing')

        if self.debug:
            logger.info("> IML: finishing profiling\n{stack}".format(stack=get_stacktrace(indent=1)))

        if self.unit_test:
            self.ut.stop()

        self.maybe_terminate_utilization_sampler(warn_terminated=False)
        timer.end_operation('maybe_terminate_utilization_sampler')

        # Record an event [PROC:<process_name>] that marks when this process started/finished execution.
        if not ( self.disable or self.disable_pyprof_interceptions or self.disable_pyprof ):
            self._record_process_event()

        if rlscope_api.is_used():
            # Print sampling results.
            rlscope_api.print()
            timer.end_operation('rlscope_api.print')
            # NOTE: ideally, we should run async_dump() for everything, then wait on everything to finish.
            rlscope_api.await_dump()
            timer.end_operation('rlscope_api.await_dump')
        # NOTE: don't record any events past this point since they will be lost;
        # we will get an abort() error from C++ if that happens.

        # For each active session, schedule its results to be dumped.
        # for sess in ACTIVE_SESSIONS:
        #   self.dump_tfprof(sess)
        # logger.info("> IML: Schedule any remaining traces to be dumped.")
        # for sess in iml_profiler.profiler.session.ACTIVE_SESSIONS:
        #     self._dump_tfprof(sess, debug=self.debug)
        # At the very least, make sure to dump the [PROC:<process_name>] we recorded above.
        # Q: How frequently should we dump pyprof data?
        # A: We'd like to keep it under a file-size limit...but computing exact proto-size isn't practical.
        #    Instead, lets just roughly count the # of events, and use that as a proxy for proto file size.
        # if clib_wrap.should_dump_pyprof():
        #     self._dump_pyprof(debug=self.debug)
        self._dump_process_metadata(debug=self.debug)
        timer.end_operation('_dump_process_metadata')
        self._dump_training_progress(debug=self.debug, dump_always=not self._failing)
        timer.end_operation('_dump_training_progress')

        if self.unit_test:
            logger.info("> IML: _dump_unit_test")
            self._dump_unit_test()
            timer.end_operation('_dump_unit_test')
            logger.info("> IML: _dump_unit_test done")
            # Make sure ALL unit-test data has been recorded before we exit.
            if not self.ut.is_empty:
                self.ut.debug_empty()
                assert self.ut.is_empty

        # Wait on all dump-processes to finish executing.
        # logger.info("> IML: Waiting for trace-dump background threads to complete.")
        # self.bg_dumper.shutdown()
        # timer.end_operation('bg_dumper.shutdown')

        # Wait for any async tfprof trace file dumps in C++ to finish.
        # logger.info("> IML: Wait for tfprof trace-file dumps in TensorFlow C++ to finish.")
        # c_api_util.await_trace_data_dumps()

        # TODO: rlscope_api.await_trace_dumps()

        # logger.info("> IML: Done")

        # Prevent weird bugs from happening at exit, like exceptions thrown during __del__ functions.
        clib_wrap.unwrap_libs()
        timer.end_operation('clib_wrap.unwrap_libs')

        if py_config.DEBUG_CRITICAL_PATH:
            logger.info("Profiler.finish critical-path inflation: {msg}".format(
                msg=pprint_msg(timer),
            ))

        if should_exit:
            logger.info("> IML: Exiting training script early")
            sys.exit(0)

    # def _dump_tfprof(self, session, debug=False):
    #     """
    #     Dump a tfprof proto file for a given session.
    #     Should get called whenever the number of traces for this session exceeds a threshold.
    #
    #     :param session:
    #     :param pctx:
    #     :return:
    #     """
    #     should_skip_dump = False
    #     if self.disable_tfprof_dump:
    #         should_skip_dump = True
    #
    #     if hasattr(session, 'iml_skip_dump') and session.iml_skip_dump:
    #         logger.info('session.iml_skip_dump was set; skipping dumping tfprof for session={s}'.format(
    #             s=session,
    #         ))
    #         should_skip_dump = True
    #
    #     pctx = ProfileContextManager.get_profile_context(session)
    #
    #     trace_id = self.next_trace_id
    #     self.next_trace_id += 1
    #     tfprof_path = self.tfprof_path(session.session_id, trace_id)
    #
    #     if pctx.iml_traced_calls == 0:
    #         # Silently skip dumping this pctx since it contains no trace-data (unless --iml-debug).
    #         if pctx.phase is None and debug:
    #             logger.info("Skip dumping tfprof @ {path}: your training script creates a tf.Session() object that never gets used so it has 0 traced-calls.".format(path=tfprof_path))
    #         elif debug:
    #             logger.info("Skip dumping tfprof @ {path}: since it has 0 traced-calls.".format(path=tfprof_path))
    #         should_skip_dump = True
    #
    #     if not should_skip_dump:
    #         tfprof_dumper = TfprofDumper(trace_id, session, self.process_name, tfprof_path, debug=debug)
    #         tfprof_dumper.dump()
    #
    #     # self.bg_dumper.submit(
    #     #     name='TfprofDumper.dump({path})'.format(path=tfprof_path),
    #     #     fn=tfprof_dumper.dump)
    #     # After process has forked for dumping trace-data, clear the current process' trace-data.
    #
    #     pctx.clear()

    # def _old_dump_tfprof(self, session, debug=False):
    #     """
    #     Dump a tfprof proto file for a given session.
    #     Should get called whenever the number of traces for this session exceeds a threshold.
    #
    #     :param session:
    #     :param pctx:
    #     :return:
    #     """
    #     pctx = ProfileContextManager.get_profile_context(session)
    #     trace_id = self.next_trace_id
    #     # tfprof_path = self.tfprof_path(ses)
    #     tfprof_path = self.tfprof_path(session.session_id, trace_id)
    #     tfprof_dumper = TfprofDumper(trace_id, session, self.process_name, tfprof_path, debug=debug)
    #     self.bg_dumper.submit(
    #         name='TfprofDumper.dump({path})'.format(path=tfprof_path),
    #         fn=tfprof_dumper.dump)
    #     # After process has forked for dumping trace-data, clear the current process' trace-data.
    #     pctx.clear()
    #     self.next_trace_id += 1
    #
    # # TODO: Don't set to True until we've made sure bg_dumper actually works!
    # UNIT_TEST_ASYNC_DUMP = False
    # def _dump_unit_test(self):
    #     assert self.unit_test
    #     assert self.unit_test_name is not None
    #     trace_id = self.next_trace_id
    #     self.next_trace_id += 1
    #     dump_kwargs = {
    #         'directory':self.out_dir,
    #         'trace_id':trace_id,
    #         'bench_name':self.bench_name,
    #         'process_name':self.process_name,
    #         'test_name':self.unit_test_name,
    #     }
    #     if Profiler.UNIT_TEST_ASYNC_DUMP:
    #         self.bg_dumper.submit(
    #             name='UnitTestDataDumper.dump(trace_id={trace_id})'.format(
    #                 trace_id=trace_id),
    #             fn=self.ut.dump,
    #             **dump_kwargs)
    #     else:
    #         self.ut.dump(**dump_kwargs)
    #     # We might be in the middle of a phase...
    #     # i.e.
    #     # - phase_start[cur_phase] is recorded, but NOT phase_end[cur_phase]
    #     # - The dump we are performing right now WON'T include cur_phase; but a future dump will
    #     # => clear should NOT forget what the current phase is.
    #     self.ut.clear_dump()

    # @property
    # def _should_dump_pyprof(self):
    #     return clib_wrap.should_dump_pyprof()

    # TODO: decide where to call dump_tfprof / dump_pyprof from.
    # def _old_dump_pyprof(self, config_kwargs=dict(), debug=False):
    #     if self.disable_pyprof_dump:
    #         return
    #     start_sec = time.time()
    #     pyprof_trace = clib_wrap.get_pyprof_trace()
    #     trace_id = self.next_trace_id
    #     pyprof_trace_key = self.pyprof_proto_path(trace_id)
    #     self.pyprof_dump_manager.put(pyprof_trace_key, pyprof_trace)
    #     pyprof_dumper = PyprofDumper(
    #         trace_id=trace_id,
    #         config_path=self.config_path(trace_id),
    #         c_lib_func_pyprof_pattern=self.c_lib_func_pyprof_pattern,
    #         num_calls=self.num_calls,
    #         start_measuring_call=self.start_measuring_call,
    #         average_time_per_call_sec=self.average_time_per_call_sec,
    #         average_time_per_call_no_profile_sec=self.average_time_per_call_no_profile_sec,
    #         config_kwargs=config_kwargs,
    #         process_name=self.process_name,
    #         phase=self.phase,
    #         pyprof_proto_path=self.pyprof_proto_path(trace_id),
    #         pyprof_call_times_path=self.pyprof_call_times_path(trace_id),
    #         pyprof_step=self._pyprof_step,
    #         pyprof_dump_manager=self.pyprof_dump_manager,
    #         pyprof_trace_key=pyprof_trace_key,
    #         debug=debug)
    #     self.next_trace_id += 1
    #     self.bg_dumper.submit(
    #         name="PyprofDumper.dump({path})".format(
    #             path=self.pyprof_proto_path(trace_id)),
    #         fn=pyprof_dumper.dump,
    #     )
    #     end_sec = time.time()
    #     time_sec = end_sec - start_sec
    #     if py_config.DEBUG:
    #         logger.info("Dump pyprof took {sec} seconds on the critical path".format(
    #             sec=time_sec,
    #         ))

    def _dump_training_progress(self, debug=False, sync=False, dump_always=False):
        """
        :param debug:
        :param sync:
        :param dump_always:
            Regardless of whether 10 seconds have expired, dump training progress.
            Useful for dumping at exit.
        :return:
        """
        start_sec = time.time()
        trace_id = self.next_trace_id

        # Some training scripts (e.g. minigo) have a more complex training loop that makes it difficult for us to
        # record training progress; in particular iml.prof.report_progress calls technically needs to happen
        # "across processes".  Currently we don't support that.
        # if dump_always:
        #     assert self.phase in self._incremental_training_progress
        #     assert self._incremental_training_progress[self.phase].can_dump(self.reports_progress, expect_true=True)

        now_sec = time.time()
        if self.phase in self._incremental_training_progress and \
                self._incremental_training_progress[self.phase].can_dump(self.reports_progress) and (
                dump_always or \
                self._last_dumped_training_progress is None or \
                now_sec - self._last_dumped_training_progress > DUMP_TRAINING_PROGRESS_EVERY_SEC
        ):
            self._last_dumped_training_progress = now_sec
        else:
            # Skip dumping training_progress, 10 seconds haven't expired yet.

            # logger.info("Skip dumping IncrementalTrainingProgress")
            # fields = {
            #     'phase': self.phase,
            #     'incremental_training_progress': self._incremental_training_progress,
            #     'dump_always': dump_always,
            #     'last_dumped_training_progress': self._last_dumped_training_progress,
            # }
            # if self.phase in self._incremental_training_progress:
            #     fields['can_dump'] = self._incremental_training_progress[self.phase].can_dump(self.reports_progress)
            # logger.info(pprint_msg(fields))

            return

        training_progress = self._incremental_training_progress[self.phase].as_proto()
        proto_name = training_progress.__class__.__name__

        training_progress_proto_path = self._training_progress_proto_path(trace_id)
        os.makedirs(_d(training_progress_proto_path), exist_ok=True)
        dumper = ProtoDumper(
            name='IncrementalTrainingProgressDumper',
            trace_id=trace_id,
            proto=training_progress,
            proto_path=training_progress_proto_path,
            debug=debug or py_config.DEBUG)

        self.next_trace_id += 1
        # self.bg_dumper.submit(
        #     name="{name}.dump({path})".format(
        #         name=dumper.name,
        #         path=training_progress_proto_path),
        #     fn=dumper.dump,
        #     sync=sync,
        # )
        dumper.dump()
        end_sec = time.time()
        time_sec = end_sec - start_sec
        if py_config.DEBUG:
            logger.info("Dump {proto} took {sec} seconds on the critical path".format(
                proto=proto_name,
                sec=time_sec,
            ))

    def _dump_process_metadata(self, debug=False, sync=False):
        start_sec = time.time()
        trace_id = self.next_trace_id
        process_metadata = ProcessMetadata()

        process_metadata.process_name = self.process_name
        process_metadata.phase = self.phase
        process_metadata.machine_name = self.machine_name

        if self.parent_process_name is not None:
            process_metadata.parent_process_name = self.parent_process_name

        # Q: multiple processes reporting training progress...consider that an error?
        # if self.reports_progress and self.percent_complete is None:
        #     self._failing = True
        #     raise RuntimeError("IML ERROR: profiler was created with iml.handle_iml_args(..., reports_progress=True), but process NEVER called iml.prof.report_progress(...)")

        # This should be prevented from self.report_progress(...)
        assert not(not self.reports_progress and self.percent_complete is not None)

        if self.percent_complete is not None:
            process_metadata.training_progress.content_code = TP_HAS_PROGRESS
            # Measure the delta of training completed over the course of training.
            # This is important since, if we delay trace collection until warmup completes,
            # we don't want to inflate the percent_complete'd over that duration of training time.
            percent_complete = self.percent_complete - self._start_percent_complete
            if self.debug:
                logger.info("percent_complete ({perc}) = latest_percent_complete ({latest}) - start_percent_complete ({start})".format(
                    perc=percent_complete,
                    latest=self.percent_complete,
                    start=self._start_percent_complete,
                ))
            process_metadata.training_progress.percent_complete = percent_complete
            # Q: Is this safe is self.num_timestamps is None? NO
            if self.num_timesteps is not None:
                num_timesteps = self.num_timesteps - self._start_num_timesteps
                process_metadata.training_progress.num_timesteps = num_timesteps
            if self.total_timesteps is not None:
                process_metadata.training_progress.total_timesteps = self.total_timesteps
        else:
            process_metadata.training_progress.content_code = TP_NO_PROGRESS

        process_metadata_proto_path = self._process_metadata_proto_path(trace_id)
        dumper = ProcessMetadataDumper(
            trace_id=trace_id,
            process_metadata=process_metadata,
            process_metadata_proto_path=process_metadata_proto_path,
            debug=debug)
        self.next_trace_id += 1
        # self.bg_dumper.submit(
        #     name="ProcessMetadataDumper.dump({path})".format(
        #         path=process_metadata_proto_path),
        #     fn=dumper.dump,
        #     sync=sync,
        # )
        dumper.dump()
        end_sec = time.time()
        time_sec = end_sec - start_sec
        if py_config.DEBUG:
            logger.info("Dump ProcessMetaData took {sec} seconds on the critical path".format(
                sec=time_sec,
            ))

    def _process_metadata_proto_path(self, trace_id):
        ret = _j(self.out_dir, "process_metadata{bench}{trace}.proto".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(trace_id),
        ))
        return ret

    def _training_progress_proto_path(self, trace_id):
        ret = _j(self.out_dir, "training_progress{bench}{trace}.proto".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(trace_id),
        ))
        return ret

    @property
    def _iml_config_path(self):
        return get_iml_config_path(self.out_dir, self.bench_name)

    def _fixup_tfprof(self, path):
        """
        Add profiler specific data to ProfileProto tfprof protobuf file.

        In particular:
        - process_name
        - phase
        """
        with open(path, 'rb') as f:
            proto = ProfileProto()
            proto.ParseFromString(f.read())
        proto.process_name = self.process_name
        proto.phase = self.phase
        with open(path, 'wb') as f:
            f.write(proto.SerializeToString())

    def _maybe_warn_live_annotations(self):
        """
        If we've exceed tracing time-limit (--iml-trace-time-sec), but there are still live annotations,
        warn the user.
        """
        total_trace_time_sec = self._total_trace_time_sec()
        if self.trace_time_sec is not None and total_trace_time_sec is not None and \
                total_trace_time_sec > self.trace_time_sec and \
                len(self._op_stack) > 0 and \
                ( self._last_warned_trace_time_sec is None or time.time() - self._last_warned_trace_time_sec >= WARN_EVERY_SEC ):
            logger.warning(textwrap.dedent("""\
            IML: Warning; tracing time (sec) has exceeded limit (--iml-trace-time-sec {limit_sec}), 
            but annotations are still active:
              process_name = {proc}
              phase_name = {phase}
              Active annotations:
                {annotations}
              Stacktrace:
            {stack}
            """).format(
                sec=total_trace_time_sec,
                limit_sec=self.trace_time_sec,
                proc=self.process_name,
                phase=self.phase,
                annotations=self._op_stack,
                stack=get_stacktrace(indent=2),
            ))
            self._last_warned_trace_time_sec = time.time()

    def _maybe_warn_report_progress(self):
        """
        If the process is responsible for calling iml.prof.report_progress(...) and we have been
        collecting trace-data for much longer than we intended, they may have forgotten to call
        iml.prof.report_progress(...).

        Warn them every 30 seconds in that case.
        """
        total_trace_time_sec = self._total_trace_time_sec()
        if self.trace_time_sec is None or total_trace_time_sec is None:
            return
        warn_idx = int((total_trace_time_sec - self.trace_time_sec) / REPORT_PROGRESS_WARN_EVERY_SEC)
        if total_trace_time_sec > self.trace_time_sec and (
            warn_idx > 1 and (
                self._last_warned_report_progress_idx is None or
                warn_idx > self._last_warned_report_progress_idx
            )
        ):
            logger.warning(textwrap.dedent("""\
            IML: Warning; tracing time so far ({sec} sec) has exceeded tracing time-limit (--iml-trace-time-sec {limit_sec}), but process={proc} 
            hasn't called iml.prof.report_progress(...); did you forget to call this in that process?
              process_name = {proc}
              phase_name = {phase}
              Active annotations:
                {annotations}
              Stacktrace:
            {stack}
            """).format(
                sec=self._total_trace_time_sec(),
                limit_sec=self.trace_time_sec,
                proc=self.process_name,
                phase=self.phase,
                annotations=self._op_stack,
                stack=get_stacktrace(indent=2),
            ))
            self._last_warned_report_progress_idx = warn_idx

    def _iml_err_msg(self, msg, stack=None, msg_type='Warning'):
        if stack is None:
            stack = get_stacktrace()
        return textwrap.dedent("""\
            IML: {msg_type}; {msg} 
              process_name = {proc}
              phase_name = {phase}
              Active annotations:
                {annotations}
              Stacktrace:
            {stack}
            """).format(
            msg_type=msg_type,
            msg=msg,
            proc=self.process_name,
            phase=self.phase,
            annotations=self._op_stack,
            stack=textwrap.indent(stack, prefix="  "*2),
        )

    def should_finish(self, finish_now=False, skip_finish=False):
        total_trace_time_sec = self._total_trace_time_sec()
        ret = finish_now or (
            (
                not rlscope_api.is_used() or self.disable_gpu_hw or not self.has_next_pass
            ) and (
                (
                    self.num_traces is not None and
                    self.next_trace_id >= self.num_traces
                ) or (
                    total_trace_time_sec is not None and
                    self.trace_time_sec is not None
                    and total_trace_time_sec >= self.trace_time_sec
                ) or (
                    self.max_timesteps is not None and
                    self.num_timesteps is not None and
                    self.num_timesteps >= self.max_timesteps
                ) or (
                    self.max_training_loop_iters is not None and
                    self.num_training_loop_iters >= zero_if_none(self.delay_training_loop_iters) + self.max_training_loop_iters
                ) or (
                    self.max_passes is not None and
                    self.num_passes >= zero_if_none(self.delay_passes) + self.max_passes
                )
            )
        )
        self._should_finish_idx += 1
        if py_config.DEBUG and (ret or self._should_finish_idx % 1000 == 0):
            logger.info(textwrap.indent(textwrap.dedent("""
            - process_name = {proc}
            - finish_now = {finish_now}
            - skip_finish = {skip_finish}
            """.format(
                proc=self.process_name,
                finish_now=ret,
                skip_finish=skip_finish,
            )), prefix="  "))
            if not self.disable_gpu_hw:
                logger.info(textwrap.indent(textwrap.dedent("""
                - has_next_pass = {has_next_pass}""".format(
                    has_next_pass=self.has_next_pass,
                )), prefix="  "))
            if self.num_traces is not None:
                logger.info(textwrap.indent(textwrap.dedent("""
                - self.next_trace_id >= self.num_traces = {next_bool}
                  - self.next_trace_id = {next_trace_id}
                  - self.num_traces = {num_traces}""".format(
                    num_traces=self.num_traces,
                    next_bool=self.next_trace_id >= self.num_traces,
                    next_trace_id=self.next_trace_id,
                )), prefix="  "))
            if total_trace_time_sec is not None and self.trace_time_sec is not None:
                logger.info(textwrap.indent(textwrap.dedent("""
                - total_trace_time_sec >= self.trace_time_sec = {total_bool}
                  - total_trace_time_sec = {total_trace_time_sec}
                  - self.trace_time_sec = {trace_time_sec}""".format(
                    total_bool=total_trace_time_sec >= self.trace_time_sec,
                    total_trace_time_sec=total_trace_time_sec,
                    trace_time_sec=self.trace_time_sec,
                )), prefix="  "))
            if self.max_timesteps is not None and self.num_timesteps is not None:
                logger.info(textwrap.indent(textwrap.dedent("""
                - self.num_timesteps >= self.max_timesteps = {bool}
                  - self.num_timesteps = {num_timesteps}
                  - self.max_timesteps = {max_timesteps}""".format(
                    bool=self.num_timesteps >= self.max_timesteps,
                    num_timesteps=self.num_timesteps,
                    max_timesteps=self.max_timesteps,
                )), prefix="  "))
            if self.max_training_loop_iters is not None:
                logger.info(textwrap.indent(textwrap.dedent("""
                - self.num_training_loop_iters >= self.delay_training_loop_iters + self.max_timesteps = {bool}
                  - self.delay_training_loop_iters = {delay_training_loop_iters}
                  - self.num_training_loop_iters = {num_training_loop_iters}
                  - self.max_training_loop_iters = {max_training_loop_iters}""".format(
                    bool=self.num_timesteps >= zero_if_none(self.delay_training_loop_iters) + self.max_training_loop_iters,
                    num_training_loop_iters=self.num_training_loop_iters,
                    delay_training_loop_iters=self.delay_training_loop_iters,
                    max_training_loop_iters=self.max_training_loop_iters,
                )), prefix="  "))
            if self.max_passes is not None:
                logger.info(textwrap.indent(textwrap.dedent("""
                - self.num_passes >= self.delay_passes + self.max_timesteps = {bool}
                  - self.delay_passes = {delay_passes}
                  - self.num_passes = {num_passes}
                  - self.max_passes = {max_passes}""".format(
                    bool=self.num_passes >= zero_if_none(self.delay_passes) + self.max_passes,
                    num_passes=self.num_passes,
                    delay_passes=self.delay_passes,
                    max_passes=self.max_passes,
                )), prefix="  "))
        return ret

    def _total_trace_time_sec(self):
        if self.start_trace_time_sec is None:
            return None
        now_sec = time.time()
        return now_sec - self.start_trace_time_sec

    def _record_process_event(self):
        self._stop_us = iml_timer.now_us()
        # Record a "special" operation event that spans the prof.start()/stop() calls
        # for the currently running process.
        assert self._start_us is not None
        assert self._stop_us is not None
        event_name = op_process_event_name(self.process_name)
        # clib_wrap.set_step(self._pyprof_step,
        #                    ignore_disable=True)
        # BUG TODO: Should we use our own constants.CATEGORY_PROCESS for process-events?
        # Otherwise, we may be unable to disambiguate from a constants.CATEGORY_OPERATION of the same name as the process.
        # Looks like we recorded it as [PROC:<process-name>] to prevent conflicts.
        # I cannot remember how this is handled downstream during analysis.
        # clib_wrap.record_event(constants.CATEGORY_OPERATION, event_name, self._start_us, self._stop_us,
        #                        ignore_disable=True)
        rlscope_api.record_event(
            category=constants.CATEGORY_OPERATION,
            start_us=self._start_us,
            duration_us=self._stop_us - self._start_us,
            name=event_name)

    def report_progress(self, percent_complete, num_timesteps=None, total_timesteps=None):
        # if not self.disable or ( self.disable and self.training_progress ):

        if self.disable and not self.training_progress:
            return

        if py_config.DEBUG and py_config.DEBUG_REPORT_PROGRESS_ALL:
            logger.info("[report-progress] vars={vars}\n{stack}".format(
                vars=dict(
                    percent_complete=percent_complete,
                    num_timesteps=num_timesteps,
                    total_timesteps=total_timesteps,
                    phase=self.phase,
                ),
                stack=get_stacktrace(indent=1)
            ))

        if not self.reports_progress:
            self._failing = True
            raise RuntimeError(
                textwrap.dedent("""\
                IML ERROR: profiler was created with iml.handle_iml_args(..., reports_progress=False), but process made unexpected call to iml.prof.report_progress(...).
                If you wish to have process_name={proc} record training progress, call iml.handle_iml_args(..., reports_progress=True), 
                and make sure its the ONLY process that does so.
                """).format(proc=self.process_name))

        if not ( 0. <= percent_complete <= 1. ):
            self._failing = True
            raise RuntimeError(
                textwrap.dedent("""\
                IML ERROR: iml.prof.report_progress(percent_complete=...) expects:
                  0 <= percent_complete <= 1
                But saw percent_complete={perc}
                  
                Typical usage looks like:
                
                  # The training loop of your ML script:
                  for t in range(total_timesteps):
                      iml.prof.report_progress(percent_complete=t/float(total_timesteps))
                """).format(
                    perc=percent_complete,
                ))

        if self._should_enable_tracing():
            self._check_no_annotations(caller_name='iml.prof.report_progress()')
            self._enable_tracing()
            # percent_complete when tracing begins.
            self._start_percent_complete = percent_complete
            self._start_num_timesteps = num_timesteps

            if self.phase not in self._incremental_training_progress:
                self._incremental_training_progress[self.phase] = RecordedIncrementalTrainingProgress(self.machine_name, self.process_name, self.phase)
            self._incremental_training_progress[self.phase].report_start_of_progress(percent_complete, num_timesteps, total_timesteps, self.start_trace_time_sec)

        if self._tracing_enabled and percent_complete < 1 and self._hw_pass_running:
            self._end_pass()

        if py_config.DEBUG and py_config.DEBUG_GPU_HW:
            logger.info(rls_log_msg('GPU_HW', f"tracing_enabled = {self._tracing_enabled}, percent_complete = {percent_complete}"))
        if self._tracing_enabled and percent_complete > 0 and percent_complete < 1.:
            self._start_pass()

        if self.max_timesteps is not None and num_timesteps is None:
            self._failing = True
            raise RuntimeError("IML ERROR: if you use --iml-max-timesteps, you must call iml.prof.report_progress(num_timesteps=NUMBER)")

        if num_timesteps is not None:
            self.num_timesteps = num_timesteps

        if self.total_timesteps is not None:
            self.total_timesteps = total_timesteps

        if self.percent_complete is not None and percent_complete < self.percent_complete:
            self._failing = True
            raise RuntimeError("IML ERROR: percent_complete should be monotonically increasing but saw {from_perc} -> {to_perc}".format(
                from_perc=self.percent_complete,
                to_perc=percent_complete,
            ))
        self.percent_complete = percent_complete

        if self.phase not in self._incremental_training_progress:
            self._incremental_training_progress[self.phase] = RecordedIncrementalTrainingProgress(self.machine_name, self.process_name, self.phase)
        self._incremental_training_progress[self.phase].report_progress(percent_complete, num_timesteps, total_timesteps, self.start_trace_time_sec)

        dump_always = (percent_complete == 1)
        if dump_always:
            # If this fails, then your training loop executed zero iterations,
            # so iml.prof.report_progress was NEVER called.
            #
            # Q: Should we allow this by making the phase basically 0 seconds...?
            if not self.tracing_enabled:
                self._failing = True
                raise RuntimeError("IML ERROR: profiler was created with iml.handle_iml_args(..., reports_progress=True), but process NEVER called iml.prof.report_progress(...)")
            # assert self.tracing_enabled
        self._dump_training_progress(debug=self.debug, dump_always=dump_always)

        self._maybe_finish(debug=self.debug)

        if self._has_called_enable_tracing:
            # They've called iml.prof.enable_tracing() since their algorithm has "warmed up";
            # start counting training loop iterations.
            self.num_training_loop_iters += 1
            self.num_passes += 1

    def _maybe_finish(self, finish_now=False, should_exit=True, debug=False):

        should_finish = self.should_finish(finish_now)
        if not should_finish:
            return

        self._stop_us = iml_timer.now_us()

        while len(self._op_stack) > 0:
            # Pass skip_finish=True to avoid recursively calling this.
            self.end_operation(self._cur_operation, skip_finish=True)

        self.finish(should_exit=should_exit)

    @property
    def total_calls_to_run(self):
        if self.start_measuring_call is None:
            return self.num_calls
        return self.num_calls + self.start_measuring_call - 1

    # def _force_load_libcupti(self):
    #     """
    #     TensorFlow calls dlopen("libcupti.so") lazily.
    #     Instead of modifying tensorflow to load it eagerly, lets just trigger the code-path that loads it.
    #     In particular, enable tfprof briefly for a very simple session.run() call.
    #
    #     libcupti takes about 0.001841 seconds to load with dlopen().
    #
    #     You can observe this by running like this:
    #     $ export TF_DEBUG_LOAD_LIBRARY=yes
    #     $ train.py ...
    #     ...
    #     2019-07-26 15:50:42.337735: I tensorflow/core/platform/posix/load_library.cc:64] > LoadLibrary library=libcupti.so.10.0 took 0.001841 sec
    #
    #     :return:
    #     """
    #     if self._loaded_libcupti:
    #         return
    #
    #     logger.info("Forcing libcupti to load before tracing begins.")
    #
    #     import tensorflow as tf
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     graph = tf.Graph()
    #     sess = tf.Session(graph=graph, config=config)
    #     # We don't want to keep any of the collected traces from this.
    #     sess.iml_skip_dump = True
    #
    #     # NOTE: we aren't actually beginning tracing for the problem, hence skip_init_trace_time=True.
    #     self._start_tfprof(skip_init_trace_time=True)
    #
    #     name = 'ForceLoadLibcupti'
    #     N = 1000
    #     zeros = np.zeros((N, N))
    #     with sess, tf.name_scope(name):
    #         a = tf.placeholder(float, name='a')
    #         b = tf.placeholder(float, name='b')
    #         c = a * b
    #
    #         feed_dict = {
    #             a: zeros,
    #             b: zeros,
    #         }
    #         c_result = sess.run(c, feed_dict=feed_dict)
    #         assert np.equal(c_result, 0.).all()
    #
    #     self._stop_tfprof()
    #
    #     self._loaded_libcupti = True

# class CUDAProfiler:
#     def __init__(self):
#         # NOTE: CUDA profiling output has already been specified when this script was launched.
#         # self.profile_basename = profile_basename
#         self.already_enabled = False
#
#     def start(self):
#         # NOTE: we assume the CUDA
#         self.already_enabled = cudaprofile.is_profiler_enabled()
#         if not self.already_enabled:
#             cudaprofile.start()
#
#     def stop(self):
#         if not self.already_enabled:
#             cudaprofile.stop()
#
#     def dump(self):
#         # Dumping is performed externally by nvprof once the program terminates.
#         pass
#
#     def __enter__(self):
#         self.start()
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.stop()

class Operation:
    def __init__(self, operation, prof, skip):
        self.operation = operation
        self.prof = prof
        self.skip = skip

    def __enter__(self):
        if not self.skip:
            self.prof.set_operation(self.operation)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Q: Should we call this when an exception is thrown?
        if not self.skip:
            self.prof.end_operation(self.operation)

class Profile:
    def __init__(self, prof, process_name, phase_name=constants.DEFAULT_PHASE, handle_utilization_sampler=True):
        self.process_name = process_name
        self.phase_name = phase_name
        self.prof = prof
        self.handle_utilization_sampler = handle_utilization_sampler

    def __enter__(self):
        self.prof.set_process_name(self.process_name)
        self.prof.set_phase(self.phase_name)
        self.prof.start(handle_utilization_sampler=self.handle_utilization_sampler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prof.stop()

class PythonProfiler:
    """
    Run profiler on python code block.

    with profiler(python_profile_basename="some/profile"):
            # profile this code
    """
    def __init__(self, directory,
                 # record_call_times=True,
                 clock='monotonic_clock'):
        self.profile = cProfile.Profile()
        self.directory = directory
        self.clock = clock
        assert self.clock in ['monotonic_clock']
        # self.use_cycle_counter = (self.clock == 'cycle_counter')
        self.use_monotonic_clock = (self.clock == 'monotonic_clock')
        # self.record_call_times = record_call_times

        # assert not ( self.use_cycle_counter and self.use_monotonic_clock )

        # if self.use_cycle_counter:
        #     self.profile.make_use_cycle_counter()

        if self.use_monotonic_clock:
            if hasattr(self.profile, 'make_use_monotonic_clock'):
                self.profile.make_use_monotonic_clock()
            else:
                logger.info("WARNING: couldn't enable monotonic_clock for cProfiler; "
                      "are you using a modified python3 with support for collecting raw start/end timestamps?")

        # if self.record_call_times:
        if hasattr(self.profile, 'make_record_call_times'):
            self.profile.make_record_call_times()
        else:
            logger.info("WARNING: couldn't enable make_record_call_times for cProfiler; "
                  "are you using a modified python3 with support for collecting raw start/end timestamps?")

    def __enter__(self):
        self.start()

    def enable(self):
        self.start()

    def disable(self):
        self.stop()

    def start(self):
        self.profile.enable()

    def stop(self):
        self.profile.disable()

    def dump(self, call_times_path, process_name, phase,
             pstats_txt_path=None,
             pstats_path=None):

        # if pstats_txt_path is not None:
        # sortby = ('calls', 'filename', 'name', 'line')
        sortby = ('tottime', 'filename', 'line', 'name')
        # os.makedirs(os.path.dirname(pstats_txt_path), exist_ok=True)
        # with open(pstats_txt_path, mode='w') as f:
        #     # ps = pstats.Stats(self.profile, stream=f).sort_stats(*sortby)
        #     ps = pstats.Stats(self.profile)
        #     # if self.record_call_times:
        #     call_times = ps.call_times
        #     # ps.print_stats()
        ps = pstats.Stats(self.profile)
        call_times = ps.call_times

        # if self.record_call_times:
        # self._dump_call_times_json(call_times, call_times_path)
        call_times_data = {
            'process_name': process_name,
            'phase': phase,
            'call_times': call_times,
        }
        self._dump_call_times_pickle(call_times_data, call_times_path)

        os.makedirs(os.path.dirname(self._prof_path), exist_ok=True)
        if pstats_path is not None:
            ps.dump_stats(pstats_path)

        # Clear any python profiler data.
        self.profile.clear()

    def _dump_call_times_pickle(self, data, call_times_path):
        os.makedirs(os.path.dirname(call_times_path), exist_ok=True)
        logger.info("> dump pyprof call_times data @ {path}".format(
            path=call_times_path))
        with open(call_times_path, 'wb') as f:
            # -1 specifies highest binary protocol
            pickle.dump(data, f, -1)

    def _dump_call_times_json(self, call_times, call_times_path):

        # Tuple keys are not OK; convert to strings.
        new_call_times = dict()
        for func_tuple, times in call_times.items():
            func = pyprof_func_std_string(func_tuple)
            new_call_times[func] = times
        json.dump(new_call_times,
                  codecs.open(call_times_path, mode='w', encoding='utf-8'),
                  sort_keys=True, indent=4)

    @property
    def _prof_path(self):
        ret = _j(self.directory, "python_profile.prof")
        return ret

    @property
    def _stats_path(self):
        return _j(self.directory, "python_profile.txt")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def add_gflags_argument(flags, *args, **kwargs):
    """
    Translate add_argument from argparse into gflags DEFINE_* call.

    :param flags:
        from absl import flags
                         -----
    :param args:
    :param kwargs:
        ArgumentParser.add_argument(...)
                                    ---
    :return:
    """

    def raise_error():
        raise NotImplementedError("Not sure how to translate argparse argument into absl.flags argument; add_argument(...) was called with\n{msg}".format(
            msg=textwrap.indent(pprint.pprint({
                'args': args,
                'kwargs': kwargs,
            }), prefix='  '),
        ).rstrip())

    def gflags_name(opt, underscores=False):
        gflags_opt = opt
        gflags_opt = re.sub('^--', '', gflags_opt)
        if underscores:
            gflags_opt = re.sub('-', '_', gflags_opt)
        return gflags_opt

    def mark_required(gflags_opt):
        required = kwargs.get('required', False)
        if required:
            flags.mark_flag_as_required(gflags_opt)

    opt = args[0]
    def _add_argument(gflags_opt):
        arg_type = kwargs.get('type', None)
        help = kwargs.get('help', None)
        dflt = kwargs.get('default', None)

        action = kwargs.get('action', None)
        if action is not None:
            if action == 'store_true':
                assert dflt is None
                dflt = False
            elif action == 'store_false':
                assert dflt is None
                dflt = True
            else:
                raise_error()
            flags.DEFINE_bool(name=gflags_opt, default=dflt, help=help)
            mark_required(gflags_opt)
            return

        if arg_type is None:
            flags.DEFINE_string(name=gflags_opt, default=dflt, help=help)
            mark_required(gflags_opt)
            return
        if arg_type == int:
            flags.DEFINE_integer(name=gflags_opt, default=dflt, help=help)
            mark_required(gflags_opt)
            return
        elif arg_type == float:
            flags.DEFINE_float(name=gflags_opt, default=dflt, help=help)
            mark_required(gflags_opt)
            return
        else:
            pass

        raise_error()

    dashed_gflags_opt = gflags_name(opt)
    # underscore_gflags_opt = gflags_name(opt, underscores=True)

    _add_argument(dashed_gflags_opt)
    # _add_argument(underscore_gflags_opt)

def fix_gflags_iml_args(FLAGS):
    for attr in dir(FLAGS):
        if re.search(r'^iml-', attr):
            gflags_opt = re.sub(r'-', '_', attr)
            if not hasattr(FLAGS, gflags_opt):
                FLAGS[gflags_opt] = FLAGS[attr]

def add_argument(parser, *args, **kwargs):
    if isinstance(parser, argparse.ArgumentParser) or isinstance(parser, argparse._ArgumentGroup):
        parser.add_argument(*args, **kwargs)
    elif isinstance(parser, ClickCtx):
        add_click_argument(parser, *args, **kwargs)
    else:
        add_gflags_argument(parser, *args, **kwargs)

# decorator
class ClickCtx:
    def __init__(self, f):
        self.f = f

    def add_argument(self, *args, **kwargs):
        # import click
        # ArgumentClass = kwargs.pop('cls', click.Argument)
        # click.decorators._param_memo(self.f, ArgumentClass(args, **kwargs))
        # return self.f
        return self._option(*args, **kwargs)

    def _option(self, *param_decls, **attrs):
        """Attaches an option to the command.  All positional arguments are
        passed as parameter declarations to :class:`Option`; all keyword
        arguments are forwarded unchanged (except ``cls``).
        This is equivalent to creating an :class:`Option` instance manually
        and attaching it to the :attr:`Command.params` list.

        :param cls: the option class to instantiate.  This defaults to
                    :class:`Option`.
        """
        import click
        # def decorator(f):
        # Issue 926, copy attrs, so pre-defined options can re-use the same cls=
        option_attrs = attrs.copy()

        if 'help' in option_attrs:
            option_attrs['help'] = inspect.cleandoc(option_attrs['help'])
        OptionClass = option_attrs.pop('cls', click.Option)
        click.decorators._param_memo(self.f, OptionClass(param_decls, **option_attrs))
        return self.f
        # return decorator


def click_add_arguments():
    def decorator(f):
        click_ctx = ClickCtx(f)
        add_iml_arguments(click_ctx)
        return f
    return decorator

def add_click_argument(click_ctx : ClickCtx, *args, **kwargs):
    """
    Translate add_argument from argparse into click  @click.argument(...) call.

    :param click:
        import click
               -----
    :param args:
    :param kwargs:
        ArgumentParser.add_argument(...)
                                    ---
    :return:
    """

    def raise_error():
        raise NotImplementedError("Not sure how to translate argparse argument into click argument; add_argument(...) was called with\n{msg}".format(
            msg=textwrap.indent(pprint.pprint({
                'args': args,
                'kwargs': kwargs,
            }), prefix='  '),
        ).rstrip())

    def click_name(opt, underscores=False):
        gflags_opt = opt
        gflags_opt = re.sub('^--', '', gflags_opt)
        if underscores:
            gflags_opt = re.sub('-', '_', gflags_opt)
        return gflags_opt

    def click_opt_name(attr, underscores=False, is_flag=False):
        name = click_name(attr, underscores=underscores)
        dashes = re.sub(r'_', '-', attr)
        if is_flag:
            return f"--{dashes}/--no-{dashes}"
        return f"--{dashes}"

        # if is_flag:
        #     return f"{attr}/no_{attr}"
        # return attr

    def _add_argument(attr):
        arg_type = kwargs.get('type', None)
        # NOTE: click doesn't let us to add help text to each argument; it's supposed to reside in the usage
        # text...
        help = kwargs.get('help', None)
        dflt = kwargs.get('default', None)
        required = kwargs.get('required', False)


        action = kwargs.get('action', None)
        if action is not None:
            if action == 'store_true':
                assert dflt is None
                dflt = False
            elif action == 'store_false':
                assert dflt is None
                dflt = True
            else:
                raise_error()
            click_opt = click_opt_name(attr, is_flag=True)
            # import pdb; pdb.set_trace()
            decorator = click_ctx.add_argument(click_opt, default=dflt, help=help, required=required)
            return decorator

        click_opt = click_opt_name(attr)
        decorator = click_ctx.add_argument(click_opt, default=dflt, type=arg_type, help=help, required=required)
        return decorator

    opt = args[0]

    # underscore_gflags_opt = click_name(opt, underscores=True)
    # decorator = _add_argument(underscore_gflags_opt)

    dashed_gflags_opt = click_name(opt)
    decorator = _add_argument(dashed_gflags_opt)

    return decorator

def add_iml_arguments(parser):
    if isinstance(parser, argparse.ArgumentParser):
        iml_parser = parser.add_argument_group("IML")
    else:
        iml_parser = parser
    add_argument(iml_parser, '--iml-nvprof-enabled', action='store_true', help=textwrap.dedent("""
        IML: is nvprof running?
        
        Internal use only; 
        used to determine whether this python script has been invoked using nvprof.
        If it hasn't, the script will re-invoke itself with nvprof.
    """))
    # add_argument(iml_parser, '--iml-tfprof', action='store_true', help=textwrap.dedent("""
    #     IML: use tfprof TensorFlow profiling utility INSTEAD of nvprof.
    # """))
    add_argument(iml_parser, '--iml-num-calls', type=int, default=1000,
                        help="IML: how many calls should be measured in a single trace?")
    add_argument(iml_parser, '--iml-env',
                 help="IML: Name of environment")
    add_argument(iml_parser, '--iml-algo',
                 help="IML: Name of RL algorithm")
    add_argument(iml_parser, '--iml-trace-time-sec', type=float,
                        help="IML: how long should we profile for, in seconds; "
                             "tracing will stop when either "
                             "we've collected --iml-num-traces OR "
                             "--iml-trace-time-sec has been exceeded")
    add_argument(iml_parser, '--iml-max-timesteps', type=int,
                            help=textwrap.dedent("""
                            IML: how long should we profile for, in timesteps; 
                            timestep progress is reported by calling 
                            iml.prof.report_progress(...)
                            """))
    add_argument(iml_parser, '--iml-max-training-loop-iters', type=int,
                            help=textwrap.dedent("""
                            IML: how long should we profile for, in "training loop iterations"; 
                            a single "training loop iteration" is one call to iml.prof.report_progress. 
                            NOTE: a training loop iteration is the same as a timestep, if every time an algorithm 
                            call iml.prof.report_progress, it advances the timestep by one.
                            - e.g. DDPG advances 100 timesteps (nb_rollout_steps) before calling iml.prof.report_progress
                            - e.g. DQN advances 1 timestep before calling iml.prof.report_progress
                            """))
    add_argument(iml_parser, '--iml-delay-training-loop-iters', type=int,
                            help=textwrap.dedent("""
                            IML: Delay trace collection for the first X "training loop iterations"; 
                            see --iml-max-training-loop-iters for a description of training loop iterations.
                            """))

    add_argument(iml_parser, '--iml-max-passes', type=int,
                            help=textwrap.dedent("""
                            IML: how long should we profile for, in "passes" (i.e., calls to iml.prof.report_progress); 
                            a single "pass" is one call to iml.prof.report_progress. 
                            """))
    add_argument(iml_parser, '--iml-delay-passes', type=int,
                            help=textwrap.dedent("""
                            IML: Delay trace collection for the first X "passes" (i.e., calls to iml.prof.report_progress).
                            """))

    add_argument(iml_parser, '--iml-internal-start-trace-time-sec', type=float,
                        help=textwrap.dedent("""
        IML: (internal use)
        The start time of tracing (in seconds). 
        This gets inherited by child processes.
    """))
    add_argument(iml_parser, '--iml-phase',
                        help=textwrap.dedent("""
        IML: (internal use)
        The "phase" of training captured by this script. 
        The phase covered by a script may change during training.
        E.g. a single script could handle "simulator" and "gradient_update" phases.
        This gets inherited by child processes.
    """))
    add_argument(iml_parser, '--iml-internal-parent-process-name',
                        help=textwrap.dedent("""
        IML: (internal use)
        The process name of the parent that launched this child python process.
        i.e. whatever was passed to iml_profiler.api.prof.set_process_name('forker')
        Internally, this is used for tracking "process dependencies".
    """))
    add_argument(iml_parser, '--iml-util-sampler-pid',
                        help=textwrap.dedent("""
        IML: (internal use)
        The pid of the utilization_sampler.py script that samples CPU/GPU utilization during training.
        We need to keep this so we can terminate it once we are done.
    """))

    add_argument(iml_parser, '--iml-num-traces', type=int,
                        # default=10,
                        help="IML: how many traces should be measured?")
    add_argument(iml_parser, '--iml-keep-traces', action='store_true', help=textwrap.dedent("""
        IML: DON'T delete any existing trace files; keep them and append to them.
        
        Useful if your ML script launches worker processes repeatedly.
    """))
    add_argument(iml_parser, '--iml-python', action='store_true', help=textwrap.dedent("""
        IML: Collecting python profiler (pyprof) data for profiled operations.
        
        Python profiling data is grouped into per-operation summaries, instead of 
        presenting profiling data process-wide.
        
        This prevent overwhelming the user with too much information.
    """))
    add_argument(iml_parser, '--iml-fuzz', action='store_true', help=textwrap.dedent("""
        IML: \"Fuzz\" the script for calls to TensorFlow API's.
        
        Useful if you have no idea where the training-loop of an ML script is located. 
        
        Adds breakpoints / dumps stack traces when certain TensorFlow API's are called; 
        for e.g. sesssion.run(...) for running the computational graph
        (currently this is the only thing we trace).
    """))
    add_argument(iml_parser, '--iml-disable', action='store_true', help=textwrap.dedent("""
        IML: Skip any profiling.
    """))
    add_argument(iml_parser, '--iml-calibration', action='store_true', help=textwrap.dedent("""
        IML: This is a calibration run. 
        Calibration runs change the semantics of the "iml-prof --config uninstrumented"; 
        in particular, usually "--config uninstrumented" would disable all of IML.
        However, for calibration runs, we use uninstrumented to disable CUPTI/CUDA-API level tracing, BUT 
        still run with python-level stuff (annotations, interceptions) enabled.
    """))
    add_argument(iml_parser, '--iml-disable-pyprof-annotations', action='store_true', help=textwrap.dedent("""
        IML: Skip recording op-events.
    """))
    add_argument(iml_parser, '--iml-disable-pyprof-interceptions', action='store_true', help=textwrap.dedent("""
        IML: Skip recording of pyprof events by intercepting Python -> C-library calls.
        ( used for collecting simulator and TensorFlow C++ API time ).
    """))
    add_argument(iml_parser, '--iml-disable-pyprof', action='store_true', help=textwrap.dedent("""
        IML: Skip any profiling (i.e. trace-collection, trace-dumping) related to python times.
    """))
    add_argument(iml_parser, '--iml-disable-tfprof', action='store_true', help=textwrap.dedent("""
        IML: Skip any profiling (i.e. trace-collection, trace-dumping) related to GPU times.
    """))
    add_argument(iml_parser, '--iml-disable-pyprof-dump', action='store_true', help=textwrap.dedent("""
        IML: Skip pyprof trace-dumping, but NOT trace-collection.
    """))
    add_argument(iml_parser, '--iml-disable-tfprof-dump', action='store_true', help=textwrap.dedent("""
        IML: Skip tfprof trace-dumping, but NOT trace-collection.
    """))
    add_argument(iml_parser, '--iml-disable-pyprof-trace', action='store_true', help=textwrap.dedent("""
        IML: Disable most of pyprof trace-collection (but not entirely).
    """))
    add_argument(iml_parser, '--iml-disable-gpu-hw', action='store_true', help=textwrap.dedent("""
        IML: Disable GPU HW sampling trace-collection.
    """))
    add_argument(iml_parser, '--iml-delay', action='store_true', help=textwrap.dedent("""
        IML: Delay trace collection until your training script has warmed up; 
        you must signal this to IML by calling iml.prof.enable_tracing() when that happens 
        (as is done in the annotated stable-baselines algorithm implementations e.g. DQN).
        
        If you DON'T provide this, then tracing begins immediately starting from "with iml.prof.profiler(...)".
    """))
    add_argument(iml_parser, '--iml-just-sample-util', action='store_true', help=textwrap.dedent("""
        IML: collect machine utilization data and output it to --iml-directory.
        
        NOTE: this will NOT collect profiling information.
    """))
    add_argument(iml_parser, '--iml-training-progress', action='store_true', help=textwrap.dedent("""
        IML: collect training progress data and output it to --iml-directory.
        
        NOTE: This is ON by default, except if --iml-disable is given, in which case you must provide this.
    """))
    add_argument(iml_parser, '--iml-unit-test',
                        action='store_true',
                        help=textwrap.dedent("""
    IML: (for unit-testing) Record "actual results" needed for doing basics unit-test checks.
    """))
    add_argument(iml_parser, '--iml-unit-test-name',
                        help=textwrap.dedent("""
    IML: (for unit-testing) name to store in IMLUnitTest.test_name.
    """))
    add_argument(iml_parser, '--iml-debug', action='store_true', help=textwrap.dedent("""
        IML: debug profiler.
    """))
    add_argument(iml_parser, '--iml-start-measuring-call', default=1, type=int,
                        help="IML: when should measuring begin?")
    add_argument(iml_parser, '--iml-bench-name',
                        default=NO_BENCH_NAME,
                        help=textwrap.dedent("""
    IML: which code block should we measure?
    i.e. --iml-bench-name=some_bench
        # Just measure "some_bench", nothing else.
        profiler.profile('some_bench', do_some_bench)
    """))
    add_argument(iml_parser, '--iml-directory',
                        help=textwrap.dedent("""
    IML: profiling output directory.
    """))
    add_argument(iml_parser, '--iml-skip-rm-traces', action='store_true', help=textwrap.dedent("""
    DON'T remove traces files from previous runs rooted at --iml-directory.
    Useful if your training script has multiple training scripts that need to be traced with IML.
    """))

# Match input/output to PythonProfilerParser
PYPROF_REGEX = r'(?:python_profile.*|microbenchmark\.json|config.*\.json)'
# Match input/output to CUDASQLiteParser
NVPROF_REGEX = r'(?:nvidia.*\.nvprof|microbenchmark\.json|config.*\.json|nvidia.*\.pretty\.txt)'
def is_iml_file(path):
    base = _b(path)
    return re.search(r'{pyprof}|{nvprof}'.format(
        pyprof=PYPROF_REGEX,
        nvprof=NVPROF_REGEX),
        base)

def iml_argv_and_env(prof : Profiler, keep_executable=False, keep_non_iml_args=False, env=None):
    iml_argv = _iml_argv(prof, keep_executable=keep_executable, keep_non_iml_args=keep_non_iml_args)
    iml_env = _iml_env(prof, keep_executable=keep_executable, keep_non_iml_args=keep_non_iml_args, env=env)
    return iml_argv, iml_env

def _iml_env(prof : Profiler, keep_executable=False, keep_non_iml_args=False, env=None):
    if env is None:
        env = dict(os.environ)

    # Only modify LD_PRELOAD to load.
    # Rationale: want to be able to easily run nvprof on RLScope-instrumented scripts
    if not prof.disable:
        ld_preloads = []
        if 'LD_PRELOAD' in env:
            ld_preloads.append(env['LD_PRELOAD'])
        ld_preloads.append(rlscope_api.RLSCOPE_CLIB)
        env['LD_PRELOAD'] = ':'.join(ld_preloads)

    return env

def _iml_argv(prof : Profiler, keep_executable=False, keep_non_iml_args=False):
    """
    Return a list of string arguments related to IML that were passed to the current running python process.

    Useful for forwarding IML arguments to python child processes instrumented with IML.
    """
    # If this fails and your using profiler.glbl, make sure you call iml.handle_iml_args(...)
    # before spawning child processes.
    assert prof is not None
    # JAMES TODO: forward set_phase to children.
    parser = argparse.ArgumentParser()
    add_iml_arguments(parser)
    logger.info("> argv: {argv}".format(argv=' '.join(sys.argv)))
    # NOTE: sys.argv[0] is the python script name.
    args, extra_argv = parser.parse_known_args(sys.argv[1:])
    logger.info("> extra_argv: {argv}".format(argv=' '.join(extra_argv)))
    # Inherit arguments in our fork-ed children.
    args.iml_internal_start_trace_time_sec = prof.get_start_trace_time_sec()
    args.iml_phase = prof.phase
    if prof.process_name is None:
        prof._failing = True
        raise RuntimeError("IML: You must call iml_profiler.api.prof.set_process_name('some_name') before forking children!")
    args.iml_internal_parent_process_name = prof.process_name
    args.iml_util_sampler_pid = prof.util_sampler_pid
    argv = args_to_cmdline(parser, args, keep_executable=keep_executable, use_pdb=False)
    if keep_non_iml_args:
        return argv + extra_argv
    return argv

def run_with_nvprof(directory, parser, args,
                    bench_name=NO_BENCH_NAME):
    logger.info("> Reinvoking script with nvprof; bench_name={b}".format(
        b=bench_name))

    nvprof_logfile = _j(directory, "nvidia{bench}.nvprof_logfile.txt".format(
        bench=bench_suffix(bench_name)))
    if _e(nvprof_logfile):
        os.remove(nvprof_logfile)
    nvprof_sqlite_file = _j(directory, "nvidia{bench}.nvprof".format(
        bench=bench_suffix(bench_name)))
    if _e(nvprof_sqlite_file):
        # Nvprof fails if the output file already exists.
        os.remove(nvprof_sqlite_file)
    os.makedirs(_d(nvprof_logfile), exist_ok=True)
    nvprof_args = ["nvprof",
                   "-o", nvprof_sqlite_file,
                   "--log-file", nvprof_logfile,
                   "--profile-from-start", "off"]
    cmdline = args_to_cmdline(parser, args)
    argv_exec = nvprof_args + cmdline + [
        "--iml-nvprof-enabled",
    ]
    if bench_name != NO_BENCH_NAME:
        argv_exec.extend(["--iml-bench-name", bench_name])

    print_cmd(argv_exec)
    subprocess.run(argv_exec, stdout=sys.stdout, stderr=sys.stderr, check=True)

def check_avail_gpus():
    avail_gpus = get_available_gpus()
    avail_cpus = get_available_cpus()
    # We want to be CERTAIN about which device TensorFlow is using.
    # If no GPUs are available, TF will use the CPU.
    # If a GPU is available, make sure only 1 is available so we are certain it's using that one.
    if not( (len(avail_gpus) == 1) or
            (len(avail_gpus) == 0 and len(avail_cpus) == 1) ):
        CUDA_VISIBLE_DEVICES = ENV.get('CUDA_VISIBLE_DEVICES', None)
        logger.error(textwrap.dedent("""
        Multiple GPUs were found; IML benchmark requires only one GPU to be visible to TensorFlow via (for example) "export CUDA_VISIBLE_DEVICES=0".
        Use one of the below available GPUs:
        """))
        pprint.pprint({
            'avail_gpus':avail_gpus,
            'avail_cpus':avail_cpus,
            'CUDA_VISIBLE_DEVICES':CUDA_VISIBLE_DEVICES,
        }, indent=2)
        return False
    return True

# If you search for function names matching this pattern in pyprof output, they will match TensorFlow C++ API calls.
CLIB_TENSORFLOW_REGEX = r'(?:built-in.*pywrap_tensorflow)'
# We can manually wrap a c-library in order to record C API call times.  See test_call_c.py for how to do this.
CLIB_WRAPPER_REGEX = r'CLIB__.*'
def dump_config(path, **kwargs):
    config = dict()

    avail_gpus = get_available_gpus()
    avail_cpus = get_available_cpus()
    # We want to be CERTAIN about which device TensorFlow is using.
    # If no GPUs are available, TF will use the CPU.
    # If a GPU is available, make sure only 1 is available so we are certain it's using that one.
    if not( (len(avail_gpus) == 1) or
            (len(avail_gpus) == 0 and len(avail_cpus) == 1) ):
        CUDA_VISIBLE_DEVICES = ENV.get('CUDA_VISIBLE_DEVICES', None)
        pprint.pprint({
            'avail_gpus':avail_gpus,
            'avail_cpus':avail_cpus,
            'CUDA_VISIBLE_DEVICES':CUDA_VISIBLE_DEVICES,
        }, indent=2)
        msg = textwrap.dedent("""
        > IML ERROR: Multiple GPUs were found; IML benchmark requires only one GPU to be visible to TensorFlow via (for example) "export CUDA_VISIBLE_DEVICES=0".
        Use one of the below available GPUs:
        """)
        raise RuntimeError(msg)
    if len(avail_gpus) == 1:
        device_dict = avail_gpus[0]
    else:
        device_dict = avail_cpus[0]

    config.update(device_dict)

    c_lib_func_pyprof_pattern = kwargs.get('c_lib_func_pyprof_pattern', CLIB_WRAPPER_REGEX)
    defaults = {
        'clock': "monotonic_clock",
        'device_name': None,
        'impl_name': None,
        # For tensorflow: r'(?:built-in.*pywrap_tensorflow)'
        'c_lib_func_pyprof_pattern':c_lib_func_pyprof_pattern,
        # Discard the first nvprof sample since it's typically 700ms
        # (not sure why, presumably some initialization time).
        'discard_first_sample':True,

        # 'bench_name_labels': {
        #     ...
        # },
    }
    config = dict(defaults)
    config.update(kwargs)
    assert 'num_calls' in config or (
        'iterations' in config and \
        'repetitions' in config
    )
    dump_json(config, path)

def load_json(path):
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def dump_json(data, path):
    os.makedirs(_d(path), exist_ok=True)
    json.dump(data,
              codecs.open(path, mode='w', encoding='utf-8'),
              sort_keys=True, indent=4,
              skipkeys=False)

def list_files(direc):
    def _list_files(direc):
        def _path(path):
            return _j(direc, path)
        return [_path(path) for path in os.listdir(direc)]

    if type(direc) == list:
        all_files = []
        for d in direc:
            all_files.extend(_list_files(d))
        return all_files

    return _list_files(direc)

def _device_proto_as_dict(device_proto):
    # For GPU's
    # ipdb> device_proto.physical_device_desc
    # 'device: 0, name: Quadro P4000, pci bus id: 0000:04:00.0, compute capability: 6.1'

    # For CPU's
    # ipdb> device_proto
    # name: "/device:CPU:0"
    # device_type: "CPU"
    # memory_limit: 268435456
    # locality {
    # }
    # incarnation: 11976653475402273625

    m = re.search(r'device: (?P<device>\d+), name: (?P<name>.*), pci bus id: (?P<pci_bus_id>[^,]+), compute capability: (?P<compute_capability>.*)',
                  device_proto.physical_device_desc)
    device = int(m.group('device'))
    name = m.group('name')
    return {"device_number":device, "device_name":name}

def ls_files(directory):
    """
    Same as list_files but allow directory to not exist.
    """
    if not os.path.isdir(directory):
        return []
    return list_files(directory)

def process_directory(directory, process_name):
    direc = _j(directory, "process", process_name)
    return direc

def phase_directory(directory, process_name, phase):
    process_direc = process_directory(directory, process_name)
    direc = _j(process_direc, "phase", str(phase))
    return direc


class RecordedIncrementalTrainingProgress:
    def __init__(self, machine_name, process_name, phase):
        self.machine_name = machine_name
        self.process_name = process_name
        self.phase = phase

        self.total_timesteps = None

        self.start_trace_time_sec = None

        # NOTE: this won't get filled in if we start collecting traces
        # from the VERY start of training (e.g. minigo)...
        # In that case, we want to assume:
        #   start_num_timesteps = 0
        #   start_percent_complete = 0
        #   start_training_time_us =
        #     (very start of program ideally, since iml.prof.report_progress won't be getting called.)
        #     Alternatively, we can use the earliest known start time of trace-collection:
        #     iml.prof.start_trace_time_sec
        self.start_num_timesteps = None
        self.start_percent_complete = None
        self.start_training_time_us = None

        self.end_num_timesteps = None
        self.end_percent_complete = None
        self.end_training_time_us = None

    def can_dump(self, reports_progress, expect_true=False):
        if reports_progress:
            if expect_true:
                assert self.start_percent_complete is not None
                assert self.start_trace_time_sec is not None
            # Wait until report_start_of_progress has been called.
            return self.start_percent_complete is not None and self.start_trace_time_sec is not None
        # Assume that tracing starts from the very beginning the ML script starts;
        # i.e. we don't delay until iml.prof.report_progress() is called.
        if expect_true:
            assert self.start_trace_time_sec is not None
        return self.start_trace_time_sec is not None

    def report_progress(self, percent_complete, num_timesteps, total_timesteps, start_trace_time_sec, start_usec=None):
        self.end_percent_complete = percent_complete
        self.end_num_timesteps = num_timesteps
        self.total_timesteps = total_timesteps
        if start_trace_time_sec is not None:
            self.start_trace_time_sec = start_trace_time_sec

        if start_usec is None:
            start_usec = iml_timer.now_us()

        self.end_training_time_us = start_usec

    def report_start_of_progress(self, percent_complete, num_timesteps, total_timesteps, start_trace_time_sec):
        assert self.start_percent_complete is None
        assert self.start_num_timesteps is None
        assert self.start_training_time_us is None

        self.start_percent_complete = percent_complete
        self.start_num_timesteps = num_timesteps
        self.start_training_time_us = iml_timer.now_us()

        self.report_progress(
            percent_complete, num_timesteps, total_timesteps, start_trace_time_sec,
            start_usec=self.start_training_time_us)

    def _or_zero(self, num):
        if num is None:
            return 0
        return num

    def _or(self, num, default):
        if num is None:
            return default
        return num

    def as_proto(self, training_progress=None):
        if training_progress is None:
            training_progress = IncrementalTrainingProgress()

        if self.start_percent_complete is None:
            training_progress.content_code = TP_NO_PROGRESS
            return training_progress

        training_progress.content_code = TP_HAS_PROGRESS

        training_progress.total_timesteps = self.total_timesteps

        training_progress.machine_name = self.machine_name
        training_progress.process_name = self.process_name
        training_progress.phase = self.phase

        # if self.start_percent_complete is not None:
        training_progress.start_percent_complete = self._or(self.start_percent_complete, 0.)
        training_progress.start_num_timesteps = self._or(self.start_num_timesteps, 0)
        training_progress.start_training_time_us = int(self._or(
            self.start_training_time_us,
            self.start_trace_time_sec * constants.USEC_IN_SEC))

        training_progress.end_percent_complete = self.end_percent_complete
        training_progress.end_training_time_us = int(self.end_training_time_us)
        training_progress.end_num_timesteps = self.end_num_timesteps

        training_progress.start_trace_time_us = int(self.start_trace_time_sec * constants.USEC_IN_SEC)

        return training_progress

    def __repr__(self):
        return as_str(self)

def get_tensorflow_config():
    """
    Report values of environment variables that affect IML-modified tensorflow execution.
    """
    conf = dict()
    def _add(var, skip_if_missing=True, default=None):
        if var in os.environ:
            conf[var] = os.environ[var]
        elif not skip_if_missing:
            conf[var] = default

    _add('TF_CUPTI_EMPTY_TRACING_CALLBACKS')
    _add('TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS')

    return conf

def rls_log_msg(flag_name, msg):
    return f"[{flag_name}] {msg}"
