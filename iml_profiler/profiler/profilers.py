import logging
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

import iml_profiler

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


from tensorflow.python.profiler import profile_context
from tensorflow.python.framework import c_api_util
from tensorflow.python.client import session

from iml_profiler.profiler import unit_test_util

from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto

# pip install py-cpuinfo
import cpuinfo
import psutil

from os import environ as ENV

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler.parser.common import *
from iml_profiler.profiler import cudaprofile
from iml_profiler.profiler import clib_wrap
from iml_profiler.profiler.clib_wrap import MICROSECONDS_IN_SECOND
from iml_profiler.profiler import tensorflow_profile_context

from iml_profiler.profiler import glbl
from iml_profiler.profiler import proto_util
import iml_profiler.profiler.estimator
import iml_profiler.profiler.session

from iml_profiler import py_config

DEBUG = tensorflow_profile_context.DEBUG
# DEBUG_OP_STACK = True
DEBUG_OP_STACK = False

# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

DEFAULT_PHASE = 'default_phase'

TF_PRINT_TIMESTAMP = ENV.get('TF_PRINT_TIMESTAMP', 'no') == 'yes'

UTILIZATION_SAMPLER_PY = _j(py_config.ROOT, 'python', 'scripts', 'utilization_sampler.py')
PYTHON_BIN = 'python3'

# If we exceed 1000 session.run(...) calls without dumping a trace, only print a warning every 100 calls.
WARN_EVERY_CALL_MODULO = 100

# For warnings that go off all the time, how frequently should we inform the user about them?
WARN_EVERY_SEC = 10

# Number of steps before we dump a trace file
# STEPS_PER_TRACE = 1000
# STEPS_PER_TRACE = 10

MP_SPAWN_CTX = multiprocessing.get_context('spawn')

_TF_MODIFIED = False
def modify_tensorflow():
    global _TF_MODIFIED
    if _TF_MODIFIED:
        return

    setup()
    iml_profiler.profiler.session.setup()
    iml_profiler.profiler.estimator.setup()
    iml_profiler.profiler.tensorflow_profile_context.setup()

    # from tensorflow.python.profiler import profile_context
    # profile_context.MAX_TRACED_STEPS = 99999999

    _TF_MODIFIED = True

# All currently active Profiler objects (there should really only be one).
# Used for hooking into sess.as_default()
PROFILERS = []


SETUP_DONE = False
def setup(allow_skip=False):
    global SETUP_DONE
    if allow_skip and SETUP_DONE:
        return
    assert not SETUP_DONE

    """
    Usually TensorFlow only measures 100 steps at most.
    Set a big upper limit so it will measure each iteration we measure.
    """
    tensorflow_profile_context.MAX_TRACED_STEPS = 99999999

    # setup_wrap_BaseSession_as_default()

    iml_profiler.profiler.session.register_session_active_hook(AddProfileContextHook)
    iml_profiler.profiler.session.register_session_inactive_hook(RemoveProfileContextHook)
    iml_profiler.profiler.session.register_session_run_hook(MaybeDumperTfprofContextHook)
    clib_wrap.register_record_event_hook(DumpPyprofTraceHook)

    sys.excepthook = cleanup_profiler_excepthook

    SETUP_DONE = True

# """
# Wrap sess.as_default().
# """
# old_BaseSession_as_default = None
# def setup_wrap_BaseSession_as_default():
#     global old_BaseSession_as_default
#     old_BaseSession_as_default = getattr(session.BaseSession, 'as_default')
#     setattr(session.BaseSession, 'as_default', wrap_BaseSession_as_default)
# def wrap_BaseSession_as_default(self):
#     if DEBUG:
#         print_stacktrace("> wrapped sess.as_default()")
#     for prof in PROFILERS:
#         prof.set_session(self)
#     return old_BaseSession_as_default(self)

class _ProfileContextManager:
    def __init__(self):
        self._session_to_context = dict()

    def add_profile_context(self, session, phase=None):
        assert session not in self._session_to_context
        if iml_profiler.api.prof is not None:
            disabled = not iml_profiler.api.prof.is_tfprof_enabled
        else:
            disabled = False
        pctx = tensorflow_profile_context.ProfileContext(
            # We handle dumping explicitly.
            # Do NOT set this to true; otherwise we'll start dumping during the critical path
            # when __exit__ is called in remove_profile_context.
            dump_on_finished=False,
            # Need to explicitly use empty trace steps otherwise profiler
            # "auto decides" which steps to trace.
            trace_steps=[],
            trace_all=True,
            session=session,
            phase=phase)
        if disabled:
            pctx.disable_tracing()
        pctx.__enter__()
        self._session_to_context[session] = pctx
        return pctx

    def get_profile_context(self, session, allow_none=False, default=None):
        if allow_none:
            pctx = self._session_to_context.get(session, default)
            return pctx

        pctx = self._session_to_context[session]
        return pctx

    def recreate_sessions_profile_contexts(self, phase=None):
        sessions = self._session_to_context.keys()
        # NOTE: tfprof files have a phase_name attribute; we need to dump whatever events we have in the current phase.
        # In particular, we must avoid accidentally putting events from the next phase into the file labelled as
        # the previous phase.
        for session in sessions:
            self.recreate_profile_context(session, phase)

    def recreate_profile_context(self, session, phase=None):
        """
        We are about to switches phases.
        Dump the current profile-context for this session,
        and initialize a new profile-context.
        """
        self.remove_profile_context(session)
        pctx = self.add_profile_context(session, phase)
        return pctx

        # assert session not in self._session_to_context
        # if iml_profiler.api.prof is not None:
        #     disabled = not iml_profiler.api.prof.is_tfprof_enabled
        # else:
        #     disabled = False
        # pctx = tensorflow_profile_context.ProfileContext(
        #     # We handle dumping explicitly.
        #     # Do NOT set this to true; otherwise we'll start dumping during the critical path
        #     # when __exit__ is called in remove_profile_context.
        #     dump_on_finished=False,
        #     # Need to explicitly use empty trace steps otherwise profiler
        #     # "auto decides" which steps to trace.
        #     trace_steps=[],
        #     trace_all=True,
        #     session=session)
        # if disabled:
        #     pctx.disable_tracing()
        # pctx.__enter__()
        # self._session_to_context[session] = pctx
        # return pctx

    def remove_profile_context(self, session):
        assert session in self._session_to_context
        pctx = self._session_to_context[session]
        # TODO: cleanup profile context here?

        # if iml_profiler.api.prof is not None:
        #     process_name = iml_profiler.api.prof.process_name
        #     phase = iml_profiler.api.prof.phase
        #     dump_path = iml_profiler.api.prof.tfprof_path(session.session_id)
        # else:
        #     process_name = None
        #     phase = None
        #     dump_path = None
        #     raise NotImplementedError("Not sure what to use for dump_path...")

        # PROBLEM: this will dump right in the middle of executing...
        # would be nicer if dump was delayed until the fixed dump period.
        # prof.dump_session_tfprof(session) / pctx.dump(dump_path, process_name, phase)

        pctx.__exit__(None, None, None)

        # SOLUTION: delay the dump in a DumpThunk.
        # add_dump_thunk(session, pctx)
        iml_profiler.api.prof._dump_tfprof(session, debug=iml_profiler.api.prof.debug)

        del self._session_to_context[session]

class MyProcess(MP_SPAWN_CTX.Process):
    """
    Record exceptions in child process, and make them accessible
    in the parent process after the child exits.

    e.g.

    def target():
        raise ValueError('Something went wrong...')

    p = Process(target=target)
    p.start()
    p.join()

    if p.exception:
        error, traceback = p.exception
        logging.info(traceback)



    From:
    https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = MP_SPAWN_CTX.Pipe()
        self._exception = None

    def run(self):
        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # Re-raise exception so that proc.exitcode
            # still gets set to non-zero.
            raise

    @property
    def exception(self):
        if self._exception is not None:
            return self._exception

        if self._pconn.poll():
            self._exception = self._pconn.recv()

        return self._exception

class ForkedProcessPool:
    """
    Fork a child and run "target".

    NOTE: We make this class explicit, since ProcessPoolExecutor DOESN'T fork a
    process on every call to pool.submit(...)
    """
    WAIT_WORKERS_TIMEOUT_SEC = 0.01
    UNLIMITED = 0
    def __init__(self, name=None, max_workers=UNLIMITED, debug=False):
        self.name = name
        self.process = None
        self.max_workers = max_workers
        self.active_workers = []
        self.debug = debug
        # self.inactive_workers = []

    def _join_finished_workers(self):
        i = 0
        while i < len(self.active_workers):
            if not self.active_workers[i].is_alive():
                if self.debug:
                    logging.info("> Pool(name={name}): Join pid={pid} active_workers[i={i}]={proc}".format(
                        name=self.name,
                        i=i,
                        pid=self.active_workers[i].pid,
                        proc=self.active_workers[i],
                    ))
                self._join_child(self.active_workers[i])
                # self.inactive_workers.append(self.active_workers[i])
                del self.active_workers[i]
            else:
                i += 1

    def _wait_for_workers(self):
        while True:
            self._join_finished_workers()
            if self.max_workers == ForkedProcessPool.UNLIMITED or len(self.active_workers) < self.max_workers:
                break
            if self.debug:
                logging.info("> Pool.sleep for {sec} sec".format(
                    sec=ForkedProcessPool.WAIT_WORKERS_TIMEOUT_SEC))
            time.sleep(ForkedProcessPool.WAIT_WORKERS_TIMEOUT_SEC)

    def submit(self, name, fn, *args, sync=False, **kwargs):
        if sync:
            return fn(*args, **kwargs)
        self._wait_for_workers()
        # proc = multiprocessing.Process(target=fn, name=name, args=args, kwargs=kwargs)
        proc = MyProcess(target=fn, name=name, args=args, kwargs=kwargs)
        proc.start()
        if self.debug:
            logging.info("> Pool(name={name}): submit pid={pid}, proc={proc}".format(
                name=self.name,
                pid=proc.pid,
                proc=proc))
        self.active_workers.append(proc)

    def shutdown(self):
        for proc in self.active_workers:
            if self.debug:
                logging.info("> Pool(name={name}): Join pid={pid}, proc={proc}".format(
                    pid=proc.pid,
                    name=self.name,
                    proc=proc))
            self._join_child(proc)
            if self.debug:
                logging.info("> Pool(name={name}): Joined pid={pid}, proc={proc}".format(
                    pid=proc.pid,
                    name=self.name,
                    proc=proc))

    def _join_child(self, proc):
        proc.join()
        if proc.exception:
            ex, tb_string = proc.exception
            ExceptionKlass = type(ex)
            parent_ex = ExceptionKlass(("Exception in child process pid={pid} proc={proc} of ForkedProcessPool(name={name}): {error}\n"
                                        "{tb}").format(
                name=self.name,
                pid=proc.pid,
                proc=proc,
                error=ex.args[0],
                tb=proc.exception[1]))
            raise parent_ex

class TfprofDumper:
    def __init__(self,
                 trace_id,
                 session,
                 process_name,
                 tfprof_path,
                 debug=False):
        self.trace_id = trace_id
        self.session = session
        self.process_name = process_name
        self.tfprof_path = tfprof_path
        self.debug = debug

        # Q: How can we detect when we've traced a certain number of calls?
        # A: After every session.run(...), check session.trace_count; hook into that event and register
        # a callback that runs:
        # prof.dump_tfprof(session)
        #
        # def dump_tfprof(self, session):
        #   tfprof_dumper = TfprofDumper(...)
        #   self.bg_dumper.submit(tfprof_dumper.dump)

    # def dump_sessions_tfprof(self):
    #     for dump_thunk in DUMP_THUNKS:
    #         dump_thunk.dump(self.next_trace_id, self.process_name, prof=self)
    #         self.next_trace_id += 1
    #     DUMP_THUNKS.clear()
    #
    #     # TODO: we must make sure to clear the trace-data AFTER sending the DumpThunk to a child-process.
    #
    #     # Q: When should we dump a profile?  When we exceed that max # of traces per file.
    #     for session in profiler.session.ACTIVE_SESSIONS:
    #         self.dump_session_tfprof(session)
    #
    #     # All the traces have been dumped, reset the counter for
    #     # "number of session.run(...) calls whose trace data we haven't flushed yet".
    #     # tensorflow_profile_context.reset_session_run_calls_traced()

    def dump(self):
        if self.debug:
            logging.info(("> TfprofDumper.dump: start\n"
                   "{dict}").format(
                dict=textwrap.indent(pprint.pformat(self.__dict__), prefix="  "),
            ))

        pctx = ProfileContextManager.get_profile_context(self.session)
        # tfprof_path = self.prof.tfprof_path(session.session_id, self.trace_id)
        pctx.dump(self.tfprof_path, self.process_name)

        if self.debug:
            logging.info(("> TfprofDumper.dump: done\n"
                   "{dict}").format(
                dict=textwrap.indent(pprint.pformat(self.__dict__), prefix="  "),
            ))

class PyprofDumper:
    def __init__(self,
                 trace_id,
                 config_path,
                 c_lib_func_pyprof_pattern,
                 num_calls,
                 start_measuring_call,
                 average_time_per_call_sec,
                 average_time_per_call_no_profile_sec,
                 config_kwargs,
                 process_name,
                 phase,
                 pyprof_proto_path,
                 pyprof_call_times_path,
                 pyprof_step,
                 pyprof_dump_manager,
                 pyprof_trace_key,
                 debug=False):
        self.trace_id = trace_id
        self.config_path = config_path
        self.c_lib_func_pyprof_pattern = c_lib_func_pyprof_pattern
        self.num_calls = num_calls
        self.start_measuring_call = start_measuring_call
        self.average_time_per_call_sec = average_time_per_call_sec
        self.average_time_per_call_no_profile_sec = average_time_per_call_no_profile_sec
        self.config_kwargs = config_kwargs
        self.process_name = process_name
        self.phase = phase
        self.pyprof_proto_path = pyprof_proto_path
        self.pyprof_call_times_path = pyprof_call_times_path
        self.pyprof_step = pyprof_step
        self.pyprof_dump_manager = pyprof_dump_manager
        self.pyprof_trace_key = pyprof_trace_key
        self.debug = debug

    def dump(self):
        """
        Dump trace data to:
        - pyprof.trace_<next_trace_id>.proto
        - profile.trace_<next_trace_id>.<session_id>.proto
        - config.trace_<next_trace_id>.proto
        """
        if self.debug:
            logging.info("PyprofDumper.dump start, path={path}".format(
                path=self.pyprof_proto_path))

        # Q: Should we be calling this again...?  We'd like to update num_calls if it was computed dynamically...
        if self.c_lib_func_pyprof_pattern is not None and \
                'c_lib_func_pyprof_pattern' not in self.config_kwargs:
            self.config_kwargs['c_lib_func_pyprof_pattern'] = self.c_lib_func_pyprof_pattern
        dump_config(self.config_path,
                    num_calls=self.num_calls,
                    start_measuring_call=self.start_measuring_call,
                    # profile_sec=self.profile_sec,
                    # no_profile_sec=self.no_profile_sec,
                    average_time_per_call_sec=self.average_time_per_call_sec,
                    average_time_per_call_no_profile_sec=self.average_time_per_call_no_profile_sec,
                    process_name=self.process_name,
                    **self.config_kwargs) # <-- this triggers GPU allocation

        pyprof_trace = self.pyprof_dump_manager.get(self.pyprof_trace_key)
        pyprof_trace.dump(self.pyprof_proto_path, self.process_name, self.phase)

        if self.debug:
            logging.info("PyprofDumper.dump done, path={path}".format(
                path=self.pyprof_proto_path))

def get_tfprof_path(directory, bench_name, session_id, trace_id):
    tfprof_path = _j(
        directory,
        "profile{bench}{trace}{sess}.proto".format(
            bench=bench_suffix(bench_name),
            trace=trace_suffix(trace_id),
            sess=sess_suffix(session_id),
        ))
    return tfprof_path

# class DumpThunk:
#     """
#     A Session object has become inactive (i.e. sess.close()).
#
#     Any traced sess.run(...) need to be dumped to a tfprof proto file.
#     However, we don't want to dump on the critical path of what we are measuring.
#     So, instead we delay the dump inside a "thunk" (i.e. a DumpThunk).
#     """
#     def __init__(self, session, pctx, async=True):
#
#         self.session = session
#         self.pctx = pctx
#         assert self.pctx.phase is not None
#         self.phase = self.pctx.phase
#         self.async = async
#
#         if self.async:
#             # TODO: bind process to a different core to avoid interference?
#             # self.bg_dumper = ProcessPoolExecutor(max_workers=1)
#             self.bg_dumper = ForkedProcessPool()
#
#     def _dump(self, trace_id, process_name, prof):
#         dump_path = prof.tfprof_path(self.session.session_id,
#                                      trace_id=trace_id)
#         self.pctx.dump(dump_path, process_name)
#
#     def dump(self, trace_id, process_name, prof):
#         if self.async:
#             ret = self.bg_dumper.submit(self._dump, trace_id, process_name, prof)
#             # After process has forked for dumping trace-data, clear the current process' trace-data.
#             self.pctx.clear()
#             return ret
#         return self._dump(trace_id, process_name, prof)

# DUMP_THUNKS = []
# def add_dump_thunk(session, pctx):
#     DUMP_THUNKS.append(DumpThunk(session, pctx))

ProfileContextManager = _ProfileContextManager()

"""
Add after-inactive hooks to call remove_profile_context, and after-active hooks to add_profile_context.
"""
class _AddProfileContextHook(iml_profiler.profiler.session.SessionActiveHook):
    def __init__(self):
        pass

    def after_active(self, session):
        """
        Run after tf.Session() is called, in case we wish to do anything that requires the C++ API.
        """
        ProfileContextManager.add_profile_context(session)
AddProfileContextHook = _AddProfileContextHook()

class _RemoveProfileContextHook(iml_profiler.profiler.session.SessionInactiveHook):
    def __init__(self):
        pass

    def before_inactive(self, session):
        """
        Run before session.close() is called, in case we wish to do anything that requires the C++ API.
        """
        ProfileContextManager.remove_profile_context(session)
RemoveProfileContextHook = _RemoveProfileContextHook()

"""
Add after-inactive hooks to call remove_profile_context, and after-active hooks to add_profile_context.
"""

class _MaybeDumperTfprofContextHook(iml_profiler.profiler.session.SessionRunHook):
    def __init__(self):
        pass

    def after_run(self, session):
        """
        Run after session.run(...) is called.

        If the number of traces collected for this session exceeds a threshold, then dump the traces.
        """
        if not hasattr(session, '_iml_num_runs_traced'):
            session._iml_num_runs_traced = 0

        if session._iml_num_runs_traced >= iml_profiler.api.prof.num_calls:
            iml_profiler.api.prof._dump_tfprof(session, debug=iml_profiler.api.prof.debug)
            session._iml_num_runs_traced = 0
        else:
            session._iml_num_runs_traced += 1
MaybeDumperTfprofContextHook = _MaybeDumperTfprofContextHook()

class _DumpPyprofTraceHook(clib_wrap.RecordEventHook):
    """
    After each TensorFlow C++ API call event is recorded, check
    if the number of total events recorded exceeds the pre-defined threshold
    (roughly corresponds to a 1MB file size limit on the Pyprof.proto file).

    If we've hit 1MB; dump the Pyprof.proto.
    """
    def __init__(self):
        pass

    def after_record_event(self, pyprof_trace, event):
        if pyprof_trace.get_num_events() >= clib_wrap.PROTO_MAX_PYPROF_PY_EVENTS:
            iml_profiler.api.prof._dump_pyprof()
            num_events = clib_wrap.num_events_recorded()
            # NOTE: I've seen this assertion fail with minigo, not sure why though...
            if num_events != 0:
                logging.info(("> IML: WARNING, after dumping pyprof data, there were "
                       "still {n} pyprof events recorded").format(n=num_events))
            # assert num_events == 0
DumpPyprofTraceHook = _DumpPyprofTraceHook()


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

    :param exit_early
        if True, exit ML script immediately after benchmarking bench_name.
        Othwerwise, continue executing ML script until it finishes.
    """
    def __init__(self, directory=None,
                 bench_name=NO_BENCH_NAME,
                 num_calls=None, start_measuring_call=None,
                 trace_time_sec=None,
                 num_traces=None,
                 keep_traces=None,
                 tfprof=True,
                 c_lib_func_pyprof_pattern=None,
                 # tfprof=True,
                 repetition_time_limit_sec=10.,
                 debug=None,
                 exit_early=True,
                 require_end_operation=False,
                 python=None,
                 disable=None,
                 unit_test=None,
                 unit_test_name=None,
                 args=None):
        modify_tensorflow()

        self.manager = multiprocessing.Manager()
        self.pyprof_dump_manager = clib_wrap.PyprofDumpManager(self.manager)

        global _prof_singleton
        if _prof_singleton is not None:
            raise RuntimeError("IML: Only a single profiler.Profiler object can be created; use iml.handle_iml_args + iml.prof instead.")
        _prof_singleton = self

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

            if hasattr(args, iml_argname) and getattr(args, iml_argname) is not None:
                argval = getattr(args, iml_argname)
                return argval

            if not allow_none and default_arg is None:
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

        # TODO: bind process to a different core to avoid interference?
        # self.bg_dumper = ProcessPoolExecutor(max_workers=1)
        self.debug = get_argval('debug', debug, False)
        self.bg_dumper = ForkedProcessPool(name="bg_dumper", debug=self.debug)

        self._op_stack = []
        self._start_us = None
        self._stop_us = None

        """
        If set, require the user to call prof.end_operation
        (don't allow a call to prof.set_operation to also count as a call to prof.end_operation)
        """
        self.require_end_operation = require_end_operation
        self.start_call_us = dict()
        self.end_call_us = dict()

        self.next_trace_id = None
        self.process_name = None
        # self.init_trace_id()
        self._tfprof_enabled = False
        self._pyprof_enabled = False
        self.total_profile_time_sec = 0
        self.directory = get_argval('directory', directory, None, allow_none=False)
        self.disable = get_argval('disable', disable, False)
        self.python = get_argval('python', python, False)
        self.exit_early = exit_early
        self.tfprof = tfprof
        self.c_lib_func_pyprof_pattern = c_lib_func_pyprof_pattern
        self.repetition_time_limit_sec = repetition_time_limit_sec
        self.num_calls = get_argval('num_calls', num_calls, None)
        self.trace_time_sec = get_argval('trace_time_sec', trace_time_sec, None)
        self._last_warned_trace_time_sec = None
        self._should_finish_idx = 0
        self.start_trace_time_sec = None
        self.num_traces = get_argval('num_traces', num_traces, None)
        self.keep_traces = get_argval('keep_traces', keep_traces, False)
        self.bench_name = get_argval('bench_name', bench_name, None)

        self.util_sampler_pid = get_internal_argval('util_sampler_pid')
        self.handle_utilization_sampler = False

        self.start_trace_time_sec = get_internal_argval('start_trace_time_sec')
        self.phase = get_internal_argval('phase', DEFAULT_PHASE)

        self.start_measuring_call = get_argval('start_measuring_call', start_measuring_call, None)

        self.unit_test = get_argval('unit_test', unit_test, False)
        self.unit_test_name = get_argval('unit_test_name', unit_test_name, None)
        if self.unit_test:
            self.ut = unit_test_util.UnitTestDataDumper(debug=self.debug)

        if not self.tfprof:
            self.cuda_profiler = CUDAProfiler()
        self.start_t = dict()
        self.end_t = dict()
        self.time_sec = dict()
        # How many times has a block of code that we are intending to profile been run?
        # We expect to run that block of code at least
        # (self.start_measuring_call + self.num_calls) times.
        self.code_count = dict()
        self.steps = 0
        clib_wrap.set_step(self._pyprof_step, ignore_disable=True)
        # self.step_start_tracing = None
        self.average_time_per_call_sec = None
        self.average_time_per_call_no_profile_sec = None

        if self.python:
            self.pyprof = PythonProfiler(self.directory)

        # self.sess = None
        # self.pctx = None

        # Total times collected from running profiled operations.
        # self.profile_sec = []
        # Total times collected from running un-profiled operations.
        # self.no_profile_sec = []

        # clib_wrap.wrap_libs()

        # assert ( self.num_calls is None and self.start_measuring_call is None ) or \
        #        ( self.num_calls is not None and self.start_measuring_call is not None )
        # assert self.start_measuring_call is not None

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
        logging.info("> Delete trace files rooted at {dir}".format(dir=self.directory))
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                # logging.info("> Consider {path}".format(path=path))
                if is_trace_file(path):
                    logging.info("> RM {path}".format(path=path))
                    os.remove(path)
                    # trace_id = int(m.group('trace_id'))
                    # self.next_trace_id = max(self.next_trace_id, trace_id + 1)

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

        if DEBUG:
            logging.info("> Using next_trace_id = {id}".format(id=self.next_trace_id))

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
            logging.info("> Dynamic iterations for bench_name={b} decided on: {it} iterations".format(
                it=iterations,
                b=bench_name))
            logging.info("  1 repetition takes ~ {sec} seconds".format(sec=total_time_sec))

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

    def _should_measure_call(self, bench_name=NO_BENCH_NAME):
        # return self.start_measuring_call is None or self.bench_name == bench_name
        return ( self.bench_name == NO_BENCH_NAME or self.bench_name == bench_name ) and (
                self.start_measuring_call is None or \
                # (self.code_count[bench_name] - 1) >= self.start_measuring_call
                self.steps + 1 >= self.start_measuring_call
                # (self.code_count[bench_name] - 1) >= self.start_measuring_call
        )

    def enable_profiling(self, bench_name=NO_BENCH_NAME):
        if not self._should_measure_call(bench_name):
            return False

        self.start_t[bench_name] = time.time()
        return True

    def profile_time_sec(self, bench_name):
        return self.time_sec[bench_name]

    def disable_profiling(self, bench_name=NO_BENCH_NAME, num_calls=1):
        if not self._should_measure_call(bench_name):
            self.code_count[bench_name] = self.code_count.get(bench_name, 0) + 1
            return

        end_time_sec = time.time()
        self.end_t[bench_name] = end_time_sec
        self.time_sec[bench_name] = self.time_sec.get(bench_name, 0.) + end_time_sec - self.start_t[bench_name]
        self.code_count[bench_name] = self.code_count.get(bench_name, 0) + num_calls

    # def set_session(self, sess):
    #     if self.disable:
    #         return
    #
    #     if DEBUG:
    #         print_stacktrace("> set_session = {sess}".format(sess=sess))
    #
    #     assert sess is not None
    #
    #     if self.sess is not None:
    #         raise NotImplementedError("Haven't implemented profiling using multiple session objects.")
    #     self.sess = sess
    #
    #     if self.pctx is not None:
    #         self.pctx.set_session(self.sess)
    #
    #     if self._cur_operation != NO_BENCH_NAME and not self._tfprof_enabled:
    #         self._start_tfprof(allow_skip=False)

    # def _cur_session(self, allow_none=False):
    #     if self.sess is not None:
    #         return self.sess
    #
    #     sess = tf.get_default_session()
    #     if sess is None and not allow_none:
    #         raise RuntimeError(
    #             "Couldn't find current session; you either need to call Profiler.set_session(sess), "
    #             "or do \"with sess.as_default():\"")
    #
    #     return sess

    def start(self, start_utilization_sampler=False, handle_utilization_sampler=False):
        PROFILERS.append(self)

        if self.disable:
            return

        self.handle_utilization_sampler = handle_utilization_sampler
        if start_utilization_sampler or handle_utilization_sampler:
            self._launch_utilization_sampler()

        self._start_us = now_us()

        if self.unit_test:
            self.ut.start()
        self._start_profiling()

    def _maybe_end_operations(self):
        while len(self._op_stack) != 0:
            self.end_operation(self._cur_operation, skip_finish=True)

    def stop(self, stop_utilization_sampler=False):
        PROFILERS.remove(self)

        if self.disable:
            return

        if stop_utilization_sampler:
            # Q: Any way to avoid forgetting to terminate utilization sampler?
            # A: Not really...can add a cleanup() script/python-API to call in the code when the programmer expects to terminate...
            # harder to forget than a call to stop() that's missing a parameter.
            self._terminate_utilization_sampler()

        self._maybe_end_operations()
        self._maybe_finish(finish_now=True, should_exit=False)

    def _start_tfprof(self):
        """
        Meant to be called right before we start measuring individual operations.

        Does setup needed for profiling:
        - Wrap TF library to measure python API time
        - Enable tfprof to hook into session.run(...) for measuring TF-side GPU/C++ API time

        :return:
        """
        if self.process_name is None:
            raise RuntimeError("You need to call profiler.set_process_name(...) before profiling.")
        assert self.phase is not None

        if self._tfprof_enabled:
            return

        # if self.step_start_tracing is None:
        #     self.step_start_tracing = self.steps

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

        self._tfprof_enable_tracing()
        # self.pctx.enable_tracing()

        self._tfprof_enabled = True

    def _tfprof_enable_tracing(self):
        for session in iml_profiler.profiler.session.ACTIVE_SESSIONS:
            pctx = ProfileContextManager.get_profile_context(session)
            pctx.enable_tracing()

    def _tfprof_disable_tracing(self):
        for session in iml_profiler.profiler.session.ACTIVE_SESSIONS:
            pctx = ProfileContextManager.get_profile_context(session)
            pctx.disable_tracing()

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
        self._tfprof_disable_tracing()

        self._tfprof_enabled = False

    @property
    def is_tfprof_enabled(self):
      return self._tfprof_enabled

    def _check_profiling_started(self):
        global PROFILERS
        started = self in PROFILERS
        if not started:
            raise RuntimeError("IML: You need to call profiler.start() before profiling.")

    def _push_operation(self, bench_name):
        # Currently we don't bother to support the following:
        # prof.set_operation('op1')
        # prof.set_operation('op1') <-- probably a bug.
        assert bench_name not in self._op_stack
        assert bench_name != NO_BENCH_NAME
        self._op_stack.append(bench_name)

    def _pop_operation(self, bench_name):
        assert self._op_stack[-1] == bench_name
        self._op_stack.pop()

    @property
    def _cur_operation(self):
        if len(self._op_stack) == 0:
            return NO_BENCH_NAME
        return self._op_stack[-1]

    def set_operation(self, bench_name):
        if self.disable:
            return

        self._check_profiling_started()

        if not(
            self._should_measure_call(bench_name)
        ):
            return

        if DEBUG:
            logging.info("> set_operation(op={op})".format(op=bench_name))

        self._push_operation(bench_name)
        # if len(self._op_stack) == 1:
        #     self._start_tfprof()
        #     self._start_pyprof()

        self.enable_profiling(bench_name)
        self.start_call_us[bench_name] = clib_wrap.now_us()

    def operation(self, operation):
        return Operation(operation, prof=self)

    def profile(self, process_name, phase_name=DEFAULT_PHASE, handle_utilization_sampler=True):
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
            handle_utilization_sampler=handle_utilization_sampler)

    def _start_profiling(self):
        """
        Do any initialization/setup needed to trace events.

        e.g. wrap tensorflow library C++ API calls, wrap simulator API calls, etc.

        :return:
        """
        self._start_tfprof()
        self._start_pyprof()

    def _init_trace_time(self):
        """
        Record the start time-since-epoch of tracing information being collected.

        (i.e. the time should proceed start_time_us of all recorded events)
        """
        if self.start_trace_time_sec is None:
            self.start_trace_time_sec = time.time()

    def _start_pyprof(self):
        if self._pyprof_enabled:
            return
        self._init_trace_time()
        clib_wrap.wrap_libs()
        # if self.step_start_tracing is None:
        #     self.step_start_tracing = self.steps
        clib_wrap.enable_tracing()
        clib_wrap.set_step(self._pyprof_step, expect_traced=True)
        if (tensorflow_profile_context.DEBUG or TF_PRINT_TIMESTAMP) and clib_wrap.is_recording():
            logging.info("> RECORDING pyprof_step = {step}".format(step=self._pyprof_step))
        if self.python:
            self.pyprof.enable()
        self._pyprof_enabled = True

    def _stop_pyprof(self):
        if not self._pyprof_enabled:
            return
        clib_wrap.disable_tracing()
        self._pyprof_enabled = False
        if self.python:
            self.pyprof.disable()

    @property
    def _pyprof_step(self):
        """
        Some operations just won't call session.run(...).
        In that case, we cannot depend on using the tfprof step number since
        it won't increment between next_step calls.
        So, just use our internal next_step counter.
        """
        return self.steps

    # @property
    # def _tfprof_step(self):
    #     """
    #     Internally, tfprof profiler (tensorflow_profile_context.ProfileContext) increments a step counter
    #     every time session.run() is called.
    #
    #     Keep pyprof step number in sync by returning it directly.
    #
    #     :return:
    #     """
    #     assert self.pctx is not None
    #     return self.pctx._step

    def end_operation(self, bench_name, skip_finish=False):
        assert bench_name != NO_BENCH_NAME

        if self.disable:
            return

        if DEBUG:
            should_finish = self.should_finish()
            logging.info('> end_operation({op}), tracing time = {sec}, should_finish = {should_finish}'.format(
                sec=self._total_trace_time_sec(),
                should_finish=should_finish,
                op=bench_name))


        # if self.calls_traced > self.num_calls and len(self._op_stack) > 0 and self.calls_traced % WARN_EVERY_CALL_MODULO == 0:
        #     logging.info("> IML: WARNING, use more fine grained operations so we can free memory by dumping traces more frequently")
        #     logging.info("  - calls traced = {calls_traced}, number of calls per-trace = {num_calls}".format(
        #         calls_traced=self.calls_traced,
        #         num_calls=self.num_calls,
        #     ))
        #     logging.info("  - currently active operations: {ops} <-- make these more fine-grained!".format(
        #         ops=self._op_stack))

        if self._cur_operation == NO_BENCH_NAME and bench_name != self._cur_operation:
            """
            start_operation was called, but was skipped since _should_measure_call 
            returned false.
            """
            assert len(self._op_stack) == 0
            return

        if self._cur_operation != bench_name:
            raise RuntimeError(textwrap.dedent("""
            Detected non stack-oriented nesting of profiling statements:
                prof.set_operation({b1})
                ...
                prof.end_operation({b2})
            """.format(
                b1=self._cur_operation,
                b2=bench_name,
            )))

        self.end_call_us[bench_name] = clib_wrap.now_us()
        self.disable_profiling(bench_name, num_calls=1)
        # Record the last amount of time in between returning
        # from a call to q_forward, and finishing benchmarking.
        # This will include time spent in the tensorflow python API
        clib_wrap.record_python_event('Finish python benchmark', self.end_call_us[bench_name])
        # time_sec = (self.end_call_us[bench_name] - self.start_call_us[bench_name])/MICROSECONDS_IN_SECOND
        # if clib_wrap.is_recording():
        #     self.profile_sec.append(time_sec)
        # else:
        #     self.no_profile_sec.append(time_sec)
        clib_wrap.record_operation(self.start_call_us[bench_name], self.end_call_us[bench_name],
                                   op_name=bench_name)
        del self.start_call_us[bench_name]
        del self.end_call_us[bench_name]

        self._pop_operation(bench_name)

        if self.trace_time_sec is not None and \
                self._total_trace_time_sec() > self.trace_time_sec and \
                len(self._op_stack) > 0 and \
                ( self._last_warned_trace_time_sec is None or time.time() - self._last_warned_trace_time_sec >= WARN_EVERY_SEC ):
            stacktrace = get_stacktrace()
            logging.warning(textwrap.dedent("""\
            IML: Warning; tracing time (sec) has exceeded limit (--iml-trace-time-sec {limit_sec}), but annotations are still active:
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
                stack=textwrap.indent(''.join(stacktrace), prefix="  "*2),
            ))
            self._last_warned_trace_time_sec = time.time()

        if len(self._op_stack) == 0:
            self.steps += 1
            # self._stop_pyprof()
            # self._stop_tfprof()

        if not skip_finish:
            self._maybe_finish(debug=self.debug)

    def set_process_name(self, process_name):
        self.process_name = process_name
        clib_wrap.set_process_name(process_name)
        self.init_trace_id()
        # self._maybe_init_profile_context()

    def set_machine_name(self, machine_name):
        self.machine_name = machine_name

    @property
    def is_root_process(self):
        return self.process_name is None

    def _launch_utilization_sampler(self):
        if not self.is_root_process:
            logging.info("IML: Warning; you are starting the utilization sampler later than expected (this is not the root process of your training script")

        if self.util_sampler_pid is not None:
            logging.info("IML: Warning; you're already running utilization sampler @ pid={pid}".format(pid=self.util_sampler_pid))
            return

        util_cmdline = ['iml-util-sampler']
        util_cmdline.extend(['--iml-directory', _a(self.directory)])
        if self.debug:
            util_cmdline.append('--iml-debug')
        # if self.debug:
        logging.info("> CMDLINE: {cmd}".format(cmd=' '.join(util_cmdline)))
        # self.util_sampler_proc = subprocess.Popen(util_cmdline, creationflags=subprocess.DETACHED_PROCESS)
        self.util_sampler_proc = subprocess.Popen(util_cmdline)
        self.util_sampler_pid = self.util_sampler_proc.pid
        logging.info("IML: CPU/GPU utilization sampler running @ pid={pid}".format(pid=self.util_sampler_pid))

    def _terminate_utilization_sampler(self, warn_terminated=True):
        assert self.util_sampler_pid is not None
        logging.info("IML: terminating CPU/GPU utilization sampler @ pid={pid}".format(pid=self.util_sampler_pid))

        try:
            proc = psutil.Process(self.util_sampler_pid)
        except psutil.NoSuchProcess as e:
            if warn_terminated:
                logging.info("IML: Warning; tried to terminate utilization sampler @ pid={pid} but it wasn't running".format(pid=self.util_sampler_pid))
            return

        proc.terminate()

    def set_phase(self, phase):
        assert type(phase) == str

        if self.disable:
            return

        if self.unit_test:
            self.ut.set_phase(phase)

        if len(self._op_stack) != 0:
            raise RuntimeError("IML: ERROR, you cannot change phases while operations are in-progress: ops = {ops}".format(
                ops=self._op_stack))

        # assert self.session.pctx.phase is not None
        # Record the current tfprof for the current phase as a DumpThunk.
        # Also, create a new pctx for the next phase.
        # ProfileContextManager.recreate_profile_context(self.session, phase)
        ProfileContextManager.recreate_sessions_profile_contexts(phase)
        # Dump the DumpThunk's.
        # self.dump_trace(finish_now=False, debug=self.debug)

        self.phase = phase
        clib_wrap.set_phase(phase)

        # self.init_trace_id()
        # self._maybe_init_profile_context()

    def maybe_terminate_utilization_sampler(self, warn_terminated):
        if self.handle_utilization_sampler:
            self._terminate_utilization_sampler(warn_terminated)

    def finish(self, should_exit=True):
        if self.unit_test:
            self.ut.stop()

        self.maybe_terminate_utilization_sampler(warn_terminated=False)

        # Record an event [PROC:<process_name>] that marks when this process started/finished execution.
        self._record_process_event()

        # For each active session, schedule its results to be dumped.
        # for sess in ACTIVE_SESSIONS:
        #   self.dump_tfprof(sess)
        logging.info("> IML: Schedule any remaining traces to be dumped.")
        for session in iml_profiler.profiler.session.ACTIVE_SESSIONS:
            self._dump_tfprof(session, debug=self.debug)
        # At the very least, make sure to dump the [PROC:<process_name>] we recorded above.
        # Q: How frequently should we dump pyprof data?
        # A: We'd like to keep it under a file-size limit...but computing exact proto-size isn't practical.
        #    Instead, lets just roughly count the # of events, and use that as a proxy for proto file size.
        if self.debug:
            logging.info("> IML: _dump_pyprof")
        self._dump_pyprof(debug=self.debug)
        if self.debug:
            logging.info("> IML: _dump_pyprof done")

        # if self._should_dump_pyprof:
        #     self._dump_pyprof()

        if self.unit_test:
            logging.info("> IML: _dump_unit_test")
            self._dump_unit_test()
            logging.info("> IML: _dump_unit_test done")
            # Make sure ALL unit-test data has been recorded before we exit.
            if not self.ut.is_empty:
                self.ut.debug_empty()
                assert self.ut.is_empty

        # Wait on all dump-processes to finish executing.
        # ForkedProcessPool.shutdown(wait=True)
        logging.info("> IML: Waiting for trace-dump background threads to complete.")
        self.bg_dumper.shutdown()

        # Wait for any async tfprof trace file dumps in C++ to finish.
        logging.info("> IML: Wait for tfprof trace-file dumps in TensorFlow C++ to finish.")
        c_api_util.await_trace_data_dumps()

        logging.info("> IML: Done")
        if should_exit:
            logging.info("> IML: Exiting training script early")
            sys.exit(0)

    # def dump_trace(self, finish_now, debug=False):
    #     # We shouldn't be in the middle of measuring an operation.
    #     start_us = now_us()
    #     assert self._cur_operation == NO_BENCH_NAME
    #
    #     self._stop_pyprof()
    #     self._stop_tfprof()
    #
    #     if finish_now:
    #         # Record a "special" operation event that spans the prof.start()/stop() calls
    #         # for the currently running process.
    #         assert self._start_us is not None
    #         assert self._stop_us is not None
    #         event_name = op_process_event_name(self.process_name)
    #         clib_wrap.set_step(self._pyprof_step,
    #                            ignore_disable=True)
    #         # BUG TODO: Should we use our own CATEGORY_PROCESS for process-events?
    #         # Otherwise, we may be unable to disambiguate from a CATEGORY_OPERATION of the same name as the process.
    #         # Looks like we recorded it as [PROC:<process-name>] to prevent conflicts.
    #         # I cannot remember how this is handled downstream during analysis.
    #         clib_wrap.record_event(CATEGORY_OPERATION, event_name, self._start_us, self._stop_us,
    #                                ignore_disable=True)
    #
    #     assert not self._tfprof_enabled
    #     assert not self._pyprof_enabled
    #     # self._stop_tfprof()
    #
    #     # ProfileGlobals.files_after = ls_files(self.out_dir)
    #
    #     # Put this here to test that cleanup_files doesn't delete nvprof/pyprof files
    #     self._dump(start_us)
    #
    #     self.step_start_tracing = self.steps

    # def _discard_profiling_data(self):
    #     clib_wrap.clear_pyprof_profiling()

    def _dump_tfprof(self, session, debug=False):
        """
        Dump a tfprof proto file for a given session.
        Should get called whenever the number of traces for this session exceeds a threshold.

        :param session:
        :param pctx:
        :return:
        """
        pctx = ProfileContextManager.get_profile_context(session)
        trace_id = self.next_trace_id
        # tfprof_path = self.tfprof_path(ses)
        tfprof_path = self.tfprof_path(session.session_id, trace_id)
        tfprof_dumper = TfprofDumper(trace_id, session, self.process_name, tfprof_path, debug=debug)
        tfprof_dumper.dump()

        # self.bg_dumper.submit(
        #     name='TfprofDumper.dump({path})'.format(path=tfprof_path),
        #     fn=tfprof_dumper.dump)
        # After process has forked for dumping trace-data, clear the current process' trace-data.

        pctx.clear()
        self.next_trace_id += 1

    def _old_dump_tfprof(self, session, debug=False):
        """
        Dump a tfprof proto file for a given session.
        Should get called whenever the number of traces for this session exceeds a threshold.

        :param session:
        :param pctx:
        :return:
        """
        pctx = ProfileContextManager.get_profile_context(session)
        trace_id = self.next_trace_id
        # tfprof_path = self.tfprof_path(ses)
        tfprof_path = self.tfprof_path(session.session_id, trace_id)
        tfprof_dumper = TfprofDumper(trace_id, session, self.process_name, tfprof_path, debug=debug)
        self.bg_dumper.submit(
            name='TfprofDumper.dump({path})'.format(path=tfprof_path),
            fn=tfprof_dumper.dump)
        # After process has forked for dumping trace-data, clear the current process' trace-data.
        pctx.clear()
        self.next_trace_id += 1

    # TODO: Don't set to True until we've made sure bg_dumper actually works!
    UNIT_TEST_ASYNC_DUMP = False
    def _dump_unit_test(self):
        assert self.unit_test
        assert self.unit_test_name is not None
        trace_id = self.next_trace_id
        self.next_trace_id += 1
        dump_kwargs = {
            'directory':self.out_dir,
            'trace_id':trace_id,
            'bench_name':self.bench_name,
            'process_name':self.process_name,
            'test_name':self.unit_test_name,
        }
        if Profiler.UNIT_TEST_ASYNC_DUMP:
            self.bg_dumper.submit(
                name='UnitTestDataDumper.dump(trace_id={trace_id})'.format(
                    trace_id=trace_id),
                fn=self.ut.dump,
                **dump_kwargs)
        else:
            self.ut.dump(**dump_kwargs)
        # We might be in the middle of a phase...
        # i.e.
        # - phase_start[cur_phase] is recorded, but NOT phase_end[cur_phase]
        # - The dump we are performing right now WON'T include cur_phase; but a future dump will
        # => clear should NOT forget what the current phase is.
        self.ut.clear_dump()

    @property
    def _should_dump_pyprof(self):
        return clib_wrap.should_dump_pyprof()

    # TODO: decide where to call dump_tfprof / dump_pyprof from.
    def _dump_pyprof(self, config_kwargs=dict(), debug=False):
        start_sec = time.time()
        pyprof_trace = clib_wrap.get_pyprof_trace()
        trace_id = self.next_trace_id
        pyprof_trace_key = self.pyprof_proto_path(trace_id)
        self.pyprof_dump_manager.put(pyprof_trace_key, pyprof_trace)
        pyprof_dumper = PyprofDumper(
            trace_id=trace_id,
            config_path=self.config_path(trace_id),
            c_lib_func_pyprof_pattern=self.c_lib_func_pyprof_pattern,
            num_calls=self.num_calls,
            start_measuring_call=self.start_measuring_call,
            average_time_per_call_sec=self.average_time_per_call_sec,
            average_time_per_call_no_profile_sec=self.average_time_per_call_no_profile_sec,
            config_kwargs=config_kwargs,
            process_name=self.process_name,
            phase=self.phase,
            pyprof_proto_path=self.pyprof_proto_path(trace_id),
            pyprof_call_times_path=self.pyprof_call_times_path(trace_id),
            pyprof_step=self._pyprof_step,
            pyprof_dump_manager=self.pyprof_dump_manager,
            pyprof_trace_key=pyprof_trace_key,
            debug=debug)
        self.next_trace_id += 1
        self.bg_dumper.submit(
            name="PyprofDumper.dump({path})".format(
                path=self.pyprof_proto_path(trace_id)),
            fn=pyprof_dumper.dump,
        )
        end_sec = time.time()
        time_sec = end_sec - start_sec
        if DEBUG:
            logging.info("Dump pyprof took {sec} seconds on the critical path".format(
                sec=time_sec,
            ))

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

    def should_finish(self, finish_now=False, skip_finish=False):
        total_trace_time_sec = self._total_trace_time_sec()
        ret = finish_now or (
                (
                        self.num_traces is not None and
                        self.next_trace_id >= self.num_traces
                ) or (
                        self.trace_time_sec is not None
                        and total_trace_time_sec >= self.trace_time_sec
                )
        )
        self._should_finish_idx += 1
        if DEBUG and (ret or self._should_finish_idx % 1000 == 0):
            logging.info(textwrap.indent(textwrap.dedent("""
            - process_name = {proc}
            - finish_now = {finish_now}
            - skip_finish = {skip_finish}
            """.format(
                proc=self.process_name,
                finish_now=ret,
                skip_finish=skip_finish,
            )), prefix="  "))
            if self.num_traces is not None:
                logging.info(textwrap.indent(textwrap.dedent("""
                - self.next_trace_id >= self.num_traces = {next_bool}
                  - self.next_trace_id = {next_trace_id}
                  - self.num_traces = {num_traces}""".format(
                    num_traces=self.num_traces,
                    next_bool=self.next_trace_id >= self.num_traces,
                    next_trace_id=self.next_trace_id,
                )), prefix="  "))
            if self.trace_time_sec is not None:
                logging.info(textwrap.indent(textwrap.dedent("""
                - total_trace_time_sec >= self.trace_time_sec = {total_bool}
                  - total_trace_time_sec = {total_trace_time_sec}
                  - self.trace_time_sec = {trace_time_sec}""".format(
                    total_bool=total_trace_time_sec >= self.trace_time_sec,
                    total_trace_time_sec=total_trace_time_sec,
                    trace_time_sec=self.trace_time_sec,
                )), prefix="  "))
        return ret

    def _total_trace_time_sec(self):
        if self.trace_time_sec is None:
            return None
        now_sec = time.time()
        return now_sec - self.start_trace_time_sec

    def _record_process_event(self):
        self._stop_us = now_us()
        # Record a "special" operation event that spans the prof.start()/stop() calls
        # for the currently running process.
        assert self._start_us is not None
        assert self._stop_us is not None
        event_name = op_process_event_name(self.process_name)
        clib_wrap.set_step(self._pyprof_step,
                           ignore_disable=True)
        # BUG TODO: Should we use our own CATEGORY_PROCESS for process-events?
        # Otherwise, we may be unable to disambiguate from a CATEGORY_OPERATION of the same name as the process.
        # Looks like we recorded it as [PROC:<process-name>] to prevent conflicts.
        # I cannot remember how this is handled downstream during analysis.
        clib_wrap.record_event(CATEGORY_OPERATION, event_name, self._start_us, self._stop_us,
                               ignore_disable=True)

    def _maybe_finish(self, finish_now=False, should_exit=True, debug=False):
        should_finish = self.should_finish(finish_now)
        if not should_finish:
            return

        self._stop_us = now_us()

        while len(self._op_stack) > 0:
            # Pass skip_finish=True to avoid recursively calling this.
            self.end_operation(self._cur_operation, skip_finish=True)

        self.finish(should_exit=should_exit)

    def done_measuring(self):
        return self.num_calls is not None and \
               self.code_count[self.bench_name] >= self.total_calls_to_run

    @property
    def total_calls_to_run(self):
        if self.start_measuring_call is None:
            return self.num_calls
        return self.num_calls + self.start_measuring_call - 1

    def should_stop(self):
        # If your using the profiler this way, you need to provide self.num_calls!
        # assert self.num_calls is not None
        # Can only measure one bench_name at a time.
        return self.exit_early and \
               self.done_measuring()

class CUDAProfiler:
    def __init__(self):
        # NOTE: CUDA profiling output has already been specified when this script was launched.
        # self.profile_basename = profile_basename
        self.already_enabled = False

    def start(self):
        # NOTE: we assume the CUDA
        self.already_enabled = cudaprofile.is_profiler_enabled()
        if not self.already_enabled:
            cudaprofile.start()

    def stop(self):
        if not self.already_enabled:
            cudaprofile.stop()

    def dump(self):
        # Dumping is performed externally by nvprof once the program terminates.
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class Operation:
    def __init__(self, operation, prof):
        self.operation = operation
        self.prof = prof

    def __enter__(self):
        self.prof.set_operation(self.operation)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Q: Should we call this when an exception is thrown?
        self.prof.end_operation(self.operation)

class Profile:
    def __init__(self, prof, process_name, phase_name=DEFAULT_PHASE, handle_utilization_sampler=True):
        self.process_name = process_name
        self.phase_name = phase_name
        self.prof = prof
        self.handle_utilization_sampler = handle_utilization_sampler

    def __enter__(self):
        self.prof.set_process_name(self.process_name)
        self.prof.set_phase(self.phase_name)
        self.prof.start(handle_utilization_sampler=self.handle_utilization_sampler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Q: Should we call this when an exception is thrown?
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
                logging.info("WARNING: couldn't enable monotonic_clock for cProfiler; "
                      "are you using a modified python3 with support for collecting raw start/end timestamps?")

        # if self.record_call_times:
        if hasattr(self.profile, 'make_record_call_times'):
            self.profile.make_record_call_times()
        else:
            logging.info("WARNING: couldn't enable make_record_call_times for cProfiler; "
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
        logging.info("> dump pyprof call_times data @ {path}".format(
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

    # @property
    # def _call_times_path(self):
    #     return _j(self.directory, "python_profile{bench}.call_times.json".format(
    #         bench=bench_suffix(self.bench_name)))

    # @property
    # def _call_times_path(self):
    #     return _j(self.directory, "pyprof_call_times{bench}.call_times.json".format(
    #         bench=bench_suffix(self.bench_name)))

    @property
    def _stats_path(self):
        return _j(self.directory, "python_profile.txt")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# 100 ms
MIN_UTIL_SAMPLE_FREQUENCY_SEC = 100/MILLISECONDS_IN_SECOND
# 500 ms
DEFAULT_UTIL_SAMPLE_FREQUENCY_SEC = 500/MILLISECONDS_IN_SECOND
def get_util_sampler_parser(only_fwd_arguments=False):
    """
    :param fwd_arguments:
        Only add arguments that should be forwarded to utilization_sampler.py from ML scripts.
    :return:
    """
    parser = argparse.ArgumentParser("Sample GPU/CPU utilization over the course of training")
    parser.add_argument('--iml-directory',
                        # required=True,
                        help=textwrap.dedent("""
    IML: profiling output directory.
    """))
    parser.add_argument('--iml-debug',
                        action='store_true',
                        help=textwrap.dedent("""
    IML: debug profiler.
    """))
    if only_fwd_arguments:
        return parser

    parser.add_argument('--iml-util-sample-frequency-sec',
                        type=float,
                        default=DEFAULT_UTIL_SAMPLE_FREQUENCY_SEC,
                        help=textwrap.dedent("""
    IML: How frequently (in seconds) should we sample GPU/CPU utilization?
    default: sample every 500 ms.
    """))
    parser.add_argument('--iml-util-dump-frequency-sec',
                        type=float,
                        default=10.,
                        help=textwrap.dedent("""
    IML: How frequently (in seconds) should we sample GPU/CPU utilization?
    default: dump every 10 seconds.
    """))
    parser.add_argument('--iml-debug-single-thread',
                        action='store_true',
                        help=textwrap.dedent("""
    IML: debug with single thread.
    """))

    parser.add_argument('--measure-samples-per-sec',
                        action='store_true',
                        help=textwrap.dedent("""
    Determines reasonable values for --iml-util-sample-frequency-sec.
    
    How fast can we call nvidia-smi (to sample GPU utilization)?  
    How fast can we gather CPU utilization?
    """))

    parser.add_argument('--kill',
                        action='store_true',
                        help=textwrap.dedent("""
    Kill any existing instances of this script that are still running, then exit.
    """))

    return parser

def add_iml_arguments(parser):
    iml_parser = parser.add_argument_group("IML")
    iml_parser.add_argument('--iml-nvprof-enabled', action='store_true', help=textwrap.dedent("""
        IML: is nvprof running?
        
        Internal use only; 
        used to determine whether this python script has been invoked using nvprof.
        If it hasn't, the script will re-invoke itself with nvprof.
    """))
    # iml_parser.add_argument('--iml-tfprof', action='store_true', help=textwrap.dedent("""
    #     IML: use tfprof TensorFlow profiling utility INSTEAD of nvprof.
    # """))
    iml_parser.add_argument('--iml-num-calls', type=int, default=1000,
                        help="IML: how many calls should be measured in a single trace?")
    iml_parser.add_argument('--iml-trace-time-sec', type=float,
                        help="IML: how long should we profile for, in seconds; "
                             "tracing will stop when either "
                             "we've collected --iml-num-traces OR "
                             "--iml-trace-time-sec has been exceeded")
    iml_parser.add_argument('--iml-internal-start-trace-time-sec', type=float,
                        help=textwrap.dedent("""
        IML: (internal use)
        The start time of tracing (in seconds). 
        This gets inherited by child processes.
    """))
    iml_parser.add_argument('--iml-phase',
                        help=textwrap.dedent("""
        IML: (internal use)
        The "phase" of training captured by this script. 
        The phase covered by a script may change during training.
        E.g. a single script could handle "simulator" and "gradient_update" phases.
        This gets inherited by child processes.
    """))
    iml_parser.add_argument('--iml-internal-parent-process-name',
                        help=textwrap.dedent("""
        IML: (internal use)
        The process name of the parent that launched this child python process.
        i.e. whatever was passed to iml_profiler.api.prof.set_process_name('forker')
        Internally, this is used for tracking "process dependencies".
    """))
    iml_parser.add_argument('--iml-util-sampler-pid',
                        help=textwrap.dedent("""
        IML: (internal use)
        The pid of the utilization_sampler.py script that samples CPU/GPU utilization during training.
        We need to keep this so we can terminate it once we are done.
    """))

    iml_parser.add_argument('--iml-num-traces', type=int,
                        # default=10,
                        help="IML: how many traces should be measured?")
    iml_parser.add_argument('--iml-keep-traces', action='store_true', help=textwrap.dedent("""
        IML: DON'T delete any existing trace files; keep them and append to them.
        
        Useful if your ML script launches worker processes repeatedly.
    """))
    iml_parser.add_argument('--iml-python', action='store_true', help=textwrap.dedent("""
        IML: Collecting python profiler (pyprof) data for profiled operations.
        
        Python profiling data is grouped into per-operation summaries, instead of 
        presenting profiling data process-wide.
        
        This prevent overwhelming the user with too much information.
    """))
    iml_parser.add_argument('--iml-fuzz', action='store_true', help=textwrap.dedent("""
        IML: \"Fuzz\" the script for calls to TensorFlow API's.
        
        Useful if you have no idea where the training-loop of an ML script is located. 
        
        Adds breakpoints / dumps stack traces when certain TensorFlow API's are called; 
        for e.g. sesssion.run(...) for running the computational graph
        (currently this is the only thing we trace).
    """))
    iml_parser.add_argument('--iml-disable', action='store_true', help=textwrap.dedent("""
        IML: Skip any profiling.
    """))
    iml_parser.add_argument('--iml-unit-test',
                        action='store_true',
                        help=textwrap.dedent("""
    IML: (for unit-testing) Record "actual results" needed for doing basics unit-test checks.
    """))
    iml_parser.add_argument('--iml-unit-test-name',
                        help=textwrap.dedent("""
    IML: (for unit-testing) name to store in IMLUnitTest.test_name.
    """))
    iml_parser.add_argument('--iml-debug', action='store_true', help=textwrap.dedent("""
        IML: debug profiler.
    """))
    iml_parser.add_argument('--iml-start-measuring-call', default=1, type=int,
                        help="IML: when should measuring begin?")
    iml_parser.add_argument('--iml-bench-name',
                        default=NO_BENCH_NAME,
                        help=textwrap.dedent("""
    IML: which code block should we measure?
    i.e. --iml-bench-name=some_bench
        # Just measure "some_bench", nothing else.
        profiler.profile('some_bench', do_some_bench)
    """))
    iml_parser.add_argument('--iml-directory',
                        help=textwrap.dedent("""
    IML: profiling output directory.
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

# class _ProfileGlobals:
#     def __init__(self):
#         self.files_before = None
#         self.files_after = None
#
#     def cleanup_files(self):
#         """
#         PROBLEM: script might output result files, and not be written to handle
#         re-running itself once those files exist.
#
#         - Modify script to overwrite/delete old output files:
#           Can probably handle this with IML wrapper.
#           files_before = [ files seen before run ]
#           files_after  = [ files seen after run ]
#           iml_files    = [ files output by iml ]
#           files_to_rm  = files_after - files_before - iml_files
#         """
#         # self.iml_files = [path for path in self.files_after if is_iml_file(path)]
#         self.files_to_rm = set(self.files_after).difference(set(self.files_before))
#         self.files_to_rm = [path for path in self.files_to_rm if not is_iml_file(path)]
#         for path in self.files_to_rm:
#             opts = ""
#             if os.path.isdir(path):
#                 opts = "-r "
#             logging.info("> RM {opts}{f}".format(
#                 opts=opts, f=path))

# ProfileGlobals = _ProfileGlobals()

def handle_iml_args(output_directory, parser, args, no_bench_name=False):
    pass
    # ProfileGlobals.files_before = ls_files(output_directory)

    # if args.iml_trace_time_sec is None and args.iml_num_traces is None:
    #     logging.info('IML: ERROR, you must provided at least one of --iml-trace-time-sec or --iml-num-traces')
    #     sys.exit(1)


def iml_argv(prof : Profiler, keep_executable=False, keep_non_iml_args=False):
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
    logging.info("> argv: {argv}".format(argv=' '.join(sys.argv)))
    # NOTE: sys.argv[0] is the python script name.
    args, extra_argv = parser.parse_known_args(sys.argv[1:])
    logging.info("> extra_argv: {argv}".format(argv=' '.join(extra_argv)))
    # Inherit arguments in our fork-ed children.
    args.iml_internal_start_trace_time_sec = prof.get_start_trace_time_sec()
    args.iml_phase = prof.phase
    if prof.process_name is None:
        raise RuntimeError("IML: You must call iml_profiler.api.prof.set_process_name('some_name') before forking children!")
    args.iml_internal_parent_process_name = prof.process_name
    args.iml_util_sampler_pid = prof.util_sampler_pid
    argv = args_to_cmdline(parser, args, keep_executable=keep_executable, keep_debug=False)
    if keep_non_iml_args:
        return argv + extra_argv
    return argv

def run_with_nvprof(directory, parser, args,
                    bench_name=NO_BENCH_NAME):
    logging.info("> Reinvoking script with nvprof; bench_name={b}".format(
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

def args_to_cmdline(parser, args,
                    keep_executable=True,
                    keep_py=False,
                    keep_debug=True):
    """
    NOTE: This WON'T keep arguments from sys.argv that AREN'T captured by parser.

    # To convert args namespace into cmdline:

    if args.option == True and option in parser:
            cmdline.append(--option)
    elif args.option == False and option in parser:
            pass
    elif type(args.option) in [int, str, float]:
            cmdline.append(--option value)
    elif type(args.open) in [list]:
            cmdline.append(--option elem[0] ... elem[n])
    else:
            raise NotImplementedError
    """

    def option_in_parser(parser, option):
        return parser.get_default(option) is not None

    def optname(option):
        return "--{s}".format(s=re.sub(r'_', '-', option))

    def is_py_file(path):
        return re.search(r'\.py$', path)
    def find_py_file_idx(argv):
        for i in range(len(argv)):
            if is_py_file(argv[i]):
                return i
        return None
    py_script_idx = find_py_file_idx(sys.argv)
    if py_script_idx is None:
        # No python executable; looks more like this:
        #
        # ['iml-test',
        #      '--train-script',
        #      'run_baselines.sh',
        #      '--test-name',
        #      'PongNoFrameskip-v4/docker',
        #      '--iml-directory',
        #      '/home/james/clone/baselines/output']
        py_script_idx = 0

    extra_opts = []
    if keep_debug and hasattr(args, 'debug') and args.debug:
        extra_opts.extend(["-m", "ipdb"])
    cmdline = []
    if keep_executable:
        cmdline.append(sys.executable)
    cmdline.extend(extra_opts)
    if keep_executable or keep_py:
        # Include python script path
        cmdline.extend(sys.argv[0:py_script_idx+1])
    else:
        # Don't include python script path
        cmdline.extend(sys.argv[0:py_script_idx])
    for option, value in args.__dict__.items():
        opt = optname(option)
        if value is None:
            continue

        if type(value) == bool:
            if value and option_in_parser(parser, option):
                cmdline.extend([opt])
            else:
                pass
        elif type(value) in [int, str, float]:
            cmdline.extend([opt, value])
        elif type(value) in [list] and len(value) > 0:
            cmdline.extend([opt])
            cmdline.extend(value)
        else:
            raise NotImplemented

    return [str(x) for x in cmdline]

# If you search for function names matching this pattern in pyprof output, they will match TensorFlow C++ API calls.
CLIB_TENSORFLOW_REGEX = r'(?:built-in.*pywrap_tensorflow)'
# We can manually wrap a c-library in order to record C API call times.  See test_call_c.py for how to do this.
CLIB_WRAPPER_REGEX = r'CLIB__.*'
def dump_config(path, **kwargs):
    config = dict()

    avail_gpus = get_available_gpus() # <-- This triggers entire GPU allocation...wtf!
    avail_cpus = get_available_cpus()
    # We want to be CERTAIN about which device TensorFlow is using.
    # If no GPUs are available, TF will use the CPU.
    # If a GPU is available, make sure only 1 is available so we are certain it's using that one.
    if not( (len(avail_gpus) == 1) or
            (len(avail_gpus) == 0 and len(avail_cpus) == 1) ):
        CUDA_VISIBLE_DEVICES = ENV.get('CUDA_VISIBLE_DEVICES', None)
        logging.info(textwrap.dedent("""
        ERROR: Multiple GPUs were found; IML benchmark requires only one GPU to be visible to TensorFlow via (for example) "export CUDA_VISIBLE_DEVICES=0".
        Use one of the below available GPUs:
        """))
        pprint.pprint({
            'avail_gpus':avail_gpus,
            'avail_cpus':avail_cpus,
            'CUDA_VISIBLE_DEVICES':CUDA_VISIBLE_DEVICES,
        }, indent=2)
        sys.exit(1)
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

def print_cmd(cmd):
    logging.info(textwrap.dedent("""
    RUN:
        cwd = {cwd}
        cmd = {cmd}
    """.format(
        cwd=os.getcwd(),
        cmd=" ".join([str(x) for x in cmd]),
    )))

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

GET_AVAILABLE_CPUS_CPU_INFO = cpuinfo.get_cpu_info()
def get_available_cpus():
    """
    Report a single [0..1] value representing current system-wide CPU utilization.

    psutil.cpu_percent() returns EXACTLY this.
    From psutil.cpu_percent docstring:
        "
        Return a float representing the current system-wide CPU
        utilization as a percentage.
        "

    NOTE: It's also possible to get INDIVIDUAL utilization for each CPU,
    if we choose to do that in the future.
    """
    device_name = GET_AVAILABLE_CPUS_CPU_INFO['brand']
    return {
        'device_name':device_name,
        'device_number':0,
    }

def get_visible_gpu_ids():
    if 'CUDA_VISIBLE_DEVICES' not in ENV:
        return None
    gpu_ids = sorted(int(gpu_id) for gpu_id in re.split(r'\s*,\s*', ENV['CUDA_VISIBLE_DEVICES']))
    return gpu_ids

def get_available_gpus():
    # $ tensorflow_cuda9 git:(opt-tfprof) ✗ nvidia-smi -L
    # GPU 0: GeForce RTX 2070 (UUID: GPU-e9c6b1d8-2b80-fee2-b750-08c5adcaac3f)
    # GPU 1: Quadro K4000 (UUID: GPU-6a547b6a-ae88-2aac-feb9-ae6b7095baaf)
    proc = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = proc.stdout.decode('utf-8').splitlines()
    device_dicts = []
    for line in lines:
        # if re.search(r'^\s*$', line):
        #     continue
        m = re.search(r'^GPU (?P<gpu_id>\d+):\s*(?P<gpu_name>.*)\s+\(UUID: (?P<gpu_uuid>.*)\)\s*', line)
        if m:
            device_dicts.append({
                "device_number":int(m.group('gpu_id')),
                "device_name":m.group('gpu_name'),
            })
    visible_gpu_ids = get_visible_gpu_ids()
    keep_devices = [gpu for gpu in device_dicts
                    if visible_gpu_ids is None or gpu['device_number'] in visible_gpu_ids]
    return keep_devices

    # Don't user TensorFlow to do this since it allocates the GPU when it runs...
    #
    # config = tf.ConfigProto()
    # # Allow multiple users to use the TensorFlow API.
    # config.gpu_options.allow_growth = True  # <--- even with this, it still user 645 MB!
    #
    # logging.info("Before list_local_devices")
    # import ipdb; ipdb.set_trace()
    # local_device_protos = tf_device_lib.list_local_devices(config) # <-- this trigger GPU allocation
    # device_protos = [x for x in local_device_protos if x.device_type == 'GPU']
    # device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    # return device_dicts

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

@contextlib.contextmanager
def _tracing_disabled(prof : Profiler):
    with clib_wrap.tracing_disabled():
        # was_tracing = prof.is_tracing()
        # if was_tracing:
        #     prof.disable_tracing()

        try:
            yield
        finally:
            pass
            # if was_tracing:
            #     prof.enable_tracing()

def process_directory(directory, process_name):
    direc = _j(directory, "process", process_name)
    return direc

def phase_directory(directory, process_name, phase):
    process_direc = process_directory(directory, process_name)
    direc = _j(process_direc, "phase", str(phase))
    return direc

