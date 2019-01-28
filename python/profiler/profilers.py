import cProfile, pstats, io
import codecs
import sys
import json
import argparse
import pprint
import subprocess
import textwrap
import os
import time
import re
from glob import glob
import math
import numpy as np
import contextlib

import tensorflow as tf
from tensorflow.python.client import device_lib as tf_device_lib
from tensorflow.python.profiler import profile_context
from tensorflow.python.framework import c_api_util
from tensorflow.python.client import session

# from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto

from parser.db import is_tfprof_file, is_pyprof_file, is_config_file

# pip install py-cpuinfo
import cpuinfo

from os import environ as ENV

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from parser.common import *
from profiler import cudaprofile
from profiler import clib_wrap
from profiler.clib_wrap import MICROSECONDS_IN_SECOND
from profiler import tensorflow_profile_context

import py_config

DEBUG = tensorflow_profile_context.DEBUG

# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

TF_PRINT_TIMESTAMP = ENV.get('TF_PRINT_TIMESTAMP', 'no') == 'yes'

NUM_STEPS_TO_TRACE = 100

_TF_MODIFIED = False
def modify_tensorflow():
    # NOTE: profiling appears to take a really long time when we do this...
    pass

    global _TF_MODIFIED
    if _TF_MODIFIED:
        return
    """
    Usually TensorFlow only measures 100 steps at most.
    Set a big upper limit so it will measure each iteration we measure.
    """

    tensorflow_profile_context.MAX_TRACED_STEPS = 99999999

    setup_wrap_BaseSession_as_default()

    # from tensorflow.python.profiler import profile_context
    # profile_context.MAX_TRACED_STEPS = 99999999

    _TF_MODIFIED = True

# All currently active Profiler objects (there should really only be one).
# Used for hooking into sess.as_default()
PROFILERS = []

"""
Wrap sess.as_default().
"""
old_BaseSession_as_default = None
def setup_wrap_BaseSession_as_default():
    global old_BaseSession_as_default
    old_BaseSession_as_default = getattr(session.BaseSession, 'as_default')
    setattr(session.BaseSession, 'as_default', wrap_BaseSession_as_default)
def wrap_BaseSession_as_default(self):
    if DEBUG:
        print_stacktrace("> wrapped sess.as_default()")
    for prof in PROFILERS:
        prof.set_session(self)
    return old_BaseSession_as_default(self)

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
                 num_traces=None,
                 tfprof=True,
                 c_lib_func_pyprof_pattern=None,
                 # tfprof=True,
                 repetition_time_limit_sec=10.,
                 debug=None,
                 exit_early=True,
                 require_end_operation=False,
                 disable=None,
                 args=None):
        modify_tensorflow()

        """
        If set, require the user to call prof.end_operation
        (don't allow a call to prof.set_operation to also count as a call to prof.end_operation)
        """
        self.require_end_operation = require_end_operation

        def get_iml_argname(argname):
            name = argname
            # name = re.sub('_', '-', name)
            name = "iml_{name}".format(name=name)
            return name

        def get_argval(argname, klass_arg, default_arg, allow_none=True):
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

            iml_argname = get_iml_argname(argname)

            if hasattr(args, iml_argname) and getattr(args, iml_argname) is not None:
                argval = getattr(args, iml_argname)
                return argval

            if not allow_none and default_arg is None:
                raise RuntimeError("IML: you must provide a value for --{arg}".format(
                    arg=re.sub('_', '-', iml_argname)))

            return default_arg

        self.pctx = None
        self.next_trace_id = None
        self.process_name = None
        self.phase = 0
        self.next_trace_id = None
        # self.init_trace_id()
        self._tfprof_enabled = False
        self._pyprof_enabled = False
        self.cur_bench_name = NO_BENCH_NAME
        self.total_profile_time_sec = 0
        self.directory = get_argval('directory', directory, None, allow_none=False)
        self.disable = get_argval('disable', disable, False)
        self.exit_early = exit_early
        self.tfprof = tfprof
        self.c_lib_func_pyprof_pattern = c_lib_func_pyprof_pattern
        self.repetition_time_limit_sec = repetition_time_limit_sec
        self.num_calls = get_argval('num_calls', num_calls, None)
        self.num_traces = get_argval('num_traces', num_traces, None)
        self.bench_name = get_argval('bench_name', bench_name, None)
        self.start_measuring_call = get_argval('start_measuring_call', start_measuring_call, None)
        self.debug = get_argval('debug', debug, False)
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
        self.step_start_profiling = None
        self.step_end_profiling = None
        self.average_time_per_call_sec = None
        self.average_time_per_call_no_profile_sec = None

        self.sess = None

        # Total times collected from running profiled operations.
        self.profile_sec = []
        # Total times collected from running un-profiled operations.
        self.no_profile_sec = []

        # clib_wrap.wrap_libs()

        # assert ( self.num_calls is None and self.start_measuring_call is None ) or \
        #        ( self.num_calls is not None and self.start_measuring_call is not None )
        # assert self.start_measuring_call is not None

    def init_trace_id(self):
        if self.process_name is None:
            return
        self.next_trace_id = 0
        # NOTE: We DON'T want to keep existing trace files across runs...
        # we really should add code for deleting existing trace files...
        # for dirpath, dirnames, filenames in os.walk(self.out_dir):
        #     for base in filenames:
        #         path = _j(dirpath, base)
        #         m = is_pyprof_file(path)
        #         if m:
        #             trace_id = int(m.group('trace_id'))
        #             self.next_trace_id = max(self.next_trace_id, trace_id + 1)
        for dirpath, dirnames, filenames in os.walk(self.out_dir):
            for base in filenames:
                path = _j(dirpath, base)
                # print("> Consider {path}".format(path=path))
                if is_pyprof_file(path) or is_tfprof_file(path) or is_config_file(path):
                    print("> RM {path}".format(path=path))
                    os.remove(path)
                    # trace_id = int(m.group('trace_id'))
                    # self.next_trace_id = max(self.next_trace_id, trace_id + 1)
        if DEBUG:
            print("> Using next_trace_id = {id}".format(id=self.next_trace_id))

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
            print("> Dynamic iterations for bench_name={b} decided on: {it} iterations".format(
                it=iterations,
                b=bench_name))
            print("  1 repetition takes ~ {sec} seconds".format(sec=total_time_sec))

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
        assert self.process_name is not None
        assert self.phase is not None
        direc = phase_directory(self.directory, self.process_name, self.phase)
        os.makedirs(direc, exist_ok=True)
        return direc

    @property
    def pyprof_proto_path(self):
        ret = _j(self.out_dir, "pyprof{bench}{trace}.proto".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(self.next_trace_id),
        ))
        return ret

    @property
    def config_path(self):
        config_path = _j(self.out_dir, "config{bench}{trace}.json".format(
            bench=bench_suffix(self.bench_name),
            trace=trace_suffix(self.next_trace_id),
        ))
        return config_path

    @property
    def tfprof_path(self):
        tfprof_path = _j(
            self.out_dir,
            "profile{bench}{trace}.proto".format(
                bench=bench_suffix(self.bench_name),
                trace=trace_suffix(self.next_trace_id),
            ))
        return tfprof_path

    def _start(self):
        if not self.tfprof:
            if self.debug:
                print("    > Start CUDA profiler")
            self.cuda_profiler.start()

    def _end(self):
        if not self.tfprof:
            self.cuda_profiler.stop()
            if self.debug:
                print("> Stop CUDA profiler")

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
        self._start()
        return True

    def profile_time_sec(self, bench_name):
        return self.time_sec[bench_name]

    def disable_profiling(self, bench_name=NO_BENCH_NAME, num_calls=1):
        if not self._should_measure_call(bench_name):
            self.code_count[bench_name] = self.code_count.get(bench_name, 0) + 1
            return

        self._end()
        end_time_sec = time.time()
        self.end_t[bench_name] = end_time_sec
        self.time_sec[bench_name] = self.time_sec.get(bench_name, 0.) + end_time_sec - self.start_t[bench_name]
        self.code_count[bench_name] = self.code_count.get(bench_name, 0) + num_calls

    def set_session(self, sess):
        if DEBUG:
            print_stacktrace("> set_session = {sess}".format(sess=sess))

        assert sess is not None

        if self.sess is not None:
            raise NotImplementedError("Haven't implemented profiling using multiple session objects.")
        self.sess = sess

        if self.disable:
            return

        if self.cur_bench_name is not None and not self._tfprof_enabled:
            self._start_tfprof(bench_name=self.cur_bench_name, allow_skip=False)

        if self.pctx is not None:
            self.pctx.set_session(self.sess)

    def _cur_session(self, allow_none=False):
        if self.sess is not None:
            return self.sess

        sess = tf.get_default_session()
        if sess is None and not allow_none:
            raise RuntimeError(
                "Couldn't find current session; you either need to call Profiler.set_session(sess), "
                "or do \"with sess.as_default():\"")

        return sess

    def start(self):
        PROFILERS.append(self)

        if self.disable:
            return

        self._start_tfprof(allow_skip=True)

    def _maybe_end_operation(self):
        if self.cur_bench_name != NO_BENCH_NAME:
            self.end_operation(self.cur_bench_name)
        assert self.cur_bench_name == NO_BENCH_NAME

    def stop(self):
        PROFILERS.remove(self)

        if self.disable:
            return

        self._maybe_end_operation()
        self._maybe_finish(finish_now=True, skip_finish=False)
        # Execution shouldn't reach here.
        assert False

    def _start_tfprof(self, bench_name=NO_BENCH_NAME, allow_skip=False):
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

        sess = self._cur_session(allow_none=True)
        if sess is None:
            if allow_skip:
                # They called set_operation before calling set_session/sess.as_default().
                # Delay preallocation of tracer until session is set.
                return
            raise RuntimeError(
                "Couldn't find current session; you either need to call Profiler.set_session(sess), "
                "or do \"with sess.as_default():\"")

        self.pctx = tensorflow_profile_context.ProfileContext(self.out_dir, dump_on_finished=True,
                                                              # Need to explicitly use empty trace steps otherwise profiler
                                                              # "auto decides" which steps to trace.
                                                              trace_steps=[],
                                                              process_name=self.process_name,
                                                              phase=self.phase,
                                                              trace_all=True)
        # NOTE: this needs to be called before each session.run(...) call....
        # with _tracing_disabled(prof=self):
        #     if py_config.CUSTOM_TF:
        #         tensorflow_profile_context.preallocate_tracer(self._tfprof_step, sess)
        self.pctx.__enter__()
        self._tfprof_enabled = True

    def _stop_tfprof(self, bench_name=NO_BENCH_NAME):
        """
        Stop profiling:
        - Collect trace data, and dump it to file(s)

        :return:
        """
        if not self._tfprof_enabled:
            return
        # NOTE: this is when we collect the trace data... keep that in mind when we implement DUMP.
        if self.sess is not None:
            self.pctx.set_session(self.sess)
        self.pctx.__exit__(None, None, None)
        self._tfprof_enabled = False

    def _check_profiling_started(self):
        global PROFILERS
        started = self in PROFILERS
        if not started:
            raise RuntimeError("IML: You need to call profiler.start() before profiling.")

    def set_operation(self, bench_name):
        if self.disable:
            return

        self._check_profiling_started()

        if self.cur_bench_name != NO_BENCH_NAME:
            """
            Allow this usage (i.e. leaving out the end_operation calls):
            profiler.set_operation('op1')
            ...
            profiler.set_operation('op2')
            ...
            profiler.set_operation('op3')
            
            """
            self.end_operation(self.cur_bench_name)

        if not(
            self._should_measure_call(bench_name)
        ):
            return

        self.cur_bench_name = bench_name
        # Q: If we don't have a session.run() call, will this result in a bug?
        # self.pctx.trace_next_step()

        if DEBUG:
            print("> set_operation(op={op})".format(op=bench_name))

        self._start_tfprof(bench_name, allow_skip=True)
        self._start_pyprof(bench_name)

    def _start_pyprof(self, bench_name):
        # if self._pyprof_enabled:
        #     return
        # assert not self._pyprof_enabled
        clib_wrap.wrap_libs()
        self.step_start_profiling = self.steps
        clib_wrap.enable_tracing()
        self.enable_profiling(bench_name)
        clib_wrap.set_step(self._pyprof_step, expect_traced=True)
        if (tensorflow_profile_context.DEBUG or TF_PRINT_TIMESTAMP) and clib_wrap.is_recording():
            print("> RECORDING pyprof_step = {step}".format(step=self._pyprof_step))
        self.start_call_us = clib_wrap.now_us()
        self._pyprof_enabled = True

    def _stop_pyprof(self, bench_name):
        assert self._pyprof_enabled
        self.end_call_us = clib_wrap.now_us()
        self.disable_profiling(bench_name, num_calls=1)
        # Record the last amount of time in between returning
        # from a call to q_forward, and finishing benchmarking.
        # This will include time spent in the tensorflow python API
        clib_wrap.record_python_event('Finish python benchmark', self.end_call_us)
        time_sec = (self.end_call_us - self.start_call_us)/MICROSECONDS_IN_SECOND
        if clib_wrap.is_recording():
            self.profile_sec.append(time_sec)
        else:
            self.no_profile_sec.append(time_sec)
        clib_wrap.record_event(CATEGORY_DUMMY_EVENT, 'Start call', self.start_call_us, self.start_call_us + 1)
        clib_wrap.record_event(CATEGORY_DUMMY_EVENT, 'End call', self.end_call_us, self.end_call_us + 1)
        clib_wrap.record_operation(self.start_call_us, self.end_call_us,
                                   op_name=bench_name)
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

    @property
    def _tfprof_step(self):
        """
        Internally, tfprof profiler (tensorflow_profile_context.ProfileContext) increments a step counter
        every time session.run() is called.

        Keep pyprof step number in sync by returning it directly.

        :return:
        """
        assert self.pctx is not None
        return self.pctx._step

    def end_operation(self, bench_name=NO_BENCH_NAME):
        if self.disable:
            return

        if bench_name != NO_BENCH_NAME and self.cur_bench_name != bench_name:
            raise RuntimeError(textwrap.dedent("""
            Detected nested profiling statements:
                prof.set_operation({b1})
                prof.set_operation({b2})
            """.format(
                b1=self.cur_bench_name,
                b2=bench_name,
            )))

        if self.cur_bench_name == NO_BENCH_NAME and bench_name != self.cur_bench_name:
            """
            start_operation was called, but was skipped since _should_measure_call 
            returned false.
            """
            return

        if bench_name == NO_BENCH_NAME:
            assert self.cur_bench_name != NO_BENCH_NAME
            bench_name = self.cur_bench_name

        if DEBUG:
            print("> end_operation(op={op})".format(op=bench_name))

        self._stop_pyprof(bench_name)
        self._stop_tfprof(bench_name)

        self.cur_bench_name = NO_BENCH_NAME

    def profile(self, bench_name, func, *args, **kwargs):
        """
        Useful for quickly profiling a single portion of the training loop by running the operation repeatedly.
        Assumes the func is idempotent.

        PSEUDOCODE:

        if idempotent:
          # func(...) is idempotent, so just run all the iterations
          # at once to reduce total runtime.

          for i in range(warmup):
            func(...)
          start_t = time.time()
          profiler.start()
          for i in range(iterations):
            func(...)
          profiler.stop()

        else:
          # func(...) is not idempotent, so we must collect iterations
          # one-at-a-time.

          func(...)

        :param self:
        :param func:
        :param args:
        :param kwargs:
        :return:
        """
        should_measure = self._should_measure_call(bench_name)

        raise NotImplementedError

        if should_measure:
            # idempotent.
            # for i in range(self.start_measuring_call):
            #     func(*args, **kwargs)

            if hasattr(func, 'init'):
                # Do any kind of initialization needed before profiling
                func.init(*args, **kwargs)

            # if self.num_calls is None:
            #     # Dynamically decide # of iterations to run, such that time to
            #     # run bench_name experiment is <= 10 seconds.
            #     self._init_num_calls(bench_name, func, *args, **kwargs)
            #     assert self.num_calls is not None

            # with tf.contrib.tfprof.ProfileContext(self.out_dir) as pctx:
            if self.tfprof:
                self._start_tfprof(bench_name, allow_skip=False)

            self.profile_sec = []
            self.no_profile_sec = []

            for i in range(self.num_calls):
                # NOTE: pyprof's step counter for deciding whether to trace the current step is is 0-based.
                # Offsetting this by +1 will cause pyprof data from 1 iteration prior to be shown with tfprof
                # from 1 iteration later.
                # (We had this bug before...)
                self.set_operation(bench_name)
                ret = func(*args, **kwargs)
                self.end_operation(bench_name)
                self.next_step(bench_name)

            if len(self.profile_sec) > 1:
                self.average_time_per_call_sec = np.mean(self.profile_sec[1:])
            if len(self.no_profile_sec) > 1:
                self.average_time_per_call_no_profile_sec = np.mean(self.no_profile_sec[1:])

            if self.tfprof:
                self._stop_tfprof(bench_name)

            if hasattr(func, 'reset'):
                # Cleanup anything we did specific to profiling so we can resume
                # running the training loop.
                func.reset(*args, **kwargs)

            self.dump_trace()
            # We shouldn't return from maybe_finish for idempotent operations.
            assert False

        else:
            # Not idempotent.
            if should_measure:
                self.set_operation(bench_name)
            ret = func(*args, **kwargs)
            if should_measure:
                self.end_operation(bench_name)

        return ret

    def set_process_name(self, process_name):
        self.process_name = process_name
        self.init_trace_id()

    def set_phase(self, phase):
        assert type(phase) == int
        self.phase = phase
        self.init_trace_id()

    def finish(self):
        print("> IML: Stopping training early now that profiler is done")
        sys.exit(0)

    def dump_trace(self):
        # We shouldn't be in the middle of measuring an operation.
        start_us = now_us()
        assert self.cur_bench_name == NO_BENCH_NAME

        assert not self._tfprof_enabled
        assert not self._pyprof_enabled
        # self._stop_tfprof()

        # ProfileGlobals.files_after = ls_files(self.out_dir)

        # Put this here to test that cleanup_files doesn't delete nvprof/pyprof files
        self._dump()
        # ProfileGlobals.cleanup_files()

        # Discards profiling data now that it has been recorded.
        self._discard_profiling_data()
        end_us = now_us()
        clib_wrap.record_event(CATEGORY_PROFILING, PROFILING_DUMP_TRACE, start_us, end_us)

    def _discard_profiling_data(self):
        clib_wrap.clear_pyprof_profiling()
        self.pctx = None

    def _dump(self, config_kwargs=dict()):
        """
        Dump trace data to:
        - pyprof.trace_<next_trace_id>.proto
        - profile.trace_<next_trace_id>.proto
        - config.trace_<next_trace_id>.proto
        """

        # Q: Should we be calling this again...?  We'd like to update num_calls if it was computed dynamically...
        config_path = self.config_path
        if self.c_lib_func_pyprof_pattern is not None and \
                'c_lib_func_pyprof_pattern' not in config_kwargs:
            config_kwargs['c_lib_func_pyprof_pattern'] = self.c_lib_func_pyprof_pattern
        dump_config(config_path,
                    num_calls=self.num_calls,
                    start_measuring_call=self.start_measuring_call,
                    profile_sec=self.profile_sec,
                    no_profile_sec=self.no_profile_sec,
                    average_time_per_call_sec=self.average_time_per_call_sec,
                    average_time_per_call_no_profile_sec=self.average_time_per_call_no_profile_sec,
                    process_name=self.process_name,
                    **config_kwargs)

        if not self.tfprof:
            self.cuda_profiler.dump()
        clib_wrap.dump_pyprof(self.pyprof_proto_path, self.process_name, self.phase)

        if self.tfprof:
            # Rename: profile_100 -> profile_100.q_forward.proto
            tfprof_protos = [path for path in glob("{dir}/profile_*".format(dir=self.out_dir))
                             if re.search(r'^profile_\d+$', _b(path))]
            if len(tfprof_protos) > 1:
                pprint.pprint({'tf_protos':tfprof_protos})
            assert len(tfprof_protos) <= 1
            if len(tfprof_protos) > 0:
                # If the sub-operation doesn't call sess.run(...), a profile_100 file won't be created.
                tf_proto = tfprof_protos[0]
                tf_proto_dir = _d(tf_proto)

                new_tf_proto = self.tfprof_path
                os.rename(tf_proto, new_tf_proto)
                # self._fixup_tfprof(new_tf_proto)
            else:
                print(("> WARNING: bench_name={bench} did not run session.run(...), "
                       "so no tfprof output was generated for it").format(bench=self.bench_name))

        self.next_trace_id += 1

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

    def should_dump_trace(self, finish_now=False):
        return finish_now or ( self.steps >= self.step_start_profiling + self.num_calls )

    def should_finish(self, finish_now=False, skip_finish=False):
        return finish_now or ( not skip_finish and self.next_trace_id >= self.num_traces )

    def _maybe_finish(self, finish_now=False, skip_finish=False):
        dump_trace = self.should_dump_trace(finish_now)
        if dump_trace:
            self.dump_trace()

        if self.should_finish(finish_now, skip_finish):
            self.finish()

        if dump_trace:
            # We just dumped a trace but we're not finished;
            # start profiling again for the next trace.
            self._start_tfprof(allow_skip=True)

    def next_step(self, skip_finish=False):
        """
        If we've recorded enough samples, dump trace files.

        If we've dumped enough trace files, exit (we're done).
        """
        # Not implemented correctly, only supports idempotent operations right now...
        # self._maybe_finish()

        # Q: Should we set cur_bench_name to NO_BENCH_NAME?
        # Currently, clib_wrap._step is the previous step.
        # set_operation assumes self.steps has a certain value,
        # so end_operation ought to be able to assume self.steps doesn't change.
        # So, yes.
        self._maybe_end_operation()
        self._maybe_finish(finish_now=False,
                           skip_finish=skip_finish)
        self.steps += 1

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

class PythonProfiler:
    """
    Run profiler on python code block.

    with profiler(python_profile_basename="some/profile"):
            # profile this code
    """
    def __init__(self, directory, bench_name=NO_BENCH_NAME, record_call_times=True, clock='monotonic_clock'):
        self.profile = cProfile.Profile()
        self.directory = directory
        self.bench_name = bench_name
        self.clock = clock
        assert self.clock in ['monotonic_clock']
        self.use_cycle_counter = (self.clock == 'cycle_counter')
        self.use_monotonic_clock = (self.clock == 'monotonic_clock')
        self.record_call_times = record_call_times

        assert not ( self.use_cycle_counter and self.use_monotonic_clock )

        if self.use_cycle_counter:
            self.profile.make_use_cycle_counter()

        if self.use_monotonic_clock:
            self.profile.make_use_monotonic_clock()

        if self.record_call_times:
            self.profile.make_record_call_times()

    def __enter__(self):
        self.start()

    def start(self):
        self.profile.enable()

    def stop(self):
        self.profile.disable()
        # self.dump()

    def dump(self):
        # sortby = ('calls', 'filename', 'name', 'line')
        sortby = ('tottime', 'filename', 'line', 'name')

        os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)
        with open(self._stats_path, mode='w') as f:
            ps = pstats.Stats(self.profile, stream=f).sort_stats(*sortby)
            if self.record_call_times:
                call_times = ps.call_times
            ps.print_stats()

        if self.record_call_times:

            # Tuple keys are not OK; convert to strings.
            new_call_times = dict()
            for func_tuple, times in call_times.items():
                func = func_std_string(func_tuple)
                new_call_times[func] = times
            json.dump(new_call_times,
                                codecs.open(self._call_times_path, mode='w', encoding='utf-8'),
                                sort_keys=True, indent=4)

        os.makedirs(os.path.dirname(self._prof_path), exist_ok=True)
        ps.dump_stats(self._prof_path)

    @property
    def _prof_path(self):
        ret = _j(self.directory, "python_profile{bench}.prof".format(
            bench=bench_suffix(self.bench_name)))
        return ret

    @property
    def _call_times_path(self):
        return _j(self.directory, "python_profile{bench}.call_times.json".format(
            bench=bench_suffix(self.bench_name)))

    @property
    def _stats_path(self):
        return _j(self.directory, "python_profile{bench}.txt".format(
            bench=bench_suffix(self.bench_name)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Taken from cProfile's Lib/pstats.py;
# func_name is 3-tuple of (file-path, line#, function_name)
# e.g.
#     ('Lib/test/my_test_profile.py', 259, '__getattr__'):
def func_std_string(func_tuple): # match what old profile produced
    if func_tuple[:2] == ('~', 0):
        # special case for built-in functions
        name = func_tuple[2]
        if name.startswith('<') and name.endswith('>'):
            return '{%s}' % name[1:-1]
        else:
            return name
    else:
        path, lineno, func_name = func_tuple
        new_path = path
        # Full path is useful for vim when visiting lines; keep it.
        # new_path = re.sub(r'.*/site-packages/', '', new_path)
        # new_path = re.sub(r'.*/clone/', '', new_path)
        # new_path = re.sub(r'.*/lib/python[^/]*/', '', new_path)
        return "{path}:{lineno}({func_name})".format(
            path=new_path,
            lineno=lineno,
            func_name=func_name,
        )

def add_iml_arguments(parser):
    parser.add_argument('--iml-nvprof-enabled', action='store_true', help=textwrap.dedent("""
        IML: is nvprof running?
        
        Internal use only; 
        used to determine whether this python script has been invoked using nvprof.
        If it hasn't, the script will re-invoke itself with nvprof.
    """))
    # parser.add_argument('--iml-tfprof', action='store_true', help=textwrap.dedent("""
    #     IML: use tfprof TensorFlow profiling utility INSTEAD of nvprof.
    # """))
    parser.add_argument('--iml-num-calls', type=int, default=1000,
                        help="IML: how many calls should be measured in a single trace?")
    parser.add_argument('--iml-num-traces', type=int, default=10,
                        help="IML: how many traces should be measured?")
    parser.add_argument('--iml-fuzz', action='store_true', help=textwrap.dedent("""
        IML: \"Fuzz\" the script for calls to TensorFlow API's.
        
        Useful if you have no idea where the training-loop of an ML script is located. 
        
        Adds breakpoints / dumps stack traces when certain TensorFlow API's are called; 
        for e.g. sesssion.run(...) for running the computational graph
        (currently this is the only thing we trace).
    """))
    parser.add_argument('--iml-disable', action='store_true', help=textwrap.dedent("""
        IML: Skip any profiling.
    """))
    parser.add_argument('--iml-debug', action='store_true', help=textwrap.dedent("""
        IML: debug profiler.
    """))
    parser.add_argument('--iml-start-measuring-call', default=100, type=int,
                        help="IML: when should measuring begin?")
    parser.add_argument('--iml-bench-name',
                        default=NO_BENCH_NAME,
                        help=textwrap.dedent("""
    IML: which code block should we measure?
    i.e. --iml-bench-name=some_bench
        # Just measure "some_bench", nothing else.
        profiler.profile('some_bench', do_some_bench)
    """))
    parser.add_argument('--iml-directory',
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
#             print("> RM {opts}{f}".format(
#                 opts=opts, f=path))

# ProfileGlobals = _ProfileGlobals()

def handle_iml_args(output_directory, parser, args, no_bench_name=False):
    pass
    # ProfileGlobals.files_before = ls_files(output_directory)

def iml_argv():
    """
    Return a list of string arguments related to IML that were passed to the current running python process.

    Useful for forwarding IML arguments to python child processes instrumented with IML.
    """
    # JAMES TODO: forward set_phase to children.
    parser = argparse.ArgumentParser()
    add_iml_arguments(parser)
    args, extra_argv = parser.parse_known_args(sys.argv)
    argv = args_to_cmdline(parser, args, keep_executable=False, keep_debug=False)
    return argv

def run_with_nvprof(directory, parser, args,
                    bench_name=NO_BENCH_NAME):
    print("> Reinvoking script with nvprof; bench_name={b}".format(
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

    py_script_idx = 0
    while not re.search(r'\.py$', sys.argv[py_script_idx]):
        py_script_idx += 1
    extra_opts = []
    if keep_debug and hasattr(args, 'debug') and args.debug:
        extra_opts.extend(["-m", "ipdb"])
    cmdline = []
    if keep_executable:
        cmdline.append(sys.executable)
    cmdline.extend(extra_opts)
    if keep_executable:
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

    avail_gpus = get_available_gpus()
    avail_cpus = get_available_cpus()
    # We want to be CERTAIN about which device TensorFlow is using.
    # If no GPUs are available, TF will use the CPU.
    # If a GPU is available, make sure only 1 is available so we are certain it's using that one.
    if not( (len(avail_gpus) == 1) or
            (len(avail_gpus) == 0 and len(avail_cpus) == 1) ):
        CUDA_VISIBLE_DEVICES = ENV.get('CUDA_VISIBLE_DEVICES', None)
        print(textwrap.dedent("""
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
    print(textwrap.dedent("""
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

def bench_suffix(bench):
    if bench != NO_BENCH_NAME:
        return ".{bench}".format(bench=bench)
    return ""

def trace_suffix(trace_id, allow_none=False):
    if trace_id is None and not allow_none:
        raise RuntimeError("trace_id must be >= 0, got None")

    if trace_id is not None:
        return ".trace_{id}".format(id=trace_id)
    return ""

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

def get_available_cpus():
    local_device_protos = tf_device_lib.list_local_devices()
    device_protos = [x for x in local_device_protos if x.device_type != 'GPU']
    assert len(device_protos) == 1
    cpu = cpuinfo.get_cpu_info()
    # 'brand': 'Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz',
    device_dict = {
        'device_name':cpu['brand'],
        'device_number':0,
    }
    return [device_dict]
    # device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    # return device_dicts

def get_available_gpus():
    local_device_protos = tf_device_lib.list_local_devices()
    device_protos = [x for x in local_device_protos if x.device_type == 'GPU']
    device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    return device_dicts

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

