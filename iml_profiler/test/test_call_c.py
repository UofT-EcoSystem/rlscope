import argparse
import textwrap
from glob import glob
import codecs
import json
import time
import math
import os
import subprocess
import re
import sys
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e

from iml_profiler.profiler.profilers import Profiler
from iml_profiler.profiler import profilers

from iml_profiler.test import py_interface

from iml_profiler.profiler.profilers import tensorflow_profile_context
from iml_profiler.profiler.profilers import clib_wrap

from iml_profiler import py_config

def disable_test_timestamp():
    """
    Make sure pyprof and tfprof timestamps match.
    """
    tfprof_t = tensorflow_profile_context.now_in_usec()
    pyprof_t = clib_wrap.now_us()
    diff_us = math.fabs(tfprof_t - pyprof_t)
    margin_of_error_us = 10
    # Be generous; we expect tfprof/pyprof timestamps to be within 10 us of each other.
    # Typical values:
    # - 2.5, 4.0, 2.25, ...
    assert diff_us < margin_of_error_us

    print("> diff_us = {diff_us} usec".format(diff_us=diff_us))

class CallCTest:
    def __init__(self, args, parser):
        self.args = args
        self.parser = parser
        self.lib = py_interface.PythonInterface()

        self.profiler = Profiler(directory=self.directory,
                                 debug=self.debug,
                                 num_calls=args.num_calls,
                                 exit_early=False)

        self.time_slept_gpu_sec = 0.
        self.time_slept_cpp_sec = 0.
        self.time_run_python_sec = 0.

    @property
    def directory(self):
        return self.args.directory

    @property
    def debug(self):
        return self.args.debug

    def init(self):
        args = self.args
        self.lib.call_c()
        if self.debug and _e(self.args.gpu_clock_freq_json):
            self.load_json()
            self.lib.set_gpu_freq_mhz(self.gpu_mhz)
        else:
            self._gpu_mhz = self.lib.guess_gpu_freq_mhz()
            self.dump_json()
        print("GPU mhz = {mhz}".format(mhz=self.gpu_mhz))

    def dump_json(self):
        print("> Dump GPU clock frequency data to: {path}".format(path=self.args.gpu_clock_freq_json))
        self.gpu_clock_freq_data = {
            'gpu_mhz':self._gpu_mhz,
        }
        profilers.dump_json(self.gpu_clock_freq_data, self.args.gpu_clock_freq_json)

    def load_json(self):
        print("> Load GPU clock frequency from: {path}".format(path=self.args.gpu_clock_freq_json))
        self.gpu_clock_freq_data = profilers.load_json(self.args.gpu_clock_freq_json)

    @property
    def gpu_mhz(self):
        return self.gpu_clock_freq_data['gpu_mhz']

    def run_gpu(self):
        args = self.args
        if self.debug:
            print("> Running on GPU for {sec} seconds".format(sec=args.gpu_time_sec))
        self.time_slept_gpu_sec += self.lib.gpu_sleep(args.gpu_time_sec)

    def run_cpp(self):
        args = self.args
        if self.debug:
            print("> Running in CPP for {sec} seconds".format(sec=args.gpu_time_sec))
        self.time_slept_cpp_sec += self.lib.run_cpp(args.cpp_time_sec)

    def run_python(self):
        args = self.args
        if self.debug:
            print("> Running inside python for {sec} seconds".format(sec=args.python_time_sec))
        start_t = time.time()
        time.sleep(args.python_time_sec)

        # NOTE: This creates huge profiler output because of a lot of function calls...
        # instead just sleep.
        #
        # while True:
        #     end_t = time.time()
        #     total_sec = end_t - start_t
        #     if total_sec >= args.python_time_sec:
        #         break
        end_t = time.time()
        total_sec = end_t - start_t
        self.time_run_python_sec += total_sec

    def iteration(self):
        self.run_python()
        self.run_cpp()
        self.run_gpu()

    def run(self):
        args = self.args

        self.init()

        print(textwrap.dedent("""
        > Running {r} repetitions, {i} iterations, each iteration is:
            Run in python for {python_sec} seconds
            Run in C++ for {cpp_sec} seconds
            Run in GPU for {gpu_sec} seconds
        """.format(
            r=args.repetitions,
            i=args.iterations,
            python_sec=args.python_time_sec,
            cpp_sec=args.cpp_time_sec,
            gpu_sec=args.gpu_time_sec,
        )))

        # IML:
        self.profiler.profile(profilers.NO_BENCH_NAME, self.iteration)

        results_json = _j(self.directory, "test_call_c.json")
        print("> Dump test_call_c.py results @ {path}".format(path=results_json))
        results = {
            'time_gpu_sec':self.time_slept_gpu_sec,
            'time_python_sec':self.time_run_python_sec,
            'time_cpp_sec':self.time_slept_cpp_sec,
            'time_profile_sec':self.profiler.profile_time_sec(profilers.NO_BENCH_NAME),
        }
        profilers.dump_json(results, results_json)

# Default time period for Python/C++/GPU.
DEFAULT_TIME_SEC = 5
# DEFAULT_TIME_SEC_DEBUG = 1
DEFAULT_TIME_SEC_DEBUG = 5
def main():
    parser = argparse.ArgumentParser(textwrap.dedent("""
    Test profiling scripts to make sure we correctly measure time spent in Python/C++/GPU.
    """))
    parser.add_argument("--debug", action='store_true',
                        help=textwrap.dedent("""
                        Run quickly.
                        """))
    parser.add_argument("--debug-single-thread", action='store_true',
                        help=textwrap.dedent("""
                        Run with a single thread to make debugging easier.
                        """))
    parser.add_argument("--gpu-time-sec",
                        help=textwrap.dedent("""
                        Time to spend in GPU.
                        5 seconds (default)
                        """))
    parser.add_argument("--cpp-time-sec",
                        help=textwrap.dedent("""
                        Time to spend inside C++.
                        5 seconds (default)
                        """))
    parser.add_argument("--python-time-sec",
                        help=textwrap.dedent("""
                        Time to spend inside Python. 
                        5 seconds (default)
                        """))
    parser.add_argument("--directory",
                        help=textwrap.dedent("""
                        Where to store results.
                        """))
    parser.add_argument("--gpu-clock-freq-json",
                        help=textwrap.dedent("""
                        Internal use only.
                        """))
    parser.add_argument("--iterations",
                        type=int,
                        help=textwrap.dedent("""
                        --num-calls = --iterations * --repetitions
                        """),
                        default=3)
    parser.add_argument("--repetitions",
                        type=int,
                        help=textwrap.dedent("""
                        --num-calls = --iterations * --repetitions
                        """),
                        default=1)
    # IML:
    profilers.add_iml_arguments(parser)
    args = parser.parse_args()
    num_calls = args.iterations * args.repetitions

    if args.directory is None:
        args.directory = _j(py_config.ROOT, "checkpoints", "test_call_c")

    if args.gpu_clock_freq_json is None:
        args.gpu_clock_freq_json = _j(args.directory, "gpu_clock_freq.json")

    for attr in ['gpu_time_sec', 'cpp_time_sec', 'python_time_sec']:
        if args.debug:
            default_time_sec = DEFAULT_TIME_SEC_DEBUG
        else:
            default_time_sec = DEFAULT_TIME_SEC
        setattr(args, attr, default_time_sec)

    # IML:
    profilers.handle_iml_args(args.directory, parser, args, no_bench_name=True)

    args.num_calls = num_calls
    test_call_c = CallCTest(args, parser)
    test_call_c.run()

if __name__ == '__main__':
    main()
