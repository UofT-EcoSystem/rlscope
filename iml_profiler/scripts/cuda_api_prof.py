import logging
import shutil
import subprocess
import argparse
import textwrap
import sys
import os
import numpy as np

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler import py_config

from iml_profiler.parser.common import *

from iml_profiler.profiler import iml_logging

def main():
    iml_logging.setup_logging()

    iml_prof_argv, cmd_argv = gather_argv(sys.argv[1:])

    parser = argparse.ArgumentParser("Sample time spent in CUDA API calls, and call counts.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--iml-debug', action='store_true')
    # parser.add_argument('--iml-disable', action='store_true', help=textwrap.dedent("""
    #     IML: Skip any profiling. Used for uninstrumented runs.
    #     Useful for ensuring minimal libcupti registration when we run --cuda-api-calls during config_uninstrumented.
    #
    #     Effect: sets "export IML_DISABLE=1" for libsample_cuda_api.so.
    # """))
    parser.add_argument('--cuda-api-calls', action='store_true',
                        help=textwrap.dedent("""
                        Trace CUDA API runtime/driver calls.
                        
                        i.e. total number of calls, and total time (usec) spent in a given API call.
                        
                        Effect: sets "export IML_CUDA_API_CALLS=1" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--cuda-activities', action='store_true',
                        help=textwrap.dedent("""
                        Trace CUDA activities (i.e. GPU kernel runtimes, memcpy's).
                        
                        Effect: sets "export IML_CUDA_ACTIVITIES=yes" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--fuzz-cuda-api', action='store_true',
                        help=textwrap.dedent("""
                        Use libcupti to trace ALL CUDA runtime API calls (# of calls, and total time spent in them).
                        This is useful for determining which CUDA API's we need to "calibrate subtractions" for.
                        NOTE: this SHOULDN'T be used for finding profiling book-keeping "subtractions", since it 
                        adds a LOT of overhead to add start/end callbacks to all CUDA API functions.
                        
                        Effect: sets "export IML_FUZZ_CUDA_API=yes" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--cuda-api-events', action='store_true',
                        help=textwrap.dedent("""
                        Trace all the start/end timestamps of CUDA API calls.
                        Needed during instrumented runs so we know when to subtract profiling overheads.
                        
                        Effect: sets "export IML_CUDA_API_EVENTS=yes" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--pc-sampling', action='store_true',
                        help=textwrap.dedent("""
                        Perform sample-profiling using CUDA's "PC Sampling" API.
                        
                        Currently, we're just going to record GPUSamplingState.is_gpu_active.
                        
                        Effect: sets "export IML_PC_SAMPLING=1" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--trace-at-start', action='store_true',
                        help=textwrap.dedent("""
                        Start tracing right at application startup.
                        
                        Effect: sets "export IML_TRACE_AT_START=yes" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--stream-sampling', action='store_true',
                        help=textwrap.dedent("""
                        Poll cudaStreamQuery() to see if the GPU is being used.
                        
                        Effect: sets "export IML_STREAM_SAMPLING=yes" for libsample_cuda_api.so.
                        """))
    parser.add_argument('--config',
                        choices=['interception',
                                 'no-interception',
                                 'gpu-activities',
                                 'no-gpu-activities',
                                 'full',
                                 'uninstrumented',
                                 ],
                        help=textwrap.dedent("""
                        For measuring LD_PRELOAD CUDA API interception overhead:
                            interception:
                                Enable LD_PRELOAD CUDA API interception.
                                $ iml-prof --debug --cuda-api-calls --cuda-api-events --iml-disable
                            no-interception:
                                Disable LD_PRELOAD CUDA API interception.
                                $ iml-prof --debug --iml-disable
                                
                        For measuring CUPTI GPU activity gathering overhead on a per CUDA API call basis.
                            gpu-activities:
                                Enable CUPTI GPU activity recording.
                                $ iml-prof --debug --cuda-api-calls --cuda-activities --iml-disable
                            no-gpu-activities:
                                Disable CUPTI GPU activity recording.
                                $ iml-prof --debug --cuda-api-calls --iml-disable
                                
                        Expect (for the above configurations):
                        You should run train.py with these arguments set
                        
                            # We are comparing total training time across each configuration 
                            --iml-training-progress
                        
                            # Since we are comparing total training time, 
                            # run each configuration with the same number of training loop steps.
                            --iml-max-timesteps $N
                            
                            # Disable any pyprof or old tfprof tracing code.
                            --iml-disable
                                
                        For collecting full IML traces for using with iml-analyze / iml-drill:
                            full:
                                Enable all of tfprof and pyprof collection.
                                $ iml-prof --cuda-api-calls --cuda-api-events --cuda-activities --iml-disable
                                NOTE: we still use --iml-disable to prevent "old" tfprof collection.
                        """))
    args = parser.parse_args(iml_prof_argv)

    env = dict(os.environ)
    # TODO: figure out how to install pre-built .so file with "pip install iml_profiler"
    so_path = py_config.LIB_SAMPLE_CUDA_API
    if not _e(so_path):
        sys.stderr.write(textwrap.dedent("""
        IML ERROR: couldn't find CUDA sampling library @ {path}; to build it, do:
          $ cd {root}
          # Download library dependencies
          $ bash ./setup.sh
        
          # Perform cmake build
          $ mkdir build
          $ cd build
          # Assuming you installed protobuf 3.9.1 at --prefix=$HOME/protobuf
          $ cmake ..
          $ make -j$(nproc)
        """.format(
            root=py_config.ROOT,
            path=so_path,
        )))
        sys.exit(1)
    add_env = dict()
    add_env['LD_PRELOAD'] = "{ld}:{so_path}".format(
        ld=env.get('LD_PRELOAD', ''),
        so_path=so_path)

    if args.config is not None:
        add_env['IML_CONFIG'] = args.config
        if args.config == 'interception':
            "iml-prof --debug --cuda-api-calls --cuda-api-events"
            args.cuda_api_calls = True
            args.cuda_api_events = True
        elif args.config in ['no-interception', 'uninstrumented']:
            "iml-prof --debug"
            pass
        elif args.config == 'gpu-activities':
            "$ iml-prof --debug --cuda-api-calls --cuda-activities"
            args.cuda_api_calls = True
            args.cuda_activities = True
        elif args.config == 'no-gpu-activities':
            "$ iml-prof --debug --cuda-api-calls"
            args.cuda_api_calls = True
        elif args.config == 'full':
            "$ iml-prof --cuda-api-calls --cuda-api-events --cuda-activities"
            args.cuda_api_calls = True
            args.cuda_api_events = True
            args.cuda_activities = True
        else:
            raise NotImplementedError()

    if args.fuzz_cuda_api and args.cuda_api_calls:
        parser.error("Can only run iml-prof with --fuzz-cuda-api or --cuda-api-calls, not both")

    if args.debug or args.iml_debug or is_env_true('IML_DEBUG'):
        logging.info("Detected debug mode; enabling C++ logging statements (export IML_CPP_MIN_VLOG_LEVEL=1)")
        add_env['IML_CPP_MIN_VLOG_LEVEL'] = 1

    # if args.iml_disable:
    #     add_env['IML_DISABLE'] = 'yes'

    if args.cuda_api_calls:
        add_env['IML_CUDA_API_CALLS'] = 'yes'

    if args.cuda_activities:
        add_env['IML_CUDA_ACTIVITIES'] = 'yes'

    if args.pc_sampling:
        add_env['IML_PC_SAMPLING'] = 'yes'

    if args.fuzz_cuda_api:
        add_env['IML_FUZZ_CUDA_API'] = 'yes'

    if args.cuda_api_events:
        add_env['IML_CUDA_API_EVENTS'] = 'yes'

    if args.trace_at_start:
        add_env['IML_TRACE_AT_START'] = 'yes'

    if args.stream_sampling:
        add_env['IML_STREAM_SAMPLING'] = 'yes'

    exe_path = shutil.which(cmd_argv[0])
    if exe_path is None:
        print("IML ERROR: couldn't locate {exe} on $PATH; try giving a full path to {exe} perhaps?".format(
            exe=cmd_argv[0],
        ))
        sys.exit(1)
    # cmd = argv
    cmd = [exe_path] + cmd_argv[1:]
    print_cmd(cmd, env=add_env)

    env.update(add_env)
    for k in list(env.keys()):
        env[k] = str(env[k])

    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(exe_path, cmd, env)
    # Shouldn't return.
    assert False

def gather_argv(argv):
    """

    $ iml-prof [options]         cmd_exec ...
               ---------         ------------
               iml_prof_argv     cmd_argv

    Split sys.argv into:
    - iml_prof_argv: Arguments that iml-prof should handle.
    - cmd_argv: Arguments that the profiled script should handle.

    :param argv:
        sys.argv
    :return:
    """
    iml_prof_argv = []
    i = 0
    def is_executable(opt):
        return shutil.which(opt) is not None
    has_dashes = any(opt == '--' for opt in argv)
    while i < len(argv):

        if has_dashes:
            if argv[i] == '--':
                i += 1
                break
        elif is_executable(argv[i]):
            break

        iml_prof_argv.append(argv[i])
        i += 1
    cmd_argv = argv[i:]
    return iml_prof_argv, cmd_argv

def is_env_true(var, env=None):
    if env is None:
        env = os.environ
    return env.get(var, 'no').lower() not in {'no', 'false', '0', 'None', 'null'}

if __name__ == '__main__':
    main()
