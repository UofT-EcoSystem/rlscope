from iml_profiler.profiler.iml_logging import logger
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
from iml_profiler.profiler.util import gather_argv, print_cmd

from iml_profiler.clib import rlscope_api
from iml_profiler.profiler.iml_logging import logger

DEFAULT_CONFIG = 'full'

def main():

    iml_prof_argv, cmd_argv = gather_argv(sys.argv[1:])

    parser = argparse.ArgumentParser("Sample time spent in CUDA API calls, and call counts.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--iml-debug', action='store_true')
    parser.add_argument('--iml-rm-traces-from', help=textwrap.dedent("""
    Delete traces rooted at this --iml-directory. 
    Useful if your training script has multiple training scripts, and you need to use --iml-skip-rm-traces 
    when launching the other scripts.
    """))
    # parser.add_argument('--iml-disable', action='store_true', help=textwrap.dedent("""
    #     IML: Skip any profiling. Used for uninstrumented runs.
    #     Useful for ensuring minimal libcupti registration when we run --cuda-api-calls during config_uninstrumented.
    #
    #     Effect: sets "export IML_DISABLE=1" for librlscope.so.
    # """))

    add_bool_arg(parser, '--cuda-api-calls',
                        help=textwrap.dedent("""
                        Trace CUDA API runtime/driver calls.
                        
                        i.e. total number of calls, and total time (usec) spent in a given API call.
                        
                        Effect: sets "export IML_CUDA_API_CALLS=1" for librlscope.so.
                        """))
    add_bool_arg(parser, '--cuda-activities',
                        help=textwrap.dedent("""
                        Trace CUDA activities (i.e. GPU kernel runtimes, memcpy's).
                        
                        Effect: sets "export IML_CUDA_ACTIVITIES=yes" for librlscope.so.
                        """))
    add_bool_arg(parser, '--cuda-api-events',
                        help=textwrap.dedent("""
                        Trace all the start/end timestamps of CUDA API calls.
                        Needed during instrumented runs so we know when to subtract profiling overheads.
                        
                        Effect: sets "export IML_CUDA_API_EVENTS=yes" for librlscope.so.
                        """))
    add_bool_arg(parser, '--gpu-hw',
                 help=textwrap.dedent("""
                        Collect GPU hardware counters.
                        
                        Effect: sets "export IML_GPU_HW=yes" for librlscope.so.
                        """))

    parser.add_argument('--fuzz-cuda-api', action='store_true',
                        help=textwrap.dedent("""
                        Use libcupti to trace ALL CUDA runtime API calls (# of calls, and total time spent in them).
                        This is useful for determining which CUDA API's we need to "calibrate subtractions" for.
                        NOTE: this SHOULDN'T be used for finding profiling book-keeping "subtractions", since it 
                        adds a LOT of overhead to add start/end callbacks to all CUDA API functions.
                        
                        Effect: sets "export IML_FUZZ_CUDA_API=yes" for librlscope.so.
                        """))

    parser.add_argument('--pc-sampling', action='store_true',
                        help=textwrap.dedent("""
                        Perform sample-profiling using CUDA's "PC Sampling" API.
                        
                        Currently, we're just going to record GPUSamplingState.is_gpu_active.
                        
                        Effect: sets "export IML_PC_SAMPLING=1" for librlscope.so.
                        """))
    parser.add_argument('--trace-at-start', action='store_true',
                        help=textwrap.dedent("""
                        Start tracing right at application startup.
                        
                        Effect: sets "export IML_TRACE_AT_START=yes" for librlscope.so.
                        """))
    parser.add_argument('--stream-sampling', action='store_true',
                        help=textwrap.dedent("""
                        Poll cudaStreamQuery() to see if the GPU is being used.
                        
                        Effect: sets "export IML_STREAM_SAMPLING=yes" for librlscope.so.
                        """))
    parser.add_argument('--calibrate', action='store_true',
                        help=textwrap.dedent("""
                        Perform multiple runs in order to calibrate for profiling overhead 
                        specific to the workload being run.
                        """))
    parser.add_argument("--re-calibrate",
                        action='store_true',
                        help=textwrap.dedent("""
                            Remove existing profiling overhead calibration files, and recompute them.
                            """))
    parser.add_argument("--re-plot",
                        action='store_true',
                        help=textwrap.dedent("""
                            Remove existing plots and remake them (NOTE: doesn't recompute analysis; see --re-calibrate).
                            """))
    parser.add_argument("--parallel-runs",
                        action='store_true',
                        help=textwrap.dedent("""
                            Parallelize running configurations across GPUs on this machine (assume no CPU inteference). See --gpus
                            """))
    parser.add_argument("--retry",
                        type=int,
                        help=textwrap.dedent("""
                            If a command fails, retry it up to --retry times.
                            Default: don't retry.
                            """))
    parser.add_argument("--dry-run",
                        action='store_true',
                        help=textwrap.dedent("""
                            Dry run
                            """))
    # parser.add_argument("--gpus",
    #                     action='store_true',
    #                     help=textwrap.dedent("""
    #                         Parallelize running configurations across GPUs on this machine (assume no CPU inteference). See --iml-gpus
    #                         """))
    parser.add_argument("--gpus",
                        help=textwrap.dedent("""
                        # Run on the first GPU only
                        --gpus 0
                        # Run on the first 2 GPUs
                        --gpus 0,1
                        # Run on all available GPUs
                        --gpus all
                        # Don't allow running with any GPUs (CUDA_VISIBLE_DEVICES="")
                        --gpus none
                        """))
    parser.add_argument('--config',
                        choices=['interception',
                                 'no-interception',
                                 'gpu-activities',
                                 'gpu-activities-api-time',
                                 'no-gpu-activities',
                                 'full',
                                 'time-breakdown',
                                 'gpu-hw',
                                 'uninstrumented',
                                 ],
                        # Detect if user provides --config or not.
                        # By default, run with full IML instrumentation.
                        # default=DEFAULT_CONFIG,
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
                                
                        gpu-hw:
                          ONLY collect GPU hardware counters
                        """))
    args = parser.parse_args(iml_prof_argv)

    if args.iml_rm_traces_from is not None:
        logger.info("iml-prof: Delete trace-files rooted at --iml-directory = {dir}".format(
            dir=args.iml_rm_traces_from))
        return

    rlscope_api.find_librlscope()
    so_path = rlscope_api.RLSCOPE_CLIB
    assert so_path is not None
    env = dict(os.environ)
    add_env = dict()
    add_env['LD_PRELOAD'] = "{ld}:{so_path}".format(
        ld=env.get('LD_PRELOAD', ''),
        so_path=so_path)
    # Q: I just want LD_LIBRARY_PATH to get printed...
    if 'LD_LIBRARY_PATH' in env:
        add_env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']
    # if 'LD_LIBRARY_PATH' in env:
    #     add_env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']

    def _set_if_none(attr, value):
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    if args.calibrate:
        if args.config is not None:
            logger.error("Only --calibrate or --config should be provided for iml-prof.")
            parser.exit(1)
        # Run calibrate.py
        cmd = ['iml-calibrate', 'run']
        if args.parallel_runs:
            cmd.extend(['--parallel-runs'])
            iml_prof_argv.remove('--parallel-runs')
        if args.retry is not None:
            cmd.extend(['--retry', str(args.retry)])

        # Q: Can't we just pass this through?
        # if args.re_calibrate:
        #     cmd.extend(['--re-calibrate'])
        #     iml_prof_argv.remove('--re-calibrate')

        # if args.gpus is not None:
        #     cmd.extend(['--gpus', args.gpus])
        iml_prof_argv.remove('--calibrate')
        cmd.extend(iml_prof_argv)
        cmd.extend(cmd_argv)
        # cmd.remove('--calibrate')
        print_cmd(cmd)
        try:
            proc = subprocess.run(cmd, check=False)
            sys.exit(proc.returncode)
        except KeyboardInterrupt:
            logger.info("Saw Ctrl-C during calibration; aborting remaining runs.")
            sys.exit(1)

    if args.config is None:
        args.config = DEFAULT_CONFIG

    add_env['IML_CONFIG'] = args.config
    if args.config == 'interception':
        "iml-prof --debug --cuda-api-calls --cuda-api-events"
        _set_if_none('cuda_api_calls', True)
        _set_if_none('cuda_api_events', True)
    elif args.config in ['no-interception', 'uninstrumented']:
        "iml-prof --debug"
        pass
    elif args.config == 'gpu-hw':
        "$ iml-prof --debug --gpu-hw"
        _set_if_none('cuda_api_calls', False)
        _set_if_none('cuda_api_events', False)
        _set_if_none('cuda_activities', False)
        _set_if_none('gpu_hw', True)
    elif args.config == 'no-gpu-activities':
        "$ iml-prof --debug --cuda-api-calls"
        _set_if_none('cuda_api_calls', True)
        _set_if_none('gpu_hw', False)
    elif args.config == 'gpu-activities':
        "$ iml-prof --debug --cuda-api-calls --cuda-activities"
        _set_if_none('cuda_api_calls', True)
        _set_if_none('cuda_activities', True)
        _set_if_none('gpu_hw', False)
    elif args.config == 'gpu-activities-api-time':
        "$ iml-prof --debug --cuda-api-calls --cuda-api-events --cuda-activities"
        _set_if_none('cuda_api_calls', True)
        _set_if_none('cuda_api_events', True)
        _set_if_none('cuda_activities', True)
        _set_if_none('gpu_hw', False)
    elif args.config is None or args.config in {'full', 'time-breakdown'}:
        "$ iml-prof --cuda-api-calls --cuda-api-events --cuda-activities"
        _set_if_none('cuda_api_calls', True)
        _set_if_none('cuda_api_events', True)
        _set_if_none('cuda_activities', True)
        _set_if_none('gpu_hw', False)
    else:
        raise NotImplementedError()

    if args.fuzz_cuda_api and args.cuda_api_calls:
        parser.error("Can only run iml-prof with --fuzz-cuda-api or --cuda-api-calls, not both")

    if args.debug or args.iml_debug or is_env_true('IML_DEBUG'):
        logger.info("Detected debug mode; enabling C++ logging statements (export IML_CPP_MIN_VLOG_LEVEL=1)")
        add_env['IML_CPP_MIN_VLOG_LEVEL'] = 1

    # if args.iml_disable:
    #     add_env['IML_DISABLE'] = 'yes'

    def set_yes_no(attr, env_var):
        if getattr(args, attr):
            add_env[env_var] = 'yes'
        else:
            add_env[env_var] = 'no'

    set_yes_no('cuda_api_calls', 'IML_CUDA_API_CALLS')

    set_yes_no('cuda_activities', 'IML_CUDA_ACTIVITIES')

    set_yes_no('gpu_hw', 'IML_GPU_HW')

    set_yes_no('pc_sampling', 'IML_PC_SAMPLING')

    set_yes_no('fuzz_cuda_api', 'IML_FUZZ_CUDA_API')

    set_yes_no('cuda_api_events', 'IML_CUDA_API_EVENTS')

    set_yes_no('gpu_hw', 'IML_GPU_HW')

    set_yes_no('trace_at_start', 'IML_TRACE_AT_START')

    set_yes_no('stream_sampling', 'IML_STREAM_SAMPLING')

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
    # os.execve shouldn't return.
    assert False

def is_env_true(var, env=None):
    if env is None:
        env = os.environ
    return env.get(var, 'no').lower() not in {'no', 'false', '0', 'None', 'null'}

def add_bool_arg(parser, opt, dest=None, default=None, **add_argument_kwargs):
    if dest is None:
        dest = opt
        dest = re.sub(r'^--', '', dest)
        dest = re.sub(r'-', '_', dest)
    opt = re.sub(r'^--', '', opt)
    # print(f"ADD: --{opt}, dest={dest}")
    parser.add_argument(f"--{opt}", dest=dest, action='store_true', **add_argument_kwargs)
    parser.add_argument(f"--no-{opt}", dest=dest, action='store_false', **add_argument_kwargs)
    # if default is not None:
    parser.set_defaults(**{
        dest: default,
    })

if __name__ == '__main__':
    main()
