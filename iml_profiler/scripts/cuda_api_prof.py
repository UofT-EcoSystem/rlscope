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

def add_cuda_api_prof_arguments(parser):
    """
    Arguments parsed by iml-prof that should NOT be forwarded to the training-script.
    """
    pass

def add_common_arguments(parser):
    """
    Arguments parsed by iml-prof that SHOULD be forwarded to the training-script,
    but are also likely recognized by the training script (e.g. --debug)
    """
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--iml-debug', action='store_true')

def main():
    iml_logging.setup_logging()

    parser_help = "Sample time spent in CUDA API calls, and call counts."
    parser = argparse.ArgumentParser(parser_help)
    add_cuda_api_prof_arguments(parser)
    _, argv = parser.parse_known_args()

    opt_parser = argparse.ArgumentParser(parser_help)
    add_common_arguments(opt_parser)
    args, _ = opt_parser.parse_known_args(sys.argv[1:])

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

    if args.debug or args.iml_debug or is_env_true('IML_DEBUG'):
        logging.info("Detected debug mode; enabling C++ logging statements (export IML_CPP_MIN_VLOG_LEVEL=1)")
        add_env['IML_CPP_MIN_VLOG_LEVEL'] = 1

    exe_path = shutil.which(argv[0])
    if exe_path is None:
        print("IML ERROR: couldn't locate {exe} on $PATH; try giving a full path to {exe} perhaps?".format(
            exe=argv[0],
        ))
        sys.exit(1)
    # cmd = argv
    cmd = [exe_path] + argv[1:]
    print_cmd(cmd, env=add_env)

    env.update(add_env)
    for k in list(env.keys()):
        env[k] = str(env[k])

    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(exe_path, cmd, env)
    # Shouldn't return.
    assert False

def is_env_true(var, env=None):
    if env is None:
        env = os.environ
    return env.get(var, 'no').lower() not in {'no', 'false', '0', 'None', 'null'}

if __name__ == '__main__':
    main()
