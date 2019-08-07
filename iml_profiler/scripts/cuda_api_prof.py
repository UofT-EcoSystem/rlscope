import logging
import subprocess
import argparse
import textwrap
import sys
import numpy as np

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler import py_config

from iml_profiler.parser.common import *

from iml_profiler.profiler import iml_logging

def add_cuda_api_prof_arguments(parser):
    """
    Arguments parsed by iml-cuda-api-prof that should NOT be forwarded to the training-script.
    """
    pass

def add_common_arguments(parser):
    """
    Arguments parsed by iml-cuda-api-prof that SHOULD be forwarded to the training-script,
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
    so_path = _j(py_config.ROOT, 'build', 'libsample_cuda_api.so')
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
          $ cmake .. -DProtobuf_INCLUDE_DIR=$HOME/protobuf/include
                     -DProtobuf_LIBRARY=$HOME/protobuf/lib/libprotobuf.so
                     
          $ make -j$(nproc)
        """.format(
            root=py_config.ROOT,
            path=so_path,
        )))
        sys.exit(1)
    env['LD_PRELOAD'] = "{ld}:{so_path}".format(
        ld=env.get('LD_PRELOAD', ''),
        so_path=so_path)
    # if args.debug or args.iml_debug:
    print_cmd(argv, env={
        'LD_PRELOAD': env['LD_PRELOAD'],
    })

    proc = subprocess.run(argv, env=env)
    sys.exit(proc.returncode)

    # os.execve(argv[1], argv, env)
    # # Shouldn't return.
    # assert False

if __name__ == '__main__':
    main()
