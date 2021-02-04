"""
Python wrapper for locating and running C++ ``rls-analyze`` binary
distributed with RL-Scope python wheel.
"""
from rlscope.profiler.rlscope_logging import logger
import sys
import textwrap
import os

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.util import print_cmd
import rlscope

DEBUG = False

INSTALL_ROOT = _d(os.path.realpath(rlscope.__file__))
CPP_LIB = _j(INSTALL_ROOT, 'cpp', 'lib')
CPP_BIN = _j(INSTALL_ROOT, 'cpp', 'bin')
CPP_INCLUDE = _j(INSTALL_ROOT, 'cpp', 'include')

def execve_rlscope_binary(binary):
    exe_path = _j(CPP_BIN, binary)
    if not os.path.exists(exe_path):
        logger.error("Couldn't find {bin} binary @ {path}".format(
            bin=binary,
            path=exe_path,
        ))
        sys.exit(1)
    cmd = [exe_path] + sys.argv[1:]
    if DEBUG:
        print_cmd(cmd)
    env = dict(os.environ)

    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(exe_path, cmd, env)
    # Shouldn't return from os.execve
    assert False

def rls_analyze():
    execve_rlscope_binary('rls-analyze')

def rls_test():
    execve_rlscope_binary('rls-test')

def rlscope_pip_installed():
    """
    Install this script on our PATH in production mode.
    DON'T install this script on our PATH in development mode.
    """
    logger.info( textwrap.dedent("""\
    NOTE: this is a 'dummy' executable that tells us that RL-Scope was installed 
    in production mode from a wheel file (i.e., "pip install rlscope==... -f ...").
    """.rstrip()))

def rlscope_is_development_mode():
    """
    Install this script on our PATH in production mode.
    DON'T install this script on our PATH in development mode.
    """
    logger.info( textwrap.dedent("""\
    NOTE: this is a 'dummy' executable that tells us that RL-Scope was installed in 
    development mode (i.e., "python setup.py develop") instead of from a wheel file.
    """.rstrip()))

# NOTE: This file should only be called from entrypoints registered via setup.py
# if __name__ == '__main__':
#     main()
