"""
Run RL-Scope unit tests.
"""
import argparse
import subprocess
import sys
import textwrap
import os
import contextlib

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.iml_logging import logger
from rlscope.profiler.util import print_cmd

from rlscope import py_config

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        formatter_class=argparse.RawTextHelpFormatter)
    # TODO: add --pdb to break on failed python tests, and gdb on failed C++ tests.
    parser.add_argument("--debug",
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug unit tests.
                        """))
    parser.add_argument("--Werror",
                        action='store_true',
                        help=textwrap.dedent("""
                        Treat warnings as errors (pytest)
                        """))
    parser.add_argument("--tests",
                        choices=['py', 'cpp', 'all'],
                        default='all',
                        help=textwrap.dedent("""
                        Which unit tests to run:
                        py:
                          Just python unit tests.
                        cpp:
                          Just C++ unit tests (rls-test).
                        all:
                          Both python and C++ unit tests.
                        """))
    args = parser.parse_args()
    unit_tests = RLSUnitTests(args)
    unit_tests.run()

class RLSUnitTests:
    def __init__(self, args):
        self.args = args

    def run_py(self):
        # TODO: run pytest with appropriate cmdline options.
        # Q: record output?
        args = self.args
        with with_chdir(py_config.INSTALL_ROOT):

            # 'python'
            cmd = [sys.executable]
            if args.Werror:
                cmd.append('-Werror')
            # '-Wignore:::_pytest.assertion.rewrite' Suppresses deprecation warnings
            # in pytest (up to at least version 6.1.1)
            #
            # https://github.com/pytest-dev/pytest/issues/1403#issuecomment-443533232
            cmd.extend(['-Wignore:::_pytest.assertion.rewrite', '-m', 'pytest'])
            if args.debug:
                cmd.append(['--pdb', '-s'])

            print_cmd(cmd)
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                logger.error("RL-Scope python unit tests failed")
                sys.exit(proc.returncode)
        logger.info("RL-Scope python unit tests PASSED")

    def run_cpp(self):
        args = self.args
        cmd = ['rls-test']
        if args.debug:
            cmd = ['gdb', '--args'] + cmd
        print_cmd(cmd)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error("RL-Scope C++ unit tests failed")
            sys.exit(proc.returncode)
        logger.info("RL-Scope C++ unit tests PASSED")

    def run(self):
        args = self.args

        os.environ['RLS_RUNNING_UNIT_TESTS'] = 'yes'

        if args.tests in ['py', 'all']:
            self.run_py()

        if args.tests in ['cpp', 'all']:
            self.run_cpp()

@contextlib.contextmanager
def with_chdir(directory):
    cur_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(cur_directory)

if __name__ == '__main__':
    main()
