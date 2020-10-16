"""
Run RL-Scope unit tests.
"""
import argparse
import subprocess
import sys
import textwrap
import os

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.profiler.iml_logging import logger
from iml_profiler.profiler.util import print_cmd

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
        cmd = ['pytest']
        if args.debug:
            cmd.append(['--pdb', '-s'])
        print_cmd(cmd)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error("RL-Scope python unit tests failed")
            sys.exit(proc.returncode)

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

    def run(self):
        args = self.args

        if args.tests in ['py', 'all']:
            self.run_py()

        if args.tests in ['cpp', 'all']:
            self.run_cpp()


if __name__ == '__main__':
    main()
