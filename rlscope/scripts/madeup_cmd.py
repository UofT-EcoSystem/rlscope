"""
Used for testing ``rls-run-expr`` program which parallelizes running
multiple training script configurations across available GPUs.
"""
import argparse
import textwrap
import time

import os

from rlscope.profiler.rlscope_logging import logger

def main():
    parser = argparse.ArgumentParser(
        description="Test: rls-run-expr",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--rlscope-directory',
                        help=textwrap.dedent("""
                        The output directory of the command being run.
                        This is where logfile.out will be output.
                        """))
    parser.add_argument('--wait-sec',
                        default=10,
                        help=textwrap.dedent("""
                        Busy wait for --wait-sec seconds
                        """))
    parser.add_argument('--print-every-sec',
                        default=2,
                        help=textwrap.dedent("""
                        Print every seconds
                        """))
    parser.add_argument('--fail',
                        action='store_true',
                        help=textwrap.dedent("""
                        Fail with unhandled exception. 
                        """))
    args = parser.parse_args()

    if args.fail:
        raise RuntimeError("Fail with unhandled exception")

    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')
    start_t = time.time()
    end_t = start_t + args.wait_sec
    now_sec = time.time()
    while now_sec <= end_t:
        logger.info(f"--rlscope-directory={args.rlscope_directory}, CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}")
        time.sleep(min(end_t - now_sec, args.print_every_sec))
        now_sec = time.time()

if __name__ == '__main__':
    main()
