"""
iml-analyze script for processing trace-data produced from an ML script.
"""
import logging
import subprocess
import sys
import os
import traceback
import ipdb
import luigi
import argparse
import pprint
import textwrap
import multiprocessing
from iml_profiler.parser import tasks
from iml_profiler.profiler import glbl

from iml_profiler.profiler import glbl
def main():
    glbl.setup_logging()
    parser = argparse.ArgumentParser(
        textwrap.dedent("""\
        Process trace-files collected from running an ML script with the IML profiler.
        
        For task-specific help, provided task-name and --help, e.g.:
        $ iml-analyze --task SQLParserTask --help
        
        NOTE: 
        - This script is a thin usage/debugging wrapper around a "luigi" DAG execution script. 
          It just forwards arguments to it.
        - Any unparsed args are forward to the luigi script.
        """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument('--pdb', action='store_true',
                        help="Break into pdb when an exception occurs")
    parser.add_argument('--task',
                        choices=[klass.__name__ for klass in tasks.IML_TASKS],
                        help="Name of a runnable IMLTask defined in iml_profiler.parser.tasks")
    parser.add_argument('--workers',
                        type=int,
                        default=multiprocessing.cpu_count(),
                        help="Maximum number of parallel tasks to run (luigi parameter)")
    parser.add_argument('--help', '-h',
                        action='store_true')
    args, luigi_argv = parser.parse_known_args(sys.argv)

    if args.help and not args.task:
        # Print available tasks.
        parser.print_usage()
        sys.exit(0)

    if args.task is None and not args.help:
        # If they just run this:
        # $ iml-analyze --iml-directory <dir>
        # Then run all the targets.
        args.task = 'All'

    extra_argv = [
        '--module', 'iml_profiler.parser.tasks',
        '--local-scheduler',
    ]
    luigi_argv.extend(extra_argv)
    if args.task:
        # Task needs to be the first argument after iml-analyze.
        luigi_argv.insert(1, args.task)

    if args.help:
        luigi_argv.extend(['--help'])

    if args.pdb:
        register_pdb_breakpoint()

    # pprint.pprint({
    #     'luigi_argv':luigi_argv,
    #     'sys.argv':sys.argv,
    # })

    tasks.main(argv=luigi_argv[1:])

def print_cmd(cmd, file=sys.stdout):
    if type(cmd) == list:
        cmd_str = " ".join([str(x) for x in cmd])
    else:
        cmd_str = cmd
    print(("> CMD:\n"
           "  $ {cmd}\n"
           "  PWD={pwd}"
           ).format(
        cmd=cmd_str,
        pwd=os.getcwd(),
    ), file=file)

def register_pdb_breakpoint():
    def pdb_breakpoint(task, ex):
        logging.info("> Detected unhandled exception {ex} in {task}; entering pdb".format(
            ex=ex.__class__.__name__,
            task=task.__class__.__name__,
        ))
        ipdb.post_mortem()
    register_failure_event = luigi.Task.event_handler(luigi.Event.FAILURE)
    register_failure_event(pdb_breakpoint)

if __name__ == '__main__':
    main()
