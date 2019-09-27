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
from iml_profiler.parser.common import print_cmd
from iml_profiler.profiler import iml_logging

def main():
    iml_logging.setup_logging()

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
                        # DISABLE --workers for now to prevent opening to many postgres connections by accident;
                        # we parallelize internally instead
                        # e.g. ResourceOverlap with 32 worker threads, each of which opens a SQL
                        # connection.
                        # default=multiprocessing.cpu_count(),
                        default=1,
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
        # Default log-level from luigi is DEBUG which is too noisy.
        # Make the default level INFO instead.
        '--log-level', 'INFO',
    ]
    luigi_argv.extend(extra_argv)
    if args.task:
        # Task needs to be the first argument after iml-analyze.
        luigi_argv.insert(1, args.task)

    if args.help:
        luigi_argv.extend(['--help'])

    if args.workers > 1:
        logging.warning("Each overlap plot uses all the cores; forcing --workers=1")
        args.workers = 1

    if args.pdb:
        logging.info("Registering pdb breakpoint (--pdb)")
        register_pdb_breakpoint()
        # Debugger is useless when multithreaded.
        args.workers = 1

    luigi_argv.extend(['--workers', str(args.workers)])

    # pprint.pprint({
    #     'luigi_argv':luigi_argv,
    #     'sys.argv':sys.argv,
    # })

    tasks.main(argv=luigi_argv[1:], should_exit=False)

def register_pdb_breakpoint():

    def pdb_breakpoint(task, ex):
        logging.info("> Detected unhandled exception {ex} in {task}; entering pdb".format(
            ex=ex.__class__.__name__,
            task=task.__class__.__name__,
        ))
        ipdb.post_mortem()

    def register_pdb_on(luigi_event):
        register_failure_event = luigi.Task.event_handler(luigi_event)
        register_failure_event(pdb_breakpoint)

    register_pdb_on(luigi.Event.FAILURE)
    register_pdb_on(luigi.Event.PROCESS_FAILURE)

if __name__ == '__main__':
    main()
