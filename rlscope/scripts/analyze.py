"""
``rls-run`` script for processing trace files (e.g., cross-stack event overlap)
produced from a training script.
"""
from rlscope.profiler.rlscope_logging import logger
import subprocess
import sys
import warnings
import os
import traceback
import pdb
import luigi
import argparse
import pprint
import textwrap
import multiprocessing
from rlscope.parser import tasks
from rlscope.profiler.rlscope_logging import logger
from rlscope.parser import check_host
from rlscope.parser.exceptions import RLScopeConfigurationError

def main():

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
        Process trace-files collected from running an ML script with the RL-Scope profiler.
        
        For task-specific help, provided task-name and --help, e.g.:
        $ rls-run --task OverlapStackedBarTask --help
        
        NOTE: 
        - This script is a thin usage/debugging wrapper around a "luigi" DAG execution script. 
          It just forwards arguments to it.
        - Any unparsed args are forward to the luigi script.
        """),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument('--pdb', action='store_true',
                        help="Break into pdb when an exception occurs")
    parser.add_argument('--task',
                        choices=[klass.__name__ for klass in tasks.RLSCOPE_TASKS],
                        help="Name of a runnable IMLTask defined in rlscope.parser.tasks")
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
        parser.print_help()
        sys.exit(0)

    if args.task is None and not args.help:
        # If they just run this:
        # $ rls-run --rlscope-directory <dir>
        # Then run all the targets.
        args.task = 'All'

    extra_argv = [
        '--module', 'rlscope.parser.tasks',
        '--local-scheduler',
        # Default log-level from luigi is DEBUG which is too noisy.
        # Make the default level INFO instead.
        '--log-level', 'INFO',
    ]
    luigi_argv.extend(extra_argv)
    if args.task:
        # Task needs to be the first argument after rls-run.
        luigi_argv.insert(1, args.task)

    if args.help:
        luigi_argv.extend(['--help'])

    if args.workers > 1:
        logger.warning("Each overlap plot uses all the cores; forcing --workers=1")
        args.workers = 1

    if args.pdb:
        logger.debug("Registering pdb breakpoint (--pdb)")
        register_pdb_breakpoint()
        # Debugger is useless when multithreaded.
        args.workers = 1

    luigi_argv.extend(['--workers', str(args.workers)])

    # logger.debug("Luigi arguments:\n{msg}".format(msg=textwrap.indent(pprint.pformat({
    #     'luigi_argv':luigi_argv,
    #     'sys.argv':sys.argv,
    # }), prefix='  ')))

    with warnings.catch_warnings():
        # I don't really take much advantage of luigi's DFS scheduler and instead run things manually.
        # Oh well.
        warnings.filterwarnings('ignore', category=UserWarning, message=r'.*without outputs has no custom complete', module=r'luigi')
        warnings.filterwarnings('ignore', category=UserWarning, message=r'Parameter.*with value "None" is not of type string', module=r'luigi')
        tasks.main(argv=luigi_argv[1:], should_exit=False)

def register_pdb_breakpoint():

    def pdb_breakpoint(task, ex):
        logger.error("> Detected unhandled exception {ex} in {task}; entering pdb".format(
            ex=ex.__class__.__name__,
            task=task.__class__.__name__,
        ))
        pdb.post_mortem()

    def register_pdb_on(luigi_event):
        register_failure_event = luigi.Task.event_handler(luigi_event)
        register_failure_event(pdb_breakpoint)

    register_pdb_on(luigi.Event.FAILURE)
    register_pdb_on(luigi.Event.PROCESS_FAILURE)

if __name__ == '__main__':
    main()
