"""
``rls-run-expr`` command for running shell commands across the
available GPUs on this machine.
Used by ``rls-prof --calibrate --parallel-runs`` to run multiple training
script configurations in parallel.

Usage:
  # --append: append the shell command to run_expr.sh.  If logfile.out already exists and is done, SKIP it.
  $ rls-run-expr --append --sh run_expr.sh rls-prof ...

  # --run: run the shell command as-is (note: --sh is optional and ignored)
  $ rls-run-expr --run --sh run_expr.sh rls-prof ...

  # --append --run: append the shell command to run_expr.sh, then run the command as-is
  $ rls-run-expr --append --run --sh run_expr.sh rls-prof ...

  # Run all the experiments in run_expr.sh
  $ rls-run-expr --run-sh --sh run_expr.sh
"""
import argparse
import time
import pprint
import traceback
import shlex
import shutil
import socket
import textwrap
import re
import os
import sys
import multiprocessing
import queue

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import progressbar

from rlscope import py_config
from rlscope.profiler import rlscope_logging
from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler.util import gather_argv, run_with_pdb, error, get_available_cpus, get_available_gpus
from rlscope.experiment.util import expr_run_cmd, expr_already_ran
from rlscope.parser import check_host
from rlscope.parser.exceptions import RLScopeConfigurationError


# This is slow.
# from rlscope.parser.common import *

class RunExpr:
    def __init__(self,
                 cmd=None,
                 run=False,
                 append=False,
                 rlscope_directory=None,
                 sh=None,
                 tee=False,
                 run_sh=False,
                 retry=None,
                 debug=False,
                 line_numbers=False,
                 verbosity='progress',
                 skip_final_error_message=False,
                 dry_run=False,
                 skip_errors=False,
                 gpus=None):
        self.cmd = cmd
        self.run = run
        self.tee = tee
        self.append = append
        self.rlscope_directory = rlscope_directory
        self.sh = sh
        self.run_sh = run_sh
        self.retry = retry
        self.debug = debug
        self.line_numbers = line_numbers
        self.verbosity = verbosity
        self.skip_final_error_message = skip_final_error_message
        self.dry_run = dry_run
        self.skip_errors = skip_errors
        self.gpus = gpus
        # TODO: make a "worker process" for each gpu, make it launch commands with "export CUDA_VISIBLE_DEVICES=${GPU},
        # if command fails, forward the failure to run_expr.py (decide what to do based on --skip-errors)
        self.cmd_queue = multiprocessing.Queue()
        self.gpu_workers = dict()
        self.should_stop = multiprocessing.Event()
        self.worker_failed = multiprocessing.Event()

    def start_gpu_workers(self):
        for gpu in self.gpus:
            self.gpu_workers[gpu] = multiprocessing.Process(target=self._gpu_worker, args=(self, gpu))
            self.gpu_workers[gpu].start()

    @staticmethod
    def _gpu_worker(self, gpu, *args, **kwargs):
        try:
            self.gpu_worker(gpu, *args, **kwargs)
        except KeyboardInterrupt:
            logger.debug(f"GPU[{gpu}] worker saw Ctrl-C; exiting early")
            return
        except Exception as e:
            logger.error(textwrap.dedent("""\
            BUG: GPU[{gpu}] worker failed with unhandled exception:
            {error}
            """).format(
                gpu=gpu,
                error=textwrap.indent(traceback.format_exc(), prefix='  '),
            ).rstrip())
            sys.exit(1)

    def gpu_worker(self, gpu):
        # :\n{msg}
        logger.debug("Start GPU[{gpu}] worker; queue contains {len} items".format(
            len=self.cmd_queue.qsize(),
            gpu=gpu,
            # msg=textwrap.indent(pprint.pformat(list(self.cmd_queue)), prefix='  '),
        ))
        while True:
            run_cmd = None
            try:
                # run_cmd = self.cmd_queue.get(block=False)
                logger.debug(f"Get: GPU[{gpu}] worker...")
                # This hangs forever when timeout isn't provided...
                # run_cmd = self.cmd_queue.get()
                run_cmd = self.cmd_queue.get(timeout=1)
            except queue.Empty:
                logger.debug(f"No more items in queue; exiting GPU[{gpu}] worker")
                break
            # except socket.timeout:
            #     logger.debug(f"Saw timeout; no more items in queue (len={self.cmd_queue.qsize()})? Exiting GPU[{gpu}] worker")
            #     assert self.cmd_queue.qsize() == 0
            #     break

            logfile = run_cmd.logfile
            logger.debug(
                textwrap.dedent("""\
                    GPU[{gpu}] worker running command (queue size = {len}):
                    > CMD:
                      logfile={logfile}
                      $ {cmd}
                    """).format(
                    len=self.cmd_queue.qsize(),
                    cmd=' '.join(run_cmd.cmd),
                    gpu=gpu,
                    logfile=logfile,
                ).rstrip())
            if self.retry is None:
                retries = 1
            else:
                retries = self.retry
            for attempt in range(1, retries+1):
                proc = self.run_cmd(gpu, run_cmd, tee_output=self.tee or self.should_show_output, tee_prefix=f"GPU[{gpu}] :: ", attempt=attempt)
                if self.dry_run or proc is None or proc.returncode == 0:
                    break

            if self.should_stop.is_set() or self.worker_failed.is_set():
                logger.debug(f"Got exit signal; exiting GPU[{gpu}] worker")
                break

    def as_run_cmd(self, cmd, output_directory=None, logfile=None):
        if output_directory is None:
            output_directory = self.cmd_output_directory(cmd)
            assert output_directory is not None
        if logfile is None:
            logfile = self.cmd_logfile(cmd, output_directory)
        # Strip --rlscope-logfile if present
        keep_cmd = self.cmd_keep_args(cmd)
        run_cmd = RunCmd(keep_cmd, output_directory, logfile)
        return run_cmd

    def run_commands(self):
        run_commands = []
        with open(self.sh, 'r') as f:
            for line in f:
                line = line.rstrip()
                if re.search(r'^\s*$', line):
                    # Skip empty lines.
                    continue
                # Remove comments
                # PROBLEM: comment may be inside string...
                # re.sub(r'#.*', line)
                cmd = shlex.split(line)
                run_cmd = self.as_run_cmd(cmd)
                run_commands.append(run_cmd)
        return run_commands

    @staticmethod
    def _run_cmd(self, *args, **kwargs):
        return self.run_cmd(*args, **kwargs)

    def cmd_output_directory(self, cmd):
        parser = argparse.ArgumentParser(
            description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--rlscope-directory',
                            help=textwrap.dedent("""
                            The output directory of the command being run.
                            This is where logfile.out will be output.
                            """))
        args, _ = parser.parse_known_args(cmd)
        if args.rlscope_directory is None:
            return os.getcwd()
        return args.rlscope_directory

    def cmd_keep_args(self, cmd):
        parser = argparse.ArgumentParser(
            description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--rlscope-logfile',
                            help=textwrap.dedent("""
                            Where to output stdout/stderr of running cmd.
                            """))
        args, keep_args = parser.parse_known_args(cmd)
        # if args.rlscope_directory is None:
        #     return os.getcwd()
        # return args.rlscope_directory
        return keep_args

    def cmd_logfile(self, cmd, output_directory):
        parser = argparse.ArgumentParser(
            description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--rlscope-logfile',
                            help=textwrap.dedent("""
                            Where to output stdout/stderr of running cmd.
                            """))
        args, _ = parser.parse_known_args(cmd)
        if args.rlscope_logfile is not None:
            return args.rlscope_logfile

        task = "RunExpr"
        logfile = _j(
            output_directory,
            self._logfile_basename(task),
        )
        return logfile

    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def run_cmd(self, gpu, run_cmd, tee_output=False, tee_prefix=None, attempt=None):
        cmd = run_cmd.cmd
        # output_directory = run_cmd.output_directory

        # NOTE: rls-prof will output its own logfile.out...?
        # logfile = self.cmd_logfile(output_directory)
        logfile = run_cmd.logfile
        # Only use one gpu.
        env = dict(os.environ)
        if gpu is None:
            # Explicitly don't use any GPUs
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        can_retry = self.retry is not None and attempt is not None and attempt < self.retry
        if can_retry:
            log_func = logger.warning
            attempt_str = "[ATTEMPT {attempt}/{retry}] ".format(attempt=attempt, retry=self.retry)
        else:
            log_func = logger.error
            attempt_str = ""
        tee_cmd = self.should_show_commands
        proc = expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            env=env,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            # Only output to logfile to avoid interleaved output from commands.
            tee_output=tee_output,
            tee_cmd=tee_cmd,
            tee_prefix=tee_prefix,
            log_errors=False,
            log_func=log_func,
            skip_error=True,
            debug=self.debug,
            only_show_env={
                'CUDA_VISIBLE_DEVICES',
            },
        )
        if not self.dry_run and proc is not None and proc.returncode != 0:
            if not self.skip_errors:
                log_func(
                    # ; use --skip-errors to ignore failure and continue with other commands
                    textwrap.dedent("""\
                    {attempt_str}Saw failed cmd in GPU[{gpu}] worker.
                    > CMD:
                      logfile={logfile}
                      $ {cmd}
                    """).format(
                        attempt_str=attempt_str,
                        cmd=' '.join(cmd),
                        gpu=gpu,
                        logfile=logfile,
                    ).rstrip())
                if not can_retry:
                    # Signal stop
                    self.worker_failed.set()
            else:
                log_func(
                    textwrap.dedent("""\
                    {attempt_str}--skip-errors: SKIP failed cmd in GPU[{gpu}] worker:
                    > CMD:
                      logfile={logfile}
                      $ {cmd}
                    """).format(
                        attempt_str=attempt_str,
                        cmd=' '.join(cmd),
                        gpu=gpu,
                        logfile=logfile,
                    ).rstrip())
        return proc

    def _expr_run_cmd(self, *args, **kwargs):
        tee_output = kwargs.get('tee_output', None)
        if tee_output is None:
            tee_output = (self.verbosity == 'output')
        tee_cmd = (self.verbosity == 'commands')
        return expr_run_cmd(
            *args,
            raise_exception=True,
            tee_output=tee_output,
            tee_cmd=tee_cmd,
            log_errors=False,
            **kwargs)

    @property
    def should_show_progress(self):
        return self.verbosity == 'progress'

    @property
    def should_show_commands(self):
        return self.verbosity == 'commands'

    @property
    def should_show_output(self):
        return self.verbosity == 'output'

    def mode_run_sh(self):
        # Fill queue with commands to run.
        run_commands = self.run_commands()
        for run_cmd in run_commands:
            logger.debug(f"Put: {run_cmd}")
            self.cmd_queue.put(run_cmd)

        self.start_gpu_workers()

        bar = None
        if self.should_show_progress:
            bar = progressbar.ProgressBar(max_value=len(run_commands))
        last_completed = None

        # Wait for workers to terminate
        try:
            while True:
                if self.should_show_progress:
                    completed = len(run_commands) - self.cmd_queue.qsize()
                    if last_completed is None or completed > last_completed:
                        bar.update(completed)
                    last_completed = completed

                if self.worker_failed.is_set():
                    self.stop_workers()
                    # ; use --skip-errors to ignore failed commands.
                    if not self.skip_final_error_message:
                        logger.error("At least one command failed with non-zero exit status")
                    if self.should_show_progress:
                        bar.finish()
                    sys.exit(1)

                alive_workers = 0
                failed_workers = 0
                for gpu, worker in self.gpu_workers.items():
                    if worker.is_alive():
                        alive_workers += 1
                        continue

                    if worker.exitcode < 0:
                        logger.error("GPU[{gpu}] worker failed with exitcode={ret} (unhandled exception)".format(
                            gpu=gpu,
                            ret=worker.exitcode,
                        ))
                        self.worker_failed.set()
                        failed_workers += 1

                if failed_workers > 0:
                    self.stop_workers()
                    if self.should_show_progress:
                        bar.finish()
                    sys.exit(1)

                if alive_workers == 0:
                    if self.cmd_queue.qsize() > 0:
                        logger.warning("GPU workers have finished with {len} remaining commands unfinished".format(
                            len=self.cmd_queue.qsize()
                        ))
                        sys.exit(1)
                    logger.debug("GPU workers have finished successfully".format(
                        len=self.cmd_queue.qsize()
                    ))
                    if self.should_show_progress:
                        bar.finish()
                    sys.exit(0)

                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Saw Ctrl-C; waiting for workers to terminate")
            self.stop_workers()
            logger.warning("{len} remaining commands went unprocessed".format(len=self.cmd_queue.qsize()))
            if self.should_show_progress:
                bar.finish()
            sys.exit(1)

    def stop_workers(self):
        self.should_stop.set()
        for gpu, worker in self.gpu_workers.items():
            logger.debug(f"Wait for GPU[{gpu}] worker...")
            worker.join()

    def run_program(self):
        if self.run_sh:
            self.mode_run_sh()
            return

        if self.append:
            with open(self.sh, 'a') as f:
                quoted_cmd = [shlex.quote(opt) for opt in self.cmd]
                f.write(' '.join(quoted_cmd))
                f.write('\n')

        if self.run:
            run_cmd = self.as_run_cmd(self.cmd, output_directory=self.rlscope_directory)
            proc = self.run_cmd(self.gpus[0], run_cmd, tee_output=True)
            if not self.dry_run and proc is not None and proc.returncode != 0:
                sys.exit(proc.returncode)


class RunCmd:
    def __init__(self, cmd, output_directory, logfile):
        self.cmd = cmd
        self.output_directory = output_directory
        self.logfile = logfile

    def __repr__(self):
        return "{klass}(cmd=\"{cmd}\", output_directory={output_directory}, logfile={logfile})".format(
            klass=self.__class__.__name__,
            cmd=' '.join(self.cmd),
            output_directory=self.output_directory,
            logfile=self.logfile
        )



def main():

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run",
                        action='store_true',
                        help=textwrap.dedent("""\
                        Run the command as-is.
                        """))
    parser.add_argument("--append",
                        action='store_true',
                        help=textwrap.dedent("""\
                        Append the command to --sh
                        """))
    parser.add_argument("--sh",
                        help=textwrap.dedent("""\
                        Shell file to append commands to (see --append).
                        """))
    parser.add_argument('--run-sh',
                        action='store_true',
                        help=textwrap.dedent("""\
                        Run all the commands in --sh on the available --gpus
                        """))
    parser.add_argument('--rlscope-directory',
                        help=textwrap.dedent("""\
                        The output directory of the command being run.
                        This is where logfile.out will be output.
                        """))
    parser.add_argument("--verbosity",
                        choices=['progress', 'commands', 'output'],
                        default='progress',
                        help=textwrap.dedent("""\
                            Output information about running commands.
                            --verbosity progress (Default)
                                Only show high-level progress bar information.
                              
                            --verbosity commands
                                Show the command-line of commands that are being run.
                                
                            --verbosity output
                                Show the output of each analysis (not configuration) command on sys.stdout.
                                NOTE: This may cause interleaving of lines.
                            """))
    parser.add_argument('--line-numbers', action='store_true', help=textwrap.dedent("""\
    Show line numbers and timestamps in RL-Scope logging messages.
    """))
    parser.add_argument('--debug',
                        action='store_true',
                        help=textwrap.dedent("""\
                        Debug
                        """))
    parser.add_argument('--skip-final-error-message',
                        action='store_true',
                        help=textwrap.dedent("""\
                        Skip error message printed at the end if at least one command fails.
                        """))
    parser.add_argument("--retry",
                        type=int,
                        help=textwrap.dedent("""\
                            If a command fails, retry it up to --retry times.
                            Default: don't retry.
                            """))
    parser.add_argument("--tee",
                        action='store_true',
                        help=textwrap.dedent("""\
                        (debug)
                        tee output of parallel processes to stdout (prefix output with worker name)
                        """))
    parser.add_argument("--pdb",
                        action='store_true',
                        help=textwrap.dedent("""\
                        Debug
                        """))
    parser.add_argument('--dry-run',
                        action='store_true',
                        help=textwrap.dedent("""\
                        Dry run
                        """))
    parser.add_argument('--skip-errors',
                        action='store_true',
                        help=textwrap.dedent("""\
                        If a command fails, ignore the failure and continue running other commands.
                        """))
    parser.add_argument("--gpus",
                        default='all',
                        help=textwrap.dedent("""\
                        # Run on the first GPU only
                        --gpus 0
                        # Run on the first 2 GPUs
                        --gpus 0,1
                        # Run on all available GPUs
                        --gpus all
                        # Don't allow running with any GPUs (CUDA_VISIBLE_DEVICES="")
                        --gpus none
                        """))
    all_args, _ = parser.parse_known_args(sys.argv)
    ignore_opts = set()
    if all_args.sh is not None:
        ignore_opts.add(all_args.sh)
    run_expr_argv, cmd = gather_argv(
        sys.argv[1:],
        ignore_opts=ignore_opts)
    args = parser.parse_args(run_expr_argv)

    if args.debug:
        logger.debug({
            'run_expr_argv': run_expr_argv,
            'cmd': cmd,
        })

    rlscope_logging.setup_logger(
        debug=args.debug,
        line_numbers=args.debug or args.line_numbers or py_config.is_development_mode(),
    )

    if args.sh is None and ( args.run_sh or args.append ):
        error("--sh is required when either --run-sh or --append are given", parser=parser)

    if args.run_sh and ( args.append or args.run ):
        error("When --run-sh is given, you cannot provide either --append or --run", parser=parser)

    available_gpus = get_available_gpus()
    if args.gpus == 'all':
        gpus = sorted([gpu['device_number'] for gpu in available_gpus])
    elif args.gpus.lower() == 'none':
        args.gpus = [None]
    else:
        try:
            gpus = sorted([int(gpu) for gpu in re.split(r',', args.gpus)])
        except ValueError:
            error("Failed to parser --gpus={gpus}".format(gpus=args.gpus), parser=parser)

    assert len(gpus) >= 1

    if args.run or args.append:
        if len(cmd) == 0:
            error("Expected cmd to run in arguments, but none was provided",
                  parser=parser)

        if shutil.which(cmd[0]) is None:
            error("Couldn't find {exec} on PATH".format(
                exec=cmd[0]), parser=parser)

    if all_args.rlscope_directory is None:
        # No --rlscope-directory argument; just use current directory?
        args.rlscope_directory = os.getcwd()
    else:
        args.rlscope_directory = all_args.rlscope_directory
    # # error("\n  {cmd}".format(cmd=' '.join(cmd)))
    # error(textwrap.dedent("""\
    # --rlscope-directory must be provided so we know where to output logfile.out for cmd:
    #   > CMD:
    #     $ {cmd}
    #   """).format(
    #   cmd=' '.join(cmd),
    # ).rstrip())
    # # "Copy" --rlscope-directory argument from cmd.
    # args.rlscope_directory = all_args.rlscope_directory

    args_dict = dict(vars(args))
    args_dict.pop('gpus')
    args_dict.pop('pdb')
    obj = RunExpr(
        cmd=cmd,
        gpus=gpus,
        **args_dict,
    )

    def _run():
        obj.run_program()
    run_with_pdb(args, _run)

if __name__ == '__main__':
    main()
