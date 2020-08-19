"""
Run shell commands across the available GPUs on this machine.

Usage:
  # --append: append the shell command to run_expr.sh.  If logfile.out already exists and is done, SKIP it.
  $ iml-run-expr --append --sh run_expr.sh iml-prof ...

  # --run: run the shell command as-is (not  --sh is optional and ignored)
  $ iml-run-expr --run --sh run_expr.sh iml-prof ...

  # --append --run: append the shell command to run_expr.sh, then run the command as-is
  $ iml-run-expr --append --run --sh run_expr.sh iml-prof ...

  # Run all the experiments in run_expr.sh
  $ iml-run-expr --run-sh --sh run_expr.sh
"""
import argparse
import time
import pprint
import shlex
import shutil
import textwrap
import re
import os
import sys

import multiprocessing
import queue

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.profiler import iml_logging
from iml_profiler.profiler.iml_logging import logger
from iml_profiler.profiler.util import gather_argv, run_with_pdb, error, get_available_cpus, get_available_gpus
from iml_profiler.experiment.util import tee, expr_run_cmd, expr_already_ran

DEBUG = True

# This is slow.
# from iml_profiler.parser.common import *

SENTINEL = object()

class RunExpr:
    def __init__(self,
                 cmd=None,
                 run=False,
                 append=False,
                 iml_directory=None,
                 sh=None,
                 run_sh=False,
                 debug=False,
                 dry_run=False,
                 skip_errors=False,
                 gpus=None):
        self.cmd = cmd
        self.run = run
        self.append = append
        self.iml_directory = iml_directory
        self.sh = sh
        self.run_sh = run_sh
        self.debug = debug
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

    def gpu_worker(self, gpu):
        while True:
            run_cmd = None
            try:
                run_cmd = self.cmd_queue.get(block=False)
            except queue.Empty:
                logger.debug(f"No more items in queue; exiting GPU[{gpu}] worker")
                break

            logfile = run_cmd.logfile
            logger.debug(
                textwrap.dedent("""\
                    GPU[{gpu}] worker running command:
                    > CMD:
                      logfile={logfile}
                      $ {cmd}
                    """).format(
                    cmd=' '.join(run_cmd.cmd),
                    gpu=gpu,
                    logfile=logfile,
                ).rstrip())
            self.run_cmd(gpu, run_cmd, tee_output=False)

            if self.should_stop.is_set() or self.worker_failed.is_set():
                logger.debug(f"Got exit signal; exiting GPU[{gpu}] worker")
                break

    def as_run_cmd(self, cmd, output_directory=None, logfile=None):
        if output_directory is None:
            output_directory = self.cmd_output_directory(cmd)
            assert output_directory is not None
        if logfile is None:
            logfile = self.cmd_logfile(cmd, output_directory)
        # Strip --iml-logfile if present
        keep_cmd = self.cmd_keep_args(cmd)
        run_cmd = RunCmd(keep_cmd, output_directory, logfile)
        return run_cmd

    def each_run_command(self):
        with open(self.sh, 'r') as f:
            for line in f:
                line = line.rstrip()
                # Remove comments
                # PROBLEM: comment may be inside string...
                # re.sub(r'#.*', line)
                cmd = shlex.split(line)
                run_cmd = self.as_run_cmd(cmd)
                yield run_cmd

    @staticmethod
    def _run_cmd(self, *args, **kwargs):
        return self.run_cmd(*args, **kwargs)

    def cmd_output_directory(self, cmd):
        parser = argparse.ArgumentParser(
            description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--iml-directory',
                            help=textwrap.dedent("""
                            The output directory of the command being run.
                            This is where logfile.out will be output.
                            """))
        args, _ = parser.parse_known_args(cmd)
        if args.iml_directory is None:
            return os.getcwd()
        return args.iml_directory

    def cmd_keep_args(self, cmd):
        parser = argparse.ArgumentParser(
            description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--iml-logfile',
                            help=textwrap.dedent("""
                            Where to output stdout/stderr of running cmd.
                            """))
        args, keep_args = parser.parse_known_args(cmd)
        # if args.iml_directory is None:
        #     return os.getcwd()
        # return args.iml_directory
        return keep_args

    def cmd_logfile(self, cmd, output_directory):
        parser = argparse.ArgumentParser(
            description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--iml-logfile',
                            help=textwrap.dedent("""
                            Where to output stdout/stderr of running cmd.
                            """))
        args, _ = parser.parse_known_args(cmd)
        if args.iml_logfile is not None:
            return args.iml_logfile

        task = "RunExpr"
        logfile = _j(
            output_directory,
            self._logfile_basename(task),
        )
        return logfile

    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def run_cmd(self, gpu, run_cmd, tee_output=False):
        cmd = run_cmd.cmd
        # output_directory = run_cmd.output_directory

        # NOTE: iml-prof will output its own logfile.out...?
        # logfile = self.cmd_logfile(output_directory)
        logfile = run_cmd.logfile
        # Only use one gpu.
        env = dict(os.environ)
        if gpu is None:
            # Explicitly don't use any GPUs
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        proc = expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            env=env,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            # Only output to logfile to avoid interleaved output from commands.
            tee_output=tee_output,
            skip_error=True,
            debug=self.debug,
            only_show_env={
                'CUDA_VISIBLE_DEVICES',
            },
        )
        if not self.dry_run and proc is not None and proc.returncode != 0:
            if not self.skip_errors:
                logger.error(
                    textwrap.dedent("""\
                    Saw failed cmd in GPU[{gpu}] worker; use --skip-errors to ignore failure and continue with other commands.
                    > CMD:
                      logfile={logfile}
                      $ {cmd}
                    """).format(
                        cmd=' '.join(cmd),
                        gpu=gpu,
                        logfile=logfile,
                    ).rstrip())
                # Signal stop
                self.worker_failed.set()
            else:
                logger.error(
                    textwrap.dedent("""\
                    --skip-errors: SKIP failed cmd in GPU[{gpu}] worker:
                    > CMD:
                      logfile={logfile}
                      $ {cmd}
                    """).format(
                        cmd=' '.join(cmd),
                        gpu=gpu,
                        logfile=logfile,
                    ).rstrip())
        return proc

    def mode_run_sh(self):
        # Fill queue with commands to run.
        for run_cmd in self.each_run_command():
            logger.debug(f"Put: {run_cmd}")
            self.cmd_queue.put(run_cmd)

        self.start_gpu_workers()

        # Wait for workers to terminate
        try:
            while True:
                if self.worker_failed.is_set():
                    self.stop_workers()
                    logger.error("At least one command failed with non-zero exit status; use --skip-errors to ignore failed commands.")
                    sys.exit(1)

                if all(not worker.is_alive() for worker in self.gpu_workers.values()):
                    logger.info("All GPU workers have finished")
                    break

                time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Saw Ctrl-C; waiting for workers to terminate")
            self.stop_workers()

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
            run_cmd = self.as_run_cmd(self.cmd, output_directory=self.iml_directory)
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
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run",
                        action='store_true',
                        help=textwrap.dedent("""
                        Run the command as-is.
                        """))
    parser.add_argument("--append",
                        action='store_true',
                        help=textwrap.dedent("""
                        Append the command to --sh
                        """))
    parser.add_argument("--sh",
                        help=textwrap.dedent("""
                        Shell file to append commands to (see --append).
                        """))
    parser.add_argument('--run-sh',
                        action='store_true',
                        help=textwrap.dedent("""
                        Run all the commands in --sh on the available --gpus
                        """))
    parser.add_argument('--iml-directory',
                        help=textwrap.dedent("""
                        The output directory of the command being run.
                        This is where logfile.out will be output.
                        """))
    parser.add_argument('--debug',
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug
                        """))
    parser.add_argument("--pdb",
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug
                        """))
    parser.add_argument('--dry-run',
                        action='store_true',
                        help=textwrap.dedent("""
                        Dry run
                        """))
    parser.add_argument('--skip-errors',
                        action='store_true',
                        help=textwrap.dedent("""
                        If a command fails, ignore the failure and continue running other commands.
                        """))
    parser.add_argument("--gpus",
                        default='all',
                        help=textwrap.dedent("""
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

    if DEBUG:
        logger.debug({
            'run_expr_argv': run_expr_argv,
            'cmd': cmd,
        })

    if args.debug:
        iml_logging.enable_debug_logging()
    else:
        iml_logging.disable_debug_logging()


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

    if all_args.iml_directory is None:
        # No --iml-directory argument; just use current directory?
        args.iml_directory = os.getcwd()
    else:
        args.iml_directory = all_args.iml_directory
    # # error("\n  {cmd}".format(cmd=' '.join(cmd)))
    # error(textwrap.dedent("""\
    # --iml-directory must be provided so we know where to output logfile.out for cmd:
    #   > CMD:
    #     $ {cmd}
    #   """).format(
    #   cmd=' '.join(cmd),
    # ).rstrip())
    # # "Copy" --iml-directory argument from cmd.
    # args.iml_directory = all_args.iml_directory

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
