import subprocess
from iml_profiler.profiler.iml_logging import logger
import contextlib

import os
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler.parser.common import *
import tempfile

@contextlib.contextmanager
def in_directory(directory, allow_none=True):
    assert not( not allow_none and directory is None )
    if directory is not None:
        original_directory = os.getcwd()
    try:
        if directory is not None:
            os.chdir(directory)
        yield
    finally:
        if directory is not None:
            os.chdir(original_directory)

def tee(cmd, to_file, cwd=None, append=False, makedirs=True, check=True, dry_run=False, **kwargs):

    # In case there are int's or float's in cmd.
    cmd = [str(opt) for opt in cmd]

    if dry_run:
        with in_directory(cwd):
            print_cmd(cmd, files=[sys.stdout], env=kwargs.get('env', None), dry_run=dry_run)
        return

    with ScopedLogFile(to_file, append=append, makedirs=makedirs) as f:
        with in_directory(cwd):
            print_cmd(cmd, files=[sys.stdout, f], env=kwargs.get('env', None))

            # NOTE: Regarding the bug mentioned below, using p.communicate() STILL hangs.
            #
            # p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
            # with p:
            #     outs, errs = p.communicate()
            #     sys.stdout.write(outs)
            #     sys.stdout.write(errs)
            #     sys.stdout.flush()
            #
            #     f.write(outs)
            #     f.write(errs)
            #     f.flush()

            debug = False

            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
            with p:
                # NOTE: if this blocks it may be because there's a zombie utilization_sampler.py still running
                # (that was created by the training script) that hasn't been terminated!
                # for line in p.stdout:

                while True:

                    if debug:
                        logger.info("RUN [05]: p.poll()")
                    rc = p.poll()
                    if rc is not None:
                        break

                    # BUG: SOMETIMES (I don't know WHY), this line will HANG even after
                    # train_stable_baselines.sh is terminated (it shows up as a Zombie process in htop/top).
                    # Sadly, this only happens occasionally, and I have yet to understand WHY it happens.
                    if debug:
                        logger.info("RUN [06]: line = p.stdout.readline()")
                    line = p.stdout.readline()

                    # b'\n'-separated lines
                    if debug:
                        logger.info("RUN [07]: line.decode")
                    line = line.decode("utf-8")

                    if debug:
                        logger.info("RUN [08]: line (\"{line}\") == '' (result={result})".format(
                            result=(line == ''),
                            line=line))
                    if line == '':
                        break

                    if re.search(r'> train\.py has exited', line):
                        pass
                        # logger.info("> ENABLE TEE DEBUGGING")
                        # debug = True

                    if debug:
                        logger.info("RUN [01]: sys.stdout.write(line)")
                    sys.stdout.write(line)
                    if debug:
                        logger.info("RUN [02]: sys.stdout.flush()")
                    sys.stdout.flush()

                    if debug:
                        logger.info("RUN [03]: f.write(line)")
                    f.write(line)
                    if debug:
                        logger.info("RUN [04]: f.flush()")
                    f.flush()
            sys.stdout.flush()
            f.flush()

            if check and p.returncode != 0:
                raise subprocess.CalledProcessError(p.returncode, p.args)
            return p

EXPERIMENT_SUCCESS_LINE = "IML BENCH DONE"

class ScopedLogFile:
    def __init__(self, file, append=False, makedirs=True):
        self.file = file
        self.append = append
        self.makedirs = makedirs

    def __enter__(self):
        if self._is_path:
            # if self.append:
            #         self.mode = 'ab'
            # else:
            #         self.mode = 'wb'

            if self.append:
                self.mode = 'a'
            else:
                self.mode = 'w'
            if self.makedirs:
                # logger.info("mkdirs {path}".format(path=_d(self.file)))
                os.makedirs(_d(self.file), exist_ok=True)
                # logger.info("ScopedLogFile.file = {path}".format(path=self.file))
            self.f = open(self.file, self.mode)
            return self.f
        else:
            # We can only append to a stream.
            self.f = self.file
            return self.f

    @property
    def _is_path(self):
        return type(self.file) == str

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.flush()
        if self._is_path:
            self.f.close()

def expr_run_cmd(cmd, to_file,
                 cwd=None,
                 env=None,
                 replace=False,
                 dry_run=False,
                 skip_error=False,
                 # extra_argv=[],
                 debug=False):
    """
    Run an experiment, if it hasn't been run already.
    We check if an experiment as already been run by looking for a log file, and whether that logfile has a success-line in it
    (we search for "IML BENCH DONE")
    :param self:
    :param cmd:
    :param to_file:
    :param env:
    :param replace:
    :param debug:
    :return:
    """

    if env is None:
        # Make sure iml-analyze get IML_POSTGRES_HOST
        env = dict(os.environ)

    proc = None
    if replace or not expr_already_ran(to_file, debug=debug):

        try:
            proc = tee(
                # cmd=cmd + extra_argv,
                cmd=cmd,
                to_file=to_file,
                cwd=cwd,
                env=env,
                dry_run=dry_run,
            )
            failed = False
        except subprocess.CalledProcessError as e:
            if not skip_error:
                logger.info((
                                 "> Command failed: see {path}; exiting early "
                                 "(use --skip-error to ignore individual experiment errors)"
                             ).format(path=to_file))
                ret = 1
                if debug:
                    logger.info("Exiting with ret={ret}\n{stack}".format(
                        ret=ret,
                        stack=get_stacktrace(),
                    ))
                sys.exit(ret)
            logger.info(
                "> Command failed; see {path}; continuing (--skip-error was set)".format(
                    path=to_file,
                ))
            failed = True

        if not failed:
            if not dry_run and proc.returncode != 0:
                logger.info("BUG: saw returncode = {ret}, expected 0".format(
                    ret=proc.returncode))
                assert proc.returncode == 0
            if not dry_run:
                with open(to_file, 'a') as f:
                    f.write("{success_line}\n".format(success_line=EXPERIMENT_SUCCESS_LINE))
            if not dry_run:
                assert expr_already_ran(to_file, debug=debug)

    return proc

def expr_already_ran(to_file, debug=False):
    if not _e(to_file):
        return False
    with open(to_file) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip()
            if re.search(r'{success_line}'.format(success_line=EXPERIMENT_SUCCESS_LINE), line):
                if debug:
                    logger.info("Saw \"{success_line}\" in {path} @ line {lineno}; skipping.".format(
                        success_line=EXPERIMENT_SUCCESS_LINE,
                        lineno=lineno,
                        path=to_file))
                return True
    return False

def test_tee():
    def test_tee_01():
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="test_tee_")
        tmp_f = os.fdopen(tmp_fd, "w")
        tmp_f.close()
        try:
            with open(tmp_path, 'w') as f:
                # p = tee(['ls', '-l'], to_file=f)
                # p = tee(['bash', '-c', 'while true; do sleep 1; date; done'], to_file=f)
                p = tee(['bash', '-c', 'for i in `seq 1 1000`; do echo $i; done; echo "DONE TEST";'], to_file=f)
            with open(f.name) as readf:
                lines = readf.readlines()
                pprint.pprint({'lines':lines})
                has_done = any(re.search(r'DONE TEST', line) for line in lines)
                assert has_done
        finally:
            if _e(tmp_path):
                os.remove(tmp_path)
    test_tee_01()

    def test_tee_02():
        for i in range(100):
            test_tee_01()
    test_tee_02()
