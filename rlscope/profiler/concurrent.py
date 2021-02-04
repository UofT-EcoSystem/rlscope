"""
Process pool implementations that are easy to debug in single-threaded mode,
and have friendlier behaviour during exceptions in child processes.
"""
from rlscope.profiler.rlscope_logging import logger
import os
import shutil
import traceback
import re
import sys
import time
import multiprocessing

import progressbar

from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.rlscope_logging import logger

from rlscope.parser.common import *

MP_SPAWN_CTX = multiprocessing.get_context('spawn')
MP_FORK_CTX = multiprocessing.get_context('fork')
MP_CTX = MP_SPAWN_CTX
# MP_CTX = MP_FORK_CTX

class FailedProcessException(Exception):
    def __init__(self, msg, proc, exitcode):
        super().__init__(msg)
        self.proc = proc
        self.exitcode = exitcode

class ProcessPoolExecutorWrapper:
    def __init__(self, name=None, max_workers=None, debug=False, cpu_affinity=None):
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        self.name = name
        self.debug = debug
        self.cpu_affinity = cpu_affinity
        self.pending_results = []

    def submit(self, name, fn, *args, sync=False, **kwargs):
        if sync:
            return fn(*args, **kwargs)
        self._wait_for_workers()
        fut = self._pool.submit(fn, *args, **kwargs)
        self.pending_results.append(fut)
        return fut

    def _wait_for_workers(self, block=False, show_progress=False):
        if block:

            bar = None
            num_results = len(self.pending_results)
            if show_progress and num_results > 0:
                bar = progressbar.ProgressBar(max_value=num_results)

            num_done = 0
            while len(self.pending_results) > 0:
                done, not_done = concurrent.futures.wait(self.pending_results, return_when=concurrent.futures.FIRST_COMPLETED)
                self.pending_results = list(not_done)
                num_done += len(done)
                if bar is not None:
                    bar.update(num_done)
                for result in done:
                    # Might re-raise exception.
                    result.result()
                    # if isinstance(result, Exception):
                    #     raise result
            if bar is not None:
                bar.finish()
            return

        i = 0
        keep = []
        while i < len(self.pending_results):
            if self.pending_results[i].done():
                # Might re-raise exception.
                self.pending_results[i].result()
            else:
                keep.append(self.pending_results[i])
            i += 1
        self.pending_results = keep

    def shutdown(self, ignore_exceptions=False, show_progress=False):
        if ignore_exceptions:
            self._pool.shutdown(wait=True)
        else:
            self._wait_for_workers(block=True, show_progress=show_progress)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wait_for_workers(block=True)
        return False


class ForkedProcessPool:
    """
    Fork a child and run "target".

    NOTE: We make this class explicit, since ProcessPoolExecutor DOESN'T fork a
    process on every call to pool.submit(...)

    TODO: add test-case to make sure child processes that fail with sys.exit(1) are detected
    (I don't think they current are...)
    """
    WAIT_WORKERS_TIMEOUT_SEC = 0.01
    UNLIMITED = 0
    # Used for unit-testing; regex should match exception message when a child does sys.exit(1)
    NONZERO_EXIT_STATUS_ERROR_REGEX = r''

    def __init__(self, name=None, max_workers=UNLIMITED, debug=False, cpu_affinity=None):
        self.name = name
        self.process = None
        self.max_workers = max_workers
        self.cpu_affinity = cpu_affinity
        self.active_workers = []
        self.debug = debug
        # self.ctx = multiprocessing.get_context('spawn')
        # self.inactive_workers = []

    def _join_finished_workers(self):
        i = 0
        while i < len(self.active_workers):
            if not self.active_workers[i].is_alive():
                self._join_child(i)
            else:
                i += 1

    def _wait_for_workers(self):
        while True:
            self._join_finished_workers()
            if self.max_workers == ForkedProcessPool.UNLIMITED or len(self.active_workers) < self.max_workers:
                break
            if self.debug:
                logger.info("> Pool.sleep for {sec} sec".format(
                    sec=ForkedProcessPool.WAIT_WORKERS_TIMEOUT_SEC))
            time.sleep(ForkedProcessPool.WAIT_WORKERS_TIMEOUT_SEC)

    def submit(self, name, fn, *args, sync=False, **kwargs):
        if sync:
            return fn(*args, **kwargs)
        self._wait_for_workers()
        # proc = multiprocessing.Process(target=fn, name=name, args=args, kwargs=kwargs)
        # def fn_wrapper(*args, **kwargs):
        #     if self.cpu_affinity is not None:
        #         proc = psutil.Process(pid=os.getpid())
        #         proc.cpu_affinity(self.cpu_affinity)
        #     return fn(*args, **kwargs)
        # proc = MyProcess(target=fn_wrapper, name=name, args=args, kwargs=kwargs)
        proc = MyProcess(target=fn, name=name, args=args, kwargs=kwargs)
        proc.start()
        if self.debug:
            logger.info("> Pool(name={name}): submit pid={pid}, proc={proc}".format(
                name=self.name,
                pid=proc.pid,
                proc=proc))
        self.active_workers.append(proc)

    def shutdown(self):
        while len(self.active_workers) > 0:
            self._join_child(0)

    def _join_child(self, i):
        if self.debug:
            logger.info("> Pool(name={name}): Join pid={pid} active_workers[i={i}]={proc}".format(
                name=self.name,
                i=i,
                pid=self.active_workers[i].pid,
                proc=self.active_workers[i],
            ))

        proc = self.active_workers[i]
        del self.active_workers[i]

        # NOTE: MyProcess, unlike multiprocessing.Process, will re-raise the child-process exception
        # in the parent during a join.
        proc.join()

        # If child process did sys.exit(1), raise a RuntimeError in the parent.
        # (if we don't this, we'll end up ignoring a failed child!)
        if proc.exitcode is not None and proc.exitcode > 0:
            msg = (
                "Child process pid={pid} proc={proc} of ForkedProcessPool(name={name}) "
                "exited with non-zero exit-status: sys.exit({ret}).\n"
            ).format(
                name=self.name,
                pid=proc.pid,
                proc=proc,
                ret=proc.exitcode,
            )
            assert re.search(ForkedProcessPool.NONZERO_EXIT_STATUS_ERROR_REGEX, msg)
            parent_ex = FailedProcessException(msg, proc=proc, exitcode=proc.exitcode)
            raise parent_ex


class MyProcess(MP_CTX.Process):
    """
    Record exceptions in child process, and make them accessible
    in the parent process after the child exits.

    e.g.

    def target():
        raise ValueError('Something went wrong...')

    p = Process(target=target)
    p.start()
    p.join()

    if p.exception:
        error, traceback = p.exception
        logger.info(traceback)



    From:
    https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = MP_CTX.Pipe()
        self._exception = None

    def run(self):
        try:
            # We need to do this otherwise log messages don't appear (on some machines).
            # Not sure why...
            super().run()
            self._cconn.send(None)
            self._cconn.close()
            self._pconn.close()
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            self._cconn.close()
            self._pconn.close()
            # Re-raise exception so that proc.exitcode
            # still gets set to non-zero.
            raise

    def join(self):
        super().join()

        # NOTE: querying for exception also forces self._pconn to close
        # (need to do this to avoid hitting POSIX open file limit!)
        exception = self.exception
        if exception:
            ex, tb_string = exception
            ExceptionKlass = type(ex)
            parent_ex = ExceptionKlass(("Exception in child process pid={pid} proc={proc} of ForkedProcessPool(name={name}): {error}\n"
                                        "{tb}").format(
                name=self.name,
                pid=self.pid,
                proc=self,
                error=ex.args,
                tb=tb_string))
            raise parent_ex

    @property
    def exception(self):
        if self._exception is not None:
            return self._exception

        if self._pconn.poll():
            self._exception = self._pconn.recv()
            self._pconn.close()
            self._cconn.close()

        return self._exception


def map_pool(pool, func, kwargs_list, desc=None, show_progress=False, sync=False):
    if len(kwargs_list) == 1 or sync:
        # logger.info("Running map_pool synchronously")
        # Run it on the current thread.
        # Easier to debug with pdb.
        results = []
        for kwargs in kwargs_list:
            result = func(kwargs)
            results.append(result)
        return results

    # logger.info("Running map_pool in parallel with n_workers")
    results = []
    for i, result in enumerate(progress(
        pool.map(func, kwargs_list),
        desc=desc,
        total=len(kwargs_list),
        show_progress=show_progress)
    ):
        results.append(result)
    return results

def _sys_exit_1():
    """
    For unit-testing.
    """
    logger.info("Running child in ForkedProcessPool that exits with sys.exit(1)")
    sys.exit(1)

class MadeupForkedProcessException(Exception):
    pass

def _exception():
    """
    For unit-testing.
    """
    logger.info("Running child in ForkedProcessPool that raises an exception")
    raise MadeupForkedProcessException("Child process exception")

def _do_nothing():
    pass

def _check_dir_exists(path):
    if not os.path.isdir(path):
        print("FAIL: path={path} doesn't exist".format(path=path))
    assert os.path.isdir(path)

class TestForkedProcessPool:

    def test_01_sys_exit_1(self):
        """
        Check that processes that exit with sys.exit(1) are detected.

        :return:
        """
        pool = ForkedProcessPool(name=self.test_01_sys_exit_1.__name__)
        import pytest
        with pytest.raises(FailedProcessException, match=ForkedProcessPool.NONZERO_EXIT_STATUS_ERROR_REGEX) as exec_info:
            pool.submit(_sys_exit_1.__name__, _sys_exit_1)
            pool.shutdown()

    def test_02_exception(self):
        """
        Check that processes that exit with sys.exit(1) are detected.

        :return:
        """
        pool = ForkedProcessPool(name=self.test_02_exception.__name__)
        import pytest
        with pytest.raises(MadeupForkedProcessException, match="Child process exception") as exec_info:
            pool.submit(_exception.__name__, _exception)
            pool.shutdown()

    def test_03_mkdir(self):
        """
        Test that directory created in parent process is seen in child process.

        :return:
        """
        TEST_DIR = './rlscope_test_data.test_03_mkdir'
        logger.info("TEST_DIR = {path}".format(path=_a(TEST_DIR)))

        def get_dir_path(i):
            return _j(TEST_DIR, 'dir_{i}'.format(i=i))

        num_dirs = 100
        if _e(TEST_DIR):
            shutil.rmtree(TEST_DIR)
        # debug = True
        debug = False
        pool = ForkedProcessPool(name=self.test_03_mkdir.__name__, debug=debug)
        for i in range(num_dirs):
            path = get_dir_path(i)
            os.makedirs(path, exist_ok=True)
            name = "{func}(i={i})".format(func=self.test_03_mkdir.__name__, i=i)
            pool.submit(name, _check_dir_exists, path)
        pool.shutdown()
        shutil.rmtree(TEST_DIR)

    # This test takes a while...
    #
    # try:
    #     # resource module not supported on Windows
    #     import resource
    #     run_test_04_file_limit = True
    # except ImportError:
    #     run_test_04_file_limit = False
    # def test_04_file_limit(self):
    #     """
    #     Test MyProcess doesn't induce open-file limit by creating
    #     multiprocessing.Pipe() objects.
    #
    #     :return:
    #     """
    #     import resource
    #     test_name = test_04_file_limit.__name__
    #     logger.info("Running {test}; this may take a minute...".format(
    #         test=test_name))
    #     soft_file_limit, hard_file_limit = resource.getrlimit(resource.RLIMIT_OFILE)
    #     # debug = True
    #     debug = False
    #     pool = ForkedProcessPool(name=test_name, debug=debug)
    #     for i in progressbar.progressbar(range(soft_file_limit + 10), prefix=test_name):
    #         pool.submit(_do_nothing.__name__, _do_nothing)
    #     pool.shutdown()
    #     # We reached here, test passed.
    # if run_test_04_file_limit:
    #     test_04_file_limit()
