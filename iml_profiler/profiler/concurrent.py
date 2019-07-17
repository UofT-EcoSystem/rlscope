import logging
import traceback
import re
import pytest
import sys
import time
import multiprocessing

from iml_profiler.profiler import iml_logging

MP_SPAWN_CTX = multiprocessing.get_context('spawn')

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
        # self.inactive_workers = []

    def _join_finished_workers(self):
        i = 0
        while i < len(self.active_workers):
            if not self.active_workers[i].is_alive():
                if self.debug:
                    logging.info("> Pool(name={name}): Join pid={pid} active_workers[i={i}]={proc}".format(
                        name=self.name,
                        i=i,
                        pid=self.active_workers[i].pid,
                        proc=self.active_workers[i],
                    ))
                self._join_child(self.active_workers[i])
                # self.inactive_workers.append(self.active_workers[i])
                del self.active_workers[i]
            else:
                i += 1

    def _wait_for_workers(self):
        while True:
            self._join_finished_workers()
            if self.max_workers == ForkedProcessPool.UNLIMITED or len(self.active_workers) < self.max_workers:
                break
            if self.debug:
                logging.info("> Pool.sleep for {sec} sec".format(
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
            logging.info("> Pool(name={name}): submit pid={pid}, proc={proc}".format(
                name=self.name,
                pid=proc.pid,
                proc=proc))
        self.active_workers.append(proc)

    def shutdown(self):
        for proc in self.active_workers:
            if self.debug:
                logging.info("> Pool(name={name}): Join pid={pid}, proc={proc}".format(
                    pid=proc.pid,
                    name=self.name,
                    proc=proc))
            self._join_child(proc)
            if self.debug:
                logging.info("> Pool(name={name}): Joined pid={pid}, proc={proc}".format(
                    pid=proc.pid,
                    name=self.name,
                    proc=proc))

    def _join_child(self, proc):
        proc.join()

        # If child process raised an exception, re-raise it in the parent.
        if proc.exception:
            ex, tb_string = proc.exception
            ExceptionKlass = type(ex)
            parent_ex = ExceptionKlass(("Exception in child process pid={pid} proc={proc} of ForkedProcessPool(name={name}): {error}\n"
                                        "{tb}").format(
                name=self.name,
                pid=proc.pid,
                proc=proc,
                error=ex.args[0],
                tb=proc.exception[1]))
            raise parent_ex

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
            parent_ex = RuntimeError(msg)
            raise parent_ex


class MyProcess(MP_SPAWN_CTX.Process):
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
        logging.info(traceback)



    From:
    https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pconn, self._cconn = MP_SPAWN_CTX.Pipe()
        self._exception = None

    def run(self):
        try:
            # We need to do this otherwise log messages don't appear (on some machines).
            # Not sure why...
            iml_logging.setup_logging()
            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # Re-raise exception so that proc.exitcode
            # still gets set to non-zero.
            raise

    @property
    def exception(self):
        if self._exception is not None:
            return self._exception

        if self._pconn.poll():
            self._exception = self._pconn.recv()

        return self._exception

def _sys_exit_1():
    """
    For unit-testing.
    """
    logging.info("Running child in ForkedProcessPool that exits with sys.exit(1)")
    sys.exit(1)

def _exception():
    """
    For unit-testing.
    """
    logging.info("Running child in ForkedProcessPool that raises an exception")
    raise RuntimeError("Child process exception")

def test_forked_process_pool():

    def test_01_sys_exit_1():
        """
        Check that processes that exit with sys.exit(1) are detected.

        :return:
        """
        pool = ForkedProcessPool(name=test_01_sys_exit_1.__name__)
        with pytest.raises(RuntimeError, match=ForkedProcessPool.NONZERO_EXIT_STATUS_ERROR_REGEX) as exec_info:
            pool.submit(_sys_exit_1.__name__, _sys_exit_1)
            pool.shutdown()
    test_01_sys_exit_1()

    def test_02_exception():
        """
        Check that processes that exit with sys.exit(1) are detected.

        :return:
        """
        pool = ForkedProcessPool(name=test_02_exception.__name__)
        with pytest.raises(RuntimeError, match="Child process exception") as exec_info:
            pool.submit(_exception.__name__, _exception)
            pool.shutdown()
    test_02_exception()


