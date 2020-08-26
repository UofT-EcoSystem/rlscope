from iml_profiler.profiler.iml_logging import logger
import contextlib
import sys
import multiprocessing
import multiprocessing.managers
import inspect
import functools

from iml_profiler.parser.common import *

from iml_profiler.protobuf.pyprof_pb2 import CategoryEventsProto, Event

from iml_profiler.profiler import proto_util

# https://stackoverflow.com/questions/9386636/profiling-a-system-with-extensively-reused-decorators
import types

MICROSECONDS_IN_SECOND = float(1e6)

from iml_profiler.profiler import wrap_util
from iml_profiler import py_config
from iml_profiler.profiler.util import get_stacktrace

from iml_profiler.clib import sample_cuda_api

DEFAULT_PREFIX = "CLIB__"

# 39870 events in CategoryEventsProto ~ 1.6M
PROTO_MAX_PYPROF_PY_EVENTS = 40000

# Store PyprofTrace's in a shared-memory process to speedup async dumping of PyprofTrace's.
# In particular, each PyprofTrace.record_event instead results in a remote method call.
#
# NOTE: we DON'T use this, since I noticed it significantly slows down training with DQN + Atari.
# i.e. speeding up async dumping delays come at the cost of slowing down trace-collection.
USE_PROXY_PYPROF_TRACE = False

# PROBLEM: We're only counting the number C++ API calls;
# we AREN'T counting python events being collected via the pyprof profiler!
# Q: Why do we need to record a python event for the C++ API call;
#    shouldn't it be counted by the pyprof profiler?
# A: It looks like we AREN'T actually using the pyprof profiler;
#    instead we are strictly using python timings recorded here...
#    that's correct.
#
# NOTE:
# Currently, we prefer NOT to collect all the pyprof profiler events
# (i.e. individual python function calls) since it's a LOT more data,
# Hence (I would suspect but have not measured) using pyprof profiler
# would lead to more profiling overhead.
# Of course, we cannot provide fine-grained python-code information
# (e.g. FlameGraph) with this approach.
# For our current purposes of isolating python time to "operations",
# this is "good enough".

class RecordEventHook:
    """
    Interface for adding hooks for when a TensorFlow C++ API event / Python event is recorded.
    """
    def __init__(self):
        pass

    # Can't see a reason for providing this.
    #
    # def before_record_event(self, pyprof_trace, event):
    #     pass

    def after_record_event(self, pyprof_trace, event):
        pass

RECORD_EVENT_HOOKS = []
def register_record_event_hook(hook : RecordEventHook):
    RECORD_EVENT_HOOKS.append(hook)
def unregister_record_event_hook(hook : RecordEventHook):
    RECORD_EVENT_HOOKS.remove(hook)


#
# Module globals.
#
# _pyprof_trace = mk_PyprofTrace()
# _step = None
# _process_name = None
# _phase = None
# Q: Should we initialize this to now_us()...?
# By default tracing is OFF.
_TRACING_ON = False
# print("LOADING clib_wrap: _TRACING_ON = {val}".format(val=_TRACING_ON))

# Just measure small aspects of pyprof trace-collection overhead.
_PYROF_TRACE_FULLY_ENABLED = True

class _ProfilingOverheadTracker:
    def __init__(self):
        self.profiling_overhead_us = 0
        self.start_t = None
        self.end_t = None
        self.started = False

    def start(self, start_t=None):
        assert not self.started
        if start_t is None:
            start_t = now_us()
        self.start_t = start_t
        self.started = True

    def end(self):
        assert self.started
        self.end_t = now_us()
        self.profiling_overhead_us = self.profiling_overhead_us + ( self.end_t - self.start_t )
        self.started = False

    def get_overhead_us(self):
        assert not self.started
        start_profiling_overhead_us = self.start_t
        duration_profiling_overhead_us = self.profiling_overhead_us

        # To ensure subsequent recordings record 0 profiling overhead.
        self.profiling_overhead_us = 0
        self.start_t = 0

        self.end_t = 0
        return start_profiling_overhead_us, duration_profiling_overhead_us


ProfilingOverheadTracker = _ProfilingOverheadTracker()

class CFuncWrapper:
    def __init__(self, func, category, prefix=DEFAULT_PREFIX, debug=False):
        # NOTE: to be as compatible as possible with intercepting existing code,
        # we forward setattr/getattr on this object back to the func we are wrapping
        # (e.g. func might be some weird SWIG object).
        super().__setattr__('func', func)
        super().__setattr__('prefix', prefix)
        super().__setattr__('category', category)
        super().__setattr__('debug', debug)

        name = self.wrapper_name(func.__name__)
        if py_config.DEBUG_WRAP_CLIB:
            logger.info("> call.name = {name}".format(name=name))

        def call(*args, **kwargs):
            # NOTE: for some reason, if we don't use a local variable here,
            # it will return None!  Bug in python3...?
            if self.debug:
                logger.info("call: {name}".format(name=name))
            ret = self.func(*args, **kwargs)
            return ret

        #
        # If we comment this out, then "call" returns the gpu_sleep time...
        # Not sure WHY.
        #

        # c = call.func_code
        # Python3
        c = call.__code__

        # https://docs.python.org/3.0/whatsnew/3.0.html
        #
        # """
        # The function attributes named func_X have been renamed to use the __X__ form,
        # freeing up these names in the function attribute namespace for user-defined attributes.
        # To wit, func_closure, func_code, func_defaults, func_dict, func_doc, func_globals,
        # func_name were renamed to __closure__, __code__, __defaults__, __dict__, __doc__, __globals__,
        # __name__, respectively.
        # """

        new_code = types.CodeType(
            c.co_argcount, c.co_kwonlyargcount, c.co_nlocals, c.co_stacksize,
            c.co_flags, c.co_code, c.co_consts,
            c.co_names, c.co_varnames, c.co_filename,
            name, c.co_firstlineno, c.co_lnotab,
            c.co_freevars, c.co_cellvars
        )
        call = types.FunctionType(
            new_code, call.__globals__, name, call.__defaults__,
            call.__closure__)

        super().__setattr__('call', call)

    def wrapper_name(self, name):
        return "{prefix}{name}".format(
            prefix=self.prefix,
            name=name)

    def __call__(self, *args, **kwargs):
        name = self.func.__name__
        with CallStack.frame(category=self.category, name=name):
            ret = self.call(*args, **kwargs)
            return ret

    # def old_pyprof_call(self, *args, **kwargs):
    #     global _pyprof_trace, _python_start_us, _step, _TRACING_ON
    #
    #     start_us = now_us()
    #     ret = self.call(*args, **kwargs)
    #     end_us = now_us()
    #     start_profiling_overhead_us, duration_profiling_overhead_us = ProfilingOverheadTracker.get_overhead_us()
    #     ProfilingOverheadTracker.start(start_t=end_us)
    #
    #     # NOTE: profiling overhead.
    #     # Doing "extra stuff" in function-call wrappers for TensorFlow C++ API / Simulators is causing huge overheads
    #     # during pyprof tracing.
    #     # Even if the record_event() callbacks are just no-op function calls, we still experience the large overheads!
    #     if _PYROF_TRACE_FULLY_ENABLED and _TRACING_ON:
    #         name = self.func.__name__
    #
    #         # We are about to call from python into a C++ API.
    #         # That means we stopping executing python while C++ runs.
    #         # So, we must add a python execution and C++ execution event.
    #         #
    #         # [ last C++ call ][ python call ][ C++ call ]
    #         #                 |               |          |
    #         #         _python_start_us     start_us   end_us
    #         #
    #         # BUG: this FAILS when using tf.py_function.
    #         # That's because C++ calls BACK into Python, whereas we assumed it's always [ Python -> C++ ]
    #         # Effectively what happens is (NOTE the "!" lines are the unexpected parts of the execution):
    #         # Calls Python: collect_driver.run() ->
    #         # Calls C++: TFE_Py_Execute ->
    #         #   Intercept clib call: python_start_us_before_call = _python_start_us
    #         # ! Calls Python: tf.py_function callback into "step" ->
    #         # !  records Python "Finish benchmark" in iml.prof.operation("step"), updating _python_start_us
    #         # ! Calls Python: tf.py_function callback into "step" ->
    #         # ! Calls C++ Simulator: PyBullet.step ->
    #         # ! Return from C++ Simulator: PyBullet.step ->
    #         # !  Intercept clib return: _python_start_us = now_us()
    #         # ! Return from Python: tf.py_function callback into "step" ->
    #         # Return from C++: TFE_Py_Execute ->
    #         #   assertion failure: _python_start_us != python_start_us_before_call
    #         #   Intercept clib return: _python_start_us = now_us()
    #         # Return from Python: collect_driver.run() ->
    #         #
    #         # Q: What events do we WANT to record in this scenario?
    #         # [ Python: before collect_driver.run() ]
    #         # [ C++: TFE_Py_Execute before step() callback ]
    #         # [ Python: start step() callback ]
    #         # [ C++: Simulator ]
    #         # [ Python: end step() callback ]
    #         # [ C++: TFE_Py_Execute after step() callback ]
    #
    #         # NOTE: record python-side pyprof-related profiling overhead.
    #         # Profiling overhead being generated now will be recorded by the NEXT python event gets recorded;
    #         # We record profiling overhead for the event before this (profiling_overhead_us).
    #         #
    #         # NOTE: ProfilingOverheadTracker will ALSO end up capturing async dumping overhead,
    #         # since register dumping "hooks" get called at the end of PyprofTrace.record_event.
    #         _pyprof_trace.record_python_event(
    #             _pyprof_trace.get_step(), name,
    #             start_us=_python_start_us,
    #             end_us=start_us,
    #             start_profiling_overhead_us=start_profiling_overhead_us,
    #             duration_profiling_overhead_us=duration_profiling_overhead_us,
    #         )
    #         _pyprof_trace.record_event(
    #             _pyprof_trace.get_step(), self.category, name,
    #             start_us=start_us,
    #             end_us=end_us,
    #             debug=self.debug)
    #
    #     ProfilingOverheadTracker.end()
    #     _python_start_us = end_us
    #
    #     return ret

    def __setattr__(self, name, value):
        return setattr(self.func, name, value)

    def __getattr__(self, name):
        return getattr(self.func, name)

def record_python_event(name, end_us, ignore_disable=False):
    """
    Useful for recording the last amount of time in between returning
    from a call to q_forward, and finishing benchmarking.
    This will include time spent in the tensorflow python API
    (i.e. not doing C++ calls, just returning back to the benchmarking script).
    """

    # Record an event for the current active frame, but we don't want to pop it
    # (it's still active).  We want to specify the name of the event, and adjust the start_us
    # marker of the active frame.
    CallStack._record_event(
        end_us=end_us,
        name=name,
    )

def enable_tracing():
    global _TRACING_ON
    _TRACING_ON = True
    # global _python_start_us
    # _python_start_us = now_us()
    start_us = now_us()
    # if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
    if py_config.DEBUG:
        logger.debug(f"iml.enable_tracing(): python_start_us.new = {start_us}")
    # NOTE: name=None => name of python events inherit the name of the c-library call they are making.
    CallStack._entry_point(category=CATEGORY_PYTHON, name=None, start_us=start_us)

def disable_tracing():
    global _TRACING_ON
    _TRACING_ON = False
    if py_config.DEBUG:
        logger.info("Disable pyprof tracing: _TRACING_ON={val}\n{stack}".format(
            val=_TRACING_ON,
            stack=get_stacktrace()))
    CallStack._exit_point()

#
# Some pre-written C++ library wrappers.
#

class WrappedModule:
    def __init__(self, wrap_module, unwrap_module):
        self.wrap_module = wrap_module
        self.unwrap_module = unwrap_module

WRAPPED_MODULES = []
def register_wrap_module(wrap_module, unwrap_module):
    if _LIBS_WRAPPED:
        raise RuntimeError("IML ERROR: Registering module too late; you must call iml.register_wrap_module(...) right after calling iml.add_iml_arguments(...)")
    WRAPPED_MODULES.append(WrappedModule(wrap_module, unwrap_module))

_LIBS_WRAPPED = False
def wrap_libs():
    global _LIBS_WRAPPED
    if _LIBS_WRAPPED:
        return
    for wrapped_module in WRAPPED_MODULES:
        wrapped_module.wrap_module()
    _LIBS_WRAPPED = True
def unwrap_libs():
    global _LIBS_WRAPPED
    if not _LIBS_WRAPPED:
        return
    for wrapped_module in reversed(WRAPPED_MODULES):
        wrapped_module.unwrap_module()
    _LIBS_WRAPPED = False

SETUP_DONE = False
def setup(allow_skip=False):
    global SETUP_DONE
    if allow_skip and SETUP_DONE:
        return
    assert not SETUP_DONE

    register_detected_libs()
    wrap_libs()

    SETUP_DONE = True

def register_detected_libs():
    try:
        import atari_py
        register_wrap_module(wrap_atari, unwrap_atari)
    except ImportError:
        pass

    try:
        import tensorflow
        if is_tensorflow_v2():
            if py_config.DEBUG_WRAP_CLIB:
                logger.debug("Detected TensorFlow v2")
            register_wrap_module(wrap_tensorflow_v2, unwrap_tensorflow_v2)
        else:
            if py_config.DEBUG_WRAP_CLIB:
                logger.debug("Detected TensorFlow v1")
            register_wrap_module(wrap_tensorflow_v1, unwrap_tensorflow_v1)
    except ImportError:
        if py_config.DEBUG_WRAP_CLIB:
            logger.debug("TensorFlow is NOT installed")

    try:
        import pybullet
        register_wrap_module(wrap_pybullet, unwrap_pybullet)
    except ImportError:
        if py_config.DEBUG_WRAP_CLIB:
            logger.debug("PyBullet is NOT installed")


def wrap_tensorflow_v1(category=CATEGORY_TF_API, debug=False):
    logger.info("> IML: Wrapping module=tensorflow call with category={category} annotations".format(
        category=category,
    ))
    success = wrap_util.wrap_lib(
        CFuncWrapper,
        import_libname='tensorflow',
        wrap_libname='tensorflow.pywrap_tensorflow',
        wrapper_args=(category, DEFAULT_PREFIX, debug),
        func_regex='^TF_')
    assert success
def unwrap_tensorflow_v1():
    if py_config.DEBUG_WRAP_CLIB:
        logger.info("> IML: Unwrapping module=tensorflow")
    wrap_util.unwrap_lib(
        CFuncWrapper,
        import_libname='tensorflow',
        wrap_libname='tensorflow.pywrap_tensorflow')

def is_tensorflow_v2():
    import tensorflow as tf
    m = re.search(r'(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)', tf.__version__)
    return int(m.group('major')) >= 2

def wrap_tensorflow_v2(category=CATEGORY_TF_API, debug=False):
    logger.info("> IML: Wrapping module=tensorflow.python.pywrap_tfe call with category={category} annotations".format(
        category=category,
    ))
    success = wrap_util.wrap_lib(
        CFuncWrapper,
        import_libname='tensorflow.python.pywrap_tfe',
        wrap_libname='tensorflow.python.pywrap_tfe',
        wrapper_args=(category, DEFAULT_PREFIX, debug),
        # Q: Should we only wrap TFE_Py_Execute and TFE_Py_FastPathExecute?
        # func_regex='^TFE_',
        func_regex='^TFE_Py_(Execute|FastPathExecute)',
    )
    assert success
    success = wrap_util.wrap_lib(
        CFuncWrapper,
        import_libname='tensorflow.python.client.pywrap_tf_session',
        wrap_libname='tensorflow.python.client.pywrap_tf_session',
        wrapper_args=(category, DEFAULT_PREFIX, debug),
        func_regex='^TF_')
    assert success
def unwrap_tensorflow_v2():
    if py_config.DEBUG_WRAP_CLIB:
        logger.info("> IML: Unwrapping module=tensorflow.python.pywrap_tfe")
    wrap_util.unwrap_lib(
        CFuncWrapper,
        import_libname='tensorflow.python.pywrap_tfe',
        wrap_libname='tensorflow.python.pywrap_tfe')
    wrap_util.unwrap_lib(
        CFuncWrapper,
        import_libname='tensorflow.python.client.pywrap_tf_session',
        wrap_libname='tensorflow.python.client.pywrap_tf_session')

def wrap_atari(category=CATEGORY_ATARI):
    try:
        import atari_py
    except ImportError:
        return
    wrap_module(atari_py.ale_python_interface.ale_lib, category)
def unwrap_atari():
    try:
        import atari_py
    except ImportError:
        return
    wrap_util.unwrap_module(
        CFuncWrapper,
        atari_py.ale_python_interface.ale_lib)

def wrap_module(module, category, debug=False, **kwargs):
    if py_config.DEBUG_WRAP_CLIB:
        logger.info("> IML: Wrapping module={mod} call with category={category} annotations".format(
            mod=module,
            category=category,
        ))
    wrap_util.wrap_module(
        CFuncWrapper, module,
        wrapper_args=(category, DEFAULT_PREFIX, debug), **kwargs)
def unwrap_module(module):
    if py_config.DEBUG_WRAP_CLIB:
        logger.info("> IML: Unwrapping module={mod}".format(
            mod=module))
    wrap_util.unwrap_module(
        CFuncWrapper,
        module)


class LibWrapper:
    """
    Yet another way of wrapping a python module.
    Instead modifying the .so module object (wrap_module), we replace the entire py-module object
    with a wrapper.

    NOTE: Don't use this class directly, use iml.wrap_entire_module instead.

    Details:

    # Say you want to replace the pybullet library, which is a C++ library.

    # This imports pybullet as a shared-library .so file
    import pybullet

    # LibWrapper is a wrapper-class around the .so file
    import sys
    lib_wrapper = LibWrapper(pybullet)
    # Get rid of original import
    del sys.modules['pybullet']

    # Replace imported pybullet with wrapper;
    # when other run "import pybullet", the wrapper will be imported.
    sys.modules['pybullet'] = lib_wrapper

    ...
    import pybullet
    # Accessing some_func will return a CFuncWrapper around the original pybullet.some_func.
    pybullet.some_func()
    """
    def __init__(self, lib, category, debug=False, prefix=DEFAULT_PREFIX):
        self.lib = lib
        self.category = category
        self.prefix = prefix
        self.debug = debug

    def __getattr__(self, name):
        """
        __getattr__ gets called is LibWrapper.name didn't exist.
        In that case, they're probably trying to call a function from the .so lib (self.lib).
        We wrap the function with CFuncWrapper(lib[name]), and set it as:
          LibWrapper[name] = CFuncWrapper(lib[name])
        Subsequent calls to lib[name] WON'T come through here (since LibWrapper[name] is set).

        :param name:
          Name of a .so function they're trying to call (probably, could be constant though).
        :return:
        """
        func = getattr(self.lib, name)
        if not callable(func):
            return func

        if self.debug:
            logger.info("Wrap: {name}".format(name=name))
        func_wrapper = CFuncWrapper(func, self.category, self.prefix, self.debug)
        setattr(self, name, func_wrapper)
        return func_wrapper

def wrap_entire_module(import_libname, category, debug=False, **kwargs):
    if py_config.DEBUG_WRAP_CLIB:
        logger.info("> IML: Wrapping module={mod} call with category={category} annotations".format(
            mod=import_libname,
            category=category,
        ))
    exec("import {import_lib}".format(import_lib=import_libname))
    wrap_libname = import_libname
    lib = eval("{wrap_lib}".format(wrap_lib=wrap_libname))
    assert lib is not None
    if import_libname in sys.modules:
        del sys.modules[import_libname]
    lib_wrapper = LibWrapper(lib, category, debug, **kwargs)
    # "import pybullet" will now return LibWrapper(pybullet)
    sys.modules[import_libname] = lib_wrapper
def unwrap_entire_module(import_libname):
    if import_libname not in sys.modules:
        return
    if py_config.DEBUG_WRAP_CLIB:
        logger.info("> IML: Unwrapping module={mod}".format(
            mod=import_libname))
    lib_wrapper = sys.modules[import_libname]
    sys.modules[import_libname] = lib_wrapper.lib

#
# Wrap PyBullet.
#

pybullet = None
try:
    import pybullet
    import pybullet_envs.bullet.bullet_client
    import pybullet_envs.minitaur.envs.bullet_client
    import pybullet_utils.bullet_client
except ImportError as e:
    logger.warning("IML: Failed to import PyBullet ({error}), skip wrapping library".format(
        error=str(e)))
    pass

def wrap_pybullet():

    # iml.wrap_entire_module(
    #     'pybullet',
    #     category=CATEGORY_SIMULATOR_CPP,
    #     debug=True)
    # _wrap_bullet_clients()

    def should_wrap(name, func):
        return inspect.isbuiltin(func)
    wrap_module(
        pybullet, category=CATEGORY_SIMULATOR_CPP,
        should_wrap=should_wrap,
        # debug=True,
    )
    _wrap_bullet_clients()

def unwrap_pybullet():

    # _unwrap_bullet_clients()
    # iml.unwrap_entire_module('pybullet')

    _unwrap_bullet_clients()
    unwrap_module(pybullet)

#
# IML: pybullet specific work-around!
#
# pybullet python library does some weird dynamic stuff when accessing shared-library functions.
# Basically BulletClient class is checking whether a function that's getting fetched is a built-in.
# If it is, then an extra physicsClientId argument is being given to it.
# So, when we manually wrap this library, the inspect.isbuiltin check will FAIL, and physicsClientId WON'T get supplied!
# So, to work around this, we must also wrap the BulletClient class, and forward physicsClientId.

OldBulletClients = dict()
# pybullet has 3 different implementations of the BulletClient class that essentially look the same.
def _wrap_bullet_clients():
    OldBulletClients['pybullet_envs.bullet.bullet_client.BulletClient'] = pybullet_envs.bullet.bullet_client.BulletClient
    OldBulletClients['pybullet_envs.minitaur.envs.bullet_client.BulletClient'] = pybullet_envs.minitaur.envs.bullet_client.BulletClient
    OldBulletClients['pybullet_utils.bullet_client.BulletClient'] = pybullet_utils.bullet_client.BulletClient
    pybullet_envs.bullet.bullet_client.BulletClient = MyBulletClient
    pybullet_envs.minitaur.envs.bullet_client.BulletClient = MyBulletClient
    pybullet_utils.bullet_client.BulletClient = MyBulletClient
def _unwrap_bullet_clients():
    pybullet_envs.bullet.bullet_client.BulletClient = OldBulletClients['pybullet_envs.bullet.bullet_client.BulletClient']
    pybullet_envs.minitaur.envs.bullet_client.BulletClient = OldBulletClients['pybullet_envs.minitaur.envs.bullet_client.BulletClient']
    pybullet_utils.bullet_client.BulletClient = OldBulletClients['pybullet_utils.bullet_client.BulletClient']

class MyBulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=pybullet.DIRECT, options=""):
        """Create a simulation and connect to it."""
        self._client = pybullet.connect(pybullet.SHARED_MEMORY)
        if (self._client < 0):
            # print("options=", options)
            self._client = pybullet.connect(connection_mode, options=options)
        self._shapes = {}

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if (
            inspect.isbuiltin(attribute) or
            ( isinstance(attribute, CFuncWrapper) and inspect.isbuiltin(attribute.func) )
        ) and name not in [
            "invertTransform",
            "multiplyTransforms",
            "getMatrixFromQuaternion",
            "getEulerFromQuaternion",
            "computeViewMatrixFromYawPitchRoll",
            "computeProjectionMatrixFOV",
            "getQuaternionFromEuler",
        ]:  # A temporary hack for now.
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class CallFrame:
    def __init__(self, category, name, start_us):
        self.name = name
        self.category = category
        self.start_us = start_us

class CallStackEntry:
    def __init__(self, category, name, start_us=None):
        self.category = category
        self.name = name
        self.start_us = start_us

    def __enter__(self):
        CallStack._entry_point(category=self.category, name=self.name, start_us=self.start_us)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        CallStack._exit_point()


class _CallStack:
    """
    Track Python -> C -> Python -> ... language transitions.

    We maintain a "callstack" of (Category=[Python | C], start_us) entry point stack.
    When we enter Python->C (Simulator, tensorflow API call), we push to it.
    When we enter C->Python (tf.numpy_function) we push to it.
    When we return Python->C we pop from it.
    Events are recorded during both entry and exit.

    Usage:

    with CallStack.frame(category=CATEGORY_SIMULATOR, name="step"):
        ...call native library...
    """
    def __init__(self):
        self.frames = []

    def frame(self, category, name, start_us=None):
        return CallStackEntry(category, name, start_us=start_us)

    def _entry_point(self, category, name, start_us=None):
        if start_us is None:
            start_us = now_us()
        if len(self) > 0:
            last_frame = self.frames[-1]
            if last_frame.name is None:
                record_name = name
            else:
                record_name = last_frame.name
            if _TRACING_ON:
                sample_cuda_api.record_event(
                    category=last_frame.category,
                    start_us=last_frame.start_us,
                    duration_us=start_us - last_frame.start_us,
                    name=record_name)

        self.frames.append(CallFrame(category, name, start_us))

    def _exit_point(self):
        # NOTE: This shouldn't fail; if it does its a BUG (exit-point without an entry-point).
        assert len(self.frames) > 0
        last_frame = self.frames.pop()
        start_us = now_us()
        if _TRACING_ON:
            sample_cuda_api.record_event(
                category=last_frame.category,
                start_us=last_frame.start_us,
                duration_us=start_us - last_frame.start_us,
                name=last_frame.name)
        if len(self.frames) > 0:
            self.frames[-1].start_us = start_us

    def _record_event(self, end_us, name=None):
        """
        Record an event for the currently active frame, and advance the start_us of the active (last) frame.
        :param end_us:
            End time of event to record.
            end_us must be >= start_us of current active frame.
        :param name:
            Name of event to record if active frame is un-named.
        :return:
        """
        assert len(self.frames) > 0
        last_frame = self.frames[-1]
        assert last_frame.start_us <= end_us
        if last_frame.name is None:
            record_name = name
        else:
            record_name = last_frame.name
        sample_cuda_api.record_event(
            category=last_frame.category,
            start_us=last_frame.start_us,
            duration_us=end_us - last_frame.start_us,
            name=record_name,
        )

    def __index__(self, idx):
        return self.frames[idx]

    def __len__(self):
        return len(self.frames)

CallStack = _CallStack()

