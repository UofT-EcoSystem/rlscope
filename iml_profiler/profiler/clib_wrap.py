import logging
import contextlib
import sys
import multiprocessing
import multiprocessing.managers

from iml_profiler.parser.common import *

from iml_profiler.protobuf.pyprof_pb2 import Pyprof, Event

from iml_profiler.profiler import proto_util

# https://stackoverflow.com/questions/9386636/profiling-a-system-with-extensively-reused-decorators
import types

MICROSECONDS_IN_SECOND = float(1e6)

from iml_profiler.profiler import wrap_util
from iml_profiler import py_config

DEFAULT_PREFIX = "CLIB__"

# 39870 events in Pyprof ~ 1.6M
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

class PyprofTrace:
    def __init__(self):
        self.pyprof = Pyprof()
        self._num_events = 0

    def finish(self, process_name, phase):
        self.pyprof.process_name = process_name
        self.pyprof.phase = phase

    def get_step(self):
        return self._step
    def get_num_events(self):
        return self._num_events

    def set_step(self, step):
        if step is None:
            return
        self._step = step
        if step not in self.pyprof.steps:
            # if py_config.DEBUG:
            #     logging.info("> ADD PYPROF STEP: {s}".format(s=self._step))

            self.pyprof.steps.extend([step])

            if py_config.DEBUG:
                pprint.pprint({
                    'len(self.pyprof.steps)':len(self.pyprof.steps),
                }, indent=2)

    def dump(self, path, process_name, phase):
        self.pyprof.process_name = process_name
        self.pyprof.phase = phase

        with open(path, 'wb') as f:
            # logging.info("> dump pyprof.steps:")
            # pprint.pprint({
            #     'len(pyprof.steps)':len(self.pyprof.steps),
            #     'pyprof.process_name':self.pyprof.process_name,
            #     'pyprof.phase':self.pyprof.phase,
            # }, indent=2)
            f.write(self.pyprof.SerializeToString())

    def record_event(self, step, category, name, start_us, end_us, attrs=None, python_event=False, debug=False):
        assert step is not None

        event = proto_util.mk_event(
            name=name,
            start_us=start_us,
            end_us=end_us,
            attrs=attrs)

        if debug:
            logging.info("Record event: name={name}, category={cat}, duration={ms} ms".format(
                name=name,
                cat=category,
                ms=(end_us - start_us)*1e3,
            ))

        # NOTE: extend() makes a copy of everything we add, but it's more familiar so who cares.
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-message-fields
        if python_event:
            self.pyprof.python_events[step].events.extend([event])
        else:
            self.pyprof.clibs[step].clibs[category].events.extend([event])

        self._num_events += 1

        # Call any RecordEvent callbacks
        # (e.g. you could register a hook to dump this PyprofTrace
        # when the number of events exceeds a threshold)
        for hook in RECORD_EVENT_HOOKS:
            hook.after_record_event(pyprof_trace=self, event=event)

    def record_python_event(self, step, name, start_us, end_us, ignore_disable=False):
        """
        Useful for recording the last amount of time in between returning
        from a call to q_forward, and finishing benchmarking.
        This will include time spent in the tensorflow python API
        (i.e. not doing C++ calls, just returning back to the benchmarking script).
        """
        self.record_event(step, CATEGORY_PYTHON, name, start_us, end_us,
                          python_event=True)

class PyprofDumpManager:
    """
    NOTE: In the future, we could make pyprof_trace a Proxy object itself to avoid serialization
    during DumpManager.put.
    """
    # def __init__(self):
    def __init__(self, manager):
        # self._manager = multiprocessing.Manager()
        # self.pyprof_traces = self._manager.dict()
        # self.lock = self._manager.Lock()

        self.pyprof_traces = manager.dict()
        self.lock = manager.Lock()

    def put(self, key, pyprof_trace):
        with self.lock:
            self.pyprof_traces[key] = pyprof_trace
            # self.pyprof_traces.append(pyprof_trace)

    def get(self, key):
        with self.lock:
            pyprof_trace = self.pyprof_traces[key]
            del self.pyprof_traces[key]
            # if len(self.pyprof_traces) == 0:
            #     return None
            # pyprof_trace = self.pyprof_traces.get()
        return pyprof_trace

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('PyprofTrace', PyprofTrace)
if USE_PROXY_PYPROF_TRACE:
    _manager = MyManager()
    _manager.start()

def mk_PyprofTrace():
    if USE_PROXY_PYPROF_TRACE:
        return _manager.PyprofTrace()
    else:
        return PyprofTrace()

#
# Module globals.
#
_pyprof_trace = mk_PyprofTrace()
_step = None
_process_name = None
_phase = None
# Q: Should we initialize this to now_us()...?
_python_start_us = None
# By default tracing is OFF.
_TRACING_ON = False
# print("LOADING clib_wrap: _TRACING_ON = {val}".format(val=_TRACING_ON))

def clear_pyprof_profiling():
    global _pyprof_trace, _python_start_us, _process_name
    _pyprof_trace = mk_PyprofTrace()
    _pyprof_trace.set_step(_step)
    _python_start_us = now_us()
def get_pyprof_trace():
    global _pyprof_trace, _process_name, _phase
    trace = _pyprof_trace
    trace.finish(_process_name, _phase)

    clear_pyprof_profiling()
    return trace

def num_events_recorded():
    global _pyprof_trace
    return _pyprof_trace.get_num_events()

def should_dump_pyprof():
    global _pyprof_trace
    return _pyprof_trace.get_num_events() >= PROTO_MAX_PYPROF_PY_EVENTS

def set_step(step, expect_traced=False, ignore_disable=False):
    global _pyprof_trace, _step, _python_start_us, _TRACING_ON
    _step = step
    _python_start_us = now_us()
    if _TRACING_ON or ignore_disable:
        _pyprof_trace.set_step(step)

def set_process_name(process_name):
    global _process_name
    _process_name = process_name

def set_phase(phase):
    global _phase
    _phase = phase

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
        logging.info("> call.name = {name}".format(name=name))

        def call(*args, **kwargs):
            # NOTE: for some reason, if we don't use a local variable here,
            # it will return None!  Bug in python3...?
            if self.debug:
                logging.info("call: {name}".format(name=name))
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
        global _pyprof_trace, _python_start_us, _step, _TRACING_ON

        start_us = now_us()
        # if self.debug:
        #     logging.info("_TRACING_ON = {val}".format(
        #         val=_TRACING_ON,
        #     ))
        #     # logging.info("_TRACING_ON = {val}\n{stack}".format(
        #     #     val=_TRACING_ON,
        #     #     stack=get_stacktrace(),
        #     # ))
        ret = self.call(*args, **kwargs)
        end_us = now_us()

        if _TRACING_ON:
            name = self.func.__name__

            # We are about to call from python into a C++ API.
            # That means we stopping executing python while C++ runs.
            # So, we must add a python execution and C++ execution event.
            #
            # [ last C++ call ][ python call ][ C++ call ]
            #                 |               |          |
            #         _python_start_us     start_us   end_us

            # TODO: BUG: sometimes, this results in a negative duration_us.
            # HACK: just filter out these rare events when building SQL database.
            # That must mean _python_start_us has been UPDATED after "start_us = now_us()" was called...
            # The only way this can happen is if self.call(...) triggered an update to _python_start_us.
            # This happens when calling "TF_Output"; possibilities:
            # - TF_Output causes a call to another wrapped function (don't think so, its a SWIG entry)
            # - multiple threads updating _python_start_us (e.g. by calling set_step/clear_pyprof_profiling/calling wrapped functions)
            #   ... unlikely since code tends to use fork() via multiprocessing if at all

            # python_event = Event(
            #     start_time_us=int(_python_start_us),
            #     duration_us=int(start_us - _python_start_us),
            #     thread_id=tid,
            #     name=name)
            # category_event = Event(
            #     start_time_us=int(start_us),
            #     duration_us=int(end_us - start_us),
            #     thread_id=tid,
            #     name=name)

            _pyprof_trace.record_python_event(
                _pyprof_trace.get_step(), name,
                start_us=_python_start_us,
                end_us=start_us)
            _pyprof_trace.record_event(
                _pyprof_trace.get_step(), self.category, name,
                start_us=start_us,
                end_us=end_us,
                debug=self.debug)

        _python_start_us = end_us

        return ret

    def __setattr__(self, name, value):
        return setattr(self.func, name, value)

    def __getattr__(self, name):
        return getattr(self.func, name)

def record_event(category, name, start_us, end_us, attrs=None, python_event=False, ignore_disable=False):
    global _pyprof_trace

    if _TRACING_ON or ignore_disable:
        _pyprof_trace.record_event(
            _pyprof_trace.get_step(), category, name, start_us, end_us,
            attrs=attrs, python_event=python_event)

def record_python_event(name, end_us, ignore_disable=False):
    """
    Useful for recording the last amount of time in between returning
    from a call to q_forward, and finishing benchmarking.
    This will include time spent in the tensorflow python API
    (i.e. not doing C++ calls, just returning back to the benchmarking script).
    """
    global _start_us, _python_start_us
    if _TRACING_ON or ignore_disable:
        record_event(CATEGORY_PYTHON, name, _python_start_us, end_us,
                     python_event=True,
                     ignore_disable=ignore_disable)
        _python_start_us = now_us()

def record_operation(start_us, end_us,
                     # attrs
                     op_name,
                     ignore_disable=False):
    """
    Useful for recording the last amount of time in between returning
    from a call to q_forward, and finishing benchmarking.
    This will include time spent in the tensorflow python API
    (i.e. not doing C++ calls, just returning back to the benchmarking script).
    """
    if _TRACING_ON or ignore_disable:
        record_event(CATEGORY_OPERATION, op_name, start_us, end_us,
                     attrs={
                         'op_name': op_name,
                     },
                     python_event=False,
                     ignore_disable=ignore_disable)

def is_recording():
    global _TRACING_ON
    return _TRACING_ON

def should_record(step):
    global _TRACING_ON
    return _TRACING_ON

def enable_tracing():
    global _TRACING_ON
    _TRACING_ON = True
    if py_config.DEBUG:
        logging.info("Enable pyprof tracing: _TRACING_ON={val}\n{stack}".format(
            val=_TRACING_ON,
            stack=get_stacktrace()))

def disable_tracing():
    global _TRACING_ON
    _TRACING_ON = False
    if py_config.DEBUG:
        logging.info("Disable pyprof tracing: _TRACING_ON={val}\n{stack}".format(
            val=_TRACING_ON,
            stack=get_stacktrace()))

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
        register_wrap_module(wrap_tensorflow, unwrap_tensorflow)
    except ImportError:
        pass

def wrap_tensorflow(category=CATEGORY_TF_API, debug=False):
    logging.info("> IML: Wrapping module=tensorflow call with category={category} annotations".format(
        category=category,
    ))
    success = wrap_util.wrap_lib(
        CFuncWrapper,
        import_libname='tensorflow',
        wrap_libname='tensorflow.pywrap_tensorflow',
        wrapper_args=(category, DEFAULT_PREFIX, debug),
        func_regex='^TF_')
    assert success
def unwrap_tensorflow():
    logging.info("> IML: Unwrapping module=tensorflow")
    wrap_util.unwrap_lib(
        CFuncWrapper,
        import_libname='tensorflow',
        wrap_libname='tensorflow.pywrap_tensorflow')

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
    logging.info("> IML: Wrapping module={mod} call with category={category} annotations".format(
        mod=module,
        category=category,
    ))
    wrap_util.wrap_module(
        CFuncWrapper, module,
        wrapper_args=(category, DEFAULT_PREFIX, debug), **kwargs)
def unwrap_module(module):
    logging.info("> IML: Unwrapping module={mod}".format(
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
            logging.info("Wrap: {name}".format(name=name))
        func_wrapper = CFuncWrapper(func, self.category, self.prefix, self.debug)
        setattr(self, name, func_wrapper)
        return func_wrapper

def wrap_entire_module(import_libname, category, debug=False, **kwargs):
    logging.info("> IML: Wrapping module={mod} call with category={category} annotations".format(
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
    logging.info("> IML: Unwrapping module={mod}".format(
        mod=import_libname))
    lib_wrapper = sys.modules[import_libname]
    sys.modules[import_libname] = lib_wrapper.lib
