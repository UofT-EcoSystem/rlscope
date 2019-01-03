import re
import time
import pprint
import importlib

from ctypes import *
import threading

from proto.protobuf.pyprof_pb2 import Pyprof, Event
# from proto.protobuf import pyprof_pb2

from profiler import tensorflow_profile_context

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import py_config

# https://stackoverflow.com/questions/9386636/profiling-a-system-with-extensively-reused-decorators
import types

MICROSECONDS_IN_SECOND = float(1e6)

DEBUG = True

DEFAULT_PREFIX = "CLIB__"

_pyprof = Pyprof()
def clear_pyprof_profiling():
    global _pyprof, _python_start_us
    _pyprof = Pyprof()
    _python_start_us = now_us()

_step = None
def set_step(step):
    global _step, _trace_steps
    _step = step
    if _step in _trace_steps and _step not in _pyprof.steps:
        if tensorflow_profile_context.DEBUG:
            print("> ADD PYPROF STEP: {s}".format(s=_step))
        _pyprof.steps.extend([step])

_trace_steps = None
def set_trace_steps(trace_steps):
    global _trace_steps
    _trace_steps = trace_steps
    pprint.pprint({'_trace_steps':trace_steps})

def now_us():
    return time.time()*MICROSECONDS_IN_SECOND
_python_start_us = None

class CLibWrapper:
    """
    # Old library import
    py_lib = cdll.LoadLibrary(_j(py_config.BUILD_DIR, 'libpy_interface.so'))

    # Wrapped library import, add "CLIB__" prefix to C function names.
    py_lib = CLibWrapper(_j(py_config.BUILD_DIR, 'libpy_interface.so'))

    # Use the imported library as before:
    py_lib.NewLibHandle.argtypes = None
    py_lib.NewLibHandle.restype = c_void_p
    """
    def __init__(self, name, LibraryLoader=cdll, prefix=DEFAULT_PREFIX):
        self.LibraryLoader = LibraryLoader
        self.lib = self.LibraryLoader.LoadLibrary(name)
        self.prefix = prefix


        self.FuncWrapper = FuncWrapper

    def __getattr__(self, name):
        # NOTE: __getattr__ only ever gets called if the object doesn't have the attribute set yet.
        c_func = getattr(self.lib, name)
        c_func_wrapper = FuncWrapper(c_func, self.prefix)
        # https://stackoverflow.com/questions/13184281/python-dynamic-function-creation-with-custom-names
        # c_func_wrapper.__call__.__name__ = "MyFuncWrapper__{name}".format(name=c_func.__name__)
        # wrapper_01
        # c_func_wrapper.__name__ = "MyFuncWrapper__{name}".format(name=c_func.__name__)
        setattr(self, name, c_func_wrapper)
        return c_func_wrapper

    def __getitem__(self, name):
        return getattr(self, name)

class FuncWrapper:
    def __init__(self, c_func, category, prefix=DEFAULT_PREFIX):
        # self.c_func = c_func
        super().__setattr__('c_func', c_func)
        super().__setattr__('prefix', prefix)
        super().__setattr__('category', category)

        def call(*args, **kwargs):
            # NOTE: for some reason, if we don't use a local variable here,
            # it will return None!  Bug in python3...?
            ret = self.c_func(*args, **kwargs)
            return ret

        #
        # If we comment this out, then "call" returns the gpu_sleep time...
        # Not sure WHY.
        #

        name = self.wrapper_name(c_func.__name__)
        print("> call.name = {name}".format(name=name))
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
        global _python_start_us, _step, _trace_steps

        start_us = now_us()
        ret = self.call(*args, **kwargs)
        end_us = now_us()

        if _trace_steps is not None and _step in _trace_steps:
            # Q: What if _step isn't present?
            tid = threading.get_ident()

            # We are about to call from python into a C++ API.
            # That means we stopping executing python while C++ runs.
            # So, we must add a python execution and C++ execution event.
            python_event = Event(
                start_time_us=int(_python_start_us),
                duration_us=int(start_us - _python_start_us),
                thread_id=tid)
            category_event = Event(
                start_time_us=int(start_us),
                duration_us=int(end_us - start_us),
                thread_id=tid)

            # category = self.__getattr__('category')
            category = self.category
            # NOTE: extend() makes a copy of everything we add, but it's more familiar so who cares.
            # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-message-fields
            _pyprof.python_events[_step].events.extend([python_event])
            _pyprof.clibs[_step].clibs[category].events.extend([category_event])
            # if _step not in _pyprof.steps:
            #     print("> ADD STEP: {s}".format(s=_step))
            #     _pyprof.steps.extend([_step])

        _python_start_us = end_us

        return ret

    def __setattr__(self, name, value):
        return setattr(self.c_func, name, value)

    def __getattr__(self, name):
        return getattr(self.c_func, name)

def wrap_module(module, category,
                func_regex=None, ignore_func_regex="^_",
                prefix=DEFAULT_PREFIX):
    for name in dir(module):
        if not re.search(ignore_func_regex, name) and (
            func_regex is None or re.search(func_regex, name)
        ):
            func = getattr(module, name)
            if type(func) == FuncWrapper or not callable(func):
                continue
            func_wrapper = FuncWrapper(func, category, prefix)
            setattr(module, name, func_wrapper)

def unwrap_module(module):
    for name in dir(module):
        # if re.search(func_regex, name):
        func_wrapper = getattr(module, name)
        if type(func_wrapper) != FuncWrapper:
            continue
        func = func_wrapper.c_func
        setattr(module, name, func)

def dump_pyprof(path):
    with open(path, 'wb') as f:
        print("> dump pyprof.steps:")
        pprint.pprint({'_pyprof.steps':list(_pyprof.steps)}, indent=2)
        f.write(_pyprof.SerializeToString())

#
# Some pre-written C++ library wrappers.
#

_LIBS_WRAPPED = False
def wrap_libs():
    global _LIBS_WRAPPED
    if _LIBS_WRAPPED:
        return
    wrap_tensorflow()
    wrap_atari()
    _LIBS_WRAPPED = True
def unwrap_libs():
    global _LIBS_WRAPPED
    if not _LIBS_WRAPPED:
        return
    unwrap_tensorflow()
    unwrap_atari()
    _LIBS_WRAPPED = False

CATEGORY_TF_API = "Framework API C"


def wrap_lib(import_libname, category, func_regex=None, wrap_libname=None):
    if wrap_libname is None:
        wrap_libname = import_libname
    lib = None
    try:

        # if DEBUG:
        #     print('> import {libname}...'.format(libname=import_libname))
        # importlib.import_module(import_libname)

        # May throw NameError
        # stmt = "import {import_lib}; lib = {wrap_lib}".format(import_lib=import_libname, wrap_lib=wrap_libname)
        # lib = eval(wrap_libname)
        # lib = eval(stmt)
        exec("import {import_lib}".format(import_lib=import_libname))
        # exec("lib = {wrap_lib}".format(wrap_lib=wrap_libname))
        lib = eval("{wrap_lib}".format(wrap_lib=wrap_libname))
        # exec(stmt)
        assert lib is not None

        # import tensorflow.pywrap_tensorflow
        if DEBUG:
            print('  ... success')
    except (ImportError, NameError) as e:
        # Failed to import library; skip wrapping the library.
        if DEBUG:
            print('  ... FAILED: cannot wrap module {lib}; stacktrace:'.format(lib=wrap_libname))
            print(e)
        return False
    wrap_module(lib, category, func_regex=func_regex)
    return True
def unwrap_lib(wrap_libname):
    try:
        if DEBUG:
            print('> lookup {libname}...'.format(libname=wrap_libname))
        # May throw NameError
        lib = eval(wrap_libname)
        if DEBUG:
            print('  ... success')
    except NameError as e:
        if DEBUG:
            print('  ... FAILED: cannot unwrap module {lib}; stacktrace:'.format(lib=wrap_libname))
            print(e)
        return
    unwrap_module(lib)

def wrap_tensorflow(category=CATEGORY_TF_API):
    success = wrap_lib(import_libname='tensorflow',
                       wrap_libname='tensorflow.pywrap_tensorflow',
                       category=category,
                       func_regex='^TF_')
    assert success
    # try:
    #     if DEBUG:
    #         print('> import tensorflow.pywrap_tensorflow...')
    #     import tensorflow.pywrap_tensorflow
    #     if DEBUG:
    #         print('  ... success')
    # except ImportError:
    #     # Failed to import library; skip wrapping the library.
    #     if DEBUG:
    #         print('  ... FAILED: cannot wrap module')
    #     return
    # func_regex = '^TF_'
    # wrap_module(tensorflow.pywrap_tensorflow, category, func_regex=func_regex)
def unwrap_tensorflow():
    unwrap_lib('tensorflow.pywrap_tensorflow')
    # try:
    #     import tensorflow.pywrap_tensorflow
    # except ImportError:
    #     return
    # unwrap_module(tensorflow.pywrap_tensorflow)

CATEGORY_SIMULATOR_CPP = "Simulator C"
CATEGORY_ATARI = CATEGORY_SIMULATOR_CPP
def wrap_atari(category=CATEGORY_ATARI):
    try:
        import atari_py
    except ImportError:
        return
    func_regex = None
    wrap_module(atari_py.ale_python_interface.ale_lib, category, func_regex=func_regex)
def unwrap_atari():
    try:
        import atari_py
    except ImportError:
        return
    unwrap_module(atari_py.ale_python_interface.ale_lib)
