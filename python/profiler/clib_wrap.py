import re
import time
import pprint
import importlib
import contextlib

from ctypes import *
import threading

from parser.common import *

from proto.protobuf.pyprof_pb2 import Pyprof, Event
# from proto.protobuf import pyprof_pb2

from profiler import tensorflow_profile_context

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import py_config

# https://stackoverflow.com/questions/9386636/profiling-a-system-with-extensively-reused-decorators
import types

MICROSECONDS_IN_SECOND = float(1e6)

from profiler import wrap_util

DEBUG = wrap_util.DEBUG

DEFAULT_PREFIX = "CLIB__"

_pyprof = Pyprof()
def clear_pyprof_profiling():
    global _pyprof, _python_start_us, _step, _process_name
    _pyprof = Pyprof()
    if _process_name is not None:
        _pyprof.process_name = _process_name
    _python_start_us = now_us()
    _step = None

_step = None
def set_step(step, expect_traced=False):
    global _step, _python_start_us
    _step = step
    _python_start_us = now_us()
    # if expect_traced:
    #     # print("> is traced = {t}".format(t=step in _trace_steps))
    #     # pprint.pprint({'step':step, 'trace_steps':_trace_steps})
    #     assert step in _trace_steps
    # if _step in _trace_steps and _step not in _pyprof.steps:
    if _TRACING_ON and step not in _pyprof.steps:
        if tensorflow_profile_context.DEBUG:
            print("> ADD PYPROF STEP: {s}".format(s=_step))

        _pyprof.steps.extend([step])

        if tensorflow_profile_context.DEBUG:
            pprint.pprint({'len(_pyprof.steps)':len(_pyprof.steps)}, indent=2)

_process_name = None
def set_process_name(process_name):
    global _process_name
    _process_name = process_name
    _pyprof.process_name = process_name

# _trace_steps = None
# def set_trace_steps(trace_steps):
#     global _trace_steps
#     _trace_steps = trace_steps
#     pprint.pprint({'_trace_steps':trace_steps})

_python_start_us = None

class CFuncWrapper:
    def __init__(self, func, category, prefix=DEFAULT_PREFIX):
        # NOTE: to be as compatible as possible with intercepting existing code,
        # we forward setattr/getattr on this object back to the func we are wrapping
        # (e.g. func might be some weird SWIG object).
        super().__setattr__('func', func)
        super().__setattr__('prefix', prefix)
        super().__setattr__('category', category)

        def call(*args, **kwargs):
            # NOTE: for some reason, if we don't use a local variable here,
            # it will return None!  Bug in python3...?
            ret = self.func(*args, **kwargs)
            return ret

        #
        # If we comment this out, then "call" returns the gpu_sleep time...
        # Not sure WHY.
        #

        name = self.wrapper_name(func.__name__)
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
        global _python_start_us, _step, _TRACING_ON

        start_us = now_us()
        ret = self.call(*args, **kwargs)
        end_us = now_us()

        # if _TRACING_ON and _trace_steps is not None and _step in _trace_steps:
        if _TRACING_ON:
            # Q: What if _step isn't present?
            tid = threading.get_ident()

            # We are about to call from python into a C++ API.
            # That means we stopping executing python while C++ runs.
            # So, we must add a python execution and C++ execution event.
            name = self.func.__name__
            python_event = Event(
                start_time_us=int(_python_start_us),
                duration_us=int(start_us - _python_start_us),
                thread_id=tid,
                name=name)
            category_event = Event(
                start_time_us=int(start_us),
                duration_us=int(end_us - start_us),
                thread_id=tid,
                name=name)

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
        return setattr(self.func, name, value)

    def __getattr__(self, name):
        return getattr(self.func, name)

def record_event(category, name, start_us, end_us, attrs=None, python_event=False):
    global _step, _pyprof
    # if _trace_steps is not None and _step in _trace_steps:
    if _TRACING_ON:
        tid = threading.get_ident()
        event = Event(
            start_time_us=int(start_us),
            duration_us=int(end_us - start_us),
            thread_id=tid,
            name=name,
            attrs=attrs)
        if python_event:
            _pyprof.python_events[_step].events.extend([event])
        else:
            _pyprof.clibs[_step].clibs[category].events.extend([event])

def record_python_event(name, end_us):
    """
    Useful for recording the last amount of time in between returning
    from a call to q_forward, and finishing benchmarking.
    This will include time spent in the tensorflow python API
    (i.e. not doing C++ calls, just returning back to the benchmarking script).
    """
    global _step, _start_us, _python_start_us
    # if _trace_steps is not None and _step in _trace_steps:
    if _TRACING_ON:
        record_event(CATEGORY_PYTHON, name, _python_start_us, end_us, python_event=True)
        _python_start_us = now_us()

def record_operation(start_us, end_us,
                     # attrs
                     op_name):
    """
    Useful for recording the last amount of time in between returning
    from a call to q_forward, and finishing benchmarking.
    This will include time spent in the tensorflow python API
    (i.e. not doing C++ calls, just returning back to the benchmarking script).
    """
    global _step
    # if _trace_steps is not None and _step in _trace_steps:
    if _TRACING_ON:
        record_event(CATEGORY_OPERATION, op_name, start_us, end_us,
                     attrs={
                         'op_name': op_name,
                     },
                     python_event=False)

def is_recording():
    global _step
    return _TRACING_ON
    # return _trace_steps is not None and _step in _trace_steps

def should_record(step):
    # global _trace_steps
    # return _trace_steps is not None and step in _trace_steps
    return _TRACING_ON

# By default tracing is OFF.
_TRACING_ON = False

@contextlib.contextmanager
def tracing_disabled():
    with tracing_as(should_enable=False):
        try:
            yield
        finally:
            pass

@contextlib.contextmanager
def tracing_enabled():
    with tracing_as(should_enable=True):
        try:
            yield
        finally:
            pass

def enable_tracing():
    global _TRACING_ON
    _TRACING_ON = True

def disable_tracing():
    global _TRACING_ON
    _TRACING_ON = False

class tracing_as:
    def __init__(self, should_enable):
        self.should_enable = should_enable

    def __enter__(self):
        global _TRACING_ON
        self._tracing_on = _TRACING_ON
        _TRACING_ON = self.should_enable

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _TRACING_ON
        _TRACING_ON = self._tracing_on

def dump_pyprof(path, process_name, phase):
    with open(path, 'wb') as f:
        print("> dump pyprof.steps:")
        pprint.pprint({
            '_pyprof.steps':list(_pyprof.steps),
            '_pyprof.process_name':_pyprof.process_name,
            '_pyprof.phase':_pyprof.phase,
        }, indent=2)
        _pyprof.process_name = process_name
        _pyprof.phase = phase
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

def wrap_tensorflow(category=CATEGORY_TF_API):
    success = wrap_util.wrap_lib(
        CFuncWrapper,
        import_libname='tensorflow',
        wrap_libname='tensorflow.pywrap_tensorflow',
        wrapper_args=(category, DEFAULT_PREFIX),
        func_regex='^TF_')
    assert success
def unwrap_tensorflow():
    wrap_util.unwrap_lib(CFuncWrapper, 'tensorflow.pywrap_tensorflow')

def wrap_atari(category=CATEGORY_ATARI):
    try:
        import atari_py
    except ImportError:
        return
    func_regex = None
    wrap_util.wrap_module(
        CFuncWrapper, atari_py.ale_python_interface.ale_lib,
        wrapper_args=(category, DEFAULT_PREFIX),
        func_regex=func_regex)
def unwrap_atari():
    try:
        import atari_py
    except ImportError:
        return
    wrap_util.unwrap_module(
        CFuncWrapper,
        atari_py.ale_python_interface.ale_lib)
