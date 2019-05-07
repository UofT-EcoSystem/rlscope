import contextlib

from iml_profiler.parser.common import *

from iml_profiler.protobuf.pyprof_pb2 import Pyprof, Event

from iml_profiler.profiler import tensorflow_profile_context

from iml_profiler.profiler import proto_util

# https://stackoverflow.com/questions/9386636/profiling-a-system-with-extensively-reused-decorators
import types

MICROSECONDS_IN_SECOND = float(1e6)

from iml_profiler.profiler import wrap_util

DEBUG = wrap_util.DEBUG

DEFAULT_PREFIX = "CLIB__"

# 39870 events in Pyprof ~ 1.6M
PROTO_MAX_PYPROF_PY_EVENTS = 40000

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
        self.num_events = 0

    def finish(self, process_name, phase):
        self.pyprof.process_name = process_name
        self.pyprof.phase = phase

    def set_step(self, step):
        if step is None:
            return
        self.step = step
        if step not in self.pyprof.steps:
            if tensorflow_profile_context.DEBUG:
                print("> ADD PYPROF STEP: {s}".format(s=self.step))

            self.pyprof.steps.extend([step])

            if tensorflow_profile_context.DEBUG:
                pprint.pprint({
                    'len(self.pyprof.steps)':len(self.pyprof.steps),
                }, indent=2)

    def dump(self, path, process_name, phase):
        self.pyprof.process_name = process_name
        self.pyprof.phase = phase

        with open(path, 'wb') as f:
            print("> dump pyprof.steps:")
            pprint.pprint({
                'len(pyprof.steps)':len(self.pyprof.steps),
                'pyprof.process_name':self.pyprof.process_name,
                'pyprof.phase':self.pyprof.phase,
            }, indent=2)
            f.write(self.pyprof.SerializeToString())

    def record_event(self, step, category, name, start_us, end_us, attrs=None, python_event=False):
        assert step is not None

        event = proto_util.mk_event(
            name=name,
            start_us=start_us,
            end_us=end_us,
            attrs=attrs)

        # NOTE: extend() makes a copy of everything we add, but it's more familiar so who cares.
        # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-message-fields
        if python_event:
            self.pyprof.python_events[step].events.extend([event])
        else:
            self.pyprof.clibs[step].clibs[category].events.extend([event])

        self.num_events += 1

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

#
# Module globals.
#
_pyprof_trace = PyprofTrace()
_step = None
_process_name = None
_phase = None
# Q: Should we initialize this to now_us()...?
_python_start_us = None
# By default tracing is OFF.
_TRACING_ON = False

def clear_pyprof_profiling():
    global _pyprof_trace, _python_start_us, _process_name
    _pyprof_trace = PyprofTrace()
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
    return _pyprof_trace.num_events

def should_dump_pyprof():
    global _pyprof_trace
    return _pyprof_trace.num_events >= PROTO_MAX_PYPROF_PY_EVENTS

def set_step(step, expect_traced=False, ignore_disable=False):
    global _pyprof_trace, _step, _python_start_us
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
        global _pyprof_trace, _python_start_us, _step, _TRACING_ON

        start_us = now_us()
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
                _pyprof_trace.step, name,
                start_us=_python_start_us,
                end_us=start_us)
            _pyprof_trace.record_event(
                _pyprof_trace.step, self.category, name,
                start_us=start_us,
                end_us=end_us)

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
            _pyprof_trace.step, category, name, start_us, end_us,
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
    return _TRACING_ON

def should_record(step):
    return _TRACING_ON

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
