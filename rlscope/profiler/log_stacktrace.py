"""
Code for recording and logging stack traces of training scripts to help determine
how Python interfaces with native C libraries.

This code was useful for determining how Python calls into PyTorch; PyTorch
has multiple native shared libraries it calls into.
"""
import textwrap
import traceback
import contextlib
from io import StringIO

import typing

# import tensorflow as tf

from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler import wrap_util

# Intercept tf.Session.run(...) calls to see when calls to TensorFlow graph computations are made.
#
# Never called with --python-mode...

def print_indent(ss, indent):
    if indent == 0 or indent is None:
        return
    ss.write('  '*indent)

def with_indent(txt, indent):
    if indent == 0 or indent is None:
        return txt
    return textwrap.indent(txt, prefix='  '*indent)

class LoggedStackTrace:
    def __init__(self, name, format_stack):
        self.name = name
        self.format_stack = format_stack
        self.num_calls = 0
        self.printed = False

    def add_call(self):
        self.num_calls += 1
        self.printed = False

    def print(self, ss, skip_last=0, indent=0):
        keep_stack = self.format_stack[:len(self.format_stack)-skip_last]
        ss.write(with_indent(''.join(keep_stack), indent))
        self.printed = True

class _LoggedStackTraces:
    def __init__(self):
        # traceback.format_stack() ->
        self.stacktraces = dict()

    def _key(self, name, format_stack):
        return tuple(format_stack)

    def log_call(self, name, format_stack):
        key = self._key(name, format_stack)
        stacktrace = self.stacktraces.get(key, None)
        if stacktrace is None:
            stacktrace = LoggedStackTrace(name, format_stack)
            self.stacktraces[key] = stacktrace
        stacktrace.add_call()

    def num_to_print(self):
        n = 0
        for st in self.stacktraces.values():
            if not st.printed:
                n += 1
        return n

    def print(self, ss, skip_last=0, indent=0):
        # Only print stacktraces for functions that have been called since we last printed.
        stacktraces = [st for st in self.stacktraces.values() if not st.printed]
        # Sort by number of calls
        stacktraces.sort(key=lambda st: (st.num_calls, st.name))
        print_indent(ss, indent)
        ss.write("Stacktraces ordered by number of calls (since last printed)\n")
        for i, st in enumerate(stacktraces):
            print_indent(ss, indent+1)
            ss.write("Stacktrace[{i}] num_calls={num_calls}: {name}\n".format(
                i=i,
                num_calls=st.num_calls,
                name=st.name,
            ))
            st.print(ss, indent=indent+2, skip_last=skip_last)

    def wrap_module(self, module, should_wrap=None):
        wrap_util.wrap_module(LoggedCall, module, should_wrap=should_wrap)

    def unwrap_module(self, module):
        wrap_util.unwrap_module(LoggedCall, module)

    def wrap_func(self, module, name, should_wrap=None):
        wrap_util.wrap_func(LoggedCall, module, name, should_wrap=should_wrap)

    def unwrap_func(self, module, name):
        wrap_util.unwrap_func(LoggedCall, module, name)

def log_call(func, name, *args, **kwargs):
    if LoggedStackTraces is not None:
        stack = traceback.format_stack()
        LoggedStackTraces.log_call(name, stack)
    return func(*args, **kwargs)

class LoggedCall:
    def __init__(self, func, name=None):
        self.func = func
        if name is None:
            name = self.func.__name__
        self.name = name

    # -> typing.Any:
    def __call__(self, *args, **kwargs):
        if LoggedStackTraces is not None:
            stack = traceback.format_stack()
            LoggedStackTraces.log_call(self.name, stack)
        ret = self.func(*args, **kwargs)
        return ret

LoggedStackTraces = _LoggedStackTraces()

# LoggedStackTraces = None
# def setup_logging_stack_traces(FLAGS):
#     global LoggedStackTraces
#     WRAP_TF_SESSION_RUN = FLAGS.log_stacktrace_freq is not None
#     if WRAP_TF_SESSION_RUN:
#         LoggedStackTraces = _LoggedStackTraces()
#
#         original_tf_Session_run = tf.compat.v1.Session.run
#         def wrapped_tf_Session_run(self, fetches, feed_dict=None, options=None, run_metadata=None):
#             return log_call(original_tf_Session_run, "tf.Session.run", self, fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
#         tf.compat.v1.Session.run = wrapped_tf_Session_run
#
#         from tensorflow.python import pywrap_tfe
#
#         original_pywrap_tfe_TFE_Py_Execute = pywrap_tfe.TFE_Py_Execute
#         def wrapped_pywrap_tfe_TFE_Py_Execute(*args, **kwargs):
#             return log_call(original_pywrap_tfe_TFE_Py_Execute, "TFE_Py_Execute", *args, **kwargs)
#         pywrap_tfe.TFE_Py_Execute = wrapped_pywrap_tfe_TFE_Py_Execute
#
#         original_pywrap_tfe_TFE_Py_FastPathExecute = pywrap_tfe.TFE_Py_FastPathExecute
#         def wrapped_pywrap_tfe_TFE_Py_FastPathExecute(*args, **kwargs):
#             return log_call(original_pywrap_tfe_TFE_Py_FastPathExecute, "TFE_Py_FastPathExecute", *args, **kwargs)
#         pywrap_tfe.TFE_Py_FastPathExecute = wrapped_pywrap_tfe_TFE_Py_FastPathExecute


@contextlib.contextmanager
def with_log_stacktraces():
    """Context manager for soft device placement, allowing summaries on CPU.

    Eager and graph contexts have different default device placements. See
    b/148408921 for details. This context manager should be used whenever using
    summary writers contexts to make sure summaries work when executing on TPUs.

    Yields:
      Sets `tf.config.set_soft_device_placement(True)` within the context
    """
    try:
        yield
    finally:
        log_stacktraces()

def log_stacktraces():
    if LoggedStackTraces is not None and LoggedStackTraces.num_to_print() > 0:
        ss = StringIO()
        # stack[-1] = Call to "traceback.format_stack()"
        # stack[-2] = Call to "return log_call(...)"
        # LoggedStackTraces.print(ss, skip_last=2, indent=0)
        LoggedStackTraces.print(ss, skip_last=1, indent=0)
        logger.info(ss.getvalue().rstrip())
