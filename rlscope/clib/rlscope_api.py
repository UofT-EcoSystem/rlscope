r"""
Python wrapper around the ``librlscope.so`` ``LD_PRELOAD`` library.
We offload most profiling work to the C++ library.
In particular:

#. Every :py:meth:`rlscope.profiler.profilers.Profiler.operation` calls into
   :py:func:`push_operation` upon entering the ``with`` block and
   :py:func:`pop_operation` upon exiting the ``with`` block.
   This records operation start/end timestamps.
#. Every :py:meth:`rlscope.profiler.profilers.Profiler.report_progress`
   call at the start of a training loop iteration calls
   :py:func:`.end_pass` then :py:func:`.start_pass`.
   This is used to start/stop collection of GPU hardware metrics.
#. Trace files are dumped asynchronously in ``librlscope.so``
   once collected traces exceed :math:`\approx` 20MB.

"""
import ctypes
from os import environ as ENV

from io import StringIO
from rlscope.profiler.rlscope_logging import logger
from rlscope.parser.common import *

import ctypes
import ctypes.util
from ctypes import *

c_int_p = ctypes.POINTER(ctypes.c_int)

from rlscope import py_config

from rlscope.profiler.rlscope_logging import logger
from rlscope import py_config


RLSCOPE_LIBNAME = 'rlscope'
RLSCOPE_CLIB = None
def find_librlscope():
    global RLSCOPE_CLIB
    if RLSCOPE_CLIB is not None:
        return

    # Older version of python (<=3.6) need 'LIBRARY_PATH' to be defined for find_library to work.
    # assert 'LIBRARY_PATH' not in ENV or ENV['LIBRARY_PATH'] == ENV['LD_LIBRARY_PATH']

    # First, try to find librlscope.so using our current LD_LIBRARY_PATH.
    #
    # NOTE: This will succeed in development mode (i.e., "python setup.py develop")
    # since we set LD_LIBRARY_PATH in source_me.sh.
    # In "pip install rlscope" distribution mode, this will fail, since
    # librlscope.lib is packaged inside:
    #   rlscope/cpp/lib/librlscope.so
    ENV['LIBRARY_PATH'] = ENV.get('LD_LIBRARY_PATH', '')
    RLSCOPE_CLIB = ctypes.util.find_library(RLSCOPE_LIBNAME)

    if RLSCOPE_CLIB is None:
        orig_LD_LIBRARY_PATH = ENV.get('LD_LIBRARY_PATH', '')
        # Locations to search for librlscope.so
        # Currently, we just search for rlscope/cpp/lib/librlscope.so
        rlscope_lib_dirs = [py_config.CPP_LIB]
        for path in rlscope_lib_dirs:
            if not os.path.isdir(path):
                continue
            ENV['LD_LIBRARY_PATH'] = "{LD_LIBRARY_PATH}:{path}".format(
                path=path,
                LD_LIBRARY_PATH=ENV.get('LD_LIBRARY_PATH', ''),
            )
            ENV['LIBRARY_PATH'] = ENV['LD_LIBRARY_PATH']
            RLSCOPE_CLIB = ctypes.util.find_library(RLSCOPE_LIBNAME)
            if RLSCOPE_CLIB is not None:
                break
            ENV['LD_LIBRARY_PATH'] = orig_LD_LIBRARY_PATH
            ENV['LIBRARY_PATH'] = ENV['LD_LIBRARY_PATH']

    if RLSCOPE_CLIB is None:
        if py_config.is_development_mode():
            """
            RL-Scope has been installed using "python setup.py develop", and is being 
            run from a github repo checkout.
            
            Provide instructions on how to build librlscope.so from scratch, and add it to 
            the user's LD_LIBRARY_PATH so we can find it.
            """
            logger.error(textwrap.dedent("""\
            RL-Scope ERROR: couldn't find RL-Scope library (lib{name}.so); to build it, do:
              $ cd {root}
              $ bash ./setup.sh
              # To modify your LD_LIBRARY_PATH to include lib{name}.so, run:
              $ source source_me.sh
            """.format(
                name=RLSCOPE_LIBNAME,
                root=py_config.ROOT,
            )).rstrip())
        else:
            """
            RL-Scope has been installed using "pip install rlscope".
            
            librlscope.so SHOULD be bundled with the install python package; 
            if it isn't then this is a BUG.
            """
            logger.error(textwrap.dedent("""\
            RL-Scope ERROR: couldn't find RL-Scope library (lib{name}.so) inside {lib_dir}.
            This looks like a BUG in RL-Scope; please report it at:
              https://github.com/UofT-EcoSystem/rlscope/issues
            """.format(
                lib_dir=py_config.CPP_LIB,
                name=RLSCOPE_LIBNAME,
            )).rstrip())
        sys.exit(1)


TF_OK = 0
TF_CANCELLED = 1
TF_UNKNOWN = 2
TF_INVALID_ARGUMENT = 3
TF_DEADLINE_EXCEEDED = 4
TF_NOT_FOUND = 5
TF_ALREADY_EXISTS = 6
TF_PERMISSION_DENIED = 7
TF_UNAUTHENTICATED = 16
TF_RESOURCE_EXHAUSTED = 8
TF_FAILED_PRECONDITION = 9
TF_ABORTED = 10
TF_OUT_OF_RANGE = 11
TF_UNIMPLEMENTED = 12
TF_INTERNAL = 13
TF_UNAVAILABLE = 14
TF_DATA_LOSS = 15

_ret_to_name = dict()
_ret_to_name[TF_OK] = 'TF_OK'
_ret_to_name[TF_CANCELLED] = 'TF_CANCELLED'
_ret_to_name[TF_UNKNOWN] = 'TF_UNKNOWN'
_ret_to_name[TF_INVALID_ARGUMENT] = 'TF_INVALID_ARGUMENT'
_ret_to_name[TF_DEADLINE_EXCEEDED] = 'TF_DEADLINE_EXCEEDED'
_ret_to_name[TF_NOT_FOUND] = 'TF_NOT_FOUND'
_ret_to_name[TF_ALREADY_EXISTS] = 'TF_ALREADY_EXISTS'
_ret_to_name[TF_PERMISSION_DENIED] = 'TF_PERMISSION_DENIED'
_ret_to_name[TF_UNAUTHENTICATED] = 'TF_UNAUTHENTICATED'
_ret_to_name[TF_RESOURCE_EXHAUSTED] = 'TF_RESOURCE_EXHAUSTED'
_ret_to_name[TF_FAILED_PRECONDITION] = 'TF_FAILED_PRECONDITION'
_ret_to_name[TF_ABORTED] = 'TF_ABORTED'
_ret_to_name[TF_OUT_OF_RANGE] = 'TF_OUT_OF_RANGE'
_ret_to_name[TF_UNIMPLEMENTED] = 'TF_UNIMPLEMENTED'
_ret_to_name[TF_INTERNAL] = 'TF_INTERNAL'
_ret_to_name[TF_UNAVAILABLE] = 'TF_UNAVAILABLE'
_ret_to_name[TF_DATA_LOSS] = 'TF_DATA_LOSS'

_so = None

_IS_USED = None
def is_used():
    global _IS_USED
    if _IS_USED is None:
        _IS_USED = any(
            is_rlscope_api_lib(path)
            for path in re.split(r':', os.environ.get('LD_PRELOAD', '')))
    return _IS_USED

def load_library(allow_fail=None):
    global _so

    if allow_fail is None:
        allow_fail = is_used()

    # except OSError as e:
    try:
        _so = ctypes.cdll.LoadLibrary(RLSCOPE_CLIB)
        # os.error
    except OSError as e:
        if not allow_fail or not re.search(r'no such file', str(e), re.IGNORECASE):
            raise
        # import pprint; pprint.pprint({
        #     'e.__dict__':e.__dict__,
        #     'e.errno':e.errno,
        # })
        logger.info(f"Failed to load {RLSCOPE_CLIB}")
        return

    _so.rlscope_setup.argtypes = []
    _so.rlscope_setup.restype = c_int
    _set_api_wrapper('setup')

    _so.rlscope_enable_tracing.argtypes = []
    _so.rlscope_enable_tracing.restype = c_int
    _set_api_wrapper('enable_tracing')

    _so.rlscope_disable_tracing.argtypes = []
    _so.rlscope_disable_tracing.restype = c_int
    _set_api_wrapper('disable_tracing')

    _so.rlscope_print.argtypes = []
    _so.rlscope_print.restype = c_int
    _set_api_wrapper('print')

    _so.rlscope_is_enabled.argtypes = [POINTER(c_int)]
    _so.rlscope_is_enabled.restype = c_int
    _set_api_wrapper('is_enabled')

    _so.rlscope_set_metadata.argtypes = [ c_char_p, c_char_p, c_char_p, c_char_p ]
    _so.rlscope_set_metadata.restype = c_int
    # _set_api_wrapper('set_metadata')

    _so.rlscope_record_event.argtypes = [ c_char_p, c_int64, c_int64, c_char_p ]
    _so.rlscope_record_event.restype = c_int
    # _set_api_wrapper('record_event')

    _so.rlscope_record_overhead_event.argtypes = [ c_char_p, c_int ]
    _so.rlscope_record_overhead_event.restype = c_int
    # _set_api_wrapper('record_overhead_event')

    _so.rlscope_record_overhead_event_for_operation.argtypes = [ c_char_p, c_char_p, c_int ]
    _so.rlscope_record_overhead_event_for_operation.restype = c_int
    # _set_api_wrapper('record_overhead_event_for_operation')

    _so.rlscope_push_operation.argtypes = [ c_char_p ]
    _so.rlscope_push_operation.restype = c_int
    # _set_api_wrapper('push_operation')

    _so.rlscope_pop_operation.argtypes = []
    _so.rlscope_pop_operation.restype = c_int
    # _set_api_wrapper('pop_operation')

    _so.rlscope_start_pass.argtypes = []
    _so.rlscope_start_pass.restype = c_int

    _so.rlscope_end_pass.argtypes = []
    _so.rlscope_end_pass.restype = c_int

    _so.rlscope_has_next_pass.argtypes = [c_int_p]
    _so.rlscope_has_next_pass.restype = c_int

    _so.rlscope_disable_gpu_hw.argtypes = []
    _so.rlscope_disable_gpu_hw.restype = c_int

    _so.rlscope_await_dump.argtypes = []
    _so.rlscope_await_dump.restype = c_int
    _set_api_wrapper('await_dump')

    _so.rlscope_async_dump.argtypes = []
    _so.rlscope_async_dump.restype = c_int
    _set_api_wrapper('async_dump')

    logger.info(f"Loaded symbols from {RLSCOPE_CLIB}")

def _full_api_name(api_name):
    return f"rlscope_{api_name}"

def _set_api_wrapper(api_name):
    full_api_name = _full_api_name(api_name)
    from rlscope.clib import rlscope_api
    func = getattr(_so, full_api_name)
    def api_wrapper(*args, **kwargs):
        if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
            logger.info(_log_api_call_msg(api_name,
                                           *args, **kwargs))
        ret = func(*args, **kwargs)
        if ret != TF_OK:
            raise RLScopeLibraryError(ret)
        return ret
    setattr(rlscope_api, api_name, api_wrapper)

def set_metadata(directory, process_name, machine_name, phase):
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('set_metadata',
                                       directory, process_name, machine_name, phase))
    ret = _so.rlscope_set_metadata(
        _as_c_string(directory),
        _as_c_string(process_name),
        _as_c_string(machine_name),
        _as_c_string(phase),
    )
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def record_event(
    category,
    start_us,
    duration_us,
    name):
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('record_event',
                                       category, start_us, duration_us, name))
    if py_config.DEBUG and duration_us < 0:
        logger.debug("BUG: recorded event with negative duration: Event(category={category}, name={name}, start_us={start_us}, dur_us={duration_us})".format(
            category=category,
            start_us=start_us,
            duration_us=duration_us,
            name=name,
        ))
    ret = _so.rlscope_record_event(
        _as_c_string(category),
        c_int64(int(start_us)),
        c_int64(int(duration_us)),
        _as_c_string(name),
    )
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def record_overhead_event(
    overhead_type,
    num_events):
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('record_overhead_event',
                                       overhead_type, num_events))
    ret = _so.rlscope_record_overhead_event(
        _as_c_string(overhead_type),
        c_int(num_events)
    )
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def record_overhead_event_for_operation(
    overhead_type,
    operation,
    num_events):
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('record_overhead_event_for_operation',
                                       overhead_type, operation, num_events))
    ret = _so.rlscope_record_overhead_event_for_operation(
        _as_c_string(overhead_type),
        _as_c_string(operation),
        c_int(num_events)
    )
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def push_operation(operation):
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('push_operation', operation))
    ret = _so.rlscope_push_operation(
        _as_c_string(operation),
    )
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def set_max_operations(operation, num_pushes):
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('set_max_operations', operation, num_pushes))
    ret = _so.rlscope_set_max_operations(
        _as_c_string(operation),
        c_int(num_pushes),
    )
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def pop_operation():
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('pop_operation'))
    ret = _so.rlscope_pop_operation()
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def start_pass():
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('start_pass'))
    ret = _so.rlscope_start_pass()
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def end_pass():
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('end_pass'))
    ret = _so.rlscope_end_pass()
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

def has_next_pass():
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('has_next_pass'))
    has_next_pass = c_int(0)
    ret = _so.rlscope_has_next_pass(ctypes.byref(has_next_pass))
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    value = bool(has_next_pass.value)
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        if value:
            logger.info(f"[PY_RLSCOPE_LIB]  returned: {value}")
    return value

def disable_gpu_hw():
    if py_config.DEBUG and py_config.DEBUG_RLSCOPE_LIB_CALLS:
        logger.info(_log_api_call_msg('disable_gpu_hw'))
    ret = _so.rlscope_disable_gpu_hw()
    if ret != TF_OK:
        raise RLScopeLibraryError(ret)
    return ret

#
# API wrapper helpers.
#

class RLScopeLibraryError(Exception):
    """
    Error raised when an error is returned by a ``librlscope.so`` API call.
    """
    def __init__(self, errcode):
        self.errcode = errcode
        super().__init__(self._gen_msg())

    def _gen_msg(self):
        msg = textwrap.dedent("""\
        RL-Scope API Error-code = {err} (see logs above for detailed C++ error messages)
        """.format(err=error_string(self.errcode)))
        return msg

def error_string(retcode):
    return _ret_to_name[retcode]

def _as_c_string(py_string):
    return c_char_p(py_string.encode('ascii'))

def _arg_string(arg):
    if type(arg) == str:
        return f'"{arg}"'
    return str(arg)

def _log_api_call_msg(name, *args, **kwargs):
    if len(kwargs) == 0:
        kwargs_str = ''
    else:
        ss = StringIO()
        for i, key, value in enumerate(kwargs.items()):
            if len(args) > 0 or i > 0:
                ss.write(', ')
            ss.write(key)
            ss.write('=')
            ss.write(_arg_string(value))
        kwargs_str = ss.getvalue()
    return "[PY_RLSCOPE_LIB] {func}({args}{kwargs})".format(
        func=name,
        args=', '.join([_arg_string(arg) for arg in args]),
        kwargs=kwargs_str,
    )

