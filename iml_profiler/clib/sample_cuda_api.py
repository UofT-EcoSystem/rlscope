# Wrapper around libsample_cuda_api.so LD_PRELOAD library.
import ctypes

from iml_profiler.parser.common import *

from ctypes import *

from iml_profiler.profiler import iml_logging
iml_logging.setup_logging()

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
            is_sample_cuda_api_lib(path)
            for path in re.split(r':', os.environ.get('LD_PRELOAD', '')))
    return _IS_USED

def load_library(allow_fail=None):
    global _so

    if allow_fail is None:
        allow_fail = is_used()

    # except OSError as e:
    try:
        _so = ctypes.cdll.LoadLibrary('libsample_cuda_api.so')
        # os.error
    except OSError as e:
        if not allow_fail or not re.search(r'no such file', str(e), re.IGNORECASE):
            raise
        # import pprint; pprint.pprint({
        #     'e.__dict__':e.__dict__,
        #     'e.errno':e.errno,
        # })
        logging.info("Failed to load libsample_cuda_api.so")
        return

    from iml_profiler.clib import sample_cuda_api

    def set_api_wrapper(api_name):
        func = getattr(_so, api_name)
        def api_wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            if ret != TF_OK:
                raise IMLProfError(ret)
            return ret
        setattr(sample_cuda_api, api_name, api_wrapper)

    _so.setup.argtypes = []
    _so.setup.restype = c_int
    # sample_cuda_api.setup = _so.setup
    set_api_wrapper('setup')

    _so.enable_tracing.argtypes = []
    _so.enable_tracing.restype = c_int
    # sample_cuda_api.enable_tracing = _so.enable_tracing
    set_api_wrapper('enable_tracing')

    _so.disable_tracing.argtypes = []
    _so.disable_tracing.restype = c_int
    # sample_cuda_api.disable_tracing = _so.disable_tracing
    set_api_wrapper('disable_tracing')

    _so.print.argtypes = []
    _so.print.restype = c_int
    # sample_cuda_api.print = _so.print
    set_api_wrapper('print')

    _so.is_enabled.argtypes = [POINTER(c_int)]
    # _so.is_enabled.argtypes = [pointer(c_int)]
    _so.is_enabled.restype = c_int
    # sample_cuda_api.is_enabled = _so.is_enabled
    set_api_wrapper('is_enabled')

    _so.collect.argtypes = []
    _so.collect.restype = c_int
    # sample_cuda_api.collect = _so.collect
    set_api_wrapper('collect')

    logging.info("Loaded symbols from libsample_cuda_api.so")

class IMLProfError(Exception):
    def __init__(self, errcode):
        self.errcode = errcode
        super().__init__(self._gen_msg())

    def _gen_msg(self):
        msg = textwrap.dedent("""\
        IML API Error-code = {err} (see logs above for detailed C++ error messages)
        """.format(err=error_string(self.errcode)))
        return msg

def error_string(retcode):
    return _ret_to_name[retcode]

# Q: Make wrapper that converts status code into exception + error message?
