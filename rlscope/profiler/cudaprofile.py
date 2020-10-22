"""
Wrappers around libcudart.so symbols for starting/stopping nvprof profiling.

.. deprecated:: 1.0.0
    We don't use nvprof anymore.
"""
import ctypes

from rlscope.parser.common import *

from ctypes import *
# _cudart = ctypes.CDLL('libcudart.so')
_cudart = ctypes.cdll.LoadLibrary('libcudart.so')

_cudart.cudaProfilerInitialize.argtypes = [c_char_p, c_char_p, c_int]
_cudart.cudaProfilerInitialize.restype = c_int

_cudart.cudaProfilerStart.argtypes = None
_cudart.cudaProfilerStart.restype = c_int

_cudart.cudaProfilerStop.argtypes = None
_cudart.cudaProfilerStop.restype = c_int

# https://github.com/lebedov/scikit-cuda/blob/master/skcuda/cudart.py
_cudart.cudaGetErrorString.restype = ctypes.c_char_p
_cudart.cudaGetErrorString.argtypes = [ctypes.c_int]

# NOTE: we ASSUME the user has invoked nvprof like this:
#   nvprof --profile-from-start off python3 some/script.py
PROFILER_IS_ON = False

def is_profiler_enabled():
    return PROFILER_IS_ON

def check_cuda_error(func_name, ret):
    if ret != 0:
        errstr = cudaGetErrorString(ret)
        raise Exception("{f} returned {ret}; \"{errstr}\"".format(
            f=func_name, ret=ret, errstr=errstr))

def start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in 
    # the future.
    ret = _cudart.cudaProfilerStart()
    check_cuda_error("cudaProfilerStart", ret)
    global PROFILER_IS_ON
    PROFILER_IS_ON = True

def stop():
    ret = _cudart.cudaProfilerStop()
    check_cuda_error("cudaProfilerStop", ret)
    global PROFILER_IS_ON
    assert PROFILER_IS_ON
    PROFILER_IS_ON = False

# __host__ ​cudaError_t cudaProfilerInitialize ( const char* configFile, const char* outputFile, cudaOutputMode_t outputMode )
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html#group__CUDART__PROFILER_1gcd07e875aa4030363bca13159c3c8494
# Initialize nvprof.
#
# Q: Can we call this multiple times in the same program in order to
# create multiple nvprof output files?

CUDA_OUTPUT_MODE_cudaKeyValuePair = 0x00
CUDA_OUTPUT_MODE_cudaCSV = 0x01
CUDA_OUTPUT_MODES = set([CUDA_OUTPUT_MODE_cudaKeyValuePair, CUDA_OUTPUT_MODE_cudaCSV])

def initialize(output_file, output_mode=CUDA_OUTPUT_MODE_cudaKeyValuePair):
    """
    enum cudaOutputMode
        CUDA Profiler Output modes

        Values
        cudaKeyValuePair = 0x00
        Output mode Key-Value pair format.
        cudaCSV = 0x01
        Output mode Comma separated values format.
    
    :param output_file: 
    :return: 
    """
    # output_file.encode()
    assert output_mode in CUDA_OUTPUT_MODES
    ret = _cudart.cudaProfilerInitialize(output_file, output_mode)
    check_cuda_error("cudaProfilerInitialize", ret)

def cudaGetErrorString(e):
    """
    Retrieve CUDA error string.
    Return the string associated with the specified CUDA error status
    code.
    Parameters
    ----------
    e : int
        Error number.
    Returns
    -------
    s : str
        Error string.
    """

    return _cudart.cudaGetErrorString(e)
