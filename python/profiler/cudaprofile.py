import ctypes

_cudart = ctypes.CDLL('libcudart.so')

# NOTE: we ASSUME the user has invoked nvprof like this:
#   nvprof --profile-from-start off python3 some/script.py
PROFILER_IS_ON = False

def is_profiler_enabled():
    return PROFILER_IS_ON

def start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in 
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)
    global PROFILER_IS_ON
    PROFILER_IS_ON = True

def stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)
    global PROFILER_IS_ON
    assert PROFILER_IS_ON
    PROFILER_IS_ON = False


