"""
Constants used in collected trace files.

WARNING: if you change the values of these (e.g., CATEGORY_*),
future analysis and plot generation will fail...so don't do that!

Attributes
----------
CATEGORY_PYTHON : str
  Python time.
CATEGORY_CUDA_API_CPU  : str
  CUDA API time.
CATEGORY_GPU  : str
  GPU kernel and memcpy time.
CATEGORY_SIMULATOR_CPP : str
  Simulator native library time.
CATEGORY_TF_API : str
  DL backend native library time.
CATEGORY_OPERATION : str
  User annotation (i.e., :py:meth:`rlscope.api.Profiler.operation`)
CATEGORY_PROF_CUPTI : str
  Profiling overhead: closed-source inflation within CUDA API calls
  (e.g., ``cudaLaunchKernel``, ``cudaMemcpyAsync``) when CUPTI is enabled
CATEGORY_PROF_LD_PRELOAD : str
  Profiling overhead: overhead from intercepting CUDA API using CUPTI library.
CATEGORY_PROF_PYTHON_ANNOTATION : str
  Profiling overhead: overhead from user-level annotations.
CATEGORY_PROF_PYTHON_INTERCEPTION : str
  Profiling overhead: overhead from intercepting Python :math:`\leftrightarrow` C library calls.
"""
USEC_IN_SEC = 1e6
PSEC_IN_USEC = 1e3

OPERATION_PYTHON_PROFILING_OVERHEAD = "Python profiling overhead"

CATEGORY_TF_API = "Framework API C"
CATEGORY_PYTHON = 'Python'
CATEGORY_PYTHON_PROFILER = 'Python profiler'
CATEGORY_CUDA_API_CPU = 'CUDA API CPU'
CATEGORY_UNKNOWN = 'Unknown'
CATEGORY_GPU = 'GPU'
CATEGORY_DUMMY_EVENT = 'Dummy event'
# Category captures when operations of a given type start/end.
# That way, if we have a profile with multiple operations in it,
# we can reduce the scope to just an operation of interest (e.g. Q-forward).
#
# NOTE: This operation category should NOT show up in compute_overlap.
CATEGORY_OPERATION = 'Operation'
# Category captures when we are executing a TRACE/WARMUP/DUMP phase of profiling.
# Can be useful for ignoring parts of the execution (e.g. DUMP).
# constants.CATEGORY_PHASE = 'Phase'
CATEGORY_SIMULATOR_CPP = "Simulator C"
CATEGORY_ATARI = CATEGORY_SIMULATOR_CPP

CATEGORY_PROF_CUPTI = 'Profiling: CUPTI'
CATEGORY_PROF_LD_PRELOAD = 'Profiling: LD_PRELOAD'
CATEGORY_PROF_PYTHON_ANNOTATION = 'Profiling: Python annotation'
CATEGORY_PROF_PYTHON_INTERCEPTION = 'Profiling: Python interception'
CATEGORIES_PROF = {
    CATEGORY_PROF_CUPTI,
    CATEGORY_PROF_LD_PRELOAD,
    CATEGORY_PROF_PYTHON_ANNOTATION,
    CATEGORY_PROF_PYTHON_INTERCEPTION,
}


# Categories that represent C events during a "Python -> C" interception:
#                                  T1       T2
#                                  |        |
#     [        Python.call        ][ C.call ][       Python.return         ]
CATEGORIES_C_EVENTS = {
    CATEGORY_TF_API,
    CATEGORY_SIMULATOR_CPP,
}

CATEGORIES_DEPRECATED = {
    CATEGORY_UNKNOWN,
    CATEGORY_DUMMY_EVENT,
    CATEGORY_PYTHON_PROFILER,
}

CATEGORIES_ALL = {
    CATEGORY_PYTHON,
    CATEGORY_CUDA_API_CPU,
    CATEGORY_GPU,
    CATEGORY_OPERATION,
}.union(CATEGORIES_C_EVENTS).union(CATEGORIES_PROF)

DEFAULT_PHASE = 'default_phase'

CUDA_API_ASYNC_CALLS = {'cudaLaunchKernel', 'cudaMemcpyAsync'}

CATEGORIES_CPU = set([
    CATEGORY_TF_API,
    CATEGORY_PYTHON,
    CATEGORY_CUDA_API_CPU,
    CATEGORY_SIMULATOR_CPP,
    CATEGORY_PYTHON_PROFILER,
])

CATEGORIES_GPU = set([
    CATEGORY_GPU,
])

# Not a category used during tracing;
# represents a group of categories.
CATEGORY_CPU = 'CPU'

CATEGORY_TOTAL = 'Total'

MICROSECONDS_IN_MS = float(1e3)
MICROSECONDS_IN_SECOND = float(1e6)
MILLISECONDS_IN_SECOND = float(1e3)
NANOSECONDS_IN_SECOND = float(1e9)
