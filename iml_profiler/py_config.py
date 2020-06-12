from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
from os import environ as ENV
import numpy as np
import ctypes.util
import sys
import textwrap



USE_NUMBA = False
nb = None
if USE_NUMBA:
  import numba as nb

ROOT = _d(_d(_a(__file__)))

IML_DIR = ROOT

IML_TEST_DIR = _j(IML_DIR, 'test_results')

DEBUG = False

RLSCOPE_LIBNAME = 'rlscope'
# Older version of python (<=3.6) need 'LIBRARY_PATH' to be defined for find_library to work.
assert 'LIBRARY_PATH' not in ENV or ENV['LIBRARY_PATH'] == ENV['LD_LIBRARY_PATH']
ENV['LIBRARY_PATH'] = ENV['LD_LIBRARY_PATH']
RLSCOPE_CLIB = ctypes.util.find_library(RLSCOPE_LIBNAME)
# RLSCOPE_CLIB = 'lib{name}.so'.format(
#   name=RLSCOPE_LIBNAME)

# if not _e(so_path):
if RLSCOPE_CLIB is None:
  sys.stderr.write(textwrap.dedent("""
      IML ERROR: couldn't find RLScope library (lib{name}.so); to build it, do:
        $ cd {root}
        $ bash ./setup.sh
        # To modify your LD_LIBRARY_PATH to include lib{name}.so, run:
        $ source source_me.sh
      """.format(
    name=RLSCOPE_LIBNAME,
    root=ROOT,
  )))
  sys.exit(1)

CLONE = _j(ENV['HOME'], 'clone')
BASELINES_ROOT = _j(CLONE, 'baselines')
BENCH_DQN_PY = _j(BASELINES_ROOT, 'baselines/deepq/experiments/benchmark_dqn.py')

SQL_IMPL = 'psql'
assert SQL_IMPL in {'psql', 'sqlite'}

ANALYSIS_PY = _j(ROOT, "python/scripts", "analyze.py")
GENERATE_INDEX_PY = _j(ROOT, "python/scripts", "generate_iml_profiler_plot_index.py")

# If true, when you provide --iml-debug, log messages when:
# - A session is created by calling tf.Session()
# - A session is used by calling sess.run()
# - IML trace-data belonging to a session is dumped
DEBUG_TRACE_SESSION = False

# Verbose debugging: Print calls to set_operation/end_operation
DEBUG_OPERATIONS = False

# If true, have iml-util-sampler log to stdout the GPU/CPU info it queries.
# This can be noisy.
DEBUG_UTIL_SAMPLER = False

# If true, dump stack when we initially start recording progress
# for a phase during iml.prof.report_progress(...)
DEBUG_REPORT_PROGRESS = False
# If true, dump stack for ALL calls to iml.prof.report_progress(...) that update progress
# (not just the first call for a phase)
DEBUG_REPORT_PROGRESS_ALL = False

# If True, print out how event are grouped into OpStack's when split_op_stacks.
DEBUG_SPLIT_STACK_OPS = False
# If True, skip inserting any profiling-overhead events.
DEBUG_SKIP_PROFILING_OVERHEAD = False

# If True, then log all calls into librlscope.so
# DEBUG_SAMPLE_CUDA_API = True

# If True, then log all calls into a wrapped library (e.g. TensorFlow, simulator, etc.).
# DEBUG_C_LIB_CALLS = True

# In case docker causes different runtime behaviour, we can use this switch.
#
# Current issues:
# - nvidia-smi doesn't work properly from within a container.
#   Docker uses a DIFFERENT pid namespace within the container.
#   i.e. the pid of the proc in the container will be DIFFERENT
#   from the proc outside the the container, when you look at it with 'top'.
#   HOWEVER, nvidia-smi within the container still reports the pid of process
#   OUTSIDE the container.
#   - This bug affects GPU memory/compute utilization sampling;
#     if we're in a docker container just sample the entire memory usage of a GPU.
IS_DOCKER = False
if 'IML_LOGGED_IN' in ENV:
  IS_DOCKER = True

# Print the names of all the TensorFlow/Simulator function calls that get wrapped.
DEBUG_WRAP_CLIB = False

# Print information about delays that happen during Profiler.finish().
DEBUG_CRITICAL_PATH = False

# LIB_SAMPLE_CUDA_API = None
# if _e(_j(ROOT, 'Debug', RLSCOPE_LIB)):
#   LIB_SAMPLE_CUDA_API = _j(ROOT, 'Debug', RLSCOPE_LIB)
# elif _e(_j(ROOT, 'Release', RLSCOPE_LIB)):
#   LIB_SAMPLE_CUDA_API = _j(ROOT, 'Release', RLSCOPE_LIB)
# elif _e(_j(ROOT, 'build', RLSCOPE_LIB)):
#   LIB_SAMPLE_CUDA_API =  _j(ROOT, 'build', RLSCOPE_LIB)

# Use a custom-built/modified version of TF for benchmarking things.
# Modifies C++ code to make tfprof add less overhead to the critical path.
# CUSTOM_TF = True
# CUSTOM_TF = False

class _EnvVars:
  """
  Typed wrapper around os.environ

  # E.g.
  $ export DEBUG=yes
  # OR
  $ export DEBUG=true
  ...
  >>> EnvVars.get_bool('DEBUG')
  True

  # E.g.
  $ export DEBUG=0
  # OR
  $ export DEBUG=false
  ...
  >>> EnvVars.get_bool('DEBUG')
  False
  """
  def __init__(self, env=None):
    if env is None:
      env = os.environ
    self.env = env

  def _as_bool(self, string):
    string = string.lower()
    if string in {'0', 'no', 'false', 'None'}:
      return False
    return True

  def _as_int(self, string):
    return int(string)

  def _as_float(self, string):
    return float(string)

  def _as_string(self, string):
    return string

  def _as_number(self, string):
    try:
      val = int(string)
      return val
    except ValueError:
      pass

    val = float(string)
    return val

  def _get_typed(self, var, converter, dflt):
    if var not in self.env:
      return dflt
    string = self.env[var]
    val = converter(string)
    return val

  def get_string(self, var, dflt=None):
    val = self._get_typed(var, self._as_string, dflt)
    return val

  def get_bool(self, var, dflt=False):
    val = self._get_typed(var, self._as_bool, dflt)
    return val

  def get_int(self, var, dflt=0):
    val = self._get_typed(var, self._as_int, dflt)
    return val

  def get_float(self, var, dflt=0.):
    val = self._get_typed(var, self._as_float, dflt)
    return val

  def get_number(self, var, dflt=0.):
    val = self._get_typed(var, self._as_number, dflt)
    return val
EnvVars = _EnvVars(env=ENV)

NUMBA_DISABLE_JIT = EnvVars.get_bool('NUMBA_DISABLE_JIT', dflt=False)

IML_DISABLE_JIT = EnvVars.get_bool('IML_DISABLE_JIT', dflt=False)

# Use numbafied implementation of event overlap computation.
IML_USE_NUMBA = EnvVars.get_bool('IML_USE_NUMBA', dflt=False)
# More verbose debugging information during unit-tests.
IML_DEBUG_UNIT_TESTS = EnvVars.get_bool('IML_DEBUG_UNIT_TESTS', dflt=False)
IML_DEBUG_UNIQUE_SPLITS_BASE = EnvVars.get_string('IML_DEBUG_UNIQUE_SPLITS_BASE', dflt=None)

HOST_INSTALL_PREFIX = _j(ROOT, "local.host")
DOCKER_INSTALL_PREFIX = _j(ROOT, "local.docker")

HOST_BUILD_PREFIX = _j(ROOT, "build.host")
DOCKER_BUILD_PREFIX = _j(ROOT, "build.docker")

# NOTE: Don't touch this, it gets set manually by unit-tests.
IS_UNIT_TEST = False

# NUMPY_TIME_USEC_TYPE = np.uint64
# NUMBA_TIME_USEC_TYPE = nb.uint64
#
# NUMPY_CATEGORY_KEY_TYPE = np.uint64
# NUMBA_CATEGORY_KEY_TYPE = nb.uint64

# NOTE: numba likes to assume np.int64 as the default platform type when dealing with integers
# (e.g. indices in a "for in range(n)" loop, "x = 0" statements).
# As a result, attempts to use np.uint64 result in lots of compilation errors.
# I haven't bothered too hard to try to fix them... hopefully an int64 is big enough for our purposes...
# should probably add a check for that!

if USE_NUMBA:
  NUMPY_TIME_USEC_TYPE = np.int64
  NUMBA_TIME_USEC_TYPE = nb.int64

  NUMPY_TIME_PSEC_TYPE = np.int64
  NUMBA_TIME_PSEC_TYPE = nb.int64

  NUMPY_CATEGORY_KEY_TYPE = np.int64
  NUMBA_CATEGORY_KEY_TYPE = nb.int64
