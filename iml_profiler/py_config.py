from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
from os import environ as ENV

ROOT = _d(_d(_a(__file__)))

IML_DIR = ROOT

IML_TEST_DIR = _j(IML_DIR, 'test_results')

DEBUG = False

BUILD_DIR = None
DEBUG_BUILD_DIR = _j(ROOT, "Debug")
RELEASE_BUILD_DIR = _j(ROOT, "Release")
if _e(DEBUG_BUILD_DIR):
  BUILD_DIR = DEBUG_BUILD_DIR
elif _e(RELEASE_BUILD_DIR):
  BUILD_DIR = RELEASE_BUILD_DIR

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

# If True, then log all calls into libsample_cuda_api.so
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

LIB_SAMPLE_CUDA_API = _j(ROOT, 'build', 'libsample_cuda_api.so')

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

# Use numbafied implementation of event overlap computation.
IML_USE_NUMBA = EnvVars.get_bool('IML_USE_NUMBA', dflt=False)
# More verbose debugging information during unit-tests.
IML_DEBUG_UNIT_TESTS = EnvVars.get_bool('IML_DEBUG_UNIT_TESTS', dflt=False)

# NOTE: Don't touch this, it gets set manually by unit-tests.
IS_UNIT_TEST = False
