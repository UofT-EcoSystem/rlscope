"""
RL-Scope configuration settings.

``DEBUG_*`` are useful for enabling certain debug logging statements.
:py:class:`_EnvVars` is for reading environment variables.
"""
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
from os import environ as ENV
import shutil
import re
import ctypes.util
import sys
import os
import textwrap

# NOTE: Don't import logger here since rlscope_logging needs py_config.
# from rlscope.profiler.rlscope_logging import logger
import rlscope

INSTALL_ROOT = _d(os.path.realpath(rlscope.__file__))

CPP_LIB = _j(INSTALL_ROOT, 'cpp', 'lib')
CPP_BIN = _j(INSTALL_ROOT, 'cpp', 'bin')
CPP_INCLUDE = _j(INSTALL_ROOT, 'cpp', 'include')

USE_NUMBA = False
nb = None
if USE_NUMBA:
  import numba as nb

ROOT = _d(_d(_a(__file__)))

RLSCOPE_DIR = ROOT

GIT_REPO_URL = "https://github.com/UofT-EcoSystem/rlscope"

RLSCOPE_TEST_DIR = _j(RLSCOPE_DIR, 'test_results')

DEBUG = False

VERSION_TXT = _j(ROOT, "version.txt")

CLONE = _j(ENV['HOME'], 'clone')
BASELINES_ROOT = _j(CLONE, 'baselines')
BENCH_DQN_PY = _j(BASELINES_ROOT, 'baselines/deepq/experiments/benchmark_dqn.py')

SQL_IMPL = 'psql'
assert SQL_IMPL in {'psql', 'sqlite'}

ANALYSIS_PY = _j(ROOT, "python/scripts", "analyze.py")
GENERATE_INDEX_PY = _j(ROOT, "python/scripts", "generate_rlscope_plot_index.py")

CPP_UNIT_TEST_CMD = 'rls-test'

# If true, when you provide --rlscope-debug, log messages when:
# - A session is created by calling tf.Session()
# - A session is used by calling sess.run()
# - RL-Scope trace-data belonging to a session is dumped
DEBUG_TRACE_SESSION = False

# Verbose debugging: Print calls to set_operation/end_operation
DEBUG_OPERATIONS = False

# Verbose debugging: Print calls to librlscope.so
DEBUG_RLSCOPE_LIB_CALLS = False

# If true, have rls-util-sampler log to stdout the GPU/CPU info it queries.
# This can be noisy.
DEBUG_UTIL_SAMPLER = False

# If true, dump stack when we initially start recording progress
# for a phase during rlscope.prof.report_progress(...)
DEBUG_REPORT_PROGRESS = False
# If true, dump stack for ALL calls to rlscope.prof.report_progress(...) that update progress
# (not just the first call for a phase)
DEBUG_REPORT_PROGRESS_ALL = False

# If True, print out how event are grouped into OpStack's when split_op_stacks.
DEBUG_SPLIT_STACK_OPS = False
# If True, skip inserting any profiling-overhead events.
DEBUG_SKIP_PROFILING_OVERHEAD = False

DEBUG_GPU_HW = False

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
if 'RLSCOPE_LOGGED_IN' in ENV:
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

RLSCOPE_DISABLE_JIT = EnvVars.get_bool('RLSCOPE_DISABLE_JIT', dflt=False)

# Use numbafied implementation of event overlap computation.
RLSCOPE_USE_NUMBA = EnvVars.get_bool('RLSCOPE_USE_NUMBA', dflt=False)
# More verbose debugging information during unit-tests.
RLSCOPE_DEBUG_UNIT_TESTS = EnvVars.get_bool('RLSCOPE_DEBUG_UNIT_TESTS', dflt=False)
RLSCOPE_DEBUG_UNIQUE_SPLITS_BASE = EnvVars.get_string('RLSCOPE_DEBUG_UNIQUE_SPLITS_BASE', dflt=None)

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
  import numpy as np
  NUMPY_TIME_USEC_TYPE = np.int64
  NUMBA_TIME_USEC_TYPE = nb.int64

  NUMPY_TIME_PSEC_TYPE = np.int64
  NUMBA_TIME_PSEC_TYPE = nb.int64

  NUMPY_CATEGORY_KEY_TYPE = np.int64
  NUMBA_CATEGORY_KEY_TYPE = nb.int64

def is_development_mode():
  return shutil.which('rlscope-is-development-mode') is not None

def is_production_mode():
  return not is_development_mode()

def yes_as_bool(yes_or_no):
  if yes_or_no.lower() in {'yes', 'y', 'on', '1'}:
    return True
  return False

def is_running_unit_tests():
  return yes_as_bool(os.environ.get('RLS_RUNNING_UNIT_TESTS', 'no'))

def read_rlscope_version():
  lines = []
  with open(VERSION_TXT) as f:
      for line in f:
        line = line.rstrip()
        if re.search(r'^\s*#.*', line):
          continue
        lines.append(line)
  if len(lines) == 0:
    return None
  return ''.join(lines)

if 'pytest' in sys.modules and 'RLS_RUNNING_UNIT_TESTS' not in os.environ:
  # Prevent users from running with "pytest" so we can detect in-code if we're running with pytest.
  # Useful for avoiding loading modules where optional imports may be used (e.g., "import tensorflow").
  # logger.error(textwrap.dedent("""
  sys.stderr.write(textwrap.dedent("""\
  Don't run \"pytest\" directly; instead:
  
    # Run Python unit tests using pytest:
    $ rls-unit-tests --tests py
    
    # Or, run Python and C++ unit tests:
    $ rls-unit-tests
  """))
  sys.exit(1)
