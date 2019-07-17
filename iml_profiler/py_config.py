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

DEBUG_REPORTS_PROGRESS = True

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

# Use a custom-built/modified version of TF for benchmarking things.
# Modifies C++ code to make tfprof add less overhead to the critical path.
# CUSTOM_TF = True
# CUSTOM_TF = False
