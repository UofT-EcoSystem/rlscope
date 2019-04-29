from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
from os import environ as ENV

ROOT = _d(_d(_a(__file__)))

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

ANALYSIS_PY = _j(ROOT, "python/scripts", "run_benchmark.py")
GENERATE_INDEX_PY = _j(ROOT, "python/scripts", "generate_plot_index.py")

# Use a custom-built/modified version of TF for benchmarking things.
# Modifies C++ code to make tfprof add less overhead to the critical path.
# CUSTOM_TF = True
# CUSTOM_TF = False
