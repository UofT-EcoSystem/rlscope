#!/usr/bin/env bash
set -e

# To avoid running everything more than once, we will stick ALL (algo, env_id) pairs inside of $IML_DIR/output/iml_bench/all.
#
# (1) On vs off policy:
# Run Atari Pong on all environments that support it (that we have annotated):
# - Ppo2 [sort of on-policy]
# - A2c [on-policy]
# - Dqn [off-policy]
# We can use this to compare on-policy vs off-policy
#
# (2) Compare environments:
# Run ALL bullet environments for at least one algorithm (ppo2).
#
# (3) Compare algorithms:
# Run ALL algorithms on 1 bullet environment (Walker)
#
# (4) Compare all RL workloads:
# Run ALL algorithms on ALL bullet environments

ROOT="$(readlink -f $(dirname "$0"))"

DRY_RUN=${DRY_RUN:-no}
DEBUG=${DEBUG:-no}

STABLE_BASELINES_DIR=${STABLE_BASELINES_DIR:-$HOME/clone/stable-baselines}
IML_DIR=${IML_DIR:-$HOME/clone/iml}
RL_BASELINES_ZOO_DIR=${RL_BASELINES_ZOO_DIR:-$HOME/clone/rl-baselines-zoo}
TF_AGENTS_DIR=${TF_AGENTS_DIR:-$HOME/clone/agents}
REAGENT_DIR=${REAGENT_DIR:-$HOME/clone/ReAgent}
#ENJOY_TRT=${ENJOY_TRT:-${RL_BASELINES_ZOO_DIR}/enjoy_trt.py}

if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

CALIB_DIR=$IML_DIR/calibration/microbench
CALIB_OPTS=( \
    --cupti-overhead-json $CALIB_DIR/cupti_overhead.json \
    --LD-PRELOAD-overhead-json $CALIB_DIR/LD_PRELOAD_overhead.json \
    --python-clib-interception-tensorflow-json $CALIB_DIR/category_events.python_clib_interception.tensorflow.json \
    --python-clib-interception-simulator-json $CALIB_DIR/category_events.python_clib_interception.simulator.json \
    --python-annotation-json $CALIB_DIR/category_events.python_annotation.json \
    )

_run_bench() {
    local all_dir=$IML_DIR/output/iml_bench/all

    _do iml-bench --dir $all_dir "$@"
}
old_run_stable_baselines() {
    sb_all_reps "$@"
}
sb_train() {
    _run_bench "$@" stable-baselines
}

iml_analyze() {
    iml-analyze "${CALIB_OPTS[@]}" "$@"
}

run_total_training_time_plot() {
    _do iml-quick-expr --expr plot_fig --fig fig_13_overhead_correction \
        "${CALIB_OPTS[@]}" \
        --root-dir $IML_DATA_DIR/iml/output/expr_total_training_time \
        --debug-memoize "$@"
}

sb_analyze() {
    _run_bench "$@" --analyze stable-baselines "${CALIB_OPTS[@]}"
}

sb_plot() {
    _run_bench "$@" --analyze stable-baselines "${CALIB_OPTS[@]}" --mode plot
}

sb_all_reps() {
    "$@" --repetition 1
    "$@" --repetition 2
    "$@" --repetition 3
}
sb_reps_first() {
    sb_all_reps sb_train "$@"
    sb_all_reps sb_analyze "$@"
    sb_all_reps sb_plot "$@"
}

sb_one_rep() {
    sb_train "$@"
    sb_analyze "$@"
    # SKIP plotting until we fix it, so we can process all the repetitions over night!
#    sb_plot "$@"
}
sb_one_rep_plot() {
#    sb_train "$@"
#    sb_analyze "$@"
    sb_plot "$@"
}
sb_reps_last() {
    sb_one_rep --repetition 1 "$@"
    sb_one_rep --repetition 2 "$@"
    sb_one_rep --repetition 3 "$@"
}

# 1. Algo choice - medium complexity (Walker2D)
#    - 511 + 383 + 270 + 44 = 1,208
# 2. Env choice (PPO2):
#    - 1.4 + 4.8 + 14 + 14 + 23 + 124 + 489 + 511 + 565 = 1,746.2
# 3. Algo choice - low complexity (MountainCar)
#    - 1.4 + 2.9 + 3.0 + 14 + 100 + 292 + 313 = 726.3
# 4. All RL workloads
#    - NOTE: we SHOULD be able to skip analyzing all (algo, env) pairs...
#    - 1.4 + 2.9 + 3.0 + 14 + 100 + 292 + 313 = 726.3
run_fig_algorithm_choice_1a_med_complexity() {
    # logan
    sb_reps_last --algo-env-group algorithm_choice_1a_med_complexity "$@"
}
run_fig_all_rl_workloads_plot_01() {
    # logan
    sb_one_rep_plot --repetition 1 "$@" --algo-env-group all_rl_workloads "$@"
}
run_fig_algorithm_choice_1a_med_complexity_plot_01() {
    # logan
    sb_one_rep_plot --repetition 1 "$@" --algo-env-group algorithm_choice_1a_med_complexity "$@"
}
run_fig_dqn_detailed_plot_01() {
    # eco-13
    sb_one_rep_plot --repetition 1 --algo-env-group dqn_detailed "$@"
}

run_fig_environment_choice() {
    # eco-13
    sb_reps_last --algo-env-group environment_choice "$@"
}
run_fig_environment_choice_plot_01() {
    # eco-13
    sb_one_rep_plot --repetition 1 --algo-env-group environment_choice "$@"
}

run_fig_algorithm_choice_1b_low_complexity() {
    # eco-14
    sb_reps_last --algo-env-group algorithm_choice_1b_low_complexity "$@"
}
run_fig_algorithm_choice_1b_low_complexity_plot_01() {
    # eco-14
    sb_one_rep_plot --repetition 1 --algo-env-group algorithm_choice_1b_low_complexity "$@"
}

#run_fig_all_rl_workloads() {
#    sb_reps_last --algo-env-group all_rl_workloads "$@"
#}
run_stable_baselines_all() {
    run_fig_algorithm_choice_1a_med_complexity "$@"
    run_fig_environment_choice "$@"
    run_fig_algorithm_choice_1b_low_complexity "$@"
    # SKIP: until we can avoid runnning iml-analyze on all (algo, env) pairs.
#    run_fig_all_rl_workloads "$@"
}
#run_fig_on_vs_off_policy() {
#    sb_reps_last --algo-env-group on_vs_off_policy "$@"
#}

RUN_DEBUG_ARGS=(--subdir debug)
INSTRUMENTED_ARGS=(--instrumented)

_run_debug_ppo_full_training() {
    local num_training_loop_iterations=1
    local n_envs=8
    local n_steps=2048
    iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr total_training_time --repetitions 1 --bullet --instrumented --n-timesteps $((n_envs*n_steps*num_training_loop_iterations)) --algo ppo2 --env MinitaurBulletEnv-v0 "$@"
}
_run_debug_dqn_full_training() {
    iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr total_training_time --repetitions 1 --instrumented --n-timesteps 20000 --algo dqn --env PongNoFrameskip-v4 "$@"
}
_run_debug_sac_full_training() {
    iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr total_training_time --repetitions 1 --instrumented --n-timesteps 20000 --algo sac --env AntBulletEnv-v0 "$@"
}
_run_debug_a2c_full_training() {
    iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr total_training_time --repetitions 1 --instrumented --n-timesteps 20000 --algo a2c --env HopperBulletEnv-v0 "$@"
}
_run_debug_ddpg_full_training() {
    iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr total_training_time --repetitions 1 --instrumented --n-timesteps 20000 --algo ddpg --env Walker2DBulletEnv-v0 "$@"
}

fig_9_simulator_choice_environments() {
  # echo AirLearningEnv
  # echo PongNoFrameskip-v4
  echo AntBulletEnv-v0
  echo HalfCheetahBulletEnv-v0
  echo HopperBulletEnv-v0
  echo Walker2DBulletEnv-v0
}
fig_9_simulator_choice_algo() {
  echo ppo2
}

fig_11_rl_framework_choice_environment() {
  echo Walker2DBulletEnv-v0
}
fig_10_algo_choice_environment() {
  echo Walker2DBulletEnv-v0
}
fig_10_algo_choice_algorithms() {
  echo a2c
  echo ddpg
  echo ppo2
  echo sac
}
tf_agents_fig_9_simulator_choice_algo() {
  echo ddpg
}
tf_agents_fig_10_algo_choice_algos() {
  echo ddpg
  # Disable until --stable_baselines_hyperparams is supported...
#  echo sac
#  echo td3
}
tf_agents_fig_11_rl_framework_choice_algos() {
  echo td3
}
stable_baselines_fig_11_rl_framework_choice_algos() {
  echo a2c
  echo ddpg
  echo ppo2
  echo sac
}
stable_baselines_fig_10_algo_choice_algos() {
  echo a2c
  echo ddpg
  echo ppo2
  echo sac
}
stable_baselines_fig_9_simulator_choice_algo() {
  echo ddpg
}

reagent_fig_11_rl_framework_choice_algos() {
  echo td3
}
reagent_fig_10_algo_choice_algos() {
  echo td3
}
reagent_fig_9_simulator_choice_algo() {
  echo td3
}


run_debug_ppo_full_training_instrumented() {
    _run_debug_ppo_full_training "${INSTRUMENTED_ARGS[@]}" "$@"
}
run_debug_dqn_full_training_instrumented() {
    _run_debug_dqn_full_training "${INSTRUMENTED_ARGS[@]}" "$@"
}
run_debug_sac_full_training_instrumented() {
    _run_debug_sac_full_training "${INSTRUMENTED_ARGS[@]}" "$@"
}
run_debug_a2c_full_training_instrumented() {
    _run_debug_a2c_full_training "${INSTRUMENTED_ARGS[@]}" "$@"
}
run_debug_ddpg_full_training_instrumented() {
    _run_debug_ddpg_full_training "${INSTRUMENTED_ARGS[@]}" "$@"
}

run_debug_ppo_full_training_uninstrumented() {
    _run_debug_ppo_full_training "$@"
}
run_debug_dqn_full_training_uninstrumented() {
    _run_debug_dqn_full_training "$@"
}
run_debug_sac_full_training_uninstrumented() {
    _run_debug_sac_full_training "$@"
}
run_debug_a2c_full_training_uninstrumented() {
    _run_debug_a2c_full_training "$@"
}
run_debug_ddpg_full_training_uninstrumented() {
    _run_debug_ddpg_full_training "$@"
}

run_debug_full_training_instrumented() {
    run_debug_ppo_full_training_instrumented
    run_debug_dqn_full_training_instrumented
    run_debug_sac_full_training_instrumented
    run_debug_a2c_full_training_instrumented
    run_debug_ddpg_full_training_instrumented
}

run_debug_full_training_uninstrumented() {
    run_debug_ppo_full_training_uninstrumented
    run_debug_dqn_full_training_uninstrumented
    run_debug_sac_full_training_uninstrumented
    run_debug_a2c_full_training_uninstrumented
    run_debug_ddpg_full_training_uninstrumented
}

run_full_training_instrumented() {
    iml-quick-expr --expr total_training_time --instrumented "$@"
}
run_full_training_uninstrumented() {
    iml-quick-expr --expr total_training_time "$@"
}

run_calibration_all() {
    run_subtraction_calibration "$@"
    run_subtraction_validation "$@"
}

NUM_REPS=5
run_subtraction_validation() {
    if [ $# -lt 2 ]; then
        echo "ERROR: Saw $# arguments, but expect:"
        echo "  run_subtraction_validation <algo> <environ>"
        return 1
    fi
    # Default in the past: (HalfCheetahBulletEnv-v0, ppo2)
    local algo="$1"
    local environ="$2"
    shift 2
    _cmd() {
        _do iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr subtraction_validation --calibration-mode validation --repetitions $NUM_REPS --only-runs 2 --env "$environ" --algo "$algo" "$@"
    }
    _cmd "$@"
    _cmd "$@" --plot
}

run_pyprof_calibration_fig_algorithm_choice_1a_med_complexity() {
    iml-quick-expr --expr microbenchmark --algo-env-group algorithm_choice_1a_med_complexity "$@"
}

run_pyprof_calibration_fig_environment_choice() {
    iml-quick-expr --expr microbenchmark --algo-env-group environment_choice "$@"
}

run_pyprof_calibration_iterations_20000() {
    iml-quick-expr --expr microbenchmark --env HalfCheetahBulletEnv-v0 --algo ppo2 --iterations $((2*10**4)) "$@"
}

run_pyprof_calibration_all() {
    run_pyprof_calibration_fig_environment_choice "$@"
    run_pyprof_calibration_fig_algorithm_choice_1a_med_complexity "$@"
    run_pyprof_calibration_iterations_20000 "$@"
}

run_subtraction_calibration() {
#    if [ $# -lt 2 ]; then
#        echo "ERROR: Saw $# arguments, but expect:"
#        echo "  run_subtraction_calibration <algo> <environ>"
#        return 1
#    fi
#    # Default in the past: (HalfCheetahBulletEnv-v0, ppo2)
#    local algo="$1"
#    local environ="$2"
#    shift 2

    if [ $# -eq 0 ]; then
        echo "ERROR: Saw $# arguments, but expected arguments that select specific (algo, env) combinations like:"
        echo "  run_subtraction_calibration --algo ppo2 --env HalfCheetahBulletEnv-v0"
        echo "  run_subtraction_calibration --algo ppo2 --env Walker2DBulletEnv-v0"
        echo "  run_subtraction_calibration --algo-env-group environment_choice"
        return 1
    fi

    _cmd() {
        _do iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr subtraction_validation --calibration-mode calibration --repetitions $NUM_REPS --only-runs 2 "$@"
    }
    _cmd "$@"
    _cmd "$@" --plot
}

run_subtraction_validation_long() {
    _cmd() {
        _do iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr subtraction_validation --calibration-mode validation --repetitions $NUM_REPS --env HalfCheetahBulletEnv-v0 --algo ppo2 "$@"
    }
    # - Run using long iterations (most important...)
    # - Re-plot stuff.
    # - Run using different numbers of iterations
    # - Re-plot stuff.
    _cmd "$@" --num-runs 0 --long-run
    _cmd "$@" --num-runs 0 --long-run --plot
#    _cmd "$@" --num-runs 3
#    _cmd "$@" --num-runs 3 --plot
}

run_debug_all() {
    run_debug_full_training_uninstrumented
    run_debug_full_training_instrumented
}

# If we want to include dqn in every single figure, we need to show a low complexity simulator (simulator time will be understated).
# If we want to show a medium complexity simulator (Walker2D) we cannot include dqn.
# So, lets do both and decide later.
# NOTE: we will run both LunarLander and LunarLanderContinuous for ppo so we can make sure these environment have identical complexity (except for the input type)
fig_1a_all_algo() {
    # (1a) All algorithms (low complexity): LunarLander
    # GOAL: 3 repetitions
    iml-quick-expr --lunar "$@"
}
fig_1b_all_algo() {
    # (1b) Mostly all algorithms (except dqn) (medium complexity): Walker2D
    # GOAL: 3 repetitions
    iml-quick-expr --env Walker2DBulletEnv-v0 "$@"
}
fig_2_all_env() {
    # (2) All environments (low and medium complexity): ppo2
    # GOAL: 3 repetitions
    iml-quick-expr --lunar --atari --bullet --algo ppo2 "$@"
}
fig_3_all_algo_env()  {
    # (3) All (algo, env) combos
    # GOAL: 1 repetition
    iml-quick-expr --lunar --atari --bullet "$@"
}
fig_all() {
    fig_1a_all_algo "$@"
    fig_1b_all_algo "$@"
    fig_2_all_env "$@"
    fig_3_all_algo_env "$@"
}
fig_all_algo_then_all_env() {
    fig_1a_all_algo "$@"
    fig_1b_all_algo "$@"
    fig_2_all_env "$@"
}
_run_reps() {
    "$@" --repetitions 1
    "$@" --repetitions 2
    "$@" --repetitions 3
}

run_all() {
    # Need 1 uninstrumented run for all (algo, env) pairs for generating nvidia-smi utilization figure (Figure 8)
    fig_all --expr total_training_time --repetitions 1 "$@"
    # Prioritize uninstrumented runs (needed for some figures).
    fig_all_algo_then_all_env --expr total_training_time --repetitions 1 "$@"
    # For generating non-extrapolated overhead breakdown.
    fig_all_algo_then_all_env --expr total_training_time --repetitions 1 --instrumented "$@"

    # For getting error bounds on "extrapolation justification" figure.
    fig_all_algo_then_all_env --expr total_training_time --repetitions 3 "$@"
}

run_perf_debug() {
    _do iml-analyze --iml-directory $IML_DIR/output/perf_debug "${CALIB_OPTS[@]}"
}

run_perf_debug_short() {
    _do iml-analyze --iml-directory $IML_DIR/output/perf_debug_short "${CALIB_OPTS[@]}"
}

_find_extension() {
    local root_dir="$1"
    local extension_regex="$2"
    shift 2
    local regex="/(.*\.(${extension_regex}))$"
    find "$root_dir" -type f | grep --perl-regexp "$regex"
}
setup_cmakelists_windows() {

    local src_ext_regex="cpp|cc|c|cu"
    local hdr_ext_regex="h|hpp|cuh"
    local all_ext_regex="${src_ext_regex}|${hdr_ext_regex}"

    local include_dirs=( \
        ./windows_includes \
        ./windows_includes/CUPTI/include \
        ./windows_includes/cuda/include \
        ./build \
        ./src \
        ./tensorflow \
    )

    local src_dirs=( \
        ./src \
        ./tensorflow \
        ./test \
    )

    local path=CMakeLists.windows.txt

    if [ -e "$path" ]; then
        rm "$path"
    fi

# Header already included by main CMakeLists.txt file.
#
#    cat <<EOF >> "$path"
#cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
#project(iml)
##https://github.com/robertmaynard/code-samples/blob/master/posts/cmake/CMakeLists.txt
#
## We want to be able to do std::move() for lambda captures (c++11 doesn't have that).
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#
#EOF

    echo "include_directories(" >> "$path"
    for direc in "${include_dirs[@]}"; do
        echo "    $direc" >> "$path"
    done
    echo ") # include_directories" >> "$path"
    echo "" >> "$path"

    echo "add_executable(dummy_exe" >> "$path"
    for src_dir in "${src_dirs[@]}"; do
        _find_extension "$src_dir" "$src_ext_regex" | sed 's/^/  /' >> "$path"
    done
    echo ") # add_executable" >> "$path"
    echo "" >> "$path"

    echo "> Success: wrote $path"
}

all_run() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}
  subdir=${subdir:-}
  fig=${fig:-all}
  framework=${framework:-all}

  if [ "$framework" = 'tf_agents' ] || [ "$framework" = 'all' ]; then
    all_run_tf_agents
  fi
  if [ "$framework" = 'stable_baselines' ] || [ "$framework" = 'all' ]; then
    all_run_stable_baselines
  fi
  if [ "$framework" = 'reagent' ] || [ "$framework" = 'all' ]; then
    all_run_reagent
  fi

)
}

all_run_tf_agents() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}
  subdir=${subdir:-}
  fig=${fig:-all}

  if [ "$fig" = 'framework_choice' ] || [ "$fig" = 'all' ]; then

    # Fig 11: RL framework choice
    for algo in $(fig_framework_choice_algos); do
      for env_id in $(fig_framework_choice_envs); do
        # for use_tf_functions in no; do
        for use_tf_functions in yes no; do
          run_tf_agents
        done
      done
    done

    # Fig 11(b): RL framework choice DDPG
    for algo in $(fig_framework_choice_algos_ddpg); do
      for env_id in $(fig_framework_choice_envs); do
        # for use_tf_functions in no; do
        for use_tf_functions in yes no; do
          run_tf_agents
        done
      done
    done
  fi

  # NOTE: we generate algo and simulator choice using stable-baselines.

#  if [ "$fig" = 'algo_choice' ] || [ "$fig" = 'all' ]; then
#    # Fig 10: Algorithm choice
#    # tf-agents
#    #
#    # ppo doesn't work (BUG in tf-agents)
#    for algo in $(tf_agents_fig_10_algo_choice_algos); do
#      env_id="$(fig_10_algo_choice_environment)"
#      for use_tf_functions in yes no; do
#        run_tf_agents
#      done
#    done
#  fi
#
#  if [ "$fig" = 'simulator_choice' ] || [ "$fig" = 'all' ]; then
#    # Fig 9: Simulator choice
#    # tf-agents
#    #
#    # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
#    for env_id in $(fig_9_simulator_choice_environments); do
#      algo=$(tf_agents_fig_9_simulator_choice_algo)
#      for use_tf_functions in yes no; do
#        run_tf_agents
#      done
#    done
#  fi

#  if [ "$fig" = 'algo_choice' ] || [ "$fig" = 'all' ]; then
#    plot_tf_agents_fig_10_algo_choice
#  fi
#  if [ "$fig" = 'simulator_choice' ] || [ "$fig" = 'all' ]; then
#    plot_tf_agents_fig_9_simulator_choice
#  fi

)
}

all_run_reagent() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}
  subdir=${subdir:-}
  fig=${fig:-all}

  if [ "$fig" = 'framework_choice' ] || [ "$fig" = 'all' ]; then
    # Fig 11: RL framework choice
    for algo in $(fig_framework_choice_algos); do
      for env_id in $(fig_framework_choice_envs); do
        run_reagent
      done
    done

    # NOTE: no Fig 11(b): doesn't implement DDPG.
  fi

#  # Fig 10: Algorithm choice
#  # tf-agents
#  #
#  # ppo doesn't work (BUG in tf-agents)
#  for algo in $(reagent_fig_10_algo_choice_algos); do
#    env_id="$(fig_10_algo_choice_environment)"
#    run_reagent
#  done
#
#  # Fig 9: Simulator choice
#  # tf-agents
#  #
#  # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
#  for env_id in $(fig_9_simulator_choice_environments); do
#    algo=$(reagent_fig_9_simulator_choice_algo)
#    run_reagent
#  done

  if [ "$fig" = 'framework_choice' ] || [ "$fig" = 'all' ]; then
    plot_reagent
  fi

)
}

all_run_stable_baselines() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}
  fig=${fig:-all}

  if [ "$fig" = 'framework_choice' ] || [ "$fig" = 'all' ]; then
    # Fig 11: RL framework choice
    for algo in $(fig_framework_choice_algos); do
      for env_id in $(fig_framework_choice_envs); do
        run_stable_baselines
      done
    done

    # Fig 11(b): RL framework choice DDPG
    for algo in $(fig_framework_choice_algos_ddpg); do
      for env_id in $(fig_framework_choice_envs); do
        run_stable_baselines
      done
    done
  fi

  if [ "$fig" = 'algo_choice' ] || [ "$fig" = 'all' ] || [ "$fig" = 'algo_env' ]; then
    # Fig 10: Algorithm choice
    # tf-agents
    #
    # ppo doesn't work (BUG in tf-agents)
    for algo in $(stable_baselines_fig_10_algo_choice_algos); do
      env_id="$(fig_10_algo_choice_environment)"
      run_stable_baselines
    done
  fi

  if [ "$fig" = 'simulator_choice' ] || [ "$fig" = 'all' ] || [ "$fig" = 'algo_env' ]; then
    # Fig 9: Simulator choice
    # tf-agents
    #
    # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
    for env_id in $(fig_9_simulator_choice_environments); do
      algo=$(stable_baselines_fig_9_simulator_choice_algo)
      run_stable_baselines
    done
  fi

#  if [ "$fig" = 'algo_choice' ] || [ "$fig" = 'all' ] || [ "$fig" = 'algo_env' ]; then
#    plot_stable_baselines_fig_10_algo_choice
#  fi
#  if [ "$fig" = 'simulator_choice' ] || [ "$fig" = 'all' ] || [ "$fig" = 'algo_env' ]; then
#    plot_stable_baselines_fig_9_simulator_choice
#  fi

)
}

fig_framework_choice_algos() {
  echo td3
}
fig_framework_choice_algos_ddpg() {
  echo ddpg
}
fig_framework_choice_envs() {
  echo Walker2DBulletEnv-v0
}

gen_tex_framework_choice() {
(
  set -eu

  dry_run=${dry_run:-no}

  args=(
    iml-analyze
    --task TexMetricsTask
    --directory $(framework_choice_plots_direc)
    --framework-choice-csv $(framework_choice_plots_direc)/OverlapStackedBarPlot.*.operation_training_time.csv
    --framework-choice-ddpg-csv $(framework_choice_plots_ddpg_direc)/OverlapStackedBarPlot.*.operation_training_time.csv
    --framework-choice-trans-csv $(framework_choice_plots_direc)/CategoryTransitionPlot.combined.csv
    --framework-choice-ddpg-trans-csv $(framework_choice_plots_ddpg_direc)/CategoryTransitionPlot.combined.csv
  )
  if [ "${dry_run}" = 'yes' ]; then
    args+=(
      --dry-run
    )
  fi
  if [ "${DEBUG}" = 'yes' ]; then
    args+=(--pdb --debug --debug-single-thread)
  fi
  _do "${args[@]}" "$@"

)
}

gen_tex_algo_choice() {
(
  set -eu

  dry_run=${dry_run:-no}

  args=(
    iml-analyze
    --task TexMetricsTask
    --directory $(stable_baselines_plots_direc)/stable_baselines_fig_10_algo_choice
    --algo-choice-csv $(stable_baselines_plots_direc)/stable_baselines_fig_10_algo_choice/OverlapStackedBarPlot.*.operation_training_time.csv
  )
  if [ "${dry_run}" = 'yes' ]; then
    args+=(
      --dry-run
    )
  fi
  if [ "${DEBUG}" = 'yes' ]; then
    args+=(--pdb --debug --debug-single-thread)
  fi
  _do "${args[@]}" "$@"

)
}

plot_framework_choice() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-3}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}

  # Fig: Framework comparison
  # WANT: plot:
  # - (DDPG, Walker)
  #   - stable-baselines
  #   - tf-agents: use_tf_functions=yes
  #   - tf-agents: use_tf_functions=no
  local iml_dirs=()
  for algo in $(fig_framework_choice_algos); do
    for env_id in $(fig_framework_choice_envs); do
      iml_dirs+=($(stable_baselines_iml_direc))
      for use_tf_functions in yes no; do
        iml_dirs+=($(tf_agents_iml_direc))
      done
      iml_dirs+=($(reagent_iml_direc))
    done
  done

#    --y2-logscale

  # --title "Framework choice"
  local args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(framework_choice_plots_direc)
    --OverlapStackedBarTask-hack-upper-right-legend-bbox-x 0.365
    --CategoryTransitionPlotTask-hack-upper-right-legend-bbox-x 0.365
    --GpuHwPlotTask-width 6
    --GpuHwPlotTask-height 5
  )
  _plot_framework_choice "${args[@]}"
)
}

plot_framework_choice_ddpg() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-3}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}

  # Fig 11(b): Framework comparison DDPG
  local iml_dirs=()
  for algo in $(fig_framework_choice_algos_ddpg); do
    for env_id in $(fig_framework_choice_envs); do
      iml_dirs+=($(stable_baselines_iml_direc))
      for use_tf_functions in yes no; do
        iml_dirs+=($(tf_agents_iml_direc))
      done
      # NOTE: ReAgent doesn't implement DDPG.
      # iml_dirs+=($(reagent_iml_direc))
    done
  done

#    --y2-logscale

  # --title "Framework choice"
  local args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(framework_choice_plots_ddpg_direc)
    --OverlapStackedBarTask-hack-upper-right-legend-bbox-x 0.365
    --CategoryTransitionPlotTask-hack-upper-right-legend-bbox-x 0.365
    --GpuHwPlotTask-width 6
    --GpuHwPlotTask-height 5
  )
  _plot_framework_choice "${args[@]}"
)
}

test_run_expr() {
(
  set -eu

  test_dir=$IML_DIR/output/run_expr/debug
  run_expr_sh=${test_dir}/run_expr.sh
#  n_launches=10
  n_launches=1
  mkdir -p $test_dir
  if [ -e $run_expr_sh ]; then
    rm $run_expr_sh
  fi
   run_expr_args=(--debug --tee --sh ${run_expr_sh} --retry 3)
   py_args=(--fail)
#  run_expr_args=(--tee --sh ${run_expr_sh})
#  py_args=()
  for i in $(seq ${n_launches}); do
    iml-run-expr "${run_expr_args[@]}" --append \
      python $IML_DIR/iml_profiler/scripts/madeup_cmd.py --iml-directory ${test_dir}/process_${i} "${py_args[@]}"
  done
  iml-run-expr "${run_expr_args[@]}" --run-sh
)
}

run_tf_agents() {
(
  set -eu

#  export CUDA_VISIBLE_DEVICES=0

  algo=${algo:-ddpg}
  env_id=${env_id:-Walker2DBulletEnv-v0}
  just_plot=${just_plot:-no}
  calibrate=${calibrate:-yes}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  # Roughly 1 minute when running --config time-breakdown
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  retry=${retry:-}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}
  subdir=${subdir:-}
  use_tf_functions=${use_tf_functions:-yes}

  echo "run_tf_agents: use_tf_functions = ${use_tf_functions}"

  local py_script=$TF_AGENTS_DIR/tf_agents/agents/${algo}/examples/v2/train_eval.rlscope.py
  if [ ! -f ${py_script} ]; then
    echo "ERROR: Couldn't find tf-agents training script @ ${py_script}"
    return 1
  fi

  args=(iml-prof)
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --calibrate
      --parallel-runs
    )
  fi
  if [ "${dry_run}" = 'yes' ]; then
    args+=(
      # iml-calibrate option
      --dry-run
    )
  fi
  if [ "${re_calibrate}" = 'yes' ]; then
    args+=(
      --re-calibrate
    )
  fi
  if [ "${re_plot}" = 'yes' ]; then
    args+=(
      --re-plot
    )
  fi
  if [ "${retry}" != "" ]; then
    args+=(
      --retry ${retry}
    )
  fi
  args+=(
    python ${py_script}
    --env_name "${env_id}"
    # --iml-delay
  )
  if [ "${max_passes}" != "" ]; then
    args+=(
      --iml-max-passes ${max_passes}
    )
  fi
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --iml-repetitions ${repetitions}
    )
  fi
  if [ "${stable_baselines_hyperparams}" = 'yes' ]; then
    args+=(
      --stable_baselines_hyperparams
    )
  fi


  gin_params=()
  set_gin_params() {
    gin_params=()
    if [ "${use_tf_functions}" = 'yes' ]; then
     gin_params+=(--gin_param=train_eval.use_tf_functions=True)
    else
     gin_params+=(--gin_param=train_eval.use_tf_functions=False)
    fi
#    gin_params+=("--gin_param=train_eval.env_name=\"${env_id}\"")
  }

  if [ "${just_plot}" = "no" ]; then
    set_gin_params
    iml_direc="$(tf_agents_iml_direc)"
    _do "${args[@]}" "${gin_params[@]}" --iml-directory ${iml_direc}
  fi

)
}

run_reagent() {
(
  set -eu

#  export CUDA_VISIBLE_DEVICES=0

  algo=${algo:-ddpg}
  env_id=${env_id:-Walker2DBulletEnv-v0}
  just_plot=${just_plot:-no}
  calibrate=${calibrate:-yes}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  # Roughly 1 minute when running --config time-breakdown
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  retry=${retry:-}
  dry_run=${dry_run:-no}
  stable_baselines_hyperparams=${stable_baselines_hyperparams:-yes}
  subdir=${subdir:-}
  use_tf_functions=${use_tf_functions:-yes}

  local py_script=$REAGENT_DIR/reagent/workflow/cli.py
  if [ ! -f ${py_script} ]; then
    echo "ERROR: Couldn't find ReAgent training script @ ${py_script}"
    return 1
  fi

  local args=(iml-prof)
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --calibrate
      --parallel-runs
    )
  fi
  if [ "${dry_run}" = 'yes' ]; then
    args+=(
      # iml-calibrate option
      --dry-run
    )
  fi
  if [ "${re_calibrate}" = 'yes' ]; then
    args+=(
      --re-calibrate
    )
  fi
  if [ "${re_plot}" = 'yes' ]; then
    args+=(
      --re-plot
    )
  fi
  if [ "${retry}" != "" ]; then
    args+=(
      --retry ${retry}
    )
  fi
  if [ "${stable_baselines_hyperparams}" != 'yes' ]; then
    echo "ERROR: Only stable_baselines_hyperparams=yes supported for ReAgent"
  fi
  args+=(
    python ${py_script}
    run-stable-baselines reagent.gym.tests.test_gym.run_test
    --algo "${algo}"
    --env "${env_id}"
  )
  if [ "${max_passes}" != "" ]; then
    args+=(
      --iml-max-passes ${max_passes}
    )
  fi
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --iml-repetitions ${repetitions}
    )
  fi

  if [ "${just_plot}" = "no" ]; then
    iml_direc="$(reagent_iml_direc)"
    _do "${args[@]}" --iml-directory ${iml_direc}
  fi

)
}

stable_baselines_setup_display() {
    local display="$1"
    shift 1

    # Wait for the file to come up
    local file="/tmp/.X11-unix/X$display"

    if [ -e "$file" ]; then
        return
    fi

    # Taken from https://github.com/openai/gym/
    # Set up display; otherwise rendering will fail
    sudo Xvfb :1 -screen 0 1024x768x24 &

    sleep 1

    for i in $(seq 1 10); do
        if [ -e "$file" ]; then
             break
        fi

        echo "Waiting for $file to be created (try $i/10)"
        sleep "$i"
    done
    if ! [ -e "$file" ]; then
        echo "Timing out: $file was not created"
        exit 1
    fi
}

run_stable_baselines() {
(
  set -eu

  algo=${algo:-ddpg}
  env_id=${env_id:-Walker2DBulletEnv-v0}
  just_plot=${just_plot:-no}
  calibrate=${calibrate:-yes}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  retry=${retry:-}
  dry_run=${dry_run:-no}

  display=1
  _do stable_baselines_setup_display $display
  export DISPLAY=:$display

  local py_script=$RL_BASELINES_ZOO_DIR/train.py
  if [ ! -f ${py_script} ]; then
    echo "ERROR: Couldn't find tf-agents training script @ ${py_script}"
    return 1
  fi

  args=(iml-prof)
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --calibrate
      --parallel-runs
    )
  fi
  if [ "${dry_run}" = 'yes' ]; then
    args+=(
      # iml-calibrate option
      --dry-run
    )
  fi
  if [ "${re_calibrate}" = 'yes' ]; then
    args+=(
      --re-calibrate
    )
  fi
  if [ "${re_plot}" = 'yes' ]; then
    args+=(
      --re-plot
    )
  fi
  if [ "${retry}" != "" ]; then
    args+=(
      --retry ${retry}
    )
  fi
  args+=(
    python ${py_script}
    --algo "${algo}"
    --env "${env_id}"
  )
  if [ "${max_passes}" != "" ]; then
    args+=(
      --iml-max-passes ${max_passes}
    )
  fi
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --iml-repetitions ${repetitions}
    )
  fi

  if [ "${just_plot}" = "no" ]; then
    iml_direc="$(stable_baselines_iml_direc)"
#    _do "${args[@]}" --iml-directory ${iml_direc}
    # NOTE: train.py derives iml directory based on --log-folder...
    _do "${args[@]}" --log-folder ${iml_direc} --iml-directory ${iml_direc}
  fi

)
}

_tf_agents_root_direc() {
  local iml_root_dir=$IML_DIR/output/tf_agents/calibration.parallel_runs_yes
  subdir=${subdir:-}
  if [ "${subdir}" != "" ]; then
    iml_root_dir="${iml_root_dir}/${subdir}"
  fi
  echo "${iml_root_dir}"
}

tf_agents_plots_direc() {
(
  set -eu
  local iml_plots_direc=$(_tf_agents_root_direc)/plots
  if [ "${stable_baselines_hyperparams}" = 'yes' ]; then
    iml_plots_direc="${iml_plots_direc}/stable_baselines_hyperparams_yes"
  else
    iml_plots_direc="${iml_plots_direc}/stable_baselines_hyperparams_no"
  fi
  echo "${iml_plots_direc}"
)
}

_reagent_root_direc() {
  local iml_root_dir=$IML_DIR/output/reagent/calibration.parallel_runs_yes
  subdir=${subdir:-}
  if [ "${subdir}" != "" ]; then
    iml_root_dir="${iml_root_dir}/${subdir}"
  fi
  echo "${iml_root_dir}"
}

reagent_plots_direc() {
(
  set -eu
  local iml_plots_direc=$(_reagent_root_direc)/plots
  if [ "${stable_baselines_hyperparams}" = 'yes' ]; then
    iml_plots_direc="${iml_plots_direc}/stable_baselines_hyperparams_yes"
  else
    iml_plots_direc="${iml_plots_direc}/stable_baselines_hyperparams_no"
  fi
  echo "${iml_plots_direc}"
)
}

framework_choice_plots_direc() {
(
  set -eu
  local iml_plots_dir=$IML_DIR/output/plots/framework_choice
  echo "${iml_plots_dir}"
)
}

framework_choice_metrics_direc() {
(
  set -eu
  local iml_plots_dir=$IML_DIR/output/plots/framework_choice.metrics
  echo "${iml_plots_dir}"
)
}

framework_choice_plots_ddpg_direc() {
(
  set -eu
  local iml_plots_dir=$IML_DIR/output/plots/framework_choice_ddpg
  echo "${iml_plots_dir}"
)
}

tf_agents_iml_direc() {
(
  set -eu
  max_passes=${max_passes:-}
  local iml_root_dir=$(_tf_agents_root_direc)/algo_${algo}/env_${env_id}
#  iml_direc="${iml_root_dir}/$(_bool_attr use_tf_functions $use_tf_functions)"
  iml_direc="${iml_root_dir}/use_tf_functions_$use_tf_functions"
  if [ "${max_passes}" != "" ]; then
    iml_direc="${iml_direc}.max_passes_${max_passes}"
  fi
  if [ "${stable_baselines_hyperparams}" = 'yes' ]; then
    iml_direc="${iml_direc}.stable_baselines_hyperparams_${stable_baselines_hyperparams}"
  fi
  echo "${iml_direc}"
)
}

reagent_iml_direc() {
(
  set -eu
  max_passes=${max_passes:-}
  local iml_root_dir=$(_reagent_root_direc)/algo_${algo}/env_${env_id}
  local iml_subdir=""
  if [ "${max_passes}" != "" ]; then
    if [ "${iml_subdir}" != "" ]; then
      iml_subdir="${iml_subdir}."
    fi
    iml_subdir="${iml_subdir}max_passes_${max_passes}"
  fi
  if [ "${stable_baselines_hyperparams}" = 'yes' ]; then
    if [ "${iml_subdir}" != "" ]; then
      iml_subdir="${iml_subdir}."
    fi
    iml_subdir="${iml_subdir}stable_baselines_hyperparams_${stable_baselines_hyperparams}"
  fi
  local iml_direc=
  if [ "${iml_subdir}" != "" ]; then
    iml_direc="${iml_root_dir}/${iml_subdir}"
  else
    iml_direc="${iml_root_dir}"
  fi
  echo "${iml_direc}"
)
}

_stable_baselines_root_direc() {
  echo $IML_DIR/output/stable_baselines/calibration.parallel_runs_yes
}
stable_baselines_iml_direc() {
(
  set -eu
  local iml_direc=$(_stable_baselines_root_direc)/algo_${algo}/env_${env_id}
  max_passes=${max_passes:-}
  if [ "${max_passes}" != "" ]; then
    iml_direc="${iml_direc}.max_passes_${max_passes}"
  fi
  echo "${iml_direc}"
)
}
stable_baselines_plots_direc() {
(
  set -eu
  local iml_plots_direc=$(_stable_baselines_root_direc)/plots
  echo "${iml_plots_direc}"
)
}

_py_tf_agents_fix_operation() {
  # APPLY:
  # new_df['operation'] = new_df['operation'].apply(tf_agents_fix_operation)
  cat <<EOF
def tf_agents_fix_operation(operation):
  if operation == 'step':
    return 'Simulation'
  elif operation == 'collect_data':
    return 'Inference'
  elif operation == 'train_step':
    return 'Backpropagation'
  else:
    raise NotImplementedError(f"Not sure what legend label to use for operation={operation}")
EOF
}
_tf_agents_remap_df() {
  cat <<EOF
$(_py_fix_region)
$(_py_tf_agents_fix_operation)

if 'region' in new_df:
  new_df['region'] = new_df.apply(fix_region, axis=1)
new_df['operation'] = new_df['operation'].apply(tf_agents_fix_operation)
EOF
}

_py_reagent_fix_operation() {
  # APPLY:
  # new_df['operation'] = new_df['operation'].apply(reagent_fix_operation)
  cat <<EOF
def reagent_fix_operation(operation):
  if operation == 'step':
    return 'Simulation'
  elif operation in {'sample_action', 'replay_buffer_add'}:
    return 'Inference'
  elif operation in {'train_step', 'training_loop'}:
    return 'Backpropagation'
  else:
    raise NotImplementedError(f"Not sure what legend label to use for operation={operation}")
EOF
}
_reagent_remap_df() {
  cat <<EOF
$(_py_fix_region)
$(_py_reagent_fix_operation)

if 'region' in new_df:
  new_df['region'] = new_df.apply(fix_region, axis=1)
new_df['operation'] = new_df['operation'].apply(reagent_fix_operation)
EOF
}

_framework_choice_remap_df() {
  # ONLY apply tf_agents_fix_operation is its a tf_agents row
  cat <<EOF
$(_py_fix_region)

$(_py_tf_agents_fix_operation)

$(_py_stable_baselines_pretty_operation)

$(_py_reagent_fix_operation)

def framework_choice_operation(row):
  global stable_baselines_pretty_operation
  global tf_agents_fix_operation
  global reagent_fix_operation

  if re.search(r'/stable_baselines/', row['iml_directory']):
    return stable_baselines_pretty_operation(row['algo'], row['operation'])
  elif re.search(r'/tf_agents/', row['iml_directory']):
    return tf_agents_fix_operation(row['operation'])
  elif re.search(r'/reagent/', row['iml_directory']):
    return reagent_fix_operation(row['operation'])
  else:
    raise NotImplementedError(f"Not sure how to remap framework_choice operation for iml_directory={row['iml_directory']}")

if 'region' in new_df:
  new_df['region'] = new_df.apply(fix_region, axis=1)
new_df['operation'] = new_df.apply(framework_choice_operation, axis=1)
EOF
}

_py_fix_region() {
  # APPLY:
  # new_df['region'] = new_df.apply(fix_region, axis=1)
  cat <<EOF
def fix_region(row):
  region = set(row['region'])
  new_region = set(region)

  # "CUDA + TensorFlow" => "CUDA"
  if {CATEGORY_TF_API, CATEGORY_CUDA_API_CPU}.issubset(new_region):
    new_region.remove(CATEGORY_TF_API)

  # "Python + TensorFlow" => "Python"
  if {CATEGORY_PYTHON, CATEGORY_TF_API}.issubset(new_region):
    new_region.remove(CATEGORY_TF_API)

  # "CUDA + Simulation" => "Simulation"
  if {CATEGORY_CUDA_API_CPU, CATEGORY_SIMULATOR_CPP}.issubset(new_region):
    new_region.remove(CATEGORY_CUDA_API_CPU)

  # "CUDA + Python" (CPU or GPU) => "Python"
  if {CATEGORY_CUDA_API_CPU, CATEGORY_PYTHON}.issubset(new_region):
    new_region.remove(CATEGORY_CUDA_API_CPU)

  if len(new_region) == 0:
    raise RuntimeError("row['region'] = {region} became empty during dataframe processing for iml_directory={iml_directory}; row was:\n{row}".format(
      region=sorted(region),
      iml_directory=row['iml_directory'],
      row=row,
    ))
    # import pdb; pdb.set_trace()

  return tuple(sorted(new_region))
EOF
}

_py_policy_type() {
  # APPLY:
  # new_df['policy_type'] = new_df.apply(get_policy_type, axis=1)
  cat <<EOF
def get_policy_type(row):
  algo = row['algo'].lower()
  if re.search(r'a2c|ppo', algo):
    return 'On-policy'
  elif re.search(r'dqn|ddpg|sac|td3', algo):
    return 'Off-policy'
  raise NotImplementedError(f"Not sure whether algo={row['algo']} is on/off-policy for iml_dir={row['iml_directory']}")
EOF
}

_py_stable_baselines_pretty_operation() {
  # APPLY:
  # new_df['operation'] = np.vectorize(stable_baselines_pretty_operation, otypes=[str])(new_df['algo'], new_df['operation'])
  cat <<EOF
def stable_baselines_pretty_operation(algo, op_name):
  if algo == 'ppo2':
    if op_name in {'compute_advantage_estimates', 'optimize_surrogate', 'training_loop'}:
      return "Backpropagation"
    elif op_name == 'sample_action':
      return "Inference"
    elif op_name == 'step':
      return "Simulation"
  elif algo == 'ddpg':
    if op_name in {'evaluate', 'train_step', 'training_loop', 'update_target_network'}:
      return "Backpropagation"
    elif op_name == 'sample_action':
      return "Inference"
    elif op_name == 'step':
      return "Simulation"
  elif algo == 'a2c':
    if op_name in {'train_step', 'training_loop'}:
      return "Backpropagation"
    elif op_name == 'sample_action':
      return "Inference"
    elif op_name == 'step':
      return "Simulation"
  elif algo == 'sac':
    if op_name in {'training_loop', 'update_actor_and_critic', 'update_target_network'}:
      return "Backpropagation"
    elif op_name == 'sample_action':
      return "Inference"
    elif op_name == 'step':
      return "Simulation"
  elif algo == 'td3':
    if op_name in {'training_loop', 'train_actor_and_critic', 'train_critic', 'evaluate'}:
      return "Backpropagation"
    elif op_name in {'sample_action', 'replay_buffer_add'}:
      return "Inference"
    elif op_name == 'step':
      return "Simulation"
  raise NotImplementedError("Not sure what pretty-name to use for algo={algo}, op_name={op_name}".format(
    algo=algo,
    op_name=op_name))
EOF
}

_stable_baselines_remap_df() {
  cat <<EOF
$(_py_fix_region)
$(_py_stable_baselines_pretty_operation)
$(_py_policy_type)

def get_algo_x_order(row):
  # Put off-policy algos together.
  return (row['policy_type'], row['algo'])

new_df['policy_type'] = new_df.apply(get_policy_type, axis=1)
new_df['algo_x_order'] = new_df.apply(get_algo_x_order, axis=1)
if 'region' in new_df:
  new_df['region'] = new_df.apply(fix_region, axis=1)
new_df['operation'] = np.vectorize(stable_baselines_pretty_operation, otypes=[str])(new_df['algo'], new_df['operation'])

#def check_category(row):
#  if row['category'] is None or row['category'] == '':
#    import pdb; pdb.set_trace()
#new_df.apply(check_category, axis=1)

EOF
}

_py_stable_baselines_op_mapping() {
  # APPLY:
  # def mapping(**op_kwargs):
  #   return stable_baselines_op_mapping(**op_kwargs)

  # 'Inference': 'sample_action',
  cat <<EOF
def stable_baselines_op_mapping(algo, iml_directory, x_field):
    # All stable-baselines algorithms use the same gpu-hw operation mapping.
    if algo == 'td3':
      return {
        'Backpropagation': CompositeOp(add=['training_loop'], subtract=['sample_action', 'step']),
        'Inference': CompositeOp(add=['sample_action', 'replay_buffer_add']),
        'Simulation': 'step',
      }
    return {
      'Backpropagation': CompositeOp(add=['training_loop'], subtract=['sample_action', 'step']),
      'Inference': CompositeOp(add=['sample_action']),
      'Simulation': 'step',
    }
EOF
}

_py_tf_agents_op_mapping() {
  # APPLY:
  # def mapping(**op_kwargs):
  #   return tf_agents_op_mapping(**op_kwargs)
  cat <<EOF
def tf_agents_op_mapping(algo, iml_directory, x_field):
    # All tf-agents algorithms use the same gpu-hw operation mapping.
    return {
      'Backpropagation': 'train_step',
      'Inference': CompositeOp(add=['collect_data'], subtract=['step']),
      'Simulation': 'step',
    }
EOF
}

_py_reagent_op_mapping() {
  # APPLY:
  # def mapping(**op_kwargs):
  #   return reagent_op_mapping(**op_kwargs)
  # 'Inference': 'sample_action',
  cat <<EOF
def reagent_op_mapping(algo, iml_directory, x_field):
    # All ReAgent algorithms use the same gpu-hw operation mapping.
    return {
      'Backpropagation': CompositeOp(add=['training_loop'], subtract=['sample_action', 'step']),
      'Inference': CompositeOp(add=['sample_action', 'replay_buffer_add']),
      'Simulation': 'step',
    }
EOF
}

_stable_baselines_op_mapping() {
  cat <<EOF
$(_py_stable_baselines_op_mapping)

def mapping(**op_kwargs):
  global stable_baselines_op_mapping
  return stable_baselines_op_mapping(**op_kwargs)
EOF
}

_tf_agents_op_mapping() {
  cat <<EOF
$(_py_tf_agents_op_mapping)

def mapping(**op_kwargs):
  global tf_agents_op_mapping
  return tf_agents_op_mapping(**op_kwargs)
EOF
}

_framework_choice_op_mapping() {
  cat <<EOF
$(_py_tf_agents_op_mapping)
$(_py_stable_baselines_op_mapping)
$(_py_reagent_op_mapping)

def mapping(iml_directory, **op_kwargs):
  global stable_baselines_op_mapping
  global tf_agents_op_mapping
  global reagent_op_mapping

  if re.search(r'/stable_baselines/', iml_directory):
    return stable_baselines_op_mapping(iml_directory=iml_directory, **op_kwargs)
  elif re.search(r'/tf_agents/', iml_directory):
    return tf_agents_op_mapping(iml_directory=iml_directory, **op_kwargs)
  elif re.search(r'/reagent/', iml_directory):
    return reagent_op_mapping(iml_directory=iml_directory, **op_kwargs)
  else:
    raise NotImplementedError(f"Not sure what op_mapping to use for framework_choice with iml_directory={iml_directory}")
EOF
}

test_plot_simulator() {
(
  set -eu
  iml_direc=$IML_DIR/output/tf_agents/debug/simulation/calibrate.tf_py_function/algo_ddpg/env_HalfCheetahBulletEnv-v0
#  --plots time-breakdown
  args=(
    --re-plot
    --iml-directories ${iml_direc}
    --output-directory ${iml_direc}/plots
    --OverlapStackedBarTask-remap-df "$(_tf_agents_remap_df)"
    --CategoryTransitionPlotTask-remap-df "$(_tf_agents_remap_df)"
    "$@"
  )
  iml-plot "${args[@]}"
)

}

test_plot_stable_baselines() {
(
  set -eu

  repetitions=${repetitions:-3}

  iml_direc=$IML_DIR/output/stable_baselines/calibration.parallel_runs_yes/algo_ddpg/env_Walker2DBulletEnv-v0
#  --plots time-breakdown
  args=(
    --re-plot
    --iml-directories ${iml_direc}
    --iml-repetitions ${repetitions}
    --output-directory ${iml_direc}/plots
    --OverlapStackedBarTask-remap-df "$(_stable_baselines_remap_df)"
    --CategoryTransitionPlotTask-remap-df "$(_stable_baselines_remap_df)"
    --GpuHwPlotTask-op-mapping "$(_stable_baselines_op_mapping)"
    --plots gpu-hw
    "$@"
  )
  iml-plot "${args[@]}"
)

}

_framework_choice_xtick_expression() {
  cat <<EOF
def pretty_autograph(iml_directory):
  if re.search(r'use_tf_functions_no', iml_directory):
    return "autograph OFF"
  elif re.search(r'use_tf_functions_yes', iml_directory):
    return "autograph ON"
  return None

def pretty_rl_framework(iml_directory):
  if re.search(r'/stable_baselines/', iml_directory):
    return "stable-baselines"
  elif re.search(r'/tf_agents/', iml_directory):
    return "tf-agents"
  elif re.search(r'/reagent/', iml_directory):
    return "ReAgent"
  return None

def xfield_detailed(row):
  each_field = [
    row['algo_env'],
    pretty_rl_framework(row['iml_directory']),
    pretty_autograph(row['iml_directory']),
  ]
  each_field = [x for x in each_field if x is not None]
  x_field = '\n'.join(each_field)
  return x_field

def pretty_DL_backend(iml_directory):
  if re.search(r'/stable_baselines/', iml_directory) or re.search(r'/tf_agents/', iml_directory):
    return "TensorFlow"
  elif re.search(r'/reagent/', iml_directory):
    return "PyTorch"
  raise NotImplementedError(f"Not sure what DL back-end to use for iml_dir={iml_directory}")

def pretty_exec_model(iml_directory):
  if re.search(r'/stable_baselines/', iml_directory):
    return "Graph"
  elif re.search(r'/tf_agents/', iml_directory):
    if re.search(r'use_tf_functions_no', iml_directory):
      return "Eager"
    else:
      assert re.search(r'use_tf_functions_yes', iml_directory), f"iml_dir = {iml_directory}"
      return "Autograph"
  elif re.search(r'/reagent/', iml_directory):
    return "Eager"
  raise NotImplementedError(f"Not sure what DL back-end to use for iml_dir={iml_directory}")

def xfield_short(row):
  """
  PyTorch eager
  TensorFlow eager
  TensorFlow autograph
  TensorFlow graph
  """
  global pretty_DL_backend
  global pretty_exec_model

  each_field = [
    pretty_DL_backend(row['iml_directory']),
    pretty_exec_model(row['iml_directory']),
  ]
  each_field = [x for x in each_field if x is not None]
  x_field = '\n'.join(each_field)
  return x_field

# x_field = xfield_detailed(row)
x_field = xfield_short(row)

EOF
}

_stable_baselines_fig_10_algo_choice_xtick_expression() {
  cat <<EOF
def pretty_policy(row):
  if re.search(r'a2c|ppo', row['algo'].lower()):
    return 'On-policy'
  elif re.search(r'dqn|ddpg|sac', row['algo'].lower()):
    return 'Off-policy'
  raise NotImplementedError(f"Not sure whether algo={row['algo']} is on/off-policy for iml_dir={row['iml_directory']}")

def xfield_short(row):
  """
  PPO2 On-policy
  A2C On-policy
  DDPG Off-policy
  SAC Off-policy
  """
  global pretty_policy

  each_field = [
    row['algo'].upper(),
    pretty_policy(row),
  ]
  each_field = [x for x in each_field if x is not None]
  x_field = '\n'.join(each_field)
  return x_field

x_field = xfield_short(row)

EOF
}

plot_tf_agents_fig_10_algo_choice() {
(
  set -eu

  # Fig 10: Algorithm choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents)
  iml_dirs=()
  for algo in $(tf_agents_fig_10_algo_choice_algos); do
    env_id="$(fig_10_algo_choice_environment)"
    for use_tf_functions in yes no; do
      iml_dirs+=($(tf_agents_iml_direc))
    done
  done

  _plot_tf_agents \
    --iml-directories "${iml_dirs[@]}" \
    --output-directory $(tf_agents_plots_direc)/tf_agents_fig_10_algo_choice \
    --x-title "RL algorithm configuration" \
    --xtick-expression "$(_framework_choice_xtick_expression)" \
    --title "RL algorithm choice" \
    --y2-logscale \
    --GpuHwPlotTask-width 6 \
    --GpuHwPlotTask-height 5 \


)
}
plot_tf_agents_fig_9_simulator_choice() {
(
  set -eu

  # Fig 9: Simulator choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
  iml_dirs=()
  for env_id in $(fig_9_simulator_choice_environments); do
    algo=$(tf_agents_fig_9_simulator_choice_algo)
    for use_tf_functions in yes no; do
      iml_dirs+=($(tf_agents_iml_direc))
    done
  done

  _plot_tf_agents \
    --iml-directories "${iml_dirs[@]}" \
    --output-directory $(tf_agents_plots_direc)/tf_agents_fig_9_simulator_choice \
    --x-title "Simulation configuration" \
    --xtick-expression "$(_tf_agents_fig_9_simulator_choice_xtick_expression)" \
    --title "Simulation choice" \
    --y2-logscale \
    --OverlapStackedBarTask-width 12 \
    --OverlapStackedBarTask-height 5 \
    --GpuHwPlotTask-width 9 \
    --GpuHwPlotTask-height 5 \


)
}
plot_tf_agents() {
(
  set -eu

  plot_tf_agents_fig_9_simulator_choice
  plot_tf_agents_fig_10_algo_choice
)
}

plot_reagent() {
(
  set -eu

  # Fig 11: RL framework choice
  iml_dirs=()
  for algo in $(fig_framework_choice_algos); do
    for env_id in $(fig_framework_choice_envs); do
      iml_dirs+=($(reagent_iml_direc))
    done
  done

  local args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(reagent_plots_direc)/reagent_fig_11_rl_framework_choice
    --x-title "RL algorithm configuration"
    # --xtick-expression "$(_reagent_fig_10_algo_choice_xtick_expression)"
    --title "RL algorithm choice"
    # --y2-logscale
    --OverlapStackedBarTask-width 6
    --OverlapStackedBarTask-height 5
    --GpuHwPlotTask-width 6
    --GpuHwPlotTask-height 5
  )
  _plot_reagent "${args[@]}"

)
}


plot_stable_baselines_fig_10_algo_choice() {
(
  set -eu

  # Fig 10: Algorithm choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents)
  iml_dirs=()
  for algo in $(stable_baselines_fig_10_algo_choice_algos); do
    env_id="$(fig_10_algo_choice_environment)"
    iml_dirs+=($(stable_baselines_iml_direc))
  done

  args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(stable_baselines_plots_direc)/stable_baselines_fig_10_algo_choice
    --x-title "RL algorithm"
    --xtick-expression "$(_stable_baselines_fig_10_algo_choice_xtick_expression)"
    --OverlapStackedBarTask-x-order-by "algo_x_order"
    # --xtick-expression "$(_framework_choice_xtick_expression)"
    # --xtick-expression "$(_tf_agents_fig_10_algo_choice_xtick_expression)"
    # --title "RL algorithm choice"
    # --OverlapStackedBarTask-y2-logscale
    --GpuHwPlotTask-width 6
    --GpuHwPlotTask-height 5
    --rotation 0
  )
  _plot_stable_baselines "${args[@]}"

)
}
plot_stable_baselines_fig_9_simulator_choice() {
(
  set -eu

  # Fig 9: Simulator choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
  iml_dirs=()
  for env_id in $(fig_9_simulator_choice_environments); do
    algo=$(stable_baselines_fig_9_simulator_choice_algo)
    iml_dirs+=($(stable_baselines_iml_direc))
  done

  # --title "Simulator choice"
  args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(stable_baselines_plots_direc)/stable_baselines_fig_9_simulator_choice
    --x-title "Simulator"
    # --xtick-expression "$(_tf_agents_fig_9_simulator_choice_xtick_expression)"
    # --OverlapStackedBarTask-y2-logscale
    --OverlapStackedBarTask-width 12
    --OverlapStackedBarTask-height 5
    --GpuHwPlotTask-width 9
    --GpuHwPlotTask-height 5
  )
  _plot_stable_baselines "${args[@]}"


)
}

_tf_agents_fig_9_simulator_choice_xtick_expression() {
  cat <<EOF
x_field = regex_match(row['iml_directory'], [
    [r'use_tf_functions_no',
     f"{row['short_env']}\nautograph\nOFF"],
    [r'use_tf_functions_yes',
     f"{row['short_env']}\nautograph\nON"]
])
EOF
}

_tf_agents_fig_10_algo_choice_xtick_expression() {
  cat <<EOF
x_field = regex_match(row['iml_directory'], [
    [r'use_tf_functions_no',
     f"{row['pretty_algo']}\nautograph\nOFF"],
    [r'use_tf_functions_yes',
     f"{row['pretty_algo']}\nautograph\nON"]
])
EOF
}

_plot_tf_agents() {

  plots=${plots:-}

  if [ "${calibrate}" = 'yes' ]; then
    iml_plots_direc=$(tf_agents_plots_direc)
    args=(
      iml-plot
        --iml-repetitions ${repetitions}
        # --iml-directories $(_tf_agents_root_direc)/*tf_functions
        --output-directory ${iml_plots_direc}
        # --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'use_tf_functions_no', f\"{row['algo_env']}\nWithout autograph\"], [r'use_tf_functions_yes', f\"{row['algo_env']}\nWith autograph\"]])"
        # --x-title "Configuration"
        --OverlapStackedBarTask-remap-df "$(_tf_agents_remap_df)"
        --CategoryTransitionPlotTask-remap-df "$(_tf_agents_remap_df)"
    )
    if [ "${dry_run}" = 'yes' ]; then
      args+=(
        # iml-calibrate option
        --dry-run
      )
    fi
    # Leave re-calibration to iml-prof.
    if [ "${re_plot}" = 'yes' ] || [ "${re_calibrate}" = 'yes' ]; then
      args+=(
        --re-plot
      )
    fi
    if [ "${plots}" != "" ]; then
      args+=(
        --plots "${plots}"
      )
    fi
    _do "${args[@]}" "$@"
  fi
}

_plot_reagent() {

  plots=${plots:-}

  if [ "${calibrate}" = 'yes' ]; then
    iml_plots_direc=$(reagent_plots_direc)
    args=(
      iml-plot
        --iml-repetitions ${repetitions}
        # --iml-directories $(_reagent_root_direc)/*tf_functions
        --output-directory ${iml_plots_direc}
        # --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'use_tf_functions_no', f\"{row['algo_env']}\nWithout autograph\"], [r'use_tf_functions_yes', f\"{row['algo_env']}\nWith autograph\"]])"
        # --x-title "Configuration"
        --OverlapStackedBarTask-remap-df "$(_reagent_remap_df)"
        --CategoryTransitionPlotTask-remap-df "$(_reagent_remap_df)"
    )
    if [ "${dry_run}" = 'yes' ]; then
      args+=(
        # iml-calibrate option
        --dry-run
      )
    fi
    # Leave re-calibration to iml-prof.
    if [ "${re_plot}" = 'yes' ] || [ "${re_calibrate}" = 'yes' ]; then
      args+=(
        --re-plot
      )
    fi
    if [ "${plots}" != "" ]; then
      args+=(
        --plots "${plots}"
      )
    fi
    if [ "${DEBUG}" = 'yes' ]; then
      args+=(--pdb --debug --debug-single-thread)
    fi
    _do "${args[@]}" "$@"
  fi
}

_plot_stable_baselines() {

  plots=${plots:-}

  if [ "${calibrate}" = 'yes' ]; then
    # iml_plots_direc=$(stable_baselines_plots_direc)
    args=(
      iml-plot
        --iml-repetitions ${repetitions}
        # --iml-directories $(_stable_baselines_root_direc)/*tf_functions
        # --output-directory ${iml_plots_direc}
        # --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'use_tf_functions_no', f\"{row['algo_env']}\nWithout autograph\"], [r'use_tf_functions_yes', f\"{row['algo_env']}\nWith autograph\"]])"
        # --x-title "Configuration"
        --OverlapStackedBarTask-remap-df "$(_stable_baselines_remap_df)"
        --CategoryTransitionPlotTask-remap-df "$(_stable_baselines_remap_df)"
        --GpuHwPlotTask-op-mapping "$(_stable_baselines_op_mapping)"

        # NOTE: x-title depends on whether it is algo or env comparison.
    )
    if [ "${dry_run}" = 'yes' ]; then
      args+=(
        # iml-calibrate option
        --dry-run
      )
    fi
    # Leave re-calibration to iml-prof.
    if [ "${re_plot}" = 'yes' ] || [ "${re_calibrate}" = 'yes' ]; then
      args+=(
        --re-plot
      )
    fi
    if [ "${plots}" != "" ]; then
      args+=(
        --plots "${plots}"
      )
    fi
    if [ "${DEBUG}" = 'yes' ]; then
      args+=(--pdb --debug --debug-single-thread)
    fi
    _do "${args[@]}" "$@"
  fi
}

_plot_framework_choice() {

  plots=${plots:-}

  if [ "${calibrate}" = 'yes' ]; then
    # iml_plots_direc=$(framework_choice_plots_direc)
    args=(
      iml-plot
        --iml-repetitions ${repetitions}
        # --iml-directories $(_framework_choice_root_direc)/*tf_functions
        # --output-directory ${iml_plots_direc}
        # --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'use_tf_functions_no', f\"{row['algo_env']}\nWithout autograph\"], [r'use_tf_functions_yes', f\"{row['algo_env']}\nWith autograph\"]])"
        # --x-title "Configuration"
        --OverlapStackedBarTask-remap-df "$(_framework_choice_remap_df)"
        --CategoryTransitionPlotTask-remap-df "$(_framework_choice_remap_df)"
        --GpuHwPlotTask-op-mapping "$(_framework_choice_op_mapping)"

        --x-title "Framework configuration"
        --xtick-expression "$(_framework_choice_xtick_expression)"
        --rotation 0
    )
    if [ "${dry_run}" = 'yes' ]; then
      args+=(
        # iml-calibrate option
        --dry-run
      )
    fi
    # Leave re-calibration to iml-prof.
    if [ "${re_plot}" = 'yes' ] || [ "${re_calibrate}" = 'yes' ]; then
      args+=(
        --re-plot
      )
    fi
    if [ "${plots}" != "" ]; then
      args+=(
        --plots "${plots}"
      )
    fi
    if [ "${DEBUG}" = 'yes' ]; then
      args+=(--pdb --debug --debug-single-thread)
    fi
    _do "${args[@]}" "$@"
  fi
}

test_tf_agents() {
(
  set -eu

  export CUDA_VISIBLE_DEVICES=0

  just_plot=${just_plot:-no}
  calibrate=${calibrate:-yes}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  # Roughly 1 minute when running --config time-breakdown
  max_passes=2500
  repetitions=${repetitions:-3}
  dry_run=${dry_run:-no}

  args=(iml-prof)
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --calibrate
    )
  fi
  args+=(
    python $TF_AGENTS_DIR/tf_agents/agents/dqn/examples/v2/train_eval.rlscope.py
    --iml-delay
    --iml-max-passes ${max_passes}
  )
  if [ "${calibrate}" = 'yes' ]; then
    args+=(
      --iml-repetitions ${repetitions}
    )
  fi
  if [ "${re_calibrate}" = 'yes' ]; then
    args+=(
      --re-calibrate
    )
  fi
  if [ "${dry_run}" = 'yes' ]; then
    args+=(
      # iml-calibrate option
      --dry-run
    )
  fi

  iml_root_dir=$IML_DIR/output/tf_agents/examples/dqn/calibration.max_passes_${max_passes}

  if [ "${just_plot}" = "no" ]; then
    iml_direc=${iml_root_dir}/use_tf_functions
    _do "${args[@]}" --iml-directory ${iml_direc}

    iml_direc=${iml_root_dir}/nouse_tf_functions
    _do "${args[@]}" --iml-directory ${iml_direc} --nouse_tf_functions
  fi

  if [ "${calibrate}" = 'yes' ]; then
    args=(
      iml-plot
        --iml-repetitions ${repetitions}
        --iml-directories ${iml_root_dir}/*tf_functions
        --output-directory ${iml_root_dir}/plots
  #      --debug-single-thread
  #      --debug
  #      --pdb
        --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'nouse_tf_functions', 'Without autograph'], [r'use_tf_functions', 'With autograph']])"
        --x-title "Configuration"
    )
    if [ "${dry_run}" = 'yes' ]; then
      args+=(
        # iml-calibrate option
        --dry-run
      )
    fi
    # Leave re-calibration to iml-prof.
    if [ "${re_plot}" = 'yes' ] || [ "${re_calibrate}" = 'yes' ]; then
      args+=(
        --re-plot
      )
    fi
    _do "${args[@]}"
  fi

)
}

_do() {
  (
  set +x
  local dry_str=""
  if [ "${DRY_RUN}" = 'yes' ]; then
    dry_str=" [dry-run]"
  fi
  echo "> CMD [run-bench]${dry_str}:"
  echo "  PWD=$PWD"
  echo "  $ $@"
  if [ "${DRY_RUN}" != 'yes' ]; then
    "$@"
  fi
  )
}
_do_with_logfile() {
  (
  set +x
  set -eu
  # If command we're running fails, propagate it (don't let "tee" mute the error)
  set -o pipefail
  local dry_str=""
  logfile_append=${logfile_append:-no}
  logfile_quiet=${logfile_quiet:-no}
  if [ "${DRY_RUN}" = 'yes' ]; then
    dry_str=" [dry-run]"
  fi
  if [ "${logfile_quiet}" != 'yes' ]; then
    echo "> CMD${dry_str}:"
    echo "  PWD=$PWD"
    echo "  $ $@"
  fi
  local tee_opts=""
  if [ "${logfile_append}" = 'yes' ]; then
    tee_opts="${tee_opts} --append"
  fi
  local ret=
  if [ "${DRY_RUN}" != 'yes' ]; then
    mkdir -p "$(dirname "$logfile")"
    "$@" 2>&1 | tee $tee_opts "$logfile"
  fi
  )
}

_bool_attr() {
    local opt="$1"
    local yes_or_no="$2"
    shift 2
    echo ".${opt}_${yes_or_no}"
}
_bool_opt() {
    local opt="$1"
    local yes_or_no="$2"
    shift 2
    if [ "$yes_or_no" = 'yes' ]; then
        echo "--${opt}"
    fi
}

do_all() {
(
  set -eu
  for cmd in "$@"; do
    _do $cmd
  done
)
}


if [ $# -gt 0 ]; then
    echo "> CMD: $@"
    echo "  $ $@"
    "$@"
else
    echo "Usage:"
    echo "$ ./run_bench.sh [cmd]:"
    echo "Where cmd is one of:"
    echo "  run_all"
    echo "    run_full_training_instrumented"
    echo "    run_full_training_uninstrumented"
    echo "  run_debug_all"
    echo "    run_debug_full_training_instrumented"
    echo "    run_debug_full_training_uninstrumented"
    echo "  old_run_stable_baselines"
    exit 1
fi

