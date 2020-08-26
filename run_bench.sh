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

STABLE_BASELINES_DIR=${STABLE_BASELINES_DIR:-$HOME/clone/stable-baselines}
IML_DIR=${IML_DIR:-$HOME/clone/iml}
RL_BASELINES_ZOO_DIR=${RL_BASELINES_ZOO_DIR:-$HOME/clone/rl-baselines-zoo}
TF_AGENTS_DIR=${TF_AGENTS_DIR:-$HOME/clone/agents}
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
stable_baselines_fig_10_algo_choice_algos() {
  echo a2c
  echo ddpg
  echo ppo2
  echo sac
}
stable_baselines_fig_9_simulator_choice_algo() {
  echo ddpg
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

all_run_tf_agents() {
(
  set -eu

  calibrate=${calibrate:-yes}
  max_passes=${max_passes:-}
  repetitions=${repetitions:-5}
  re_calibrate=${re_calibrate:-no}
  re_plot=${re_plot:-no}
  dry_run=${dry_run:-no}

  # Fig 10: Algorithm choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents)
  for algo in $(tf_agents_fig_10_algo_choice_algos); do
    env_id="$(fig_10_algo_choice_environment)"
    run_tf_agents
  done

  # Fig 9: Simulator choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
  for env_id in $(fig_9_simulator_choice_environments); do
    algo=$(tf_agents_fig_9_simulator_choice_algo)
    run_tf_agents
  done

  plot_tf_agents_fig_10_algo_choice
  plot_tf_agents_fig_9_simulator_choice

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

  # Fig 10: Algorithm choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents)
  for algo in $(stable_baselines_fig_10_algo_choice_algos); do
    env_id="$(fig_10_algo_choice_environment)"
    run_stable_baselines
  done

  # Fig 9: Simulator choice
  # tf-agents
  #
  # ppo doesn't work (BUG in tf-agents).  Use ddpg instead.
  for env_id in $(fig_9_simulator_choice_environments); do
    algo=$(stable_baselines_fig_9_simulator_choice_algo)
    run_stable_baselines
  done

  plot_stable_baselines_fig_10_algo_choice
  plot_stable_baselines_fig_9_simulator_choice

)
}

test_run_expr() {
(
  set -eu

  test_dir=$IML_DIR/output/run_expr/debug
  run_expr_sh=${test_dir}/run_expr.sh
  n_launches=10
  mkdir -p $test_dir
  if [ -e $run_expr_sh ]; then
    rm $run_expr_sh
  fi
  # run_expr_args=(--debug --tee --sh ${run_expr_sh})
  run_expr_args=(--tee --sh ${run_expr_sh})
  # py_args=(--fail)
  py_args=()
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
  dry_run=${dry_run:-no}
#  use_tf_functions=${use_tf_functions:-yes}

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

  use_tf_functions=yes
  set_gin_params
  iml_direc="$(tf_agents_iml_direc)"

  if [ "${just_plot}" = "no" ]; then
    for use_tf_functions in yes no; do
      set_gin_params
      iml_direc="$(tf_agents_iml_direc)"
      _do "${args[@]}" "${gin_params[@]}" --iml-directory ${iml_direc}
    done
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
  echo $IML_DIR/output/tf_agents/calibration.parallel_runs_yes
}

tf_agents_plots_direc() {
(
  set -eu
  local iml_plots_direc=$(_tf_agents_root_direc)/plots
  echo "${iml_plots_direc}"
)
}

tf_agents_iml_direc() {
(
  set -eu
  local iml_root_dir=$(_tf_agents_root_direc)/algo_${algo}/env_${env_id}
  if [ "${max_passes}" != "" ]; then
    iml_direc="${iml_root_dir}.max_passes_${max_passes}"
  fi
#  iml_direc="${iml_root_dir}/$(_bool_attr use_tf_functions $use_tf_functions)"
  iml_direc="${iml_root_dir}/use_tf_functions_$use_tf_functions"
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

_tf_agents_remap_df() {
  cat <<EOF
$(_common_fix_region_remap_df)

def fix_operation(operation):
  if operation == 'step':
    return 'Simulation'
  elif operation == 'collect_data':
    return 'Inference'
  elif operation == 'train_step':
    return 'Backpropagation'
  else:
    raise NotImplementedError(f"Not sure what legend label to use for operation={operation}")
new_df['operation'] = new_df['operation'].apply(fix_operation)
EOF
}

_common_fix_region_remap_df() {
  cat <<EOF
def fix_region(row):
  region = set(row['region'])

  # "CUDA + TensorFlow" => "CUDA"
  if {CATEGORY_TF_API, CATEGORY_CUDA_API_CPU}.issubset(region):
    region.remove(CATEGORY_TF_API)

  # "Python + TensorFlow" => "Python"
  if {CATEGORY_PYTHON, CATEGORY_TF_API}.issubset(region):
    region.remove(CATEGORY_TF_API)

  # "CUDA + Simulator" => "Simulator"
  if {CATEGORY_CUDA_API_CPU, CATEGORY_SIMULATOR_CPP}.issubset(region):
    region.remove(CATEGORY_CUDA_API_CPU)

  # "CUDA + Python" (CPU or GPU) => "Python"
  if {CATEGORY_CUDA_API_CPU, CATEGORY_PYTHON}.issubset(region):
    region.remove(CATEGORY_CUDA_API_CPU)

  return tuple(sorted(region))
new_df['region'] = new_df.apply(fix_region, axis=1)
EOF
}

_stable_baselines_remap_df() {
  cat <<EOF
$(_common_fix_region_remap_df)

def pretty_operation(algo, op_name):
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
    raise NotImplementedError("Not sure what pretty-name to use for algo={algo}, op_name={op_name}".format(
        algo=algo,
        op_name=op_name))
new_df['operation'] = np.vectorize(pretty_operation, otypes=[str])(new_df['algo'], new_df['operation'])
EOF
}

_stable_baselines_op_mapping() {
#    raise NotImplementedError("Not sure what mapping(algo) to use for algo={algo}".format(
#        algo=algo,
#    ))
  cat <<EOF
def mapping(algo):
    # All stable-baselines algorithms use the same gpu-hw operation mapping.
    return {
      'Backpropagation': CompositeOp(add=['training_loop'], subtract=['sample_action', 'step']),
      'Inference': 'sample_action',
      'Simulator': 'step',
    }
EOF
}

#        def mapping(algo):
#            if algo == 'ddpg':
#              {
#              "Backpropagation": ComposeOp(add=["training_loop"], subtract=["sample_action", "step"]),
#              "Inference": ComposeOp(add=["sample_action"]) # or just "sample_action"
#              "Simulator": "step",
#              }
#            elif ...:

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
    --GpuHwPlotTask-op-mapping "$(_stable_baselines_op_mapping)"
    --plots gpu-hw
    "$@"
  )
  iml-plot "${args[@]}"
)

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
    --xtick-expression "$(_tf_agents_fig_10_algo_choice_xtick_expression)" \
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
    --x-title "Simulator configuration" \
    --xtick-expression "$(_tf_agents_fig_9_simulator_choice_xtick_expression)" \
    --title "Simulator choice" \
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
    run_stable_baselines
  done

  args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(stable_baselines_plots_direc)/stable_baselines_fig_10_algo_choice
    --x-title "RL algorithm configuration"
    # --xtick-expression "$(_tf_agents_fig_10_algo_choice_xtick_expression)"
    --title "RL algorithm choice"
    --y2-logscale
    --GpuHwPlotTask-width 6
    --GpuHwPlotTask-height 5
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

  args=(
    --iml-directories "${iml_dirs[@]}"
    --output-directory $(stable_baselines_plots_direc)/stable_baselines_fig_9_simulator_choice
    --x-title "Simulator configuration"
    # --xtick-expression "$(_tf_agents_fig_9_simulator_choice_xtick_expression)"
    --title "Simulator choice"
    --y2-logscale
    --OverlapStackedBarTask-width 12
    --OverlapStackedBarTask-height 5
    --GpuHwPlotTask-width 9
    --GpuHwPlotTask-height 5
  )
  _plot_stable_baselines "${args[@]}"


)
}

_tf_agents_fig_9_simulator_choice_xtick_expression() {
  # --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'use_tf_functions_no', f\"{row['algo_env']}\nWithout autograph\"], [r'use_tf_functions_yes', f\"{row['algo_env']}\nWith autograph\"]])"
#  cat <<EOF
#  x_field = regex_match(row['iml_directory'], [
#      [r'use_tf_functions_no',
#       f"{row['algo_env']}\nWithout autograph"],
#      [r'use_tf_functions_yes',
#       f"{row['algo_env']}\nWith autograph"]
#  ])
#EOF
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

_plot_stable_baselines() {

  plots=${plots:-}

  if [ "${calibrate}" = 'yes' ]; then
    iml_plots_direc=$(stable_baselines_plots_direc)
    args=(
      iml-plot
        --iml-repetitions ${repetitions}
        # --iml-directories $(_stable_baselines_root_direc)/*tf_functions
        --output-directory ${iml_plots_direc}
        # --xtick-expression "x_field = regex_match(row['iml_directory'], [[r'use_tf_functions_no', f\"{row['algo_env']}\nWithout autograph\"], [r'use_tf_functions_yes', f\"{row['algo_env']}\nWith autograph\"]])"
        # --x-title "Configuration"
        --OverlapStackedBarTask-remap-df "$(_stable_baselines_remap_df)"
        --GpuHwPlotTask-op-mapping "$(_stable_baselines_op_mapping)"
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


