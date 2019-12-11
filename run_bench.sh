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

if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

_do() {
    echo "> [run_bench] CMD:"
    echo "  $ $@"
    "$@"
}

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
run_stable_baselines() {
    sb_all_reps "$@"
}
sb_train() {
    _run_bench "$@" stable-baselines
}

iml_analyze() {
    iml-analyze "${CALIB_OPTS[@]}" "$@"
}

run_total_training_time_plot() {
    iml-quick-expr --expr plot_fig --fig fig_13_overhead_correction \
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
    sb_analyze "$@"
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
run_fig_algorithm_choice_1a_med_complexity_plot_01() {
    # logan
    sb_one_rep_plot --repetition 1 "$@" --algo-env-group algorithm_choice_1a_med_complexity "$@"
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
    echo "  run_stable_baselines"
    exit 1
fi
