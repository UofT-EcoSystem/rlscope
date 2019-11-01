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

_run_bench() {
    local all_dir=$IML_DIR/output/iml_bench/all

    _do iml-bench --dir $all_dir "$@"
}
run_stable_baselines() {
    _run_bench "$@" stable-baselines
    _run_bench "$@" stable-baselines --analyze
    _run_bench "$@" stable-baselines --mode plot
}

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

run_subtraction_validation() {
    _cmd() {
        _do iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr subtraction_validation --calibration-mode validation --repetitions 3 --only-runs 2 --env HalfCheetahBulletEnv-v0 --algo ppo2 "$@"
    }
    _cmd "$@"
    _cmd "$@" --plot
}

run_subtraction_calibration() {
    _cmd() {
        _do iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr subtraction_validation --calibration-mode calibration --repetitions 3 --only-runs 2 --env HalfCheetahBulletEnv-v0 --algo ppo2 "$@"
    }
    _cmd "$@"
    _cmd "$@" --plot
}

run_subtraction_validation_long() {
    _cmd() {
        _do iml-quick-expr "${RUN_DEBUG_ARGS[@]}" --expr subtraction_validation --calibration-mode validation --repetitions 3 --env HalfCheetahBulletEnv-v0 --algo ppo2 "$@"
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
