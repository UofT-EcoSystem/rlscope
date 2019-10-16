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
    echo "> CMD: $@"
    echo "  $ $@"
    "$@"
}

_run_bench() {
    local all_dir=$IML_DIR/output/iml_bench/all

    _do iml-bench --dir $all_dir "$@" stable-baselines
}
run_stable_baselines() {
    _run_bench "$@"
    _run_bench "$@" --analyze
    _run_bench "$@" --mode plot
}

DEBUG_ALGO_ENV_ARGS=(--algo ppo2 --env Walker2DBulletEnv-v0)

run_full_training_instrumented() {
    iml-quick-expr --expr total_training_time --repetitions 3 --bullet --instrumented
}
run_debug_full_training_instrumented() {
    iml-quick-expr --expr total_training_time --repetitions 1 --bullet --instrumented \
        "${DEBUG_ALGO_ENV_ARGS[@]}"
}

run_full_training_uninstrumented() {
    iml-quick-expr --expr total_training_time --repetitions 3 --bullet
}
run_debug_full_training_uninstrumented() {
    iml-quick-expr --expr total_training_time --repetitions 1 --bullet \
        "${DEBUG_ALGO_ENV_ARGS[@]}"
}

if [ $# -gt 0 ]; then
    "$@"
else
    echo "Usage:"
    echo "$ ./run_bench.sh [cmd]:"
    echo "Where cmd is one of:"
    echo "  run_stable_baselines"
    echo "  run_full_training_instrumented"
    echo "  run_debug_full_training_instrumented"
    echo "  run_full_training_uninstrumented"
    echo "  run_debug_full_training_uninstrumented"
    exit 1
fi
