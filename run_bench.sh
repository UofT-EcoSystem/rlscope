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
_do() {
    echo "> CMD: $@"
    echo "  $ $@"
    "$@"
}

_run_bench() {
    local all_dir=$IML_DIR/output/iml_bench/all
    # (1) On vs off policy:
    _do iml-bench --dir $all_dir "$@" train_stable_baselines.sh --env-id PongNoFrameskip-v4
    # (2) Compare environments:
    _do iml-bench --dir $all_dir "$@" train_stable_baselines.sh --bullet --algo ppo2
    # (3) Compare algorithms:
    _do iml-bench --dir $all_dir "$@" train_stable_baselines.sh --env-id Walker2DBulletEnv-v0
    # (4) Compare all RL workloads:
    _do iml-bench --dir $all_dir "$@" train_stable_baselines.sh --all --bullet
}

_run_bench "$@"
_run_bench "$@" --analyze
