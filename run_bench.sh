#!/usr/bin/env bash
set -e
set -x

# FAILS to create environment.
# - BipedalWalker-v2
# - HumanoidBulletEnv-v0

# # Lots of implementations available for this environment.
# iml-bench --dir $IML_DIR/output/iml_bench/algorithms "$@" train_stable_baselines.sh --env-id Walker2DBulletEnv-v0
#
# # Lots of environments implemented using ppo2
# iml-bench --dir $IML_DIR/output/iml_bench/environments "$@" train_stable_baselines.sh --algo ppo2 --bullet

# Lots of environments implemented using ppo2
# iml-bench --dir $IML_DIR/output/iml_bench/debug_inference/pybullet_wrapped "$@" inference_stable_baselines.sh --algo ppo2 --env-id Walker2DBulletEnv-v0
# iml-bench --dir $IML_DIR/output/iml_bench/debug_inference/pybullet_not_wrapped "$@" inference_stable_baselines.sh --algo ppo2 --env-id Walker2DBulletEnv-v0
# iml-bench --dir $IML_DIR/output/iml_bench/debug_inference/pybullet_wrapped_logging "$@" inference_stable_baselines.sh --algo ppo2 --env-id Walker2DBulletEnv-v0
# iml-bench --dir $IML_DIR/output/iml_bench/debug_inference/pybullet_wrapped_logging.2 "$@" inference_stable_baselines.sh --algo ppo2 --env-id Walker2DBulletEnv-v0
# iml-bench --dir $IML_DIR/output/iml_bench/debug_training/simulator "$@" train_stable_baselines.sh --algo ppo2 --env-id Walker2DBulletEnv-v0 
iml-bench --dir $IML_DIR/output/iml_bench/debug_training/a2c --debug "$@" train_stable_baselines.sh --algo a2c --env-id Walker2DBulletEnv-v0 
