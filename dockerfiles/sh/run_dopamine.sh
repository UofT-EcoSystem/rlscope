#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
set -x
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env

_check_DOPAMINE_DIR

cd $DOPAMINE_DIR
# Train Pong (the default "atari_lib.create_atari_environment.game_name" in dqn.gin)

# To override, do (for e.g. to run Breakout):
# $ run_dopamine.sh --gin_bindings="atari_lib.create_atari_environment.game_name=Breakout"
python -um dopamine.discrete_domains.train \
  --base_dir="$DOPAMINE_DIR/output" \
  --gin_files="$DOPAMINE_DIR/dopamine/agents/dqn/configs/dqn.gin" \
  "$@"
