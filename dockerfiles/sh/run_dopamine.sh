#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/docker_runtime_common.sh

_check_apt
_check_env
_upgrade_pip

_check_DOPAMINE_DIR

cd $DOPAMINE_DIR

if [ "$OUTPUT_DIR" == '' ]; then
    OUTPUT_DIR="$DOPAMINE_DIR/output/docker"
    echo "> env.OUTPUT_DIR not set; defaulting to:"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
fi
mkdir -p $OUTPUT_DIR

# Train Pong (the default "atari_lib.create_atari_environment.game_name" in dqn.gin)

# To override, do (for e.g. to run Breakout):
# $ run_dopamine.sh --gin_bindings="atari_lib.create_atari_environment.game_name=Breakout"
_do python -um dopamine.discrete_domains.train \
  --base_dir="$OUTPUT_DIR" \
  --gin_files="$DOPAMINE_DIR/dopamine/agents/dqn/configs/dqn.gin" \
  "$@"
