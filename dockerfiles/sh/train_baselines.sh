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

source $SH_DIR/make_utils.sh

_check_env
_upgrade_pip

_check_BASELINES_DIR

_check_tensorflow
_check_rlscope

py_maybe_install 'baselines' install_baselines.sh

if [ "$OUTPUT_DIR" == '' ]; then
    OUTPUT_DIR="$BASELINES_DIR/output/PongNoFrameskip-v4/docker/train_baselines"
    echo "> env.OUTPUT_DIR not set; defaulting to:"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
fi
mkdir -p $OUTPUT_DIR

_do python $BASELINES_DIR/baselines/deepq/experiments/run_atari.py \
    --env PongNoFrameskip-v4 \
    --rlscope-start-measuring-call 1 \
    --checkpoint-path $OUTPUT_DIR \
    --rlscope-trace-time-sec $((2*60))
