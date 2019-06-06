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

_check_MLPERF_DIR

#py_maybe_install 'baselines' install_baselines.sh

if [ "$OUTPUT_DIR" == '' ]; then
    OUTPUT_DIR="$MLPERF_DIR/output/minigo/docker/train_minigo"
    echo "> env.OUTPUT_DIR not set; defaulting to:"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
fi
mkdir -p $OUTPUT_DIR

# K4000
#export CUDA_VISIBLE_DEVICES="1"

SEED=1
export BASE_DIR=$OUTPUT_DIR
export GOPARAMS=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/multiple_workers.json

mkdir -p $BASE_DIR

cd $MLPERF_DIR/reinforcement/tensorflow
./run_and_time.sh $SEED --iml-keep-traces
