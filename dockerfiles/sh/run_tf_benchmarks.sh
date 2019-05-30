#!/usr/bin/env bash
# NOTE: This should run OUTSIDE a docker container.
# This script is used to build AND start the docker container.
set -e
set -x
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env
_check_TENSORFLOW_BENCHMARKS_DIR

cd $TENSORFLOW_BENCHMARKS_DIR
cd scripts/tf_cnn_benchmarks
python tf_cnn_benchmarks.py \
    --num_gpus=1 \
    --batch_size=32 \
    --model=resnet50 \
    --variable_update=parameter_server \
    "$@"
