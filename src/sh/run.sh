#!/usr/bin/env bash
# Currently we're in ROOT/src/sh.
_script_dir="$(realpath "$(dirname "$0")/../..")"
ROOT=$_script_dir
cd $ROOT

CLONE=$HOME/clone
MLPERF_DIR=$CLONE/mlperf_training
MLPERF_MINIGO_DIR=$MLPERF_DIR/reinforcement/tensorflow/minigo
IML=$HOME/clone/dnn_tensorflow_cpp
CHECKPOINTS=$IML/checkpoints/
SEED=1

BASELINES_DIR=$CLONE/baselines
BASELINES_CHECKPOINTS=$BASELINES_DIR/checkpoints

GYM_DIR=$CLONE/gym

_activate_tensorflow() {
    local prev_dir=$PWD
    cd $HOME/clone/tensorflow_cuda9
    source source_me.sh
    activate_tf
    cd $prev_dir
}

_activate_iml() {
    export PYTHONPATH="$PYTHONPATH:\
$ROOT/python\
"
}

_train_minigo() {
    local base_dir="$1"
    local goparams="$2"
    shift 2

(
    _activate_tensorflow

    # K4000
    export CUDA_VISIBLE_DEVICES="1"

    export BASE_DIR=$base_dir
    export GOPARAMS=$goparams

    mkdir -p $BASE_DIR

    cd $MLPERF_DIR/reinforcement/tensorflow ~/clone/mlperf_training/reinforcement/tensorflow
#    ./run_and_time.sh $SEED --iml-keep-traces --iml-disable 2>&1 | tee --ignore-interrupts ${BASE_DIR}/benchmark.txt
    ./run_and_time.sh $SEED --iml-keep-traces 2>&1 | tee --ignore-interrupts ${BASE_DIR}/benchmark.txt
)
}

train_minigo_multiple() {
    local base_dir=$CHECKPOINTS/minigo/vector_test_multiple_workers_k4000
    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/test_multiple_workers.json
    _train_minigo $base_dir $goparams
}

train_minigo_multiple_final() {
    local base_dir=$CHECKPOINTS/minigo/vector_final_multiple_workers_k4000
    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/final_multiple_workers.json
    _train_minigo $base_dir $goparams
}

train_minigo_single() {
    local base_dir=$CHECKPOINTS/minigo/vector_test_single_worker_k4000
    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/test_single_worker.json
    _train_minigo $base_dir $goparams
}

_activate_baselines() {
#    local prev_dir=$PWD
#    cd $BASELINES_DIR
#    source source_me.sh
#    cd $prev_dir

    export PYTHONPATH="$PYTHONPATH:\
$BASELINES_DIR:\
$GYM_DIR\
"
#$HOME/clone/atari-py
# $HOME/clone/Arcade-Learning-Environment

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:\
#$HOME/.mujoco/mjpro150/bin:\
#/usr/lib/nvidia-384"
}

_train_pong() {
    local chkpt_path="$1"
    shift 1
    (
    _activate_tensorflow
    _activate_baselines
    _activate_iml
    echo $PYTHONPATH
    export TF_PRINT_TIMESTAMP=yes
    mkdir -p $chkpt_path
    export CUDA_VISIBLE_DEVICES="1"
    python3 $BASELINES_DIR/baselines/deepq/experiments/run_atari.py \
        --env PongNoFrameskip-v4 \
        --iml-start-measuring-call 1 \
        --checkpoint-path $chkpt_path \
        --iml-num-calls=100 \
        --iml-num-traces 2 \
        --iml-trace-time-sec 40 \
        --iml-python
    )
}

train_pong() {
    local chkpt_path=$BASELINES_CHECKPOINTS/PongNoFrameskip-v4/vector_k4000
    _train_pong $chkpt_path
}

#train_minigo_multiple() {
#    local base_dir=$CHECKPOINTS/minigo/vector_test_multiple_workers_k4000
#    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/test_multiple_workers.json
#    _train_minigo $base_dir $goparams
#}

if [ $# -gt 0 ]; then
    set -x
    "$@"
fi
