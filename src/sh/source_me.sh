#!/usr/bin/env bash
export IML=$HOME/clone/dnn_tensorflow_cpp
export IML_DRILL=$HOME/clone/iml-drill

_activate_virtualenv_cuda9() {
    # tensorflow_cuda9 github repo checkout
    source $HOME/envs/cuda9/bin/activate
#    source $HOME/envs/tf_patch_cuda9/bin/activate
#    source $HOME/envs/py37/bin/activate
# TODO: Make PILCO run without python3.7 requirement.
}

_activate_cuda9() {
export PATH="\
/usr/local/cuda-9.0/bin:\
$HOME/bin:\
$PATH"
export LD_LIBRARY_PATH="\
/usr/local/cuda-9.0/lib64:\
/usr/local/cuda-9.0/extras/CUPTI/lib64:\
$LD_LIBRARY_PATH"
}

_add_python_path() {
export PYTHONPATH="\
$IML:\
$IML/python:\
$PYTHONPATH"
}
_activate_virtualenv_cuda9
_activate_cuda9
_add_python_path

rlscope_gen_protobufs() {
(
    set -e
    set -x
    cd $IML
    protoc -I$PWD --python_out=. prof_protobuf/*.proto
)
}

_process_minigo() {
(
    set -e
#    set -x
    set -o pipefail

    _do() {
        echo "> CMD: $@"
        "$@"
    }

    DIR=$IML/checkpoints/minigo/vector_multiple_workers_k4000

    cd $IML

#    for overlap_type in ResourceOverlap ResourceSubplot OperationOverlap; do
#    for overlap_type in ResourceOverlap ResourceSubplot; do
#    for overlap_type in ResourceSubplot; do
#    for overlap_type in OperationOverlap; do
#    for overlap_type in ResourceSubplot; do
#    for overlap_type in OperationOverlap CategoryOverlap; do
#    for overlap_type in CategoryOverlap; do
#    for overlap_type in OperationOverlap; do
    for overlap_type in ResourceOverlap ResourceSubplot CategoryOverlap OperationOverlap; do
        (
        echo "> overlap_type = $overlap_type"
        _do python3 $IML/rlscope/scripts/analyze.py \
            --directories $DIR \
            --rules UtilizationPlot \
            --overlap-type $overlap_type \
            --debug
        echo "> DONE"
        ) | tee ${overlap_type}.txt &
    done

    local plot_type="HeatScale"
    (
    echo "> plot_type = $plot_type"
    _do python3 $IML/rlscope/scripts/analyze.py \
      --directories $DIR \
      --rules HeatScalePlot
    echo "> DONE"
    ) | tee ${plot_type}.txt &

    echo "Wait for things to finish..."
    wait

)
}

_maybe_add_path() {
    local direc="$1"
    shift 1

    if [ -d "$direc" ]; then
        echo "> INFO: Add to PATH: $direc"
        export PATH="$direc:$PATH"
    else
        echo "> WARNING: SKIP adding non-existent directory to PATH: $direc"
        echo "  Perhaps you forgot to run setup:"
        echo "  $ bash setup.sh"
    fi
}

# NOTE: this will modify your PATH to include IML
# dependencies that were built from source (e.g. protobuf v3).
# Fails in container...not sure why.
ROOT="$(readlink -f "$(dirname "$0")")"
_local_dir=$ROOT/local.$(hostname)
#_local_dir=$ROOT/install
#if [ -e $ROOT/venv ]; then
#  source $ROOT/venv/bin/activate
#else
#  echo "ERROR: didn't find python3 virtualenv at $ROOT/venv; try running setup.sh first:"
#  echo "  $ cd $ROOT"
#  echo "  $ bash setup.sh"
#fi

#PROTOBUF_INSTALL_DIR="$(ls -d $ROOT/third_party/protobuf-*/build.$(hostname)/bin)"
#_maybe_add_path "$PROTOBUF_INSTALL_DIR"
_maybe_add_path "$_local_dir/bin"
# Keep gdb-symbols for debugging.
# For some reason, cmake "install()" isn't finding compiled third_party libraries.
#_maybe_add_path "$ROOT/local/Debug/bin"
#_maybe_add_path "$ROOT/Debug.$(hostname)"
#unset PROTOBUF_INSTALL_DIR

unset _maybe_add_path
unset ROOT
unset _local_dir
