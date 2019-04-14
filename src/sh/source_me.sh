#!/usr/bin/env bash
export IML=$HOME/clone/dnn_tensorflow_cpp

_activate_virtualenv_cuda9() {
    source $HOME/envs/cuda9/bin/activate
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
$IML/python:\
$PYTHONPATH"
}
_activate_virtualenv_cuda9
_activate_cuda9
_add_python_path

_process_minigo() {
(
    set -e
#    set -x
    set -o pipefail

    _do() {
        echo "> CMD: $@"
        "$@"
    }

    cd $IML

#    for overlap_type in ResourceOverlap ResourceSubplot CategoryOverlap OperationOverlap; do
#    for overlap_type in ResourceOverlap ResourceSubplot OperationOverlap; do
    for overlap_type in OperationOverlap; do
        (
        echo "> overlap_type = $overlap_type"
        _do python3 $IML/python/profiler/run_benchmark.py \
            --directories $IML/checkpoints/minigo/vector_test_multiple_workers_k4000 \
            --rules UtilizationPlot \
            --overlap-type $overlap_type
        echo "> DONE"
        ) | tee ${overlap_type}.txt
    done
)
}
