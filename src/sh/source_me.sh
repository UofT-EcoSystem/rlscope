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
