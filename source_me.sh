#@IgnoreInspection BashAddShebang
# This is needed for built pxd files:
#   $HOME/build/lib.linux-x86_64-3.5
_script_dir="$(readlink -f "$(dirname "$0")")"
_TF_CPP_ROOT=$_script_dir
_TF_CPP_CLONE=$HOME/clone
_TF_CPP_TF_ROOT="$_TF_CPP_CLONE/tensorflow.benchmark_c_api"
_TF_CPP_TF_DBG_ROOT="$_TF_CPP_CLONE/tensorflow.dbg"

export PATH="\
$_TF_CPP_ROOT/local/bin:\
$PATH"

export PYTHONPATH="\
$_TF_CPP_ROOT/python:\
$_TF_CPP_CLONE/clone/baselines:\
$_TF_CPP_CLONE/gym:\
$PYTHONPATH"

# $_TF_CPP_TF_ROOT/_python_build:

unset _script_dir

# activate_tf_cpp() {
#     source ~/envs/dopamine/bin/activate
# }

_tf_cpp_python() {
    if [ "$DEBUG" = 'yes' ]; then
        echo python3 -m ipdb
    else
        echo python3
    fi
}

activate_tf() {
    local environ=$HOME/envs/cycle_counter_prod
    source $environ/bin/activate
}

activate_tf_dbg() {
    local environ=$HOME/new_envs/tf_dbg
    source $environ/bin/activate
}

btf_dbg() {
    activate_tf_dbg
    if [ ! -e $_TF_CPP_TF_DBG_ROOT/source_me.sh ]; then
        echo "ERROR: couldn't find tensorflow.dbg checkout: missing $_TF_CPP_TF_DBG_ROOT/source_me.sh"
        return 1
    fi
    (
    source $_TF_CPP_TF_DBG_ROOT/source_me.sh
    btf "$@"
    )
}

tf_sync() {
    # Sync sources between tensorflow.dbg and 
    local DIRECS=( \
        tensorflow \
    )
    _do() {
        echo "$@"
        "$@"
    }
    local opts="-avz"
    local logfile="$HOME/tf_sync.txt"
    for direc in "${DIRECS[@]}"; do
        ( _do rsync $opts $_TF_CPP_TF_ROOT/$direc/ $_TF_CPP_TF_DBG_ROOT/$direc/ 2>&1 ) \
            | tee "$logfile"
    done
    echo "> rsync output saved to $logfile"
}

tf_cpp_tensorflow_env_settings() {
    # We need this, otherwise TensorFlow will ignore the Quadro K4000 GPU with the following message:
    #
    #   2018-10-31 10:35:54.318150: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1469] Ignoring visible gpu device (device: 1, name: Quadro K4000,
    #   pci bus id: 0000:08:00.0, compute capability: 3.0) with Cuda multiprocessor count: 4.
    #   The minimum required count is 8. You can adjust this requirement
    #   with the env var TF_MIN_GPU_MULTIPROCESSOR_COUNT.
    #
    export TF_MIN_GPU_MULTIPROCESSOR_COUNT=4
}

tf_cpp_source_all() {
    local _olddir=$PWD
    cd $_olddir

    # To avoid confusing building tensorflow.dbg vs non-debug builds, don't source this.
    # cd ~/clone/tensorflow.benchmark_c_api
    # source source_me.sh
    # activate_tf
    # cd $_olddir

    # cd ~/clone/cpython
    # source source_me.sh
    # cd $_olddir

    # cd ~/clone/dnn_tensorflow_cpp
    # source source_me.sh
    # cd $_olddir

}

iml_build_protos() {
(
    cd $_TF_CPP_ROOT
    protoc --python_out=$_TF_CPP_ROOT/python/proto protobuf/pyprof.proto
)
}

# activate_tf_cpp
# tf_cpp_source_all
tf_cpp_tensorflow_env_settings
tf_cpp_source_all
