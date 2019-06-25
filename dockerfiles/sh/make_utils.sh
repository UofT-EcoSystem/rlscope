#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
#set -e
#set -x

_do() {
    echo "> CMD: $@"
    "$@"
}

BAZEL_BUILD_FLAGS=()
BAZEL_BUILD_TARGET=
BAZEL_EXPLAIN_LOG=
_bazel_build() {
    if [ "$BAZEL_BUILD_TARGET" = "" ]; then
        BAZEL_BUILD_TARGET="tensorflow/tools/pip_package:build_pip_package"
    fi
    if [ "$BAZEL_EXPLAIN_LOG" = "" ]; then
        BAZEL_EXPLAIN_LOG="$BAZEL_BUILD_DIR/bazel_explain.log"
    fi

    local BAZEL_GPU_CONFIG=()
    if [ "$TF_NEED_ROCM" = "1" ]; then
        echo "> Add ROCm to bazel build config: --config=rocm"
        BAZEL_GPU_CONFIG=("${BAZEL_GPU_CONFIG[@]}" --config=rocm)
    elif [ "$TF_NEED_CUDA" = "1" ]; then
        echo "> Add CUDA to bazel build config: --config=cuda"
        BAZEL_GPU_CONFIG=("${BAZEL_GPU_CONFIG[@]}" --config=cuda)
    fi
    (
    cd $TENSORFLOW_DIR
    _do bazel \
        --output_user_root=$BAZEL_BUILD_DIR \
        build \
        "${BAZEL_BUILD_FLAGS[@]}" \
        --copt=-mavx \
        "${BAZEL_GPU_CONFIG[@]}" \
        --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        $BAZEL_BUILD_TARGET \
        --explain=$BAZEL_EXPLAIN_LOG \
        --verbose_explanations \
        --verbose_failures \
        -s
    )
    BAZEL_BUILD_TARGET=
    BAZEL_BUILD_FLAGS=()
    BAZEL_EXPLAIN_LOG=
}

_dbg_add_per_file_copt() {
    local rel_dir="$1"
    shift 1
    get_per_file_copt() {
        local cc_file="$1"
        shift 1
        echo "--per_file_copt=$cc_file@${_DEBUG_OPTS}"
    }
    find $rel_dir -name '*.cc' | grep -v --perl-regexp '\.cu\.cc$' | while read cc_file; do
        get_per_file_copt "$cc_file"
    done
}

#_DEBUG_OPTS="-O0,-g"
_DEBUG_OPTS="-g"
_IGNORE_CUDA="-.*\.cu\.cc$"
_dbg_subdir_add_per_file_copt() {
    # Q: Does THIS include tensorflow/c/c_api.cc?
    # A: YES.
    # SO: to include a subdir and ALL its *.cc files (in all subdirectories), we need:
    #    "--per_file_copt=//$SUBDIR:.*\.cc,${_IGNORE_CUDA}@${_DEBUG_OPTS}"
    #    "--per_file_copt=//$SUBDIR/.*:.*\.cc,${_IGNORE_CUDA}@${_DEBUG_OPTS}"
    local subdir="$1"
    shift 1

    echo "--per_file_copt=//$subdir:.*\.cc,${_IGNORE_CUDA}@${_DEBUG_OPTS}"
    echo "--per_file_copt=//$subdir/.*:.*\.cc,${_IGNORE_CUDA}@${_DEBUG_OPTS}"
}
_dbg_file_add_per_file_copt() {
    # NOTE: Some files are stubborn and just WON'T compile with "-O0 -g" unless I EXPLICITLY add a --per_file_copt that mentions them by their path.
    #
    # e.g. This regex fails to capture direct_session.cc:
    #    "--per_file_copt=//tensorflow/core/common_runtime:.*\.cc,${_IGNORE_CUDA}@${_DEBUG_OPTS}"
    # So, we need to add this
    #    "--per_file_copt=tensorflow/core/common_runtime/direct_session.cc@${_DEBUG_OPTS}"
    local cc_file="$1"
    shift 1
    echo "--per_file_copt=$cc_file@${_DEBUG_OPTS}"
}

_set_bazel_dbg_build_flags() {
    BAZEL_BUILD_FLAGS=( \
        --compilation_mode=opt \
        $(_dbg_subdir_add_per_file_copt "tensorflow/c") \
        $(_dbg_subdir_add_per_file_copt "tensorflow/python/client") \
        $(_dbg_subdir_add_per_file_copt "tensorflow/core/common_runtime") \
        $(_dbg_subdir_add_per_file_copt "tensorflow/core/graph") \
        $(_dbg_file_add_per_file_copt   "tensorflow/core/common_runtime/direct_session.cc") \
    )
}

_check_env() {
    #
    # TODO: these should be optional (i.e. if TENSORFLOW_DIR isn't provided, install it from a fresh git repo checkout).
    #

    if [ "$IML_DIR" = "" ] || [ ! -d "$IML_DIR" ]; then
        echo "ERROR: environment variable IML_DIR should be set to: The root directory of the iml_profiler repo checkout."
        exit 1
    fi

    if ! which install_iml.sh; then
        export PATH="$IML_DIR/dockerfiles/sh:$PATH"
        if ! which install_iml.sh; then
            echo "ERROR: failed trying to push $IML_DIR/dockerfiles/sh on \$PATH."
            exit 1
        fi
    fi

}

_check_tensorflow_build() {
    # environment variables that should be defined for tensorflow builds.

    if [ "$TENSORFLOW_DIR" = "" ] || [ ! -d "$TENSORFLOW_DIR" ]; then
        echo "ERROR: environment variable TENSORFLOW_DIR should be set to: The root directory of a 'patched' TensorFlow checkout."
        exit 1
    fi

    if [ "$BAZEL_BUILD_DIR" = "" ] || [ ! -d "$BAZEL_BUILD_DIR" ]; then
        echo "ERROR: environment variable TENSORFLOW_DIR should be set to: The local path where we should output bazel objects (overrides $HOME/.cache/bazel)."
        exit 1
    fi

}

_check_tensorflow() {
    if ! py_module_installed "tensorflow"; then
        echo "ERROR: doesn't look like you installed tensorflow with IML modifications yet."
        echo "  You have two options:"
        echo "    (1) Install a pre-compiled TensorFlow with IML modifications from https://github.com/UofT-EcoSystem/iml/releases"
        echo "    (2) Compile it yourself, then run install_tensorflow.sh"
        echo "  For details, see the wiki: "
        echo "    https://github.com/UofT-EcoSystem/iml/wiki/Installing-tensorflow-with-IML-modifications"
        exit 1
    fi
}

_check_iml() {
    if ! py_module_installed "iml_profiler"; then
        install_iml.sh
        if ! py_module_installed "iml_profiler"; then
            echo "ERROR: tried to install iml_profiler but failed:"
            echo "> CMD:"
            echo "  $ install_iml.sh"
            exit 1
        fi
    fi
}

_check_env_var() {
    local varname="$1"
    local description="$2"
    shift 2

    eval "local varvalue=\"\$${varname}\""

    if [ "$varvalue" = "" ] || [ ! -d "$varvalue" ]; then
        echo "ERROR: environment variable $varname should be set to: $description"
        exit 1
    fi
}

_check_TENSORFLOW_BENCHMARKS_DIR() {
    local description="The root directory of a checkout of TensorFlow benchmarks repo (https://github.com/tensorflow/benchmarks)"
    _check_env_var TENSORFLOW_BENCHMARKS_DIR "$description"
}

_check_DOPAMINE_DIR() {
    local description="The root directory of a checkout of the dopamine repo (https://github.com/google/dopamine)"
    _check_env_var DOPAMINE_DIR "$description"
}

_check_BASELINES_DIR() {
    local description="The root directory of a checkout of the baselines repo (https://github.com/openai/baselines)"
    _check_env_var BASELINES_DIR "$description"
}

_check_MLPERF_DIR() {
    local description="The root directory of a checkout of the mlperf-training repo (https://github.com/mlperf/training)"
    _check_env_var MLPERF_DIR "$description"
}

_check_STABLE_BASELINES_DIR() {
    local description="The root directory of a checkout of the stable-baselines repo (https://github.com/hill-a/stable-baselines)"
    _check_env_var STABLE_BASELINES_DIR "$description"
}

_check_RL_BASELINES_ZOO_DIR() {
    local description="The root directory of a checkout of the rl-baselines-zoo repo (https://github.com/araffin/rl-baselines-zoo)"
    _check_env_var RL_BASELINES_ZOO_DIR "$description"
}

PY_MODULE_INSTALLED_SILENT=no
py_module_installed() {
    local py_module="$1"
    shift 1
    # Returns 1 if ImportError is thrown.
    # Returns 0 if import succeeds.
    if [ "$PY_MODULE_INSTALLED_SILENT" == 'yes' ]; then
        python -c "import ${py_module}" > /dev/null 2>&1
    else
        echo "> Make sure we can import ${py_module}"
        _do python -c "import ${py_module}"
    fi
}

py_maybe_install() {
    local py_module="$1"
    local sh_install="$2"
    shift 2

    if ! py_module_installed $py_module; then
        ${sh_install}
        echo "> Installed python module $py_module: $sh_install"
        if ! py_module_installed $py_module; then
            echo "ERROR: $py_module still failed even after running $sh_install"
            exit 1
        fi
        echo "> $py_module installed"
    fi

}

#_check_container_env() {
#}
