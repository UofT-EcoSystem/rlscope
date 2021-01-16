#!/usr/bin/env bash
# Shell script that is available at Docker RUN time (NOT at build time).
# Modifying this file WILL NOT trigger container rebuilds (modifying docker_build_common.sh will).
#
# NOTE: This should run inside a docker container.

SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
source $SH_DIR/exports.sh
source $SH_DIR/docker_build_common.sh

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
        $(_dbg_file_add_per_file_copt   "tensorflow/core/framework/op_kernel.cc") \
    )
}

_check_env() {
    #
    # TODO: these should be optional (i.e. if TENSORFLOW_DIR isn't provided, install it from a fresh git repo checkout).
    #

    _check_rlscope_dir

#    if ! which develop_rlscope.sh; then
#        export PATH="$RLSCOPE_DIR/dockerfiles/sh:$PATH"
#        if ! which develop_rlscope.sh; then
#            echo "ERROR: failed trying to push $RLSCOPE_DIR/dockerfiles/sh on \$PATH."
#            exit 1
#        fi
#    fi

}

_check_rlscope_dir() {
    if [ "$RLSCOPE_DIR" = "" ] || [ ! -d "$RLSCOPE_DIR" ]; then
        echo "ERROR: environment variable RLSCOPE_DIR should be set to: The root directory of the rlscope repo checkout."
        exit 1
    fi
}

_check_apt() {
    # Sometimes apt will fail on fresh containers unless we do this first.
    sudo apt update
}

check_tensorflow_build() {
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
#  (
#  set -u
#  pip install --no-cache-dir tensorflow-gpu==$TENSORFLOW_VERSION
#  )
  if ! py_module_installed "tensorflow"; then
    echo "ERROR: you need to install tensorflow."
    echo "  Recommended way to install tensorflow:"
    echo "  $ pip install tensorflow-gpu==\$TENSORFLOW_VERSION"
    exit 1
  fi
}

_upgrade_pip() {
    # I've seen pip commands fail when pip isn't up-to-date.
    # pip install --upgrade pip

    # HACK: bug in pip is causing optuna installation to fail:
    # https://github.com/pypa/pip/issues/6264#issuecomment-572896391
    # pip install --upgrade pip==19.0

    pip install --upgrade pip
}

_check_rlscope() {
    if ! py_module_installed "rlscope"; then
        develop_rlscope.sh
        if ! py_module_installed "rlscope"; then
            echo "ERROR: tried to install rlscope but failed:"
            echo "> CMD:"
            echo "  $ install_rlscope.sh"
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

_check_MINIGO_DIR() {
    local description="The root directory of a checkout of the minigo repo (https://github.com/tensorflow/minigo)"
    _check_env_var MINIGO_DIR "$description"
}

_check_STABLE_BASELINES_DIR() {
    local description="The root directory of a checkout of the stable-baselines repo (https://github.com/hill-a/stable-baselines)"
    _check_env_var STABLE_BASELINES_DIR "$description"
}

_check_TF_AGENTS_DIR() {
    local description="The root directory of a checkout of the tf-agents repo (https://github.com/tensorflow/agents)"
    _check_env_var TF_AGENTS_DIR "$description"
}

_check_RL_BASELINES_ZOO_DIR() {
    local description="The root directory of a checkout of the rl-baselines-zoo repo (https://github.com/araffin/rl-baselines-zoo)"
    _check_env_var RL_BASELINES_ZOO_DIR "$description"
}

_check_REAGENT_DIR() {
    local description="The root directory of a checkout of the ReAgent repo (https://github.com/facebookresearch/ReAgent)"
    _check_env_var REAGENT_DIR "$description"
}

_clone() {
    local path="$1"
    local repo="$2"
    shift 2
    local commit=
    if [ $# -ge 1 ]; then
        commit="$1"
        shift 1
    fi
    if [ ! -e "$path" ]; then
        git clone --recursive $repo $path "$@"
        echo "> Git clone:"
        echo "  Repository: $repo"
        echo "  Directory: $path"
    fi
    (
    cd $path
    git pull || true
    git checkout $commit
    git submodule update --init
    )
}