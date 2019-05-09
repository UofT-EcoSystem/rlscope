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

#BAZEL_BUILD_FLAGS=(--compilation_mode=dbg)
BAZEL_BUILD_FLAGS=( \
    --compilation_mode=opt \
    $(_dbg_subdir_add_per_file_copt "tensorflow/c") \
    $(_dbg_subdir_add_per_file_copt "tensorflow/python/client") \
    $(_dbg_subdir_add_per_file_copt "tensorflow/core/common_runtime") \
    $(_dbg_subdir_add_per_file_copt "tensorflow/core/graph") \
    $(_dbg_file_add_per_file_copt   "tensorflow/core/common_runtime/direct_session.cc") \
)

BAZEL_EXPLAIN_LOG="$BAZEL_BUILD_DIR/bazel_explain.dbg.log"
cd $TENSORFLOW_DIR
_bazel_build
