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

check_tensorflow_build

#BAZEL_BUILD_FLAGS=(--compilation_mode=dbg)
_set_bazel_dbg_build_flags

BAZEL_EXPLAIN_LOG="$BAZEL_BUILD_DIR/bazel_explain.dbg.log"
cd $TENSORFLOW_DIR
_bazel_build
