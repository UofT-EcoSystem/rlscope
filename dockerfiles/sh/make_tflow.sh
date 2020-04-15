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
_upgrade_pip

# Equivalent to: --compilation_mode=opt
# See: https://docs.bazel.build/versions/master/user-manual.html
BAZEL_BUILD_FLAGS=(-c opt)
BAZEL_EXPLAIN_LOG="$BAZEL_BUILD_DIR/bazel_explain.opt.log"
_bazel_build
