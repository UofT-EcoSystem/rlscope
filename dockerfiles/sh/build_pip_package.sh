#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: you should run this AFTER running ./make_tflow.sh
# All this does is "package up" pre-built files into an installable pip package.
#
# === pip *.whl install instructions ===
#
#   To install the pip package in TENSORFLOW_DIR/tmp/pip, run this (either inside/outside the container):
#   $ pip --no-cache-dir install --upgrade $TENSORFLOW_DIR/tmp/pip/tensorflow-*.whl
set -e
set -x
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

check_tensorflow_build

WHL_DIR=$TENSORFLOW_DIR/tmp/pip
mkdir -p $WHL_DIR
cd $TENSORFLOW_DIR
./bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHL_DIR
