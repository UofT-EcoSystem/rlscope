#!/usr/bin/env bash
# NOTE: This should run OUTSIDE a docker container.
# This script is used to build AND start the docker container.
set -e
set -x
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env


# pip install tensorflow-gpu==1.13.1
# The Dockerfile downloaded a tensorflow .whl package in advance.
pip install $HOME/pip_whl/tensorflow_gpu*.whl
