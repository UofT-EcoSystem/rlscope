#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
DEBUG=${DEBUG:-no}
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env
_upgrade_pip

_check_TF_AGENTS_DIR

_check_tensorflow
_check_iml

set -u

#if [ "${STABLE_BASELINES_DIR}" = "" ]; then
#    # Install directly from git repo.
#    _do pip install git+https://github.com/jagleeso/stable-baselines.git@iml
#else
# Install from local checkout of repo.
_do cd "${TF_AGENTS_DIR}"
_do python setup.py develop
#fi

# For some reason, v0.5 "requires" tensorflow probability that's designed for TensorFlow v2.3+
_do pip install tensorflow-probability==0.10.0
