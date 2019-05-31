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

_check_MLPERF_DIR

# mpi4py
#sudo apt-get -y install libopenmpi-dev

pip install -r $MLPERF_DIR/reinforcement/tensorflow/minigo/requirements.txt

#if [ "${MLPERF_DIR}" = "" ]; then
#    # Install directly from git repo.
#    pip install git+https://github.com/jagleeso/baselines.git
#else
#    # Install from local checkout of repo.
#    cd "${MLPERF_DIR}"
#    python setup.py develop
#    # NOTE: setup.py will install mujoco_py; however we DON'T want mujoco_py installed since
#    # without the actual libmujoco.so + license installed, baselines will complain it's not setup.
#    pip uninstall -y mujoco_py
#fi
