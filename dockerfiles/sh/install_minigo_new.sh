#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env
_upgrade_pip

_check_MINIGO_DIR

#_check_tensorflow
#_check_iml

# mpi4py
#sudo apt-get -y install libopenmpi-dev

#pip install -r $MLPERF_DIR/reinforcement/tensorflow/minigo/requirements.txt
cd $MINIGO_DIR
#export BAZEL_OPTS="--jobs 4"
# Already defined in environment.
#export TF_CUDA_VERSION=10.0
# Already defined in environment.
#CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
./cc/configure_tensorflow.sh

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