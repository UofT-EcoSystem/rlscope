#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
else
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
cd "$SH_DIR"

source $SH_DIR/docker_runtime_common.sh

_check_apt
_check_env
_upgrade_pip

_check_BASELINES_DIR

_check_tensorflow
_check_rlscope

# mpi4py
# Install OpenMPI from source.
# For details, see https://github.com/UofT-EcoSystem/rlscope/wiki/Issues-and-TODOs
#
# sudo apt-get -y install libopenmpi-dev

# TODO: remove this dependency (I caused it...)
pip install 'cymem==2.0.2' 'Cython==0.29.7'

if [ "${BASELINES_DIR}" = "" ]; then
    # Install directly from git repo.
    _do pip install git+https://github.com/UofT-EcoSystem/baselines.git
else
    # Install from local checkout of repo.
    _do cd "${BASELINES_DIR}"
    _do python setup.py develop
    # NOTE: setup.py will install mujoco_py; however we DON'T want mujoco_py installed since
    # without the actual libmujoco.so + license installed, baselines will complain it's not setup.
    _do pip uninstall -y mujoco_py
fi
