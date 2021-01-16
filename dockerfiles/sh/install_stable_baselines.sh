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

source $SH_DIR/docker_runtime_common.sh

_check_apt
_check_env
_upgrade_pip

_check_STABLE_BASELINES_DIR
_check_RL_BASELINES_ZOO_DIR

_check_tensorflow
_check_rlscope

# Dependencies taken from:
# rl-baselines-zoo/docker/Dockerfile.gpu
# commit/tag: v1.2

sudo apt-get -y install python-dev python3-dev \
                   libglib2.0-0 \
                   libsm6 libxext6 libfontconfig1 libxrender1 \
                   swig zlib1g-dev ffmpeg \
                   freeglut3-dev xvfb \
                   libopenmpi-dev \
                   ssh
# NOTE: mpi4py wants ssh to be installed

# As of roughly June 21 2019, "pip install atari-py==0.2.0" has stopped
# working for python3.5.  Looks like the python3.5 package is no longer
# present at https://pypi.org/project/atari-py/#files
#
# Work-around: install atari-py 0.2.0 straight from github using tag=0.2.0
if ! py_module_installed "atari_py"; then
  pip install git+https://github.com/openai/atari-py.git@0.2.0
fi

# NOTE: we DON'T install stable-baselines from pip;
#   we want to use our custom stable-baselines repo with RL-Scope annotations added ($STABLE_BASELINES_DIR).
# box2d-py==2.3.5

# HACK: optuna fails to install with pip >= 19.0
#pip install --upgrade pip==19.0
#pip install optuna==0.12.0

# HACK: gym-minigrid fails to install with pip <= 19.0
#pip install --upgrade pip
pip install gym-minigrid==0.0.4

pip install \
    'pyyaml>=5.1' \
    'Box2D==2.3.2' \
    pybullet==2.5.1 \
    pytablewriter==0.36.0 \
    progressbar2 \
    ipdb \
    ipython

# LunarLander requires the python package `box2d`.
# You can install it using ``apt install swig`` and then ``pip install box2d box2d-kengz``
#sudo apt-get install -y swig
pip install \
    'Box2D-kengz==2.3.3'

if [ "${STABLE_BASELINES_DIR}" = "" ]; then
    # Install directly from git repo.
    _do pip install git+https://github.com/jagleeso/stable-baselines.git@rlscope
else
    # Install from local checkout of repo.
    _do cd "${STABLE_BASELINES_DIR}"
    # _do python setup.py develop
    # NOTE: the setup.py extra_requires [mpi] is needed for td3/ddpg.
    _do pip install -e ./[mpi]
fi
