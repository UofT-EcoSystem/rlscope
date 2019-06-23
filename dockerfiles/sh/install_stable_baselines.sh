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

_check_STABLE_BASELINES_DIR
_check_RL_BASELINES_ZOO_DIR

_check_tensorflow
_check_iml

# Dependencies taken from:
# rl-baselines-zoo/docker/Dockerfile.gpu
# commit/tag: v1.2

# NOTE: We don't install libopenmpi-dev from Ubuntu 16.04 repo, since
# it is pretty out of date and ML scripts that use mpi4py print warnings
# due to poor interaction with Docker.
# For details, see:
#   https://github.com/UofT-EcoSystem/iml/wiki/Issues-and-TODOs
#
# sudo apt-get -y install libopenmpi-dev

sudo apt-get -y install python-dev python3-dev \
                   libglib2.0-0 \
                   libsm6 libxext6 libfontconfig1 libxrender1 \
                   swig cmake zlib1g-dev ffmpeg \
                   freeglut3-dev xvfb

# As of roughly June 21 2019, "pip install atari-py==0.2.0" has stopped
# working for python3.5.  Looks like the python3.5 package is no longer
# present at https://pypi.org/project/atari-py/#files
#
# Work-around: install atari-py 0.2.0 straight from github using tag=0.2.0
pip install git+https://github.com/openai/atari-py.git@0.2.0

# NOTE: we DON'T install stable-baselines from pip;
#   we want to use our custom stable-baselines repo with IML annotations added ($STABLE_BASELINES_DIR).
pip install \
    'pyyaml>=5.1' \
    box2d-py==2.3.5 \
    pybullet==2.5.1 \
    gym-minigrid==0.0.4 \
    optuna==0.12.0 \
    pytablewriter==0.36.0 \
    progressbar2 \
    ipdb \
    ipython

if [ "${STABLE_BASELINES_DIR}" = "" ]; then
    # Install directly from git repo.
    _do pip install git+https://github.com/jagleeso/stable-baselines.git@iml
else
    # Install from local checkout of repo.
    _do cd "${STABLE_BASELINES_DIR}"
    _do python setup.py develop
fi
