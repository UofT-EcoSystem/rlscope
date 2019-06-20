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

_check_STABLE_BASELINES_DIR
_check_RL_BASELINES_ZOO_DIR

# Dependencies taken from:
# rl-baselines-zoo/docker/Dockerfile.gpu
# commit/tag: v1.2
sudo apt-get -y install python-dev python3-dev libopenmpi-dev \
                   libglib2.0-0 \
                   libsm6 libxext6 libfontconfig1 libxrender1 \
                   swig cmake libopenmpi-dev zlib1g-dev ffmpeg \
                   freeglut3-dev xvfb

pip install \
    pyyaml \
    box2d-py==2.3.5 \
    stable-baselines \
    pybullet \
    gym-minigrid \
    progressbar2 \
    ipdb \
    ipython \
    optuna \
    pytablewriter==0.36.0

if [ "${STABLE_BASELINES_DIR}" = "" ]; then
    # Install directly from git repo.
    pip install git+https://github.com/jagleeso/stable-baselines.git@iml
else
    # Install from local checkout of repo.
    cd "${STABLE_BASELINES_DIR}"
    python setup.py develop
fi
