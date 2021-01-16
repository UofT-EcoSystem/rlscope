#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -eu
FORCE={FORCE:-no}
DEBUG=${DEBUG-no}
# Python version to using in ReAgent virtual environment (using pyenv)
PYTHON_VERSION=3.8.2
TORCH_VERSION=1.6.0
TORCHVISION_VERSION=0.7.0
# CUDA version of PyTorch install
CUDA_VERSION=${CUDA_VERSION:-10.1}
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/docker_runtime_common.sh

_check_apt
_check_env
_upgrade_pip

_check_REAGENT_DIR

#_check_tensorflow
_check_rlscope

# Dependencies taken from:
# rl-baselines-zoo/docker/Dockerfile.gpu
# commit/tag: v1.2

source $RLSCOPE_DIR/source_me.sh


PYENV_ROOT_DIR=$RLSCOPE_INSTALL_PREFIX/pyenv
setup_pyenv() {
    if [ "$FORCE" != 'yes' ] && [ -f $PYENV_ROOT_DIR/bin/pyenv ]; then
        return
    fi

    local apt_packages=(
        make
        build-essential
        libssl-dev
        zlib1g-dev
        libbz2-dev
        libreadline-dev
        libsqlite3-dev
        wget
        curl
        llvm
        libncurses5-dev
        xz-utils
        tk-dev
        libxml2-dev
        libxmlsec1-dev
        libffi-dev
        liblzma-dev
    )
    sudo apt install -y --no-install-recommends "${apt_packages[@]}"


    # local commit="$(_git_latest_tag)"
    local commit="master"
    _clone $PYENV_ROOT_DIR \
        https://github.com/pyenv/pyenv.git \
        $commit

    if ! $PYENV_ROOT_DIR/bin/pyenv help > /dev/null; then
        echo "ERROR: setup_pyenv failed with retcode=$?"
        return $?
    fi
}
setup_pyenv_virtualenv_plugin() {
    local pyenv_virtualenv_plugin_dir=$PYENV_ROOT_DIR/plugins/pyenv-virtualenv
    if [ "$FORCE" != 'yes' ] && [ -f $pyenv_virtualenv_plugin_dir ]; then
        return
    fi
    if [ ! -d $PYENV_ROOT_DIR ]; then
        return
    fi
    local commit="master"
    _clone $pyenv_virtualenv_plugin_dir \
        https://github.com/pyenv/pyenv-virtualenv.git \
        $commit
    if ! $PYENV_ROOT_DIR/bin/pyenv help virtualenv > /dev/null; then
        echo "ERROR: setup_pyenv_virtualenv_plugin failed with retcode=$?"
        return $?
    fi
}
setup_reagent_pyenv() {
  _do setup_pyenv
  _do setup_pyenv_virtualenv_plugin
  _do source $RLSCOPE_DIR/source_me.sh
  if ! pyenv help > /dev/null; then
      ret=$?
      echo "ERROR: setup_pyenv failed with retcode=$ret"
      return $ret
  fi

  _do cd $REAGENT_DIR
  pyenv install ${PYTHON_VERSION} --skip-existing
  local venv="reagent-pyenv-${PYTHON_VERSION}"
  if pyenv virtualenvs | grep ${venv} --quiet; then
    true # pass
  else
    _do pyenv virtualenv ${PYTHON_VERSION} reagent-pyenv-${PYTHON_VERSION}
  fi
  _do pyenv activate reagent-pyenv-${PYTHON_VERSION}
}

_py_version() {
  python --version | perl -lape 's/^Python\s*//'
}

_cuda_version() {
  # e.g., cu101 for CUDA 10.1
  echo "${CUDA_VERSION}" | grep --only-matching --perl-regexp '^\d+\.\d+'
}

_pytorch_cuda_version() {
  local cuda_version="cu$(_cuda_version | perl -lape 's/\.//')"
  echo $cuda_version
}

_pytorch_version() {
  echo "${TORCH_VERSION}+$(_pytorch_cuda_version)"
}

_torchvision_version() {
  echo "${TORCHVISION_VERSION}+$(_pytorch_cuda_version)"
}

main() {
(

  # Looks like we don't need this.
  # Ubuntu's existing 3.6.2 python install works fine?
  # _do setup_reagent_pyenv

  # Needed to "pip install scipy==1.3.1" from ReAgent requirements.txt,
  # otherwise you get compilation errors: "no lapack/blas resources found"
  _do sudo apt-get install -y gfortran libopenblas-dev liblapack-dev

  _do sudo apt-get install -y swig


  # Install from local checkout of repo.
  _do cd "${REAGENT_DIR}"
  _do pip install -r requirements.txt

  # Based on official PyTorch pip installation instructions @ https://pytorch.org
  # e.g.,
  # pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  _do pip install torch==$(_pytorch_version) torchvision==$(_torchvision_version) -f https://download.pytorch.org/whl/torch_stable.html

  _do python setup.py develop

)
}
main "$@"
