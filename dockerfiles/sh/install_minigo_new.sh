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

CUDA=10.1

# What I used for v1.15.0 tensorflow (even though TensorFlow says to use 0.26.1 ...)
#BAZEL_VERSION=0.24.1
BAZEL_VERSION=2.0.0

bazel_version() {
  bazel version | grep --perl-regexp "Build label:" | perl -lape 's/Build label:\s*(.*)/$1/'
}

install_bazel() {
  if which bazel >/dev/null; then
    local cur_bazel_version="$(bazel_version)"
    if [ "$cur_bazel_version" == "$BAZEL_VERSION" ]; then
      # Bazel already installed.
      return
    fi
  fi
  if [ ! -f bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh ]; then
    wget --quiet https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  fi
  chmod 755 bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  sudo ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
  local cur_bazel_version="$(bazel_version)"
  if [ "$cur_bazel_version" != "$BAZEL_VERSION" ]; then
    echo "ERROR: failed to install bazel version=$BAZEL_VERSION, saw version $cur_bazel_version installed instead." 1>&2
    return 1
  fi
}

install_cuda() {
#  # Install cuDNN library.
#  CUDNN=7.6.5.32-1
#  CUDNN_MAJOR_VERSION=7
#  apt-get install -y \
#      libcudnn7=${CUDNN}+cuda${CUDA} \
#      libcudnn7-dev=${CUDNN}+cuda${CUDA}

  # Install TensorRT
  NVINFER_VERSION=5.1.5-1
  NVINFER_MAJOR_VERSION=5
  sudo apt-get install -y \
      libnvinfer${NVINFER_MAJOR_VERSION}=${NVINFER_VERSION}+cuda${CUDA}

##  cuda-cublas-dev-${CUDA/./-} \
##  libcublas-dev=${CUDA/./-} \
#  sudo apt-get install -y \
#      cuda-command-line-tools-${CUDA/./-} \
#      cuda-cufft-dev-${CUDA/./-} \
#      cuda-curand-dev-${CUDA/./-} \
#      cuda-cusolver-dev-${CUDA/./-} \
#      cuda-cusparse-dev-${CUDA/./-}

}

#_check_tensorflow
#_check_rlscope

# mpi4py
#sudo apt-get -y install libopenmpi-dev

#pip install -r $MLPERF_DIR/reinforcement/tensorflow/minigo/requirements.txt
cd $MINIGO_DIR
#export BAZEL_OPTS="--jobs 4"
# Already defined in environment.
#export TF_CUDA_VERSION=10.0
# Already defined in environment.
#CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

install_bazel
install_cuda

# Lets hope it can build with 10.1 so we can using CUDA Profiling API...

# Use this as search path for "CUDA libraries and headers"
# /usr/lib/x86_64-linux-gnu,/usr/include/x86_64-linux-gnu,/usr/include,/home/jgleeson/clone/minigo/cudnn7
export TF_CUDA_VERSION=${CUDA}
export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
export TF_NEED_TENSORRT=y
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
