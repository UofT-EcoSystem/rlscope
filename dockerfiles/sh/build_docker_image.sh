#!/usr/bin/env bash
# NOTE: This should run OUTSIDE a docker container.
# This script is used to build AND start the docker container.
set -e
set -x
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

# This will build a clean dev environment where we can mount
# tensorflow source-code as a volume and build...
#
# NOTE: apt-get for CUDA dependencies from nvidia's server
# time out sometimes causing this to fail.
cd $TENSORFLOW_DIR/tensorflow/tools/dockerfiles
(
export TF_CUDA_COMPUTE_CAPABILITIES=7.5
# Just compile compute capability 7.5 for RTX 2070 since that's all it uses.
#
# Q: How does "nvidia-docker run" know to run this docker image?
# Why doesn't this command refer to tensorflow/tensorflow:devel-dockerized-gpu-py3?
# -t tf?
docker build -f ./dockerfiles/devel-gpu.Dockerfile \
    --build-arg USE_PYTHON_3_NOT_2=3 \
    --build-arg USER_NAME=${USER} \
    --build-arg USER_ID=$(id -u ${USER}) \
    --build-arg GROUP_ID=$(id -u ${USER}) \
    -t tensorflow/tensorflow:devel-dockerized-gpu-py3 \
    .
)

# NOTE:
# -v $TENSORFLOW_DIR:$TENSORFLOW_DIR
#    ---------- ----------
#    host path  container path
#
# We make the host/container path match for BOTH the source-directory
# (TENSORFLOW_DIR) and the bazel build directory (BAZEL_BUILD_DIR),
# so that any symbolic links created by the bazel build process inside the
# container will be valid outside the container.
#
# That way "python setup.py develop" for doing recompiles without building a
# full pip package and installing it should work.
nvidia-docker run -it \
    -p 8888:8888 \
    -v $TENSORFLOW_DIR:$TENSORFLOW_DIR \
    -v $BAZEL_BUILD_DIR:$BAZEL_BUILD_DIR \
    --env "BAZEL_BUILD_DIR=$BAZEL_BUILD_DIR" \
    --env "TENSORFLOW_DIR=$TENSORFLOW_DIR" \
    tensorflow/tensorflow:devel-dockerized-gpu-py3
