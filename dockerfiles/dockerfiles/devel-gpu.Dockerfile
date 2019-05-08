# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=16.04

FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION} as base

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libcudnn7-dev=7.4.1.5-1+cuda10.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        git \
        && \
    find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0 \
        && rm -rf /var/lib/apt/lists/*

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
ENV TF_CUDA_VERSION=10.0
ENV TF_CUDNN_VERSION=7

# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

RUN ${PIP} --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
    enum34

# Install bazel
ARG BAZEL_VERSION=0.19.2
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh

#RUN apt search libnccl
#RUN exit 1

#libnccl-dev/unknown 2.4.2-1+cuda10.1 amd64
#  NVIDIA Collectives Communication Library (NCCL) Development Files
#
#libnccl1/unknown 1.2.3-1+cuda8.0 amd64
#  NVIDIA Communication Collectives Library (NCCL) Runtime
#
#libnccl2/unknown 2.4.2-1+cuda10.1 amd64
#  NVIDIA Collectives Communication Library (NCCL) Runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libnccl-dev=2.4.2-1+cuda10.1 \
        libnccl2=2.4.2-1+cuda10.1

#libnccl2=2.2.13-1+cuda10.0
#libnccl-dev=2.2.13-1+cuda10.0

# Link NCCL libray and header where the build script expects them.
RUN mkdir -p /usr/local/cuda-10.0/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h

# Configure the build for our CUDA configuration.
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
#ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
# Just RTX 2070 (faster build times)
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5
ENV TF_CUDA_VERSION=9.0
ENV TF_CUDNN_VERSION=7
# NCCL 2.x
ENV TF_NCCL_VERSION=2

# JAMES NOTE: add user for permissions when mounting volumes
#RUN useradd -u 1001 -ms /bin/bash james

# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/#ok-so-what-actually-works
ARG USER_ID
ARG GROUP_ID

# REQUIRED:
# --build-arg USER_ID=(id -u ${USER})
ARG USER_ID
RUN test -n "$USER_ID"
ENV USER_ID $USER_ID

# REQUIRED:
# --build-arg GROUP_ID=(id -u ${USER})
ARG GROUP_ID
RUN test -n "$GROUP_ID"
ENV GROUP_ID $GROUP_ID

# REQUIRED:
# --build-arg USER_NAME=${USER}
ARG USER_NAME
RUN test -n "$USER_NAME"
ENV USER_NAME $USER_NAME

#RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
#    userdel -f ${USER_NAME} &&\
#    if getent group ${USER_NAME} ; then groupdel ${USER_NAME}; fi &&\
#    groupadd -g ${GROUP_ID} ${USER_NAME} &&\
#    useradd -l -u ${USER_ID} -g ${USER_NAME} ${USER_NAME} &&\
#    install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME} \
#;fi
#USER ${USER_NAME}

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN groupadd -g ${GROUP_ID} ${USER_NAME}
RUN useradd -l -u ${USER_ID} -g ${USER_NAME} ${USER_NAME}
RUN install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME}
USER ${USER_NAME}

