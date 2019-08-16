ARG UBUNTU_VERSION=18.04
ARG CUDA=10.1
# ARG CUDA_10=10.0
# FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
FROM nvidia/cuda:${CUDA}-cudnn7-devel-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG CUDA
ARG UBUNTU_VERSION
# ARG CUDNN=7.4.1.5-1
ARG CUDNN_MAJOR_VERSION=7

# cuda-command-line-tools-${CUDA/./-}
# cuda-cublas-dev-${CUDA/./-}
# cuda-cufft-dev-${CUDA/./-}
# cuda-curand-dev-${CUDA/./-}
# cuda-cusolver-dev-${CUDA/./-}
# cuda-cusparse-dev-${CUDA/./-}
# libcudnn7=${CUDNN}+cuda${CUDA}
# libcudnn7-dev=${CUDNN}+cuda${CUDA}

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# libpng12-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
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
    find /usr/local/cuda-${CUDA}/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

# TODO: symlink fix for libcublas; use to be inside /usr/local/cuda/lib64, but not resides in /usr/lib/x86_64-linux-gnu/libcublas*
# https://github.com/tensorflow/tensorflow/issues/26150

# apt-get install nvinfer-runtime-trt-repo-ubuntu${UBUNTU_VERSION/./}-5.0.2-ga-cuda${CUDA}
# libnvinfer5=5.0.2-1+cuda${CUDA}
# libnvinfer-dev=5.0.2-1+cuda${CUDA}
RUN { apt-get update \
        && apt-get install -y --no-install-recommends \
            libnvinfer5 \
            libnvinfer-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# Configure the build for our CUDA configuration.
#ENV CI_BUILD_PYTHON python
#ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#ENV TF_NEED_CUDA 1
#ENV TF_NEED_TENSORRT 1
#ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
#ENV TF_CUDA_VERSION=${CUDA}
#ENV TF_CUDNN_VERSION=7

# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
ARG CHECKOUT_TF_SRC=0
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

RUN echo hi
