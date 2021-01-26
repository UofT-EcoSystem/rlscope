# >= CUDA 10.0
# https://forums.developer.nvidia.com/t/cublas-for-10-1-is-missing/71015
USER root

#ARG CUBLAS_VERSION=10.1.0.105-1
RUN apt-get install -y --no-install-recommends \
    libcublas-${CUDA/./-} \
    libcublas-dev-${CUDA/./-}

##<= CUDA 10.0
#RUN apt-get install -y --no-install-recommends \
#    cuda-cublas-dev-${CUDA/./-}

RUN apt-get install -y --no-install-recommends \
        build-essential

RUN apt-get install -y --no-install-recommends \
        cuda-command-line-tools-${CUDA/./-}

#RUN apt-get install -y --no-install-recommends \
#        cuda-cufft-dev-${CUDA/./-} \
#        cuda-curand-dev-${CUDA/./-} \
#        cuda-cusolver-dev-${CUDA/./-} \
#        cuda-cusparse-dev-${CUDA/./-}

# Q: do we need these...?
RUN apt-get install -y --no-install-recommends \
        libcufft-${CUDA/./-} \
        libcufft-dev-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcurand-dev-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusolver-dev-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        libcusparse-dev-${CUDA/./-}

# Install cuDNN library.
#
# To install a different cuDNN version, list available versions using:
# $ apt list libcudnn7 --all-versions
#ARG CUDNN=7.6.5.32-1
#ARG CUDNN_MAJOR_VERSION=7
#libcudnn8=${CUDNN}+cuda${CUDA}
#libcudnn8-dev=${CUDNN}+cuda${CUDA}
RUN apt-get install -y \
    libcudnn8 \
    libcudnn8-dev

## CUDA 10.0
#RUN apt-get install -y --no-install-recommends \
#        libpng12-dev

# CUDA 10.1
RUN apt-get install -y --no-install-recommends \
        libpng-dev

# Set timezone to avoid interactive prompt during container build
ARG TZ="America/New_York"
RUN ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get install -y --no-install-recommends \
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
        git

## CUDA 10.0
#RUN find /usr/local/cuda-10.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
#    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

## <= CUDA 10.0
#RUN { apt-get update && \
#        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda${CUDA} \
#        && apt-get update \
#        && apt-get install -y --no-install-recommends \
#            libnvinfer5=5.0.2-1+cuda${CUDA} \
#            libnvinfer-dev=5.0.2-1+cuda${CUDA} \
#        && apt-get clean \
#        && rm -rf /var/lib/apt/lists/*; }


#ARG LIBNVINFER_VERSION_NUMBER_FULL=5.1.5-1
#ARG LIBNVINFER_VERSION_NUMBER_PARTIAL=5.1.5
## <= CUDA 10.0
#RUN { apt-get update && \
#        apt-get install nvinfer-runtime-trt-repo-${UBUNTU_VERSION/./}-${LIBNVINFER_VERSION_NUMBER_PARTIAL}-ga-cuda${CUDA} \
#        && apt-get update \
#        && apt-get install -y --no-install-recommends \
#            libnvinfer5=${LIBNVINFER_VERSION} \
#            libnvinfer-dev=${LIBNVINFER_VERSION} \
#        && apt-get clean \
#        && rm -rf /var/lib/apt/lists/*; }


# Disable this for now... not needed for RL-Scope.
#ARG LIBNVINFER_VERSION=5.1.5-1+cuda${CUDA}
#RUN { \
#        apt-get install -y --no-install-recommends \
#            libnvinfer5=${LIBNVINFER_VERSION} \
#            libnvinfer-dev=${LIBNVINFER_VERSION} \
#            python3-libnvinfer=${LIBNVINFER_VERSION} \
#        && apt-get clean \
#        && rm -rf /var/lib/apt/lists/*; }

# Configure the build for our CUDA configuration.
#ENV CI_BUILD_PYTHON python
#ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#ENV TF_NEED_CUDA 1
#ENV TF_NEED_TENSORRT 1
#ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
#ENV TF_CUDA_VERSION=10.0
#ENV TF_CUDNN_VERSION=7

# Check out TensorFlow source code if --build_arg CHECKOUT_TENSORFLOW=1
#ARG CHECKOUT_TF_SRC=0
#RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

USER ${RLSCOPE_USER}
