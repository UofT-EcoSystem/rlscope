# >= CUDA 10.0
# https://forums.developer.nvidia.com/t/cublas-for-10-1-is-missing/71015
USER root

#        cuda-runtime-${CUDA/./-}
#        cuda-command-line-tools-${CUDA/./-}
RUN apt-get install -y --no-install-recommends \
        cuda-cupti-${CUDA/./-}

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH /usr/local/cuda/bin:$PATH

#ARG CUBLAS_VERSION=10.1.0.105-1
#RUN apt-get install -y --no-install-recommends \
#    libcublas10=${CUBLAS_VERSION} \
#    libcublas-dev=${CUBLAS_VERSION}
#
#RUN apt-get install -y --no-install-recommends \
#        build-essential
#
#RUN apt-get install -y --no-install-recommends \
#        cuda-command-line-tools-${CUDA/./-}
#
#RUN apt-get install -y --no-install-recommends \
#        cuda-cufft-dev-${CUDA/./-} \
#        cuda-curand-dev-${CUDA/./-} \
#        cuda-cusolver-dev-${CUDA/./-} \
#        cuda-cusparse-dev-${CUDA/./-}
#
## Install cuDNN library.
##
## To install a different cuDNN version, list available versions using:
## $ apt list libcudnn7 --all-versions
#ARG CUDNN=7.6.5.32-1
#ARG CUDNN_MAJOR_VERSION=7
#RUN apt-get install -y \
#    libcudnn7=${CUDNN}+cuda${CUDA} \
#    libcudnn7-dev=${CUDNN}+cuda${CUDA}
#
## CUDA 10.1
#RUN apt-get install -y --no-install-recommends \
#        libpng-dev
#
#RUN apt-get install -y --no-install-recommends \
#        libcurl3-dev \
#        libfreetype6-dev \
#        libhdf5-serial-dev \
#        libzmq3-dev \
#        pkg-config \
#        rsync \
#        software-properties-common \
#        unzip \
#        zip \
#        zlib1g-dev \
#        wget \
#        git
#
#ARG LIBNVINFER_VERSION=5.1.5-1+cuda${CUDA}
#RUN { \
#        apt-get install -y --no-install-recommends \
#            libnvinfer5=${LIBNVINFER_VERSION} \
#            libnvinfer-dev=${LIBNVINFER_VERSION} \
#            python3-libnvinfer=${LIBNVINFER_VERSION} \
#        && apt-get clean \
#        && rm -rf /var/lib/apt/lists/*; }

USER ${RLSCOPE_USER}
