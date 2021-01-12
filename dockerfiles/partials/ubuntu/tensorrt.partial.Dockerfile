# Create a virtualenv, and override

## Ubuntu 18.04 uses python3.6
## Ubuntu 16.04 uses python3.5
#ARG PYTHON_BIN_BASENAME=python3
## https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
#ENV VIRTUALENV=/root/venv
## NOTE: We use --system-site-packages since some python packages we need get installed through deb packages.
## e.g.
##  python3-libnvinfer:
##     # Allows us to use this package:
##     import tensorrt
##     # Sadly, cannot pip install this separately (unavailable in pip).
#RUN python -m virtualenv -p /usr/bin/${PYTHON_BIN_BASENAME} $VIRTUALENV --system-site-packages
## We can't do "actiate venv" from Dockerfile.
## So instead, just overwrite PATH so it finds /root/bin/python instead of
## /usr/bin/python.
#ENV PATH="$VIRTUALENV/bin:$PATH"

## TensorRT 6.0.1
#ARG TENSOR_RT_VERSION=6.0.1-1+cuda10.1
#ARG TENSOR_RT_VERSION_MAJOR=6
#ARG TENSOR_RT_CUDA_VERSION=10.1
#ARG TENSOR_RT_VERSION_THIRD_PARTY=6.0.1.5
#ARG TENSOR_RT_TF_VERSION=1.14.0

# TensorRT 7.1.3
ARG TENSOR_RT_VERSION=7.1.3-1+cuda10.2
ARG TENSOR_RT_VERSION_MAJOR=7
ARG TENSOR_RT_CUDA_VERSION=10.2
ARG TENSOR_RT_VERSION_THIRD_PARTY=7.1.3.4
ARG TENSOR_RT_TF_VERSION=1.15.2
# NOTE: Need to install libcudnn8 before libnvinfer, otherwise libnvinfer
# will automatically install libcudnn8=cuda-11.0 (not sure why...)
ARG CUDNN8_VERSION=8.0.0.180-1+cuda${TENSOR_RT_CUDA_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=${CUDNN8_VERSION} \
    libcudnn8-dev=${CUDNN8_VERSION}


# NOTE: For some reason I had cuDNN installed for CUDA 10.2…why?
ARG CUDNN_VERSION=7.6.5.32-1+cuda${CUDA}
#ARG CUDNN_VERSION=7.6.5.32-1+cuda${TENSOR_RT_CUDA_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=${CUDNN_VERSION} \
    libcudnn7-dev=${CUDNN_VERSION}

RUN apt list --installed | grep 'cuda11\.0'
#RUN exit 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer${TENSOR_RT_VERSION_MAJOR}=${TENSOR_RT_VERSION}

RUN apt list --installed | grep 'cuda11\.0'
#RUN exit 1

#libnvinfer${TENSOR_RT_VERSION_MAJOR}=${TENSOR_RT_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvonnxparsers${TENSOR_RT_VERSION_MAJOR}=${TENSOR_RT_VERSION} \
    libnvparsers${TENSOR_RT_VERSION_MAJOR}=${TENSOR_RT_VERSION} \
    libnvinfer-plugin${TENSOR_RT_VERSION_MAJOR}=${TENSOR_RT_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer-dev=${TENSOR_RT_VERSION} \
    libnvonnxparsers-dev=${TENSOR_RT_VERSION} \
    libnvparsers-dev=${TENSOR_RT_VERSION} \
    libnvinfer-plugin-dev=${TENSOR_RT_VERSION} \
    python-libnvinfer=${TENSOR_RT_VERSION} \
    python3-libnvinfer=${TENSOR_RT_VERSION}
RUN sudo apt install cuda-nvrtc-${TENSOR_RT_CUDA_VERSION/./-}

# NOTE: ideally, happens after pip setup.... could just install globally though.

# Very Ubuntu version specific...oh well.
ARG CP_PYTHON_VERSION=cp36
ADD third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY} /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}
RUN pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/graphsurgeon/*.whl
RUN pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/python/tensorrt*-${CP_PYTHON_VERSION}-*.whl
RUN pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/uff/*.whl
#RUN python3 -c 'import sys; print("cp{major}{minor}".format(major=sys.version_info[0], minor=sys.version_info[1]))'

# The "convert-to-uff" doesn't work with tensorflow v2.
# Create a separate python virtual env with tensorflow v1 and so we can use convert-to-uff.
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV TF_V1_VIRTUALENV=/root/venv_tf_v1
# NOTE: We use --system-site-packages since some python packages we need get installed through deb packages.
# e.g.
#  python3-libnvinfer:
#     # Allows us to use this package:
#     import tensorrt
#     # Sadly, cannot pip install this separately (unavailable in pip).
RUN python3 -m virtualenv -p /usr/bin/python3 $TF_V1_VIRTUALENV --system-site-packages
# We can't do "activate venv" from Dockerfile.
# So instead, just overwrite PATH so it finds /root/bin/python instead of
# /usr/bin/python.
#ENV PATH="$VIRTUALENV/bin:$PATH"
#RUN $TF_V1_VIRTUALENV/bin/pip install 'tensorflow-gpu==1.14.0'

# TensorRT 7.1.3
RUN $TF_V1_VIRTUALENV/bin/pip install "tensorflow-gpu==${TENSOR_RT_TF_VERSION}"
RUN $TF_V1_VIRTUALENV/bin/pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/graphsurgeon/*.whl
RUN $TF_V1_VIRTUALENV/bin/pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/python/tensorrt*-${CP_PYTHON_VERSION}-*.whl
RUN $TF_V1_VIRTUALENV/bin/pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/uff/*.whl

RUN rm -rf /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}

ARG NCCL_VERSION=2.7.5-1+cuda10.1
# Install nccl (belongs in nvidia_nccl.partial.Dockerfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl-dev=${NCCL_VERSION} \
    libnccl2=${NCCL_VERSION}
