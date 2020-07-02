# Create a virtualenv, and override

## Ubuntu 18.04 uses python3.6
## Ubuntu 16.04 uses python3.5
#ARG PYTHON_BIN_BASENAME=python3
## https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
#ENV VIRTUAL_ENV=/root/venv
## NOTE: We use --system-site-packages since some python packages we need get installed through deb packages.
## e.g.
##  python3-libnvinfer:
##     # Allows us to use this package:
##     import tensorrt
##     # Sadly, cannot pip install this separately (unavailable in pip).
#RUN python -m virtualenv -p /usr/bin/${PYTHON_BIN_BASENAME} $VIRTUAL_ENV --system-site-packages
## We can't do "actiate venv" from Dockerfile.
## So instead, just overwrite PATH so it finds /root/bin/python instead of
## /usr/bin/python.
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ARG TENSOR_RT_VERSION=6.0.1-1+cuda10.1
#USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer6=${TENSOR_RT_VERSION} \
    libnvonnxparsers6=${TENSOR_RT_VERSION} \
    libnvparsers6=${TENSOR_RT_VERSION} \
    libnvinfer-plugin6=${TENSOR_RT_VERSION} \
    libnvinfer-dev=${TENSOR_RT_VERSION} \
    libnvonnxparsers-dev=${TENSOR_RT_VERSION} \
    libnvparsers-dev=${TENSOR_RT_VERSION} \
    libnvinfer-plugin-dev=${TENSOR_RT_VERSION} \
    python-libnvinfer=${TENSOR_RT_VERSION} \
    python3-libnvinfer=${TENSOR_RT_VERSION}
RUN sudo apt install cuda-nvrtc-${CUDA/./-}
#USER ${USER}

# NOTE: For some reason I had cuDNN installed for CUDA 10.2…why?
ARG CUDNN_VERSION=7.6.5.32-1+cuda${CUDA}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=${CUDNN_VERSION} \
    libcudnn7-dev=${CUDNN_VERSION}

# NOTE: ideally, happens after pip setup.... could just install globally though.

# Very Ubuntu version specific...oh well.
ARG CP_PYTHON_VERSION=cp36
ARG TENSOR_RT_VERSION_THIRD_PARTY=6.0.1.5
ADD third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY} /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}
RUN pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/graphsurgeon/*.whl
RUN pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/python/tensorrt*-${CP_PYTHON_VERSION}-*.whl
RUN pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/uff/*.whl
#RUN python3 -c 'import sys; print("cp{major}{minor}".format(major=sys.version_info[0], minor=sys.version_info[1]))'

# The "convert-to-uff" doesn't work with tensorflow v2.
# Create a separate python virtual env with tensorflow v1 and so we can use convert-to-uff.
# https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV TF_V1_VIRTUAL_ENV=/root/venv_tf_v1
# NOTE: We use --system-site-packages since some python packages we need get installed through deb packages.
# e.g.
#  python3-libnvinfer:
#     # Allows us to use this package:
#     import tensorrt
#     # Sadly, cannot pip install this separately (unavailable in pip).
RUN python -m virtualenv -p /usr/bin/python3 $TF_V1_VIRTUAL_ENV --system-site-packages
# We can't do "activate venv" from Dockerfile.
# So instead, just overwrite PATH so it finds /root/bin/python instead of
# /usr/bin/python.
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN $TF_V1_VIRTUAL_ENV/bin/pip install 'tensorflow-gpu==1.14.0'
RUN $TF_V1_VIRTUAL_ENV/bin/pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/graphsurgeon/*.whl
RUN $TF_V1_VIRTUAL_ENV/bin/pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/python/tensorrt*-${CP_PYTHON_VERSION}-*.whl
RUN $TF_V1_VIRTUAL_ENV/bin/pip install /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}/uff/*.whl

RUN rm -rf /root/third_party/TensorRT-${TENSOR_RT_VERSION_THIRD_PARTY}
