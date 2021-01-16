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

#
# NOTE: nvidia image already includes nccl
#
# ARG NCCL_VERSION=2.4.2-1
# libnccl-dev=${NCCL_VERSION}+cuda${CUDA}
# libnccl2=${NCCL_VERSION}+cuda${CUDA}
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         libnccl-dev \
#         libnccl2

#libnccl2=2.2.13-1+cuda10.0
#libnccl-dev=2.2.13-1+cuda10.0
USER root

# Link NCCL libray and header where the build script expects them.
RUN mkdir -p /usr/local/cuda-${CUDA}/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h

# TensorFlow v1.3.1 has includes like this:
#
# e.g. inside tensorflow/core/kernels/cuda_solvers.cc
# #include "cuda/include/cublas_v2.h"
#
# I'm not sure why I was able to compile this before, but it's failing now.
# HACK: create symlink: /usr/include/cuda -> /usr/local/cuda
# As a result, compiler should find /usr/include/cuda/include/cublas_v2.h
#
# NOTE: in newever versions I think they are including CUDA headers in the tensorflow
# source tree (e.g. third_party/gpus/cuda/include/cublas_v2.h).
RUN ln -s -T /usr/local/cuda /usr/include/cuda

USER ${RLSCOPE_USER}
