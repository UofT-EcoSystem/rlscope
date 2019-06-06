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
ARG NCCL_VERSION=2.4.2-1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libnccl-dev=${NCCL_VERSION}+cuda${CUDA} \
        libnccl2=${NCCL_VERSION}+cuda${CUDA}

#libnccl2=2.2.13-1+cuda10.0
#libnccl-dev=2.2.13-1+cuda10.0

# Link NCCL libray and header where the build script expects them.
RUN mkdir -p /usr/local/cuda-${CUDA}/lib &&  \
    ln -s /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/local/cuda/lib/libnccl.so.2 && \
    ln -s /usr/include/nccl.h /usr/local/cuda/include/nccl.h
