ARG CUDA=10.1
#FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION} as base
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ENV RLSCOPE_CUDA_VERSION 10.1
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG CUDA
#ARG CUDNN=7.4.1.5-1
#ARG CUDNN=7.6.0.64-1
#ARG CUDNN_MAJOR_VERSION=7

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

RUN apt-get update
