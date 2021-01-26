ARG CUDA=11.0
#FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ARG DOCKER_CUDA=11.0
FROM nvidia/cuda:${DOCKER_CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ENV RLSCOPE_CUDA_VERSION 11.0
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG CUDA

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

RUN apt-get update
