# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV TF_NEED_CUDA 1
ENV TF_NEED_TENSORRT 1
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#ENV TF_CUDA_COMPUTE_CAPABILITIES=3.5,5.2,6.0,6.1,7.0
# Just RTX 2070 (faster build times)

# RTX 2070: 7.5
# Quadro P4000: 6.1
ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5
# ENV TF_CUDA_COMPUTE_CAPABILITIES=7.5,6.1
ENV TF_CUDA_VERSION=${CUDA}
ENV TF_CUDNN_VERSION=${CUDNN_MAJOR_VERSION}
# NCCL 2.x
ENV TF_NCCL_VERSION=2
# The default install path for cudnn for the libcudnn7 NVIDIA package.
# This will be used as the default value for TensorFlow's ./configure
ENV CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu
