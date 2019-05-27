# https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/develop-upstream/rocm_docs/tensorflow-build-from-source.md
ENV HCC_HOME /opt/rocm/hcc
ENV HIP_PATH /opt/rocm/hip
ENV PATH $HCC_HOME/bin:$HIP_PATH/bin:$PATH
