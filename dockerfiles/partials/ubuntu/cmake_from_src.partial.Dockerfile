#
# Ubuntu 16.04 cmake is out-of-date.
# E.g. doesn't support CUDA.
#

ARG CMAKE_VERSION=3.15.1

USER root
WORKDIR /root/tar_files
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh
RUN chmod +x cmake-${CMAKE_VERSION}-Linux-x86_64.sh
RUN ./cmake-3.15.1-Linux-x86_64.sh --skip-license --prefix=/usr/local
RUN cmake --version
WORKDIR ${HOME}
USER ${RLSCOPE_USER}
