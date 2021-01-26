#
# TensorFlow wheel file is python-version dependent.
# We need to create separate pip-packages for python3.6 and python3.5.
#

# NOTE: Had to bump version to 3.9.1 since 3.6.1 had an include-file bug
# when compiling things using cmake (strutil.h).
#ARG PROTOBUF_VERSION=3.9.1
ARG PROTOBUF_VERSION=3.14.0

USER root
WORKDIR /root/tar_files
RUN wget --quiet https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz
WORKDIR /root/protobuf
RUN tar -xf /root/tar_files/protobuf-all-${PROTOBUF_VERSION}.tar.gz
WORKDIR /root/protobuf/protobuf-${PROTOBUF_VERSION}
RUN ./configure "CFLAGS=-fPIC" "CXXFLAGS=-fPIC"
RUN make -j$(nproc)
RUN make install
RUN protoc --version
WORKDIR ${HOME}
USER ${RLSCOPE_USER}
