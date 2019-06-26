#
# TensorFlow wheel file is python-version dependent.
# We need to create separate pip-packages for python3.6 and python3.5.
#
ARG PROTOBUF_VERSION=3.6.1

WORKDIR /root/tar_files
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz
WORKDIR /root/protobuf
RUN tar -xf /root/tar_files/protobuf-all-${PROTOBUF_VERSION}.tar.gz
WORKDIR /root/protobuf/protobuf-3.6.1
RUN ./configure
RUN make -j$(nproc)
RUN make install
RUN protoc --version
