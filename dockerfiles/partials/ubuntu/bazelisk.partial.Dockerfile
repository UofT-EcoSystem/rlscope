# Install bazelisk
#ARG BAZELISK_VERSION=v1.4.0
#RUN wget --quiet -O /usr/local/bin/bazelisk "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64"
#RUN chmod +x /usr/local/bin/bazelisk

## Install nccl (belongs in nvidia_nccl.partial.Dockerfile)
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libnccl-dev \
#    libnccl2
