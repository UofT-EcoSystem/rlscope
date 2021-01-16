RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

# pip_dependencies.
# Must be installed AFTER setting up virtualenv.
# We use a virtualenv so that $USER can still modify the python enviroment.
# We DON'T want to have to run as root.
# But, we need to be able to install iml and rlscope-drill using "python setup.py devel"
# from a mounted volumes AFTER the container starts.
#RUN ${PIP} --no-cache-dir install \
#    Pillow \
#    h5py \
#    keras_applications \
#    keras_preprocessing \
#    matplotlib \
#    mock \
#    numpy \
#    scipy \
#    sklearn \
#    pandas \
#    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
#    enum34

# Install bazel
ARG BAZEL_VERSION=0.19.2
RUN mkdir /bazel && \
    wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh && \
    rm -f /bazel/installer.sh
