# NOTE: we need nvcc / nvprof; need "devel" not "runtime"
#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# The results I produced were based on the following ( from nvidia-smi ).
# However, tensorflow from tf-nightly-gpu depends on CUDA 9.0...
# Must have worked because I built tensorflow from source?
#
# Driver Version: 410.66       CUDA Version: 10.0
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update

# Install general dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git

# Install dependencies for building python from source.
RUN apt-get install -y build-essential git libexpat1-dev libssl-dev zlib1g-dev \
  libncurses5-dev libbz2-dev liblzma-dev \
  libsqlite3-dev libffi-dev tcl-dev linux-headers-generic libgdbm-dev \
  libreadline-dev tk tk-dev

# Install dependencies for git clone
RUN apt-get install -y ssh

#
# setup ssh
#
ENV HOME /root
RUN mkdir -p $HOME
# Make ssh dir
RUN mkdir $HOME/.ssh/
# Copy over private key, and set permissions
ADD id_rsa $HOME/.ssh/id_rsa
RUN chmod -R ugo-rwx $HOME/.ssh
ADD ssh_config $HOME/.ssh/config
# Create known_hosts
RUN touch $HOME/.ssh/known_hosts
# Add bitbuckets key
RUN ssh-keyscan github.com >> $HOME/.ssh/known_hosts

#
# clone git repo's
#
ENV CLONE_DIR $HOME/clone
RUN mkdir -p $CLONE_DIR
RUN git clone git@github.com:jagleeso/cpython.git $CLONE_DIR/cpython
WORKDIR $CLONE_DIR/cpython
RUN git checkout cycle-counter

WORKDIR $CLONE_DIR/cpython
RUN mkdir build
WORKDIR build

RUN apt-get install -y wget

# Install CMake version manually to support newer features.
RUN wget -nv https://cmake.org/files/v3.13/cmake-3.13.0-rc3-Linux-x86_64.sh -P $CLONE_DIR/
#ADD https://cmake.org/files/v3.13/cmake-3.13.0-rc3-Linux-x86_64.sh $CLONE_DIR/
WORKDIR $CLONE_DIR
RUN bash cmake-3.13.0-rc3-Linux-x86_64.sh --prefix=/usr --skip-license --exclude-subdir
#RUN apt-get install -y cmake

#RUN apt-get install -y curl
# JAMES NOTE: Apparently downloading files using ADD is a bad practice and we should use wget instead?
# I don't quite follow why that's the case... I guess ADD will ADD the file to the docker image?
# Can't we just rm the file?
#ADD https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz -O $CLONE_DIR/
ENV ROOT /dnn_tensorflow_cpp
RUN wget -nv https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz -P $CLONE_DIR/
WORKDIR $CLONE_DIR
ENV EXTERNAL_LIB_DIR $ROOT/external_libs
WORKDIR $EXTERNAL_LIB_DIR
RUN tar -xf $CLONE_DIR/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz

# Install json C++ header library (no configure/make required)
RUN mkdir -p $ROOT
RUN git clone git@github.com:nlohmann/json.git $ROOT/third_party/json
WORKDIR $ROOT/third_party/json
RUN git checkout v3.4.0

RUN apt-get install -y libboost1.58

RUN apt-get install -y libgflags2v5 libgflags-dev libgtest-dev
WORKDIR /usr/src/gtest
ENTRYPOINT ["/bin/bash"]
RUN cmake . && make -j$(nproc)
RUN mv libg* /usr/lib

#
# Build modified python3.
#
# Python is needed to build python...?
# python ../Objects/typeslots.py < ../Include/typeslots.h > Objects/typeslots.inc
# /bin/sh: 1: python: not found
RUN apt-get install -y python
# For faster build times remove this option; need to run tests to get profiling information used by python build
#      --enable-optimizations
WORKDIR $CLONE_DIR/cpython/build
RUN ../configure \
      --prefix=/usr \
      --enable-loadable-sqlite-extensions \
      --enable-shared \
      --with-lto \
      --with-system-expat \
      --with-system-ffi \
      --enable-ipv6
RUN make -j$(nproc)
RUN make install

# TODO: put before any pip install's
RUN pip3 install --upgrade pip

# JAMES NOTE: my stuff
# Test out python installation.
RUN pip3 install ipdb ipython

# analyze.py requirements
RUN pip3 install numpy scipy pandas \
    py-cpuinfo \
    progressbar2 \
    seaborn \
    matplotlib \
    cxxfilt

RUN pip3 install tf-nightly-gpu
#RUN python3 -c 'import tensorflow; print("> Importing TensorFlow Works!")'
# For some reason, this FAILS form the Dockerfile,
# but it WORKS if we add an entry point here and run it manually...
# I don't know why
# FAILS WITH:
#   ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
#ENTRYPOINT ["/bin/bash"]

#
# Add repo src files to container.
#
WORKDIR $HOME
ADD docker $ROOT/docker
ADD config $ROOT/config
ADD python $ROOT/python
ADD src $ROOT/src
ADD tensorflow $ROOT/tensorflow
ADD test $ROOT/test
ADD CMakeLists.txt $ROOT/CMakeLists.txt
ADD normalized_car_features.csv $ROOT/normalized_car_features.csv

#
# Build cmake for repo.
#
WORKDIR $ROOT
RUN mkdir Debug
WORKDIR Debug
RUN cmake -DCMAKE_BUILD_TYPE=Debug ..
RUN make -j$(nproc)

# VOLUME command triggers build to happen over again (it appears...)
ENV CHECKPOINT_DIR $ROOT/checkpoints
RUN mkdir -p $CHECKPOINT_DIR
VOLUME $CHECKPOINT_DIR

##    python python-pip
##RUN apt-get install -y python-setuptools
##RUN apt-get install -y python-pip python3-pip virtualenv htop
#
##RUN pip3 install --upgrade numpy scipy sklearn tf-nightly-gpu
##
### When to use ADD vs VOLUME in Dockerfile's:
### https://stackoverflow.com/questions/27735706/docker-add-vs-volume
##
### ADD vs COPY:
### https://nickjanetakis.com/blog/docker-tip-2-the-difference-between-copy-and-add-in-a-dockerile
### TLDR: ADD allows the src to be a url/tar-file.
##
### Mount data into the docker
##ADD . $HOME/resnet
##
##WORKDIR $HOME/resnet
##RUN pip3 install -r official/requirements.txt

#RUN pip3 install virtualenv virtualenvwrapper
#RUN python3 -m virtualenv -p python3 $HOME/pyenv

ENV PYTHONPATH "${PYTHONPATH}:${ROOT}/python"
WORKDIR $ROOT
ENTRYPOINT ["/bin/bash"]
