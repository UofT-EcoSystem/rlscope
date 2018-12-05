#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

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

ENV ROOT /dnn_tensorflow_cpp
ENV PYENV_ROOT $ROOT/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

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

# Python is needed to build python...?
# python ../Objects/typeslots.py < ../Include/typeslots.h > Objects/typeslots.inc
# /bin/sh: 1: python: not found
RUN apt-get install -y python

# For faster build times remove this option; need to run tests to get profiling information used by python build
#      --enable-optimizations
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

RUN mkdir -p $ROOT

# Install CMake version manually to support newer features.
ADD https://cmake.org/files/v3.13/cmake-3.13.0-rc3-Linux-x86_64.sh $CLONE_DIR/
WORKDIR $CLONE_DIR
RUN bash cmake-3.13.0-rc3-Linux-x86_64.sh --prefix=/usr --skip-license --exclude-subdir
#RUN apt-get install -y cmake

# Install json C++ header library (no configure/make required)
RUN git clone git@github.com:nlohmann/json.git $ROOT/third_party/json
WORKDIR $ROOT/third_party/json
RUN git checkout v3.4.0

#RUN apt-get install -y curl
# JAMES NOTE: Apparently downloading files using ADD is a bad practice and we should use wget instead?
# I don't quite follow why that's the case... I guess ADD will ADD the file to the docker image?
# Can't we just rm the file?
ADD https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz $CLONE_DIR/
WORKDIR $CLONE_DIR
ENV EXTERNAL_LIB_DIR $ROOT/external_libs
WORKDIR $EXTERNAL_LIB_DIR
RUN tar -xf $CLONE_DIR/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz

RUN apt-get install -y libboost1.58

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


#ENTRYPOINT ["/bin/bash"]
#
#
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
#
## JAMES NOTE: my stuff
## Test out python installation.
#RUN pip3 install ipdb ipython
#
#ENTRYPOINT ["/bin/bash"]
