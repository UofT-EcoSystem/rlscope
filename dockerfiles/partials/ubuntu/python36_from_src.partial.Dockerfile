#
# TensorFlow wheel file is python-version dependent.
# We need to create separate pip-packages for python3.6 and python3.5.
#
ARG PYTHON_VERSION=3.6.8

USER root
WORKDIR /root/tar_files
RUN wget --quiet https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz
WORKDIR /root/python3.6
RUN tar -xf /root/tar_files/Python-${PYTHON_VERSION}.tar.xz
WORKDIR /root/python3.6/Python-${PYTHON_VERSION}
# Enable profile-guided optimizations.
# This takes a long time to build...is it worth it?
#RUN ./configure --enable-optimizations
#		--prefix=/usr \

# Enable sources; needed for "apt-get build-dep"
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies needed to build all the various python modules:
# _bz2                  _curses               _curses_panel
# _dbm                  _gdbm                 _lzma
# _sqlite3              _ssl                  _tkinter
# readline
RUN apt-get install -y build-essential zlib1g-dev libbz2-dev liblzma-dev libncurses5-dev \
    libreadline6-dev libsqlite3-dev libssl-dev \
    tk8.5-dev lzma lzma-dev libgdbm-dev
RUN ./configure \
    --enable-ipv6 \
    --enable-loadable-sqlite-extensions \
    --with-dbmliborder=bdb:gdbm \
    --with-computed-gotos \
    --without-ensurepip \
    --with-system-expat \
    --with-system-libmpdec \
    --with-system-ffi

RUN make -j$(nproc)
# Using altinstall will ensure that you don’t mess with the default system Python.
RUN make altinstall
RUN python3.6 --version
WORKDIR ${HOME}
USER ${RLSCOPE_USER}
