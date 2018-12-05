#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

NPROC="$(nproc)"

#CLONE_DIR
setup_python() {
#    apt-get install -y build-essential git libexpat1-dev libssl-dev zlib1g-dev \
#      libncurses5-dev libbz2-dev liblzma-dev \
#      libsqlite3-dev libffi-dev tcl-dev linux-headers-generic libgdbm-dev \
#      libreadline-dev tk tk-dev
    (
    cd $CLONE_DIR/cpython
    mkdir build
    cd build
    # https://stackoverflow.com/questions/8097161/how-would-i-build-python-myself-from-source-code-on-ubuntu
    #
    #     cd cpython && ./configure --prefix=/usr \
    #  --enable-loadable-sqlite-extensions \
    #  --enable-shared \
    #  --with-lto \
    #  --enable-optimizations \
    #  --with-system-expat \
    #  --with-system-ffi \
    #  --enable-ipv6 --with-threads --with-pydebug --disable-rpath \

    # MINE
    #    --prefix=/usr
#    ../configure --enable-optimizations \
#      --enable-loadable-sqlite-extensions \

    ../configure \
      --prefix=/usr \
      --enable-loadable-sqlite-extensions \
      --enable-shared \
      --with-lto \
      --enable-optimizations \
      --with-system-expat \
      --with-system-ffi \
      --enable-ipv6 --with-threads --with-pydebug --disable-rpath \
    make -j$NPROC
    make install
    )
}

main() {
#    echo HELLO WORLD
    setup_python
}

main "$@"