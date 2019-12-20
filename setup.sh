#!/usr/bin/env bash
set -e

# Need a checkout of tensorflow to build:
# 1) Python pip package
# 2) C-API
TENSORFLOW_SRC_ROOT=$HOME/clone/tensorflow

# Force re-running setup
if [ "$FORCE" = "" ]; then
    FORCE=no
fi

if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

ROOT="$(readlink -f $(dirname "$0"))"

NCPU=$(grep -c ^processor /proc/cpuinfo)

info() {
    echo "$@"
}

_wget() {
    local url="$1"
    shift 1

    local path=$(_wget_output_path "$url")

    if [ ! -e "$path" ]; then
        info "> Download:"
        info "  URL: $url"
        info "  Path: $path"
        wget "$url" -O "$path"
    fi
}
_wget_output_path() {
    local url="$1"
    shift 1

    local path="$ROOT/$(basename $url)"
    echo "$path"
}

_archive_first_file() {
    local tar_file="$1"
    shift 1
    tar tf "$tar_file" | \
        perl -lape 's|^./||' | \
        sed '/^$/d' | \
        head -n 1
}
_archive_first_dir() {
    local tar_file="$1"
    shift 1
    tar tf "$tar_file" | \
        perl -lape 's|^./||' | \
        sed '/^$/d' | \
        grep --perl-regexp '/$' | \
        perl -lape 's|(^[^/]+)/.*|$1|' | \
        sort --unique | \
        head -n 1
}
_wget_tar() {
    local url="$1"
    local tar_dir="$2"
    shift 2

    _wget "$url"
    local path="$ROOT/$(basename $url)"

    local first_file="$(_archive_first_file $path)"
    # local first_dir=$(tar tf $path | perl -lape 's/(^[^\/]+)(\/.*)?/$1/' | sort --unique | head -n 1)
    # local wget_output_dir="$(dirname $path)/$first_dir" 
    if [ ! -e $ROOT/$tar_dir/$first_file ]; then
        (
        mkdir -p $ROOT/$tar_dir
        cd $ROOT/$tar_dir
        info "> Untar:"
        info "  Archive: $path"
        info "  Directory: $PWD"
        tar xf "$path"
        )
    fi
}

_untar() {
    local tar_file="$1"
    local tar_dir="$2"
    shift 2

    local first_file="$(_archive_first_file $tar_file)"
    if [ ! -e $ROOT/$tar_dir/$first_file ]; then
        (
        mkdir -p $ROOT/$tar_dir
        cd $ROOT/$tar_dir
        info "> Untar:"
        info "  Archive: $tar_file"
        info "  Directory: $PWD"
        tar xf "$tar_file"
        )
    fi

}

CLONE_TAG_PATTERN=
_clone() {
    local path="$1"
    local name_slash_repo="$2"
    shift 2
    local repo="$(github_url $name_slash_repo)"
    local commit=
    if [ $# -ge 1 ]; then
        commit="$1"
        shift 1
    elif [ "$CLONE_TAG_PATTERN" != '' ]; then
        commit="$(
            cd $dir
            _git_latest_tag_like "$CLONE_TAG_PATTERN"
        )"
    fi
    if [ ! -e "$path" ]; then
        git clone --recursive $repo $path
        info "> Git clone:"
        info "  Repository: $repo"
        info "  Directory: $path"
    fi
    (
    cd $path
    git checkout $commit
    git submodule update --init
    )
    CLONE_TAG_PATTERN=
}

_hg_clone() {
    local path="$1"
    local repo="$2"
    local commit="$3"
    shift 3

    if [ ! -e "$path" ]; then
        hg clone $repo $path
        info "> Hg clone:"
        info "  Repository: $repo"
        info "  Directory: $path"
    fi
    (
    cd $path
    hg update -r $commit
   )
}

github_url() {
    local name_slash_repo="$1"
    shift 1

    # Doesn't work inside docker container (no ssh key)
    # echo "git@github.com:$name_slash_repo"

    echo "https://github.com/$name_slash_repo"
}

CMAKE_VERSION=3.15.1
setup_cmake() {
    if [ "$FORCE" != 'yes' ] && [ -e $ROOT/local/bin/cmake ]; then
        return
    fi
    local url=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh
    _wget "$url"
    local path=$(_wget_output_path "$url")
    chmod +x $path
    mkdir -p $ROOT/local
    bash $path --prefix=$ROOT/local --skip-license --exclude-subdir
}
JSON_CPP_LIB_DIR="$ROOT/third_party/json"
setup_json_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $JSON_CPP_LIB_DIR ]; then
        return
    fi
    local commit="v3.4.0"
    _clone "$JSON_CPP_LIB_DIR" \
        nlohmann/json.git \
        $commit
#    cd $JSON_CPP_LIB_DIR
#    _configure_make_install
}

ABSEIL_CPP_LIB_DIR="$ROOT/third_party/abseil-cpp"
setup_abseil_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $ABSEIL_CPP_LIB_DIR ]; then
        return
    fi
    local commit="20181200"
    _clone "$ABSEIL_CPP_LIB_DIR" \
        abseil/abseil-cpp.git \
        $commit
}

GTEST_CPP_LIB_DIR="$ROOT/third_party/googletest"
setup_gtest_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $GTEST_CPP_LIB_DIR ]; then
        return
    fi
    local commit="release-1.8.1"
    _clone "$GTEST_CPP_LIB_DIR" \
        google/googletest.git \
        $commit
    (
    cd $GTEST_CPP_LIB_DIR
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/cmake_install
    make -j$(nproc)
    make install
    )
}

NSYNC_CPP_LIB_DIR="$ROOT/third_party/nsync"
setup_nsync_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $NSYNC_CPP_LIB_DIR ]; then
        return
    fi
    local commit="1.21.0"
    _clone "$NSYNC_CPP_LIB_DIR" \
        google/nsync.git \
        $commit
}

BACKWARD_CPP_LIB_DIR="$ROOT/third_party/backward-cpp"
setup_backward_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $BACKWARD_CPP_LIB_DIR ]; then
        return
    fi
    local commit="v1.4"
    _clone "$BACKWARD_CPP_LIB_DIR" \
        bombela/backward-cpp.git \
        $commit
}

SPDLOG_CPP_LIB_DIR="$ROOT/third_party/spdlog"
setup_spdlog_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $SPDLOG_CPP_LIB_DIR/build ]; then
        return
    fi
    local commit="v1.4.2"
    _clone "$SPDLOG_CPP_LIB_DIR" \
        gabime/spdlog.git \
        $commit
    cmake_build "$SPDLOG_CPP_LIB_DIR"
}

CTPL_CPP_LIB_DIR="$ROOT/third_party/CTPL"
setup_ctpl_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $CTPL_CPP_LIB_DIR ]; then
        return
    fi
    local commit="v.0.0.2"
    _clone "$CTPL_CPP_LIB_DIR" \
        vit-vit/CTPL.git \
        $commit
}

BOOST_CPP_LIB_VERSION="1.70.0"
BOOST_CPP_LIB_VERSION_UNDERSCORES="$(perl -lape 's/\./_/g'<<<"$BOOST_CPP_LIB_VERSION")"
BOOST_CPP_LIB_DIR="$ROOT/third_party/boost_${BOOST_CPP_LIB_VERSION_UNDERSCORES}"
BOOST_CPP_LIB_URL="https://dl.bintray.com/boostorg/release/${BOOST_CPP_LIB_VERSION}/source/boost_${BOOST_CPP_LIB_VERSION_UNDERSCORES}.tar.gz"
setup_boost_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $BOOST_CPP_LIB_DIR/build ]; then
        return
    fi
    _wget_tar "$BOOST_CPP_LIB_URL" "third_party"
    (
    cd $BOOST_CPP_LIB_DIR
    ./bootstrap.sh --prefix=$BOOST_CPP_LIB_DIR/build
    if [ ! -e project-config.jam.bkup ]; then
        # https://github.com/boostorg/build/issues/289#issuecomment-515712785
        # Compiling boost fails inside a virtualenv; complains about pyconfig.h not existing
        # since include-headers aren't properly detected in a virtualenv.
        # This is a documented (but not yet upstreamed) work-around.
        local py_ver="$(python --version | perl -lape 's/^Python //; s/(\d+\.\d+)\.\d+/$1/')";
        local py_path="$(which python)";
        local py_include="$(python -c "from sysconfig import get_paths; info = get_paths(); print(info['include']);")";
        sed --in-place=.bkup "s|using python : .*|using python : ${py_ver} : ${py_path} : ${py_include} ;|" project-config.jam
    fi
#    ./b2 -j$(nproc) install
    ./b2 cxxflags="-fPIC" cflags="-fPIC" -j$(nproc) install
    )
}

EIGEN_CPP_LIB_DIR="$ROOT/third_party/eigen"
setup_eigen_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $EIGEN_CPP_LIB_DIR ]; then
        return
    fi
    local commit="9f48e814419e"
    _hg_clone "$EIGEN_CPP_LIB_DIR" \
        https://bitbucket.org/eigen/eigen \
        $commit
    (
    cd $EIGEN_CPP_LIB_DIR
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/cmake_install
    make -j$(nproc)
    make install
    )
}

cmake_build() {
    local src_dir="$1"
    shift 1

    (
    cd "$src_dir"
    mkdir -p build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/cmake_install
    make -j$(nproc)
    make install
    )
}

#PROTOBUF_VERSION='3.6.1.2'

#PROTOBUF_VERSION='3.6.1'
PROTOBUF_VERSION='3.9.1'

PROTOBUF_CPP_LIB_DIR="$ROOT/third_party/protobuf-${PROTOBUF_VERSION}"
PROTOBUF_URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz"
#PROTOBUF_URL="https://github.com/protocolbuffers/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz"
setup_protobuf_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $PROTOBUF_CPP_LIB_DIR ]; then
        return
    fi
    _wget_tar "$PROTOBUF_URL" "third_party"
    (
    cd $PROTOBUF_CPP_LIB_DIR
    ./configure \
        "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" \
        --prefix="${PROTOBUF_CPP_LIB_DIR}/build"
    make -j$(nproc)
    make install
    ${PROTOBUF_CPP_LIB_DIR}/build/bin/protoc --version
    )
}

LIBPQXX_CPP_LIB_DIR="$ROOT/third_party/json"
LIBPQXX_VERSION="6.4.5"
setup_cpp_libpqxx() {
    if [ "$FORCE" != 'yes' ] && [ -e $LIBPQXX_CPP_LIB_DIR ]; then
        return
    fi
    local commit="$LIBPQXX_VERSION"
    _clone "$LIBPQXX_CPP_LIB_DIR" \
        jtv/libpqxx.git \
        $commit
#    cd $LIBPQXX_CPP_LIB_DIR
#    _configure_make_install
}


_configure() {
    local install_dir="$1"
    shift 1
    if [ ! -e ./configure ]; then
        if [ -e ./autogen.sh ]; then
            ./autogen.sh
        elif [ -e ./configure.ac ]; then
            autoreconf
        fi
    fi
    _maybe ./configure "${CONFIG_FLAGS[@]}" --prefix=$install_dir
}
_configure_make_install()
{
    (
#    export LDFLAGS="$LDFLAGS ${CONFIG_LDFLAGS[@]}"
#    export CXXFLAGS="$CXXFLAGS ${CONFIG_CXXFLAGS[@]}"
#    export CFLAGS="$CFLAGS ${CONFIG_CFLAGS[@]}"
    _configure
    _maybe make -j$NCPU
    _maybe make install
    )
    CONFIG_FLAGS=()
}


_DEBUG() {
    [ "$DEBUG" = 'yes' ]
}
_DEBUG_SHELL() {
    [ "$DEBUG_SHELL" = 'yes' ]
}
#CONFIG_FLAGS=()
#CONFIG_LDFLAGS=(-Wl,-rpath,$INSTALL_DIR/lib -L$INSTALL_DIR/lib)
#CONFIG_CFLAGS=(-I$INSTALL_DIR/include)
#CONFIG_CXXFLAGS=(-I$INSTALL_DIR/include)
_maybe() {
    set +x
    # Maybe run the command.
    # If DEBUG, just print it, else run it.
    if _DEBUG; then
        echo "$ $@"
        if _DEBUG_SHELL; then
            set -x
        fi
        return
    fi
    if _DEBUG_SHELL; then
        set -x
    fi
    "$@"
}

# $ROOT/external_libs
# |--- lib/
#      |--- libtensorflow.so
#      |--- libtensorflow_framework.so
# |--- include/
# ...
TENSORFLOW_LIB_DIR=external_libs
_download_tensorflow_c_api() {
    local libtensorflow_cpu_url="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz"
    _wget_tar "$libtensorflow_cpu_url" $TENSORFLOW_LIB_DIR
}
_build_tensorflow_c_api() {
    local output_tar=$ROOT/libtensorflow.tar.gz
    if [ ! -e $ROOT/libtensorflow.tar.gz ]; then
        # Build tensorflow C-API package using bazel.
        (
        info "> Build TensorFlow C-API from source using Bazel:"
        info "  SrcDir: $TENSORFLOW_SRC_ROOT"
        info "  Output: $output_tar"

        cd $TENSORFLOW_SRC_ROOT
        bazel build --config opt //tensorflow/tools/lib_package:libtensorflow
        )
        cp $TENSORFLOW_SRC_ROOT/bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz \
           $ROOT
    fi
    _untar $output_tar $TENSORFLOW_LIB_DIR
}

_do() {
    echo "> CMD: $@"
    echo "  $ $@"
    "$@"
}

_apt_install() {
    sudo apt install -y "$@"
}

setup_apt_packages() {
    # binutils-dev
    #   Needed for pretty stack-traces from backward-cpp library during segfaults:
    #   https://github.com/bombela/backward-cpp#libraries-to-read-the-debug-info
    _apt_install \
        mercurial \
        git \
        'libgflags*' \
        binutils-dev \

        # libgflags-dev libgflags2v5

}

main() {
    # PROBLEM: I've found that if you don't use the same
    # version of TensorFlow when sharing trained models across C++ and python, C++ can
    # deadlock when loading a model trained in python.
    #
    # To be safe, you're better off building everything from source; i.e.
    # 1) Build TensorFlow from source and install with pip
    # 2) Build TensorFlow C-API and build against that.
#    _download_tensorflow_c_api
#    _build_tensorflow_c_api

    if [ $# -gt 0 ]; then
        _do "$@"
        return
    fi

    _do setup_apt_packages
    _do setup_json_cpp_library
    _do setup_abseil_cpp_library
    _do setup_gtest_cpp_library
    _do setup_nsync_cpp_library
    _do setup_backward_cpp_library
    _do setup_eigen_cpp_library
    _do setup_protobuf_cpp_library
    _do setup_boost_cpp_library
    _do setup_spdlog_cpp_library
    _do setup_ctpl_cpp_library
    # TODO: setup protobuf
   _do setup_cmake
    echo "> Success!"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    (
    cd $ROOT
    main "$@"
    )
else
    echo "> BASH: Sourcing ${BASH_SOURCE[0]}"
fi 

