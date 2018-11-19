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
    local repo="$2"
    shift 2
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

setup_cmake() {
    if [ "$FORCE" != 'yes' ] && [ -e $ROOT/local/bin/cmake ]; then
        return
    fi
    local url=https://cmake.org/files/v3.13/cmake-3.13.0-rc3-Linux-x86_64.sh
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
        git@github.com:nlohmann/json.git \
        $commit
#    cd $JSON_CPP_LIB_DIR
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

main() {
    # PROBLEM: I've found that if you don't use the same
    # version of TensorFlow when sharing trained models across C++ and python, C++ can
    # deadlock when loading a model trained in python.
    #
    # To be safe, you're better off building everything from source; i.e.
    # 1) Build TensorFlow from source and install with pip
    # 2) Build TensorFlow C-API and build against that.
#    _download_tensorflow_c_api
    _build_tensorflow_c_api
    setup_json_cpp_library
    setup_cmake
}

(
cd $ROOT
main "$@"
)
