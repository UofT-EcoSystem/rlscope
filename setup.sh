#!/usr/bin/env bash
set -e
set -u

IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  echo "ERROR: please run setup.sh using bash (not zsh)"
  exit 1
fi


# Need a checkout of tensorflow to build:
# 1) Python pip package
# 2) C-API
#TENSORFLOW_SRC_ROOT=$HOME/clone/tensorflow
JOBS=${JOBS:-$(nproc)}
echo "> Using JOBS=$JOBS"

# If no, clone using https://github.com/$user/$repo
#   Works better in docker containers (no ssh key).
# If yes, clone using git@github.com:$user/$repo
#   Works better outside docker containers (ssh key and config present).
GIT_USE_SSH=${GIT_USE_SSH:-no}

LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

# If BUILD_PIP=yes, then build a python wheel.
BUILD_PIP=${BUILD_PIP:-no}
# If EXPERIMENTS=yes, then clone all experiments needed in experiments.
EXPERIMENTS=${EXPERIMENTS:-no}
# If SKIP_CPACK=yes, then skip calling "make package" to create a cmake binary archive (*.tar.gz)
# from rlscope binaries/libraries.
SKIP_CPACK=${SKIP_CPACK:-no}

if [ "$BUILD_PIP" = 'yes' ]; then
  RLSCOPE_BUILD_TYPE=Release
else
  RLSCOPE_BUILD_TYPE=Debug
fi

# NOTE: This is set in the Dockerfile, but you can override it.
RLSCOPE_CUDA_VERSION=${RLSCOPE_CUDA_VERSION:-10.1}

# Force re-running setup
FORCE=${FORCE:-no}
#if [ "$FORCE" = "" ]; then
#    FORCE=no
#fi

DEBUG=${DEBUG:-no}
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  ROOT="$(readlink -f "$(dirname "${0:A}")")"
else
  ROOT="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
source $ROOT/dockerfiles/sh/docker_runtime_common.sh

## Only add "_cuda_10_1" suffix to rlscope build directory.
#RLSCOPE_BUILD_SUFFIX="$(_rlscope_build_suffix ${RLSCOPE_CUDA_VERSION})"
#RLSCOPE_BUILD_DIR="$(cmake_build_dir "$ROOT")"
#unset RLSCOPE_BUILD_SUFFIX

#NCPU=$(grep -c ^processor /proc/cpuinfo)

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

GIT_PULL=no
GIT_RECURSIVE=yes
GIT_CLONE_OPTS=()
_git_clone() {
    local path="$1"
    local repo="$2"
    shift 2
    local commit=
    if [ $# -ge 1 ]; then
        commit="$1"
        shift 1
    fi
    if [ ! -e "$path" ]; then
        if [ "$GIT_RECURSIVE" != "no" ]; then
          GIT_CLONE_OPTS+=(--recursive)
        fi
        _do git clone "${GIT_CLONE_OPTS[@]}" $repo $path
        info "> Git clone:"
        info "  Repository: $repo"
        info "  Directory: $path"
    fi
    (
    cd $path
    git checkout $commit
    if [ "$GIT_PULL" = "yes" ]; then
      git pull
    fi
    if [ "$GIT_RECURSIVE" != "no" ]; then
      git submodule update --init
    fi
    )
}

_GIT_HAS_WARNED=no
_github_clone() {
    local path="$1"
    local name_slash_repo="$2"
    shift 2
    if [ "$GIT_USE_SSH" = "no" ] && [ "$_GIT_HAS_WARNED" = "no" ]; then
      log_warning "RL-Scope NOTE: set GIT_USE_SSH=yes if you have a github ssh key and want to clone repos using git@github.com:$name_slash_repo"
      _GIT_HAS_WARNED=yes
    fi
    local repo="$(github_url $name_slash_repo)"
    _git_clone "$path" "$repo" "$@"
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

    if [ "$GIT_USE_SSH" = "yes" ]; then
      # Doesn't work inside docker container (no ssh key)
      echo "git@github.com:$name_slash_repo"
    else
      echo "https://github.com/$name_slash_repo"
    fi
}

CMAKE_VERSION=3.15.1
setup_cmake() {
    if [ "$FORCE" != 'yes' ] && [ -e "$(_local_dir)"/bin/cmake ]; then
        return
    fi
    local url=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh
    _wget "$url"
    local path=$(_wget_output_path "$url")
    chmod +x $path
    mkdir -p "$(_local_dir)"
    bash $path --prefix="$(_local_dir)" --skip-license --exclude-subdir
}
JSON_CPP_LIB_DIR="$ROOT/third_party/json"
setup_json_cpp_library() {
#    if [ "$FORCE" != 'yes' ] && [ -e $JSON_CPP_LIB_DIR ]; then
    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$JSON_CPP_LIB_DIR")/include/nlohmann/json.hpp"; then
        return
    fi

    # Fails to build with gcc 9 on Ubuntu 20.04
    # https://github.com/nlohmann/json/pull/1492
    # local commit="v3.4.0"

    local commit="v3.9.1"
    _github_clone "$JSON_CPP_LIB_DIR" \
        nlohmann/json.git \
        $commit
    cmake_build "$JSON_CPP_LIB_DIR"
#    cd $JSON_CPP_LIB_DIR
#    _configure_make_install
}

#
# Repositories with RL-Scope annotations added for experiments needed to
# generate figures in RL-Scope paper.
#

setup_clone_experiments() {
  # _do setup_experiment_baselines
  _do setup_experiment_rl_baselines_zoo
  _do setup_experiment_stable_baselines
  _do setup_experiment_tf_agents
  _do setup_experiment_reagent
  _do setup_experiment_mlperf_training
  # _do setup_experiment_minigo
}

setup_install_experiments() {
  # _do install_baselines.sh
  _do install_stable_baselines.sh
  _do install_reagent.sh
  _do install_tf_agents.sh
  _do install_minigo.sh
}

setup_experiment_baselines() {
#    if [ "$FORCE" != 'yes' ] && [ -e $BASELINES_DIR/setup.py ]; then
#        return
#    fi
    local commit="iml"
    (
    GIT_PULL=yes
    _github_clone "$BASELINES_DIR" \
        UofT-EcoSystem/baselines.git \
        $commit
    )
}

setup_experiment_mlperf_training() {
#    if [ "$FORCE" != 'yes' ] && [ -e $MLPERF_DIR/README.md ]; then
#        return
#    fi
    local commit="iml"
    (
    GIT_PULL=yes
    # "community" submodule fails to pull.
    GIT_RECURSIVE=no
    _github_clone "$MLPERF_DIR" \
        UofT-EcoSystem/rlscope_mlperf_training.git \
        $commit
    )
}

setup_experiment_stable_baselines() {
#    if [ "$FORCE" != 'yes' ] && [ -e $STABLE_BASELINES_DIR/setup.py ]; then
#        return
#    fi
    local commit="iml-td3"
    (
    GIT_PULL=yes
    _github_clone "$STABLE_BASELINES_DIR" \
        UofT-EcoSystem/rlscope_stable-baselines.git \
        $commit
    )
}

setup_experiment_rl_baselines_zoo() {
#    if [ "$FORCE" != 'yes' ] && [ -e $RL_BASELINES_ZOO_DIR/train.py ]; then
#        return
#    fi
    local commit="iml-td3"
    (
    GIT_PULL=yes
    _github_clone "$RL_BASELINES_ZOO_DIR" \
        UofT-EcoSystem/rlscope_rl-baselines-zoo.git \
        $commit
    )
}

setup_experiment_tf_agents() {
#    if [ "$FORCE" != 'yes' ] && [ -e $TF_AGENTS_DIR/setup.py ]; then
#        return
#    fi
    local commit="iml"
    (
    GIT_PULL=yes
    _github_clone "$TF_AGENTS_DIR" \
        UofT-EcoSystem/rlscope_agents.git \
        $commit
    )
}

setup_experiment_reagent() {
#    if [ "$FORCE" != 'yes' ] && [ -e $REAGENT_DIR/setup.py ]; then
#        return
#    fi
    local commit="iml"
    (
    GIT_PULL=yes
    _github_clone "$REAGENT_DIR" \
        UofT-EcoSystem/rlscope_ReAgent.git \
        $commit
    )
}

#MINIGO_DIR="$ROOT/third_party/minigo"
#setup_experiment_minigo() {
#    if [ "$FORCE" != 'yes' ] && [ -e $MINIGO_DIR/setup.py ]; then
#        return
#    fi
#    local commit="iml"
#    (
#    GIT_PULL=yes
#    _github_clone "$MINIGO_DIR" \
#        UofT-EcoSystem/minigo.git \
#        $commit
#    )
#}

ABSEIL_CPP_LIB_DIR="$ROOT/third_party/abseil-cpp"
setup_abseil_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $ABSEIL_CPP_LIB_DIR ]; then
        return
    fi
    local commit="20181200"
    _github_clone "$ABSEIL_CPP_LIB_DIR" \
        abseil/abseil-cpp.git \
        $commit
}

# Return true if we see at least one file in the glob pattern.
glob_any() {
  local glob="$1"
  shift 1
  test -n "$(eval ls $glob 2>/dev/null || true)"
}

GTEST_CPP_LIB_DIR="$ROOT/third_party/googletest"
setup_gtest_cpp_library() {
    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$GTEST_CPP_LIB_DIR")/lib/libgtest*"; then
        return
    fi
    local commit="release-1.8.1"
    _github_clone "$GTEST_CPP_LIB_DIR" \
        google/googletest.git \
        $commit
    cmake_build "$GTEST_CPP_LIB_DIR"
}

GFLAGS_CPP_LIB_DIR="$ROOT/third_party/gflags"
setup_gflags_cpp_library() {
    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$GFLAGS_CPP_LIB_DIR")/lib/libgflags*"; then
        return
    fi
    local commit="v2.2.2"
    _github_clone "$GFLAGS_CPP_LIB_DIR" \
        gflags/gflags.git \
        $commit
    CMAKE_OPTS=(-DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON)
    cmake_build "$GFLAGS_CPP_LIB_DIR"
}

third_party_install_prefix() {
    local third_party_dir="$1"
    shift 1

    # Local install directory within cmake build directory:
    # echo "$(cmake_build_dir "$third_party_dir")/cmake_install"

    # Install in $ROOT/local.$(hostname)
    _local_dir
}

##
## BEGIN source_me.sh
##
_rlscope_build_suffix() {
  local cuda_version="$1"
  shift 1
  # e.g. RLSCOPE_BUILD_SUFFIX="_cuda_10_2"
  echo "_cuda_${cuda_version}" | sed 's/[\.]/_/g'
}
cmake_build_dir() {
    local third_party_dir="$1"
    shift 1

    local build_prefix=
    if _is_non_empty RLSCOPE_BUILD_PREFIX; then
      # Docker container environment.
      build_prefix="$RLSCOPE_BUILD_PREFIX/${RLSCOPE_BUILD_TYPE}"
    else
      # Assume we're running in host environment.
      build_prefix="$ROOT/local.host/${RLSCOPE_BUILD_TYPE}"
    fi
    local build_dir="$build_prefix/$(basename "$third_party_dir")"
    if _is_non_empty RLSCOPE_BUILD_SUFFIX; then
      local build_dir="${build_dir}${RLSCOPE_BUILD_SUFFIX}"
    fi
    echo "$build_dir"
}
_is_non_empty() {
  # Check if an environment variable is defined and not equal to empty string
  (
  set +u
  local varname="$1"
  # Indirect variable dereference that works with both bash and zsh.
  # https://unix.stackexchange.com/questions/68035/foo-and-zsh
  local value=
  eval "value=\"\$${varname}\""
  [ "${value}" != "" ]
  )
}
_rlscope_install_prefix() {
    # When installing things with configure/make-install
    # $ configure --prefix="$(_local_dir)"
#    if _is_non_empty RLSCOPE_INSTALL_PREFIX; then
    if _is_non_empty RLSCOPE_IS_DOCKER && [ "$RLSCOPE_IS_DOCKER" = 'yes' ]; then
      # Docker container environment.
      echo "$RLSCOPE_INSTALL_PREFIX"
    else
      # Assume we're running in host environment.
      echo "$ROOT/local.host"
    fi
}
_rlscope_build_prefix() {
    local build_prefix=
#    if _is_non_empty RLSCOPE_BUILD_PREFIX; then
    if _is_non_empty RLSCOPE_IS_DOCKER && [ "$RLSCOPE_IS_DOCKER" = 'yes' ]; then
      # Docker container environment.
      build_prefix="$RLSCOPE_BUILD_PREFIX"
    else
      # Assume we're running in host environment.
      build_prefix="$ROOT/build.host"
    fi
    echo "$build_prefix"
}
_local_dir() {
    echo "$(_rlscope_install_prefix)/${RLSCOPE_BUILD_TYPE}"
}
_build_dir() {
    echo "$(_rlscope_build_prefix)/${RLSCOPE_BUILD_TYPE}"
}
_add_PATH() {
    local direc="$1"
    shift 1

    echo "> INFO: Add to PATH: $direc"
    export PATH="$direc:$PATH"
}
_add_LD_LIBRARY_PATH() {
  local lib_dir="$1"
  shift 1

  echo "> INFO: Add to LD_LIBRARY_PATH: $lib_dir"
  export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
}
_add_PYTHONPATH() {
  local lib_dir="$1"
  shift 1

  echo "> INFO: Add to PYTHONPATH: $lib_dir"
  export PYTHONPATH="$lib_dir:$PYTHONPATH"
}
##
## END source_me.sh
##

NSYNC_CPP_LIB_DIR="$ROOT/third_party/nsync"
setup_nsync_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $NSYNC_CPP_LIB_DIR ]; then
        return
    fi
    local commit="1.21.0"
    _github_clone "$NSYNC_CPP_LIB_DIR" \
        google/nsync.git \
        $commit
}

BACKWARD_CPP_LIB_DIR="$ROOT/third_party/backward-cpp"
setup_backward_cpp_library() {
    if [ "$FORCE" != 'yes' ] && [ -e $BACKWARD_CPP_LIB_DIR ]; then
        return
    fi
    # local commit="v1.4"
    # Compilation fix for Ubuntu 20.04 binutils update.
    local commit="v1.5"
    _github_clone "$BACKWARD_CPP_LIB_DIR" \
        bombela/backward-cpp.git \
        $commit
}

SPDLOG_CPP_LIB_DIR="$ROOT/third_party/spdlog"
setup_spdlog_cpp_library() {
    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$SPDLOG_CPP_LIB_DIR")/lib/libspdlog*"; then
        return
    fi
    local commit="v1.4.2"
    _github_clone "$SPDLOG_CPP_LIB_DIR" \
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
    _github_clone "$CTPL_CPP_LIB_DIR" \
        vit-vit/CTPL.git \
        $commit
}

#BOOST_CPP_LIB_VERSION="1.70.0"
BOOST_CPP_LIB_VERSION="1.73.0"
BOOST_CPP_LIB_VERSION_UNDERSCORES="$(perl -lape 's/\./_/g'<<<"$BOOST_CPP_LIB_VERSION")"
BOOST_CPP_LIB_DIR="$ROOT/third_party/boost_${BOOST_CPP_LIB_VERSION_UNDERSCORES}"
BOOST_CPP_LIB_URL="https://dl.bintray.com/boostorg/release/${BOOST_CPP_LIB_VERSION}/source/boost_${BOOST_CPP_LIB_VERSION_UNDERSCORES}.tar.gz"
setup_boost_cpp_library() {
    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$BOOST_CPP_LIB_DIR")/lib/libboost_*"; then
        return
    fi
    _wget_tar "$BOOST_CPP_LIB_URL" "third_party"
    (
    cd $BOOST_CPP_LIB_DIR
    ./bootstrap.sh --prefix="$(third_party_install_prefix "$BOOST_CPP_LIB_DIR")"
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
#    ./b2 -j${JOBS} install
    # HACK: boost installer likes to run unit tests and I don't know how to disable them.
    # Inveitably, it ends up failing some unit tests but seems to install fine:
    #   ...failed updating 66 targets...
    #   ...skipped 24 targets...
    #   ...updated 16973 targets...
    # So, just ignore bad exit status so we can continue with everything else...
    ./b2 cxxflags="-fPIC" cflags="-fPIC" -j${JOBS} install || true
    )
}

EIGEN_CPP_LIB_DIR="$ROOT/third_party/eigen"
setup_eigen_cpp_library() {
    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$EIGEN_CPP_LIB_DIR")/include/eigen*"; then
        return
    fi
    # mercurial repo on bitbucket no longer supported; they use gitlab now.
#    local commit="9f48e814419e"
#    _hg_clone "$EIGEN_CPP_LIB_DIR" \
#        https://bitbucket.org/eigen/eigen \
#        $commit
    local commit="3.3.9"
    _git_clone "$EIGEN_CPP_LIB_DIR" \
        https://gitlab.com/libeigen/eigen.git \
        $commit
    cmake_build "$EIGEN_CPP_LIB_DIR"
}

CMAKE_OPTS=()
CMAKE_VERBOSE=${CMAKE_VERBOSE:-}
CMAKE_NO_CLEAR_OPTS=${CMAKE_NO_CLEAR_OPTS:-no}
cmake_build() {
    local src_dir="$1"
    shift 1

    local old_DO_VERBOSE="$DO_VERBOSE"
    DO_VERBOSE="$CMAKE_VERBOSE"
    (
    cd "$src_dir"
    mkdir -p "$(cmake_build_dir "$src_dir")"
    _dov cd "$(cmake_build_dir "$src_dir")"
    _dov cmake "$src_dir" -DCMAKE_INSTALL_PREFIX="$(third_party_install_prefix "$src_dir")" "${CMAKE_OPTS[@]}"
    _dov make -j${JOBS}
#    _dov make install -j${JOBS}
    _dov make install
    )
    if [ "$CMAKE_NO_CLEAR_OPTS" != 'yes' ]; then
      CMAKE_OPTS=()
      CMAKE_VERBOSE=
    fi
    DO_VERBOSE="$old_DO_VERBOSE"
}
cmake_prepare() {
    local src_dir="$1"
    shift 1

    local old_DO_VERBOSE="$DO_VERBOSE"
    DO_VERBOSE="$CMAKE_VERBOSE"
    (
    cd "$src_dir"
    mkdir -p "$(cmake_build_dir "$src_dir")"
    _dov cd "$(cmake_build_dir "$src_dir")"
    _dov cmake "$src_dir" -DCMAKE_INSTALL_PREFIX="$(third_party_install_prefix "$src_dir")" "${CMAKE_OPTS[@]}"
    )
    if [ "$CMAKE_NO_CLEAR_OPTS" != 'yes' ]; then
      CMAKE_OPTS=()
      CMAKE_VERBOSE=
    fi
    DO_VERBOSE="$old_DO_VERBOSE"
}
cmake_make() {
    local src_dir="$1"
    shift 1

    local old_DO_VERBOSE="$DO_VERBOSE"
    DO_VERBOSE="$CMAKE_VERBOSE"
    (
    cd "$src_dir"
    _dov cd "$(cmake_build_dir "$src_dir")"
    _dov make -j${JOBS} "$@"
    # _dov make install
    )
    if [ "$CMAKE_NO_CLEAR_OPTS" != 'yes' ]; then
      CMAKE_OPTS=()
      CMAKE_VERBOSE=
    fi
    DO_VERBOSE="$old_DO_VERBOSE"
}
cmake_install() {
    local src_dir="$1"
    shift 1

    local old_DO_VERBOSE="$DO_VERBOSE"
    DO_VERBOSE="$CMAKE_VERBOSE"
    (
    cd "$src_dir"
    _dov cd "$(cmake_build_dir "$src_dir")"
    _dov make install
    )
    if [ "$CMAKE_NO_CLEAR_OPTS" != 'yes' ]; then
      CMAKE_OPTS=()
      CMAKE_VERBOSE=
    fi
    DO_VERBOSE="$old_DO_VERBOSE"
}

#PROTOBUF_VERSION='3.6.1.2'

#PROTOBUF_VERSION='3.6.1'
#PROTOBUF_VERSION='3.9.1'
PROTOBUF_VERSION='3.14.0'

PROTOBUF_CPP_LIB_DIR="$ROOT/third_party/protobuf-${PROTOBUF_VERSION}"
PROTOBUF_URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz"
#PROTOBUF_URL="https://github.com/protocolbuffers/protobuf/archive/v${PROTOBUF_VERSION}.tar.gz"
setup_protobuf_cpp_library() {
    # NOTE: get errors when trying to use container built protobuf (-fPIC flag wasn't used)
#    if [ "$FORCE" != 'yes' ] && glob_any "$(third_party_install_prefix "$PROTOBUF_CPP_LIB_DIR")/lib/libprotobuf*"; then
    if [ "$FORCE" != 'yes' ] && (
      # protobuf is already installed during container build.
      ( which protoc > /dev/null 2>&1 ) ||
      ( glob_any "$(third_party_install_prefix "$PROTOBUF_CPP_LIB_DIR")/lib/libprotobuf*" )
    ); then
        return
    fi
    _wget_tar "$PROTOBUF_URL" "third_party"
    (
    cd $PROTOBUF_CPP_LIB_DIR
    # NOTE: if --prefix changes, we need to run "make clean" first.
    # NOTE: clean target may not exist.
    make clean || true
    ./configure \
        "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" \
        --prefix="$(third_party_install_prefix "$PROTOBUF_CPP_LIB_DIR")"
    make -j${JOBS}
    make install
    "$(third_party_install_prefix "$PROTOBUF_CPP_LIB_DIR")"/bin/protoc --version
    )
}

LIBPQXX_CPP_LIB_DIR="$ROOT/third_party/json"
LIBPQXX_VERSION="6.4.5"
setup_cpp_libpqxx() {
    if [ "$FORCE" != 'yes' ] && [ -e $LIBPQXX_CPP_LIB_DIR ]; then
        return
    fi
    local commit="$LIBPQXX_VERSION"
    _github_clone "$LIBPQXX_CPP_LIB_DIR" \
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
    _maybe make -j${JOBS}
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
#TENSORFLOW_LIB_DIR=external_libs
#_download_tensorflow_c_api() {
#    local libtensorflow_cpu_url="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.12.0.tar.gz"
#    _wget_tar "$libtensorflow_cpu_url" $TENSORFLOW_LIB_DIR
#}
#_build_tensorflow_c_api() {
#    local output_tar=$ROOT/libtensorflow.tar.gz
#    if [ ! -e $ROOT/libtensorflow.tar.gz ]; then
#        # Build tensorflow C-API package using bazel.
#        (
#        info "> Build TensorFlow C-API from source using Bazel:"
#        info "  SrcDir: $TENSORFLOW_SRC_ROOT"
#        info "  Output: $output_tar"
#
#        cd $TENSORFLOW_SRC_ROOT
#        bazel build --config opt //tensorflow/tools/lib_package:libtensorflow
#        )
#        cp $TENSORFLOW_SRC_ROOT/bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz \
#           $ROOT
#    fi
#    _untar $output_tar $TENSORFLOW_LIB_DIR
#}

_do() {
    echo "> CMD [setup.sh]:"
    echo "  $ $@"
    echo "  PWD=$PWD"
    "$@"
}
DO_VERBOSE=
_dov() {
  if [ "$DO_VERBOSE" = 'yes' ]; then
    echo "> CMD [setup.sh]:"
    echo "  $ $@"
    echo "  PWD=$PWD"
  fi
  "$@"
}

_apt_install() {
    # To make this as fast as possible, just depend on the user to do this.
    # sudo apt-get update
    sudo apt-get install -y "$@"
}

setup_apt_packages() {
    # binutils-dev
    #   Needed for pretty stack-traces from backward-cpp library during segfaults:
    #   https://github.com/bombela/backward-cpp#libraries-to-read-the-debug-info
    local APT_DEPS=(
        mercurial
        git
        binutils-dev
        gdb
        gdbserver
    )
    _apt_install "${APT_DEPS[@]}"
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

    _add_LD_LIBRARY_PATH "$(_local_dir)/lib"
    _add_PATH "$(_local_dir)/bin"

    if [ $# -gt 0 ]; then
        _do "$@"
        log_info "> Success!"
        return
    fi

    if [ "$EXPERIMENTS" = 'yes' ]; then
      _do setup_clone_experiments
      _do setup_install_experiments
      log_info "> Third-party RL repositories have been cloned in $RLSCOPE_DIR/third_party/* and installed in the current virtualenv $(which python)"
      log_info "  You can now run experiments:"
      log_info "    $ experiment_algorithm_choice.sh"
      log_info "    $ experiment_RL_framework_comparison.sh"
      log_info "    $ experiment_simulator_choice.sh"
      return
    fi

    #
    # Build RL-Scope from source (possibly a *.whl file also if BUILD_PIP=yes).
    #
    _do setup_cmake
    _do setup_apt_packages
    _do setup_json_cpp_library
    _do setup_abseil_cpp_library
    _do setup_gtest_cpp_library
    _do setup_gflags_cpp_library
    _do setup_nsync_cpp_library
    _do setup_backward_cpp_library
    _do setup_eigen_cpp_library
    _do setup_protobuf_cpp_library
    _do setup_boost_cpp_library
    _do setup_spdlog_cpp_library
    _do setup_ctpl_cpp_library
    # NOTE: IDEALLY we would install it do a SEPARATE directory... but I'm not 100% sure how to make that work nicely
    # and still have all the "installed" stuff in the same directory.
    (
    cuda_version=${RLSCOPE_CUDA_VERSION}
    _do _setup_project_with_cuda
    )

    log_info "> RL-Scope C++ components have been built (i.e., librlscope.so, rls-analyze)."
    log_info "  You can now use rls-prof to profile your RL training scripts."
}

setup_pip_package() {
  # Build a python wheel package.
  local build_dir=$(cmake_build_dir "$ROOT")
  if [ "$SKIP_CPACK" != 'yes' ]; then
    cmake_make "$ROOT" package
    local cpp_pkg=$(ls $build_dir/rlscope*.tar.gz)
    if [ "$cpp_pkg" = "" ]; then
      log_error "ERROR: setup_pip_package failed; couldn't find CPack archive at $build_dir/rlscope*.tar.gz"
      return 1
    fi
    # Cleanup anything from previously built packages.
    rm -rf $ROOT/rlscope/cpp/bin
    rm -rf $ROOT/rlscope/cpp/lib
    rm -rf $ROOT/rlscope/cpp/include
    # Extract newly built package into python tree.
    _do tar xf $cpp_pkg -C $ROOT/rlscope/cpp --strip-components=1
  fi

  # NOTE: I include "clean --all" since I've seen weird behaviour from setup.py where
  # if try to remove files from the wheel (e.g., with --debug-skip-cpp), they still get added.
  #
  # This has been observed elsewhere also...
  # https://github.com/pypa/wheel/issues/147
  _do python setup.py \
    clean --all \
    bdist_wheel --plat-name "$(_pip_platform_name)" \
    sdist
}

_pip_platform_name() {
  local PY_SCRIPT=
  read -r -d '' PY_SCRIPT << EOF || true
import distutils.util
platform = distutils.util.get_platform()
platform = platform.replace('linux', 'manylinux1')
print(platform)
EOF
  python -c "$PY_SCRIPT"
}

_setup_project() {
  # cmake will use whatever nvcc is on PATH, even if its outside of CUDA_TOOLKIT_ROOT_DIR
  export PATH=$CUDA_TOOLKIT_ROOT_DIR/bin:$PATH
  cmake_prepare "$ROOT"

  # The first time we build RLScope, for some reason, if we just do the normal "cmake .. && make && make install",
  # the "make install" triggers ANOTHER full build.  I have no idea WHY this happens.
  # It also only happens on the "intial" cmake build (i.e. subsequent builds only happen once).
  # Anyways, hacky workaround is to just call "make" twice the first time we build RLScope.
  cmake_make "$ROOT"
  cmake_make "$ROOT"

  cmake_install "$ROOT"

  if [ "$BUILD_PIP" = 'yes' ]; then
    _do setup_pip_package
  fi

  echo "To re-build RL-Scope library faster when you change source files, do:"
  echo "  $ cd $(cmake_build_dir "$ROOT")"
  echo "  $ make -j\$(nproc) install"
}

setup_project_cuda_10_2() {
  (
  cuda_version=10.2
  _setup_project_with_cuda
  )
}

setup_project_cuda_10_1() {
  (
  cuda_version=10.1
  _setup_project_with_cuda
  )
}

_rlscope_build_suffix() {
  local cuda_version="$1"
  shift 1
  # e.g. RLSCOPE_BUILD_SUFFIX="_cuda_10_2"
  echo "_cuda_${cuda_version}" | sed 's/[\.]/_/g'
}

_setup_project_with_cuda() {
  (
  set -u

  echo "> RL-Scope build environment variables:"
  echo "  - LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
  echo "  - PATH = ${PATH}"
  echo "  - cmake = $(which cmake), $(cmake --version | grep version)"
  # echo "  - LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"

  RLSCOPE_BUILD_SUFFIX="$(_rlscope_build_suffix ${cuda_version})"
  CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${cuda_version}
  CMAKE_OPTS=(-DCMAKE_BUILD_TYPE=${RLSCOPE_BUILD_TYPE} -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR})
  CMAKE_VERBOSE=yes
  CMAKE_NO_CLEAR_OPTS=yes
  DO_VERBOSE=yes
  _setup_project
  )
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    (
    cd $ROOT
    main "$@"
    )
else
    echo "> BASH: Sourcing ${BASH_SOURCE[0]}"
fi 


