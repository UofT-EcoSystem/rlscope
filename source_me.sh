# NOTE: this will modify your PATH to include IML
# dependencies that were built from source (e.g. protobuf v3).
if [ "$RLSCOPE_DIR" != "" ]; then
    ROOT="$RLSCOPE_DIR"
else
    # Fails in container...not sure why.
    # ROOT="$(readlink -f "$(dirname "$0")")"

    IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
    if [ "$IS_ZSH" = 'yes' ]; then
      ROOT="$(readlink -f "$(dirname "${0:A}")")"
    else
      ROOT="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
    fi
fi
source $ROOT/dockerfiles/sh/exports.sh

PYTHONPATH=${PYTHONPATH:-}

# NOTE: put Debug paths (out of build_rlscope) on our PATH by default.
# build_wheel will trigger Release builds.
RLSCOPE_BUILD_TYPE=Debug

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

(

RLSCOPE_CUDA_VERSION=${RLSCOPE_CUDA_VERSION:-10.1}
# Only add "_cuda_10_1" suffix to rlscope build directory.
RLSCOPE_BUILD_SUFFIX="$(_rlscope_build_suffix ${RLSCOPE_CUDA_VERSION})"
RLSCOPE_BUILD_DIR="$(cmake_build_dir "$ROOT")"
unset RLSCOPE_BUILD_SUFFIX

RLSCOPE_INSTALL_PREFIX_DEFINED=no
if _is_non_empty RLSCOPE_INSTALL_PREFIX; then
  RLSCOPE_INSTALL_PREFIX_DEFINED=yes
fi
RLSCOPE_BUILD_PREFIX_DEFINED=no
if _is_non_empty RLSCOPE_BUILD_PREFIX; then
  RLSCOPE_BUILD_PREFIX_DEFINED=yes
fi
set +u
echo "> INFO: [defined=${RLSCOPE_INSTALL_PREFIX_DEFINED}] RLSCOPE_INSTALL_PREFIX=${RLSCOPE_INSTALL_PREFIX}"
echo "> INFO: [defined=${RLSCOPE_BUILD_PREFIX_DEFINED}] RLSCOPE_BUILD_PREFIX=${RLSCOPE_BUILD_PREFIX}"
)
echo "> INFO: Using CMAKE_INSTALL_PREFIX=$(_local_dir)"
_add_LD_LIBRARY_PATH "$(_local_dir)/lib"
_add_PATH "$(_local_dir)/bin"
_add_PATH "${ROOT}/dockerfiles/sh"
# NOTE: these scripts are out of date and probably don't work.
#_add_PATH "$ROOT/dev/sh"
_add_PYTHONPATH "$(_local_dir)/lib/python3/dist-packages"

if [ "${RLSCOPE_INSTALL_PREFIX:-}" = "" ]; then
  echo "> INFO: export RLSCOPE_INSTALL_PREFIX=$(_rlscope_install_prefix)"
  export RLSCOPE_INSTALL_PREFIX="$(_rlscope_install_prefix)"
fi
if [ "${RLSCOPE_BUILD_PREFIX:-}" = "" ]; then
  echo "> INFO: export RLSCOPE_BUILD_PREFIX=$(_rlscope_build_prefix)"
  export RLSCOPE_BUILD_PREFIX="$(_rlscope_build_prefix)"
fi

unset _add_LD_LIBRARY_PATH
unset _add_PATH
unset ROOT
unset _local_dir
unset _is_non_empty

