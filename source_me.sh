# NOTE: this will modify your PATH to include IML
# dependencies that were built from source (e.g. protobuf v3).
if [ "$IML_DIR" != "" ]; then
    ROOT="$IML_DIR"
else
    # Fails in container...not sure why.
    ROOT="$(readlink -f "$(dirname "$0")")"
fi

PYTHONPATH=${PYTHONPATH:-}

# From setup.sh
_iml_build_suffix() {
  local cuda_version="$1"
  shift 1
  # e.g. IML_BUILD_SUFFIX="_cuda_10_2"
  echo "_cuda_${cuda_version}" | sed 's/[\.]/_/g'
}
cmake_build_dir() {
    local third_party_dir="$1"
    shift 1

    local build_prefix=
    if _is_non_empty IML_BUILD_PREFIX; then
      # Docker container environment.
      build_prefix="$IML_BUILD_PREFIX"
    else
      # Assume we're running in host environment.
      build_prefix="$ROOT/local.host"
    fi
    local build_dir="$build_prefix/$(basename "$third_party_dir")"
    if _is_non_empty IML_BUILD_SUFFIX; then
      local build_dir="${build_dir}${IML_BUILD_SUFFIX}"
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
_local_dir() {
    # When installing things with configure/make-install
    # $ configure --prefix="$(_local_dir)"
    if _is_non_empty IML_INSTALL_PREFIX; then
      # Docker container environment.
      echo "$IML_INSTALL_PREFIX"
    else
      # Assume we're running in host environment.
      echo "$ROOT/local.host"
    fi
}
_build_dir() {
    local build_prefix=
    if _is_non_empty IML_BUILD_PREFIX; then
      # Docker container environment.
      build_prefix="$IML_BUILD_PREFIX"
    else
      # Assume we're running in host environment.
      build_prefix="$ROOT/local.host"
    fi
    echo "$build_prefix"
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
_add_pyenv() {
  local pyenv_dir=$IML_INSTALL_PREFIX/pyenv
  local bin_dir=$pyenv_dir/bin

  if [ -e $bin_dir ]; then
      export PYENV_ROOT="$pyenv_dir"
      _add_PATH "$PYENV_ROOT/bin"
      # export PATH="$PYENV_ROOT/bin:$PATH"

      if [ -x "$(which pyenv)" ]; then
          eval "$(pyenv init -)"
      fi

      # if [ -e $bin_dir/plugins/pyenv-virtualenv ]; then
      #     # automatically active pyenv-virtualenv if .python-version file present in directory:
      #     # https://github.com/pyenv/pyenv-virtualenv#activate-virtualenv
      #     eval "$(pyenv virtualenv-init -)"
      # fi
  fi

#  export PYENV_ROOT_DIR
}

(

IML_CUDA_VERSION=${IML_CUDA_VERSION:-10.1}
# Only add "_cuda_10_1" suffix to iml build directory.
IML_BUILD_SUFFIX="$(_iml_build_suffix ${IML_CUDA_VERSION})"
IML_BUILD_DIR="$(cmake_build_dir "$ROOT")"
unset IML_BUILD_SUFFIX

IML_INSTALL_PREFIX_DEFINED=no
if _is_non_empty IML_INSTALL_PREFIX; then
  IML_INSTALL_PREFIX_DEFINED=yes
fi
IML_BUILD_PREFIX_DEFINED=no
if _is_non_empty IML_BUILD_PREFIX; then
  IML_BUILD_PREFIX_DEFINED=yes
fi
set +u
echo "> INFO: [defined=${IML_INSTALL_PREFIX_DEFINED}] IML_INSTALL_PREFIX=${IML_INSTALL_PREFIX}"
echo "> INFO: [defined=${IML_BUILD_PREFIX_DEFINED}] IML_BUILD_PREFIX=${IML_BUILD_PREFIX}"
)
echo "> INFO: Using CMAKE_INSTALL_PREFIX=$(_local_dir)"
_add_LD_LIBRARY_PATH "$(_local_dir)/lib"
_add_PATH "$(_local_dir)/bin"
_add_PYTHONPATH "$(_local_dir)/lib/python3/dist-packages"
_add_pyenv

if [ "${IML_INSTALL_PREFIX:-}" = "" ]; then
  echo "> INFO: export IML_INSTALL_PREFIX=$(_local_dir)"
  export IML_INSTALL_PREFIX="$(_local_dir)"
fi
if [ "${IML_BUILD_PREFIX:-}" = "" ]; then
  echo "> INFO: export IML_BUILD_PREFIX=$(_build_dir)"
  export IML_BUILD_PREFIX="$(_build_dir)"
fi

unset _add_LD_LIBRARY_PATH
unset _add_PATH
unset ROOT
unset _local_dir
unset _is_non_empty
unset _add_pyenv
