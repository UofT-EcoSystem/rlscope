# NOTE: this will modify your PATH to include IML 
# dependencies that were built from source (e.g. protobuf v3).
if [ "$IML_DIR" != "" ]; then
    ROOT="$IML_DIR"
else
    # Fails in container...not sure why.
    ROOT="$(readlink -f "$(dirname "$0")")"
fi

# From setup.sh
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

(
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

unset _add_LD_LIBRARY_PATH
unset _add_PATH
unset ROOT
unset _local_dir
unset _is_non_empty
