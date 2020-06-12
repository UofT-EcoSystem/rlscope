# NOTE: this will modify your PATH to include IML 
# dependencies that were built from source (e.g. protobuf v3).
if [ "$IML_DIR" != "" ]; then
    ROOT="$IML_DIR"
else
    # Fails in container...not sure why.
    ROOT="$(readlink -f "$(dirname "$0")")"
fi

# From setup.sh
_is_defined() {
  # Check if an environment variable is defined:
  # https://unix.stackexchange.com/questions/402067/bash-find-if-all-env-variables-are-declared-by-variable-name
  # NOTE: returns TRUE if varname is empty-string.
  # If you want empty-string to be false, check should be:
  #   if _is_defined $VAR && [ "$VAR" != "" ]; then
  #     ...
  #   fi
  (
  set +u
  local varname="$1"
  [ -z ${!varname}+x ]
  )
}
_local_dir() {
    # When installing things with configure/make-install
    # $ configure --prefix="$(_local_dir)"
    if _is_defined IML_INSTALL_PREFIX; then
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

_add_LD_LIBRARY_PATH "$(_local_dir)/lib"
_add_PATH "$(_local_dir)/bin"

unset _add_LD_LIBRARY_PATH
unset _add_PATH
unset ROOT
unset _local_dir
unset _is_defined
