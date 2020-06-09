# NOTE: this will modify your PATH to include IML 
# dependencies that were built from source (e.g. protobuf v3).
if [ "$IML_DIR" != "" ]; then
    ROOT="$IML_DIR"
else
    # Fails in container...not sure why.
    ROOT="$(readlink -f "$(dirname "$0")")"
fi

# From setup.sh
_local_dir() {
    # When installing things with configure/make-install
    # $ configure --prefix="$(local_dir)"
    if [ "$IML_INSTALL_PREFIX" != "" ]; then
      echo "$IML_INSTALL_PREFIX"
    else
      # Assume we're running in host environment.
      echo "$ROOT/local.host"
    fi
    # echo "$ROOT/local.$(hostname)"
}

_maybe_add_path() {
    local direc="$1"
    shift 1

    if [ -d "$direc" ]; then
        echo "> INFO: Add to PATH: $direc"
        export PATH="$direc:$PATH"
    else
        echo "> WARNING: SKIP adding non-existent directory to PATH: $direc"
        echo "  Perhaps you forgot to run setup:"
        echo "  $ bash setup.sh"
    fi
}

#BOOST_VERSION=1.73.0
#BOOST_VERSION_UNDERSCORES=1_73_0
#PROTOBUF_VERSION=3.9.1
#THIRD_PARTY_LIB_DIRNAMES=(
#        json
#        abseil-cpp
#        nsync
#        backward-cpp
#        eigen
#        protobuf-${PROTOBUF_VERSION}
#        # libpqxx
#        boost_${BOOST_VERSION_UNDERSCORES}
#        spdlog
#        CTPL
#        googletest
#        )
_maybe_add_LD_LIBRARY_PATH() {
  local lib_dir="$1"
  shift 1
  if [ -d "$lib_dir" ]; then
    # Prepend in case there exists a system installation.
    echo "> INFO: Add to LD_LIBRARY_PATH: $lib_dir"
    export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
#  else
#    echo "> SKIP LD_LIBRARY_PATH (doesn't exist): $lib_dir"
  fi
}
#_add_third_party_LD_LIBRARY_PATH() {
#  for third_party in "${THIRD_PARTY_LIB_DIRNAMES[@]}"; do
#    local lib_dir=
#    # Things we built in setup.sh using cmake.
#    lib_dir="$PWD/third_party/$third_party/build.$(hostname)/cmake_install/lib"
#    _maybe_add_LD_LIBRARY_PATH "$lib_dir"
#    # Things we built in setup.sh using other tools (configure/make-install, boost's weird custom build system)
#    # e.g., protobuf, boost
#    lib_dir="$PWD/third_party/$third_party/build.$(hostname)/lib"
#    _maybe_add_LD_LIBRARY_PATH "$lib_dir"
#  done
#}
#_add_third_party_LD_LIBRARY_PATH

_maybe_add_LD_LIBRARY_PATH "$(_local_dir)/lib"

unset _maybe_add_LD_LIBRARY_PATH
#unset _add_third_party_LD_LIBRARY_PATH
#unset BOOST_VERSION
#unset BOOST_VERSION_UNDERSCORES
#unset PROTOBUF_VERSION
#unset THIRD_PARTY_LIB_DIRNAMES

#PROTOBUF_INSTALL_DIR="$(ls -d $ROOT/third_party/protobuf-*/build.$(hostname)/bin)"
#_maybe_add_path "$PROTOBUF_INSTALL_DIR"

_maybe_add_path "$(_local_dir)/bin"
# Keep gdb-symbols for debugging.
# For some reason, cmake "install()" isn't finding compiled third_party libraries.
#_maybe_add_path "$ROOT/local/Debug/bin"
#_maybe_add_path "$ROOT/Debug.$(hostname)/bin"
unset PROTOBUF_INSTALL_DIR

unset _maybe_add_path
unset ROOT
unset _local_dir
