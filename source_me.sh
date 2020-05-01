# NOTE: this will modify your PATH to include IML 
# dependencies that were built from source (e.g. protobuf v3).
if [ "$IML_DIR" != "" ]; then
    ROOT="$IML_DIR"
else
    # Fails in container...not sure why.
    ROOT="$(readlink -f "$(dirname "$0")")"
fi
_local_dir=$ROOT/local.$(hostname)

_maybe_add_path() {
    local direc="$1"
    shift 1

    if [ -d "$direc" ]; then
        echo "> INFO: Add to PATH: $direc"
        export PATH="$direc:$PATH"
    else
        echo "> WARNING: SKIP adding non-existent directory to PATH: $direc"
        echo "  Did you run ./setup.sh?"
    fi
}

PROTOBUF_INSTALL_DIR="$(ls -d $ROOT/third_party/protobuf-*/build.$(hostname)/bin)"
_maybe_add_path "$PROTOBUF_INSTALL_DIR"
_maybe_add_path "$_local_dir/bin"
# Keep gdb-symbols for debugging.
# For some reason, cmake "install()" isn't finding compiled third_party libraries.
#_maybe_add_path "$ROOT/local/Debug/bin"
_maybe_add_path "$ROOT/Debug.$(hostname)"
unset PROTOBUF_INSTALL_DIR

unset _maybe_add_path
unset ROOT
unset _local_dir
