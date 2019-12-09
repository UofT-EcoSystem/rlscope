# NOTE: this will modify your PATH to include IML 
# dependencies that were built from source (e.g. protobuf v3).
ROOT="$(readlink -f "$(dirname "$0")")"

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

PROTOBUF_INSTALL_DIR="$(ls -d $ROOT/third_party/protobuf-*/build/bin)"
_maybe_add_path "$PROTOBUF_INSTALL_DIR"
_maybe_add_path "$ROOT/local/bin"
unset PROTOBUF_INSTALL_DIR

unset _maybe_add_path
unset ROOT
