#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
#set -e
#set -x

_do() {
    echo "> CMD: $@"
    "$@"
}

BAZEL_BUILD_FLAGS=()
BAZEL_BUILD_TARGET=
BAZEL_EXPLAIN_LOG=
_bazel_build() {
    if [ "$BAZEL_BUILD_TARGET" = "" ]; then
        BAZEL_BUILD_TARGET="tensorflow/tools/pip_package:build_pip_package"
    fi
    if [ "$BAZEL_EXPLAIN_LOG" = "" ]; then
        BAZEL_EXPLAIN_LOG="$BAZEL_BUILD_DIR/bazel_explain.log"
    fi
    _do bazel \
        --output_user_root=$BAZEL_BUILD_DIR \
        build \
        "${BAZEL_BUILD_FLAGS[@]}" \
        --copt=-mavx --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        $BAZEL_BUILD_TARGET \
        --explain=$BAZEL_EXPLAIN_LOG \
        --verbose_explanations \
        --verbose_failures \
        -s
    BAZEL_BUILD_TARGET=
    BAZEL_BUILD_FLAGS=()
    BAZEL_EXPLAIN_LOG=
}

_check_env() {
    #
    # TODO: these should be optional (i.e. if TENSORFLOW_DIR isn't provided, install it from a fresh git repo checkout).
    #
    if [ "$TENSORFLOW_DIR" = "" ] || [ ! -d "$TENSORFLOW_DIR" ]; then
        echo "ERROR: environment variable TENSORFLOW_DIR should be set to: The root directory of a 'patched' TensorFlow checkout."
        exit 1
    fi

    if [ "$BAZEL_BUILD_DIR" = "" ] || [ ! -d "$BAZEL_BUILD_DIR" ]; then
        echo "ERROR: environment variable TENSORFLOW_DIR should be set to: The local path where we should output bazel objects (overrides $HOME/.cache/bazel)."
        exit 1
    fi

    if [ "$IML_DIR" = "" ] || [ ! -d "$IML_DIR" ]; then
        echo "ERROR: environment variable IML_DIR should be set to: The root directory of the iml_profiler repo checkout."
        exit 1
    fi

}

#_check_container_env() {
#}
