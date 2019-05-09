#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
set -x
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env

# Equivalent to: --compilation_mode=opt
# See: https://docs.bazel.build/versions/master/user-manual.html
BAZEL_BUILD_FLAGS=(-c opt)
BAZEL_EXPLAIN_LOG="$BAZEL_BUILD_DIR/bazel_explain.opt.log"
_bazel_build

#export TF_PRINT_TIMESTAMP=yes

#python3 $IML_DRILL/tests/iml_test_harness.py --test-name pong_redo_01 --train-script ~/clone/baselines/baselines/deepq/experiments/run_atari.py --env PongNoFrameskip-v4 --iml-start-measuring-call 1 --checkpoint-path IML_DIRECTORY --iml-trace-time-sec 40

# https://stackoverflow.com/questions/2524367/inline-comments-for-bash
# echo abc `#put your comment here` \
#      def `#another chance for a comment` \
#      xyz etc

iml-test \
    `# iml-test arguments ` \
    --test-name pong_docker \
    --train-script $BASELINES_DIR/baselines/deepq/experiments/run_atari.py \
    `# test-script iml arguments ` \
    --iml-start-measuring-call 1 \
    --iml-trace-time-sec 40 \
    `# test-script arguments ` \
    --env PongNoFrameskip-v4 \
    --checkpoint-path IML_DIRECTORY \
