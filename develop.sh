#!/usr/bin/env bash
#
# Run this script to install a "development" version of rlscope.
# Do this to allow frequent modification of rlscope repo without reinstalling the .whl.
#
set -e
set -x

ROOT="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
source $ROOT/dockerfiles/sh/exports.sh
cd "$RLSCOPE_DIR"

python setup.py develop
