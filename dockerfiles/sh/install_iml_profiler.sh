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

if [ "${IML_DIR}" = "" ]; then
    # Install directly from git repo.
    pip install git+https://github.com/UofT-EcoSystem/iml.git
else
    # Install from local checkout of repo.
    cd "${IML_DIR}"
    python setup.py develop
fi
