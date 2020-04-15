#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env
_upgrade_pip

if [ "${IML_DIR}" = "" ]; then
    # Install directly from git repo.
    _do pip install git+https://github.com/UofT-EcoSystem/iml.git
else
    # Install from local checkout of repo.
    _do cd "${IML_DIR}"
    _do python setup.py develop
fi

if [ "${IML_DRILL_DIR}" = "" ]; then
    # Install directly from git repo.
    _do pip install git+https://github.com/jagleeso/iml-drill.git
else
    # Install from local checkout of repo.
    _do cd "${IML_DRILL_DIR}"
    _do python setup.py develop
fi

