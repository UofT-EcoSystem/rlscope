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

source $SH_DIR/docker_runtime_common.sh

# NOTE: we DON'T run "sudo apt update" since doing so at the start
# of the container causes an "apt busy" error.
# _check_apt
_check_env
_upgrade_pip

RLSCOPE_DRILL_DIR=${RLSCOPE_DRILL_DIR:-}

if [ "${RLSCOPE_DIR}" = "" ]; then
    # Install directly from git repo.
    _do pip install git+https://github.com/UofT-EcoSystem/rlscope.git
else
    # Install from local checkout of repo.
    _do cd "${RLSCOPE_DIR}"
    _do python setup.py develop
fi

if [ "${RLSCOPE_DRILL_DIR}" = "" ]; then
    # SKIP
    true
    # Install directly from git repo.
    # _do pip install git+https://github.com/jagleeso/rlscope-drill.git
else
    # Install from local checkout of repo.
    _do cd "${RLSCOPE_DRILL_DIR}"
    _do python setup.py develop
fi

