#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
else
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
cd "$SH_DIR"

source $SH_DIR/docker_runtime_common.sh

# NOTE: we DON'T run "sudo apt update" since doing so at the start
# of the container causes an "apt busy" error.
# _check_apt
_check_env
_upgrade_pip

_do bash -c "pip uninstall -y rlscope || true"

# Install from local checkout of repo.
_do cd "${RLSCOPE_DIR}"
_do python setup.py develop
_do pip install -r $RLSCOPE_DIR/requirements.txt
_do pip install -r $RLSCOPE_DIR/requirements.docs.txt
_do pip install -r $RLSCOPE_DIR/requirements.develop.txt

## Install from local checkout of repo.
#RLSCOPE_DRILL_DIR=${RLSCOPE_DRILL_DIR:-}
#_do cd "${RLSCOPE_DRILL_DIR}"
#_do python setup.py develop
