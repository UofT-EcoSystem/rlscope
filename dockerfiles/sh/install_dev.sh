#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/docker_runtime_common.sh

_check_apt
_check_env
_upgrade_pip

install_tensorflow_v1.3.1.sh
install_stable_baselines.sh
install_rlscope.sh
