#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

SCRIPT_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
source $ROOT/dockerfiles/sh/docker_runtime_common.sh

python ./run.py bash -c "cd Debug && make -j$(nproc)"
python ./run.py gdbserver --once localhost:$LOGAN_GDB_PORT "$@"
