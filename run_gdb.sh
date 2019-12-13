#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi
cd $(dirname $0)
./run.sh "cd Debug && make -j$(nproc)" 
./run.sh gdbserver --once localhost:$LOGAN_GDB_PORT "$@"
