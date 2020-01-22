#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi
cd $(dirname $0)
python ./run.py bash -c "cd Debug && make -j$(nproc)"
python ./run.py gdbserver --once localhost:$LOGAN_GDB_PORT "$@"
