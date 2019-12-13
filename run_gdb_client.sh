#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi
cd $(dirname $0)
gdb \
    --eval-command "source .gdbinit" \
    --eval-command "target remote localhost:$LOGAN_GDB_PORT" \
    --eval-command "continue" \

