#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi
# cd $(dirname $0)

# $ ./run_gdb_host --args Debug/program ...
#   Source .gdbinit (may contain breakpoints).
#   Then, run gdb on provided program (--args).
_IML_DIR="$(dirname "$0")"
gdb \
    --eval-command "source ${_IML_DIR}/.gdbinit" \
    --eval-command "run" \
    "$@"
