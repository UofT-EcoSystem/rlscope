#!/usr/bin/env bash
set -e
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi
# cd $(dirname $0)

# $ ./run_gdb_host --args Debug/program ...
#   Source .gdbinit (may contain breakpoints).
#   Then, run gdb on provided program (--args).

SCRIPT_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
source $ROOT/dockerfiles/sh/docker_runtime_common.sh

gdb \
    --eval-command "source ${RLSCOPE_DIR}/.gdbinit" \
    --eval-command "run" \
    "$@"
