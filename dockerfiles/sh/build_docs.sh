#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

main() {
  IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
  if [ "$IS_ZSH" = 'yes' ]; then
    SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
  else
    SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
  fi
  source $SH_DIR/docker_runtime_common.sh

  set -u
  cd $RLSCOPE_DIR
  make -f Makefile.sphinx docs
}

main "$@"

