#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
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
source $SH_DIR/docker_runtime_common.sh

main() {

  _check_rlscope_dir
  _upgrade_pip

  cd $RLSCOPE_DIR
  BUILD_PIP=yes bash ./setup.sh
  WHEEL_FILE="$(ls $RLSCOPE_DIR/dist/rlscope*.whl)"
  log_info "> RL-Scope wheel file output to: "
  log_info "  ${WHEEL_FILE}"
  log_info
  log_info "  You can install it by running:"
  log_info "  $ pip install ${WHEEL_FILE}"

}

main "$@"
