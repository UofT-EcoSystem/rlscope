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
  _do build_wheel.sh
  _do pip uninstall -y rlscope || true
  # NOTE: whl won't install if we are in the rlscope repo directory; cd to $HOME first.
  cd $HOME
  # --force-reinstall
  _do pip install $RLSCOPE_DIR/dist/rlscope*.whl
  # develop_rlscope.sh
}

main "$@"

