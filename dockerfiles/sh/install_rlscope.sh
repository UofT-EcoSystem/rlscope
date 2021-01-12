#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

main() {
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
  source $SH_DIR/make_utils.sh

  _check_rlscope_dir
  _upgrade_pip

  cd $IML_DIR
  _do build_pip.sh
  _do pip uninstall -y rlscope || true
  # NOTE: whl won't install if we are in the rlscope repo directory; cd to $HOME first.
  cd $HOME
  # --force-reinstall
  _do pip install $IML_DIR/dist/rlscope*.whl
  # develop_rlscope.sh
}

main "$@"

