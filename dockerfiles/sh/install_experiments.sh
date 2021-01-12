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
  EXPERIMENTS=yes bash ./setup.sh
}

main "$@"

