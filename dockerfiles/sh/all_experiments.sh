#!/usr/bin/env bash
#
# Generate the "RL framework comparison" figures from the RL-Scope paper.
#
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi

main() {
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
  source $SH_DIR/docker_runtime_common.sh

  _check_rlscope_dir
  _upgrade_pip

  cd $RLSCOPE_DIR

  # TODO: remove...
  export repetitions=2

  _do experiment_algorithm_choice.sh
  _do experiment_simulator_choice.sh
  _do experiment_RL_framework_comparison.sh

  log_info
  (
  TXT_BOLD=yes
  log_info "> Success!"
  )
  log_info "  Plots have been output @ $ARTIFACTS_DIR"

}

main "$@"

