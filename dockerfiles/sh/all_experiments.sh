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
  IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
  if [ "$IS_ZSH" = 'yes' ]; then
    SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
  else
    SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
  fi
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

