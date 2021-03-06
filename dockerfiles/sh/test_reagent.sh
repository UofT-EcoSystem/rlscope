#!/usr/bin/env bash
# NOTE: This should run inside a docker container.
#
# NOTE: It's up to you to call ./configure from inside the TENSORFLOW_DIR of the container!
# This script is just meant for doing iterative rebuilds (i.e. "make")
set -eu
FORCE={FORCE:-no}
DEBUG=${DEBUG-no}
# Python version to using in ReAgent virtual environment (using pyenv)
PYTHON_VERSION=3.8.2
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
else
  SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
cd "$SH_DIR"

source $SH_DIR/docker_runtime_common.sh

_check_apt
_check_env
_upgrade_pip

_check_REAGENT_DIR

#_check_tensorflow
_check_rlscope

# Dependencies taken from:
# rl-baselines-zoo/docker/Dockerfile.gpu
# commit/tag: v1.2

source $RLSCOPE_DIR/source_me.sh

main() {
(
  # Test that ReAgent installation works
  cd $REAGENT_DIR
  export CONFIG=$REAGENT_DIR/reagent/gym/tests/configs/cartpole/discrete_dqn_cartpole_online.yaml
  _do python3 $REAGENT_DIR/reagent/workflow/cli.py run reagent.gym.tests.test_gym.run_test $CONFIG
)
}
main "$@"
