#
# Location of third party experiment repo directories (e.g., BASELINES_DIR).
# These variables are used by installation scripts (e.g., setup.sh, install_baselines.sh, etc.)
#
IS_ZSH="$(ps -ef | grep $$ | grep -v --perl-regexp 'grep|ps ' | grep zsh --quiet && echo yes || echo no)"
if [ "$IS_ZSH" = 'yes' ]; then
  _SH_DIR="$(readlink -f "$(dirname "${0:A}")")"
else
  _SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
fi
_ROOT_DIR="$(dirname "$(dirname "$_SH_DIR")")"

export RLSCOPE_DIR=${RLSCOPE_DIR:-$_ROOT_DIR}
#export BASELINES_DIR="$RLSCOPE_DIR/third_party/baselines"
export MLPERF_DIR="$RLSCOPE_DIR/third_party/rlscope_mlperf_training"
export STABLE_BASELINES_DIR="$RLSCOPE_DIR/third_party/rlscope_stable-baselines"
export RL_BASELINES_ZOO_DIR="$RLSCOPE_DIR/third_party/rlscope_rl-baselines-zoo"
export TF_AGENTS_DIR="$RLSCOPE_DIR/third_party/rlscope_agents"
export REAGENT_DIR="$RLSCOPE_DIR/third_party/rlscope_ReAgent"
export ARTIFACTS_DIR="$RLSCOPE_DIR/output/artifacts"

# NOTE: keep in sync with run_docker.py
export TENSORFLOW_VERSION=${TENSORFLOW_VERSION:-"2.2.0"}

# Defined by run_docker.py
# Use this to determine where to output build files.
# container:
#   $RLSCOPE_DIR/build.docker
# host:
#   $RLSCOPE_DIR/build.host
export RLSCOPE_IS_DOCKER=${RLSCOPE_IS_DOCKER:-no}

unset _ROOT_DIR
unset _SH_DIR
