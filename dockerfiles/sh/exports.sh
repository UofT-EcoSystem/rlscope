#
# Location of third party experiment repo directories (e.g., BASELINES_DIR).
# These variables are used by installation scripts (e.g., setup.sh, install_baselines.sh, etc.)
#
_SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
_ROOT_DIR="$(dirname "$(dirname "$_SH_DIR")")"

export RLSCOPE_DIR=${RLSCOPE_DIR:-$_ROOT_DIR}
export BASELINES_DIR="$RLSCOPE_DIR/third_party/baselines"
export MLPERF_DIR="$RLSCOPE_DIR/third_party/mlperf_training"
export STABLE_BASELINES_DIR="$RLSCOPE_DIR/third_party/stable-baselines"
export RL_BASELINES_ZOO_DIR="$RLSCOPE_DIR/third_party/rl-baselines-zoo"
export TF_AGENTS_DIR="$RLSCOPE_DIR/third_party/agents"
export REAGENT_DIR="$RLSCOPE_DIR/third_party/ReAgent"
export ARTIFACTS_DIR="$RLSCOPE_DIR/output/artifacts"

unset _ROOT_DIR
unset _SH_DIR
