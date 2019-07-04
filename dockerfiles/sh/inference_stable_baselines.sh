#!/usr/bin/env bash
# Run inference using a pre-trained RL algorithm using a particular environment.
# Defaults to Atari Pong and DQN.
#
# Usage:
# $ [ENV_ID=PongNoFrameskip-v4] [ALGO=dqn] train_atari_pong.sh
#
# NOTE: This should run inside a docker container.
set -e
if [ "$DEBUG" == 'yes' ]; then
    set -x
fi
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
cd "$SH_DIR"

source $SH_DIR/make_utils.sh

_check_env

_check_STABLE_BASELINES_DIR
_check_RL_BASELINES_ZOO_DIR

_check_tensorflow
_check_iml

setup_display() {
    local display="$1"
    shift 1

    # Wait for the file to come up
    local file="/tmp/.X11-unix/X$display"

    if [ -e "$file" ]; then
        return
    fi

    # Taken from https://github.com/openai/gym/
    # Set up display; otherwise rendering will fail
    sudo Xvfb :1 -screen 0 1024x768x24 &

    sleep 1

    for i in $(seq 1 10); do
        if [ -e "$file" ]; then
             break
        fi

        echo "Waiting for $file to be created (try $i/10)"
        sleep "$i"
    done
    if ! [ -e "$file" ]; then
        echo "Timing out: $file was not created"
        exit 1
    fi
}

py_maybe_install 'stable_baselines' install_stable_baselines.sh

display=1
setup_display $display
export DISPLAY=:$display

if [ "$ENV_ID" = "" ]; then
    ENV_ID="PongNoFrameskip-v4"
fi

if [ "$ALGO" = "" ]; then
    ALGO="dqn"
fi

#OUTPUT_DIR="$RL_BASELINES_ZOO_DIR/retrained_agents"
#OUTPUT_DIR="$RL_BASELINES_ZOO_DIR/output"

#echo "> Inference ALGO = $ALGO, ENV_ID = $ENV_ID, OUTPUT_DIR = $OUTPUT_DIR"
echo "> Inference ALGO = $ALGO, ENV_ID = $ENV_ID"

# NOTE: train.py expects to find hyperparams/*.yml,
# so we must cd into $RL_BASELINES_ZOO_DIR
cd $RL_BASELINES_ZOO_DIR

# For real results:
#    --iml-trace-time-sec $((2*60))
# For quick debugging:
#    --iml-trace-time-sec $((40))

#if [ "$DEBUG" == 'yes' ]; then
#    IML_TRACE_TIME_SEC=$((40))
#else
#    IML_TRACE_TIME_SEC=$((2*60))
#fi

# Debug simulator time...
IML_TRACE_TIME_SEC=$((10))

#if [ "$DEBUG" == 'yes' ]; then
#    PYTHON=(python -m ipdb)
#else
#    PYTHON=(python)
#fi
PYTHON=(python)

#    --folder $OUTPUT_DIR
_do "${PYTHON[@]}" $RL_BASELINES_ZOO_DIR/enjoy.py \
    --algo $ALGO \
    --env $ENV_ID \
    --iml-start-measuring-call 1 \
    --iml-trace-time-sec $IML_TRACE_TIME_SEC \
    --no-render \
    "$@"
