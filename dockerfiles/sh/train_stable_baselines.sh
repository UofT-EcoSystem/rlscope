#!/usr/bin/env bash
# Train an RL algorithm using a particular environment.
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

# TODO: Make this runnable on Ubuntu 16.04/18.04 from outside of docker-env by removing dependency on env-variables being set.

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
OUTPUT_DIR="$RL_BASELINES_ZOO_DIR/output"

echo "> Training ALGO = $ALGO, ENV_ID = $ENV_ID, OUTPUT_DIR = $OUTPUT_DIR"

# NOTE: train.py expects to find hyperparams/*.yml,
# so we must cd into $RL_BASELINES_ZOO_DIR
cd $RL_BASELINES_ZOO_DIR

# For real results:
#    --iml-trace-time-sec $((2*60))
# For quick debugging:
#    --iml-trace-time-sec $((40))

#if [ "$DEBUG" == 'yes' ]; then
##    IML_TRACE_TIME_SEC=$((40))
#    IML_TRACE_TIME_SEC=$((20))
#elif [ "$IML_TRACE_TIME_SEC" == "" ]; then
#    IML_TRACE_TIME_SEC=$((2*60))
#fi
#    --iml-trace-time-sec $IML_TRACE_TIME_SEC \

#if [ "$DEBUG" == 'yes' ]; then
#    PYTHON=(python -m ipdb)
#else
#    PYTHON=(python)
#fi
PYTHON=(python)

NVPROF=()
#if [ "$IML_USE_NVPROF" == "yes" ]; then
#    NVPROF=(nvprof --source-level-analysis)
#fi


# NOTE: CUDA_API_PROF, if defined, will be a command (iml-prof) that wraps the training script
# using LD_PRELOAD=libsample_cuda_api.so in order to sample CUDA API calls during
# training.  This should be used for uninstrumented runs.
# Instrumented runs should instead use the profiler built-in to tensorflow to avoid libcupti
# callback registration conflicts.

_do $CUDA_API_PROF "${NVPROF[@]}" "${PYTHON[@]}" $RL_BASELINES_ZOO_DIR/train.py \
    --algo $ALGO \
    --env $ENV_ID \
    --log-folder $OUTPUT_DIR \
    --log-interval 1 \
    --iml-start-measuring-call 1 \
    --iml-delay \
    "$@"
# Sanity check that train.py has exited,
# for those weird runs where iml-bench hangs unexpectedly during p.stdout.readline().
echo "> train.py has exited"
