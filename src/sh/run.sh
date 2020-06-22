#!/usr/bin/env bash

# Currently we're in ROOT/src/sh.
_script_dir="$(realpath "$(dirname "$0")/../..")"
ROOT=$_script_dir
cd $ROOT

DEBUG=${DEBUG:-no}
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

DRY_RUN=${DRY_RUN:-no}

CLONE=$HOME/clone
MLPERF_DIR=$CLONE/mlperf_training
MLPERF_MINIGO_DIR=$MLPERF_DIR/reinforcement/tensorflow/minigo
IML=$HOME/clone/dnn_tensorflow_cpp
CHECKPOINTS=$IML/checkpoints/
SEED=1

BASELINES_DIR=$CLONE/baselines
BASELINES_CHECKPOINTS=$BASELINES_DIR/checkpoints

GYM_DIR=$CLONE/gym

_activate_tensorflow() {
    local prev_dir=$PWD
    cd $HOME/clone/tensorflow_cuda9
    source source_me.sh
    activate_tf
    cd $prev_dir
}

_activate_iml() {
    export PYTHONPATH="$PYTHONPATH:\
$ROOT/python\
"
}

_train_minigo() {
    local base_dir="$1"
    local goparams="$2"
    shift 2

(
    _activate_tensorflow

    # K4000
    export CUDA_VISIBLE_DEVICES="1"

    export BASE_DIR=$base_dir
    export GOPARAMS=$goparams

    mkdir -p $BASE_DIR

    cd $MLPERF_DIR/reinforcement/tensorflow
#    ./run_and_time.sh $SEED --iml-keep-traces --iml-disable 2>&1 | tee --ignore-interrupts ${BASE_DIR}/benchmark.txt
    ./run_and_time.sh $SEED --iml-keep-traces 2>&1 | tee --ignore-interrupts ${BASE_DIR}/benchmark.txt
)
}

train_minigo_test_multiple() {
    local base_dir=$CHECKPOINTS/minigo/vector_test_multiple_workers_k4000
    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/test_multiple_workers.json
    _train_minigo $base_dir $goparams
}

train_minigo_multiple() {
    local base_dir=$CHECKPOINTS/minigo/vector_multiple_workers_k4000
    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/multiple_workers.json
    _train_minigo $base_dir $goparams
}

train_minigo_single() {
    local base_dir=$CHECKPOINTS/minigo/vector_test_single_worker_k4000
    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/test_single_worker.json
    _train_minigo $base_dir $goparams
}

_activate_baselines() {
#    local prev_dir=$PWD
#    cd $BASELINES_DIR
#    source source_me.sh
#    cd $prev_dir

    export PYTHONPATH="$PYTHONPATH:\
$BASELINES_DIR:\
$GYM_DIR\
"
#$HOME/clone/atari-py
# $HOME/clone/Arcade-Learning-Environment

#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:\
#$HOME/.mujoco/mjpro150/bin:\
#/usr/lib/nvidia-384"
}

_train_pong() {
    local chkpt_path="$1"
    shift 1
    (
    _activate_tensorflow
    _activate_baselines
    _activate_iml
    echo $PYTHONPATH
    export TF_PRINT_TIMESTAMP=yes
    mkdir -p $chkpt_path
    export CUDA_VISIBLE_DEVICES="1"
    python3 $BASELINES_DIR/baselines/deepq/experiments/run_atari.py \
        --env PongNoFrameskip-v4 \
        --iml-start-measuring-call 1 \
        --checkpoint-path $chkpt_path \
        --iml-num-calls=100 \
        --iml-num-traces 2 \
        --iml-trace-time-sec 40 \
        --iml-python
    )
}

train_pong() {
#    local chkpt_path=$BASELINES_CHECKPOINTS/PongNoFrameskip-v4/vector_k4000
    local chkpt_path=$BASELINES_CHECKPOINTS/PongNoFrameskip-v4/test_k4000
    _train_pong $chkpt_path
}

#train_minigo_multiple() {
#    local base_dir=$CHECKPOINTS/minigo/vector_test_multiple_workers_k4000
#    local goparams=$MLPERF_DIR/reinforcement/tensorflow/minigo/params/test_multiple_workers.json
#    _train_minigo $base_dir $goparams
#}

_do() {
  (
  set +x
  local dry_str=""
  if [ "${DRY_RUN}" = 'yes' ]; then
    dry_str=" [dry-run]"
  fi
  echo "> CMD${dry_str}:"
  echo "  PWD=$PWD"
  echo "  $ $@"
  if [ "${DRY_RUN}" != 'yes' ]; then
    "$@"
  fi
  )
}
_do_always() {
  (
  set +x
  echo "> CMD:"
  echo "  PWD=$PWD"
  echo "  $ $@"
  "$@"
  )
}

_bool_attr() {
    local opt="$1"
    local yes_or_no="$2"
    shift 2
    echo ".${opt}_${yes_or_no}"
}
_bool_opt() {
    local opt="$1"
    local yes_or_no="$2"
    shift 2
    if [ "$yes_or_no" = 'yes' ]; then
        echo "--${opt}"
    fi
}

multithread_expr() {
  (
    subdir=multithread_expr
    # num_threads=68
    # Want to match multiprocess_expr
    num_threads=60
    processes='no'
    _multi_expr
  )
}

multiprocess_expr() {
  (
    subdir=multiprocess_expr
    # NOTE: each thread uses 177 MB of memory... cannot use all 68 SM's.
    num_threads=60
    processes='yes'
    _multi_expr
  )
}

nvidia_smi_expr() {
  (
    subdir=nvidia_smi_expr
    # NOTE: each thread uses 177 MB of memory... cannot use all 68 SM's.
    num_threads=1
    thread_blocks=1
    thread_block_size=1
    processes='no'
    _multi_expr
  )
}

_multi_expr() {
(
  set -ue
  _make_install
  export CUDA_VISIBLE_DEVICES=0
  # ~ 10 seconds of executing a GPU kernel with a single thread.
  iterations=$((10*1000*1000*1000))
  # Sample sm_id 10 times over the course of the 10 second kernel execution
  # ( I expect the sm_id to remain the same )
  n_sched_samples=10
  iterations_per_sched_sample=$((iterations/n_sched_samples))
  thread_blocks=1
  thread_block_size=1024
  hw_counters='no'
  n_launches=1
  iml_dir="$ROOT/output/gpu_util_experiment/${subdir}/thread_blocks_${thread_blocks}.thread_block_size_${thread_block_size}.n_launches_${n_launches}.iterations_${iterations}.num_threads_${num_threads}.iterations_per_sched_sample_${iterations_per_sched_sample}$(_bool_attr processes $processes)$(_bool_attr hw_counters $hw_counters)"
  _do mkdir -p ${iml_dir}
  _do iml-util-sampler --iml-directory ${iml_dir} -- \
    gpu_util_experiment \
    --mode run_kernels \
    --iml_directory ${iml_dir} \
    --gpu_clock_freq_json $IML_DIR/calibration/gpu_clock_freq/gpu_clock_freq.json \
    --kernel compute_kernel_sched_info \
    --kern_arg_iterations ${iterations} \
    --kern_arg_threads_per_block ${thread_block_size} \
    --kern_arg_num_blocks ${thread_blocks} \
    --kern_arg_iterations_per_sched_sample ${iterations_per_sched_sample} \
    --num_threads ${num_threads} \
    --n_launches ${n_launches} \
    $(_bool_opt processes $processes) \
    $(_bool_opt hw_counters $hw_counters) \
    2>&1 | tee ${iml_dir}/gpu_util_experiment.log.txt
)
}

_make_install() {
  (
  set -ue
  cd "$(cmake_build_dir "$ROOT")"
  _do_always make -j$(nproc) install
  )
}

nvidia_smi_expr() {
(
  set -ue
  _make_install
  export CUDA_VISIBLE_DEVICES=0
  iterations=$((10*1000*1000*1000))
  thread_blocks=1
  thread_block_size=1
  processes='no'
  hw_counters='no'
  num_threads=1
  n_launches=1
  iml_dir="$ROOT/output/gpu_util_experiment/nvidia_smi/thread_blocks_${thread_blocks}.thread_block_size_${thread_block_size}.n_launches_${n_launches}.iterations_${iterations}.num_threads_${num_threads}$(_bool_attr processes $processes)$(_bool_attr hw_counters $hw_counters)"
  _do mkdir -p ${iml_dir}
    #  --kernel_duration_us ${kernel_duration_us}
    #    --kernel_delay_us ${kernel_delay_us}
  _do iml-util-sampler --iml-directory ${iml_dir} -- \
    gpu_util_experiment \
    --mode run_kernels \
    --iml_directory ${iml_dir} \
    --gpu_clock_freq_json $IML_DIR/calibration/gpu_clock_freq/gpu_clock_freq.json \
    --kernel compute_kernel_sched_info \
    --kern_arg_iterations ${iterations} \
    --kern_arg_threads_per_block ${thread_block_size} \
    --kern_arg_num_blocks ${thread_blocks} \
    --num_threads ${num_threads} \
    --n_launches ${n_launches} \
    $(_bool_opt processes $processes) \
    $(_bool_opt hw_counters $hw_counters) \
    2>&1 | tee ${iml_dir}/gpu_util_experiment.log.txt
#  _do rls-analyze --mode gpu_hw --iml_directory ${iml_dir} | \
#    2>&1 | tee ${iml_dir}/rls_analyze.log.txt
)
}

cmake_build_dir() {
    local third_party_dir="$1"
    shift 1

    local build_prefix=
    if _is_non_empty IML_BUILD_PREFIX; then
      # Docker container environment.
      build_prefix="$IML_BUILD_PREFIX"
    else
      # Assume we're running in host environment.
      build_prefix="$ROOT/local.host"
    fi
    echo "$build_prefix/$(basename "$third_party_dir")"
}
_is_non_empty() {
  # Check if an environment variable is defined and not equal to empty string
  (
  set +u
  local varname="$1"
  # Indirect variable dereference that works with both bash and zsh.
  # https://unix.stackexchange.com/questions/68035/foo-and-zsh
  local value=
  eval "value=\"\$${varname}\""
  [ "${value}" != "" ]
  )
}
_local_dir() {
    # When installing things with configure/make-install
    # $ configure --prefix="$(_local_dir)"
    if _is_non_empty IML_INSTALL_PREFIX; then
      # Docker container environment.
      echo "$IML_INSTALL_PREFIX"
    else
      # Assume we're running in host environment.
      echo "$ROOT/local.host"
    fi
}
_add_PATH() {
    local direc="$1"
    shift 1

    echo "> INFO: Add to PATH: $direc"
    export PATH="$direc:$PATH"
}

_add_LD_LIBRARY_PATH() {
  local lib_dir="$1"
  shift 1

  echo "> INFO: Add to LD_LIBRARY_PATH: $lib_dir"
  export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
}

main() {
    _add_LD_LIBRARY_PATH "$(_local_dir)/lib"
    _add_PATH "$(_local_dir)/bin"

    if [ $# -gt 0 ]; then
        _do_always "$@"
        return
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    (
    cd $ROOT
    main "$@"
    )
else
    echo "> BASH: Sourcing ${BASH_SOURCE[0]}"
fi
