#!/usr/bin/env bash

# Currently we're in ROOT/src/sh.
SH_DIR="$(readlink -f "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )")"
ROOT="$(realpath "$SH_DIR/../..")"
source $ROOT/dockerfiles/sh/exports.sh
cd $ROOT

DEBUG=${DEBUG:-no}
if [ "$DEBUG" = 'yes' ]; then
    set -x
fi

FORCE=${FORCE:-no}

DRY_RUN=${DRY_RUN:-no}
SKIP_BUILD=${SKIP_BUILD:-no}

CLONE=$HOME/clone
MLPERF_MINIGO_DIR=$MLPERF_DIR/reinforcement/tensorflow/minigo
IML=$HOME/clone/dnn_tensorflow_cpp
CHECKPOINTS=$IML/checkpoints/
SEED=1

BASELINES_CHECKPOINTS=$BASELINES_DIR/checkpoints

GYM_DIR=$CLONE/gym

ENJOY_TRT=${ENJOY_TRT:-${RL_BASELINES_ZOO_DIR}/enjoy_trt.py}

_activate_tensorflow() {
    local prev_dir=$PWD
    cd $HOME/clone/tensorflow_cuda9
    source source_me.sh
    activate_tf
    cd $prev_dir
}

_activate_rlscope() {
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
#    ./run_and_time.sh $SEED --rlscope-keep-traces --rlscope-disable 2>&1 | tee --ignore-interrupts ${BASE_DIR}/benchmark.txt
    ./run_and_time.sh $SEED --rlscope-keep-traces 2>&1 | tee --ignore-interrupts ${BASE_DIR}/benchmark.txt
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
    _activate_rlscope
    echo $PYTHONPATH
    export TF_PRINT_TIMESTAMP=yes
    mkdir -p $chkpt_path
    export CUDA_VISIBLE_DEVICES="1"
    python3 $BASELINES_DIR/baselines/deepq/experiments/run_atari.py \
        --env PongNoFrameskip-v4 \
        --rlscope-start-measuring-call 1 \
        --checkpoint-path $chkpt_path \
        --rlscope-num-calls=100 \
        --rlscope-num-traces 2 \
        --rlscope-trace-time-sec 40 \
        --rlscope-python
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
_do_with_logfile() {
  (
  set +x
  set -eu
  # If command we're running fails, propagate it (don't let "tee" mute the error)
  set -o pipefail
  local dry_str=""
  logfile_append=${logfile_append:-no}
  logfile_quiet=${logfile_quiet:-no}
  if [ "${DRY_RUN}" = 'yes' ]; then
    dry_str=" [dry-run]"
  fi
  if [ "${logfile_quiet}" != 'yes' ]; then
    echo "> CMD${dry_str}:"
    echo "  PWD=$PWD"
    echo "  $ $@"
  fi
  local tee_opts=""
  if [ "${logfile_append}" = 'yes' ]; then
    tee_opts="${tee_opts} --append"
  fi
  local ret=
  if [ "${DRY_RUN}" != 'yes' ]; then
    mkdir -p "$(dirname "$logfile")"
    "$@" 2>&1 | tee $tee_opts "$logfile"
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
    num_threads=${num_threads:-60}
    processes='no'
    _multi_expr
  )
}

multiprocess_expr() {
  (
    subdir=multiprocess_expr
    # NOTE: each thread uses 177 MB of memory... cannot use all 68 SM's.
    num_threads=${num_threads:-60}
    processes='yes'
    _multi_expr
  )
}

all_gpu_util_experiment() {
(
  set -eu
  MPS_MAX_PARTITIONS=48
  num_threads=$((MPS_MAX_PARTITIONS-1))
  if is_mps; then
    multiprocess_mps_expr
  else
    multiprocess_expr
    multithread_expr
  fi
)
}

_math() {
  local expression="$1"
  shift 1
  python -c "import math; print($expression)"
}

_gpu_num_sms() {
  # Get the number of streaming multiprocessors (SMs) on device=0
  # Parse this line from deviceQuery:
  #    (68) Multiprocessors, ( 64) CUDA Cores/MP:     4352 CUDA Cores
  deviceQuery | \
    grep Multiprocessors | \
    head -n 1 | \
    perl -lape 's/.*\(\s*(\d+)\s*\) Multiprocessors.*/$1/'
}

multiprocess_mps_expr() {
  (
    subdir=multiprocess_mps_expr
    # NOTE: each thread uses 177 MB of memory... cannot use all 68 SM's.
    # According to MPS documentation, post-Volta cards have a limit of 48 connections to MPS.
    # Q:
    # https://docs.nvidia.com/deploy/mps/index.html#topic_3_3_5_1
    #   The pre-Volta MPS Server supports up to 16 client CUDA contexts per-device
    #   concurrently. Volta MPS server supports 48 client CUDA contexts per-device. These
    #   contexts may be distributed over multiple processes. If the connection limit is
    #   exceeded, the CUDA application will fail to create a CUDA Context and return an API
    #   error from cuCtxCreate() or the first CUDA Runtime API call that triggers context
    #   creation. Failed connection attempts will be logged by the MPS server.
#    MPS_MAX_PARTITIONS=48
#    num_threads=$((MPS_MAX_PARTITIONS-1))
    processes='yes'
    mps='yes'
    # TODO: add and compile deviceQuery so we can query this dynamically for a given GPU.
    NUM_SMS=$(_gpu_num_sms)
    # NUM_SMS=68
    # Q: can CUDA_MPS_ACTIVE_THREAD_PERCENTAGE be a float, or does it have to be an integer?
    # export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(_math "math.ceil(100*${sms_allocated}/${NUM_SMS})")

    # """
    # The limit will be internally rounded up to the next hardware-supported thread
    # count limit. On Volta, the executed limit is reflected through device attribute
    # cudaDevAttrMultiProcessorCount
    # """
    # So, we want to be careful to ensure we don't allocate 2 SMs instead of 1...
    # so, if we need to use integer, we can use math.floor (NOT math.ceil).
#    local mps_percent_1_sms=$(_get_mps_percentage 1)
#    local mps_percent_2_sms=$(_get_mps_percentage 2)
#    if [ "$mps_percent_1_sms" = "$mps_percent_2_sms" ]; then
#      echo "ERROR: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE cannot discern between allocating 1 SM and 2 SMs; use float instead of math.floor([# SMs allocated]/[# SMs on GPU])"
#      return 1
#    fi
    # export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(_math "math.floor(100*${sms_allocated}/${NUM_SMS})")
    # export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(_math "100*${sms_allocated}/${NUM_SMS}")

    # Try to allocate a single SM on the GPU to each process.
    sms_allocated=1
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(_get_mps_percentage 1)
    local sms_allocated=$(_gpu_num_sms)

    suffix=".num_sms_${NUM_SMS}.sms_allocated_${sms_allocated}.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE_${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}"
    _multi_expr
  )
}

_get_mps_percentage() {
  local sms="$1"
  shift 1
  local num_sms=$(_gpu_num_sms)
  # _math "math.floor(100*${sms}/${NUM_SMS})"
  # _math "100*${sms}/${NUM_SMS}"
  # Based on deviceQuery, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE can be a float, and for RTX 2080 which has 68 SMs,
  # SMs are always allocated 2 at a time (2, 4, 6, ..., 68).
  _math "100*${sms}/${num_sms}"
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
  mps=${mps:-no}
  suffix=${suffix:-}
  rlscope_dir="$ROOT/output/gpu_util_experiment/${subdir}/thread_blocks_${thread_blocks}.thread_block_size_${thread_block_size}.n_launches_${n_launches}.iterations_${iterations}.num_threads_${num_threads}.iterations_per_sched_sample_${iterations_per_sched_sample}$(_bool_attr processes $processes)$(_bool_attr hw_counters $hw_counters)$(_bool_attr mps $mps)${suffix}"
  _already_ran() {
    if [ ! -d ${rlscope_dir} ]; then
      return 1
    fi
    find ${rlscope_dir} -type f \
      | grep --perl-regexp '/GPUComputeSchedInfoKernel[^/]*\.json$' >/dev/null
  }
  if _already_ran; then
    echo "> SKIP gpu_util_experiment; already exists @ ${rlscope_dir}"
    return
  fi
  _do mkdir -p ${rlscope_dir}
  logfile=${rlscope_dir}/gpu_util_experiment.log.txt
  _do_with_logfile rls-util-sampler --rlscope-directory ${rlscope_dir} -- \
    gpu_util_experiment \
    --mode run_kernels \
    --rlscope_directory ${rlscope_dir} \
    --gpu_clock_freq_json $RLSCOPE_DIR/calibration/gpu_clock_freq/gpu_clock_freq.json \
    --kernel compute_kernel_sched_info \
    --kern_arg_iterations ${iterations} \
    --kern_arg_threads_per_block ${thread_block_size} \
    --kern_arg_num_blocks ${thread_blocks} \
    --kern_arg_iterations_per_sched_sample ${iterations_per_sched_sample} \
    --num_threads ${num_threads} \
    --n_launches ${n_launches} \
    $(_bool_opt processes $processes) \
    $(_bool_opt hw_counters $hw_counters)
)
}

_make_install() {
  (
  set -ue
  cd $ROOT
  bash setup.sh
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
  rlscope_dir="$ROOT/output/gpu_util_experiment/nvidia_smi/thread_blocks_${thread_blocks}.thread_block_size_${thread_block_size}.n_launches_${n_launches}.iterations_${iterations}.num_threads_${num_threads}$(_bool_attr processes $processes)$(_bool_attr hw_counters $hw_counters)"
  _do mkdir -p ${rlscope_dir}
    #  --kernel_duration_us ${kernel_duration_us}
    #    --kernel_delay_us ${kernel_delay_us}
  _do rls-util-sampler --rlscope-directory ${rlscope_dir} -- \
    gpu_util_experiment \
    --mode run_kernels \
    --rlscope_directory ${rlscope_dir} \
    --gpu_clock_freq_json $RLSCOPE_DIR/calibration/gpu_clock_freq/gpu_clock_freq.json \
    --kernel compute_kernel_sched_info \
    --kern_arg_iterations ${iterations} \
    --kern_arg_threads_per_block ${thread_block_size} \
    --kern_arg_num_blocks ${thread_blocks} \
    --num_threads ${num_threads} \
    --n_launches ${n_launches} \
    $(_bool_opt processes $processes) \
    $(_bool_opt hw_counters $hw_counters) \
    2>&1 | tee ${rlscope_dir}/gpu_util_experiment.log.txt
#  _do rls-analyze --mode gpu_hw --rlscope_directory ${rlscope_dir} | \
#    2>&1 | tee ${rlscope_dir}/rls_analyze.log.txt
)
}

all_trtexec_expr() {
(
  set -ue
  _make_install
  export CUDA_VISIBLE_DEVICES=0

#  BATCH_SIZES=(1 8 16 32 64 128 256 512)
#  STREAMS=(1 2 3 4 5 6 7 8)

  BATCH_SIZES=(1 8 16 32 64 128 256 512)
#  BATCH_SIZES=(1)
  STREAMS=(1)

  echo "> Running trtexec7 experiments for: uff_model_path = ${uff_model_path}"

#  local uff_model_path=$(_uff_model_path)
#  if [ ! -f $uff_model_path ]; then
#    mk_uff_model
#  fi

  for batch_size in "${BATCH_SIZES[@]}"; do
    local engine_path=$(_engine_path)
    if [ ! -f $engine_path ]; then
      build_trt_from_uff
    fi
  done

  threads=no
  for batch_size in "${BATCH_SIZES[@]}"; do
    for streams in "${STREAMS[@]}"; do
      # for threads in yes no; do
      # done
      trtexec_expr
    done
  done
)
}

_tf_model_output_dir() {
  echo $(_trtexec_output_dir)/model
}
_trtexec_output_dir() {
  echo $RLSCOPE_DIR/output/trtexec7
}

#_uff_model_path() {
#  # echo $RL_BASELINES_ZOO_DIR/tf_model.uff
#  echo $RLSCOPE_DIR/output/trtexec7/model/tf_model.uff
#}
_engine_path() {
(
  set -ue
  echo $(_tf_model_output_dir)/$(_uff_model_name).batch_size_${batch_size}.trt
)
}

_tf_inference_root_dir() {
  echo $RLSCOPE_DIR/output/tf_inference
}
_tf_inference_output_dir() {
  (
  set -u
  echo $(_tf_inference_root_dir)/batch_size_${batch_size}$(_bool_attr xla $xla)
  )
}

microbench_simulator_expr() {
(
  set -ue
  PYTHONPATH="${PYTHONPATH:-}"
  export PYTHONPATH="$STABLE_BASELINES_DIR:$RLSCOPE_DIR:$PYTHONPATH"
  export CUDA_VISIBLE_DEVICES=0

  env_id=${env_id:-BreakoutNoFrameskip-v4}
  iterations=5000

  cd $RL_BASELINES_ZOO_DIR
  local out_dir=$RLSCOPE_DIR/output/microbench_simulator/env_id_${env_id};
  local expr_file=${out_dir}/mode_microbench_simulator.json
  if [ -e ${expr_file} ]; then
    echo "> SKIP microbench_simulator_expr; already exists @ ${expr_file}"
    return
  fi
  echo "> RUN: ${expr_file}"
  _do mkdir -p ${out_dir}
  logfile=${out_dir}/log.txt
  _do_with_logfile python ${ENJOY_TRT} \
    --env ${env_id} \
    --folder trained_agents/ \
    --iterations ${iterations}  \
    --directory ${out_dir} \
    --mode microbench_simulator
)
}

all_microbench_simulator_expr() {
(
  set -ue
  _make_install

  ENV_IDS=(
    PongNoFrameskip-v4
    Walker2DBulletEnv-v0
    HopperBulletEnv-v0
    HalfCheetahBulletEnv-v0
    AntBulletEnv-v0
  )

#  ENV_IDS=(PongNoFrameskip-v4)

  for env_id in "${ENV_IDS[@]}"; do
    microbench_inference_expr
  done
)
}


is_mps() {
  # NOTE: This only works for detecting MPS within our docker container.
  [ -e /tmp/nvidia-mps ]
}

microbench_inference_multiprocess_expr() {
(
  set -ue
  PYTHONPATH="${PYTHONPATH:-}"
  export PYTHONPATH="$STABLE_BASELINES_DIR:$RLSCOPE_DIR:$PYTHONPATH"

  algo=${algo:-a2c}
  env_id=${env_id:-BreakoutNoFrameskip-v4}
  num_tasks=${num_tasks:-2}
  batch_size=${batch_size:-1}
  cpu=${cpu:-no}
  sm_alloc_strategy=${sm_alloc_strategy:-evenly}
  graph_def_pb=${graph_def_pb:-}
  # TODO: is there any way to check if we're running with MPS enabled?
  if is_mps; then
    echo "> Running WITH CUDA MPS mode"
    mps=yes
  else
    echo "> Running WITHOUT CUDA MPS"
    mps=no
  fi
  n_warmup_batches=10
  n_measure_batches=100
  # Make it long enough that we can look for nvidia-smi to change to ensure GPU is used.
#  n_measure_batches=10000
  local n_timesteps=$((n_measure_batches + n_warmup_batches))
  subdir=${subdir:-}
  if [ "$subdir" != "" ]; then
    subdir="${subdir}/"
  fi

  if [ "$cpu" = 'yes' ]; then
    # Ensure NO GPUS are visible to tensorflow.
    # Even when doing "with tf.device(CPU)", I see it using the GPU!
    export CUDA_VISIBLE_DEVICES=
  else
    export CUDA_VISIBLE_DEVICES=0
  fi

  local out_dir="$RLSCOPE_DIR/output/microbench_inference_multiprocess/${subdir}batch_size_${batch_size}.num_tasks_${num_tasks}.env_id_${env_id}$(_bool_attr mps $mps)$(_bool_attr cpu $cpu)"

  if [ "$mps" = 'yes' ]; then
    # Divide GPU's resource equally among the parallel inference tasks
    # (i.e., CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = 1/num_tasks
    local physical_num_sms=$(_gpu_num_sms)


    # MPS documentation recommends NOT to divide SMs evenly among the processes.
    # Instead, allocate TWICE as much as dividing evenly.
    #
    # I guess this means an SM won't go underutilized by the one process
    # to which it was statically allocated?
    # But presumably this also means potential for more interference?
    #
    #   https://docs.nvidia.com/deploy/mps/index.html#topic_3_3_5_2
    #   """
    #   A common provisioning strategy is to divide the available threads equally to each
    #   MPS client processes (i.e. 100% / n, for n expected MPS client processes). This strategy
    #   will allocate close to the minimum amount of execution resources, but it could restrict
    #   performance for clients that could occasionally make use of idle resources. A more
    #   optimal strategy is to divide the portion by half of the number of expected clients (i.e.
    #   100% / 0.5n) to give the load balancer more freedom to overlap execution between clients
    #   when there are idle resources.
    #   """
    if [ "${sm_alloc_strategy}" = 'evenly_x2' ]; then
      export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(_math "min(100, 100*(1/$num_tasks)*2)")
    elif [ "${sm_alloc_strategy}" = 'evenly' ]; then
      export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$(_math "100*(1/$num_tasks)")
    else
      echo "ERROR: not sure what sm_alloc_strategy=${sm_alloc_strategy} means"
      return 1
    fi

    # NOTE: deviceQuery tells us how many SMs each process will get now that CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set.
    local sms_allocated=$(_gpu_num_sms)

    out_dir="${out_dir}.num_sms_${physical_num_sms}.sms_allocated_${sms_allocated}.sm_alloc_strategy_${sm_alloc_strategy}.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE_${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}"
  fi


  local arglist=()

  if [ "${graph_def_pb}" != "" ]; then
    echo "> RUNNING WITH graph_def_pb=${graph_def_pb}"
    arglist+=(--graph-def-pb "${graph_def_pb}")
    out_dir="${out_dir}$(_model_attr_from "${graph_def_pb}")"
  fi


  local expr_file=${out_dir}/mode_microbench_inference_multiprocess.merged.json

  if [ -e ${expr_file} ]; then
    echo "> SKIP microbench_inference_multiprocess_expr; already exists @ ${expr_file}"
    return
  fi
  echo "> RUN: ${expr_file}"
#  cd $RL_BASELINES_ZOO_DIR
  _do mkdir -p ${out_dir}
  logfile=${out_dir}/log.txt
  if [ -e $logfile ]; then
    rm $logfile
  fi
  logfile_append=yes
  logfile_quiet=yes
  _do_with_logfile echo "===================================="
  _do_with_logfile echo "> DEVICES:"
  logfile_quiet=no
  # NOTE: deviceQuery exits with non-zero exit code for cpu=yes when no GPUs are visible.
  _do_with_logfile deviceQuery || true
  logfile_quiet=yes
  _do_with_logfile echo "===================================="
  _do_with_logfile echo "> ENVIRONMENT:"
  logfile_quiet=no
  _do_with_logfile env
  logfile_quiet=yes
  _do_with_logfile echo "===================================="
  logfile_quiet=no
  arglist+=(
    --algo ${algo}
    --env ${env_id}
    --folder trained_agents/
    -n ${n_timesteps}
    --directory ${out_dir}
    --mode microbench_inference_multiprocess
    --warmup-iters ${n_warmup_batches}
    --batch-size ${batch_size}
    --num-tasks ${num_tasks}
    $(_bool_opt cpu $cpu)
  )
  _do_with_logfile python ${ENJOY_TRT} "${arglist[@]}"
  logfile_quiet=no
  logfile_append=no
)
}

INFERENCE_BATCH_SIZES=(1 8 16 32 64 128 256 512)
#INFERENCE_BATCH_SIZES=(8)
#INFERENCE_BATCH_SIZES=(256)
#INFERENCE_BATCH_SIZES=(512)
all_microbench_inference_multiprocess() {
(
  set -ue
  # Need to run from minigo, cannot compile trtexec7
#  _make_install

#  NUM_TASKS=(1 2 3 4 5 6 7 8)
#   NUM_TASKS=(1 2)
   NUM_TASKS=(1)
#  NUM_TASKS=(8)
#  NUM_TASKS=(7)

  gpu_only=${gpu_only:-no}

  USE_CPU=()
  if [ "${gpu_only}" = 'yes' ]; then
    echo ">> gpu_only=${gpu_only} : Only run with GPU (e.g., for TensorRT models which otherwise error out)"
    USE_CPU=(no)
  else
    if ! is_mps; then
      USE_CPU=(no yes)
    else
      USE_CPU=(no)
    fi
  fi


#  USE_CPU=(yes)


  for cpu in "${USE_CPU[@]}"; do

    SM_ALLOC_STRATEGY=(evenly)
    if is_mps && [ "$cpu" = 'no' ]; then
      SM_ALLOC_STRATEGY=(evenly evenly_x2)
    fi

    for sm_alloc_strategy in "${SM_ALLOC_STRATEGY[@]}"; do
      for num_tasks in "${NUM_TASKS[@]}"; do
        for batch_size in "${INFERENCE_BATCH_SIZES[@]}"; do

          # WARNING: for some reason, "if ! microbench_inference_multiprocess_expr; ..."
          # acts as is "set -e" doesn't exist, causing things to execute without error detection...
          # workaround: use "set +e"
          (
          set +e
          microbench_inference_multiprocess_expr
          local ret=$?
          set -e
          if [ "$ret" != "0" ]; then
            echo "> SKIP FAILURE: microbench_inference_multiprocess_expr"
            echo "  batch_size=${batch_size}"
            echo "  num_tasks=${num_tasks}"
            echo "  sm_alloc_strategy=${sm_alloc_strategy}"
          fi
          )
        done
      done
    done
  done
)
}

test_microbench_inference_multiprocess() {
  run_proc() {
    num_tasks=1
    batch_size=256
    microbench_inference_multiprocess_expr
  }
  subdir="test_microbench_inference_multiprocess/proc_1"
  run_proc &
  sleep 15
  subdir="test_microbench_inference_multiprocess/proc_2"
  run_proc &
  wait
}

tf_inference_expr() {
(
  set -ue
  PYTHONPATH="${PYTHONPATH:-}"
  export PYTHONPATH="$STABLE_BASELINES_DIR:$RLSCOPE_DIR:$PYTHONPATH"
  export CUDA_VISIBLE_DEVICES=0

  batch_size=${batch_size:-1}
  xla=${xla:-no}

#  cd $RL_BASELINES_ZOO_DIR
  n_warmup_batches=3
  n_measure_batches=20
  local n_timesteps=$((n_measure_batches + n_warmup_batches))
  rlscope_prof_config="gpu-hw"
  local out_dir=$(_tf_inference_output_dir)
  if [ -e ${out_dir}/mode_microbench_inference.json ]; then
    echo "> SKIP tf_inference_expr; already exists @ ${out_dir}/mode_microbench_inference.json"
    return
  fi
  echo "> RUN: ${out_dir}/mode_microbench_inference.json"

  if [ "$xla" = 'yes' ]; then
    # Generate dump of what XLA did to the computational graph.
    # https://www.tensorflow.org/xla#inspect_compiled_programs
    export XLA_FLAGS="--xla_dump_to=${out_dir}/xla_codegen"
    # https://www.tensorflow.org/xla#auto-clustering
    export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
  fi

  _do mkdir -p ${out_dir}
  logfile=${out_dir}/log.txt
  _do_with_logfile rls-prof --config ${rlscope_prof_config} python ${ENJOY_TRT} \
    --algo a2c \
    --env BreakoutNoFrameskip-v4 \
    --folder trained_agents/ \
    -n ${n_timesteps} \
    --rlscope-directory ${out_dir} \
    --rlscope-delay \
    --rlscope-debug \
    --mode microbench_inference \
    --warmup-iters ${n_warmup_batches} \
    --batch-size ${batch_size}

  logfile=${out_dir}/rls_analyze.log.txt
  _do_with_logfile rls-analyze --mode gpu_hw --rlscope_directory ${out_dir}
)
}

all_tf_inference_expr() {
(
  set -ue
  _make_install

  # BATCH_SIZES=(1 8 16 32 64 128 256 512)
  XLA_MODES=(no yes)

#  # BATCH_SIZES=(1 8)
#  BATCH_SIZES=(128)
##  XLA_MODES=(no)
#  XLA_MODES=(yes)

  for xla in "${XLA_MODES[@]}"; do
    for batch_size in "${INFERENCE_BATCH_SIZES[@]}"; do
      tf_inference_expr
    done
  done
)
}

mk_trt_model() {
(
  set -ue
  PYTHONPATH="${PYTHONPATH:-}"
  export PYTHONPATH="$STABLE_BASELINES_DIR:$RLSCOPE_DIR:$PYTHONPATH"
  export CUDA_VISIBLE_DEVICES=0
#  cd $RL_BASELINES_ZOO_DIR

#    --algo a2c
#    --env BreakoutNoFrameskip-v4
#    --folder trained_agents/
#    -n 5000
#    --rlscope-directory output/tensorrt

#  trt_max_batch_size=${trt_max_batch_size:-1}
#  trt_precision=${trt_precision:-fp16}
  graph_def_pb=${graph_def_pb:-}
  saved_model_dir=${saved_model_dir:-}

  local arglist=()

  if [ "${graph_def_pb}" != "" ]; then
    arglist+=(--graph-def-pb "${graph_def_pb}")
    logfile=$(dirname ${graph_def_pb})/$(_model_name ${graph_def_pb}).convert_trt.log.txt
  fi

  if [ "${saved_model_dir}" != "" ]; then
    arglist+=(--saved-model-dir "${saved_model_dir}")
    logfile=${saved_model_dir}/convert_trt.log.txt
  fi

  arglist+=(
    --mode convert_trt
    --trt-max-batch-size ${trt_max_batch_size}
    --trt-precision ${trt_precision}
  )

  _do mkdir -p $(_tf_model_output_dir)
  _do_with_logfile python ${ENJOY_TRT} "${arglist[@]}"
##  _do mv tf_model* $(_tf_model_output_dir) || true
#  if [ "$DRY_RUN" != 'yes' ]; then
##    local uff_model_path=$(_uff_model_path)
#    if [ ! -f $uff_model_path ]; then
#      echo "INTERNAL ERROR: didn't find tensorrt uff file @ $uff_path after running save_tensorrt..."
#      exit 1
#    fi
#  fi
)
}

mk_uff_model() {
(
  set -ue
  PYTHONPATH="${PYTHONPATH:-}"
  export PYTHONPATH="$STABLE_BASELINES_DIR:$RLSCOPE_DIR:$PYTHONPATH"
  export CUDA_VISIBLE_DEVICES=0
#  cd $RL_BASELINES_ZOO_DIR

  _do mkdir -p $(_tf_model_output_dir)
  logfile=$(_tf_model_output_dir)/mk_uff_model.log.txt
  _do_with_logfile python ${ENJOY_TRT} \
    --algo a2c \
    --env BreakoutNoFrameskip-v4 \
    --folder trained_agents/ \
    -n 5000 \
    --rlscope-directory output/tensorrt \
    --mode save_tensorrt
  _do mv tf_model* $(_tf_model_output_dir) || true
  if [ "$DRY_RUN" != 'yes' ]; then
#    local uff_model_path=$(_uff_model_path)
    if [ ! -f $uff_model_path ]; then
      echo "INTERNAL ERROR: didn't find tensorrt uff file @ $uff_path after running save_tensorrt..."
      exit 1
    fi
  fi
)
}


is_stable_baselines_uff_model() {
  basename ${uff_model_path} | grep --quiet --perl-regexp 'tf_model'
}
is_minigo_uff_model() {
  basename ${uff_model_path} | grep --quiet --perl-regexp 'minigo'
}

build_trt_from_uff() {
(
  set -eu
  batch_size=${batch_size:-32}

#  local uff_model_path=$(_uff_model_path)
  if [ "$DRY_RUN" != 'yes' ]; then
    if [ ! -f $uff_model_path ]; then
      echo "ERROR: didn't find tensorrt uff file @ $uff_model_path; build it using mk_uff_model"
      exit 1
    fi
  fi

  # TODO: How does fp16 perform?  That's what NVIDIA uses for minigo apparently.
  _make_install

  logfile=${uff_model_path}.log.txt
  local common_args=(
        --uff=${uff_model_path}
        --batch=${batch_size}
        --saveEngine=${engine_path}
        --buildOnly
        --workspace=512
  )
  _build_trt_stable_baselines() {
      _do_with_logfile trtexec7 \
        "${common_args[@]}" \
        --uffNHWC \
        --uffInput=input/Ob,84,84,4 \
        --output=output/Softmax \
        --int8
  }
  _build_trt_minigo() {
      _do_with_logfile trtexec7 \
        "${common_args[@]}" \
        --uffInput=pos_tensor,13,19,19 \
        --output=policy_output \
        --output=value_output \
        --fp16
  }
  if is_stable_baselines_uff_model; then
      _build_trt_stable_baselines
  elif is_minigo_uff_model; then
      _build_trt_minigo
  else
      echo "ERROR: Not sure how to convert ${uff_model_path} into TensorRT model (need to know its outputs, and input shapes)"
      exit 1
  fi

  if [ "$DRY_RUN" != 'yes' ]; then
    local engine_path=$(_engine_path)
    if [ ! -f $engine_path ]; then
      echo "INTERNAL ERROR: didn't find tensorrt engine file @ $engine_path"
      exit 1
    fi
  fi

)
}

_model_name() {
(
  set -eu
  set -o pipefail
  local model_path="$1"
  shift 1
  basename ${model_path} | perl -lape 's/\.(uff|pb)$//; s/\./-/g;'
  #   | tr -d '\n'
)
}

_uff_model_name() {
(
  set -eu
  set -o pipefail
  _model_name "${uff_model_path}"
)
}

_model_attr() {
  echo ".model_$(_uff_model_name)"
}

_model_attr_from() {
  echo ".model_$(_model_name "$@")"
}

trtexec_expr() {
(
  set -ue

#  threads='yes'
#  streams=2
#  batch_size=32

  threads=${threads:-no}
  cuda_graph=${cuda_graph:-no}
  streams=${streams:-1}
  batch_size=${batch_size:-32}
  subdir=${subdir:-}
  hw_counters=${hw_counters:-yes}
  if [ "$subdir" != "" ]; then
    subdir="${subdir}/"
  fi

  local engine_path=$(_engine_path)
  if [ "$DRY_RUN" != 'yes' ]; then
    if [ ! -f $engine_path ]; then
      echo "ERROR: didn't find tensorrt engine file @ $engine_path; build it using build_trt_from_uff"
      exit 1
    fi
  fi

  rlscope_dir="$(_trtexec_output_dir)/${subdir}batch_size_${batch_size}.streams_${streams}$(_bool_attr threads $threads)$(_bool_attr cuda_graph $cuda_graph)$(_bool_attr hw_counters $hw_counters)$(_model_attr)"

  if [ -d $rlscope_dir ] && [ "$DRY_RUN" = 'no' ]; then
    if [ "$FORCE" != 'yes' ]; then
      echo "> SKIP: $rlscope_dir"
      return
    else
      echo "> FORCE: $rlscope_dir"
    fi
  fi

  _make_install
  export CUDA_VISIBLE_DEVICES=0

  _do mkdir -p ${rlscope_dir}
  logfile=${rlscope_dir}/trtexec.log.txt
  _do_with_logfile trtexec7 \
    --loadEngine=${engine_path} \
    --profile-dir=${rlscope_dir} \
    --batch=${batch_size} \
    --streams=${streams} \
    --exportTimes=${rlscope_dir}/times.json \
    $(_bool_opt threads $threads) \
    $(_bool_opt hw-counters $hw_counters) \
    $(_bool_opt useCudaGraph $cuda_graph)

  _do python $RLSCOPE_DIR/src/libs/trtexec7/tracer.py ${rlscope_dir}/times.json
)
}

cmake_build_dir() {
    local third_party_dir="$1"
    shift 1

    local build_prefix=
    if _is_non_empty RLSCOPE_BUILD_PREFIX; then
      # Docker container environment.
      build_prefix="$RLSCOPE_BUILD_PREFIX"
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
    if _is_non_empty RLSCOPE_INSTALL_PREFIX; then
      # Docker container environment.
      echo "$RLSCOPE_INSTALL_PREFIX"
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
    set -e
    cd $ROOT
    main "$@"
    )
else
    echo "> BASH: Sourcing ${BASH_SOURCE[0]}"
fi
