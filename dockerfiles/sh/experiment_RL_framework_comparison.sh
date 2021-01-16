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

  export fig=framework_choice
  # Run TD3 and DDPG across all the different RL frameworks:
  # - tf_agents: TensorFlow eager / autograph
  # - stable_baselines: TensorFlow graph
  # - reagent: PyTorch eager
  bash ./run_bench.sh all_run
  # TD3 plots.
  bash ./run_bench.sh plot_framework_choice
  # DDPG plots.
  bash ./run_bench.sh plot_framework_choice_ddpg

  local output_dir="$ARTIFACTS_DIR/experiment_RL_framework_comparison"
  mkdir -p "$output_dir"
  _do cp $RLSCOPE_DIR/output/plots/framework_choice_ddpg/OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.pdf "$output_dir/fig_ddpg_time_breakdown.pdf"
  _do cp $RLSCOPE_DIR/output/plots/framework_choice_ddpg/CategoryTransitionPlot.combined.pdf "$output_dir/fig_ddpg_transitions.pdf"
  _do cp $RLSCOPE_DIR/output/plots/framework_choice/OverlapStackedBarPlot.overlap_type_CategoryOverlap.operation_training_time.pdf "$output_dir/fig_td3_time_breakdown.pdf"
  _do cp $RLSCOPE_DIR/output/plots/framework_choice/CategoryTransitionPlot.combined.pdf "$output_dir/fig_td3_transitions.pdf"
  log_info
  (
  TXT_BOLD=yes
  log_info "> Success!"
  )
  log_info "  RL framework comparison plots have been output @ $output_dir"

}

main "$@"

