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

  # Compare different simulators to each other (using same RL algorithm: PPO in stable-baselines).
  export fig=simulator_choice;
  bash ./run_bench.sh all_run;
  # Plots of simulator comparison
  bash ./run_bench.sh plot_stable_baselines_fig_9_simulator_choice

  local output_dir="$ARTIFACTS_DIR/experiment_simulator_choice"
  mkdir -p "$output_dir"
  _do cp $RLSCOPE_DIR/output/stable_baselines/calibration.parallel_runs_yes/plots/stable_baselines_fig_9_simulator_choice/OverlapStackedBarPlot.overlap_type_CategoryOverlap.percent.pdf "$output_dir/fig_simulator_choice.pdf"
  log_info
  (
  TXT_BOLD=yes
  log_info "> Success!"
  )
  log_info "  Simulator comparison plots have been output @ $output_dir"

}

main "$@"

