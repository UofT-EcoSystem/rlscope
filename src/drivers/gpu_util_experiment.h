//
// Created by jgleeson on 2020-04-13.
//

#ifndef IML_GPU_UTIL_EXPERIMENT_H
#define IML_GPU_UTIL_EXPERIMENT_H

#include <boost/optional.hpp>
#include <boost/process.hpp>

namespace tensorflow {

struct GPUUtilExperimentArgs {
  // Environment variables
  boost::optional<std::string> IML_PROCESS_NAME;

  // Flags
  boost::optional<bool> FLAGS_debug;
  boost::optional<std::string> FLAGS_iml_directory;
  boost::optional<std::string> FLAGS_gpu_clock_freq_json;
  boost::optional<std::string> FLAGS_mode;

  boost::optional<int64_t> FLAGS_n_launches;
  boost::optional<int64_t> FLAGS_kernel_delay_us;
  boost::optional<int64_t> FLAGS_kernel_duration_us;
  boost::optional<double> FLAGS_run_sec;
  boost::optional<int64_t> FLAGS_num_threads;
  boost::optional<bool> FLAGS_processes;
  boost::optional<bool> FLAGS_sync;
  boost::optional<int64_t> FLAGS_repetitions;
};


boost::process::child ReinvokeProcess(const GPUUtilExperimentArgs& overwrite_args, boost::process::environment env);

}

#endif //IML_GPU_UTIL_EXPERIMENT_H
