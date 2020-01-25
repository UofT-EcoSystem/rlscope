//
// Created by jgleeson on 2020-01-23.
//

//#include "common/debug.h"

#include "experiment/gpu_freq.h"

#include "error_codes.pb.h"

#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <boost/any.hpp>

#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/debug_flags.h"

// Time breakdown:
// - metric: how many events are processed per second by compute overlap.
// - loading data from proto files
// - running overlap computation

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <backward.hpp>

#include <iostream>

#include <assert.h>

//#include "tensorflow/core/lib/core/status.h"
#include "analysis/my_status.h"
#include "analysis/sample_periods.h"

#include <list>
#include <initializer_list>

#include <gflags/gflags.h>
#include <memory>

#include "analysis/trace_file_parser.h"

DEFINE_bool(debug, false, "Debug: give additional verbose output");
DEFINE_string(iml_directory, "", "Path to --iml-directory used when collecting trace-files");
DEFINE_string(gpu_clock_freq_json, "", "--mode=run_kernels: Path to JSON file containing GPU clock frequency measurements (from --mode=gpu_clock_freq)");
DEFINE_string(mode, "", "One of: [gpu_clock_freq, run_kernels]");

DEFINE_int64(kernel_delay_us, 0, "Time between kernel launches in microseconds");
DEFINE_int64(kernel_duration_us, 0, "Duration of kernel in microseconds");
DEFINE_int64(repetitions, 5, "Repetitions when guessing GPU clock frequency");

using namespace tensorflow;

#define IF_BAD_STATUS_EXIT(msg, status)  \
      if (status.code() != MyStatus::OK().code()) { \
        std::cout << "ERROR: " << msg << ": " << status.ToString() << std::endl; \
        exit(EXIT_FAILURE); \
      }

enum Mode {
  MODE_UNKNOWN = 0,
  MODE_GPU_CLOCK_FREQ = 1,
  MODE_RUN_KERNELS = 2,
};
const std::set<Mode> VALID_MODES = {MODE_GPU_CLOCK_FREQ, MODE_RUN_KERNELS};

Mode StringToMode(const std::string& mode_str) {
  if (mode_str == "gpu_clock_freq") {
    return MODE_GPU_CLOCK_FREQ;
  } else if (mode_str == "run_kernels") {
    return MODE_RUN_KERNELS;
  } else {
    return MODE_UNKNOWN;
  }
}
const char* ModeToString(Mode mode) {
  switch (mode) {
    case MODE_UNKNOWN:
      return "MODE_UNKNOWN";
    case MODE_GPU_CLOCK_FREQ:
      return "gpu_clock_freq";
    case MODE_RUN_KERNELS:
      return "run_kernels";
  }
  assert(false);
  return "";
}

void Usage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "  # Measure GPU clock frequency, so we know how to busy wait on the GPU for 5 us in terms of GPU-cycles." << std::endl;
  std::cout << "  $ gpu_util_experiment --mode gpu_clock_freq" << std::endl;
  std::cout << "  # Use GPU clock frequency measurement to launch \"sleep\" kernels that busy wait for 5 us." << std::endl;
  std::cout << "  $ gpu_util_experiment --mode run_kernels --kernel_duration_us 5 --kernel_delay_us 5" << std::endl;
  std::cout << std::endl;
}
void UsageAndExit(const std::string& msg) {
  Usage();
  std::cout << "ERROR: " << msg << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  backward::SignalHandling sh;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // NOTE: If you only define SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG, this doesn't enable debug logging.
  // It just ensures that the SPDLOG_DEBUG statements are **compiled in**!
  // We still need to turn them on though!
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

  MyStatus status = MyStatus::OK();

  boost::filesystem::path iml_path(FLAGS_iml_directory);
  if (FLAGS_iml_directory != "" && !boost::filesystem::is_directory(iml_path)) {
    std::cout << "ERROR: --iml_directory must be a path to a root --iml-directory given when collecting traces" << std::endl;
    exit(EXIT_FAILURE);
  }

  Mode mode = StringToMode(FLAGS_mode);
  if (mode == MODE_UNKNOWN) {
    std::set<std::string> mode_strings;
    for (auto valid_mode : VALID_MODES) {
      mode_strings.insert(ModeToString(valid_mode));
    }
    std::stringstream ss;
    ss << "--mode must be one of ";
    PrintValue(ss, mode_strings);
    UsageAndExit(ss.str());
  }

  if (mode == Mode::MODE_GPU_CLOCK_FREQ) {
    if (FLAGS_iml_directory == "") {
      std::stringstream ss;
      ss << "--iml-directory is required for --mode=" << FLAGS_mode;
      UsageAndExit(ss.str());
    }
  }

  if (mode == Mode::MODE_GPU_CLOCK_FREQ) {
    GPUClockFreq gpu_clock_freq(FLAGS_repetitions, FLAGS_iml_directory);
    gpu_clock_freq.run();
    status = gpu_clock_freq.dump_json();
    IF_BAD_STATUS_EXIT("Failed to dump json for --mode=gpu_clock_freq", status);
    DBG_LOG("Dumped gpu_clock_freq json @ {}", gpu_clock_freq.json_path());
    exit(EXIT_SUCCESS);
  }

  // Shouldn't reach here.
  assert(false);

  return 0;
}

