//
// Created by jgleeson on 2020-01-23.
//

//#include "common/debug.h"

#include "experiment/gpu_freq.h"

#include "error_codes.pb.h"

#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/any.hpp>
#include <boost/optional.hpp>

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
#include "common/my_status.h"
#include "analysis/sample_periods.h"

#include <list>
#include <initializer_list>

#include <gflags/gflags.h>
#include <memory>

#include "analysis/trace_file_parser.h"
#include "gpu_util_experiment.h"

DEFINE_bool(debug, false, "Debug: give additional verbose output");
DEFINE_string(iml_directory, "", "Path to --iml-directory used when collecting trace-files");
DEFINE_string(gpu_clock_freq_json, "", "--mode=run_kernels: Path to JSON file containing GPU clock frequency measurements (from --mode=gpu_clock_freq)");
DEFINE_string(mode, "", "One of: [gpu_clock_freq, run_kernels]");

DEFINE_int64(n_launches, 0, "Number of kernels to launch per-thread.");
DEFINE_int64(kernel_delay_us, 0, "Time between kernel launches in microseconds");
DEFINE_int64(kernel_duration_us, 0, "Duration of kernel in microseconds");
DEFINE_double(run_sec, 0, "How to long to run for (in seconds)");
DEFINE_int64(num_threads, 1, "How many threads/processes to launch CUDA kernels from?");
// NOTE: launching in the same process will allow kernel overlap, whereas separate process will not (similar to minigo).
DEFINE_bool(processes, false, "When --num_threads > 1, use separate processes to launch CUDA kernels.  Default behaviour is to use separate threads.");
DEFINE_bool(sync, false, "Wait for kernel to finish after each launch. Useful for running really long kernels (e.g., 10 sec) to avoid creating long queues of kernels accidentally.");
DEFINE_bool(cuda_context, false, "Create new CUDA context for each thread.");
DEFINE_int64(repetitions, 5, "Repetitions when guessing GPU clock frequency");

//using namespace tensorflow;
namespace tensorflow {

// FlagType = int64_t, double
template <typename FlagType>
void AppendCmdArg(std::list<std::string>* cmdline, const std::string& flag_opt, FlagType FLAGS_value, boost::optional<FlagType> overwrite_value)
{
  FlagType value;
  if (overwrite_value.has_value()) {
    value = overwrite_value.value();
  } else {
    value = FLAGS_value;
  }

  std::stringstream flag_opt_ss;
  flag_opt_ss << "--" << flag_opt;
  cmdline->push_back(flag_opt_ss.str());
  std::stringstream value_ss;
  value_ss << value;
  cmdline->push_back(value_ss.str());
}

template <>
void AppendCmdArg(std::list<std::string>* cmdline, const std::string& flag_opt, bool FLAGS_value, boost::optional<bool> overwrite_value) {
  bool value;
  if (overwrite_value.has_value()) {
    value = overwrite_value.value();
  } else {
    value = FLAGS_value;
  }

  if (value) {
    std::stringstream flag_opt_ss;
    flag_opt_ss << "--" << flag_opt;
    cmdline->push_back(flag_opt_ss.str());
  }
}

template <>
void AppendCmdArg(std::list<std::string>* cmdline, const std::string& flag_opt, std::string FLAGS_value, boost::optional<std::string> overwrite_value) {
  std::string value;
  if (overwrite_value.has_value()) {
    value = overwrite_value.value();
  } else {
    value = FLAGS_value;
  }

  std::stringstream flag_opt_ss;
  flag_opt_ss << "--" << flag_opt;
  cmdline->push_back(flag_opt_ss.str());
  cmdline->push_back(value);
}

static std::string BINARY_PATH;

boost::process::child ReinvokeProcess(const GPUUtilExperimentArgs& overwrite_args, boost::process::environment env) {
//  using bp = boost::process;
//  bp::child c(bp::search_path("g++"), "main.cpp");
//  std::stringstream ss;
//  auto cmdline = ss.str();
  std::list<std::string> cmdline;
  cmdline.push_back(BINARY_PATH);

#define APPEND_CMD_ARG(flag_opt, FLAGS_var) \
  AppendCmdArg(&cmdline, flag_opt, FLAGS_var, overwrite_args.FLAGS_var);

  APPEND_CMD_ARG("debug", FLAGS_debug);
  APPEND_CMD_ARG("iml_directory", FLAGS_iml_directory);
  APPEND_CMD_ARG("gpu_clock_freq_json", FLAGS_gpu_clock_freq_json);
  APPEND_CMD_ARG("mode", FLAGS_mode);
  APPEND_CMD_ARG("n_launches", FLAGS_n_launches);
  APPEND_CMD_ARG("kernel_delay_us", FLAGS_kernel_delay_us);
  APPEND_CMD_ARG("kernel_duration_us", FLAGS_kernel_duration_us);
  APPEND_CMD_ARG("run_sec", FLAGS_run_sec);
  APPEND_CMD_ARG("num_threads", FLAGS_num_threads);
  APPEND_CMD_ARG("processes", FLAGS_processes);
  APPEND_CMD_ARG("sync", FLAGS_sync);
  APPEND_CMD_ARG("repetitions", FLAGS_repetitions);
#undef APPEND_CMD_ARG

  std::stringstream cmdline_ss;
  int i = 0;
  for (const auto& arg : cmdline) {
    if (i > 0) {
      cmdline_ss << " ";
    }
    cmdline_ss << arg;
    i += 1;
  }
  auto cmdline_str = cmdline_ss.str();
  DBG_LOG("Reinvoke gpu_util_experiment:\n  $ {}", cmdline_str);

  boost::process::child child(cmdline_str, env);

  return child;
}


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

} // tensorflow
using namespace tensorflow;

int main(int argc, char** argv) {
  BINARY_PATH = argv[0];
  backward::SignalHandling sh;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // NOTE: If you only define SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG, this doesn't enable debug logging.
  // It just ensures that the SPDLOG_DEBUG statements are **compiled in**!
  // We still need to turn them on though!
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

  MyStatus status = MyStatus::OK();

  boost::filesystem::path iml_path(FLAGS_iml_directory);
  if (FLAGS_iml_directory != "" && !boost::filesystem::is_directory(iml_path)) {
    bool success = boost::filesystem::create_directories(iml_path);
    if (!success) {
      std::cout << "ERROR: failed to create --iml_directory = " << iml_path << std::endl;
      exit(EXIT_FAILURE);
    }
//    std::cout << "ERROR: --iml_directory must be a path to a root --iml-directory given when collecting traces" << std::endl;
//    exit(EXIT_FAILURE);
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

  if (mode == Mode::MODE_GPU_CLOCK_FREQ || mode == Mode::MODE_RUN_KERNELS) {
    if (FLAGS_iml_directory == "") {
      std::stringstream ss;
      ss << "--iml-directory is required for --mode=" << FLAGS_mode;
      UsageAndExit(ss.str());
    }
  }

  if (mode == Mode::MODE_RUN_KERNELS) {
    if (FLAGS_kernel_delay_us <= 0) {
      std::stringstream ss;
      ss << "--kernel_delay_us > 0 is required for --mode=" << FLAGS_mode;
      UsageAndExit(ss.str());
    }

    if (FLAGS_n_launches < 0) {
      std::stringstream ss;
      ss << "--n_launches >= 0 is required for --mode=" << FLAGS_mode;
      UsageAndExit(ss.str());
    }

    if (FLAGS_kernel_duration_us <= 0) {
      std::stringstream ss;
      ss << "--kernel_duration_us > 0 is required for --mode=" << FLAGS_mode;
      UsageAndExit(ss.str());
    }

//    if (FLAGS_run_sec <= 0) {
//      std::stringstream ss;
//      ss << "--run_sec > 0 is required for --mode=" << FLAGS_mode;
//      UsageAndExit(ss.str());
//    }

  }

  if (mode == Mode::MODE_GPU_CLOCK_FREQ) {
    GPUClockFreq gpu_clock_freq(FLAGS_repetitions, FLAGS_iml_directory);
    gpu_clock_freq.run();
    status = gpu_clock_freq.dump_json();
    IF_BAD_STATUS_EXIT("Failed to dump json for --mode=gpu_clock_freq", status);
    DBG_LOG("Dumped gpu_clock_freq json @ {}", gpu_clock_freq.json_path());
    exit(EXIT_SUCCESS);
  }

  if (mode == Mode::MODE_RUN_KERNELS) {
    GPUClockFreq gpu_clock_freq(FLAGS_repetitions, FLAGS_iml_directory);
    status = gpu_clock_freq.load_json(FLAGS_gpu_clock_freq_json);
    IF_BAD_STATUS_EXIT("Failed to load json for --mode=gpu_clock_freq", status);
    ThreadedGPUKernelRunner gpu_kernel_runner(
        gpu_clock_freq,
        FLAGS_n_launches,
        FLAGS_kernel_delay_us,
        FLAGS_kernel_duration_us,
        FLAGS_run_sec,
        FLAGS_num_threads,
        FLAGS_processes,
        FLAGS_sync,
        FLAGS_cuda_context,
        FLAGS_iml_directory,
        FLAGS_debug);
    gpu_kernel_runner.run();
    exit(EXIT_SUCCESS);
  }

  // Shouldn't reach here.
  assert(false);

  return 0;
}

