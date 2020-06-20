//
// Created by jgleeson on 2020-01-23.
//

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <boost/any.hpp>
#include <boost/optional.hpp>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

#include <backward.hpp>

#include <gflags/gflags.h>

#include <iostream>
#include <assert.h>
#include <list>
#include <initializer_list>
#include <memory>

#include "range_sampling.h"
#include "common_util.h"

#include "gpu_util_experiment.h"
#include "gpu_freq.h"
#include "gpu_freq.cuh"

static const std::vector<std::string> DEFAULT_METRICS = rlscope::get_DEFAULT_METRICS();
static const std::string DEFAULT_METRICS_STR = rlscope::get_DEFAULT_METRICS_STR();

DEFINE_bool(debug, false, "Debug: give additional verbose output");
DEFINE_string(iml_directory, "", "Path to --iml-directory used when collecting trace-files");
DEFINE_string(gpu_clock_freq_json, "",
              "--mode=run_kernels: Path to JSON file containing GPU clock frequency measurements (from --mode=gpu_clock_freq)");
DEFINE_string(mode, "", "One of: [gpu_clock_freq, run_kernels]");
DEFINE_bool(hw_counters, false,
            "Use CUDA Profiling API to collect \"GPU occupancy\" related GPU hardware performance counters");
DEFINE_int64(samples, 1, "How many times to run the experiment in a row; useful for hw_counters");
//DEFINE_string(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active+,smsp__inst_executed.sum+,sm__cycles_active.sum+,sm__warps_active.sum+,sm__cycles_elapsed.sum+", "Comma-delimited list of CUDA Profiling API metrics to collect.");
DEFINE_string(metrics, DEFAULT_METRICS_STR.c_str(), "Comma-delimited list of CUDA Profiling API metrics to collect.");

DEFINE_int64(n_launches, 1, "Number of kernels to launch per-thread.");
DEFINE_int64(kernel_delay_us, 0, "Time between kernel launches in microseconds");
DEFINE_int64(kernel_duration_us, 0, "Duration of kernel in microseconds");
DEFINE_double(run_sec, 0, "How to long to run for (in seconds)");
DEFINE_int64(num_threads, 1, "How many threads/processes to launch CUDA kernels from?");
// NOTE: launching in the same process will allow kernel overlap, whereas separate process will not (similar to minigo).
DEFINE_bool(processes, false,
            "When --num_threads > 1, use separate processes to launch CUDA kernels.  Default behaviour is to use separate threads.");
DEFINE_bool(sync, false,
            "Wait for kernel to finish after each launch. Useful for running really long kernels (e.g., 10 sec) to avoid creating long queues of kernels accidentally.");
DEFINE_bool(cuda_context, false, "Create new CUDA context for each thread.");
DEFINE_int64(repetitions, 5, "Repetitions when guessing GPU clock frequency");
DEFINE_int32(device, 0, "zero-based CUDA device id");

DEFINE_bool(internal_is_child, false,
            "(Internal) this process is a child of some parent instance of gpu_util_experiment => open existing shared memory (don't create)");
DEFINE_int64(internal_thread_id, -1, "(Internal) 0-based thread-id for child");

DEFINE_string(kernel, "compute_kernel", "What GPU kernel should we run?");
// URL: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
DEFINE_string(gpu_sched_policy, "default",
              "What GPU scheduling policy to use for the CUDA context (see: CUDA Driver API documentation for cuCtxCreate for details)?");
DEFINE_int64(kern_arg_iterations, 1000 * 1000 * 1000,
             "(Kernel arg) compute_kernel: how many loop iterations to perform (increase compute); default = 1 million, roughly 1 second on a RTX 2080");
DEFINE_int64(kern_arg_num_blocks, 1,
             "(Kernel arg) compute_sched_info_kernel: how many GPU thread blocks to run (increase parallelism)?");
DEFINE_int64(kern_arg_threads_per_block, 1,
             "(Kernel arg) compute_kernel: how many threads for each GPU thread block (increase parallelism)?");
DEFINE_int64(kern_arg_iterations_per_sched_sample, -1,
             "(Kernel arg) compute_sched_info_kernel: how many iterations of execution before sample GPU scheduling info (sm_id/warp_id/lane_id)? Default: just take a single sample at the start of the kernel (== --kern_arg_iterations)");
// TODO: add grid size and thread block size args.


//using namespace rlscope;
namespace rlscope {

static MyStatus GetCudaContextFlags(unsigned int *flags) {
    *flags = 0;
    MyStatus status;
    if (FLAGS_gpu_sched_policy == "default") {
        *flags |= CU_CTX_SCHED_AUTO;
    } else if (FLAGS_gpu_sched_policy == "spin") {
        *flags |= CU_CTX_SCHED_SPIN;
    } else if (FLAGS_gpu_sched_policy == "block") {
        *flags |= CU_CTX_SCHED_BLOCKING_SYNC;
    } else if (FLAGS_gpu_sched_policy == "yield") {
        *flags |= CU_CTX_SCHED_YIELD;
    } else {
        std::stringstream ss;
        ss << "Not sure what GPU scheduling policy to use for --gpu_sched_policy=" << FLAGS_gpu_sched_policy
           << "; choices are: ";
        PrintValue(ss, std::vector{"default", "spin", "block", "yield"});
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
    return MyStatus::OK();
}

// FlagType = int64_t, double
template<typename FlagType>
void AppendCmdArg(std::list<std::string> *cmdline, const std::string &flag_opt, boost::optional<FlagType> arg) {
    if (!arg.has_value()) {
        return;
    }
    FlagType value = arg.value();

    std::stringstream flag_opt_ss;
    flag_opt_ss << "--" << flag_opt;
    cmdline->push_back(flag_opt_ss.str());
    std::stringstream value_ss;
    value_ss << value;
    cmdline->push_back(value_ss.str());
}

template<>
void AppendCmdArg(std::list<std::string> *cmdline, const std::string &flag_opt, boost::optional<bool> arg) {
    if (!arg.has_value()) {
        return;
    }
    bool value = arg.value();

    if (value) {
        std::stringstream flag_opt_ss;
        flag_opt_ss << "--" << flag_opt;
        cmdline->push_back(flag_opt_ss.str());
    }
}

template<>
void AppendCmdArg(std::list<std::string> *cmdline, const std::string &flag_opt, boost::optional<std::string> arg) {
    if (!arg.has_value()) {
        return;
    }
    std::string value = arg.value();

    std::stringstream flag_opt_ss;
    flag_opt_ss << "--" << flag_opt;
    cmdline->push_back(flag_opt_ss.str());
    cmdline->push_back(value);
}

static std::string BINARY_PATH;

/* static */ GPUUtilExperimentArgs GPUUtilExperimentArgs::FromFlags() {
    GPUUtilExperimentArgs args;
#define SET_FLAG(FLAGS_var) \
  args.FLAGS_var = ::FLAGS_var;

    auto env = boost::this_process::environment();
#define SET_ENV(env_var) \
  if (env.find(#env_var) != env.end()) { \
    args.env_var = env[#env_var].to_string(); \
  }

//  if (env.find("IML_PROCESS_NAME") != env.end()) {
//    args.IML_PROCESS_NAME = env["IML_PROCESS_NAME"].to_string();
//  }

    SET_ENV(IML_PROCESS_NAME);

//  args.FLAGS_debug = ::FLAGS_debug;

    SET_FLAG(FLAGS_debug);
    SET_FLAG(FLAGS_iml_directory);
    SET_FLAG(FLAGS_gpu_clock_freq_json);
    SET_FLAG(FLAGS_mode);
    SET_FLAG(FLAGS_hw_counters);
    SET_FLAG(FLAGS_n_launches);
    SET_FLAG(FLAGS_kernel_delay_us);
    SET_FLAG(FLAGS_kernel_duration_us);
    SET_FLAG(FLAGS_run_sec);
    SET_FLAG(FLAGS_num_threads);
    SET_FLAG(FLAGS_processes);
    SET_FLAG(FLAGS_sync);
    SET_FLAG(FLAGS_cuda_context);
    SET_FLAG(FLAGS_repetitions);
    SET_FLAG(FLAGS_samples);
    SET_FLAG(FLAGS_device);
    SET_FLAG(FLAGS_internal_is_child);
    if (::FLAGS_internal_thread_id != -1) {
        SET_FLAG(FLAGS_internal_thread_id);
    }

    std::vector<std::string> metrics = rlscope::StringSplit(::FLAGS_metrics, ",");
    if (metrics.size()) {
        args.FLAGS_metrics = metrics;
    }

    SET_FLAG(FLAGS_kern_arg_iterations);
    SET_FLAG(FLAGS_kern_arg_num_blocks);
    SET_FLAG(FLAGS_kern_arg_threads_per_block);
    SET_FLAG(FLAGS_kern_arg_iterations_per_sched_sample);
    SET_FLAG(FLAGS_kernel);
    SET_FLAG(FLAGS_gpu_sched_policy);
#undef SET_FLAG

    return args;
}

boost::process::child ReinvokeProcess(const GPUUtilExperimentArgs &overwrite_args, boost::process::environment env) {
//  using bp = boost::process;
//  bp::child c(bp::search_path("g++"), "main.cpp");
//  std::stringstream ss;
//  auto cmdline = ss.str();
    std::list<std::string> cmdline;
    cmdline.push_back(BINARY_PATH);

#define APPEND_CMD_ARG(flag_opt, FLAGS_var) \
  AppendCmdArg(&cmdline, flag_opt, overwrite_args.FLAGS_var);

    APPEND_CMD_ARG("debug", FLAGS_debug);
    APPEND_CMD_ARG("iml_directory", FLAGS_iml_directory);
    APPEND_CMD_ARG("gpu_clock_freq_json", FLAGS_gpu_clock_freq_json);
    APPEND_CMD_ARG("mode", FLAGS_mode);
    APPEND_CMD_ARG("hw_counters", FLAGS_hw_counters);
    APPEND_CMD_ARG("n_launches", FLAGS_n_launches);
    APPEND_CMD_ARG("kernel_delay_us", FLAGS_kernel_delay_us);
    APPEND_CMD_ARG("kernel_duration_us", FLAGS_kernel_duration_us);
    APPEND_CMD_ARG("run_sec", FLAGS_run_sec);
    APPEND_CMD_ARG("num_threads", FLAGS_num_threads);
    APPEND_CMD_ARG("processes", FLAGS_processes);
    APPEND_CMD_ARG("sync", FLAGS_sync);
    APPEND_CMD_ARG("cuda_context", FLAGS_cuda_context);
    APPEND_CMD_ARG("repetitions", FLAGS_repetitions);
    APPEND_CMD_ARG("samples", FLAGS_samples);
    APPEND_CMD_ARG("device", FLAGS_device);
    APPEND_CMD_ARG("internal_is_child", FLAGS_internal_is_child);
    APPEND_CMD_ARG("internal_thread_id", FLAGS_internal_thread_id);
    APPEND_CMD_ARG("kern_arg_iterations", FLAGS_kern_arg_iterations);
    APPEND_CMD_ARG("kern_arg_num_blocks", FLAGS_kern_arg_num_blocks);
    APPEND_CMD_ARG("kern_arg_threads_per_block", FLAGS_kern_arg_threads_per_block);
    APPEND_CMD_ARG("kern_arg_iterations_per_sched_sample", FLAGS_kern_arg_iterations_per_sched_sample);
    APPEND_CMD_ARG("kernel", FLAGS_kernel);
    APPEND_CMD_ARG("gpu_sched_policy", FLAGS_gpu_sched_policy);
#undef APPEND_CMD_ARG

    std::stringstream cmdline_ss;
    int i = 0;
    for (const auto &arg : cmdline) {
        if (i > 0) {
            cmdline_ss << " ";
        }
        cmdline_ss << arg;
        i += 1;
    }
    auto cmdline_str = cmdline_ss.str();
    RLS_LOG("GPU_UTIL", "Reinvoke gpu_util_experiment:\n  $ {}", cmdline_str);

    boost::process::child child(cmdline_str, env);

    return child;
}


enum Mode {
    MODE_UNKNOWN = 0,
    MODE_GPU_CLOCK_FREQ = 1,
    MODE_RUN_KERNELS = 2,
};
const std::set<Mode> VALID_MODES = {MODE_GPU_CLOCK_FREQ, MODE_RUN_KERNELS};

Mode StringToMode(const std::string &mode_str) {
    if (mode_str == "gpu_clock_freq") {
        return MODE_GPU_CLOCK_FREQ;
    } else if (mode_str == "run_kernels") {
        return MODE_RUN_KERNELS;
    } else {
        return MODE_UNKNOWN;
    }
}

const char *ModeToString(Mode mode) {
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
    std::cout
            << "  # Measure GPU clock frequency, so we know how to busy wait on the GPU for 5 us in terms of GPU-cycles."
            << std::endl;
    std::cout << "  $ gpu_util_experiment --mode gpu_clock_freq" << std::endl;
    std::cout << "  # Use GPU clock frequency measurement to launch \"sleep\" kernels that busy wait for 5 us."
              << std::endl;
    std::cout << "  $ gpu_util_experiment --mode run_kernels --kernel_duration_us 5 --kernel_delay_us 5" << std::endl;
    std::cout << std::endl;
}

void UsageAndExit(const std::string &msg) {
    Usage();
    std::cout << "ERROR: " << msg << std::endl;
    exit(EXIT_FAILURE);
}

} // tensorflow
using namespace rlscope;

int main(int argc, char **argv) {
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
//        if (FLAGS_kernel_delay_us <= 0) {
//            std::stringstream ss;
//            ss << "--kernel_delay_us > 0 is required for --mode=" << FLAGS_mode;
//            UsageAndExit(ss.str());
//        }

        if (FLAGS_n_launches < 0) {
            std::stringstream ss;
            ss << "--n_launches >= 0 is required for --mode=" << FLAGS_mode;
            UsageAndExit(ss.str());
        }

//        if (FLAGS_kernel_duration_us <= 0) {
//            std::stringstream ss;
//            ss << "--kernel_duration_us > 0 is required for --mode=" << FLAGS_mode;
//            UsageAndExit(ss.str());
//        }

//    if (FLAGS_run_sec <= 0) {
//      std::stringstream ss;
//      ss << "--run_sec > 0 is required for --mode=" << FLAGS_mode;
//      UsageAndExit(ss.str());
//    }

    }

    if (FLAGS_kern_arg_iterations_per_sched_sample == -1) {
        // By default, just take a single sample at the start of the kernel
        FLAGS_kern_arg_iterations_per_sched_sample = FLAGS_kern_arg_iterations;
    }

    auto args = GPUUtilExperimentArgs::FromFlags();
    if (SHOULD_DEBUG(FEATURE_GPU_CLOCK_FREQ)
        || SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)
        || SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)
        || FLAGS_debug) {
        RLS_LOG("GPU_UTIL", "{}", args);
    }


    if (args.FLAGS_hw_counters.get() && args.FLAGS_metrics.has_value() && args.FLAGS_metrics.get().size() == 0) {
        std::cerr << "ERROR: --metrics must be a comma-delimited list with at least one CUPTI Profiling API metric" << std::endl;
        exit(EXIT_FAILURE);
    }

    // NOTE: This is required.  I think we can assume a CUDA
    // code-base will make these calls for us before sampler is used.
    DRIVER_API_CALL_MAYBE_EXIT(cuInit(0));

    cudaError_t cuda_err;
    int num_devices;
    RUNTIME_API_CALL_MAYBE_EXIT(cudaGetDeviceCount(&num_devices));
    if (args.FLAGS_device.get() >= num_devices) {
        std::cout << "ERROR: --device must be in [0.." << num_devices << "]" << std::endl;
    }

    unsigned int cuda_context_flags;
    status = GetCudaContextFlags(&cuda_context_flags);
    IF_BAD_STATUS_EXIT("Failed to parse cmdline args", status);
    args.FLAGS_cuda_context_flags = cuda_context_flags;

    if (mode == Mode::MODE_GPU_CLOCK_FREQ) {
        GPUClockFreq gpu_clock_freq(args);
        gpu_clock_freq.run();
        status = gpu_clock_freq.dump_json();
        IF_BAD_STATUS_EXIT("Failed to dump json for --mode=gpu_clock_freq", status);
        RLS_LOG("GPU_UTIL", "Dumped gpu_clock_freq json @ {}", gpu_clock_freq.json_path());
        exit(EXIT_SUCCESS);
    }

    if (mode == Mode::MODE_RUN_KERNELS) {
        std::unique_ptr<GPUKernel> gpu_kernel;
        status = GetGPUKernel(args, &gpu_kernel);
        IF_BAD_STATUS_EXIT("Failed to setup --kernel", status);

        status = gpu_kernel->CheckArgs();
        IF_BAD_STATUS_EXIT("Kernel arguments looked incorrect", status);

//    GPUClockFreq gpu_clock_freq(args);
//    status = gpu_clock_freq.load_json(FLAGS_gpu_clock_freq_json);
//    IF_BAD_STATUS_EXIT("Failed to load json for --mode=gpu_clock_freq", status);
        ThreadedGPUKernelRunner gpu_kernel_runner(
                std::move(gpu_kernel),
                args);
        gpu_kernel_runner.run();
        exit(EXIT_SUCCESS);
    }

    // Shouldn't reach here.
    assert(false);

    return 0;
}

