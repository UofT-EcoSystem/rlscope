//
// Created by jgleeson on 2020-01-23.
//

#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime.h>
#include <nvToolsExtCudaRt.h>

#include "range_sampling.h"

#include <chrono>
#include <iostream>
#include <cmath>
#include <thread>
#include <memory>

#include <cassert>
#include <sstream>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

#include <pthread.h>

#include <backward.hpp>

#include "common_util.h"

#include <sys/syscall.h>

#include "gpu_freq.h"
#include "gpu_freq.cuh"
#include "gpu_util_experiment.h"

static pid_t my_gettid() {
    return syscall(SYS_gettid);
}

// Used by ActivityBuffer and DeviceTracerImpl
#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = call;                                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      const char *errstr;                                           \
      cuptiGetResultString(_status, &errstr);                       \
      RLS_LOG("GPU_UTIL", "ERROR: libcupti call {} failed with {}", #call, errstr); \
      exit(EXIT_FAILURE); \
    }                                                               \
  } while (0)

namespace rlscope {

using clock_value_t = long long;

using steady_clock = std::chrono::steady_clock;

MyStatus GPUComputeKernel::Init() {
    // MyStatus status;
    this->run_ctx.reset(new RunCtx(1));
    return MyStatus::OK();
}

MyStatus GPUComputeKernel::CheckArgs() {
    return MyStatus::OK();
}

std::unique_ptr<GPUKernel> GPUComputeKernel::clone() const {
    std::unique_ptr<GPUComputeKernel> obj(new GPUComputeKernel(this->args));
    if (this->run_ctx) {
        obj->run_ctx.reset(new RunCtx(this->run_ctx->output.num_elems()));
    }
    return obj;
    // std::unique_ptr<RunCtx> doesn't allow copy constructor...
//  return std::make_unique<GPUComputeKernel>(*this);  // requires C++ 14
}

void GPUComputeKernel::RunSync(CudaStream stream) {
    _gpu_compute_kernel(stream, /*sync=*/true);
}

void GPUComputeKernel::RunAsync(CudaStream stream) {
    _gpu_compute_kernel(stream, /*sync=*/false);
}

MyStatus GPUComputeKernel::DumpKernelInfo(int thread_id, CudaStream stream) {
    return MyStatus::OK();
}


MyStatus GPUComputeSchedInfoKernel::Init() {
    CUPTI_CALL(cuptiGetTimestamp(&gpu_base_timestamp_ns));

    CUcontext context;
    DRIVER_API_CALL_MAYBE_EXIT(cuCtxGetCurrent(&context));
    assert(context != nullptr);
    CUPTI_CALL(cuptiDeviceGetTimestamp(context, &device_base_timestamp_ns));
    cpu_base_timestamp_us = get_timestamp_us();

    RUNTIME_API_CALL_MAYBE_EXIT(cudaGetDeviceProperties(&device_prop, args.FLAGS_device.get()));

    this->run_ctx.reset(new RunCtx(
            args,
            device_prop,
            1));
    return MyStatus::OK();
}

MyStatus GPUComputeSchedInfoKernel::ResetPass() {
    this->run_ctx->ResetPass();
    return MyStatus::OK();
}

std::unique_ptr<GPUKernel> GPUComputeSchedInfoKernel::clone() const {
    std::unique_ptr<GPUComputeSchedInfoKernel> obj(new GPUComputeSchedInfoKernel(this->args));
    if (this->run_ctx) {
        obj->device_prop = device_prop;
        obj->gpu_base_timestamp_ns = gpu_base_timestamp_ns;
        obj->device_base_timestamp_ns = device_base_timestamp_ns;
        obj->cpu_base_timestamp_us = cpu_base_timestamp_us;
        obj->trace_id = trace_id;
        obj->run_ctx.reset(new RunCtx(
                args,
                device_prop,
                this->run_ctx->output.num_elems()));
    }
    return obj;
}

void GPUComputeSchedInfoKernel::RunSync(CudaStream stream) {
    _gpu_compute_kernel(stream, /*sync=*/true);
}

void GPUComputeSchedInfoKernel::RunAsync(CudaStream stream) {
    _gpu_compute_kernel(stream, /*sync=*/false);
}

MyStatus GPUComputeSchedInfoKernel::DumpKernelInfo(int thread_id, CudaStream stream) {
    auto status = MyStatus::OK();
    auto stream_id = stream._stream_id;
    // Output:
    // GPUComputeSchedInfoKernel.thread_id_<>.stream_id_<>.json
    json js;
    json params_js;
    params_js["num_total_gpu_threads"] = run_ctx->sched_info.num_elems();
    params_js["num_blocks"] = args.FLAGS_kern_arg_num_blocks.get();
    params_js["threads_per_block"] = args.FLAGS_kern_arg_threads_per_block.get();
    params_js["iterations_per_sched_sample"] = args.FLAGS_kern_arg_iterations_per_sched_sample.get();
    params_js["iterations"] = args.FLAGS_kern_arg_iterations.get();
    params_js["n_samples"] =
            args.FLAGS_kern_arg_iterations.get() / args.FLAGS_kern_arg_iterations_per_sched_sample.get();
    params_js["n_launches"] = args.FLAGS_n_launches.get();
    params_js["processes"] = args.FLAGS_processes.get();
    params_js["cuda_context"] = args.FLAGS_cuda_context.get();
    params_js["device"] = args.FLAGS_device.get();
    params_js["cpu_base_timestamp_us"] = cpu_base_timestamp_us.time_since_epoch().count();
    params_js["gpu_base_timestamp_ns"] = gpu_base_timestamp_ns;
    params_js["device_base_timestamp_ns"] = device_base_timestamp_ns;

    js["params"] = params_js;
    js["sm_id"] = args.FLAGS_kern_arg_threads_per_block.get();
//    RLS_LOG("GPU_UTIL", "run_ctx->sched_info.num_elems() = {}", run_ctx->sched_info.num_elems());
    auto vectors = GPUThreadSchedInfoVectors::FromStructArray(run_ctx->sched_info.get(),
                                                              run_ctx->sched_info.num_elems());
    js["stream_id"] = vectors.stream_id;
    js["kernel_id"] = vectors.kernel_id;
    js["sm_id"] = vectors.sm_id;
    js["warp_id"] = vectors.warp_id;
    js["lane_id"] = vectors.lane_id;
    js["globaltimer_ns"] = vectors.globaltimer_ns;
    boost::filesystem::path iml_dir(args.FLAGS_iml_directory.get());
    std::stringstream base_ss;
    base_ss << "GPUComputeSchedInfoKernel"
            << ".thread_id_" << thread_id
            << ".stream_id_" << stream_id
            << ".trace_id_" << trace_id
            << ".json";
    trace_id += 1;
    auto path = iml_dir / base_ss.str();
    status = WriteJson(path.string(), js);
    IF_BAD_STATUS_RETURN(status);
    RLS_LOG("GPU_UTIL", "Dumped kernel info to {}", path);
    return MyStatus::OK();
}

size_t GPUComputeSchedInfoKernel::RunCtx::ComputeNumTotalSchedSamples(
    const GPUUtilExperimentArgs &args,
    const cudaDeviceProp &device_prop) {
  size_t n_samples_per_warp =
      args.FLAGS_kern_arg_iterations.get() / args.FLAGS_kern_arg_iterations_per_sched_sample.get();
  // n_total_samples = ((num_blocks * threads_per_block) / warp_size) * n_samples_per_warp * n_launches
  size_t n_total_warps = static_cast<size_t>(ceil(
      (args.FLAGS_kern_arg_num_blocks.get() * args.FLAGS_kern_arg_threads_per_block.get()) /
      static_cast<double>(device_prop.warpSize)));
  assert(n_total_warps >= 1);
  size_t n_total_sched_samples = n_total_warps * n_samples_per_warp * args.FLAGS_n_launches.get();
  if (n_total_sched_samples == 0) {
    DBG_BREAKPOINT_STACKTRACE("ComputeNumTotalSchedSamples");
  }
  return n_total_sched_samples;
}

MyStatus GPUClockFreq::Init() {
    return this->load_json(args.FLAGS_gpu_clock_freq_json.get());
}

std::unique_ptr<GPUKernel> GPUClockFreq::clone() const {
    return std::make_unique<GPUClockFreq>(*this);  // requires C++ 14
}

void GPUClockFreq::RunSync(CudaStream stream) {
    gpu_sleep_us_sync(stream, args.FLAGS_kernel_duration_us.get());
}

void GPUClockFreq::RunAsync(CudaStream stream) {
    gpu_sleep_us(stream, args.FLAGS_kernel_duration_us.get());
}

MyStatus GPUClockFreq::CheckArgs() {
    return MyStatus::OK();
}

MyStatus GPUClockFreq::DumpKernelInfo(int thread_id, CudaStream stream) {
    return MyStatus::OK();
}

MyStatus GetGPUKernel(GPUUtilExperimentArgs args, std::unique_ptr<GPUKernel> *gpu_kernel) {
    if (args.FLAGS_kernel.get() == "gpu_sleep") {
        (*gpu_kernel).reset(new GPUClockFreq(args));
    } else if (args.FLAGS_kernel.get() == "compute_kernel") {
        (*gpu_kernel).reset(new GPUComputeKernel(args));
    } else if (args.FLAGS_kernel.get() == "compute_kernel_sched_info") {
        (*gpu_kernel).reset(new GPUComputeSchedInfoKernel(args));
    } else {
        (*gpu_kernel).reset(nullptr);
        std::stringstream ss;
        ss << "Not sure what kernel to run for --kernel=" << args.FLAGS_kernel.get() << "; choices are:\n";
        PrintValue(ss, std::vector{"gpu_sleep", "compute_kernel"});
        return MyStatus(error::INVALID_ARGUMENT, ss.str());

    }
    return MyStatus::OK();
}

void GPUClockFreq::guess_cycles(CudaStream stream) {
    std::cout << "> Using initial sleep_cycles=" << _sleep_cycles << std::endl;
    while (true) {
        time_type start_t, end_t;
        iter(stream, &start_t, &end_t);
        auto total_sec = elapsed_sec(start_t, end_t);
        if (total_sec > GPU_CLOCK_MIN_SAMPLE_TIME_SEC) {
            std::cout << "> Using sleep_cycles=" << _sleep_cycles << ", which takes " << total_sec << " seconds"
                      << std::endl;
            break;
        } else if (total_sec > GPU_CLOCK_MIN_GUESS_TIME_SEC) {
            double cycles_per_second = _sleep_cycles / total_sec;
            auto seconds_left = GPU_CLOCK_MIN_SAMPLE_TIME_SEC - total_sec;
            // Add fudge factor of 10% more than we believe we need.
            auto guess_cycles_left = 1.1 * seconds_left * cycles_per_second;
            auto new_sleep_cycles = _sleep_cycles + guess_cycles_left;
            if (!(new_sleep_cycles > _sleep_cycles)) {
                std::cout << "total_sec = " << total_sec
                          << ", new_sleep_cycles = " << new_sleep_cycles
                          << ",  _sleep_cycles = " << _sleep_cycles
                          << std::endl;
                assert(new_sleep_cycles > _sleep_cycles);
            }
            _sleep_cycles = new_sleep_cycles;
        } else {
            auto new_sleep_cycles = _sleep_cycles * 2;
            if (!(new_sleep_cycles > _sleep_cycles)) {
                std::cout << "total_sec = " << total_sec
                          << ", new_sleep_cycles = " << new_sleep_cycles
                          << ",  _sleep_cycles = " << _sleep_cycles
                          << std::endl;
                assert(new_sleep_cycles > _sleep_cycles);
            }
            _sleep_cycles = new_sleep_cycles;
        }
    }
}

time_type time_now() {
    time_type t = steady_clock::now();
    return t;
}

double elapsed_sec(time_type start, time_type stop) {
    double sec = ((stop - start).count()) * steady_clock::period::num / static_cast<double>(steady_clock::period::den);
    return sec;
}

//std::chrono::nanoseconds elapsed_nano(time_type start, time_type stop) {
//  return std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
//}

void GPUClockFreq::iter(CudaStream stream, time_type *start_t, time_type *end_t) {
    *start_t = time_now();
    _gpu_sleeper.gpu_sleep_cycles_sync(stream, _sleep_cycles);
    *end_t = time_now();
}

double GPUClockFreq::freq_mhz(double time_sec) {
    return _sleep_cycles / time_sec;
}

MyStatus GPUClockFreq::dump_json() const {
    json js;
    MyStatus status = MyStatus::OK();
    js["time_secs"] = _time_secs;
    js["freq_mhz"] = _freq_mhz;
//  auto avg = Average(_freq_mhz);
//  auto std = Std(_freq_mhz);
    js["avg_mhz"] = _avg_mhz;
    js["std_mhz"] = _std_mhz;
    js["sleep_cycles"] = _sleep_cycles;
    auto path = this->json_path();
    status = WriteJson(path, js);
    IF_BAD_STATUS_RETURN(status);
    return MyStatus::OK();
}

MyStatus GPUClockFreq::load_json(const std::string &path) {
    json js;
    // auto path = this->json_path();
    MyStatus status = MyStatus::OK();
    status = ReadJson(path, &js);
    IF_BAD_STATUS_RETURN(status);
//  _time_secs = js["time_secs"];
//  _freq_mhz = js["freq_mhz"];
    _time_secs = js["time_secs"].get<std::vector<double>>();
    _freq_mhz = js["freq_mhz"].get<std::vector<double>>();
//  _time_secs = js["time_secs"].get<decltype(_time_secs)>();
//  _freq_mhz = js["freq_mhz"].get<decltype(_freq_mhz)>();
//  auto avg = Average(_freq_mhz);
//  auto std = Std(_freq_mhz);
    _avg_mhz = js["avg_mhz"];
    _std_mhz = js["std_mhz"];
    _sleep_cycles = js["sleep_cycles"];
    return MyStatus::OK();
}

std::string GPUClockFreq::json_basename() const {
    std::stringstream ss;
    ss << "gpu_clock_freq.json";
    return ss.str();
}

std::string GPUClockFreq::json_path() const {
    boost::filesystem::path direc(args.FLAGS_iml_directory.get());
    boost::filesystem::path base = json_basename();
    return (direc / base).string();
}

void GPUClockFreq::run() {
    CudaStream stream;
    guess_cycles(stream);
    time_type start_t, end_t;
    for (int r = 0; r < args.FLAGS_repetitions.get(); ++r) {
        iter(stream, &start_t, &end_t);
        auto total_sec = elapsed_sec(start_t, end_t);
        _time_secs.push_back(total_sec);
        auto freq = freq_mhz(total_sec);
        _freq_mhz.push_back(freq);
        std::cout << "> freq[" << r << "] = " << freq << " MHz" << std::endl;
    }

    _avg_mhz = Average(_freq_mhz);
    _std_mhz = Std(_freq_mhz);
//  _result = GPUClockResult{.avg_mhz = avg, .std_mhz = std};
    std::cout << "> Average freq = " << _avg_mhz << " MHz" << std::endl;
    std::cout << "> Std freq = " << _std_mhz << " MHz" << std::endl;

//  return _result;
}

void GPUKernelRunner::DelayUs(int64_t usec) {
//  int ret = usleep(usec);
//  assert(ret == 0);
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (microseconds >= usec) {
            break;
        }
    }
}

void ThreadedGPUKernelRunner::run() {

    if (!args.FLAGS_internal_is_child.get()) {
        // Create shared memory segment.
        _shared_mem = SharedMem(SharedMem::Parent(SHARED_MEM_NAME, SHARED_MEM_SIZE_BYTES));
        // num_threads + 1 since barrier is across all children AND parent.
        // (parent uses barrier to time just kernel execution portion, excluding setup/teardown).
        _sync_block = SyncBlock::Parent(&_shared_mem, SYNC_BLOCK_NAME, args.FLAGS_num_threads.get() + 1);
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "Parent saw {} @ {}", *_sync_block, reinterpret_cast<void *>(_sync_block));
        }
    }

    if (args.FLAGS_internal_is_child.get()) {
        // Just one thread launches kernels (default).
        // Run in same thread.
        std::unique_ptr<GPUKernel> gpu_kernel = _gpu_kernel->clone();
        GPUKernelRunner gpu_kernel_runner(
                args.FLAGS_internal_thread_id.get(),
                std::move(gpu_kernel),
                args);
        gpu_kernel_runner.run_child_process();
        return;
    }

    run_parent();
}

MyStatus MaybeInitCUDA(int device, CUcontext* context) {
  *context = nullptr;

  // Initialize the CUDA driver API if it hasn't been already (e.g. during LD_PRELOAD).
  {
    int deviceCount;
    CUresult cu_ret = cuDeviceGetCount(&deviceCount);
    if (cu_ret == CUDA_ERROR_NOT_INITIALIZED) {
      // cuInit(0) hasn't been called yet; call it so we can make CUDA API calls.
      RLS_LOG("GPU_UTIL", "Initializing CUDA driver API with cuInit(0)", device);
      DRIVER_API_CALL_MAYBE_STATUS(cuInit(0));
    } else {
      // cuDeviceGetCount failed with an error we don't know how to handle...
      // Call it again to report the error:
      DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGetCount(&deviceCount));
    }
  }

  {
    // CUcontext context = nullptr;
    // FAILS during api.set_metadata with initialization error (3)
    DRIVER_API_CALL_MAYBE_STATUS(cuCtxGetCurrent(context));
    if (context == nullptr) {
      std::stringstream ss;
      RLS_LOG("GPU_UTIL", "Created CUDA context for device={} since it didn't already exist", device);
      DRIVER_API_CALL_MAYBE_STATUS(cuCtxCreate(context, 0, device));
//      ss << "GPUHwCounterSampler: no CUDA context has been created yet";
//      return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
  }

  return MyStatus::OK();
}

void ThreadedGPUKernelRunner::run_parent() {
    // Parent only from here on.
    // We coordinate the children.

    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
        RLS_LOG("GPU_UTIL", "ThreadedGPUKernelRunner launching kernels from {} threads", args.FLAGS_num_threads.get());
    }

    assert(!args.FLAGS_internal_is_child.get());
    assert(!args.FLAGS_internal_thread_id.has_value());

    //
    // Setup to perform before launching any threads/processes.
    //

    // NOTE: This is required.  I think we can assume a CUDA
    // code-base will make these calls for us before sampler is used.

  CUcontext context;
  IF_BAD_STATUS_EXIT_WITH(MaybeInitCUDA(args.FLAGS_device.get(), &context));

////    DRIVER_API_CALL_MAYBE_EXIT(cuInit(0));
//    CUdevice cuDevice;
//    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGet(&cuDevice, args.FLAGS_device.get()));
//    CUcontext cuContext;
////    DRIVER_API_CALL_MAYBE_EXIT(cuCtxCreate(&cuContext, 0, cuDevice));
//    DRIVER_API_CALL_MAYBE_EXIT(cuCtxGetCurrent(&cuContext));
//    assert(cuContext != nullptr);

    MyStatus ret = MyStatus::OK();
    rlscope::GPUHwCounterSampler sampler(args.FLAGS_device.get(), args.FLAGS_iml_directory.get(), "");

    if (!args.FLAGS_hw_counters.get() || args.FLAGS_processes.get()) {
        ret = sampler.Disable();
        IF_BAD_STATUS_EXIT("Failed to disable GPU hw counter profiler", ret);
    }

    ret = sampler.Init();
    IF_BAD_STATUS_EXIT("Failed to initialize GPU hw counter profiler", ret);

    //
    // Launch child threads/processes.
    //

    // >= 2 threads.
    // Launch kernels from multiple threads
    // Q: Separate stream?  Same stream?
    // A: In minigo, there are separate processes...so to be accurate we should fork().
    //    From Serina's experiments with GPU scheduling, the GPU will NOT allow separate kernels to overlap and will
    //    instead serialize their execution.  But separate streams from the same process, I expect there to be overlap.
    //    We should try both, and measure BOTH utilization AND overlap.
    std::vector<GPUKernelRunner> gpu_kernel_runners;
    gpu_kernel_runners.reserve(args.FLAGS_num_threads.get());

    GPUUtilExperimentArgs child_args = args;
    child_args.FLAGS_internal_is_child = true;
    for (int i = 0; i < static_cast<int>(args.FLAGS_num_threads.get()); i++) {
        std::unique_ptr<GPUKernel> gpu_kernel = _gpu_kernel->clone();
        gpu_kernel_runners.emplace_back(
                i,
                std::move(gpu_kernel),
                child_args);
    }
    int i;

    i = 0;
    for (auto &gpu_kernel_runner : gpu_kernel_runners) {
        if (!args.FLAGS_processes.get()) {
            gpu_kernel_runner.run_async_thread();
        } else {
            gpu_kernel_runner.run_async_process(i);
        }
        i++;
    }


    // - Parsed by: ParseMetricNameString
    //   - <metric_name>[$|&][+]
    //
    //   - default if no symbols:
    //     keepInstances = false
    //     isolated = true
    //
    //   - keepInstances = "+" present
    //     isolated = "&" is NOT present
    //     (NOTE $ is redundant? it make isolated=True, but isolated=True is the default).

    // achieved_occupancy = sm__warps_active.avg.pct_of_peak_sustained_active+
    ret = sampler.StartConfig(args.FLAGS_metrics.get());
    IF_BAD_STATUS_EXIT("Failed to configure GPU hw counter profiler", ret);

    // isolated = false
    // NOTE: this doesn't work; I've posted on NVIDIA's forum about it:
    // https://forums.developer.nvidia.com/t/multi-process-profiling-nvpa-rawmetricrequest-isolated-false/124798
//    ret = sampler.StartConfig({"sm__warps_active.avg.pct_of_peak_sustained_active&+"});
//    IF_BAD_STATUS_EXIT("Failed to configure GPU hw counter profiler", ret);

    // (PARENT BARRIER 1) Wait for threads to setup CUDA context.
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
        RLS_LOG("GPU_UTIL", "{}", "(PARENT BARRIER 1) wait for threads to setup CUDA context...");
    }
    _sync_block->barrier.arrive_and_wait("barrier 1: setup CUDA context");

    struct PassStats {
        size_t pass_idx;
        time_type start_kernels_t;
        time_type done_kernels_t;

        double DurationSec() const {
            double seconds = std::chrono::duration_cast<std::chrono::microseconds>(done_kernels_t - start_kernels_t).count() /
                             static_cast<double>(MICROSECONDS_IN_SECOND);
            return seconds;
        }

        void Print(bool config_pass) const {
            auto seconds = this->DurationSec();
            std::string pass_type;
            if (config_pass) {
                pass_type = "[CONFIG] ";
            }
            RLS_LOG("GPU_UTIL", "{}Pass {}: running kernels took {} seconds", pass_type, pass_idx + 1, seconds);
        }

    };
    std::list<PassStats> pass_stats;
    size_t pass_idx = 0;

    auto parent_run_pass = [&] (bool should_stop, bool config_pass) {
        MyStatus status;
        PassStats stats;
        stats.pass_idx = pass_idx;
        stats.start_kernels_t = time_now();

        if (should_stop) {
            _sync_block->should_stop = should_stop;
            // (PARENT BARRIER 2) Tell children to start running GPU kernels, OR, should_stop.
            if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
                RLS_LOG("GPU_UTIL", "(PARENT BARRIER 2) should_stop = {}", should_stop);
            }
            _sync_block->barrier.arrive_and_wait("barrier 2/3: start GPU kernels or should_stop");
            return;
        }

        status = sampler.StartPass();
        IF_BAD_STATUS_EXIT("Failed to start GPU hw pass", status);

        status = sampler.Push("run_kernels");
        IF_BAD_STATUS_EXIT("Failed to push range for GPU hw pass", status);

        // (PARENT BARRIER 3) Tell children to start running GPU kernels, OR, should_stop.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(PARENT BARRIER 3) start kernels");
        }
        _sync_block->barrier.arrive_and_wait("barrier 2/3: start GPU kernels or should_stop");

        // (PARENT BARRIER 4) Wait for threads finish executing GPU kernels.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(PARENT BARRIER 4) wait for kernels...");
        }
        _sync_block->barrier.arrive_and_wait("barrier 4: finish GPU kernels");
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(PARENT BARRIER 4) ... kernels done.");
        }
        stats.done_kernels_t = time_now();

        _sync_block->barrier.arrive_and_wait("barrier 5: record pass info");
        _sync_block->barrier.arrive_and_wait("barrier 6: check pass info agreement");

        // "run_kernels"
        status = sampler.Pop();
        IF_BAD_STATUS_EXIT("Failed to pop range for GPU hw pass", status);

        status = sampler.EndPass();
        IF_BAD_STATUS_EXIT("Failed to end GPU hw pass", status);

        stats.Print(config_pass);

        pass_idx += 1;

        if (!config_pass) {
            pass_stats.push_back(stats);
        }

    };

    size_t num_passes = 0;
    if (args.FLAGS_hw_counters.get()) {
        parent_run_pass(false, true);
        if (args.FLAGS_processes.get()) {
            // Processes.
            //
            // Each child will run a configuration pass for us, and inform us how many passes are required.
            num_passes = _sync_block->num_passes;
            RLS_LOG("GPU_UTIL", "GPU hw config results: num_passes = {}",
                    num_passes);
        } else {
            // Threads.
            // Run a configuration pass in the "parent" thread.
            //
            // Configuration pass.
            RLS_LOG("GPU_UTIL", "GPU hw config results: MaxNumRanges = {}, MaxNestingLevels = {}",
                    sampler.MaxNumRanges(),
                    sampler.MaxNestingLevels());
            num_passes = sampler.NumPasses();
        }
        assert(num_passes != 0);
    }

    ret = sampler.StartProfiling();
    IF_BAD_STATUS_EXIT("Failed to start GPU hw counter profiler", ret);

    for (int64_t n = 0; n < args.FLAGS_samples.get(); n++) {
        if (args.FLAGS_hw_counters.get()) {
//            while (sampler.HasNextPass()) {
            for (size_t pass_i = 0; pass_i < num_passes; pass_i++) {
                // Samples.
                parent_run_pass(false, false);
            }
            ret = sampler.RecordSample();
            IF_BAD_STATUS_EXIT("Failed to record GPU hw counter sample", ret);

            ret = sampler.DumpSync();
            IF_BAD_STATUS_EXIT("Failed to dump GPU hw counter sample", ret);
        } else {
            assert(false);
            parent_run_pass(false, false);
        }
    }
    // Signal STOP (don't actually run a pass).
    parent_run_pass(true, false);

    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
        RLS_LOG("GPU_UTIL", "{}", "Parent waits for children...");
    }
    i = 0;
    for (auto &gpu_kernel_runner : gpu_kernel_runners) {
        if (!args.FLAGS_processes.get()) {
            gpu_kernel_runner.wait_thread();
        } else {
            gpu_kernel_runner.wait_process(i);
        }
        i++;
    }
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
        RLS_LOG("GPU_UTIL", "{}", "... Parent sees children finish.");
    }

    ret = sampler.StopProfiling();
    IF_BAD_STATUS_EXIT("Failed to stop GPU hw counter profiler", ret);

    {
        for (const auto &stats : pass_stats) {
            stats.Print(false);
        }
    }

    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
        RLS_LOG("GPU_UTIL", "After children done, Parent sees: {}", *_sync_block);
    }

}

SyncBlock::~SyncBlock() {
    RLS_LOG("GPU_UTIL", "Destructor for {}", *this);
}

void GPUKernelRunner::run_async_thread() {
    assert(_async_thread == nullptr);
    _async_thread.reset(new std::thread([this] {
        this->run_child_thread();
    }));
}

void GPUKernelRunner::wait_thread() {
    assert(_async_thread != nullptr);
    _async_thread->join();
    _async_thread.reset(nullptr);
}

void GPUKernelRunner::run_async_process(int thread_id) {
    assert(_async_process == nullptr);
    _async_process.reset(new boost::process::child);
    GPUUtilExperimentArgs child_args = args;
    child_args.FLAGS_num_threads = 1;
    child_args.FLAGS_internal_is_child = true;
    child_args.FLAGS_internal_thread_id = _thread_id;

    //get a handle to the current environment
    auto env = boost::this_process::environment();
    std::stringstream process_name_ss;
    boost::process::environment child_env = env;
    process_name_ss << env["IML_PROCESS_NAME"].to_string() << ".thread_" << thread_id;
    child_env["IML_PROCESS_NAME"] = process_name_ss.str();

    *_async_process = ReinvokeProcess(child_args, child_env);
}

void GPUKernelRunner::wait_process(int thread_id) {
    assert(_async_process);
    // Q: How are non-zero exit codes handled?
    _async_process->wait();
    _async_process.reset(nullptr);
}

void GPUKernelRunner::SetCurrentThreadName(const std::string &name) {
//    nvtxNameOsThreadA(pthread_self(), name.c_str());
}

MyStatus GPUKernelRunner::_run_child_pass(SyncBlock *sync_block, RunContext* _run_ctx) {
    MyStatus status = MyStatus::OK();

    status = _gpu_kernel->ResetPass();
    IF_BAD_STATUS_EXIT("Failed to reset pass", status);

    // Launch a kernel that runs for --kernel_duration_us microseconds.
    // Launch the kernel every --kernel_delay_us microseconds.
//    time_type start_t = time_now();

//  assert(args.FLAGS_kernel_duration_us.get() % MICROSECONDS_IN_SECOND == 0);

    struct GPUKernelRun {
        int64_t time_sleeping_us;
        int64_t after_delay_us;
    };
    std::list<GPUKernelRun> kernel_runs;

    TimeHistogram hist("cudaLaunchKernel");

//    bool nvtx_is_pushed = false;

    int64_t launches = 0;

//    nvtxRangePush("run_kernels");
////  const int NVTX_PUSH_EVERY = 2;
//    const int NVTX_PUSH_EVERY = 1;
    while (true) {
//        if (launches % NVTX_PUSH_EVERY == 0) {
//            if (nvtx_is_pushed) {
//                // run_kernel_every
//                RLS_LOG("GPU_UTIL", "nvtxRangePop() @ launches = {}", launches);
//                nvtxRangePop();
//            }
//            RLS_LOG("GPU_UTIL", "nvtxRangePush(\"{}\") @ launches = {}", "run_kernel_every", launches);
//            nvtxRangePush("run_kernel_every");
//            nvtx_is_pushed = true;
//        }

        time_type now_t = time_now();
//        auto time_sec = elapsed_sec(start_t, now_t);
        if (
//        time_sec >= args.FLAGS_run_sec.get() ||
                launches >= args.FLAGS_n_launches.get()) {
            // Q: Should we wait for the stream to finish here?
            // Otherwise, cudaStreamDestroy will free the stream in ~CudaStreamWrapper without waiting for the stream to finish;
            // I suspect that could result in traces NOT being collected for remaining kernels (unsure...)
            break;
        }
        GPUKernelRun kernel_run;
        auto before_sleep_t = time_now();


        // THIS part is specific to the kernel implementation.
        // GPU clock freq is an argument that is specific to the GPU sleep kernel.
        // Kernel interface:
        // Kernel(GPUUtilExperimentArgs args)
        // Kernel.RunSync()
        // Kernel.RunAsync()

        // NOTE: this WILL wait for the kernel to finish first using cudaDeviceSynchronize()....
        // Alternatives:
        // - wait for just the launched kernel to finish using cudaEvent => multi-threaded friendly
        // - Q: Do we want to launch another kernel BEFORE the last one finishes...?
        //   ... if waiting for it to finish takes a long time (> 5 us)... then yes?

//        nvtxRangePush("run_kernel_each");

        if (args.FLAGS_sync.get()) {
            _gpu_kernel->RunSync(*_run_ctx->_stream);
//      _freq.gpu_sleep_us_sync(*_run_ctx->_stream, args.FLAGS_kernel_duration_us.get());
        } else {
            _gpu_kernel->RunAsync(*_run_ctx->_stream);
//      _freq.gpu_sleep_us(*_run_ctx->_stream, args.FLAGS_kernel_duration_us.get());
        }
        // run_kernel

//        nvtxRangePop();

        launches += 1;
        auto after_sleep_t = time_now();
        auto nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(after_sleep_t - before_sleep_t).count();
        hist.CountNanoseconds(nanosec);
        auto time_sleeping_sec = elapsed_sec(before_sleep_t, after_sleep_t);

        int64_t time_sleeping_us = time_sleeping_sec * MICROSECONDS_IN_SECOND;
        kernel_run.time_sleeping_us = time_sleeping_us;
//    if (SHOULD_DEBUG(FEATURE_GPU_CLOCK_FREQ)) {
//      auto off_by = args.FLAGS_kernel_duration_us.get() - time_sleeping_us;
//      RLS_LOG("GPU_UTIL", "GPU.sleep for {} us (off by {} us)", time_sleeping_us, off_by);
//    }
        auto before_delay_t = time_now();
        DelayUs(args.FLAGS_kernel_delay_us.get());
//    int ret = usleep(args.FLAGS_kernel_delay_us.get());
//    assert(ret == 0);
        auto after_delay_t = time_now();
        auto after_delay_sec = elapsed_sec(before_delay_t, after_delay_t);
        int64_t after_delay_us = after_delay_sec * MICROSECONDS_IN_SECOND;
        kernel_run.after_delay_us = after_delay_us;
//    if (SHOULD_DEBUG(FEATURE_GPU_CLOCK_FREQ)) {
//      auto off_by = args.FLAGS_kernel_delay_us.get() - after_delay_us;
//      RLS_LOG("GPU_UTIL", "CPU.delay for {} us (off by {} us)", after_delay_us, off_by);
//    }
        if (args.FLAGS_debug.get()) {
            kernel_runs.push_back(std::move(kernel_run));
        }
    }

//    if (nvtx_is_pushed) {
//        // run_kernel_every
//        RLS_LOG("GPU_UTIL", "nvtxRangePop() @ launches = {}", launches);
//        nvtxRangePop();
//    }
//
//    // run_kernels
//    nvtxRangePop();

    _run_ctx->synchronize();
    if (args.FLAGS_internal_is_child.get()) {
        // (CHILD BARRIER 4) Barrier synchronization to wait for ALL GPU kernels to finish executing across all threads.
        //
        // This is to ensure a fair comparison of "how long" it takes multi-process/multi-thread/multi-context
        // to launch --n_launches kernels, WITHOUT considering setup cost.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 4) wait for other children to finish kernels...");
        }
        sync_block->barrier.arrive_and_wait("barrier 4: finish GPU kernels");
    }

    _gpu_kernel->DumpKernelInfo(_thread_id, *_run_ctx->_stream);

//    args.FLAGS_debug.get()
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_KERNEL_TIME)) {
        std::stringstream ss;
        size_t i = 0;
        int indent = 0;
        PrintIndent(ss, indent);
        ss << "GPUKernelRuns: size = " << kernel_runs.size() << "\n";
        for (auto const &kernel_run : kernel_runs) {
            PrintIndent(ss, indent + 1);
            ss << "[" << i << "]\n";

            PrintIndent(ss, indent + 2);
            int64_t off_by_time_sleeping_us = args.FLAGS_kernel_duration_us.get() - kernel_run.time_sleeping_us;
            ss << "GPU.sleep for " << kernel_run.time_sleeping_us << " us (off by " << off_by_time_sleeping_us
               << " us)\n";

            PrintIndent(ss, indent + 2);
            int64_t off_by_after_delay_us = args.FLAGS_kernel_delay_us.get() - kernel_run.after_delay_us;
            ss << "CPU.delay for " << kernel_run.after_delay_us << " us (off by " << off_by_after_delay_us << " us)\n";

            i += 1;
        }
        RLS_LOG("GPU_UTIL", "{}", ss.str());
    }

    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_KERNEL_TIME)) {
        std::stringstream ss;
        ss << hist;
        RLS_LOG("GPU_UTIL", "{}", ss.str());
    }

    return MyStatus::OK();

}

void GPUKernelRunner::run_child_process() {
    // Child process must create is own GPU hw sampler:
    // - run its own configuration passes
    MyStatus status = MyStatus::OK();
    MyStatus ret = MyStatus::OK();

    // NOTE: This is required.  I think we can assume a CUDA
    // code-base will make these calls for us before sampler is used.
//    DRIVER_API_CALL_MAYBE_EXIT(cuInit(0));

//    std::unique_ptr<RunContext> _run_ctx;
//    SharedMem shared_mem;
//    SyncBlock *sync_block = nullptr;
//    this->_setup_child_thread(&_run_ctx, &shared_mem, &sync_block);

    std::unique_ptr<RunContext> _run_ctx;
    SharedMem shared_mem;
    SyncBlock *sync_block = nullptr;

    {
        std::stringstream ss;
        ss << _thread_id;
        this->SetCurrentThreadName(ss.str());
    }

    // NOTE: this is NOT a member variable, since we want it to be destructed within the child thread, NOT the parent.
    // That's because the CUDA context is bound to this thread.
//    std::unique_ptr<RunContext> _run_ctx;

//    SharedMem shared_mem;
    sync_block = nullptr;
//    RLS_LOG("GPU_UTIL", "args.FLAGS_internal_is_child = {}", args.FLAGS_internal_is_child.get());
    if (args.FLAGS_internal_is_child.get()) {
        shared_mem = SharedMem::Child(SHARED_MEM_NAME);
        sync_block = SyncBlock::Child(&shared_mem, SYNC_BLOCK_NAME);
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "Child saw {} @ {}", *sync_block, reinterpret_cast<void *>(sync_block));
        }
    }

    CUdevice cuDevice;
    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGet(&cuDevice, args.FLAGS_device.get()));
    CUcontext cuContext;
//    DRIVER_API_CALL_MAYBE_EXIT(cuCtxCreate(&cuContext, 0, cuDevice));
    DRIVER_API_CALL_MAYBE_EXIT(cuCtxGetCurrent(&cuContext));
    assert(cuContext != nullptr);

    std::stringstream suffix_ss;
    suffix_ss << ".process_" << args.FLAGS_internal_thread_id.get();

    rlscope::GPUHwCounterSampler sampler(args.FLAGS_device.get(), args.FLAGS_iml_directory.get(), suffix_ss.str());
    assert(args.FLAGS_processes.get());
    if (!args.FLAGS_hw_counters.get()) {
        ret = sampler.Disable();
        IF_BAD_STATUS_EXIT("Failed to disable GPU hw counter profiler", ret);
    }

    if (_thread_id != 0) {
        RLS_LOG("GPU_UTIL", "DISABLE GPU HW PROFILING FOR THREAD {}", _thread_id);
        ret = sampler.Disable();
        IF_BAD_STATUS_EXIT("Failed to disable GPU hw counter profiler", ret);
    }

    // IMPORTANT: for some reason, cuptiProfilerInitialize(...) must happen BEFORE RunContext calls cudaStreamCreateWithFlags(...),
    // otherwise, cuptiProfilerInitialize returns "CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED" (not sure why...).
    // Q: In RLScope, should we call cuInit(0) and cuptiProfilerInitialize(...) immediately on shared library load?
    ret = sampler.Init();
    IF_BAD_STATUS_EXIT("Failed to initialize GPU hw counter profiler", ret);

    _run_ctx.reset(new RunContext(
            _thread_id,
            args.FLAGS_cuda_context.get(),
            args.FLAGS_cuda_context_flags.get()));

    // NOTE: some GPU kernels need to initialize AFTER CUDA context has been created.
    status = _gpu_kernel->Init();
    IF_BAD_STATUS_EXIT("Failed to initialize kernel", status);


    // achieved_occupancy = sm__warps_active.avg.pct_of_peak_sustained_active+
    ret = sampler.StartConfig(args.FLAGS_metrics.get());
    IF_BAD_STATUS_EXIT("Failed to configure GPU hw counter profiler", ret);

    if (args.FLAGS_internal_is_child.get()) {
        // (CHILD BARRIER 1) Barrier synchronization to wait for ALL threads to finish creating CUDA context.
        //
        // This is to ensure a fair comparison of "how long" it takes multi-process/multi-thread/multi-context
        // to launch --n_launches kernels, WITHOUT considering setup cost.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 1) wait for other children to setup CUDA context...");
        }
        sync_block->barrier.arrive_and_wait("barrier 1: setup CUDA context");
    }

    size_t pass_idx = 0;
    auto child_run_pass = [&] (bool should_stop, bool config_pass) {
        MyStatus status;
//        PassStats stats;
//        stats.pass_idx = pass_idx;
//        stats.start_kernels_t = time_now();

        if (should_stop) {
//            sync_block->should_stop = should_stop;
            // (CHILD BARRIER 2) Tell children to start running GPU kernels, OR, should_stop.
            sync_block->barrier.arrive_and_wait("barrier 2/3: start GPU kernels or should_stop");
            if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
                RLS_LOG("GPU_UTIL", "(CHILD BARRIER 2) should_stop = {}", sync_block->should_stop.load());
            }
            assert(sync_block->should_stop.load());
            return;
        }

        status = sampler.StartPass();
        IF_BAD_STATUS_EXIT("Failed to start GPU hw pass", status);

        status = sampler.Push("run_kernels");
        IF_BAD_STATUS_EXIT("Failed to push range for GPU hw pass", status);

        // (CHILD BARRIER 3) Tell children to start running GPU kernels, OR, should_stop.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 3) start kernels");
        }
        sync_block->barrier.arrive_and_wait("barrier 2/3: start GPU kernels or should_stop");
        assert(!sync_block->should_stop.load());

        // NOTE: 4th child barrier happens in _run_child_pass
        status = this->_run_child_pass(sync_block, _run_ctx.get());
        IF_BAD_STATUS_EXIT("Failed to run GPU hw pass in child process", status);
        // (CHILD BARRIER 4) Wait for threads finish executing GPU kernels.
//        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
//            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 4) wait for kernels...");
//        }
//        sync_block->barrier.arrive_and_wait("barrier 4: finish GPU kernels");
//        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
//            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 4) ... kernels done.");
//        }
//        stats.done_kernels_t = time_now();

        // "run_kernels"
        status = sampler.Pop();
        IF_BAD_STATUS_EXIT("Failed to pop range for GPU hw pass", status);

        status = sampler.EndPass();
        IF_BAD_STATUS_EXIT("Failed to end GPU hw pass", status);

        if (config_pass && _thread_id == 0) {
            sync_block->num_passes = sampler.NumPasses();
        }
        sync_block->barrier.arrive_and_wait("barrier 5: record pass info");
        if (config_pass && sampler.Enabled()) {
            assert(sync_block->num_passes == sampler.NumPasses());
        }
        sync_block->barrier.arrive_and_wait("barrier 6: check pass info agreement");

//        stats.Print();

        pass_idx += 1;

//        if (!config_pass) {
//            pass_stats.push_back(stats);
//        }

    };

    size_t num_passes = 0;
    if (args.FLAGS_hw_counters.get()) {
        RLS_LOG("GPU_UTIL", "Running configuration pass from thread {}", _thread_id);
        child_run_pass(false, true);
        assert(args.FLAGS_processes.get());
        // Processes.
        //
        // Each child will run a configuration pass for us, and inform us how many passes are required.
        num_passes = sync_block->num_passes;
        assert(num_passes != 0);
    }

    ret = sampler.StartProfiling();
    IF_BAD_STATUS_EXIT("Failed to start GPU hw counter profiler", ret);

    for (int64_t n = 0; n < args.FLAGS_samples.get(); n++) {
        RLS_LOG("GPU_UTIL", "Child sample = {}", n);
        if (args.FLAGS_hw_counters.get()) {
//            while (sampler.HasNextPass()) {
            for (size_t pass_i = 0; pass_i < num_passes; pass_i++) {
                // Samples.
                RLS_LOG("GPU_UTIL", "Child pass_i = {}", pass_i);
                child_run_pass(false, false);
            }
            assert(!sampler.HasNextPass());
            ret = sampler.RecordSample();
            IF_BAD_STATUS_EXIT("Failed to record GPU hw counter sample", ret);

            ret = sampler.DumpSync();
            IF_BAD_STATUS_EXIT("Failed to dump GPU hw counter sample", ret);
        } else {
            assert(false);
            child_run_pass(false, false);
        }
    }
    // Signal STOP (don't actually run a pass).
    child_run_pass(true, false);
}
void GPUKernelRunner::_setup_child_thread(
        std::unique_ptr<RunContext>* _run_ctx,
        SharedMem* shared_mem,
        SyncBlock **sync_block = nullptr) {
    MyStatus status = MyStatus::OK();

    {
        std::stringstream ss;
        ss << _thread_id;
        this->SetCurrentThreadName(ss.str());
    }

    // NOTE: this is NOT a member variable, since we want it to be destructed within the child thread, NOT the parent.
    // That's because the CUDA context is bound to this thread.
    *sync_block = nullptr;
//    RLS_LOG("GPU_UTIL", "args.FLAGS_internal_is_child = {}", args.FLAGS_internal_is_child.get());
    if (args.FLAGS_internal_is_child.get()) {
        *shared_mem = SharedMem::Child(SHARED_MEM_NAME);
        *sync_block = SyncBlock::Child(shared_mem, SYNC_BLOCK_NAME);
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "Child saw {} @ {}", **sync_block, reinterpret_cast<void *>(*sync_block));
        }
    }

    (*_run_ctx).reset(new RunContext(
            _thread_id,
            args.FLAGS_cuda_context.get(),
            args.FLAGS_cuda_context_flags.get()));

    // NOTE: some GPU kernels need to initialize AFTER CUDA context has been created.
    status = _gpu_kernel->Init();
    IF_BAD_STATUS_EXIT("Failed to initialize kernel", status);

}
void GPUKernelRunner::run_child_thread() {
    MyStatus status = MyStatus::OK();

    std::unique_ptr<RunContext> _run_ctx;
    SharedMem shared_mem;
    SyncBlock *sync_block = nullptr;
    this->_setup_child_thread(&_run_ctx, &shared_mem, &sync_block);

    // TODO: remove if stmt?
    if (args.FLAGS_internal_is_child.get()) {
        // (CHILD BARRIER 1) Barrier synchronization to wait for ALL threads to finish creating CUDA context.
        //
        // This is to ensure a fair comparison of "how long" it takes multi-process/multi-thread/multi-context
        // to launch --n_launches kernels, WITHOUT considering setup cost.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 1) wait for other children to setup CUDA context...");
        }
        sync_block->barrier.arrive_and_wait("barrier 1: setup CUDA context");
    }

    while (true) {
        // (CHILD BARRIER 2/3) Tell children to start running GPU kernels, OR, should_stop.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "{}", "(CHILD BARRIER 2/3) wait for order from parent...");
        }
        sync_block->barrier.arrive_and_wait("barrier 2/3: start GPU kernels or should_stop");
        bool should_stop = sync_block->should_stop.load();
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
            RLS_LOG("GPU_UTIL", "(CHILD BARRIER 2/3) should_stop = {}", should_stop);
        }
        if (should_stop) {
           break;
        }
        status = _run_child_pass(sync_block, _run_ctx.get());
        IF_BAD_STATUS_EXIT("Failed to run pass in child", status);

        sync_block->barrier.arrive_and_wait("barrier 5: record pass info");
        sync_block->barrier.arrive_and_wait("barrier 6: check pass info agreement");
    }
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
        RLS_LOG("GPU_UTIL", "{}", "CHILD FINISHED");
    }

}

CudaContext::CudaContext() :
        _context(new CudaContextWrapper()) {
}

CudaContext::CudaContext(unsigned int flags) :
        _context(new CudaContextWrapper(flags)) {
}

CUcontext CudaContext::get() const {
    return _context->_handle;
}

void CudaContext::synchronize() {
    return _context->synchronize();
}

void CudaContextWrapper::synchronize() {
    RUNTIME_API_CALL_MAYBE_EXIT(cudaDeviceSynchronize());
}

void CudaContextWrapper::_Init() {
    DRIVER_API_CALL_MAYBE_EXIT(cuCtxCreate(&_handle, _flags, _dev));
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)) {
        std::stringstream ss;
//  RLS_LOG("GPU_UTIL", "Create CUDA context = {}", reinterpret_cast<void*>(_handle));
        backward::StackTrace st;
        st.load_here(32);
        backward::Printer p;
        ss << "Create CUDA context = " << reinterpret_cast<void *>(_handle) << "\n";
        p.print(st, ss);
        RLS_LOG("GPU_UTIL", "{}", ss.str());
    }
    assert(_handle != nullptr);
}

CudaContextWrapper::CudaContextWrapper(unsigned int flags) :
        _handle(nullptr),
        _flags(flags),
        // When running really long kernels, CUDA's default scheduling policy
        // (CU_CTX_SCHED_AUTO -> CU_CTX_SCHED_SPIN) still polls waiting for GPU kernels to finish.
        // This leads to high CPU utilization.  This can be eliminated by changing the scheduling
        // policy associated with the CUDA context to CU_CTX_SCHED_BLOCKING_SYNC.
        // This drops CPU utilization down to zero.
//    _flags(CU_CTX_SCHED_BLOCKING_SYNC),
        _dev(0) {
    _Init();
}

CudaContextWrapper::CudaContextWrapper() :
        _handle(nullptr),
        _flags(0),
        // When running really long kernels, CUDA's default scheduling policy
        // (CU_CTX_SCHED_AUTO -> CU_CTX_SCHED_SPIN) still polls waiting for GPU kernels to finish.
        // This leads to high CPU utilization.  This can be eliminated by changing the scheduling
        // policy associated with the CUDA context to CU_CTX_SCHED_BLOCKING_SYNC.
        // This drops CPU utilization down to zero.
//    _flags(CU_CTX_SCHED_BLOCKING_SYNC),
        _dev(0) {
    _Init();
}

CudaContextWrapper::~CudaContextWrapper() {
    if (_handle) {
        // Wait for remaining kernels on all streams for the context to complete.
        // (driver API assumes you have done this before destroying context.)
        CUresult result;
        this->synchronize();
        // NOTE: I've seen this segfault _sometimes_ for multi-context runs... not sure why.
        if (SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)) {
            RLS_LOG("GPU_UTIL", "Destroy cuda context @ handle={}", reinterpret_cast<void *>(_handle));
        }
        DRIVER_API_CALL_MAYBE_EXIT(cuCtxDestroy(_handle));
        _handle = nullptr;
    }
}

CudaStream::CudaStream() :
        _stream_id(0),
        _stream(new CudaStreamWrapper()) {
}

cudaStream_t CudaStream::get() const {
    return _stream->_handle;
}

void CudaStream::synchronize() {
    return _stream->synchronize();
}

void CudaStream::_set_name(const std::string &name) {
//    nvtxNameCudaStreamA(_stream->_handle, name.c_str());
}

void CudaStream::set_stream_id(uint64_t stream_id) {
    _stream_id = stream_id;
    std::stringstream ss;
    ss << _stream_id;
    _set_name(ss.str());
}

void set_stream_id(uint64_t stream_id);

void CudaStreamWrapper::synchronize() {
    RUNTIME_API_CALL_MAYBE_EXIT(cudaStreamSynchronize(_handle));
}

CudaStreamWrapper::CudaStreamWrapper() :
        _handle(nullptr) {
    cudaError_t ret;
//  ret = cudaStreamCreate(&_handle);
    RUNTIME_API_CALL_MAYBE_EXIT(cudaStreamCreateWithFlags(&_handle, cudaStreamNonBlocking));
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)) {
        RLS_LOG("GPU_UTIL", "Create CUDA stream = {}", reinterpret_cast<void *>(_handle));
    }
    assert(_handle != nullptr);
}

CudaStreamWrapper::~CudaStreamWrapper() {
    cudaError_t ret;
    if (_handle) {
        // Wait for remaining kernels on stream to complete.
        this->synchronize();
        RUNTIME_API_CALL_MAYBE_EXIT(cudaStreamDestroy(_handle));
        _handle = nullptr;
    }
}

} // namespace rlscope
