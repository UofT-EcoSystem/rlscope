//
// Created by jgleeson on 2020-01-23.
//

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <thread>

#include <cuda_runtime.h>
#include <cassert>
#include <sstream>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "experiment/gpu_freq.h"

#include "common/json.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <pthread.h>

#include <backward.hpp>

#include "common/my_status.h"

#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/debug_flags.h"

#include <sys/syscall.h>
#include "drivers/gpu_util_experiment.h"

static pid_t my_gettid() {
  return syscall(SYS_gettid);
}

namespace tensorflow {

using clock_value_t = long long;

using steady_clock = std::chrono::steady_clock;

void GPUClockFreq::guess_cycles(CudaStream stream) {
  std::cout << "> Using initial sleep_cycles=" << _sleep_cycles << std::endl;
  while (true) {
    time_type start_t, end_t;
    iter(stream, &start_t, &end_t);
    auto total_sec = elapsed_sec(start_t, end_t);
    if (total_sec > GPU_CLOCK_MIN_SAMPLE_TIME_SEC) {
      std::cout << "> Using sleep_cycles=" << _sleep_cycles << ", which takes " << total_sec << " seconds" << std::endl;
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
      DBG_LOG("Parent saw {} @ {}", *_sync_block, reinterpret_cast<void *>(_sync_block));
    }
  }

  if (args.FLAGS_num_threads.get() == 1) {
    // Just one thread launches kernels (default).
    // Run in same thread.
    GPUKernelRunner gpu_kernel_runner(
        _freq,
        args);
    gpu_kernel_runner.run();
    return;
  }

  if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
    DBG_LOG("ThreadedGPUKernelRunner launching kernels from {} threads", args.FLAGS_num_threads.get());
  }

  assert(!args.FLAGS_internal_is_child.get());

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
    gpu_kernel_runners.emplace_back(
        _freq,
        child_args);
  }
  int i;

  i = 0;
  for (auto& gpu_kernel_runner : gpu_kernel_runners) {
    if (!args.FLAGS_processes.get()) {
      gpu_kernel_runner.run_async_thread();
    } else {
      gpu_kernel_runner.run_async_process(i);
    }
    i++;
  }

  // Wait for threads to setup CUDA context.
  _sync_block->barrier.arrive_and_wait();
  auto start_kernels_t = time_now();

  // Wait for threads finish executing GPU kernels.to setup CUDA context.
  _sync_block->barrier.arrive_and_wait();
  auto done_kernels_t = time_now();

  i = 0;
  for (auto& gpu_kernel_runner : gpu_kernel_runners) {
    if (!args.FLAGS_processes.get()) {
      gpu_kernel_runner.wait_thread();
    } else {
      gpu_kernel_runner.wait_process(i);
    }
    i++;
  }

  double seconds = std::chrono::duration_cast<std::chrono::microseconds>(done_kernels_t - start_kernels_t).count() / static_cast<double>(MICROSECONDS_IN_SECOND);
  DBG_LOG("Running kernels took {} seconds", seconds);

  if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
    DBG_LOG("After children done, Parent sees: {}", *_sync_block);
  }

}

SyncBlock::~SyncBlock() {
  DBG_LOG("Destructor for {}", *this);
}

void GPUKernelRunner::run_async_thread() {
  assert(_async_thread == nullptr);
  _async_thread.reset(new std::thread([this] {
    this->run();
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

void GPUKernelRunner::run() {

  // NOTE: this is NOT a member variable, since we want it to be destructed within the child thread, NOT the parent.
  // That's because the CUDA context is bound to this thread.
  std::unique_ptr<RunContext> _run_ctx;

  SharedMem shared_mem;
  SyncBlock* sync_block = nullptr;
  DBG_LOG("args.FLAGS_internal_is_child = {}", args.FLAGS_internal_is_child.get());
  if (args.FLAGS_internal_is_child.get()) {
    shared_mem = SharedMem::Child(SHARED_MEM_NAME);
    sync_block = SyncBlock::Child(&shared_mem, SYNC_BLOCK_NAME);
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_SYNC)) {
      DBG_LOG("Child saw {} @ {}", *sync_block, reinterpret_cast<void*>(sync_block));
    }
  }

  _run_ctx.reset(new RunContext(args.FLAGS_cuda_context.get()));

  if (args.FLAGS_internal_is_child.get()) {
    // Barrier synchronization to wait for ALL threads to finish creating CUDA context.
    //
    // This is to ensure a fair comparison of "how long" it takes multi-process/multi-thread/multi-context
    // to launch --n_launches kernels, WITHOUT considering setup cost.
    sync_block->barrier.arrive_and_wait();
  }

  // Launch a kernel that runs for --kernel_duration_us microseconds.
  // Launch the kernel every --kernel_delay_us microseconds.
  time_type start_t = time_now();

//  assert(args.FLAGS_kernel_duration_us.get() % MICROSECONDS_IN_SECOND == 0);

  struct GPUKernelRun {
    int64_t time_sleeping_us;
    int64_t after_delay_us;
  };
  std::list<GPUKernelRun> kernel_runs;

  TimeHistogram hist("cudaLaunchKernel");

  int64_t launches = 0;
  while (true) {
    time_type now_t = time_now();
    auto time_sec = elapsed_sec(start_t, now_t);
    if (
//        time_sec >= args.FLAGS_run_sec.get() ||
        launches >= args.FLAGS_n_launches.get()) {
      // Q: Should we wait for the stream to finish here?
      // Otherwise, cudaStreamDestroy will free the stream in ~CudaStreamWrapper without waiting for the stream to finish;
      // I suspect that could result in traces NOT being collected for remaining kernels (unsure...)
      break;
    }
    GPUKernelRun kernel_run;
    // NOTE: this WILL wait for the kernel to finish first using cudaDeviceSynchronize()....
    // Alternatives:
    // - wait for just the launched kernel to finish using cudaEvent => multi-threaded friendly
    // - Q: Do we want to launch another kernel BEFORE the last one finishes...?
    //   ... if waiting for it to finish takes a long time (> 5 us)... then yes?
    auto before_sleep_t = time_now();
    if (args.FLAGS_sync.get()) {
      _freq.gpu_sleep_us_sync(*_run_ctx->_stream, args.FLAGS_kernel_duration_us.get());
    } else {
      _freq.gpu_sleep_us(*_run_ctx->_stream, args.FLAGS_kernel_duration_us.get());
    }
    launches += 1;
    auto after_sleep_t = time_now();
    auto nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(after_sleep_t - before_sleep_t).count();
    hist.CountNanoseconds(nanosec);
    auto time_sleeping_sec = elapsed_sec(before_sleep_t, after_sleep_t);

    int64_t time_sleeping_us = time_sleeping_sec * MICROSECONDS_IN_SECOND;
    kernel_run.time_sleeping_us = time_sleeping_us;
//    if (SHOULD_DEBUG(FEATURE_GPU_CLOCK_FREQ)) {
//      auto off_by = args.FLAGS_kernel_duration_us.get() - time_sleeping_us;
//      DBG_LOG("GPU.sleep for {} us (off by {} us)", time_sleeping_us, off_by);
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
//      DBG_LOG("CPU.delay for {} us (off by {} us)", after_delay_us, off_by);
//    }
    if (args.FLAGS_debug.get()) {
      kernel_runs.push_back(std::move(kernel_run));
    }
  }

  if (args.FLAGS_internal_is_child.get()) {
    // Barrier synchronization to wait for ALL GPU kernels to finish executing across all threads.
    //
    // This is to ensure a fair comparison of "how long" it takes multi-process/multi-thread/multi-context
    // to launch --n_launches kernels, WITHOUT considering setup cost.
    sync_block->barrier.arrive_and_wait();
  }

  if (args.FLAGS_debug.get()) {
    std::stringstream ss;
    size_t i = 0;
    int indent = 0;
    PrintIndent(ss, indent);
    ss << "GPUKernelRuns: size = " << kernel_runs.size() << "\n";
    for (auto const& kernel_run : kernel_runs) {
      PrintIndent(ss, indent + 1);
      ss << "[" << i << "]\n";

      PrintIndent(ss, indent + 2);
      int64_t off_by_time_sleeping_us = args.FLAGS_kernel_duration_us.get() - kernel_run.time_sleeping_us;
      ss << "GPU.sleep for " << kernel_run.time_sleeping_us << " us (off by " << off_by_time_sleeping_us << " us)\n";

      PrintIndent(ss, indent + 2);
      int64_t off_by_after_delay_us = args.FLAGS_kernel_delay_us.get() - kernel_run.after_delay_us;
      ss << "CPU.delay for " << kernel_run.after_delay_us << " us (off by " << off_by_after_delay_us << " us)\n";

      i += 1;
    }
    DBG_LOG("{}", ss.str());
  }

  {
    std::stringstream ss;
    ss << hist;
    DBG_LOG("{}", ss.str());
  }

}

CudaContext::CudaContext() : _context(new CudaContextWrapper()) {
}
CUcontext CudaContext::get() const {
  return _context->_handle;
}
void CudaContext::synchronize() {
  return _context->synchronize();
}

void CudaContextWrapper::synchronize() {
  cudaError_t ret;
  ret = cudaDeviceSynchronize();
  CHECK_CUDA(ret);
}
CudaContextWrapper::CudaContextWrapper() :
    _handle(nullptr),
    _flags(0),
    _dev(0)
{
  CUresult result;
  result = cuCtxCreate(&_handle, _flags, _dev);
  CHECK_CUDA_DRIVER(result);
  if (SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)) {
    std::stringstream ss;
//  DBG_LOG("Create CUDA context = {}", reinterpret_cast<void*>(_handle));
    backward::StackTrace st;
    st.load_here(32);
    backward::Printer p;
    ss << "Create CUDA context = " << reinterpret_cast<void*>(_handle) << "\n";
    p.print(st, ss);
    DBG_LOG("{}", ss.str());
  }
  assert(_handle != nullptr);
}
CudaContextWrapper::~CudaContextWrapper() {
  if (_handle) {
    // Wait for remaining kernels on all streams for the context to complete.
    // (driver API assumes you have done this before destroying context.)
    CUresult result;
    this->synchronize();
    // NOTE: I've seen this segfault _sometimes_ for multi-context runs... not sure why.
    if (SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)) {
      DBG_LOG("Destroy cuda context @ handle={}", reinterpret_cast<void*>(_handle));
    }
    result = cuCtxDestroy(_handle);
    CHECK_CUDA_DRIVER(result);
    _handle = nullptr;
  }
}

CudaStream::CudaStream() : _stream(new CudaStreamWrapper()) {
}
cudaStream_t CudaStream::get() const {
  return _stream->_handle;
}
void CudaStream::synchronize() {
  return _stream->synchronize();
}

void CudaStreamWrapper::synchronize() {
  cudaError_t ret;
  ret = cudaStreamSynchronize(_handle);
  CHECK_CUDA(ret);
}
CudaStreamWrapper::CudaStreamWrapper() :
    _handle(nullptr)
{
  cudaError_t ret;
  ret = cudaStreamCreateWithFlags(&_handle, cudaStreamNonBlocking);
//  ret = cudaStreamCreate(&_handle);
  CHECK_CUDA(ret);
  if (SHOULD_DEBUG(FEATURE_GPU_UTIL_CUDA_CONTEXT)) {
    DBG_LOG("Create CUDA stream = {}", reinterpret_cast<void *>(_handle));
  }
  assert(_handle != nullptr);
}
CudaStreamWrapper::~CudaStreamWrapper() {
  cudaError_t ret;
  if (_handle) {
    // Wait for remaining kernels on stream to complete.
    this->synchronize();
    ret = cudaStreamDestroy(_handle);
    CHECK_CUDA(ret);
    _handle = nullptr;
  }
}

} // namespace tensorflow
