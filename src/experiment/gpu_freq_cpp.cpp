//
// Created by jgleeson on 2020-01-23.
//

#include <chrono>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cassert>
#include <sstream>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "experiment/gpu_freq.h"

#include "common/json.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "common/my_status.h"

#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/debug_flags.h"


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
  boost::filesystem::path direc(_directory);
  boost::filesystem::path base = json_basename();
  return (direc / base).string();
}

void GPUClockFreq::run() {
  CudaStream stream;
  guess_cycles(stream);
  time_type start_t, end_t;
  for (int r = 0; r < _repetitions; ++r) {
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

void GPUKernelRunner::run() {
  // Launch a kernel that runs for --kernel_duration_us microseconds.
  // Launch the kernel every --kernel_delay_us microseconds.
  time_type start_t = time_now();

//  assert(_kernel_duration_us % MICROSECONDS_IN_SECOND == 0);

  struct GPUKernelRun {
    int64_t time_sleeping_us;
    int64_t after_delay_us;
  };
  std::list<GPUKernelRun> kernel_runs;

  while (true) {
    time_type now_t = time_now();
    auto time_sec = elapsed_sec(start_t, now_t);
    if (time_sec >= _run_sec) {
      break;
    }
    GPUKernelRun kernel_run;
    // NOTE: this WILL wait for the kernel to finish first using cudaDeviceSynchronize()....
    // Alternatives:
    // - wait for just the launched kernel to finish using cudaEvent => multi-threaded friendly
    // - Q: Do we want to launch another kernel BEFORE the last one finishes...?
    //   ... if waiting for it to finish takes a long time (> 5 us)... then yes?
    auto before_sleep_t = time_now();
    _freq.gpu_sleep_us(_stream, _kernel_duration_us);
    auto after_sleep_t = time_now();
    auto time_sleeping_sec = elapsed_sec(before_sleep_t, after_sleep_t);

    int64_t time_sleeping_us = time_sleeping_sec * MICROSECONDS_IN_SECOND;
    kernel_run.time_sleeping_us = time_sleeping_us;
//    if (SHOULD_DEBUG(FEATURE_GPU_CLOCK_FREQ)) {
//      auto off_by = _kernel_duration_us - time_sleeping_us;
//      DBG_LOG("GPU.sleep for {} us (off by {} us)", time_sleeping_us, off_by);
//    }
    auto before_delay_t = time_now();
    DelayUs(_kernel_delay_us);
//    int ret = usleep(_kernel_delay_us);
//    assert(ret == 0);
    auto after_delay_t = time_now();
    auto after_delay_sec = elapsed_sec(before_delay_t, after_delay_t);
    int64_t after_delay_us = after_delay_sec * MICROSECONDS_IN_SECOND;
    kernel_run.after_delay_us = after_delay_us;
//    if (SHOULD_DEBUG(FEATURE_GPU_CLOCK_FREQ)) {
//      auto off_by = _kernel_delay_us - after_delay_us;
//      DBG_LOG("CPU.delay for {} us (off by {} us)", after_delay_us, off_by);
//    }
    if (_debug) {
      kernel_runs.push_back(std::move(kernel_run));
    }
  }

  if (_debug) {
    std::stringstream ss;
    size_t i = 0;
    int indent = 0;
    PrintIndent(ss, indent);
    ss << "GPUKernelRuns: size = " << kernel_runs.size() << "\n";
    for (auto const& kernel_run : kernel_runs) {
      PrintIndent(ss, indent + 1);
      ss << "[" << i << "]\n";

      PrintIndent(ss, indent + 2);
      int64_t off_by_time_sleeping_us = _kernel_duration_us - kernel_run.time_sleeping_us;
      ss << "GPU.sleep for " << kernel_run.time_sleeping_us << " us (off by " << off_by_time_sleeping_us << " us)\n";

      PrintIndent(ss, indent + 2);
      int64_t off_by_after_delay_us = _kernel_delay_us - kernel_run.after_delay_us;
      ss << "CPU.delay for " << kernel_run.after_delay_us << " us (off by " << off_by_after_delay_us << " us)\n";

      i += 1;
    }
    DBG_LOG("{}", ss.str());
  }

}

CudaStream::CudaStream() : _stream(new CudaStreamWrapper()) {
}
cudaStream_t CudaStream::get() const {
  return _stream->_handle;
}

CudaStreamWrapper::CudaStreamWrapper() :
    _handle(nullptr)
{
  cudaError_t ret;
  ret = cudaStreamCreate(&_handle);
  CHECK_CUDA(ret);
  assert(_handle != nullptr);
}
CudaStreamWrapper::~CudaStreamWrapper() {
  cudaError_t ret;
  if (_handle) {
    ret = cudaStreamDestroy(_handle);
    CHECK_CUDA(ret);
    _handle = nullptr;
  }
}

} // namespace tensorflow
