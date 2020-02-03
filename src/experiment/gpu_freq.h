//
// Created by jgleeson on 2020-01-23.
//

#ifndef IML_GPU_FREQ_H
#define IML_GPU_FREQ_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include "common/my_status.h"

//#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (10)
#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (5)
#define GPU_CLOCK_REPETITIONS (3)
#define GPU_CLOCK_INIT_GPU_SLEEP_CYCLES (1000)
#define GPU_CLOCK_MIN_GUESS_TIME_SEC (1)

#define MICROSECONDS_IN_SECOND (1000000)

//std::stringstream _ss;
//_ss << __FILE__ << ":" << __LINE__ << " @ " << __func__ << ": CUDA Failed with (err=" << err << "): " << err_str;
//DBG_LOG("{}", _ss.str());

#define CHECK_CUDA(err) ({ \
  if (err != cudaSuccess) { \
    auto err_str = cudaGetErrorString(err); \
    std::cout << __FILE__ << ":" << __LINE__ << " @ " << __func__ << ": CUDA Failed with (err=" << err << "): " << err_str << std::endl; \
    assert(err == cudaSuccess); \
  } \
})

namespace tensorflow {

template <class Container>
double Average(Container& buffer) {
  double summation = 0;
  int n = 0;
  for (auto x : buffer) {
    summation += x;
    n += 1;
  }
  return summation / ((double) n);
}

template <class Container>
double Std(Container& buffer) {
  auto avg = Average(buffer);
  double summation = 0;
  int n = 0;
  for (auto x : buffer) {
    summation += pow(x - avg, 2);
    n += 1;
  }
  return sqrt(summation / ((double) n));
}

using clock_value_t = long long;
// https://www.softwariness.com/articles/monotonic-clocks-windows-and-posix/
//  using std::chrono::steady_clock;
using time_type = std::chrono::steady_clock::time_point;

double elapsed_sec(time_type start, time_type stop);

time_type time_now();

struct CudaStreamWrapper {
  cudaStream_t _handle;
  CudaStreamWrapper();
  ~CudaStreamWrapper();
};

class CudaStream {
public:
  // We want to inherit shared_ptr copy/move constructor functionality, but we need new/delete
  // to create a cudaStream_t using cudaStreamCreate/cudaStreamDestroy.
  std::shared_ptr<CudaStreamWrapper> _stream;
//  cudaStream_t _stream;
  CudaStream();
  cudaStream_t get() const;
};

template <typename T>
struct CudaHostBufferWrapper {
  T* _handle;
  size_t _n_elems;

  CudaHostBufferWrapper(size_t n_elems) :
      _handle(nullptr),
      _n_elems(n_elems) {
    cudaError_t ret;
    ret = cudaMallocHost(&_handle, size_bytes());
    CHECK_CUDA(ret);
    assert(_handle != nullptr);
  }
  size_t size_bytes() const {
    return _n_elems*sizeof(T);
  }
  ~CudaHostBufferWrapper() {
    cudaError_t ret;
    if (_handle) {
      ret = cudaFreeHost(_handle);
      CHECK_CUDA(ret);
      _handle = nullptr;
    }
  }
};
template <typename T>
class CudaHostBuffer {
public:
  // We want to inherit shared_ptr copy/move constructor functionality, but we need new/delete
  // to create a cudaStream_t using cudaStreamCreate/cudaStreamDestroy.
  std::shared_ptr<CudaHostBufferWrapper<T>> _buffer;
  size_t _n_elems;
//  cudaStream_t _stream;

  CudaHostBuffer(size_t n_elems) :
      _buffer(new CudaHostBufferWrapper<T>(n_elems)),
      _n_elems(n_elems) {
  }
  T* get() const {
    return _buffer->_handle;
  }

};

//struct GPUClockResult {
//  double avg_mhz;
//  double std_mhz;
//};

// We want to expose a function for sleeping for a given number of cycles
// (used by GPUClockFreq).
// Sleeping for a given number of seconds requires GPUClockFreq's calculation of GPU clock rate.
class GPUSleeper {
public:
//  CudaStream _stream;
  CudaHostBuffer<int64_t> _output;

  GPUSleeper() :
      _output(1) {
  }

  void gpu_sleep_cycles(CudaStream stream, clock_value_t sleep_cycles);
  void gpu_sleep_cycles_sync(CudaStream stream, clock_value_t sleep_cycles);

};

class GPUClockFreq {
public:
  GPUSleeper _gpu_sleeper;

  std::vector<double> _time_secs;
  std::vector<double> _freq_mhz;
  double _avg_mhz;
  double _std_mhz;
//  std::vector<clock_value_t> time_cycles;
  // Number of cycles to sleep for on the GPU.
  clock_value_t _sleep_cycles;
  int _repetitions;
//  GPUClockResult _result;
  std::string _directory;

  GPUClockFreq(int repetitions, const std::string& directory) :
      _avg_mhz(0.)
      , _std_mhz(0.)
      , _sleep_cycles(GPU_CLOCK_INIT_GPU_SLEEP_CYCLES)
      , _repetitions(repetitions)
      , _directory(directory)
  {
  }

  void guess_cycles(CudaStream stream);

  void iter(CudaStream stream, time_type *start_t, time_type *end_t);

  double freq_mhz(double time_sec);

  void gpu_sleep_sec(CudaStream stream, double seconds);
  void gpu_sleep_us(CudaStream stream, int64_t usec);
  void run();

  MyStatus dump_json() const;
  MyStatus load_json(const std::string &path);
  std::string json_path() const;
  std::string json_basename() const;
};

class GPUKernelRunner {
public:
  CudaStream _stream;

  // "Time between kernel launches in microseconds"
  int64_t _kernel_delay_us;
  // "Duration of kernel in microseconds"
  int64_t _kernel_duration_us;
  // "How to long to run for (in seconds)"
  double _run_sec;

  GPUClockFreq _freq;
  std::string _directory;
  bool _debug;

  GPUKernelRunner(
      GPUClockFreq freq,
      int64_t kernel_delay_us,
      int64_t kernel_duration_us,
      double run_sec,
      const std::string& directory,
      bool debug) :
      _kernel_delay_us(kernel_delay_us),
      _kernel_duration_us(kernel_duration_us),
      _run_sec(run_sec),
      _freq(std::move(freq)),
      _directory(directory),
      _debug(debug)
  {
  }

  void DelayUs(int64_t usec);

  void run();

//  void guess_cycles();
//
//  static time_type now();
//
//  static double elapsed_sec(time_type start, time_type stop);
//
//  void iter(time_type *start_t, time_type *end_t);
//
//  double freq_mhz(double time_sec);
//
//  static double gpu_sleep(clock_value_t sleep_cycles);
//  void run();

//  MyStatus dump_json() const;
//  MyStatus load_json(const std::string &path);
//  std::string json_path() const;
//  std::string json_basename() const;
};


} // namespace tensorflow

#endif //IML_GPU_FREQ_H
