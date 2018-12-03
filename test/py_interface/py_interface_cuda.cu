#include <chrono>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cassert>

#include "test/py_interface/py_interface_cuda.cuh"

#define CHECK_CUDA(err) ({ \
  if (err != cudaSuccess) { \
    auto err_str = cudaGetErrorString(err); \
    std::cout << __FILE__ << ":" << __LINE__ << " @ " << __func__ << ": CUDA Failed with (err=" << err << "): " << err_str << std::endl; \
    assert(err == cudaSuccess); \
  } \
})

using clock_value_t = long long;

using steady_clock = std::chrono::steady_clock;

__global__ void _gpu_sleep(clock_value_t sleep_cycles, int64_t* output)
{
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
    *output = *output + 1;
  }
  while (cycles_elapsed < sleep_cycles);
}

double GPUClockFreq::gpu_sleep(clock_value_t sleep_cycles) {
//  int64_t output = 0;
  int64_t* output = nullptr;
  cudaError_t err;
  err = cudaMallocHost((void**)&output, sizeof(int64_t));
  CHECK_CUDA(err);
  *output = 0;

  auto start_t = GPUClockFreq::now();
  _gpu_sleep<<<1, 1>>>(sleep_cycles, output); // This line alone is 0.208557334
  err = cudaDeviceSynchronize();
  CHECK_CUDA(err);
  auto end_t = GPUClockFreq::now(); // This whole block is 5.316381218, but we measure it using nvprof as 5.113029

//  std::cout << "> gpu_sleep.output=" << *output << ", sleep_cycles=" << sleep_cycles << std::endl;

  err = cudaFreeHost(output);
  CHECK_CUDA(err);

  auto time_sec = GPUClockFreq::elapsed_sec(start_t, end_t);
  return time_sec;
}

void GPUClockFreq::guess_cycles() {
  std::cout << "> Using initial sleep_cycles=" << _sleep_cycles << std::endl;
  while (true) {
    time_type start_t, end_t;
    iter(&start_t, &end_t);
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

GPUClockFreq::time_type GPUClockFreq::now() {
  time_type t = steady_clock::now();
  return t;
}

double GPUClockFreq::elapsed_sec(time_type start, time_type stop) {
  double sec = ((stop - start).count()) * steady_clock::period::num / static_cast<double>(steady_clock::period::den);
  return sec;
}

void GPUClockFreq::iter(time_type *start_t, time_type *end_t) {
  *start_t = now();
  GPUClockFreq::gpu_sleep(_sleep_cycles);
  *end_t = now();
}

double GPUClockFreq::freq_mhz(double time_sec) {
  return _sleep_cycles / time_sec;
}

GPUClockResult GPUClockFreq::run() {
  guess_cycles();
  time_type start_t, end_t;
  for (int r = 0; r < _repetitions; ++r) {
    iter(&start_t, &end_t);
    auto total_sec = elapsed_sec(start_t, end_t);
    _time_secs.push_back(total_sec);
    auto freq = freq_mhz(total_sec);
    _freq_mhz.push_back(freq);
    std::cout << "> freq[" << r << "] = " << freq << " MHz" << std::endl;
  }

  auto avg = Average(_freq_mhz);
  auto std = Std(_freq_mhz);
  _result = GPUClockResult{.avg_mhz = avg, .std_mhz = std};
  std::cout << "> Average freq = " << avg << " MHz" << std::endl;
  std::cout << "> Std freq = " << std << " MHz" << std::endl;

  return _result;
}

