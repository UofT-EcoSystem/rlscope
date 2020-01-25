//
// Created by jgleeson on 2020-01-23.
//

#include <chrono>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cassert>
#include <sstream>

#include <boost/filesystem.hpp>

#include "experiment/gpu_freq.h"

//#include "analysis/json.h"

//#include <nlohmann/json.hpp>
//using json = nlohmann::json;

#include "analysis/my_status.h"

#define CHECK_CUDA(err) ({ \
  if (err != cudaSuccess) { \
    auto err_str = cudaGetErrorString(err); \
    std::cout << __FILE__ << ":" << __LINE__ << " @ " << __func__ << ": CUDA Failed with (err=" << err << "): " << err_str << std::endl; \
    assert(err == cudaSuccess); \
  } \
})

namespace tensorflow {

using clock_value_t = long long;

using steady_clock = std::chrono::steady_clock;

__global__ void _gpu_sleep(clock_value_t sleep_cycles, int64_t *output) {
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
    *output = *output + 1;
  } while (cycles_elapsed < sleep_cycles);
}

double GPUClockFreq::gpu_sleep(clock_value_t sleep_cycles) {
//  int64_t output = 0;
  int64_t *output = nullptr;
  cudaError_t err;
  err = cudaMallocHost((void **) &output, sizeof(int64_t));
  CHECK_CUDA(err);
  *output = 0;

  auto start_t = GPUClockFreq::now();
  _gpu_sleep << < 1, 1 >> > (sleep_cycles, output); // This line alone is 0.208557334
  err = cudaDeviceSynchronize();
  CHECK_CUDA(err);
  auto end_t = GPUClockFreq::now(); // This whole block is 5.316381218, but we measure it using nvprof as 5.113029

//  std::cout << "> gpu_sleep.output=" << *output << ", sleep_cycles=" << sleep_cycles << std::endl;

  err = cudaFreeHost(output);
  CHECK_CUDA(err);

  auto time_sec = GPUClockFreq::elapsed_sec(start_t, end_t);
  return time_sec;
}

} // namespace tensorflow
