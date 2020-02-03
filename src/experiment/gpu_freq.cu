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

//#include "common/json.h"

//#include <nlohmann/json.hpp>
//using json = nlohmann::json;

#include "common/my_status.h"

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

void GPUSleeper::gpu_sleep_cycles(CudaStream stream, clock_value_t sleep_cycles) {

//  int64_t *output = nullptr;
//  cudaError_t err;
//  err = cudaMallocHost((void **) &output, sizeof(int64_t));
//  CHECK_CUDA(err);
//  *output = 0;

//  auto start_t = time_now();
  _gpu_sleep<<<1, 1, 0, stream.get()>>>(sleep_cycles, _output.get()); // This line alone is 0.208557334

//  err = cudaDeviceSynchronize();
//  CHECK_CUDA(err);

//  auto end_t = time_now(); // This whole block is 5.316381218, but we measure it using nvprof as 5.113029

//  err = cudaFreeHost(output);
//  CHECK_CUDA(err);

//  auto time_sec = elapsed_sec(start_t, end_t);
//  return time_sec;
}

void GPUSleeper::gpu_sleep_cycles_sync(CudaStream stream, clock_value_t sleep_cycles) {
  cudaError_t err;
  gpu_sleep_cycles(stream, sleep_cycles);
//  err = cudaDeviceSynchronize();
  err = cudaStreamSynchronize(stream.get());
  CHECK_CUDA(err);
}

void GPUClockFreq::gpu_sleep_sec(CudaStream stream, double seconds) {
// cycles = cycles/sec * seconds
  clock_value_t cycles = ceil(_avg_mhz * seconds);
//  std::cout << "> Sleeping on GPU for " << seconds << " seconds (" << cycles << " cycles @ " << _clock_result.avg_mhz << " MHz)." << std::endl;
//  auto start_t = GPUClockFreq::now();
  _gpu_sleeper.gpu_sleep_cycles(stream, cycles);
}

void GPUClockFreq::gpu_sleep_us(CudaStream stream, int64_t usec) {
// cycles = cycles/sec * seconds
// cycles/usec
  clock_value_t cycles = ceil((_avg_mhz / ((double)MICROSECONDS_IN_SECOND)) * usec);
//  std::cout << "> Sleeping on GPU for " << seconds << " seconds (" << cycles << " cycles @ " << _clock_result.avg_mhz << " MHz)." << std::endl;
//  auto start_t = GPUClockFreq::now();
  _gpu_sleeper.gpu_sleep_cycles(stream, cycles);
}

} // namespace tensorflow
