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

__global__ void _compute_kernel(size_t iterations, int64_t *output) {
  for (size_t i = 0; i < iterations; i++) {
    *output = *output + 1;
  }
}

void GPUSleeper::gpu_sleep_cycles(CudaStream stream, clock_value_t sleep_cycles, bool sync) {
  cudaError_t err;
  _gpu_sleep<<<1, 1, 0, stream.get()>>>(sleep_cycles, _output.get()); // This line alone is 0.208557334
  if (sync) {
    err = cudaStreamSynchronize(stream.get());
    CHECK_CUDA(err);
  }
}

void GPUSleeper::gpu_sleep_cycles_sync(CudaStream stream, clock_value_t sleep_cycles) {
  gpu_sleep_cycles(stream, sleep_cycles, /*sync=*/true);
}

void GPUClockFreq::gpu_sleep_sec(CudaStream stream, double seconds) {
// cycles = cycles/sec * seconds
  clock_value_t cycles = ceil(_avg_mhz * seconds);
//  std::cout << "> Sleeping on GPU for " << seconds << " seconds (" << cycles << " cycles @ " << _clock_result.avg_mhz << " MHz)." << std::endl;
//  auto start_t = GPUClockFreq::now();
  _gpu_sleeper.gpu_sleep_cycles(stream, cycles, /*sync=*/false);
}

void GPUClockFreq::_gpu_sleep_us(CudaStream stream, int64_t usec, bool sync) {
// cycles = cycles/sec * seconds
// cycles/usec
  clock_value_t cycles = ceil((_avg_mhz / ((double)MICROSECONDS_IN_SECOND)) * usec);
//  std::cout << "> Sleeping on GPU for " << seconds << " seconds (" << cycles << " cycles @ " << _clock_result.avg_mhz << " MHz)." << std::endl;
//  auto start_t = GPUClockFreq::now();
  _gpu_sleeper.gpu_sleep_cycles(stream, cycles, /*sync=*/sync);
}

void GPUClockFreq::gpu_sleep_us(CudaStream stream, int64_t usec) {
  _gpu_sleep_us(stream, usec, /*sync=*/false);
}
void GPUClockFreq::gpu_sleep_us_sync(CudaStream stream, int64_t usec) {
  _gpu_sleep_us(stream, usec, /*sync=*/true);
}

void GPUComputeKernel::_gpu_compute_kernel(CudaStream stream, bool sync) {
  cudaError_t err;
  _compute_kernel<<<1, 1, 0, stream.get()>>>(args.FLAGS_kern_arg_iterations.get(), run_ctx->output.get()); // This line alone is 0.208557334
  if (sync) {
    err = cudaStreamSynchronize(stream.get());
    CHECK_CUDA(err);
  }
}

} // namespace tensorflow
