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

__global__ void _gpu_sleep(clock_value_t sleep_cycles, int64_t *output);

__global__ void _compute_kernel(size_t iterations, int64_t *output);

__global__ void _compute_sched_info_kernel(
    size_t iterations, int64_t *output,
    GPUThreadSchedInfo* sched_info, uint64_t stream_id, uint64_t kernel_id, int warp_size, size_t n_samples, size_t n_elems);


//https://stackoverflow.com/questions/28881491/how-can-i-find-out-which-thread-is-getting-executed-on-which-core-of-the-gpu
static __device__ __inline__ uint32_t __sm_id();
static __device__ __inline__ uint32_t __warp_id();
static __device__ __inline__ uint32_t __lane_id();
static __device__ __inline__ uint64_t __globaltimer_ns();

} // namespace tensorflow
