//
// Created by jgleeson on 2020-01-23.
//

// nvcc bugs: cannot import json.hpp without errors:
// https://github.com/nlohmann/json/issues/1347
#define RLS_IGNORE_JSON

#include <chrono>
#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cassert>
#include <sstream>


#include <inttypes.h>

#include <boost/filesystem.hpp>

#include "gpu_freq.h"
#include "gpu_freq.cuh"

//#include <nlohmann/json.hpp>
//using json = nlohmann::json;

#include "common_util.h"

namespace rlscope {

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

static __device__ __inline__ void record_sched_info_sample(
        GPUThreadSchedInfo *sched_info,
        uint64_t stream_id, uint64_t kernel_id, int warp_size, size_t n_samples,
        int n, size_t n_elems) {

    // Kind of like warp_id, but static under preemption:
    //   static_warp_id = threadIdx.x / warp_size

    uint32_t sm_id = __sm_id();
    uint32_t warp_id = __warp_id();
    uint32_t lane_id = __lane_id();
    uint64_t globaltimer_ns = __globaltimer_ns();
    if (lane_id == 0) {
        // sched_info[kernel_id][n][block_id][static_warp_id]
        //   Where:
        //     static_warp_id = 0..warps_per_block-1
        //     block_id = 0..num_blocks-1
        //     n = 0..n_samples-1
        //     kernel_id = 0..n_launches
        //
        // sched_info[kernel_id][n][block_id][static_warp_id]
        // ===
        // sched_info[
        //   static_warp_id
        //   + block_id * warps_per_block
        //   + n * num_blocks * warps_per_block
        //   + kernel_id * n_samples * num_blocks * warps_per_block
        // ]
        //
        // static_warp_id = threadIdx.x / warp_size
        // num_blocks = gridDim.x
        // threads_per_block = blockDim.x
        // block_id = blockIdx.x
        // warps_per_block = ceil(threads_per_block/warp_size)
        //                 = (blockDim.x + warp_size - 1)/warp_size
        int static_warp_id = threadIdx.x / warp_size;
        int warps_per_block = (blockDim.x + warp_size - 1) / warp_size;
        int num_blocks = gridDim.x;
        int block_id = blockIdx.x;
        int i =
                static_warp_id
                + block_id * warps_per_block
                + n * num_blocks * warps_per_block
                + kernel_id * n_samples * num_blocks * warps_per_block;
        assert(i < n_elems);

//    if (stream_id == 0) {
//      printf(
//          "static_warp_id = %i"
//          ", block_id = %i"
//          ", warps_per_block = %i"
//          ", n = %i"
//          ", num_blocks = %i"
//          ", kernel_id = %" PRIu64 ""
//          // ", n_samples = %zu"
//          ", n_samples = %lu"
//          ", i = %i"
//          ", globaltimer_ns = %llu"
//          "\n",
//          static_warp_id
//          , block_id
//          , warps_per_block
//          , n
//          , num_blocks
//          , kernel_id
//          , (unsigned long)n_samples
//          , i
//          , (unsigned long long)globaltimer_ns
//          );
//    }

        sched_info[i].stream_id = stream_id;
        sched_info[i].kernel_id = kernel_id;
        sched_info[i].sm_id = sm_id;
        sched_info[i].warp_id = warp_id;
        sched_info[i].lane_id = lane_id;
        sched_info[i].globaltimer_ns = globaltimer_ns;
    }
}

// https://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim
//   https://stackoverflow.com/a/16619578
// gridDim.x = number of thread blocks
// blockIdx.x = block id (0 ... num_thread_blocks - 1)
//                       (0 ... gridDim.x - 1)
// blockDim.x = number of threads in a block
// threadIdx.x = id of thread, within the thread block (0 ... threads_per_block - 1)
//                                                     (0 ... blockDim.x - 1)


// N_TOTAL_SAMPLES = ((NUM_THREAD_BLOCKS * THREADS_PER_BLOCK) / WARP_SIZE) * N_SAMPLES_PER_WARP * N_LAUNCHES
// n_total_samples = ((gridDim.x * blockDim.x) / warp_size) * n_samples * n_launches
// allocate sched_info = GPUThreadSchedInfo[N_TOTAL_SAMPLES]
// WANT: these to be contiguous


// < 0 .. warps_per_block >
// [                      ][                      ][                      ]
// <                        0 .. num_thread_blocks                        >
// sched_info[warps_per_block * block_id + static_warp_id]
//   warps_per_block = blockDim.x / WARP_SIZE
//   block_id = blockIdx.x
//   // kind of like warp_id, but static under preemption:
//   static_warp_id = threadIdx.x / WARP_SIZE
// if (lane_id == 0) {
//   sched_info[(blockDim.x / WARP_SIZE) * blockIdx.x + (threadIdx.x / WARP_SIZE)] = ...
// }

// sched_info[n

__global__ void _compute_sched_info_kernel(
        size_t iterations, int64_t *output,
        GPUThreadSchedInfo *sched_info, uint64_t stream_id, uint64_t kernel_id, int warp_size, size_t n_samples,
        size_t n_elems) {
    // int id = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: refactor into function to allow recording upto n samples.  Only record if lane_id == 0.
    // OR, record across all lanes, but don't record "warp_id"...
    //   Unclear to me if this will work... no risk of cache block size being committed without writes present?
    for (int n = 0; n < static_cast<int>(n_samples); n++) {
        // NOTE: ideally we would take samples whenever there's a scheduling change...
        // but we don't know when preemption happens.
        record_sched_info_sample(
                sched_info,
                stream_id, kernel_id, warp_size, n_samples,
                n, n_elems);
        for (size_t i = 0; i < iterations / n_samples; i++) {
            *output = *output + 1;
        }
    }
}


// deviceQuery output on my desktop workstation (gambit) which has a "GeForce RTX 2060 SUPER".
// We will use this deviceQuery output as an example for discussing sm_id/lane_id/warp_id.
//
//   ./deviceQuery Starting...
//
//    CUDA Device Query (Runtime API) version (CUDART static linking)
//
//   Detected 1 CUDA Capable device(s)
//
//   Device 0: "GeForce RTX 2060 SUPER"
//     CUDA Driver Version / Runtime Version          10.2 / 10.2
//     CUDA Capability Major/Minor version number:    7.5
//     Total amount of global memory:                 7979 MBytes (8366784512 bytes)
//     (34) Multiprocessors, ( 64) CUDA Cores/MP:     2176 CUDA Cores
//     GPU Max Clock rate:                            1650 MHz (1.65 GHz)
//     Memory Clock rate:                             7001 Mhz
//     Memory Bus Width:                              256-bit
//     L2 Cache Size:                                 4194304 bytes
//     Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
//     Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
//     Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
//     Total amount of constant memory:               65536 bytes
//     Total amount of shared memory per block:       49152 bytes
//     Total number of registers available per block: 65536
//     Warp size:                                     32
//     Maximum number of threads per multiprocessor:  1024
//     Maximum number of threads per block:           1024
//     Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
//     Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
//     Maximum memory pitch:                          2147483647 bytes
//     Texture alignment:                             512 bytes
//     Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
//     Run time limit on kernels:                     Yes
//     Integrated GPU sharing Host Memory:            No
//     Support host page-locked memory mapping:       Yes
//     Alignment requirement for Surfaces:            Yes
//     Device has ECC support:                        Disabled
//     Device supports Unified Addressing (UVA):      Yes
//     Device supports Compute Preemption:            Yes
//     Supports Cooperative Kernel Launch:            Yes
//     Supports MultiDevice Co-op Kernel Launch:      Yes
//     Device PCI Domain ID / Bus ID / location ID:   0 / 65 / 0
//     Compute Mode:
//        < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
//
// deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.2, CUDA Runtime Version = 10.2, NumDevs = 1
// Result = PASS


// We can query/probe low-level GPU-side scheduling decisions, and communicate them back to the CPU.
// https://stackoverflow.com/questions/28881491/how-can-i-find-out-which-thread-is-getting-executed-on-which-core-of-the-gpu

// sm_id
//   TLDR: sm_id = 0, 1, ..., 33 (34 SMs on GeForce RTX 2060 SUPER)
//   sm_id is volatile and will change over the kernel's execution lifetime during preemption.
//   Q: So, SM-level preemption means an entire thread block (i.e. MULTIPLE warps)
//   will migrate to another SM... how frequently does that happen?
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-smid
//     """
//     Description: A predefined, read-only special register that returns the processor (SM) identifier on which
//     a particular thread is executing. The SM identifier ranges from 0 to %nsmid-1.
//     The SM identifier numbering is not guaranteed to be contiguous.
//
//     Notes: Note that %smid is volatile and returns the location of a thread at the moment when read,
//     but its value may change during execution, e.g. due to rescheduling of threads following preemption.
//     %smid is intended mainly to enable profiling and diagnostic code to sample and log information such
//     as work place mapping and load distribution.
//     """
static __device__ __inline__ uint32_t __sm_id() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// Inline PTX assembly syntax documentation:
// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
//   "h" = .u16 reg
//   "r" = .u32 reg
//   "l" = .u64 reg
//   "f" = .f32 reg
//   "d" = .f64 reg

// warp_id
//   TLDR: for warp-size = 32, CUDA cores per SM = 64,
//   warps per SM = (CUDA cores per SM)/(warp size) = 64/32 = 2 warps per SM
//   So, warp_id = 0, 1
//   NOTE: warp_id is volatile and will change over execution of the warp due to preemption...
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-warpid
//     """
//     A predefined, read-only special register that returns the thread's warp identifier.
//     The warp identifier provides a unique warp number within a CTA but not across CTAs within a grid.
//     The warp identifier will be the same for all threads within a single warp.
//
//     Note that %warpid is volatile and returns the location of a thread at the moment when read, but
//     its value may change during execution, e.g., due to rescheduling of threads following preemption.
//     For this reason, %ctaid and %tid should be used to compute a virtual warp index if such a value
//     is needed in kernel code; %warpid is intended mainly to enable profiling and diagnostic code to
//     sample and log information such as work place mapping and load distribution.
//     A predefined, read-only special register that returns the thread's warp identifier.
//     """
static __device__ __inline__ uint32_t __warp_id() {
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}

// lane_id
//   TLDR: lane_id = 0, 1, ..., 31 for a warp-size of 32 (32 lanes); index of thread into its warp
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-laneid
//   """
//   A predefined, read-only special register that returns the thread's lane within the warp.
//   The lane identifier ranges from zero to WARP_SZ-1.
//   """
static __device__ __inline__ uint32_t __lane_id() {
    uint32_t laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

// globaltimer
//   TLDR: global nanosecond timer that can be used to establish total ordering of events on GPU.
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer
//   """
//   Description: Special registers intended for use by NVIDIA tools. The behavior is target-specific
//   and may change or be removed in future GPUs. When JIT-compiled to other targets, the value of
//   these registers is unspecified.
//   """

static __device__ __inline__ uint64_t __globaltimer_ns() {
    uint64_t globaltimer;
    // asm volatile("mov.u64 %0, %%globaltimer;" : "=r"(globaltimer));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(globaltimer));
    return globaltimer;
}

void GPUSleeper::gpu_sleep_cycles(CudaStream stream, clock_value_t sleep_cycles, bool sync) {
    cudaError_t err;
    _gpu_sleep<<<1, 1, 0, stream.get()>>>(sleep_cycles, _output.get()); // This line alone is 0.208557334
    if (sync) {
        RUNTIME_API_CALL_MAYBE_EXIT(cudaStreamSynchronize(stream.get()));
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
    clock_value_t cycles = ceil((_avg_mhz / ((double) MICROSECONDS_IN_SECOND)) * usec);
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
    _compute_kernel<<<1, 1, 0, stream.get()>>>(args.FLAGS_kern_arg_iterations.get(),
                                               run_ctx->output.get()); // This line alone is 0.208557334
    if (sync) {
        RUNTIME_API_CALL_MAYBE_EXIT(cudaStreamSynchronize(stream.get()));
    }
}

void GPUComputeSchedInfoKernel::_gpu_compute_kernel(CudaStream stream, bool sync) {
    cudaError_t err;
    auto stream_id = stream._stream_id;
    size_t n_samples = args.FLAGS_kern_arg_iterations.get() / args.FLAGS_kern_arg_iterations_per_sched_sample.get();
    _compute_sched_info_kernel<<<args.FLAGS_kern_arg_num_blocks.get(), args.FLAGS_kern_arg_threads_per_block.get(), 0, stream.get()>>>(
            args.FLAGS_kern_arg_iterations.get(), run_ctx->output.get(),
            run_ctx->sched_info.get(), stream_id, run_ctx->kernel_id, device_prop.warpSize, n_samples,
            run_ctx->sched_info.num_elems());
    assert(run_ctx->kernel_id < static_cast<uint64_t>(args.FLAGS_n_launches.get()));
    run_ctx->kernel_id += 1;
    if (sync) {
        RUNTIME_API_CALL_MAYBE_EXIT(cudaStreamSynchronize(stream.get()));
    }
}

MyStatus GPUComputeSchedInfoKernel::CheckArgs() {
    cudaError_t cuda_err;

    if (args.FLAGS_kern_arg_iterations.get() % args.FLAGS_kern_arg_iterations_per_sched_sample.get() != 0) {
        std::stringstream ss;
        ss << "ERROR: --kern_arg_iterations=" << args.FLAGS_kern_arg_iterations.get()
           << " must be evenly divisible by --kern_arg_iterations_per_sched_sample="
           << args.FLAGS_kern_arg_iterations_per_sched_sample.get();
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }

    // Moved to ::Init.
//  cuda_err = cudaGetDeviceProperties(&device_prop, args.FLAGS_device.get());
//  CHECK_CUDA(cuda_err);

//  printf("Device Number: %d\n", args.FLAGS_device.get());
//  printf("  Device name: %s\n", prop.name);
//  printf("  Memory Clock Rate (KHz): %d\n",
//         prop.memoryClockRate);
//  printf("  Memory Bus Width (bits): %d\n",
//         prop.memoryBusWidth);
//  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
//         2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);


    cudaFuncAttributes func_attrs;
    RUNTIME_API_CALL_MAYBE_EXIT(cudaFuncGetAttributes(&func_attrs, reinterpret_cast<const void *>(_compute_sched_info_kernel)));
    if (args.FLAGS_kern_arg_threads_per_block.get() > func_attrs.maxThreadsPerBlock) {
        std::stringstream ss;
        ss << "ERROR: --kern-arg-threads-per-block <= " << func_attrs.maxThreadsPerBlock
           << " for compute_sched_info_kernel";
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }

    // NOTE: there's no limit on the number of thread blocks that can run.

    // If we want to saturate the GPU with thread blocks then
    // --kern-arg-num-thread-blocks <= ( deviceProp.multiProcessorCount * deviceProp.multiProcessorCount ) / args.threads_per_block
    // NOTE: you CAN schedule more thread blocks than this...
    // they'll just get scheduled in increasing order, scheduling as many as will fit on the GPU at once.

    return MyStatus::OK();
}

} // namespace rlscope
