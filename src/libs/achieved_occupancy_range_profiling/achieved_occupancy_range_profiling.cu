// nvcc bugs: cannot import json.hpp without errors:
// https://github.com/nlohmann/json/issues/1347
#define RLS_IGNORE_JSON

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
//#include <Metric.h>
//#include <Eval.h>
//#include <List.h>
//#include <FileOp.h>
#include "profilerhost_util.h"
#include "c_util/FileOp.h"
#include <cstring>
#include <assert.h>

#include <list>

#include <backward.hpp>

#include <gflags/gflags.h>

#include "common_util.h"

// DON'T do "using namespace" to avoid name clashes for things that are still duplicated in other sample programs
//using namespace rlscope;

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

// achieved_occupancy = sm__warps_active.avg.pct_of_peak_sustained_active+
// inst_executed = sm__inst_executed.sum+
// default metric used by userrange_profiling = smsp__warps_launched.avg+ (how many warps are launched on average)
#define DEFAULT_METRIC_NAME "sm__warps_active.avg.pct_of_peak_sustained_active+"

DEFINE_bool(debug, false, "Debug: give additional verbose output");
DEFINE_bool(disable_sampling, false, "Skip range profiling, just run GPU computation");
DEFINE_string(metric, DEFAULT_METRIC_NAME, "CUDA Profiling API Metric name");
DEFINE_int64(n_int32s, 50000, "Size of array for VecAdd/VecSub GPU calls");
DEFINE_int64(iterations, 20000, "number of times to call VecAdd kernel to add two int32 vectors");
DEFINE_int32(device, 0, "Device number");

DEFINE_int32(passes, 0, "How many CUPTI profiling passes to perform; required number of passes depends on number of \"ranges\" and number of metrics (and as a result, counters) being collected");
// How many unique paths from root node to leaf node in the CUPTI profiling annotation DAG formed from push()es.
DEFINE_int32(unique_paths, 1, "How many unique paths from root node to leaf node in the CUPTI profiling annotation DAG formed from push()es");
// How many push()es you do in a row before popping?
// I observe segfaults for >= 12... why?
// Looks like this directly determines how many "passes" need to be performed.
// CUDA sample program has this equal to numRanges (1).
DEFINE_int32(nesting_levels, 1, "How many CUPTI profiling push()es you do in a row before popping?");

// Actual number of unique ranges we observer at runtime.
// Q: What happens if this exceeds what we allocated for our counter-image data?
// i.e., RUNTIME_NUM_UNIQUE_RANGES > COUNTER_DATA_MAX_NUM_RANGES
// Then we simply DON'T collect all the counters; we only collect the first COUNTER_DATA_MAX_NUM_RANGES counters,
// and we only run for COUNTER_DATA_MAX_NUM_RANGES passes.
static int RUNTIME_NUM_UNIQUE_RANGES;

// IDEALLY, we would mark operations in RL script as "GPU operations" so that we can minimize the required "nesting depth" and
// "unique ranges" needed for collecting GPU profiling data.  Otherwise we end up inflating the counter-image size,
// and inflating the number of profiling passes.

// Upper bound on how many push()es you do in a row before popping.
// This directly determines how many "passes" are performed at runtime (but they may not be enough to collect all ranges if this is too low!).
// If it's too low:
//   We'll only collect ranges whose depth is <= this number.  I don't think it leads to segfault though (but it might...).
// If it's too high:
//   We'll inflate the number of profiling passes performed at runtime (=> more profiling overhead than maybe required).
// Interestingly, this DOESN'T affect the size of the output.
//
// Q: How to detect if too low?
// Count number of unique stacks from push() calls at runtime.
static int COUNTER_DATA_MAX_NUM_NESTING_LEVELS;

// How many ranges to allocate for in counter-data image.
// This is the number of "unique stacks".
// We're using this for:
//    counterDataImageOptions.maxNumRanges
//    counterDataImageOptions.maxNumRangeTreeNodes
//    beginSessionParams.maxRangesPerPass
//    beginSessionParams.maxLaunchesPerPass
// If it's too low, sometimes we will simply be missing some "stacks" in our profiling output; however it can also lead to a segfault
// (I saw this when experimenting with runtime nesting depth of 10, with 10 different "paths" of that depth).
// So, safest thing to do is to set this pretty high.
//   Q: How to detect if too low?
//   Count number of unique stacks from push() calls at runtime.
// If it's too high, then the size of the counter-data file will be inflated without storing anything useful.
//   Q: How to detect if too high?
//   Count number of unique stacks from push() calls at runtime.
// IDEALLY: postprocessing should "shrink" the counter-data file...then we could pre-allocate enough space ahead-of-time through a configuration param.
// Q: How to do this?
//static int COUNTER_DATA_MAX_NUM_RANGES = 10;
//static int COUNTER_DATA_MAX_NUM_RANGES = 20; // 800, 94K (scratch, counter-data)
//static int COUNTER_DATA_MAX_NUM_RANGES = 30;
//static int COUNTER_DATA_MAX_NUM_RANGES = 100; // 4K, 462K (scratch, counter-data)
//static int COUNTER_DATA_MAX_NUM_RANGES = 1000;
// IDEAL setting if we know the workload ahead-of-time.
//static int COUNTER_DATA_MAX_NUM_RANGES = RUNTIME_NUM_UNIQUE_RANGES; // 7.9K, 923K
static int COUNTER_DATA_MAX_NUM_RANGES;

bool CreateCounterDataImage(
    std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer,
    std::vector<uint8_t>& counterDataImagePrefix)
{

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    // Q: Does a single range get profiled in each pass...?
    counterDataImageOptions.maxNumRanges = COUNTER_DATA_MAX_NUM_RANGES;
    counterDataImageOptions.maxNumRangeTreeNodes = COUNTER_DATA_MAX_NUM_RANGES;
    counterDataImageOptions.maxRangeNameLength = 64;
    {
        std::stringstream ss;
        ss << "Runtime info: "
           << std::endl
           << "  counterDataImageOptions.maxNumRanges = " << counterDataImageOptions.maxNumRanges
           << std::endl
           << "  counterDataImageOptions.maxNumRangeTreeNodes = " << counterDataImageOptions.maxNumRangeTreeNodes
           << std::endl
           << "  counterDataImageOptions.maxRangeNameLength = " << counterDataImageOptions.maxRangeNameLength
           ;
        RLS_LOG("GPU_HW", "{}", ss.str());
    }


    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}

void run_gpu_compute() {
//    rlscope::ComputeVectorAddSubtract(FLAGS_n_int32s);
    auto start_compute_t = rlscope::get_timestamp_us();
    CuptiSamples::ComputeVecAdd(FLAGS_iterations, FLAGS_n_int32s);
    cudaDeviceSynchronize();
    auto end_compute_t = rlscope::get_timestamp_us();

    float compute_sec = (end_compute_t - start_compute_t).count()/FLOAT_MICROSECONDS_IN_SEC;
    // std::cout << "ComputeVecAdd(iterations=" << FLAGS_iterations << ", n=" << FLAGS_n_int32s << ") took " << compute_sec << " seconds" << std::endl;
}

using RangeFunc = std::function<void()>;
static void with_range(const std::string& range_name, RangeFunc func) {
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
    pushRangeParams.pRangeName = range_name.c_str();
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPushRange(&pushRangeParams));
    {
        func();
    }
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPopRange(&popRangeParams));
}


using RangeFunc = std::function<void()>;
static void with_N_ranges(int N, const std::string& range_name, RangeFunc func) {
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};

    // With nesting. Only compute in the last nest-level.
    for (int i = 0; i < N; i++) {
        std::stringstream ss;
        ss << range_name << "_" << i;
        auto range_name_i = ss.str();
        pushRangeParams.pRangeName = range_name_i.c_str();
        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPushRange(&pushRangeParams));
    }
    func();
    for (int i = 0; i < N; i++) {
        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPopRange(&popRangeParams));
    }

//    // With nesting. More compute with each added nest-level.
//    //
//    //   rangeName: userrange_0  metricName: smsp__inst_executed.sum     gpuValue: 1.87572e+11
//    //   rangeName: userrange_0/userrange_1      metricName: smsp__inst_executed.sum     gpuValue: 9.3786e+10
//    //
//    for (int i = 0; i < N; i++) {
//        std::stringstream ss;
//        ss << range_name << "_" << i;
//        auto range_name_i = ss.str();
//        pushRangeParams.pRangeName = range_name_i.c_str();
//        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPushRange(&pushRangeParams));
//        func();
//    }
//    // for (int i = N-1; i >= 0; i--) {
//    for (int i = 0; i < N; i++) {
//        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPopRange(&popRangeParams));
//    }

    // Without nesting.
//    for (int i = 0; i < N; i++) {
//        std::stringstream ss;
//        ss << range_name << "_" << i;
//        auto range_name_i = ss.str();
//        pushRangeParams.pRangeName = range_name_i.c_str();
//        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPushRange(&pushRangeParams));
//        func();
//        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPopRange(&popRangeParams));
//    }

}

size_t getNumRangeCollected(std::vector<uint8_t>& counterDataImage) {
    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = &counterDataImage[0];
    NVPW_API_CALL_MAYBE_EXIT(NVPW_CounterData_GetNumRanges(&getNumRangesParams));
    return getNumRangesParams.numRanges;
}


static size_t CUR_NESTING_LEVEL = 0;
static size_t MAX_NESTING_LEVEL = 0;
template <typename Func>
void ScopedOperation(const std::string& operation, Func func) {
    CUptiResult status;
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    pushRangeParams.pRangeName = operation.c_str();
    {
        std::stringstream ss;
        rlscope::log_func_call_impl(ss, "cuptiProfilerPushRange", operation);
        RLS_LOG("CUDA_API_TRACE", "{}", ss.str());
    }
    CUPTI_API_CALL_MAYBE_EXIT_SILENT(cuptiProfilerPushRange(&pushRangeParams));
    CUR_NESTING_LEVEL += 1;
    MAX_NESTING_LEVEL = std::max(CUR_NESTING_LEVEL, MAX_NESTING_LEVEL);

    if (CUR_NESTING_LEVEL > static_cast<size_t>(COUNTER_DATA_MAX_NUM_NESTING_LEVELS)) {
        std::stringstream ss;
        ss << "WARNING: The number of consecutive cuptiProfilerPushRange calls (" << CUR_NESTING_LEVEL << ") has exceeded setConfigParams.numNestingLevels = " << COUNTER_DATA_MAX_NUM_NESTING_LEVELS;
        RLS_LOG("GPU_HW", "{}", ss.str());
    }

    func();

    CUR_NESTING_LEVEL -= 1;

    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerPopRange(&popRangeParams));
}

// Q: if we visit the same path multiple times, should it be double-counted...?

// WITHOUT double-counting ("unique" paths):
//   [training_loop]
//   [training_loop, q_forward]
//   [training_loop, step]
//   [training_loop, q_backward]
//   [training_loop, q_update_target_network]
// This causes SEGFAULT.
// #define DQN_ATARI_UNIQUE_PATHS (5)

// WITH double-counting.
#define DQN_ATARI_UNIQUE_PATHS ((3*3)+(1*4)+(3*3)+(1*5))

#define DQN_ATARI_NESTING_LEVELS (2)
void run_pass_dqn_atari() {

    // num_paths = 3 * 3
    for (size_t i = 0; i < 3; i++) {
        ScopedOperation("training_loop", [&] {

            ScopedOperation("q_forward", [&] {
                run_gpu_compute();
            });

            ScopedOperation("step", [&] {
            });

        });
    }

    // num_paths = 1 * 4
    for (size_t i = 0; i < 1; i++) {
        ScopedOperation("training_loop", [&] {

            ScopedOperation("q_forward", [&] {
                run_gpu_compute();
            });

            ScopedOperation("step", [&] {
            });

            ScopedOperation("q_backward", [&] {
                run_gpu_compute();
            });

        });
    }

    // num_paths = 3 * 3
    for (size_t i = 0; i < 3; i++) {
        ScopedOperation("training_loop", [&] {

            ScopedOperation("q_forward", [&] {
                run_gpu_compute();
            });

            ScopedOperation("step", [&] {
            });

        });
    }

    // num_paths = 1 * 5
    for (size_t i = 0; i < 1; i++) {
        ScopedOperation("training_loop", [&] {

            ScopedOperation("q_forward", [&] {
                run_gpu_compute();
            });

            ScopedOperation("step", [&] {
            });

            ScopedOperation("q_backward", [&] {
                run_gpu_compute();
            });

            ScopedOperation("q_update_target_network", [&] {
                run_gpu_compute();
            });

        });
    }

}

void run_pass() {
    for (int32_t userrange_i = 0; userrange_i < FLAGS_unique_paths; userrange_i++) {
        std::stringstream range_ss;
        range_ss << "userrange_path";
        range_ss << userrange_i;
        with_N_ranges(FLAGS_nesting_levels, range_ss.str(), [&] {
            run_gpu_compute();
        });
    }
}

bool runTest(CUdevice cuDevice,
             std::vector<uint8_t>& configImage,
             std::vector<uint8_t>& counterDataScratchBuffer,
             std::vector<uint8_t>& counterDataImage,
             CUpti_ProfilerReplayMode profilerReplayMode,
             CUpti_ProfilerRange profilerRange)
{

    CUpti_Profiler_BeginSession_Params beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};

    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    beginSessionParams.replayMode = profilerReplayMode;
    beginSessionParams.maxRangesPerPass = COUNTER_DATA_MAX_NUM_RANGES;
    // Q: Does this matter...?  It's hard to know how many kernels might be launched.
    beginSessionParams.maxLaunchesPerPass = COUNTER_DATA_MAX_NUM_RANGES;

    {
        std::stringstream ss;
        ss << "Runtime info: "
           << std::endl
           << "  beginSessionParams.maxRangesPerPass = " << beginSessionParams.maxRangesPerPass
           << std::endl
           << "  beginSessionParams.maxLaunchesPerPass = " << beginSessionParams.maxLaunchesPerPass
           ;
        RLS_LOG("GPU_HW", "{}", ss.str());
    }

    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerBeginSession(&beginSessionParams));

    setConfigParams.pConfig = &configImage[0];
    setConfigParams.configSize = configImage.size();

    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = COUNTER_DATA_MAX_NUM_NESTING_LEVELS;
    // assert(COUNTER_DATA_MAX_NUM_NESTING_LEVELS * FLAGS_unique_paths <= COUNTER_DATA_MAX_NUM_RANGES);

    {
        std::stringstream ss;
        ss << "Using setConfigParams.numNestingLevels = " << setConfigParams.numNestingLevels;
        RLS_LOG("GPU_HW", "{}", ss.str());
    }
    // When experimenting with achieved_occupancy_range_profiling, I observed segfaults with config_data.counter_data_max_num_nesting_levels >= 12...
    // I have NO IDEA why...
    // TODO: post on nvidia forums and ask what's going on (reference ORIGINAL CUPTI sample program and how to reproduce the problem).
    assert(setConfigParams.numNestingLevels < 12);

    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerSetConfig(&setConfigParams));
    // DBG_BREAKPOINT("setConfigParams");
    /* User takes the resposiblity of replaying the kernel launches */
    CUpti_Profiler_BeginPass_Params beginPassParams = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
    CUpti_Profiler_EndPass_Params endPassParams = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
    int32_t pass_idx = 0;
    bool warned_passes = false;

    size_t num_ranges_collected = 0;
    size_t last_num_ranges_collected = 0;
    while (true) {
        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerBeginPass(&beginPassParams));
        {
            CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerEnableProfiling(&enableProfilingParams));

            // run_pass();
            run_pass_dqn_atari();

            pass_idx += 1;
            num_ranges_collected = getNumRangeCollected(counterDataImage);

            std::cout << "After pass " << pass_idx << ":" << std::endl
                    << "  total_num_ranges_collected = " << num_ranges_collected << std::endl
                    << "  increase in total_num_ranges_collected since last pass = " << (num_ranges_collected - last_num_ranges_collected) << std::endl;

            CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerDisableProfiling(&disableProfilingParams));
        }
        CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerEndPass(&endPassParams));

        last_num_ranges_collected = num_ranges_collected;

        if (FLAGS_passes != 0) {
            if (pass_idx >= FLAGS_passes) {
                if (!endPassParams.allPassesSubmitted) {
                    // NOTE: if we continue to do additional passes, endPassParams.allPassesSubmitted becomes false again.
                    // In order to have endPassParams.allPassesSubmitted become true again, we must complete another numRanges passes.
                    std::cout << "Stopping profiling with NOT allPassesSubmitted and --num_passes=" << pass_idx << std::endl;
                } else {
                    std::cout << "Stopping profiling with allPassesSubmitted and --num_passes=" << pass_idx << std::endl;
                }
                break;
            }
            if (!warned_passes && endPassParams.allPassesSubmitted) {
                std::cout << "CUPTI signalled enough passes collected after pass " << pass_idx << ", but continuing more passes since --num_passes=" << FLAGS_passes << std::endl;
                warned_passes = true;
            }
        } else if (endPassParams.allPassesSubmitted) {
            std::cout << "Stopping profiling; CUPTI signalled that it has collected enough passes after pass " << pass_idx << std::endl;
            break;
        }

    } // while
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerEndSession(&endSessionParams));

    return true;
}

void run_profiled(
        CUdevice cuDevice,
        const std::vector<std::string>& metricNames) {
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;
    std::string CounterDataFileName("SimpleCupti.counterdata");
    std::string CounterDataSBFileName("SimpleCupti.counterdataSB");
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;
    CUpti_ProfilerRange profilerRange = CUPTI_UserRange;

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerInitialize(&profilerInitializeParams));
    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = FLAGS_device;
    CUPTI_API_CALL_MAYBE_EXIT(cuptiDeviceGetChipName(&getChipNameParams));
    std::string chipName(getChipNameParams.pChipName);

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL_MAYBE_EXIT(NVPW_InitializeHost(&initializeHostParams));

    if (metricNames.size()) {
        if(!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage))
        {
            std::cout << "Failed to create configImage" << std::endl;
            exit(-1);
        }
        if(!NV::Metric::Config::GetCounterDataPrefixImage(chipName, metricNames, counterDataImagePrefix))
        {
            std::cout << "Failed to create counterDataImagePrefix" << std::endl;
            exit(-1);
        }
    }
    else
    {
        std::cout << "No metrics provided to profile" << std::endl;
        exit(-1);
    }

    if(!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer, counterDataImagePrefix))
    {
        std::cout << "Failed to create counterDataImage" << std::endl;
        exit(-1);
    }

    if(!runTest(cuDevice, configImage, counterDataScratchBuffer, counterDataImage, profilerReplayMode, profilerRange))
    {
        std::cout << "Failed to run sample" << std::endl;
        exit(-1);
    }
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

    /* Dump counterDataImage in file */
    WriteBinaryFile(CounterDataFileName.c_str(), counterDataImage);
    WriteBinaryFile(CounterDataSBFileName.c_str(), counterDataScratchBuffer);

    /* Evaluation of metrics collected in counterDataImage, this can also be done offline*/
    NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);
}

// We want an interface like this:
// Collect data like:
// GpuHWCounterSamples {
//   // <CUPTI_metric_name> -> ListOf[samples]
//   map<string, ListOf[float]> hardware_counter_samples,
//   ListOf[usec] start_us;
//   ListOf[usec] duration_us;
//   // Use this to associate with a particular "operation" instance.
//   operation_id
//   machine_name
//   process_name
//   phase_name
//   trace_id
// }
//
// GpuHWCounterConfig {
//   binary blob / configuration data needed to compute metrics offline using raw-counter values.
// }
//
// We would LIKE to support nesting...not sure what support CUPTI provides for this.

// Q: Is there any way to "scope" GPU HW counters to different operations...?
// In particular, how much of GPU occupancy was op1 responsible for vs op2?
//
// GpuHwCounterSampler<std::string> hw_sampler;
// hw_sampler.push_operation("some_op1");
//   ... op1 GPU code ...
//   hw_sampler.push_operation("some_op2");
//   ... op2 GPU code ...
//   hw_sampler.end_operation();
// hw_sampler.end_operation();
//
// Asynchronously dump files whenever we have "enough" samples.
// "Re-use" any CUPTI objects we can between collecting samples.
// Q: Should we convert from CUPTI binary format into csv-friendly protobuf format?
// Does this make it harder to compute derived metrics?
//
// NOTE: we DON'T need this for our purposes since we just want "high-level" metrics during a phase...
// could be useful in the future when "drilling down" into inference server.
//
// CSV output:
// operation_id,start_us,duration_us,metric_name,metric_value

//template <typename Op>
//struct GpuHwCounterSamplerState {
//      std::vector<>
//}
//
//template <typename Op>
//class GpuHwCounterSampler {
//      std::list<Op> operations;
//};

int main(int argc, char* argv[])
{
    backward::SignalHandling sh;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug


    // RUNTIME_NUM_UNIQUE_RANGES = FLAGS_unique_paths*FLAGS_nesting_levels;
    // COUNTER_DATA_MAX_NUM_NESTING_LEVELS = FLAGS_nesting_levels*2;
    // COUNTER_DATA_MAX_NUM_RANGES = RUNTIME_NUM_UNIQUE_RANGES*2;

    FLAGS_unique_paths = DQN_ATARI_UNIQUE_PATHS;
    FLAGS_nesting_levels = DQN_ATARI_NESTING_LEVELS;
    COUNTER_DATA_MAX_NUM_NESTING_LEVELS = FLAGS_nesting_levels;
    COUNTER_DATA_MAX_NUM_RANGES = FLAGS_unique_paths;

    {
        std::stringstream ss;
        ss << "Runtime info:"
           << std::endl
           << "  COUNTER_DATA_MAX_NUM_NESTING_LEVELS = " << COUNTER_DATA_MAX_NUM_NESTING_LEVELS
           << std::endl
           << "  COUNTER_DATA_MAX_NUM_RANGES = " << COUNTER_DATA_MAX_NUM_RANGES;
        RLS_LOG("GPU_HW", "{}", ss.str());
    }


    CUdevice cuDevice;
    std::vector<std::string> metricNames;

    int deviceCount;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

    DRIVER_API_CALL_MAYBE_EXIT(cuInit(0));
    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        return -2;
    }

    printf("CUDA Device Number: %d\n", FLAGS_device);

    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGet(&cuDevice, FLAGS_device));
    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor,computeCapabilityMinor);

    if(computeCapabilityMajor < 7) {
      printf("Sample unsupported on Device with compute capability < 7.0\n");
      return -2;
    }

    // Q: Is this required?
    // A: YES.  Otherwise cuptiProfilerBeginSession returns an error. Weird, since
    // I thought CUDA context was created implicitly...
    CUcontext cuContext;
    DRIVER_API_CALL_MAYBE_EXIT(cuCtxCreate(&cuContext, 0, cuDevice));

    // Get the names of the metrics to collect
    metricNames = rlscope::StringSplit(FLAGS_metric, ",");

    if (!FLAGS_disable_sampling) {
        run_profiled(cuDevice, metricNames);
    } else {
        std::cout << "DISABLE SAMPLING" << std::endl;
        run_gpu_compute();
    }

//    NV::Metric::Enum::ListMetrics(chipName.c_str(), true);
//    NV::Metric::Enum::ListCounters(chipName.c_str());
//    NV::Metric::Enum::ListRatios(chipName.c_str());
//    NV::Metric::Enum::ListThroughputs(chipName.c_str());
//    NV::Metric::Enum::ListMetricBases(chipName.c_str());

    DRIVER_API_CALL_MAYBE_EXIT(cuCtxDestroy(cuContext));

    {
        std::stringstream ss;
        ss << "Runtime info:"
           << std::endl
           << "  MAX_NESTING_LEVEL = " << MAX_NESTING_LEVEL
           << std::endl
           << "  setConfigParams.numNestingLevels = " << COUNTER_DATA_MAX_NUM_NESTING_LEVELS;
        RLS_LOG("GPU_HW", "{}", ss.str());
    }

    return 0;
}

