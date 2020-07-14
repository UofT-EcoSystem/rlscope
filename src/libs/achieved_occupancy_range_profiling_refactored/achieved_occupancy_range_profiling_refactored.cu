// nvcc bugs: cannot import json.hpp without errors:
// https://github.com/nlohmann/json/issues/1347
#define RLS_IGNORE_JSON

#include <cupti_target.h>
#include <cupti.h>
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

#include "range_sampling.h"
#include "common_util.h"

using rlscope::MyStatus;

// DON'T do "using namespace" to avoid name clashes for things that are still duplicated in other sample programs
//using namespace rlscope;

// achieved_occupancy = sm__warps_active.avg.pct_of_peak_sustained_active+
// inst_executed = sm__inst_executed.sum+
// default metric used by userrange_profiling = smsp__warps_launched.avg+ (how many warps are launched on average)
#define DEFAULT_METRIC_NAME "sm__warps_active.avg.pct_of_peak_sustained_active+"

DEFINE_bool(debug, false, "Debug: give additional verbose output");
DEFINE_string(print_csv, "", "File path to print all the GPU hw counter sample files (GPUHwCounterSampleProto*.proto) recursively rooted at --directory in csv format");
DEFINE_bool(disable_sampling, false, "Skip range profiling, just run GPU computation");
DEFINE_string(metric, DEFAULT_METRIC_NAME, "CUDA Profiling API Metric name");
DEFINE_int64(n_int32s, 50000, "Size of array for VecAdd/VecSub GPU calls");
DEFINE_int64(iterations, 20000, "number of times to call VecAdd kernel to add two int32 vectors");
DEFINE_int64(config_passes, 1, "how many configuration passes to perform to determine amount of space to allocate for GPU event counters");

DEFINE_int32(device, 0, "Device number");
DEFINE_string(directory, ".", "Directory to store protobuf files containing GPU hardware counter results (default: current working dir)");

DEFINE_int64(samples, 1, "How many GPU hw counter samples to take?");
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

void run_gpu_compute() {
//    CuptiSamples::ComputeVectorAddSubtract(FLAGS_n_int32s);
    auto start_compute_t = rlscope::get_timestamp_us();
    CuptiSamples::ComputeVecAdd(FLAGS_iterations, FLAGS_n_int32s);
    cudaDeviceSynchronize();
    auto end_compute_t = rlscope::get_timestamp_us();

    float compute_sec = (end_compute_t - start_compute_t).count()/FLOAT_MICROSECONDS_IN_SEC;
    std::cout << "ComputeVecAdd(iterations=" << FLAGS_iterations << ", n=" << FLAGS_n_int32s << ") took " << compute_sec << " seconds" << std::endl;
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

using RangeFunc = std::function<MyStatus()>;
static MyStatus with_N_ranges(rlscope::GPUHwCounterSampler& sampler, int N, const std::string& range_name, RangeFunc func) {
    CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
    CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};

    MyStatus ret = MyStatus::OK();

    // With nesting. Only compute in the last nest-level.
    for (int i = 0; i < N; i++) {
        std::stringstream ss;
        ss << range_name << "_" << i;
        auto range_name_i = ss.str();
//        pushRangeParams.pRangeName = range_name_i.c_str();
//        CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
        ret = sampler.Push(range_name_i);
        IF_BAD_STATUS_RETURN(ret);
    }
    ret = func();
    IF_BAD_STATUS_RETURN(ret);
    for (int i = 0; i < N; i++) {
//        CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
        ret = sampler.Pop();
        IF_BAD_STATUS_RETURN(ret);
    }

    return MyStatus::OK();
}

// void run_profiling() {
//     sampler.StartConfig({"sm_occupancy", "sm_efficiency"});
// 
//     sampler.StartProfiling();
// 
//     // Training loop.
//     for (int iteration = 0; i < N; i++) {
//         // Tell profiler when we start/end 
//         // a repeatable computation that wish 
//         // to profile.
//         sampler.StartPass();
// 
//         sampler.Push("vec_add");
//             vec_add<<<blocks, block_size>>>(A, B, C);
//             vec_add<<<blocks, block_size>>>(A, B, C);
//             vec_add<<<blocks, block_size>>>(A, B, C);
//         sampler.Pop();
// 
//         sampler.EndPass();
//     }
// 
//     sampler.StopProfiling();
// 
//     sampler.PrintCSV();
// }

MyStatus run_pass(rlscope::GPUHwCounterSampler& sampler) {
    MyStatus ret = MyStatus::OK();

    ret = sampler.StartPass();
    IF_BAD_STATUS_RETURN(ret);
    // IF_BAD_STATUS_EXIT("Failed to start configuration pass for GPU hw counter profiler", ret);

    for (int32_t userrange_i = 0; userrange_i < FLAGS_unique_paths; userrange_i++) {
        std::stringstream range_ss;
        range_ss << "userrange_path";
        range_ss << userrange_i;
        ret = with_N_ranges(sampler, FLAGS_nesting_levels, range_ss.str(), [&] {
            run_gpu_compute();
            return MyStatus::OK();
        });
    }

    ret = sampler.EndPass();
    IF_BAD_STATUS_RETURN(ret);
    // IF_BAD_STATUS_EXIT("Failed to end configuration pass for GPU hw counter profiler", ret);

    return MyStatus::OK();
}

int main(int argc, char* argv[])
{
    backward::SignalHandling sh;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // NOTE: If you only define SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG, this doesn't enable debug logging.
    // It just ensures that the SPDLOG_DEBUG statements are **compiled in**!
    // We still need to turn them on though!
    spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

#define PRINT_FLAG(ss, flags_var) \
    ss << "  " << #flags_var << " = " << flags_var << "\n";
    {
        std::stringstream ss;
        ss << "Cmdline:\n";
        PRINT_FLAG(ss, FLAGS_print_csv);
        PRINT_FLAG(ss, FLAGS_debug);
        PRINT_FLAG(ss, FLAGS_disable_sampling);
        PRINT_FLAG(ss, FLAGS_metric);
        PRINT_FLAG(ss, FLAGS_n_int32s);
        PRINT_FLAG(ss, FLAGS_iterations);
        PRINT_FLAG(ss, FLAGS_config_passes);
        PRINT_FLAG(ss, FLAGS_device);
        PRINT_FLAG(ss, FLAGS_directory);
        PRINT_FLAG(ss, FLAGS_samples);
        PRINT_FLAG(ss, FLAGS_passes);
        PRINT_FLAG(ss, FLAGS_unique_paths);
        PRINT_FLAG(ss, FLAGS_nesting_levels);

        DBG_LOG("{}", ss.str());
    }
#undef PRINT_FLAG


    // NOTE: This is required.  I think we can assume a CUDA
    // code-base will make these calls for us before sampler is used.
    DRIVER_API_CALL_MAYBE_EXIT(cuInit(0));
    CUdevice cuDevice;
    DRIVER_API_CALL_MAYBE_EXIT(cuDeviceGet(&cuDevice, FLAGS_device));
    CUcontext cuContext;
    DRIVER_API_CALL_MAYBE_EXIT(cuCtxCreate(&cuContext, 0, cuDevice));

    MyStatus ret = MyStatus::OK();
    rlscope::GPUHwCounterSampler sampler(FLAGS_device, FLAGS_directory, "");

    ret = sampler.Init();
    IF_BAD_STATUS_EXIT("Failed to initialize GPU hw counter profiler", ret);

    if (FLAGS_print_csv != "") {
        bool printed_header = false;
        std::ofstream csv_f(FLAGS_print_csv, std::ios::out | std::ios::trunc);
        if (csv_f.fail()) {
            std::cerr << "ERROR: Failed to write to csv file --print_csv=" << FLAGS_print_csv << " : " << strerror(errno) << std::endl;
            exit(EXIT_FAILURE);
        }
        ret = sampler.PrintCSV(csv_f, printed_header);
        IF_BAD_STATUS_EXIT("Failed to print GPU hw sample files in csv format", ret);
        exit(EXIT_SUCCESS);
    }

    // Get the names of the metrics to collect
    std::vector<std::string> metricNames;
    metricNames = rlscope::StringSplit(FLAGS_metric, ",");
    ret = sampler.StartConfig(metricNames);
    IF_BAD_STATUS_EXIT("Failed to configure GPU hw counter profiler", ret);

    for (int64_t i = 0; i < FLAGS_config_passes; i++) {
        ret = run_pass(sampler);
        if (ret.code() != rlscope::error::OK) {
            std::stringstream ss;
            ss << "Failed to run configuration pass " << i << " with GPU hw counter profiler enabled";
            IF_BAD_STATUS_EXIT(ss.str(), ret);
        }
    }

    std::cout << "Configuration pass results:" << std::endl
              << "  MaxNestingLevels = " << sampler.MaxNestingLevels() << std::endl
              << "  MaxNumRanges = " << sampler.MaxNumRanges() << std::endl;

    ret = sampler.StartProfiling();
    IF_BAD_STATUS_EXIT("Failed to start GPU hw counter profiler", ret);

    for (int64_t i = 0; i < FLAGS_samples; i++) {
        while (sampler.HasNextPass()) {
            DBG_LOG("Sample {}, Pass {}", i + 1, sampler._pass_idx + 1);

            ret = run_pass(sampler);
            IF_BAD_STATUS_EXIT("Failed to run pass with GPU hw counter profiler enabled", ret);
        }
        if (sampler.CanRecord()) {
            ret = sampler.RecordSample();
            IF_BAD_STATUS_EXIT("Failed to record GPU hw counter sample", ret);
        }
    }

    ret = sampler.StopProfiling();
    IF_BAD_STATUS_EXIT("Failed to stop GPU hw counter profiler", ret);

    return 0;
}
