//
// Created by jgleeson on 2020-05-14.
//

#include "RangeSampling.h"
#include "ScopeExit.h"

#include "common_util.h"

#include "range_sampling/range_sampling.pb.h"

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <nvperf_host.h>

#include <vector>
#include <list>
#include <string>
#include <regex>
#include <chrono>
#include <assert.h>
#include <nvperf_cuda_host.h>

#include <algorithm>
#include <iterator>
#include <boost/uuid/detail/md5.hpp>
#include <boost/algorithm/hex.hpp>

using boost::uuids::detail::md5;

#include <iostream>
#include <fstream>

namespace rlscope {

// 50 MB.
// In a simple script, one sample can take up 11K, so I expect the size to be large in general
// for real workloads.
const size_t GPUHwCounterSampler::MaxSampleFileSizeBytes = 50*1024*1024;

// 25K (test dumping multiple files for simple scripts)
//const size_t GPUHwCounterSampler::MaxSampleFileSizeBytes = 25*1024;

std::string md5_as_string(const md5::digest_type &digest)
{
    const auto charDigest = reinterpret_cast<const char *>(&digest);
    std::string result;
    boost::algorithm::hex(charDigest, charDigest + sizeof(md5::digest_type), std::back_inserter(result));
    return result;
}

std::string calc_md5_hash(const uint8_t* data, size_t n) {
    md5 hash;
    md5::digest_type digest;

    hash.process_bytes(data, n);
    hash.get_digest(digest);

    return md5_as_string(digest);
}

static MyStatus GetRawMetricRequests(
        NVPA_MetricsContext* pMetricsContext,
        const std::vector<std::string>& metricNames,
        std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
        std::vector<std::string>& temp);

static bool ParseMetricNameString(const std::string& metricName, std::string* reqName, bool* isolated, bool* keepInstances);

// State created for each "sample".

MyStatus CounterData::Init() {
    MyStatus ret = MyStatus::OK();

    ret = _InitConfigImagePrefix();
    IF_BAD_STATUS_RETURN(ret);

    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
    counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
    // Q: Does a single range get profiled in each pass...?
    counterDataImageOptions.maxNumRanges = counter_data_max_num_ranges;
    counterDataImageOptions.maxNumRangeTreeNodes = counter_data_max_num_ranges;
    counterDataImageOptions.maxRangeNameLength = 64;

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

    calculateSizeParams.pOptions = &counterDataImageOptions;
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.pOptions = &counterDataImageOptions;
    initializeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);
    initializeParams.pCounterDataImage = &counterDataImage[0];
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchBufferSizeParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;
    scratchBufferSizeParams.pCounterDataImage = initializeParams.pCounterDataImage;
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    initScratchBufferParams.counterDataImageSize = calculateSizeParams.counterDataImageSize;

    initScratchBufferParams.pCounterDataImage = initializeParams.pCounterDataImage;
    initScratchBufferParams.counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize;
    initScratchBufferParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];

    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return MyStatus::OK();
}

// Q: How many of these should there be...?
// A: One for each process.  Profiler should get initialized once at process start.
// I guess we could probably get away with one for the entire profiling session.
// NOTE: we should try to avoid saving this in each "trace file" of the sampler (it's a one-time thing)

MyStatus ConfigData::Init() {
    MyStatus ret = MyStatus::OK();
    ret = _InitConfigImage();
    IF_BAD_STATUS_RETURN(ret);
    return MyStatus::OK();
}

static bool ParseMetricNameString(const std::string& metricName, std::string* reqName, bool* isolated, bool* keepInstances)
{
    std::string& name = *reqName;
    name = metricName;
    if (name.empty())
    {
        return false;
    }

    // boost program_options sometimes inserts a \n between the metric name and a '&' at the end
    size_t pos = name.find('\n');
    if (pos != std::string::npos)
    {
        name.erase(pos, 1);
    }

    // trim whitespace
    while (name.back() == ' ')
    {
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }

    *keepInstances = false;
    if (name.back() == '+')
    {
        *keepInstances = true;
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }

    *isolated = true;
    if (name.back() == '$')
    {
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }
    else if (name.back() == '&')
    {
        *isolated = false;
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }

    return true;
}

static MyStatus GetRawMetricRequests(
        NVPA_MetricsContext* pMetricsContext,
        const std::vector<std::string>& metricNames,
        std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
        std::vector<std::string>& temp) {
    std::string reqName;
    bool isolated = true;
    bool keepInstances = true;

    for (auto& metricName : metricNames)
    {
        ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
        /* Bug in collection with collection of metrics without instances, keep it to true*/
        keepInstances = true;
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
        getMetricPropertiesBeginParams.pMetricName = reqName.c_str();

        NVPW_API_CALL_MAYBE_STATUS(NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams));

        for (const char** ppMetricDependencies = getMetricPropertiesBeginParams.ppRawMetricDependencies; *ppMetricDependencies; ++ppMetricDependencies)
        {
            temp.push_back(*ppMetricDependencies);
        }
        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesEndParams.pMetricsContext = pMetricsContext;
        NVPW_API_CALL_MAYBE_STATUS(NVPW_MetricsContext_GetMetricProperties_End(&getMetricPropertiesEndParams));
    }

    for (auto& rawMetricName : temp)
    {
        NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };

//        isolated = false;
//        keepInstances = false;
//
//        NVPW_RawMetricsConfig_GetMetricProperties_Params GetMetricProperties_Params = { NVPW_RawMetricsConfig_GetMetricProperties_Params_STRUCT_SIZE };
//        GetMetricProperties_Params.pMetricName = rawMetricName.c_str();
//        setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
//        setCounterDataParams.pCounterDataImage = counterDataImage;
//        setCounterDataParams.isolated = true;
//        setCounterDataParams.rangeIndex = rangeIndex;
//        NVPW_MetricsContext_SetCounterData(&setCounterDataParams);
//        NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_GetMetricProperties(NVPW_RawMetricsConfig_GetMetricProperties_Params* pParams));

        {
            std::stringstream ss;
            ss << "CONFIG: metric = " << rawMetricName << std::endl
               << "  isolated = " << isolated << std::endl
               << "  keepInstances = " << keepInstances << std::endl;
            DBG_LOG("{}", ss.str());
        }
        metricRequest.pMetricName = rawMetricName.c_str();
        metricRequest.isolated = isolated;
        metricRequest.keepInstances = keepInstances;
        rawMetricRequests.push_back(metricRequest);
    }

    return MyStatus::OK();
}

MyStatus CounterData::getNumRangeCollected(size_t* numRanges) const {
    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = &counterDataImage[0];
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CounterData_GetNumRanges(&getNumRangesParams));
    *numRanges = getNumRangesParams.numRanges;
    return MyStatus::OK();
}

MyStatus CounterData::_InitConfigImagePrefix()
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
    counterDataBuilderCreateParams.pChipName = chipName.c_str();
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
    counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    SCOPE_EXIT([&]() { NVPW_CounterDataBuilder_Destroy((NVPW_CounterDataBuilder_Destroy_Params *)&counterDataBuilderDestroyParams); });

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests = rawMetricRequests.size();
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

    size_t counterDataPrefixSize = 0;
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
    getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer = NULL;
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

    counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

    getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
    getCounterDataPrefixParams.pBuffer = &counterDataImagePrefix[0];
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

    return MyStatus::OK();
}

MyStatus ConfigData::_InitConfigImage()
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

    NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
    metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    metricsConfigOptions.pChipName = chipName.c_str();
    NVPA_RawMetricsConfig* pRawMetricsConfig;
    NVPW_API_CALL_MAYBE_STATUS(NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig));

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    SCOPE_EXIT([&]() { NVPW_RawMetricsConfig_Destroy((NVPW_RawMetricsConfig_Destroy_Params *)&rawMetricsConfigDestroyParams); });

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests = rawMetricRequests.size();
    NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
    generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
    getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    getConfigImageParams.bytesAllocated = 0;
    getConfigImageParams.pBuffer = NULL;
    NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

    configImage.resize(getConfigImageParams.bytesCopied);

    getConfigImageParams.bytesAllocated = configImage.size();
    getConfigImageParams.pBuffer = &configImage[0];
    NVPW_API_CALL_MAYBE_STATUS(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));


    return MyStatus::OK();
}

std::unique_ptr<iml::ConfigDataProto> ConfigData::AsProto() {
    std::unique_ptr<iml::ConfigDataProto> proto(new iml::ConfigDataProto);
    proto->set_data(configImage.data(), configImage.size());
    return proto;
}

MyStatus CUPTIProfilerState::_InitConfig(ConfigData& config_data) {
    // NOTE: in the original "userrange_sampling" sample program, cuptiProfilerSetConfig gets called
    // once after cuptiProfilerBeginSession and before the first call to cuptiProfilerBeginPass.
    // I've found that if I attempt to call cuptiProfilerSetConfig again because I end up changing/reallocating
    // the configImage buffer, then I get errors from the CUPTI API.  So, instead I explicitly re-use the same buffer
    // so that I don't need to call cuptiProfilerSetConfig more than once.
    assert(config_data.configImage.size() != 0);
    std::cerr << "> configImage.size() = " << config_data.configImage.size() << std::endl;
    CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pConfig = &config_data.configImage[0];
    setConfigParams.configSize = config_data.configImage.size();
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = config_data.counter_data_max_num_nesting_levels;
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerSetConfig(&setConfigParams));

    return MyStatus::OK();
}

MyStatus CUPTIProfilerState::StartProfiling(ConfigData& config_data, CounterData& counter_data) {
    MyStatus ret = MyStatus::OK();

    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;
    CUpti_ProfilerRange profilerRange = CUPTI_UserRange;

    CUpti_Profiler_BeginSession_Params beginSessionParams;
    beginSessionParams = {CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSessionParams.ctx = NULL;
    beginSessionParams.counterDataImageSize = counter_data.counterDataImage.size();
    beginSessionParams.pCounterDataImage = &counter_data.counterDataImage[0];
    beginSessionParams.counterDataScratchBufferSize = counter_data.counterDataScratchBuffer.size();
    beginSessionParams.pCounterDataScratchBuffer = &counter_data.counterDataScratchBuffer[0];
    beginSessionParams.range = profilerRange;
    beginSessionParams.replayMode = profilerReplayMode;
    beginSessionParams.maxRangesPerPass = counter_data_max_num_ranges;
    // Q: Does this matter...?  It's hard to know how many kernels might be launched.
    beginSessionParams.maxLaunchesPerPass = counter_data_max_num_ranges;
    // NOTE: if we attempt to profile multiple processes at once, the first process will succeed here,
    // but the next ones will fail.
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerBeginSession(&beginSessionParams));
    // DBG_LOG("{}", "cuptiProfilerBeginSession WORKED");

    ret = _InitConfig(config_data);
    IF_BAD_STATUS_RETURN(ret);

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerEnableProfiling(&enableProfilingParams));

    return MyStatus::OK();
}

MyStatus CUPTIProfilerState::StopProfiling(ConfigData& config_data, CounterData& counter_data) {

    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerDisableProfiling(&disableProfilingParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerUnsetConfig(&unsetConfigParams));

    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerEndSession(&endSessionParams));

    return MyStatus::OK();
}

bool CUPTIProfilerState::HasNextPass() const {
    return !_endPassParams_allPassesSubmitted;
}

MyStatus CUPTIProfilerState::StartPass(ConfigData& config_data) {
    MyStatus ret = MyStatus::OK();

    CUpti_Profiler_BeginPass_Params beginPassParamsLocal = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerBeginPass(&beginPassParamsLocal));

    return MyStatus::OK();
}

MyStatus CUPTIProfilerState::EndPass() {
    CUpti_Profiler_EndPass_Params endPassParamsLocal = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerEndPass(&endPassParamsLocal));
    _endPassParams_allPassesSubmitted = endPassParamsLocal.allPassesSubmitted;

    return MyStatus::OK();
}

MyStatus CUPTIProfilerState::Flush(ConfigData &config_data, CounterData &counter_data) {
    // Flush counters to counter-data.
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    return MyStatus::OK();
}

MyStatus CUPTIProfilerState::NextPass(ConfigData &config_data, CounterData &counter_data) {
    MyStatus ret = MyStatus::OK();

    _endPassParams_allPassesSubmitted = false;

    return MyStatus::OK();
}

bool GPUHwCounterSamplerState::CanDump() const {
    return config_data.size() > 0 && counter_data.size() > 0;
//        return _process_name != "" &&
//               _machine_name != "" &&
//               _phase_name != "" &&
//               _directory != ""
}

//void GPUHwCounterSampler::AsyncDump() {
//    std::unique_lock<std::mutex> lock(_mu);
//    _AsyncDump();
//}
//
//MyStatus GPUHwCounterSamplerState::ShouldDump() {
//    // How do we store samples?
//    assert(false);
//    return MyStatus::OK();
////    return
////        // Number of records is larger than some threshold (~ ... MB).
////            ( _events.size() ) >= CUDA_API_PROFILER_MAX_RECORDS_PER_DUMP;
//}
//
//void GPUHwCounterSampler::_AsyncDump() {
//    if (_state.CanDump()) {
//        GPUHwCounterSamplerState dump_state;
//        dump_state = _state.DumpState();
//        _AsyncDumpWithState(std::move(dump_state));
//    }
//}
//
//void GPUHwCounterSampler::_AsyncDumpWithState(GPUHwCounterSamplerState&& dump_state) {
//    _pool.Schedule([dump_state = std::move(dump_state)] () mutable {
//        auto path = dump_state.DumpPath(dump_state._trace_id);
//        mkdir_p(os_dirname(path));
//        auto proto = dump_state.AsProto();
//        std::fstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
//        if (!proto->SerializeToOstream(&out)) {
//            LOG(FATAL) << "Failed to dump " << path;
//        }
//        VLOG(1) << "Dumped " << path;
//    });
//    DCHECK(!_state.CanDump());
//}

std::unique_ptr<iml::CounterDataProto> GPUHwCounterSamplerState::AsProto() {
    std::unique_ptr<iml::CounterDataProto> proto(new iml::CounterDataProto);
    proto->set_data(counter_data.counterDataImage.data(), counter_data.counterDataImage.size());
    proto->set_start_time_us(start_profiling_t.time_since_epoch().count());
    assert(start_profiling_t <= stop_profiling_t);
    proto->set_duration_us((stop_profiling_t - start_profiling_t).count());
    return proto;
}

std::string GPUHwCounterSamplerProtoState::DumpPath() const {
//    DCHECK(directory != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
//    DCHECK(_phase_name != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
//    DCHECK(_process_name != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
    assert(_directory != "");

    std::stringstream ss;

    ss << _directory << path_separator();

//    ss << "process" << path_separator() << _process_name << path_separator();
//    ss << "phase" << path_separator() << _phase_name << path_separator();

    ss << "GPUHwCounterSampleProto";

    ss << ".trace_" << _trace_id;

    if (_dump_suffix != "") {
        ss << _dump_suffix;
    }

    ss << ".proto";

    return ss.str();
}

//MyStatus GPUHwCounterSampler::_SyncDumpWithState(GPUHwCounterSamplerState&& dump_state) {
////    _pool.Schedule([dump_state = std::move(dump_state)] () mutable {
////    auto dump_state = std::move(dump_state_);
//    auto path = dump_state.DumpPath(dump_state.trace_id);
//    auto direc = os_dirname(path);
//    if (direc != ".") {
//        mkdir_p(direc);
//    }
//    auto proto = dump_state.AsProto();
//    std::fstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
//    if (!proto->SerializeToOstream(&out)) {
//        std::cerr << "Failed to dump " << path << std::endl;
//        exit(-1);
//    }
//    DBG_LOG("Dumped {}", path);
////    });
//    return MyStatus::OK();
//}

GPUHwCounterSampler::~GPUHwCounterSampler() {
    // Q: How many times does this need to be called?
    // In particular:
    // - multi-threaded GPU workload
    //   My guess: once
    // - multi-context (single process) GPU workload
    //   My guess: N times for N threads
    //   ... lets just not bother with this scenario ...
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

MyStatus GetGPUChipName(int device, std::string* chip_name) {
    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
    getChipNameParams.deviceIndex = device;
    CUPTI_API_CALL_MAYBE_STATUS(cuptiDeviceGetChipName(&getChipNameParams));
    assert(getChipNameParams.pChipName != nullptr);
    *chip_name = getChipNameParams.pChipName;
    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::Init() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    MyStatus ret = MyStatus::OK();

    // Programmer error; you haven't called SetDevice.
    assert(_device >= 0);

    ret = CheckCUPTIProfilingAPISupported();
    IF_BAD_STATUS_RETURN(ret);

//    CUdevice cuDevice;
//    DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGet(&cuDevice, _device));

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerInitialize(&profilerInitializeParams));
    ret = GetGPUChipName(_device, &_chip_name);
    IF_BAD_STATUS_RETURN(ret);

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
    NVPW_API_CALL_MAYBE_STATUS(NVPW_InitializeHost(&initializeHostParams));

    _initialized = true;

    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::Push(const std::string &operation) {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);
    bool allow_insert = (_mode == PROFILE);
    bool would_insert = _range_tree.Push(operation, allow_insert);
    if (!allow_insert && would_insert) {
      std::stringstream ss;
      ss << "GPUHwCounterSampler: Tried to push operation=\"" << operation << "\", but we never saw this operation during the configuration pass.  Operation stack at time of push was:";
      auto node_stack = _range_tree.CurStack();
      for (auto const& node : node_stack) {
        ss << "\n";
        PrintIndent(ss, 1);
        ss << node->name;
      }
      return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
    if (_mode == PROFILE) {
        CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
        pushRangeParams.pRangeName = operation.c_str();
        pushRangeParams.rangeNameLength = operation.size();
        CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerPushRange(&pushRangeParams));
    }
    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::Pop() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);
    _range_tree.Pop();
    if (_mode == PROFILE) {
        CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
        CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerPopRange(&popRangeParams));
    }
    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::Disable() {
    _enabled = false;
    return MyStatus::OK();
}
bool GPUHwCounterSampler::Enabled() const {
    return _enabled;
}
MyStatus GPUHwCounterSampler::StartConfig(const std::vector<std::string>& metrics) {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);

    _mode = CONFIG;
    assert(metrics.size() > 0);
    _metrics = metrics;
    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::CheckCUPTIProfilingAPISupported() {
    CUdevice cuDevice;
    int deviceCount;
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

    // DRIVER_API_CALL_MAYBE_STATUS(cuInit(0));
    DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0 || _device >= deviceCount) {
        std::stringstream ss;
        ss << "There is no GPU --device=" << _device << " (device count was " << deviceCount << ")";
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }

//    printf("CUDA Device Number: %d\n", _device);

    DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGet(&cuDevice, _device));
    DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

//    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor,computeCapabilityMinor);

    if(computeCapabilityMajor < 7) {
        std::stringstream ss;
        ss << "Sample unsupported on --device=" << _device
           << " with compute capability < 7.0 "
           << "(saw compute capability " << computeCapabilityMajor << "." << computeCapabilityMinor << " )";
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }

    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::StartProfiling() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);
    MyStatus ret = MyStatus::OK();
    _mode = PROFILE;

    ret = _InitSamplerState();
    IF_BAD_STATUS_RETURN(ret);

    ret = profiler_state.StartProfiling(state.config_data, state.counter_data);
    IF_BAD_STATUS_RETURN(ret);

    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::_InitSamplerState() {
    MyStatus ret = MyStatus::OK();
//    _pass_idx = 0;
    if (MaxNestingLevels() == 0) {
        std::cerr << "WARNING: did you forget to call StartConfig()? MaxNestingLevels() == " << MaxNestingLevels() << std::endl;
    }
    if (MaxUniqueRanges() == 0) {
        std::cerr << "WARNING: did you forget to call StartConfig()? MaxUniqueRanges() == " << MaxUniqueRanges() << std::endl;
    }

    profiler_state = CUPTIProfilerState(/*counter_data_max_num_ranges=*/UseMaxUniqueRanges());

    state.directory = _directory;
    assert(_metrics.size() > 0);

    state.config_data = ConfigData(
            _chip_name,
            _metrics,
            /*counter_data_max_num_nesting_levels*/UseMaxNestingLevels());
    ret = state.config_data.Init();
    IF_BAD_STATUS_RETURN(ret);

    state.counter_data = CounterData(
            _chip_name,
            _metrics,
            /*counter_data_max_num_ranges=*/UseMaxUniqueRanges());
    ret = state.counter_data.Init();
    IF_BAD_STATUS_RETURN(ret);

    ret = _NextSamplerState();
    IF_BAD_STATUS_RETURN(ret);

    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::_NextSamplerState() {
    MyStatus ret = MyStatus::OK();
    _pass_idx = 0;
    state.start_profiling_t = rlscope::get_timestamp_us();
    return MyStatus::OK();
}
MyStatus GPUHwCounterSamplerProtoState::DumpSync() {
    MyStatus ret = MyStatus::OK();
    auto path = DumpPath();

    // NOTE: explicitly COPY the configImage buffer, leaving the original in-place so that future
    // passes for future samples can re-use it.  I found that attempting to reallocate/reconfigure
    // configImage buffer led to CUPTI errors. See CUPTIProfilerState::_InitConfig for details.
    auto proto = this->AsProto();

    auto direc = os_dirname(path);
    if (direc != ".") {
        mkdir_p(direc);
    }
    std::fstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!proto->SerializeToOstream(&out)) {
        std::stringstream ss;
        ss << "Failed to dump " << path << " : " << strerror(errno);
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
    DBG_LOG("Dumped {}", path);
    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::_MaybeRecordSample(bool* recorded) {
    MyStatus ret = MyStatus::OK();
    *recorded = false;
    if (!this->CanRecord()) {
        return MyStatus::OK();
    }

    ret = profiler_state.Flush(state.config_data, state.counter_data);
    IF_BAD_STATUS_RETURN(ret);
    // Record timestamp before or after flush?  Lets go with after.
    state.stop_profiling_t = rlscope::get_timestamp_us();

    {
        // NOTE: explicitly COPY the configImage buffer, leaving the original in-place so that future
        // passes for future samples can re-use it.  I found that attempting to reallocate/reconfigure
        // configImage buffer led to CUPTI errors. See CUPTIProfilerState::_InitConfig for details.
        std::unique_ptr<GPUHwCounterSamplerState> sample(new GPUHwCounterSamplerState);
        *sample = state;
        _size_bytes += sample->size_bytes();
        _samples.push_back(std::move(sample));
    }

    *recorded = true;
    return MyStatus::OK();
}


MyStatus GPUHwCounterSampler::StopProfiling() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);
    MyStatus ret = MyStatus::OK();

    if (CanRecord()) {
        ret = RecordSample();
        IF_BAD_STATUS_RETURN(ret);
    }

    if (CanDump()) {
        ret = DumpAsync();
        IF_BAD_STATUS_RETURN(ret);
    }

    ret = AwaitDump();
    IF_BAD_STATUS_RETURN(ret);

    if (_pass_idx > 0) {
        std::cerr << "WARNING: GPU hw counter didn't run for enough passes so the last sample will be discarded; "
                  << "only ran for " << (_pass_idx + 1) << " passes but likely needed at least " << UseMaxUniqueRanges()
                  << std::endl;
    }

    ret = profiler_state.StopProfiling(state.config_data, state.counter_data);
    IF_BAD_STATUS_RETURN(ret);

    return MyStatus::OK();
}
bool GPUHwCounterSampler::HasNextPass() const {
    if (!_enabled) {
        // Q: What's the right behaviour here...?
        return false;
    }
    assert(_initialized);
    return profiler_state.HasNextPass();
}
bool GPUHwCounterSampler::CanRecord() const {
    if (!_enabled) {
        return false;
    }
    assert(_initialized);
//    DBG_LOG("!HasNextPass() = {}, state.CanDump() = {}", !HasNextPass(), state.CanRecord());
    return !HasNextPass() && state.CanDump();
}
bool GPUHwCounterSampler::CanDump() const {
    assert(_initialized);
//    DBG_LOG("!HasNextPass() = {}, state.CanDump() = {}", !HasNextPass(), state.CanRecord());
    return _samples.size() > 0 && !CanRecord();
}
bool GPUHwCounterSampler::ShouldDump() const {
    assert(_initialized);
//    DBG_LOG("!HasNextPass() = {}, state.CanDump() = {}", !HasNextPass(), state.CanRecord());
    return CanDump() && this->size_bytes() >= GPUHwCounterSampler::MaxSampleFileSizeBytes;
}
MyStatus GPUHwCounterSampler::StartPass() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);

    MyStatus ret = MyStatus::OK();
    if (_mode == PROFILE) {

        ret = profiler_state.StartPass(state.config_data);
        IF_BAD_STATUS_RETURN(ret);

    }
    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::EndPass() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);

    MyStatus ret = MyStatus::OK();

    if (_mode == PROFILE) {
        _pass_idx += 1;
        assert(_pass_idx <= this->NumPasses());
        size_t num_ranges_collected = 0;
        ret = state.counter_data.getNumRangeCollected(&num_ranges_collected);
        IF_BAD_STATUS_RETURN(ret);

        std::cout << "After pass " << _pass_idx << ":" << std::endl
                  << "  total_num_ranges_collected = " << num_ranges_collected << std::endl;
//              << "  increase in total_num_ranges_collected since last pass = " << (num_ranges_collected - last_num_ranges_collected) << std::endl;

        ret = profiler_state.EndPass();
        IF_BAD_STATUS_RETURN(ret);


    }

    return MyStatus::OK();
}


MyStatus GPUHwCounterSampler::_DumpAsync() {
    auto proto_state = this->AsProtoState();
    boost::asio::post(_pool, [proto_state = std::move(proto_state)] () mutable {
        MyStatus ret = MyStatus::OK();
        ret = proto_state.DumpSync();
        if (!ret.ok()) {
            std::stringstream ss;
            ss << "Failed to dump GPU hw sampling state " << proto_state.DumpPath() << " asynchronously";
            IF_BAD_STATUS_EXIT(ss.str(), ret);
        }
    });
    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::_DumpSync() {
    auto proto_state = this->AsProtoState();
    MyStatus ret = MyStatus::OK();
    ret = proto_state.DumpSync();
    IF_BAD_STATUS_RETURN(ret);
    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::DumpSync() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    return _Dump(/*sync=*/true);
}
MyStatus GPUHwCounterSampler::DumpAsync() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    return _Dump(/*sync=*/false);
}
MyStatus GPUHwCounterSampler::_Dump(bool sync) {
    assert(_initialized);
    MyStatus ret = MyStatus::OK();
    if (_mode == PROFILE) {
        if (!CanDump()) {
            std::stringstream ss;
            ss << "Haven't collected enough GPU hw counter samples to dump anything (samples = " << _samples.size() << "); "
               << "only ran for " << (_pass_idx + 1) << " passes but likely needed at least " << UseMaxUniqueRanges() << " to be able to record one sample";
            return MyStatus(error::INVALID_ARGUMENT, ss.str());
        }
        if (sync) {
            ret = _DumpSync();
            IF_BAD_STATUS_RETURN(ret);
        } else {
            ret = _DumpAsync();
            IF_BAD_STATUS_RETURN(ret);
        }
        assert(!CanDump());
    }
    return MyStatus::OK();
}

std::unique_ptr<iml::GPUHwCounterSampleProto> GPUHwCounterSamplerProtoState::AsProto() {
    std::unique_ptr<iml::GPUHwCounterSampleProto> proto(new iml::GPUHwCounterSampleProto);

    auto* config_data_proto = proto->mutable_config_data();
    {
        auto tmp_config_data_proto = _config_data.AsProto();
        config_data_proto->Swap(tmp_config_data_proto.get());
    }

    proto->set_chip_name(_chip_name);

    proto->clear_metrics();
    for (size_t i = 0; i < _metrics.size(); i++) {
        proto->add_metrics(_metrics[i]);
    }

    auto it = _samples.begin();
    while (it != _samples.end()) {
        auto* counter_data_proto = proto->add_counter_data();
        {
            auto tmp_counter_data_proto = (*it)->AsProto();
            counter_data_proto->Swap(tmp_counter_data_proto.get());
        }
        it = _samples.erase(it);
    }

    proto->set_num_passes(_num_passes);

    return proto;
}

MyStatus GPUHwCounterSampler::RecordSample() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    assert(_initialized);
    MyStatus ret = MyStatus::OK();
    if (_mode == PROFILE) {
        if (HasNextPass()) {
            std::stringstream ss;
            ss << "GPU hw counter didn't run for enough passes to collect a full sample; "
               << "only ran for " << (_pass_idx + 1) << " passes but likely needed at least " << UseMaxUniqueRanges();
            return MyStatus(error::INVALID_ARGUMENT, ss.str());
        }
        // TODO: need to record sample state and dump it asynchronously (want multiple samples per proto file)...
        // for now we can just synchronously dump a new file for each sample.
        bool recorded = false;
        ret = _MaybeRecordSample(&recorded);
        IF_BAD_STATUS_RETURN(ret);
        assert(recorded);

        ret = profiler_state.NextPass(state.config_data, state.counter_data);
        IF_BAD_STATUS_RETURN(ret);

        ret = _NextSamplerState();
        IF_BAD_STATUS_RETURN(ret);

        if (ShouldDump()) {
            ret = DumpAsync();
            IF_BAD_STATUS_RETURN(ret);
        }
    }

    return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::PrintCSV(std::ostream &out, const iml::GPUHwCounterSampleProto& proto, bool& printed_header) {
    MyStatus ret = MyStatus::OK();
    size_t sample_idx = 0;
    const std::vector<std::string> extra_headers{"start_time_us", "duration_us", "num_passes"};
    for (const auto& counter_data_proto : proto.counter_data()) {
        std::vector<std::string> metrics;
        metrics.reserve(proto.metrics().size());
        for (const auto& metric : proto.metrics()) {
            metrics.push_back(metric);
        }
        const std::string& counter_data_string = counter_data_proto.data();
        auto counter_data = reinterpret_cast<const uint8_t*>(counter_data_string.data());

        std::vector<std::string> extra_fields;
#define APPEND_EXTRA_FIELD(value) { \
        std::stringstream field_ss; \
        field_ss << value; \
        extra_fields.push_back(field_ss.str()); \
    }
        APPEND_EXTRA_FIELD(counter_data_proto.start_time_us());
        APPEND_EXTRA_FIELD(counter_data_proto.duration_us());
        APPEND_EXTRA_FIELD(proto.num_passes());
        ret = this->PrintCSV(
                out,
                proto.chip_name(),
                counter_data,
                counter_data_proto.data().size(),
                metrics,
                printed_header,
                extra_headers,
                extra_fields);
        IF_BAD_STATUS_RETURN(ret);
        sample_idx += 1;
    }
    return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::PrintCSV(std::ostream &out, bool& printed_header) {
    // IMPORTANT: printing of metrics requires some intitialization (NVPW_InitializeHost specifically).
    assert(_initialized);

    MyStatus ret = MyStatus::OK();
    std::list<std::string> paths;
    ret = RecursiveFindFiles(&paths, _directory, [&] (const boost::filesystem::path& bpath) {
        if (!boost::filesystem::is_regular_file(bpath)) {
            return false;
        }
        return this->IsProtoFile(bpath);
    });
    IF_BAD_STATUS_RETURN(ret);


    for (const auto& path : paths) {
        DBG_LOG("path = {}", path);
    }

    for (const auto& path : paths) {
        iml::GPUHwCounterSampleProto proto;
        {
            std::fstream input(path, std::ios::in | std::ios::binary);
            if (!proto.ParseFromIstream(&input)) {
                std::stringstream ss;
                ss << "Failed to parse protobuf from " << path << " : " << strerror(errno);
                return MyStatus(error::INVALID_ARGUMENT, ss.str());
            }
        }
        // TODO: port metric printing code to print metrics using counter and config data images.
        DBG_LOG("path = {}", path);
        ret = this->PrintCSV(out, proto, printed_header);
        IF_BAD_STATUS_RETURN(ret);
    }

    return MyStatus::OK();
}

const std::regex GPUHwCounterSampler::FilenameRegex(R"(^GPUHwCounterSampleProto\..*\.proto$)");
bool GPUHwCounterSampler::IsProtoFile(const boost::filesystem::path& path) {
    std::smatch match;
    return std::regex_search(path.filename().string(), match, FilenameRegex);
    return false;
}


bool RangeTree::Push(const std::string& name, bool allow_insert) {
    // If there's an existing entry, set cur_node to it.
    // Otherwise create a new node, and set cur_node to it.
    assert(root != nullptr);
    auto it = cur_node->children.find(name);
    if (it == cur_node->children.end()) {
        if (allow_insert) {
          cur_node->children[name].reset(new RangeNode(cur_node, name));
          _UpdateStatsOnPush(true);
          cur_node = cur_node->children[name].get();
        }
        // This Push DID result in an insert.
        return true;
    }
    _UpdateStatsOnPush(false);
    cur_node = it->second.get();
    // This Push did not result in an insert.
    return false;
}

void RangeTree::Pop() {
    assert(cur_node != nullptr);
    cur_node = cur_node->parent;
    // Should at least point to the "root" RangeNode.
    // Otherwise, user of class probably called pop() too many times.
    assert(cur_node != nullptr);
    _UpdateStatsOnPop();
}

std::list<const RangeNode*> RangeTree::CurStack() const {
  RangeNode* node = cur_node;
  std::list<const RangeNode*> stack;
  while (node != nullptr) {
    stack.push_back(node);
    node = node->parent;
  }
  stack.reverse();
  return stack;
}

void RangeTree::_UpdateStatsOnPush(bool was_insert) {
    cur_depth += 1;
    max_nesting_levels = std::max(cur_depth, max_nesting_levels);
    if (was_insert) {
        max_unique_ranges += 1;
    }
}

void RangeTree::_UpdateStatsOnPop() {
    cur_depth -= 1;
}

MyStatus GPUHwCounterSampler::PrintCSV(
        std::ostream& out,
        const std::string& chipName,
        const uint8_t* counterDataImage,
        size_t counterDataImageSize,
        const std::vector<std::string>& metricNames,
        bool& printed_header,
        const std::vector<std::string>& extra_headers,
        const std::vector<std::string>& extra_fields) {

    if (counterDataImageSize == 0) {
        std::stringstream ss;
        ss << "Counter Data Image is empty!";
        return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }

//    auto md5_hash = calc_md5_hash(counterDataImage, counterDataImageSize);
//    out << "md5_hash = " << md5_hash << std::endl;

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = counterDataImage;
    NVPW_API_CALL_MAYBE_STATUS(NVPW_CounterData_GetNumRanges(&getNumRangesParams));

    std::vector<std::string> reqName;
    reqName.resize(metricNames.size());
    bool isolated = true;
    bool keepInstances = true;
    std::vector<const char*> metricNamePtrs;
    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
        ParseMetricNameString(metricNames[metricIndex], &reqName[metricIndex], &isolated, &keepInstances);
        metricNamePtrs.push_back(reqName[metricIndex].c_str());
    }

    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
        std::vector<const char*> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
        getRangeDescParams.pCounterDataImage = counterDataImage;
        getRangeDescParams.rangeIndex = rangeIndex;
        NVPW_API_CALL_MAYBE_STATUS(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

        descriptionPtrs.resize(getRangeDescParams.numDescriptions);

        getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
        NVPW_API_CALL_MAYBE_STATUS(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

        std::string rangeName;
        for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
        {
            if (descriptionIndex)
            {
                rangeName += "/";
            }
            rangeName += descriptionPtrs[descriptionIndex];
        }

        // const bool isolated = true;
        std::vector<double> gpuValues;
        gpuValues.resize(metricNames.size());

        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
        setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        setCounterDataParams.pCounterDataImage = counterDataImage;
        setCounterDataParams.isolated = true;
        setCounterDataParams.rangeIndex = rangeIndex;
        NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
        evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        evalToGpuParams.numMetrics = metricNamePtrs.size();
        evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
        evalToGpuParams.pMetricValues = &gpuValues[0];
        NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);

        if (!printed_header) {
            out << "range_name"
                << "," << "metric_name"
                << "," << "metric_value";
            for (const auto& header : extra_headers) {
                out << "," << header;
            }
            out << std::endl;
            printed_header = true;
        }

        for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
            out << rangeName
                << "," << metricNames[metricIndex]
                << "," << gpuValues[metricIndex];
            for (const auto& field : extra_fields) {
                out << "," << field;
            }
            out << std::endl;
        }
    }
    return MyStatus::OK();
}

GPUHwCounterSamplerProtoState GPUHwCounterSampler::AsProtoState() {
    GPUHwCounterSamplerProtoState proto_state(
            _next_trace_id
            , _directory
            , _dump_suffix
            , _chip_name
            , _metrics
            , state.config_data
            , std::move(_samples)
            , NumPasses()
    );
    this->_samples.clear();
    _next_trace_id += 1;
    _size_bytes = 0;
    return proto_state;
}

MyStatus GPUHwCounterSampler::AwaitDump() {
    if (!_enabled) {
        return MyStatus::OK();
    }
    // Wait for all outstanding work to complete.
    _pool.join();
    return MyStatus::OK();
}

size_t GPUHwCounterSampler::NumPasses() const {
//    CUpti_EventGroupSets *eventGroupSets = NULL;
//    size_t metricIdArraySize = sizeof(CUpti_MetricID) * numMetrics;
//    CUpti_MetricID metricIdArray = (CUpti_MetricID *)malloc(sizeof(CUpti_MetricID) * numMetrics);
//    // fill in metric Ids
//    cuptiMetricCreateEventGroupSets(context, metricIdArraySize, metricIdArray, &eventGroupSets);
//    passes = eventGroupSets->numSets;

    return this->UseMaxNestingLevels() * this->UseMaxUniqueRanges();
}

void GPUHwCounterSampler::SetDirectory(std::string& directory) {
  _directory = directory;
}
void GPUHwCounterSampler::SetDevice(int device) {
  _device = device;
}

GPUHwCounterSamplerMode GPUHwCounterSampler::Mode() const {
  return _mode;
}

} // namespace rlscope
