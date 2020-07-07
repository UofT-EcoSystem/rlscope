//
// Created by jgleeson on 2020-05-14.
//

#include "RangeSampling.h"
#include "ScopeExit.h"

#include "common_util.h"

#include "range_sampling/range_sampling.pb.h"

#include <spdlog/spdlog.h>
//#include <sys/types.h>
// Must be included in order operator<< to work with spd logging.
// https://github.com/gabime/spdlog#user-defined-types
#include "spdlog/fmt/ostr.h"

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
#include <boost/algorithm/string/join.hpp>

using boost::uuids::detail::md5;

#include <iostream>
#include <fstream>

// #define LOG_FUNC_ENTRY()
//   if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
//     RLS_LOG("GPU_HW_FUNC", "{}", "");
//   }

#define MY_LOG_FUNC_CALL(os, ...) rlscope::log_func_call_impl(os, "blort" __VA_OPT__(,) __VA_ARGS__)
    // MY_LOG_FUNC_CALL(ss, __VA_OPT__(,) __VA_ARGS__);

#define MY_RLS_LOG_FUNC(TAG, ...)                           \
  {                                                         \
    std::stringstream ss;                                   \
    rlscope::log_func_call_impl(ss, __func__, __VA_ARGS__); \
    RLS_LOG(TAG, "{}", ss.str());                           \
  }

// #define LOG_FUNC_ENTRY(TAG, ...)
//   {
//     std::stringstream ss;
//     rlscope::log_func_call_impl(ss, __func__, ##__VA_ARGS__);
//     RLS_LOG(TAG, "{}", ss.str());
//   }

// #define LOG_FUNC_ENTRY(...) MY_RLS_LOG_FUNC("GPU_HW_TRACE", __VA_ARGS__)
// #define LOG_FUNC_ENTRY(...) RLS_LOG_FUNC("GPU_HW_TRACE")
#define LOG_FUNC_ENTRY(...) 

namespace rlscope {

std::vector<std::string> get_DEFAULT_METRICS() {
  // - Parsed by: ParseMetricNameString
  //   - <metric_name>[$|&][+]
  //
  //   - default if NO symbols:
  //     keepInstances = false
  //     isolated = true
  //
  //   - keepInstances = "+" present
  //     isolated = "&" is NOT present
  //     (NOTE $ is redundant? it make isolated=True, but isolated=True is the default).
  std::vector<std::string> DEFAULT_METRICS = {

      // keepInstances = true
      // isolated = true

      //
      // NOTE: To figure out useful metrics to collect, grep CUPTI-samples/userrange_profiling/*.txt files
      // for deprecated CUPTI metrics that ACTUALLY HAVE DOCUMENTATION STRINGS (unlike new "profiling API"...),
      // then lookup the mapping from old metric names to new "Profiling API" metric name using this table from
      // the CUPTI documentation:
      //    https://docs.nvidia.com/cupti/Cupti/r_main.html#metrics_map_table_70
      //

      // Deprecated CUPTI metric API -- achieved_occupancy:
      //    Id        = 1205
      //    Shortdesc = Achieved Occupancy
      //    Longdesc  = Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
      "sm__warps_active.avg.pct_of_peak_sustained_active+",

      // Deprecated CUPTI metric API -- sm_efficiency:
      //    Id        = 1203
      //    Shortdesc = Multiprocessor Activity
      //    Longdesc  = The percentage of time at least one warp is active on a multiprocessor averaged over all multiprocessors on the GPU
      // See CUPTI documentation for mapping to new "Profiling API" metric name:
      //    https://docs.nvidia.com/cupti/Cupti/r_main.html#metrics_map_table_70
      "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed+",

      // Deprecated CUPTI metric API -- inst_executed:
      //    Metric# 90
      //    Id        = 1290
      //    Name      = inst_executed
      //    Shortdesc = Instructions Executed
      //    Longdesc  = The number of instructions executed
      "smsp__inst_executed.sum+",

      // Deprecated CUPTI metric API -- active_cycles:
      //    Event# 25
      //    Id        = 2629
      //    Name      = active_cycles
      //    Shortdesc = Active cycles
      //    Longdesc  = Number of cycles a multiprocessor has at least one active warp.
      //    Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION
      "sm__cycles_active.sum+",

      // Deprecated CUPTI metric API -- active_warps:
      //    Event# 26
      //    Id        = 2630
      //    Name      = active_warps
      //    Shortdesc = Active warps
      //    Longdesc  = Accumulated number of active warps per cycle. For every cycle it increments by the number of active warps in the cycle which can be in the range 0 to 64.
      //    Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION
      "sm__warps_active.sum+",

      // Deprecated CUPTI metric API -- elapsed_cycles_sm:
      //    Event# 33
      //    Id        = 2193
      //    Name      = elapsed_cycles_sm
      //    Shortdesc = Elapsed clocks
      //    Longdesc  = Elapsed clocks
      //    Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION
      "sm__cycles_elapsed.sum+"

//    // FAILS:
//    //   ERROR: Invalid argument: /home/jgleeson/clone/iml/src/libs/range_sampling/RangeSampling.cpp:384: error:
//    //   function NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams) failed with error
//    //   (1) NVPA_STATUS_ERROR: Generic error.
//    // keepInstances = true
//    // isolated = false
//    "sm__warps_active.avg.pct_of_peak_sustained_active&+",
//    "smsp__inst_executed.sum&+",
//    "sm__cycles_active.sum&+",
//    "sm__warps_active.sum&+",
//    "sm__cycles_elapsed.sum&+"

//    // keepInstances = false
//    // isolated = true
//    // No difference?
//    "sm__warps_active.avg.pct_of_peak_sustained_active",
//    "smsp__inst_executed.sum",
//    "sm__cycles_active.sum",
//    "sm__warps_active.sum",
//    "sm__cycles_elapsed.sum"

  };
  return DEFAULT_METRICS;
}
std::string get_DEFAULT_METRICS_STR() {
  auto DEFAULT_METRICS = get_DEFAULT_METRICS();
  return boost::algorithm::join(DEFAULT_METRICS, ",");
}

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
  counterDataImageOptions.maxRangeNameLength = counter_data_maxRangeNameLength;
  // counterDataImageOptions.maxRangeNameLength = 64;

  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    std::stringstream ss;
    ss << "Runtime info: "
       << std::endl
       << "  counterDataImageOptions.maxNumRanges = " << counterDataImageOptions.maxNumRanges
       << std::endl
       << "  counterDataImageOptions.maxNumRangeTreeNodes = " << counterDataImageOptions.maxNumRangeTreeNodes
       << std::endl
       << "  counterDataImageOptions.maxRangeNameLength = " << counterDataImageOptions.maxRangeNameLength;
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

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
  MyStatus status = MyStatus::OK();
  std::string reqName;
  bool isolated = true;
  bool keepInstances = true;

  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    std::stringstream ss;
    ss << "metricNames = ";
//    std::vector<std::string> metricNamesCopy = metricNames;
//    PrintValue(ss, metricNamesCopy);
    PrintValue(ss, metricNames);
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

  for (auto& metricName : metricNames)
  {
    ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
    /* Bug in collection with collection of metrics without instances, keep it to true*/
    keepInstances = true;
    NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
    getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
    getMetricPropertiesBeginParams.pMetricName = reqName.c_str();

    if (rlscope::TRACE_CUDA) {
      std::stringstream ss;
      rlscope::log_func_call_impl(ss, "NVPW_MetricsContext_GetMetricProperties_Begin", reqName);
      RLS_LOG("CUDA_API_TRACE", "{}", ss.str());
    }
//    NVPW_API_CALL_MAYBE_STATUS_SILENT(NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams));
    NVPA_Status nvpa_ret = NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams);
    if (nvpa_ret != NVPA_STATUS_SUCCESS) {
      std::stringstream err_ss;
      auto err_str = rlscope::nvperfGetErrorString(nvpa_ret);
      err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << "NVPW_MetricsContext_GetMetricProperties_Begin" << " failed for metric=" << reqName << " with error (" << nvpa_ret << ") " << err_str;
      auto my_status = MyStatus(rlscope::error::INVALID_ARGUMENT, err_ss.str());
      return my_status;
    }

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

    if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
      std::stringstream ss;
      ss << "CONFIG: metric = " << rawMetricName
         << std::endl << "  isolated = " << isolated
         << std::endl << "  keepInstances = " << keepInstances;
      RLS_LOG("GPU_HW", "{}", ss.str());
//      RLS_LOG()
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
  MyStatus status = MyStatus::OK();
  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
  metricsContextCreateParams.pChipName = chipName.c_str();
  NVPW_API_CALL_MAYBE_STATUS(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
  metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
  SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  std::vector<std::string> temp;
  status = GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);
  IF_BAD_STATUS_RETURN(status);

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
  MyStatus status = MyStatus::OK();

  NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
  metricsContextCreateParams.pChipName = chipName.c_str();
  NVPW_API_CALL_MAYBE_STATUS(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

  NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
  metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
  SCOPE_EXIT([&]() { NVPW_MetricsContext_Destroy((NVPW_MetricsContext_Destroy_Params *)&metricsContextDestroyParams); });

  std::vector<NVPA_RawMetricRequest> rawMetricRequests;
  std::vector<std::string> temp;
  status = GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);
  IF_BAD_STATUS_RETURN(status);

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

  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    std::stringstream ss;
    ss << "configImage.size() = " << config_data.configImage.size();
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

  CUpti_Profiler_SetConfig_Params setConfigParams = {CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  setConfigParams.pConfig = &config_data.configImage[0];
  setConfigParams.configSize = config_data.configImage.size();
  setConfigParams.passIndex = 0;
  setConfigParams.minNestingLevel = 1;
  setConfigParams.numNestingLevels = config_data.counter_data_max_num_nesting_levels;

  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    std::stringstream ss;
    ss << "Using setConfigParams.numNestingLevels = " << setConfigParams.numNestingLevels;
    RLS_LOG("GPU_HW", "{}", ss.str());
  }
  // When experimenting with achieved_occupancy_range_profiling, I observed segfaults with config_data.counter_data_max_num_nesting_levels >= 12...
  // I have NO IDEA why...
  // TODO: post on nvidia forums and ask what's going on (reference ORIGINAL CUPTI sample program and how to reproduce the problem).
  assert(setConfigParams.numNestingLevels < 12);

  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerSetConfig(&setConfigParams));

  return MyStatus::OK();
}

MyStatus CUPTIProfilerState::StartProfiling(ConfigData& config_data, CounterData& counter_data) {
  MyStatus ret = MyStatus::OK();

  // Already called this; fail fast to avoid weird CUPTI errors.
  assert(!_profiler_running);
  assert(!_pass_running);

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

  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    std::stringstream ss;
    ss << "Runtime info: "
       << std::endl
       << "  beginSessionParams.maxRangesPerPass = " << beginSessionParams.maxRangesPerPass
       << std::endl
       << "  beginSessionParams.maxLaunchesPerPass = " << beginSessionParams.maxLaunchesPerPass;
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

  // NOTE: if we attempt to profile multiple processes at once, the first process will succeed here,
  // but the next ones will fail.
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerBeginSession(&beginSessionParams));
  // DBG_LOG("{}", "cuptiProfilerBeginSession WORKED");

  ret = _InitConfig(config_data);
  IF_BAD_STATUS_RETURN(ret);

  _profiler_running = true;

  return MyStatus::OK();
}

MyStatus CUPTIProfilerState::StopProfiling(ConfigData& config_data, CounterData& counter_data) {
  if (!_profiler_running) {
    return MyStatus::OK();
  }

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerUnsetConfig(&unsetConfigParams));

  CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerEndSession(&endSessionParams));

  _profiler_running = false;

  return MyStatus::OK();
}

bool CUPTIProfilerState::HasNextPass() const {
  return !_endPassParams_allPassesSubmitted;
}

MyStatus CUPTIProfilerState::StartPass(ConfigData& config_data) {
  MyStatus ret = MyStatus::OK();
  assert(!_pass_running);
  _pass_running = true;

  CUpti_Profiler_BeginPass_Params beginPassParamsLocal = {CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerBeginPass(&beginPassParamsLocal));

  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerEnableProfiling(&enableProfilingParams));

  return MyStatus::OK();
}

MyStatus CUPTIProfilerState::EndPass() {
  if (!_pass_running || !_profiler_running) {
    return MyStatus::OK();
  }

  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerDisableProfiling(&disableProfilingParams));

  CUpti_Profiler_EndPass_Params endPassParamsLocal = {CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
  CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerEndPass(&endPassParamsLocal));
  _endPassParams_allPassesSubmitted = endPassParamsLocal.allPassesSubmitted;

  _pass_running = false;

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
//  if (!RLS_GPU_HW_SKIP_PROF_API) {
    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_EXIT(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
//  }

  if (_context != nullptr) {
    cuCtxDestroy(_context);
    _context = nullptr;
  }
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
  // LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  assert(_device != -1);
  if (_initialized_device == _device) {
    // We've already initialized profiling for this device.
    if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
      DBG_LOG("SKIP: GPUHwCounterSampler already initialized with device={}, dir={}",
              _device,
              _directory);
    }
    return MyStatus::OK();
  }
  if (!_enabled) {
    return MyStatus::OK();
  }
  MyStatus ret = MyStatus::OK();

  // Initialize the CUDA driver API if it hasn't been already (e.g. during LD_PRELOAD).
  {
    int deviceCount;
    CUresult cu_ret = cuDeviceGetCount(&deviceCount);
    if (cu_ret == CUDA_ERROR_NOT_INITIALIZED) {
      // cuInit(0) hasn't been called yet; call it so we can make CUDA API calls.
      RLS_LOG("GPU_HW", "Initializing CUDA driver API with cuInit(0)", _device);
      DRIVER_API_CALL_MAYBE_STATUS(cuInit(0));
    } else {
      // cuDeviceGetCount failed with an error we don't know how to handle...
      // Call it again to report the error:
      DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGetCount(&deviceCount));
    }
  }

  {
    // CUcontext _context = nullptr;
    // FAILS during api.set_metadata with initialization error (3)
    DRIVER_API_CALL_MAYBE_STATUS(cuCtxGetCurrent(&_context));
    if (_context == nullptr) {
      std::stringstream ss;
      RLS_LOG("GPU_HW", "Created CUDA context for device={} since it didn't already exist", _device);
      DRIVER_API_CALL_MAYBE_STATUS(cuCtxCreate(&_context, 0, _device));
//      ss << "GPUHwCounterSampler: no CUDA context has been created yet";
//      return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
  }

  // Programmer error; you haven't called SetDevice.
  assert(_device >= 0);

  ret = CheckCUPTIProfilingAPISupported();
  IF_BAD_STATUS_RETURN(ret);

//    CUdevice cuDevice;
//    DRIVER_API_CALL_MAYBE_STATUS(cuDeviceGet(&cuDevice, _device));

//  if (!RLS_GPU_HW_SKIP_PROF_API) {
    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerInitialize(&profilerInitializeParams));
    ret = GetGPUChipName(_device, &_chip_name);
    IF_BAD_STATUS_RETURN(ret);

    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    NVPW_API_CALL_MAYBE_STATUS(NVPW_InitializeHost(&initializeHostParams));
//  }

  _initialized = true;
  _initialized_device = _device;

  return MyStatus::OK();
}

const char* GPUHwCounterSampler::ModeString(GPUHwCounterSamplerMode mode) {
  switch (mode) {
    case PROFILE:
      return "PROFILE";
    case CONFIG:
      return "CONFIG";
    case EVAL:
      return "EVAL";
    default:
      assert(false);
      return "";
  }
}


MyStatus GPUHwCounterSampler::Push(const std::string &operation) {
  LOG_FUNC_ENTRY(operation);
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__, operation);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }
  
  MyStatus status = MyStatus::OK();
  if (!_enabled || !_running_pass) {
    return MyStatus::OK();
  }

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  _Check();
#endif

  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);

  bool update_stats = (_mode == CONFIG);
  status = _range_tree.Push(operation, update_stats);
  IF_BAD_STATUS_RETURN(status);

  // if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
  //   std::stringstream ss;
  //   const char* mode_str = GPUHwCounterSampler::ModeString(_mode);
  //   ss << "mode=" << mode_str << ", ";
  //   ss << "Push: operation = \"" << operation << "\"";
  //   RLS_LOG("GPU_HW", "{}", ss.str());
  // }

  if (_mode == PROFILE) {
    if (!RLS_GPU_HW_SKIP_PROF_API) {
      CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
      pushRangeParams.pRangeName = operation.c_str();
      pushRangeParams.rangeNameLength = operation.size();
      // assert(pushRangeParams.pRangeName[pushRangeParams.rangeNameLength] == '\0');
      assert(pushRangeParams.pRangeName[operation.size()] == '\0');
      if (rlscope::TRACE_CUDA) {
        std::stringstream ss;
        rlscope::log_func_call_impl(ss, "cuptiProfilerPushRange", operation);
        RLS_LOG("CUDA_API_TRACE", "{}", ss.str());
      }
      CUPTI_API_CALL_MAYBE_EXIT_SILENT(cuptiProfilerPushRange(&pushRangeParams));
    }
  }

  return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::Pop() {
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled || !_running_pass) {
    return MyStatus::OK();
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);
  _range_tree.Pop();
  if (_mode == PROFILE) {
    if (!RLS_GPU_HW_SKIP_PROF_API) {
      CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
      CUPTI_API_CALL_MAYBE_STATUS(cuptiProfilerPopRange(&popRangeParams));
    }
  }

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  _Check();
#endif

  return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::Disable() {
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  _enabled = false;
  return MyStatus::OK();
}
bool GPUHwCounterSampler::Enabled() const {
  return _enabled;
}
MyStatus GPUHwCounterSampler::_CheckInitialized(const char* file, int lineno, const char* func) const {
  if (!_initialized) {
    std::stringstream ss;
    ss << "You forgot to call GPUHwCounterSampler::Init() before calling GPUHwCounterSampler::" << func << " @ " << file << ":" << lineno;
    return MyStatus(error::INTERNAL, ss.str());
  }
  return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::StartConfig(const std::vector<std::string>& metrics) {
  LOG_FUNC_ENTRY(metrics);
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__, metrics);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled) {
    return MyStatus::OK();
  }

  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);

  _mode = CONFIG;
  if (metrics.size() <= 0) {
    std::stringstream ss;
    ss << "GPUHwCounterSampler.StartConfig(...) needs at least one metric, but got none";
    return MyStatus(error::INTERNAL, ss.str());
  }
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
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled) {
    return MyStatus::OK();
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);
  MyStatus ret = MyStatus::OK();

  if (_running_pass) {
    ret = EndPass();
    IF_BAD_STATUS_RETURN(ret);
  }

  _mode = PROFILE;

  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    std::stringstream ss;
    ss << "GPU HW configuration:"
       << std::endl << "  MaxNestingLevels = " << MaxNestingLevels()
       << std::endl << "  MaxNumRanges = " << MaxNumRanges()
       << std::endl << "  NumPasses = " << NumPasses();
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

  if (MaxNestingLevels() == 0 || MaxNumRanges() == 0) {
    std::stringstream ss;
    ss << "Didn't see any operation annotations " <<
       "(MaxNestingLevels = " << MaxNestingLevels() <<
       ", MaxNumRanges = " << MaxNumRanges() << ").";
    ss << "\n";
    ss << "Stacks seen during configuration passes:\n";
    _range_tree.PrintStacks(ss, 1);
    // RLS_LOG("GPU_HW", "{}", ss.str());
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }

  ret = _InitSamplerState();
  IF_BAD_STATUS_RETURN(ret);

//  if (!RLS_GPU_HW_SKIP_PROF_API) {
    ret = profiler_state.StartProfiling(state.config_data, state.counter_data);
    IF_BAD_STATUS_RETURN(ret);
//  }

  return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::_InitSamplerState() {
  MyStatus ret = MyStatus::OK();
//    _pass_idx = 0;
  if (MaxNestingLevels() == 0) {
    std::stringstream ss;
    ss << "WARNING: did you forget to call StartConfig()? MaxNestingLevels() == " << MaxNestingLevels();
    RLS_LOG("GPU_HW", "{}", ss.str());
  }
  if (MaxNumRanges() == 0) {
    std::stringstream ss;
    ss << "WARNING: did you forget to call StartConfig()? MaxNumRanges() == " << MaxNumRanges();
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

//  if (!RLS_GPU_HW_SKIP_PROF_API) {
    profiler_state = CUPTIProfilerState(/*counter_data_max_num_ranges=*/UseMaxNumRanges());

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
        /*counter_data_max_num_ranges=*/UseMaxNumRanges(),
        /*counter_data_maxRangeNameLength=*/UseMaxRangeNameLength());
    ret = state.counter_data.Init();
    IF_BAD_STATUS_RETURN(ret);

    ret = _NextSamplerState();
    IF_BAD_STATUS_RETURN(ret);
//  }

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
  if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
    RLS_LOG("GPU_HW", "Dumped {}", path);
  }
  return MyStatus::OK();
}

MyStatus GPUHwCounterSampler::_MaybeRecordSample(bool* recorded) {
  MyStatus ret = MyStatus::OK();
  *recorded = false;
  if (!this->CanRecord()) {
    return MyStatus::OK();
  }

//  if (!RLS_GPU_HW_SKIP_PROF_API) {
    ret = profiler_state.Flush(state.config_data, state.counter_data);
    IF_BAD_STATUS_RETURN(ret);
//  }

  if (!RLS_GPU_HW_SKIP_PROF_API) {
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
  }
  return MyStatus::OK();
}


MyStatus GPUHwCounterSampler::StopProfiling() {
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled) {
    return MyStatus::OK();
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);
  MyStatus ret = MyStatus::OK();

  if (_running_pass) {
    ret = EndPass();
    IF_BAD_STATUS_RETURN(ret);
  }

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
    std::stringstream ss;
    ss << "WARNING: GPU hw counter didn't run for enough passes so the last sample will be discarded; "
       << "only ran for " << (_pass_idx + 1) << " passes but likely needed at least " << UseMaxNumRanges();
    RLS_LOG("GPU_HW", "{}", ss.str());
  }

//  if (!RLS_GPU_HW_SKIP_PROF_API) {
    ret = profiler_state.StopProfiling(state.config_data, state.counter_data);
    IF_BAD_STATUS_RETURN(ret);
//  }

  return MyStatus::OK();
}
bool GPUHwCounterSampler::HasNextPass() const {
  MyStatus status = MyStatus::OK();
  if (!_enabled) {
    // Q: What's the right behaviour here...?
    return false;
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_EXIT_WITH(status);
  return profiler_state.HasNextPass();
}
bool GPUHwCounterSampler::CanRecord() const {
  MyStatus status = MyStatus::OK();
  if (!_enabled) {
    return false;
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_EXIT_WITH(status);
//    DBG_LOG("!HasNextPass() = {}, state.CanDump() = {}", !HasNextPass(), state.CanRecord());
  return !HasNextPass() && state.CanDump();
}
bool GPUHwCounterSampler::CanDump() const {
  MyStatus status = MyStatus::OK();
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_EXIT_WITH(status);
//    DBG_LOG("!HasNextPass() = {}, state.CanDump() = {}", !HasNextPass(), state.CanRecord());
  return _samples.size() > 0 && !CanRecord();
}
bool GPUHwCounterSampler::ShouldDump() const {
  MyStatus status = MyStatus::OK();
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_EXIT_WITH(status);
//    DBG_LOG("!HasNextPass() = {}, state.CanDump() = {}", !HasNextPass(), state.CanRecord());
  return CanDump() && this->size_bytes() >= GPUHwCounterSampler::MaxSampleFileSizeBytes;
}
MyStatus GPUHwCounterSampler::StartPass() {
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled || !HasNextPass()) {
    return MyStatus::OK();
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);

  _range_tree.StartPass(/*update_stats=*/_mode == CONFIG);

  MyStatus ret = MyStatus::OK();
  if (_mode == PROFILE) {

//    if (!RLS_GPU_HW_SKIP_PROF_API) {
      ret = profiler_state.StartPass(state.config_data);
      IF_BAD_STATUS_RETURN(ret);
//    }

  }
  _running_pass = true;
  return MyStatus::OK();
}
MyStatus GPUHwCounterSampler::EndPass() {
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled || !_running_pass) {
    return MyStatus::OK();
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);

  _range_tree.EndPass(/*update_stats=*/_mode == CONFIG);

  MyStatus ret = MyStatus::OK();

  if (_mode == PROFILE) {
    _pass_idx += 1;
    // GOAL: if we try to run too many passes, then just "turn off" GPU HW sampler.
    // NOTE: The number of passes we need as we increase the number of hw-counters is BIGGER than NumPasses()
    // (i.e., it's WRONG).
//    assert(_pass_idx <= this->NumPasses());

    if (!RLS_GPU_HW_SKIP_PROF_API) {
      if (SHOULD_DEBUG(FEATURE_GPU_HW)) {
        size_t num_ranges_collected = 0;
        ret = state.counter_data.getNumRangeCollected(&num_ranges_collected);
        IF_BAD_STATUS_RETURN(ret);

        std::stringstream ss;
        ss << "After pass " << _pass_idx << ":"
           << std::endl << "  total_num_ranges_collected = " << num_ranges_collected;
        // << std::endl << "  increase in total_num_ranges_collected since last pass = " << (num_ranges_collected - last_num_ranges_collected);
        RLS_LOG("GPU_HW", "{}", ss.str());
      }
    }

//    if (!RLS_GPU_HW_SKIP_PROF_API) {
      ret = profiler_state.EndPass();
      IF_BAD_STATUS_RETURN(ret);
//    }

  }
  _running_pass = false;

#ifdef CONFIG_CHECK_PROF_BUFFER_OVERFLOW
  _Check();
#endif

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
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  if (!_enabled) {
    return MyStatus::OK();
  }
  return _Dump(/*sync=*/true);
}
MyStatus GPUHwCounterSampler::DumpAsync() {
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  if (!_enabled) {
    return MyStatus::OK();
  }
  return _Dump(/*sync=*/false);
}
MyStatus GPUHwCounterSampler::_Dump(bool sync) {
  MyStatus status = MyStatus::OK();
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);
  MyStatus ret = MyStatus::OK();
  if (_mode == PROFILE) {
    if (!CanDump()) {
      std::stringstream ss;
      ss << "Haven't collected enough GPU hw counter samples to dump anything (samples = " << _samples.size() << "); "
         << "only ran for " << (_pass_idx + 1) << " passes but likely needed at least " << UseMaxNumRanges() << " to be able to record one sample";
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
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

  MyStatus status = MyStatus::OK();
  if (!_enabled || RLS_GPU_HW_SKIP_PROF_API) {
    return MyStatus::OK();
  }
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);
  MyStatus ret = MyStatus::OK();
  if (_mode == PROFILE) {
    if (HasNextPass()) {
      std::stringstream ss;
      ss << "GPU hw counter didn't run for enough passes to collect a full sample; "
         << "only ran for " << (_pass_idx + 1) << " passes but likely needed at least " << UseMaxNumRanges();
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
  MyStatus status = MyStatus::OK();
  // IMPORTANT: printing of metrics requires some intitialization (NVPW_InitializeHost specifically).
  status = _CheckInitialized(__FILE__, __LINE__, __FUNCTION__);
  IF_BAD_STATUS_RETURN(status);

  MyStatus ret = MyStatus::OK();
  std::list<std::string> paths;
  ret = RecursiveFindFiles(&paths, _directory, [&] (const boost::filesystem::path& bpath) {
    if (!boost::filesystem::is_regular_file(bpath)) {
      return false;
    }
    return this->IsProtoFile(bpath);
  });
  IF_BAD_STATUS_RETURN(ret);

  {
    std::stringstream ss;
    ss << "GPU HW trace files:";
    for (const auto& path : paths) {
      ss << std::endl;
      PrintIndent(ss, 1);
      ss << path;
    }
    RLS_INFO("GPU_HW", "{}", ss.str());
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
    // DBG_LOG("path = {}", path);
    ret = this->PrintCSV(out, proto, printed_header);
    IF_BAD_STATUS_RETURN(ret);
  }

  return MyStatus::OK();
}

const std::regex GPUHwCounterSampler::FilenameRegex(R"(^GPUHwCounterSampleProto\..*\.proto$)");
bool GPUHwCounterSampler::IsProtoFile(const boost::filesystem::path& path) {
  std::smatch match;
  return std::regex_search(path.filename().string(), match, FilenameRegex);
}

const RangeTreeStats& RangeTree::RecordedStats() const {
  // If this fails, we forgot to call RangeTree::EndPass (called from GPUHwCounterSampler::EndPass).
  assert(recorded_stats.initialized);
  return recorded_stats;
}

void RangeTree::StartPass(bool update_stats) {
  // if (update_stats) {
  cur_num_ranges = 0;
  // }
}
void RangeTree::EndPass(bool update_stats) {
  if (update_stats) {
    _RecordStats();
  }
}

// MyStatus RangeTree::CanPush(const std::string& name) {
//   // If there's an existing entry, set cur_node to it.
//   // Otherwise create a new node, and set cur_node to it.
//   assert(root != nullptr);
//   auto it = cur_node->children.find(name);
//   if (it == cur_node->children.end()) {
//     if (allow_insert) {
//       cur_node->children[name].reset(new RangeNode(cur_node, name));
//       cur_node = cur_node->children[name].get();
//       _UpdateStatsOnPush(true);
//     }
//     // This Push DID result in an insert.
//     return true;
//   }
//   cur_node = it->second.get();
//   _UpdateStatsOnPush(false);
//   // This Push did not result in an insert.
//   return false;
// }
MyStatus RangeTree::Push(const std::string& name, bool update_stats) {
  // If there's an existing entry, set cur_node to it.
  // Otherwise create a new node, and set cur_node to it.
  assert(root != nullptr);
  if (!update_stats) {
    // If this fails, we forgot to call RangeTree::EndPass.
    assert(recorded_stats.initialized);
  }
  auto it = cur_node->children.find(name);
  if (it == cur_node->children.end()) {
    if (update_stats) {
      cur_node->children[name].reset(new RangeNode(cur_node, name));
      cur_node = cur_node->children[name].get();
      _UpdateStatsOnPush(true);
      return MyStatus::OK();
    }
    // This Push would result in an insert, but our caller DOESN'T want that (it's not a configuration pass).
    std::stringstream ss;
    ss << "GPUHwCounterSampler: Tried to push operation=\"" << name << "\", but we never saw this operation during the configuration pass.  "
       << "Operation stack at time of push was:\n";
    this->PrintStack(ss, 1, this->CurStack());
    ss << "\n";
    ss << "Stacks seen during configuration passes:\n";
    this->PrintStacks(ss, 1);
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  cur_node = it->second.get();
  _UpdateStatsOnPush(false);
  if (!update_stats && cur_num_ranges > recorded_stats.max_num_ranges) {
    std::stringstream ss;
    ss << "GPUHwCounterSampler: Tried to push operation=\"" << name << "\", but this will exceed the total number of operations (Push() calls) "
       << "seen during the configuration pass, which was " << recorded_stats.max_num_ranges << ".  "
       << "Operation stack at time of push was:\n";
    this->PrintStack(ss, 1, this->CurStack());
    ss << "\n";
    ss << "Stacks seen during configuration passes:\n";
    this->PrintStacks(ss, 1);
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  // This Push did not result in an insert.
  return MyStatus::OK();
}

void RangeTree::Pop() {
  assert(cur_node != nullptr);
  _UpdateStatsOnPop();
  cur_node = cur_node->parent;
  // Should at least point to the "root" RangeNode.
  // Otherwise, user of class probably called pop() too many times.
  assert(cur_node != nullptr);
}

RangeTree::Stack RangeTree::CurStack() const {
  RangeNode* node = cur_node;
  RangeTree::Stack stack;
  while (node != root.get()) {
    stack.push_back(node);
    node = node->parent;
  }
  stack.reverse();
  return stack;
}

void RangeTree::EachStackSeen(EachStackSeenCb func) const {
  for (const auto& pair : root->children) {
    RangeTree::Stack stack;
    RangeNode* node = pair.second.get();
    assert(node != nullptr);
    stack.push_back(node);
    _EachStackSeen(node, stack, func);
    stack.pop_back();
  }
}

void RangeTree::_EachStackSeen(
    RangeNode* node,
    RangeTree::Stack stack,
    EachStackSeenCb func) const {
  assert(node != nullptr);
  if (node->children.size() == 0) {
    // Leaf node.
    func(stack);
  } else {
    for (const auto& pair : cur_node->children) {
      node = pair.second.get();
      stack.push_back(node);
      _EachStackSeen(node, stack, func);
      stack.pop_back();
    }
  }
}

void RangeTree::_RecordStats() {
  recorded_stats = stats;
  recorded_stats.initialized = true;
}

void RangeTree::_UpdateStatsOnPush(bool was_insert) {
  cur_depth += 1;
  cur_range_name_length += cur_node->name.size();
  cur_num_ranges += 1;
  stats.max_nesting_levels = std::max(cur_depth, stats.max_nesting_levels);
  stats.max_range_name_length = std::max(CurRangeNameLength(), stats.max_range_name_length);
  stats.max_num_ranges = std::max(cur_num_ranges, stats.max_num_ranges);
  if (was_insert) {
    stats.max_unique_ranges += 1;
  }
}

void RangeTree::_UpdateStatsOnPop() {
  assert(cur_depth > 0);
  assert(cur_range_name_length >= cur_node->name.size());
  cur_range_name_length -= cur_node->name.size();
  cur_depth -= 1;
}

RangeTreeStats::RangeTreeStats(const RangeTree& range_tree) {
  max_range_name_length = range_tree.CurRangeNameLength();
}

size_t RangeTree::CurRangeNameLength() const {
  // e.g. training_loop/q_forward
  // cur_range_name_length => sum of lengths of components (not including slash separators).
  // cur_depth             => slash separators.
  // +1                    => null terminator.
  size_t len = 0;
  len += cur_range_name_length;
  if (cur_depth > 0) {
    // One less slash than the number of components ( 0 slashes if 0 or 1 components ).
    len += cur_depth - 1;
  }
  // null terminator.
  len += 1;
//  {
//    std::stringstream ss;
//    ss << "CurRangeNameLength = " << len;
//    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
//  }
  return len;
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
  LOG_FUNC_ENTRY();
  if (SHOULD_DEBUG(FEATURE_GPU_HW_TRACE)) {
    std::stringstream ss;
    rlscope::log_func_call_impl(ss, __func__);
    RLS_LOG("GPU_HW_TRACE", "{}", ss.str());
  }

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

  return this->UseMaxNestingLevels() * this->UseMaxNumRanges();
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
