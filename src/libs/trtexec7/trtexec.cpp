/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <memory>

#include "nvToolsExt.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "tensorrt_common7.h"

//#ifdef RLS_ENABLE_HW_COUNTERS
//#error "RLS_ENABLE_HW_COUNTERS = ON"
//#else
//#error "RLS_ENABLE_HW_COUNTERS = OFF"
//#endif

#ifdef RLS_ENABLE_HW_COUNTERS
#include "range_sampling.h"
#endif
#include "common_util.h"
using rlscope::MyStatus;

using namespace nvinfer1;
using namespace sample;

int main(int argc, char** argv)
{
  backward::SignalHandling sh;

  // NOTE: If we DON'T call this, then the libnvToolsExt.so library WON'T be included...
//  if (0) {
  // std::string thread_name = "MainThread";
  // nvtxNameOsThreadA(pthread_self(), thread_name.c_str());
//  }

  // NOTE: If you only define SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG, this doesn't enable debug logging.
  // It just ensures that the SPDLOG_DEBUG statements are **compiled in**!
  // We still need to turn them on though!
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

  const std::string sampleName = "TensorRT.trtexec";

  auto sampleTest = sample::gLogger.defineTest(sampleName, argc, argv);

  sample::gLogger.reportTestStart(sampleTest);

  Arguments args = argsToArgumentsMap(argc, argv);
  AllOptions options;

  if (parseHelp(args))
  {
    AllOptions::help(std::cout);
    return EXIT_SUCCESS;
  }

  if (!args.empty())
  {
    bool failed{false};
    try
    {
      options.parse(args);

      if (!args.empty())
      {
        for (const auto& arg : args)
        {
          sample::gLogError << "Unknown option: " << arg.first << " " << arg.second << std::endl;
        }
        failed = true;
      }
    }
    catch (const std::invalid_argument& arg)
    {
      sample::gLogError << arg.what() << std::endl;
      failed = true;
    }

    if (failed)
    {
      AllOptions::help(std::cout);
      return sample::gLogger.reportFail(sampleTest);
    }
  }
  else
  {
    options.helps = true;
  }

  if (options.helps)
  {
    AllOptions::help(std::cout);
    return sample::gLogger.reportPass(sampleTest);
  }

  sample::gLogInfo << options;
  if (options.reporting.verbose)
  {
    sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
  }

  cudaCheck(cudaSetDevice(options.system.device));

  initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

  for (const auto& pluginPath : options.system.plugins)
  {
    sample::gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
    samplesCommon::loadLibrary(pluginPath);
  }

  size_t num_threads;
  if (options.inference.threads) {
    num_threads = options.inference.streams;
  } else {
    num_threads = 1;
  }
  InferenceEnvironment iEnv(num_threads);
  iEnv.engine = getEngine(options.model, options.build, options.system, sample::gLogError);
  if (!iEnv.engine)
  {
    sample::gLogError << "Engine set up failed" << std::endl;
    return sample::gLogger.reportFail(sampleTest);
  }
  if (options.inference.skip)
  {
    return sample::gLogger.reportPass(sampleTest);
  }

  if (options.build.safe && options.system.DLACore >= 0)
  {
    sample::gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
                        "then use dla_safety_runtime to run inference with saved DLA loadable, "
                        "or alternatively run with your own application" << std::endl;
    return sample::gLogger.reportFail(sampleTest);
  }

  if (options.reporting.profile ||
      //  !options.reporting.exportTimes.empty()
      !options.reporting.exportProfile.empty()
  )
  {
    iEnv.profiler.reset(new Profiler);
  }

  if (!setUpInference(iEnv, options.inference))
  {
    sample::gLogError << "Inference set up failed" << std::endl;
    return sample::gLogger.reportFail(sampleTest);
  }

  boost::filesystem::path profile_dir;
  if (!options.reporting.profile_dir.empty()) {
    profile_dir = options.reporting.profile_dir;
  } else if (!options.model.baseModel.model.empty()) {
    boost::filesystem::path model_dir = boost::filesystem::path(options.model.baseModel.model).parent_path();
    profile_dir = model_dir / (options.model.baseModel.model + ".profiling");
  } else if (!options.build.engine.empty()) {
    boost::filesystem::path model_dir = boost::filesystem::path(options.build.engine).parent_path();
    profile_dir = model_dir / (boost::filesystem::path(options.build.engine).filename().string() + ".profiling");
  } else {
    profile_dir = ".";
  }
  boost::filesystem::create_directories(profile_dir);

//  https://docs.nvidia.com/cupti/Cupti/r_limitations.html#r_limitations

  if (options.reporting.hw_counters && options.inference.graph) {
    sample::gLogError << "LIMITATION: CUPTI profiling does not work with the CUDA graph API (results in segfaults):"
                      << std::endl
                      << "  https://docs.nvidia.com/cupti/Cupti/r_limitations.html#r_limitations"
                      << std::endl
                      << "  \"Profiling is not supported for CUDA kernel nodes launched by a CUDA Graph\""
                      << std::endl;
    return sample::gLogger.reportFail(sampleTest);
  }

#ifdef RLS_ENABLE_HW_COUNTERS
  MyStatus status = MyStatus::OK();
  rlscope::GPUHwCounterSampler sampler(options.system.device, profile_dir.string(), "");
  if (!options.reporting.hw_counters) {
    status = sampler.Disable();
    IF_BAD_STATUS_EXIT_WITH(status);
  }

  status = sampler.Init();
  IF_BAD_STATUS_EXIT("Failed to initialize GPU hw counter profiler", status);

  status = sampler.StartConfig(options.reporting.hw_metrics);
  IF_BAD_STATUS_EXIT("Failed to configure GPU hw counter profiler", status);
#endif

  sample::gLogInfo << "Starting inference threads" << std::endl;

  std::vector<InferenceTrace> trace;
  auto run_pass = [&] (int pass_idx, bool is_config_pass) {

#ifdef RLS_ENABLE_HW_COUNTERS
    status = sampler.StartPass();
    IF_BAD_STATUS_EXIT("Failed to start GPU hw pass", status);

    status = sampler.Push("inference_iterations");
    IF_BAD_STATUS_EXIT("Failed to push range for GPU hw pass", status);
#endif

    if (is_config_pass) {
      sample::gLogInfo << "Config pass " << pass_idx << std::endl;
    } else {
      sample::gLogInfo << "Pass " << pass_idx << std::endl;
    }
    runInference(options.inference, iEnv, options.system.device, trace);

#ifdef RLS_ENABLE_HW_COUNTERS
    // "inference_iterations"
    status = sampler.Pop();
    IF_BAD_STATUS_EXIT("Failed to pop range for GPU hw pass", status);

    status = sampler.EndPass();
    IF_BAD_STATUS_EXIT("Failed to end GPU hw pass", status);
#endif

  };

#ifdef RLS_ENABLE_HW_COUNTERS
  if (sampler.Enabled()) {
    run_pass(0, true);
  }

  status = sampler.StartProfiling();
  IF_BAD_STATUS_EXIT("Failed to start GPU hw counter profiler", status);
#endif


#ifdef RLS_ENABLE_HW_COUNTERS
  int pass_idx = 0;
  do {
    run_pass(pass_idx, false);
    pass_idx += 1;
  } while (sampler.HasNextPass());
  assert(!sampler.HasNextPass());
#else
  for (int pass_idx = 0; pass_idx < options.inference.passes; pass_idx++) {
    run_pass(pass_idx, false);
  }
#endif

  // Print trace information only for the final pass.
  printPerformanceReport(trace, options.reporting, static_cast<float>(options.inference.warmup), options.inference.batch, sample::gLogInfo);

#ifdef RLS_ENABLE_HW_COUNTERS
  status = sampler.RecordSample();
  IF_BAD_STATUS_EXIT("Failed to record GPU hw counter sample", status);

  status = sampler.DumpSync();
  IF_BAD_STATUS_EXIT("Failed to dump GPU hw counter sample", status);

  if (sampler.Enabled()) {
    auto csv_path = profile_dir / "GPUHwCounterSampler.csv";
    bool printed_header = false;
    std::ofstream csv_f(csv_path.string(), std::ios::out | std::ios::trunc);
    if (csv_f.fail()) {
      std::cerr << "ERROR: Failed to write to GPU HW csv file @ " << csv_path.string() << " : " << strerror(errno) << std::endl;
      exit(EXIT_FAILURE);
    }
    status = sampler.PrintCSV(csv_f, printed_header);
    IF_BAD_STATUS_EXIT("Failed to print GPU hw sample files in csv format", status);
    std::cout << "Output GPU HW csv file @ " << csv_path.string() << std::endl;
  }
#endif

  if (options.reporting.output)
  {
    dumpOutputs(*iEnv.context.front(), *iEnv.bindings.front(), sample::gLogInfo);
  }
  if (!options.reporting.exportOutput.empty())
  {
    exportJSONOutput(*iEnv.context.front(), *iEnv.bindings.front(), options.reporting.exportOutput);
  }
  if (!options.reporting.exportTimes.empty())
  {
    exportJSONTrace(trace, options.reporting.exportTimes);
  }
  if (options.reporting.profile)
  {
    iEnv.profiler->print(sample::gLogInfo);
  }
  if (!options.reporting.exportProfile.empty())
  {
    iEnv.profiler->exportJSONProfile(options.reporting.exportProfile);
  }

  return sample::gLogger.reportPass(sampleTest);
}
