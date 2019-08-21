//
// Created by jagle on 8/19/2019.
//

#include <string>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <memory>

#include "cuda_api_profiler/cupti_logging.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/device_tracer.h"

#include "cuda_api_profiler/cuda_ld_preload.h"
#include "cuda_api_profiler/globals.h"
#include "cuda_api_profiler/get_env_var.h"

namespace tensorflow {

Globals globals;

Globals::Globals() {

  Status status = Status::OK();

#ifdef WITH_CUDA_LD_PRELOAD
  VLOG(1) << "dlopen(\"libcudart.so\")";
  cuda_library = GetCudaLibrary();
  VLOG(1) << "dlopen(\"libcudart.so\"): success!";
#endif

  std::ifstream cmdline_stream("/proc/self/cmdline");
  std::string cmdline((std::istreambuf_iterator<char>(cmdline_stream)),
                      std::istreambuf_iterator<char>());

  VLOG(1) << "Initialize globals\n"
          << "  CMD = " << cmdline;

  device_tracer = tensorflow::CreateDeviceTracer();
  auto IML_TRACE_AT_START = getenv("IML_TRACE_AT_START");
  VLOG(0) << "IML_TRACE_AT_START = " << IML_TRACE_AT_START;
  if (env_is_on("IML_TRACE_AT_START", false, true)) {
    VLOG(0) << "Starting tracing at program start (export IML_TRACE_AT_START=yes)";
    status = device_tracer->Start();
    MAYBE_LOG_ERROR(LOG(INFO), "DeviceTracerImpl::Start()", status);
    MAYBE_EXIT(status);
  }

}

Globals::~Globals() {
  // NOTE: some programs will close stdout/stderr BEFORE this gets called.
  // This will cause log message to be LOST.
  // HOWEVER, the destructor will still execute.
  // You can confirm this behaviour by creating a file.
  //
  // https://stackoverflow.com/questions/23850624/ld-preload-does-not-work-as-expected
  //
//  std::ofstream myfile;
//  myfile.open("globals.destructor.txt");
//  myfile << "Writing this to a file.\n";
//  myfile.close();

  VLOG(1) << "TODO: Stop tracing; collect traces";
  if (device_tracer->IsEnabled()) {
    VLOG(FATAL) << "Looks like DeviceTracer was still running... "
                << "please call sample_cuda_api.disable_tracing() in python BEFORE exiting to avoid stranger behavior in C++ destructors during library unload.";
  }
  // Dump CUDA API call counts and total CUDA API time to a protobuf file.
//    device_tracer->Collect();
}


}
