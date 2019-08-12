//
// Created by jagle on 8/6/2019.
//

#include <cuda.h>
#include <cupti.h>

#include <string>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <memory>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"

#include "sample_cuda_api_export.h"

namespace tensorflow {

class Globals {
public:
  Globals();
  ~Globals();

  std::unique_ptr<tensorflow::DeviceTracer> device_tracer;
};
Globals globals;

bool env_is_on(const char* var, bool dflt) {
  const char* val = getenv(var);
  if (val == nullptr) {
    return dflt;
  }
  std::string var_str(var);
  std::transform(
      var_str.begin(), var_str.end(), var_str.begin(),
      [](unsigned char c){ return std::tolower(c); });
  return var_str == "on"
         || var_str == "1"
         || var_str == "true"
         || var_str == "yes";
}

Globals::Globals() {

  std::ifstream cmdline_stream("/proc/self/cmdline");
  std::string cmdline((std::istreambuf_iterator<char>(cmdline_stream)),
                      std::istreambuf_iterator<char>());

  VLOG(1) << "Initialize globals\n"
          << "  CMD = " << cmdline;

  device_tracer = tensorflow::CreateDeviceTracer();
  if (env_is_on("IML_TRACE_AT_START", false)) {
    VLOG(0) << "Starting tracing at program start (export IML_TRACE_AT_START=yes)";
    device_tracer->Start();
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
  // Dump CUDA API call counts and total CUDA API time to a protobuf file.
//    device_tracer->Collect();
}

// #define CUPTI_CALL(call) ({
//      CUptiResult _status = call;
//      if (_status != CUPTI_SUCCESS) {
//        const char *errstr;
//        cuptiGetResultString(_status, &errstr);
//        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",
//                __FILE__, __LINE__, #call, errstr);
//      }
//      _status;
//  })

#define CUPTI_CALL(call) do { \
    CUptiResult _status = call; \
    if (_status != CUPTI_SUCCESS) { \
      const char *errstr; \
      cuptiGetResultString(_status, &errstr); \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr); \
    } \
    _status; \
} while (0);


//#define MAYBE_RETURN(cupti_status) do {
//    CUptiResult _status = cupti_status;
//    if (_status != CUPTI_SUCCESS) {
//        return ERROR;
//    }
//} while (0);

// Status initTrace()
// {
//   size_t attrValue = 0, attrValueSize = sizeof(size_t);
//   // Device activity record is created when CUDA initializes, so we
//   // want to enable it before cuInit() or any CUDA runtime call.
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE)));
//   // Enable all other activity record kinds.
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
//
//   // Register callbacks for buffer requests and for buffers completed by CUPTI.
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
//
//   // Get and set activity attributes.
//   // Attributes can be set by the CUPTI client to change behavior of the activity API.
//   // Some attributes require to be set before any CUDA context is created to be effective,
//   // e.g. to be applied to all device buffer allocations (see documentation).
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
//   printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
//   attrValue *= 2;
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
//
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
//   printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
//   attrValue *= 2;
//   MAYBE_RETURN(CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
//
//   MAYBE_RETURN(CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
// }

using StatusRet = tensorflow::error::Code;

extern "C" {

typedef enum SAMPLE_CUDA_API_EXPORT TF_Code {
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_UNAUTHENTICATED = 16,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15,
} TF_Code;

//using RetCode = TF_Code;
using RetCode = int;

RetCode SAMPLE_CUDA_API_EXPORT setup() {
  // Initialize global state.
  VLOG(1) << __func__;
  return tensorflow::Status::OK().code();
}

RetCode SAMPLE_CUDA_API_EXPORT enable_tracing() {
  // Enable call-backs.
  VLOG(1) << __func__;
  globals.device_tracer->Start();
  return tensorflow::Status::OK().code();
}

RetCode SAMPLE_CUDA_API_EXPORT is_enabled(int* retval) {
  VLOG(1) << __func__;
  if (globals.device_tracer->IsEnabled()) {
    *retval = 1;
  } else {
    *retval = 0;
  }
  return tensorflow::Status::OK().code();
}

RetCode SAMPLE_CUDA_API_EXPORT disable_tracing() {
  // Disable call-backs.
  VLOG(1) << __func__;
  globals.device_tracer->Stop();
  return tensorflow::Status::OK().code();
}

RetCode SAMPLE_CUDA_API_EXPORT collect() {
  // Collect traces (synchronously).
  VLOG(1) << __func__;
  globals.device_tracer->Stop();
  globals.device_tracer->Collect();
  return tensorflow::Status::OK().code();
}

}

}
