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

#ifdef WITH_CUDA_LD_PRELOAD
#include "cuda_api_profiler/cuda_ld_preload.h"
#endif

#include "cuda_api_profiler/cupti_logging.h"
#include "cuda_api_profiler/globals.h"

#include "sample_cuda_api_export.h"

namespace tensorflow {



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

RetCode SAMPLE_CUDA_API_EXPORT print() {
  VLOG(1) << __func__;
  auto status = globals.device_tracer->Print();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode SAMPLE_CUDA_API_EXPORT set_metadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  VLOG(1) << __func__
          << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  auto status = globals.device_tracer->SetMetadata(directory, process_name, machine_name, phase_name);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode SAMPLE_CUDA_API_EXPORT enable_tracing() {
  // Enable call-backs.
  VLOG(1) << __func__;
  auto status = globals.device_tracer->Start();
//  if (status.code() != Status::OK()) {
//    VLOG(0) << "iml-prof C++ API " << __func__ << " failed with: " << status;
//  }
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
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
  auto status = globals.device_tracer->Stop();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode SAMPLE_CUDA_API_EXPORT async_dump() {
  // Dump traces (asynchronously).
  VLOG(1) << __func__;
  Status status;
  status = globals.device_tracer->AsyncDump();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode SAMPLE_CUDA_API_EXPORT await_dump() {
  // Wait for async dump traces to complete.
  VLOG(1) << __func__;
  Status status;
  status = globals.device_tracer->AsyncDump();
  MAYBE_RETURN(status);
  status = globals.device_tracer->AwaitDump();
  // TODO: call device_tracer->Collect?
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

//RetCode SAMPLE_CUDA_API_EXPORT collect() {
//  // Collect traces (synchronously).
//  VLOG(1) << __func__;
//  Status status;
//  status = globals.device_tracer->Stop();
//  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
//  MAYBE_RETURN(status);
//  status = globals.device_tracer->Collect();
//  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
//  return status.code();
//}

}

}