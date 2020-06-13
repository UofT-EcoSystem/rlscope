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

#include "common_util.h"
#include "cuda_api_profiler/device_tracer.h"

#ifdef WITH_CUDA_LD_PRELOAD
#include "cuda_api_profiler/cuda_ld_preload.h"
#endif

#include "cuda_api_profiler/cupti_logging.h"
#include "cuda_api_profiler/globals.h"

#include "rlscope_export.h"

namespace rlscope {



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

// MyStatus initTrace()
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

using StatusRet = rlscope::error::Code;

extern "C" {

typedef enum RLSCOPE_EXPORT TF_Code {
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

RetCode RLSCOPE_EXPORT setup() {
  // Initialize global state.
  VLOG(1) << __func__;
  return rlscope::MyStatus::OK().code();
}

RetCode RLSCOPE_EXPORT print() {
  VLOG(1) << __func__;
  auto status = globals.device_tracer->Print();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT set_metadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  VLOG(1) << __func__
          << ", " << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  auto status = globals.device_tracer->SetMetadata(directory, process_name, machine_name, phase_name);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT enable_tracing() {
  // Enable call-backs.
  VLOG(1) << __func__;
  auto status = globals.device_tracer->Start();
//  if (status.code() != MyStatus::OK()) {
//    VLOG(0) << "iml-prof C++ API " << __func__ << " failed with: " << status;
//  }
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT is_enabled(int* retval) {
  VLOG(1) << __func__;
  if (globals.device_tracer->IsEnabled()) {
    *retval = 1;
  } else {
    *retval = 0;
  }
  return rlscope::MyStatus::OK().code();
}

RetCode RLSCOPE_EXPORT disable_tracing() {
  // Disable call-backs.
  VLOG(1) << __func__;
  auto status = globals.device_tracer->Stop();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT async_dump() {
  // Dump traces (asynchronously).
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->AsyncDump();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT await_dump() {
  // Wait for async dump traces to complete.
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->AsyncDump();
  MAYBE_RETURN(status);
  status = globals.device_tracer->AwaitDump();
  // TODO: call device_tracer->Collect?
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}


RetCode RLSCOPE_EXPORT record_event(
    const char* category,
    int64_t start_us,
    int64_t duration_us,
    const char* name) {
  // Wait for async dump traces to complete.
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->RecordEvent(
      category,
      start_us,
      duration_us,
      name);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT record_overhead_event(
    const char* overhead_type,
    int num_events) {
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->RecordOverheadEvent(
      overhead_type,
      num_events);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT record_overhead_event_for_operation(
    const char* overhead_type,
    const char* operation,
    int num_events) {
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->RecordOverheadEventForOperation(
      overhead_type,
      operation,
      num_events);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT push_operation(
    const char* operation) {
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->PushOperation(
      operation);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT start_pass() {
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->StartPass();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT end_pass() {
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->EndPass();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT pop_operation() {
  VLOG(1) << __func__;
  MyStatus status;
  status = globals.device_tracer->PopOperation();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

}

}
