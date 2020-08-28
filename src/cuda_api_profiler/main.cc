//
// Created by jagle on 8/6/2019.
//

#include <cuda.h>
#include <cupti_target.h>
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

#define LOG_FUNC_ENTRY() \
  if (SHOULD_DEBUG(FEATURE_RLSCOPE_LIB_TRACE)) { \
    RLS_LOG("RLSCOPE_LIB_TRACE", "{}", ""); \
  }

//#define LOG_FUNC_ENTRY()

namespace rlscope {



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
  LOG_FUNC_ENTRY();
  return rlscope::MyStatus::OK().code();
}

RetCode RLSCOPE_EXPORT print() {
  LOG_FUNC_ENTRY();
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
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->Start();
//  if (status.code() != MyStatus::OK()) {
//    VLOG(0) << "iml-prof C++ API " << __func__ << " failed with: " << status;
//  }
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT is_enabled(int* retval) {
  LOG_FUNC_ENTRY();
  if (globals.device_tracer->IsEnabled()) {
    *retval = 1;
  } else {
    *retval = 0;
  }
  return rlscope::MyStatus::OK().code();
}

RetCode RLSCOPE_EXPORT disable_tracing() {
  // Disable call-backs.
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->Stop();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT disable_gpu_hw() {
  // Disable GPU HW sampler.
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->DisableGpuHW();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT async_dump() {
  // Dump traces (asynchronously).
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->AsyncDump();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT await_dump() {
  // Wait for async dump traces to complete.
  LOG_FUNC_ENTRY();
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
  LOG_FUNC_ENTRY();
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
  LOG_FUNC_ENTRY();
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
  LOG_FUNC_ENTRY();
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
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->PushOperation(
      operation);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT start_pass() {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->StartPass();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT end_pass() {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->EndPass();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT has_next_pass(int* has_next_pass) {
  LOG_FUNC_ENTRY();
  MyStatus status;
  bool bool_has_next_pass = false;
  *has_next_pass = 0;
  status = globals.device_tracer->HasNextPass(&bool_has_next_pass);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  if (status.ok()) {
    *has_next_pass = static_cast<int>(bool_has_next_pass);
  }
  return status.code();
}

RetCode RLSCOPE_EXPORT pop_operation() {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->PopOperation();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

RetCode RLSCOPE_EXPORT set_max_operations(const char* operation, int num_pushes) {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->SetMaxOperations(operation, num_pushes);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

}

}
