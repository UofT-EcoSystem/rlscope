//
// Created by jagle on 8/6/2019.
//

#include "public_headers/rlscope_c_api.h"

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

using namespace rlscope;

extern "C" {

int RLSCOPE_EXPORT rlscope_hello_world() {
  std::cout << "FROM RLSCOPE: HELLO_WORLD" << std::endl;
  return rlscope::MyStatus::OK().code();
}

int RLSCOPE_EXPORT rlscope_setup() {
  // Initialize global state.
  LOG_FUNC_ENTRY();
  return rlscope::MyStatus::OK().code();
}

int RLSCOPE_EXPORT rlscope_print() {
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->Print();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_set_metadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  VLOG(1) << __func__
          << ", " << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  auto status = globals.device_tracer->SetMetadata(directory, process_name, machine_name, phase_name);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_enable_tracing() {
  // Enable call-backs.
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->Start();
//  if (status.code() != MyStatus::OK()) {
//    VLOG(0) << "rls-prof C++ API " << __func__ << " failed with: " << status;
//  }
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_is_enabled(int* retval) {
  LOG_FUNC_ENTRY();
  if (globals.device_tracer->IsEnabled()) {
    *retval = 1;
  } else {
    *retval = 0;
  }
  return rlscope::MyStatus::OK().code();
}

int RLSCOPE_EXPORT rlscope_disable_tracing() {
  // Disable call-backs.
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->Stop();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_disable_gpu_hw() {
  // Disable GPU HW sampler.
  LOG_FUNC_ENTRY();
  auto status = globals.device_tracer->DisableGpuHW();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_async_dump() {
  // Dump traces (asynchronously).
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->AsyncDump();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_await_dump() {
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


int RLSCOPE_EXPORT rlscope_record_event(
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

int RLSCOPE_EXPORT rlscope_record_overhead_event(
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

int RLSCOPE_EXPORT rlscope_record_overhead_event_for_operation(
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

int RLSCOPE_EXPORT rlscope_push_operation(
    const char* operation) {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->PushOperation(
      operation);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_start_pass() {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->StartPass();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_end_pass() {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->EndPass();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_has_next_pass(int* has_next_pass) {
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

int RLSCOPE_EXPORT rlscope_pop_operation() {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->PopOperation();
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

int RLSCOPE_EXPORT rlscope_set_max_operations(const char* operation, int num_pushes) {
  LOG_FUNC_ENTRY();
  MyStatus status;
  status = globals.device_tracer->SetMaxOperations(operation, num_pushes);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  return status.code();
}

} // extern "C"

