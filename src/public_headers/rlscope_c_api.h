//
// Created by jgleeson on 2020-09-21.
//

#ifndef IML_RLSCOPE_C_API_H
#define IML_RLSCOPE_C_API_H

#include <cstdint>
#include "rlscope_export.h"

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

int RLSCOPE_EXPORT rlscope_hello_world();

int RLSCOPE_EXPORT rlscope_setup();

int RLSCOPE_EXPORT rlscope_print();

int RLSCOPE_EXPORT rlscope_set_metadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name);

int RLSCOPE_EXPORT rlscope_enable_tracing();

int RLSCOPE_EXPORT rlscope_is_enabled(int* retval);

int RLSCOPE_EXPORT rlscope_disable_tracing();

int RLSCOPE_EXPORT rlscope_disable_gpu_hw();

int RLSCOPE_EXPORT rlscope_async_dump();

int RLSCOPE_EXPORT rlscope_await_dump();

int RLSCOPE_EXPORT rlscope_record_event(
    const char* category,
    int64_t start_us,
    int64_t duration_us,
    const char* name);

int RLSCOPE_EXPORT rlscope_record_overhead_event(
    const char* overhead_type,
    int num_events);

int RLSCOPE_EXPORT rlscope_record_overhead_event_for_operation(
    const char* overhead_type,
    const char* operation,
    int num_events);

int RLSCOPE_EXPORT rlscope_push_operation(
    const char* operation);

int RLSCOPE_EXPORT rlscope_start_pass();

int RLSCOPE_EXPORT rlscope_end_pass();

int RLSCOPE_EXPORT rlscope_has_next_pass(int* has_next_pass);

int RLSCOPE_EXPORT rlscope_pop_operation();

int RLSCOPE_EXPORT rlscope_set_max_operations(const char* operation, int num_pushes);

}

#endif //IML_RLSCOPE_C_API_H
