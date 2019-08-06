//
// Created by jagle on 8/2/2019.
//

#include <cuda.h>
#include <cupti.h>

enum Status {
    OK = 0,
    ERROR = 1
};

#define CUPTI_CALL(call) ({ \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
    }                                                                   \
    _status;
})

#define MAYBE_RETURN(cupti_status) ({ \
    CUptiResult _status = cupti_status; \
    if (_status != CUPTI_SUCESS) { \
        return ERROR;
    } \
})

Status initTrace()
{
  size_t attrValue = 0, attrValueSize = sizeof(size_t);
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE)));
  // Enable all other activity record kinds.
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the activity API.
  // Some attributes require to be set before any CUDA context is created to be effective,
  // e.g. to be applied to all device buffer allocations (see documentation).
  MAYBE_RETURN(CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
  attrValue *= 2;
  MAYBE_RETURN(CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

  MAYBE_RETURN(CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
  attrValue *= 2;
  MAYBE_RETURN(CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

  MAYBE_RETURN(CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}


extern "C" {

    Status setup() {
        // Initialize global state.
    }

    Status enable_tracing() {
        // Enable call-backs.
    }

    Status disable_tracing() {
        // Disable call-backs.
    }
}

