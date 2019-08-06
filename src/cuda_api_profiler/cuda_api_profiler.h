//
// Created by jagle on 8/2/2019.
//

#ifndef DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H
#define DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H

#include "tensorflow/core/platform/device_tracer.h"

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = cupti_wrapper_->call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

struct CUDAAPIStats {
    int64 total_api_time_usec;
    int64 n_calls;

    CUDAAPIStats() :
            total_api_time_usec(0),
            n_calls(0) {
    }

    void AddCall(int64 start_call_usec, int64 end_call_usec) {
        auto delta = end_call_usec - start_call_usec;
        total_api_time_usec += delta;
        n_calls += 1;
    }
};
// TODO: We need to dump this information to a protobuf, ideally separate from tfprof protobuf
// so that we can record/dump this even if tfprof is disabled (uninstrumented runs).
class CUDAAPIProfiler {
public:
    // (thread-id, api-cbid)
    using APIKey = std::tuple<pid_t, CUpti_CallbackDomain, CUpti_CallbackId>;
    using TimeUsec = int64;
    std::map<APIKey, TimeUsec> _start_t;
    std::map<APIKey, TimeUsec> _end_t;
    std::map<APIKey, CUDAAPIStats> _api_stats;
    CUDAAPIProfiler() = default;
    ~CUDAAPIProfiler();
    void ApiCallback(
            void *userdata,
            CUpti_CallbackDomain domain,
            CUpti_CallbackId cbid,
            const void *cbdata);

    template <class Stream>
    void Print(Stream&& out);

    static std::map<CUpti_CallbackId, std::string> RuntimeCallbackIDToName();
    static std::map<CUpti_CallbackId, std::string> DriverCallbackIDToName();
};

#endif //DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H
