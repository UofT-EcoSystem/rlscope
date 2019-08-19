//
// Created by jagle on 8/2/2019.
//

#ifndef DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H
#define DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H

#include <cuda.h>
#include <cupti.h>

#include "cuda_api_profiler/event_handler.h"

#include "tensorflow/core/platform/notification.h"

#include <map>
#include <list>
#include <string>
#include <tuple>
#include <thread>

namespace tensorflow {

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
    std::map<APIKey, TimeUsec> _start_t GUARDED_BY(_mu);
    std::map<APIKey, TimeUsec> _end_t GUARDED_BY(_mu);
    std::map<APIKey, CUDAAPIStats> _api_stats GUARDED_BY(_mu);
    CUDAAPIProfiler() = default;
    ~CUDAAPIProfiler();
    void ApiCallback(
            CUpti_CallbackDomain domain,
            CUpti_CallbackId cbid,
            const void *cbdata);

    void Print(std::ostream& out, int indent);

    static std::map<CUpti_CallbackId, std::string> RuntimeCallbackIDToName();
    static std::map<CUpti_CallbackId, std::string> DriverCallbackIDToName();

    mutex _mu;
};


class CUDAAPIProfilerPrinter {
    /* Every X seconds, print out the collected CUDA API stats.
     * (in the future, we will dump the stats to a proto file.)
     */
public:
  EventHandler _event_handler;
  Notification _should_stop;
  CUDAAPIProfiler& _profiler;
  float _every_sec;
  std::unique_ptr<std::thread> _printer_thread;

  CUDAAPIProfilerPrinter(CUDAAPIProfiler& profiler, float every_sec);
  ~CUDAAPIProfilerPrinter();
  void _Run();
  void Stop();
  void _EverySec();

  void Start();
};

}

#endif //DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H
