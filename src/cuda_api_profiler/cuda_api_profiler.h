//
// Created by jagle on 8/2/2019.
//

#ifndef DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H
#define DNN_TENSORFLOW_CPP_CUDA_API_PROFILER_H

#include <cuda.h>
#include <cupti.h>

#include "iml_profiler/protobuf/iml_prof.pb.h"

#include "cuda_api_profiler/op_stack.h"
#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/thread_pool_wrapper.h"

#include "tensorflow/core/platform/notification.h"

#include <map>
#include <list>
#include <string>
#include <tuple>
#include <memory>
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

struct CUDAAPICallRecord;

struct CUDAAPIProfilerState {
  // (thread-id, api-cbid)
  using APIKey = std::tuple<pid_t, CUpti_CallbackDomain, CUpti_CallbackId>;
  using TimeUsec = int64;
  using OperationName = std::string;

  std::map<APIKey, TimeUsec> _start_t;
  std::map<APIKey, TimeUsec> _end_t;
  std::map<APIKey, CUDAAPIStats> _api_stats;
  std::string _directory;
  std::string _process_name;
  std::string _machine_name;
  std::string _phase_name;
  int _next_trace_id;
  int _trace_id;
  bool _fuzzing;
  bool _event_recording;

  std::list<CUDAAPICallRecord> _events;

  // WARNING: if you add a member here, don't forget to copy the field in DumpState()!
  CUDAAPIProfilerState() :
      _next_trace_id(0),
      _trace_id(-1),
      _fuzzing(false),
      _event_recording(false)
  {
  }

  bool CanDump();
  std::string DumpPath(int trace_id);
  CUDAAPIProfilerState DumpState();
  std::unique_ptr<iml::CUDAAPIPhaseStatsProto> AsProto();
  std::tuple<pid_t, const char*> _GetTidApiName(APIKey api_key);

};

struct CUDAAPICallRecord {
  CUDAAPIProfilerState::APIKey api_key;
  CUDAAPIProfilerState::TimeUsec start_us;
  CUDAAPIProfilerState::TimeUsec duration_us;
  CUDAAPIProfilerState::OperationName active_operation;
  CUDAAPICallRecord(CUDAAPIProfilerState::APIKey api_key, CUDAAPIProfilerState::TimeUsec start_us, CUDAAPIProfilerState::TimeUsec duration_us, const CUDAAPIProfilerState::OperationName& active_operation) :
      api_key(api_key),
      start_us(start_us),
      duration_us(duration_us),
      active_operation(active_operation)
  {
  }
};

class CUDAAPIProfiler {
public:
  ThreadPoolWrapper _pool;
  mutex _mu;
  CUDAAPIProfilerState _state;
  OpStack& _op_stack;

  CUDAAPIProfiler(OpStack& op_stack);
  ~CUDAAPIProfiler();
  void EnableFuzzing();
  void EnableEventRecording();
  void ApiCallback(
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid,
      // const void *cbdata
      CUpti_ApiCallbackSite cb_site);

  void Print(std::ostream& out, int indent);

  void SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name);
  void AsyncDump();
  void _AsyncDump();
  void AwaitDump();

  static std::map<CUpti_CallbackId, std::string> RuntimeCallbackIDToName();
  static std::map<CUpti_CallbackId, std::string> DriverCallbackIDToName();

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
