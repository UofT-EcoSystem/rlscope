//
// Created by jagle on 8/12/2019.
//

#ifndef IML_PC_SAMPLING_H
#define IML_PC_SAMPLING_H

#include <list>
#include <string>
#include <thread>
#include <mutex>

#include "common_util.h"

#include "cuda_api_profiler/get_env_var.h"

//#include "rlscope/protobuf/rlscope_prof.pb.h"
#include "rlscope_prof.pb.h"

#define THREAD_ID_UNSET (-1)

namespace rlscope {

using namespace rlscope;

// Forward decls.
class SampleEvent;
class CPUSampleState;
class GPUSampleState;
class SampleEvents;

struct ThreadInfo {
  int thread_id;
  ThreadInfo() :
      thread_id(THREAD_ID_UNSET) {
  }
};

extern thread_local ThreadInfo tinfo;



class Annotation {
public:
  // "sgd_updates", "training_loop", ...
  std::string name;
  // "Operation", "Framework API", "Python", ...
  std::string category;
  Annotation() = default;
  Annotation(const std::string& name, const std::string& category) :
      name(name),
      category(category) {
  }

  void Sample(Annotation* data);

  AnnotationProto AsProto() const;
};

class CPUSampleState {
public:
  pid_t tid;
  int64_t thread_id;
  std::string device_name;
  CPUSampleState() = default;
  CPUSampleState(pid_t tid, int64_t thread_id);
  // The current active "stack" of annotations.
  std::list<Annotation> annotations;

  void Push(const std::string& name, const std::string& category);
  void Pop();

  void Sample(CPUSampleState* data);

  CPUSampleStateProto AsProto() const;
};

class GPUSampleState {
public:
  std::string device_name;
  // Currently, we JUST record whether a GPU kernel is running.
  // Ideally, in the future we would like to collect more GPU-side information.
  // Collecting additional GPU information (e.g. hardware counters) will require
  // the kernel-replay feature which CUDA provides for collecting multiple hardware
  // counters in separate runs (since register space is a limitation).
  bool is_gpu_active;

  struct SyncState {
    Notification _should_stop;
    std::condition_variable _cv_is_gpu_active;
    std::unique_ptr<std::thread> _make_gpu_inactive_thread;
    ~SyncState();
  };
  std::unique_ptr<SyncState> _sync_state;

//  GPUSampleState& operator=(GPUSampleState&& other) = default;
//  GPUSampleState& operator=(const GPUSampleState& other) = default;
//  GPUSampleState() = default;

//  GPUSampleState( GPUSampleState&& other ) :
//  device_name(std::move(other.device_name)),
//  is_gpu_active(std::move(other.is_gpu_active)),
//  _sync_state(std::move(_sync_state))
//  {
//  }

  GPUSampleState() = default;
  GPUSampleState(const std::string& device_name);
  void MarkGPUActive();

  void Sample(GPUSampleState* data);

  GPUSampleStateProto AsProto() const;
};

// A SamplingEvent captures all of the sampling-state, at a particular point in time.
//
// Ideally, we capture the sampling-state of each thread on the machine.
class SampleEvent {
public:
  int64_t sample_time_us;
  std::vector<CPUSampleState> cpu_sample_state;
  std::vector<GPUSampleState> gpu_sample_state;

  SampleEvent() = default;

  void Sample(SampleEvent* data);

  SampleEventProto AsProto() const;
};

class SampleEvents {
public:
  std::string process_name;
  std::string phase;
  std::string machine_name;

  SampleEvents() = default;

  void SetAttrs(const std::string& process_name, const std::string& phase, const std::string& machine_name);

//  std::list<SampleEvent> events;
  SampleEvent event;

  void Sample(SampleEvents* data);

  SampleEventsProto AsProto() const;
};


#define DEFAULT_IML_SAMPLE_EVERY_SEC (1.0)
class SampleAPI {
public:

  std::mutex _mu;

  // GPU device-name -> 0-based device integer ID
  std::map<std::string, int> _dev_to_dev_id;
  int _next_gpu_id;

  // POSIX thread_id -> 0-based integer ID
  std::map<pid_t, int> _tid_to_id;
  int _next_thread_id;

  float _sample_every_sec;

  SampleEvents _sample_events;

  SampleAPI() :
      _next_gpu_id(0),
      _next_thread_id(0),
      _sample_every_sec(get_IML_SAMPLE_EVERY_SEC(0)) {
  }

  void _RunMakeGPUInactive(GPUSampleState* gpu_sample_state);

  void _MaybeSetThreadID(ThreadInfo* tinfo);
  void MaybeSetThreadID(ThreadInfo* tinfo);

  // Python main thread calls into here when an annotation gets pushed.
  void Push(const std::string& name, const std::string& operation);
  // Python main thread calls into here when an annotation gets popped.
  void Pop();
  // CUDA PC sampling call-back calls into here.
  void MarkGPUActive(const std::string& device_name);

  SampleEvents Sample();

  int _SetThreadID();
  int SetThreadID();
  int SetGPUID(const std::string& device_name);
  int _SetGPUID(const std::string& device_name);

  ThreadInfo* _CurrentThreadInfo();
};
extern SampleAPI sample_api;


}

#endif //IML_PC_SAMPLING_H
