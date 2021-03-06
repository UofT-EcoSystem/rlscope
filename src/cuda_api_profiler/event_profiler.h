//
// Created by jagle on 9/3/2019.
//

#ifndef RLSCOPE_EVENT_PROFILER_H
#define RLSCOPE_EVENT_PROFILER_H

//#include "rlscope/protobuf/rlscope_prof.pb.h"
//#include "rlscope/protobuf/pyprof.pb.h"

#include "rlscope_prof.pb.h"
#include "pyprof.pb.h"

#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/thread_pool_wrapper.h"

#include <map>
#include <list>
#include <string>
#include <tuple>
#include <memory>
#include <thread>
#include <mutex>

namespace rlscope {

struct EventRecord;

// CategoryEventsProto from pyprof.proto
struct EventProfilerState {
  using Category = std::string;
  using EventName = std::string;
  using TimeUsec = int64_t;
  using TidType = int64_t;

  std::string _directory;
  std::string _process_name;
  std::string _machine_name;
  std::string _phase_name;
  size_t _size;
  int _next_trace_id;
  int _trace_id;

  std::map<Category, std::list<EventRecord>> _events;

  // WARNING: if you add a member here, don't forget to copy the field in DumpState()!
  EventProfilerState() :
      _size(0)
      , _next_trace_id(0)
      , _trace_id(-1)
  {
  }

  bool ShouldDump();
  bool CanDump();
  std::string DumpPath(int trace_id);
  EventProfilerState DumpState();
  std::unique_ptr<rlscope::CategoryEventsProto> AsProto();
  size_t size() const;
  // std::tuple<pid_t, const char*> _GetTidApiName(APIKey api_key);

};

// Event from pyprof.proto
struct EventRecord {
  EventProfilerState::TidType tid;
  EventProfilerState::TimeUsec start_us;
  EventProfilerState::TimeUsec duration_us;
  EventProfilerState::EventName name;
  EventRecord(
      EventProfilerState::TidType tid,
      EventProfilerState::TimeUsec start_us,
      EventProfilerState::TimeUsec duration_us,
      const EventProfilerState::EventName& name) :
      tid(tid),
      start_us(start_us),
      duration_us(duration_us),
      name(name)
  {
  }
};

class EventProfiler {
public:
  ThreadPoolWrapper _pool;
  std::mutex _mu;
  EventProfilerState _state;

  EventProfiler();
  ~EventProfiler();

  void Print(std::ostream& out, int indent);

  void RecordEvent(
      const EventProfilerState::Category& category,
      EventProfilerState::TimeUsec start_us,
      EventProfilerState::TimeUsec duration_us,
      const EventProfilerState::EventName& name);

  void SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name);
  void AsyncDump();
  void _AsyncDump();
  void _AsyncDumpWithState(EventProfilerState&& dump_state);
  void AwaitDump();
  void _MaybeDump();

};

}

#endif //RLSCOPE_EVENT_PROFILER_H
