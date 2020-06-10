//
// Created by jagle on 9/3/2019.
//

#include "cuda_api_profiler/event_profiler.h"
#include "cuda_api_profiler/cupti_logging.h"
#include "cuda_api_profiler/get_env_var.h"
#include "cuda_api_profiler/defines.h"
#include "common/util.h"

#include <cstdlib>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include <fstream>

//#define EVENT_PROFILER_MAX_RECORDS_PER_DUMP 1000
// 1000 -> 28kb per proto.
// x    -> 20mb
//
// 1000 / 28 = x / (20*1024)
// x = (20*1024) * (1000 / 28)
// x = 731 428.57
//
// 1000 / 28 = 10000 / y

// Q: How many events should we dump per file to reach 20MB?
#define EVENT_PROFILER_MAX_RECORDS_PER_DUMP 731500

namespace rlscope {

void EventProfiler::SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  mutex_lock lock(_mu);
  if (_state.CanDump()) {
    _AsyncDump();
  }
  VLOG(1) << "EventProfiler." << __func__
          << " " << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  _state._directory = directory;
  _state._process_name = process_name;
  _state._machine_name = machine_name;
  _state._phase_name = phase_name;
}

void EventProfiler::Print(std::ostream& out, int indent) {
    mutex_lock l(_mu);
    PrintIndent(out, indent);
    out << "EventProfiler: size = " << _state._events.size();
    const int MAX_PRINT_EVENTS = 15;
    for (auto const& pair : _state._events) {
      auto& category = pair.first;

      out << "\n";
      PrintIndent(out, indent + 1);
      out << "Category: " << category << ", size = " << pair.second.size();

      size_t i = 0;
      for (auto const& event_record : pair.second) {
        out << "\n";
        PrintIndent(out, indent + 2);
        out << "Event[" << i << "]: "
            << "name=\"" << event_record.name << "\""
            << ", " << "start_us=" << event_record.start_us << " us"
            << ", " << "duration_us=" << event_record.duration_us << " us";
        i += 1;
        if (i >= MAX_PRINT_EVENTS) {
          out << "\n";
          PrintIndent(out, indent + 2);
          out << "... " << (pair.second.size() - i)  << " more Event's ...";
          break;
        }
      }

    }
}

void EventProfiler::AsyncDump() {
  mutex_lock lock(_mu);
  _AsyncDump();
}

void EventProfiler::_AsyncDump() {
  if (_state.CanDump()) {
    auto dump_state = _state.DumpState();
    _pool.Schedule([dump_state = std::move(dump_state)] () mutable {
      auto path = dump_state.DumpPath(dump_state._trace_id);
      mkdir_p(os_dirname(path));
      auto proto = dump_state.AsProto();
      std::fstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
      if (!proto->SerializeToOstream(&out)) {
        LOG(FATAL) << "Failed to dump " << path;
      }
      VLOG(1) << "Dumped " << path;
    });
    DCHECK(!_state.CanDump());
  }
}

void EventProfiler::AwaitDump() {
  _pool.AwaitAll();
}

size_t EventProfilerState::size() const {
  return _size;
}

EventProfilerState EventProfilerState::DumpState() {
  EventProfilerState state;
  state._directory = _directory;
  state._process_name = _process_name;
  state._machine_name = _machine_name;
  state._phase_name = _phase_name;

  state._size = _size;
  _size = 0;

  state._trace_id = _next_trace_id;
  _next_trace_id += 1;

  state._events = std::move(_events);
  _events.clear();

  DCHECK(_events.size() == 0 && _size == 0);
  return state;
}


std::unique_ptr<iml::CategoryEventsProto> EventProfilerState::AsProto() {
  std::unique_ptr<iml::CategoryEventsProto> proto(new iml::CategoryEventsProto);
  proto->set_process_name(_process_name);
  proto->set_machine_name(_machine_name);
  proto->set_phase(_phase_name);
  for (auto const& pair : _events) {
    auto& category = pair.first;
    for (auto const& event_record : pair.second) {
      auto &events = (*proto->mutable_category_events())[category];
      auto event_proto = events.add_events();
      event_proto->set_thread_id(event_record.tid);
      event_proto->set_start_time_us(event_record.start_us);
      event_proto->set_duration_us(event_record.duration_us);
      event_proto->set_name(event_record.name);
    }
  }
  return proto;
}

bool EventProfilerState::ShouldDump() {
  return
    // Number of records is larger than some thresold (~ ... MB).
      ( this->size() ) >= EVENT_PROFILER_MAX_RECORDS_PER_DUMP;
}

bool EventProfilerState::CanDump() {
  return _process_name != "" &&
         _machine_name != "" &&
         _phase_name != "" &&
         _directory != "" &&
         _events.size() > 0;
}

std::string EventProfilerState::DumpPath(int trace_id) {
  DCHECK(_directory != "") << "You forgot to call EventProfiler.SetMetadata";
  DCHECK(_phase_name != "") << "You forgot to call EventProfiler.SetMetadata";
  DCHECK(_process_name != "") << "You forgot to call EventProfiler.SetMetadata";

  std::stringstream ss;

  ss << _directory << path_separator();

  ss << "process" << path_separator() << _process_name << path_separator();
  ss << "phase" << path_separator() << _phase_name << path_separator();

  ss << "category_events";

  ss << ".trace_" << trace_id;

  ss << ".proto";

  return ss.str();
}

void EventProfiler::RecordEvent(
    const EventProfilerState::Category& category,
    EventProfilerState::TimeUsec start_us,
    EventProfilerState::TimeUsec duration_us,
    const EventProfilerState::EventName& name) {
  mutex_lock lock(_mu);
  auto& events = _state._events[category];
  pid_t tid = gettid();
  events.emplace_back(tid, start_us, duration_us, name);
  _state._size += 1;

  _MaybeDump();

}

void EventProfiler::_MaybeDump() {
  if (_state.ShouldDump() && _state.CanDump()) {
    VLOG(1) << "EventProfiler saw more than " << EVENT_PROFILER_MAX_RECORDS_PER_DUMP << " records; "
            << "triggering async dump.";
    auto dump_state = _state.DumpState();
    _AsyncDumpWithState(std::move(dump_state));
  }
}

void EventProfiler::_AsyncDumpWithState(EventProfilerState&& dump_state) {
  _pool.Schedule([dump_state = std::move(dump_state)] () mutable {
    auto path = dump_state.DumpPath(dump_state._trace_id);
    mkdir_p(os_dirname(path));
    auto proto = dump_state.AsProto();
    std::fstream out(path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!proto->SerializeToOstream(&out)) {
      LOG(FATAL) << "Failed to dump " << path;
    }
    VLOG(1) << "Dumped " << path;
  });
  DCHECK(!_state.CanDump());
}

EventProfiler::EventProfiler() :
    _pool("EventProfiler.pool", /*num_threads=*/4)
{
}
EventProfiler::~EventProfiler() {
//  if (VLOG_IS_ON(1)) {
//    DECLARE_LOG_INFO(info);
//    this->Print(info, 0);
//  }
  AwaitDump();
  if (_state.CanDump()) {
    {
      DECLARE_LOG_INFO(info);
      this->Print(info, 0);
    }
    LOG(FATAL) << "Looks like you forgot to dump Event profiling state: "
               << _state.size() << " records";
  }
  if (_state._events.size() > 0 and !_state.CanDump()) {
    LOG(WARNING) << "There were " << _state.size() << " Event records left over, "
                 << "but we are not able to dump them; did you forget a call to sample_cuda_api.set_metadata(...)?";
  }
}


}