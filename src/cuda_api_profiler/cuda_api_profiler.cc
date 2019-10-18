//
// Created by jagle on 8/2/2019.
//


#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"

#include <map>

//#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <list>
#include <fstream>
#include <iostream>
#include <memory>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include "cuda_api_profiler/cuda_api_profiler.h"
#include "cuda_api_profiler/get_env_var.h"
#include "cuda_api_profiler/defines.h"
#include "cuda_api_profiler/cupti_logging.h"
#include "cuda_api_profiler/util.h"

// 1656913 events -> 87092548 bytes (84MB)
// x              -> 20mb
//
// 1656913 / 87092548 = x / (20*1024*1024)
// x = (1656913 / 87092548) * (20*1024*1024)
// x = 398 977.7
// roundup: 400 000
#define CUDA_API_PROFILER_MAX_RECORDS_PER_DUMP 400000

namespace tensorflow {

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = cupti_wrapper_->call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

CUDAAPIProfiler::~CUDAAPIProfiler() {
//  if (VLOG_IS_ON(1)) {
//    DECLARE_LOG_INFO(info);
//    this->Print(info, 0);
//  }
  AwaitDump();
  DCHECK(!_state.CanDump())
    << "Looks like you forgot to dump CUDA API profiling state: "
    << _state._api_stats.size()  << " records";
  if (_state._api_stats.size() > 0 and !_state.CanDump()) {
    LOG(WARNING) << "There were " << _state._api_stats.size() << " CUDA API stats records left over, "
                 << "but we are not able to dump them; did you forget a call to sample_cuda_api.set_metadata(...)?";
  }
}

CUDAAPIProfilerState CUDAAPIProfilerState::DumpState() {
  CUDAAPIProfilerState state;
  state._directory = _directory;
  state._process_name = _process_name;
  state._machine_name = _machine_name;
  state._phase_name = _phase_name;
  state._trace_id = _next_trace_id;
  state._fuzzing = _fuzzing;
  state._event_recording = _event_recording;
  _next_trace_id += 1;

  state._events = std::move(_events);
  _events.clear();

  // NOTE: _start_t and _end_t are used to record the start/end time of CUDA API calls that are in-progress;
  // we DON'T serialize these to the protobuf, and doing so will lead to exceptions
  // (at CUDA API exit, we will expect to find the corresponding start time but it will be missing).
  // TLDR: Leave start_t/end_t in-tact.
//  state._start_t = std::move(_start_t);
//  _start_t.clear();
//  state._end_t = std::move(_end_t);
//  _end_t.clear();

  state._api_stats = std::move(_api_stats);
  _api_stats.clear();
  DCHECK(_api_stats.size() == 0);
  return state;
}

std::tuple<pid_t, const char*> CUDAAPIProfilerState::_GetTidApiName(APIKey api_key) {
  auto tid = std::get<0>(api_key);
  auto domain = std::get<1>(api_key);
  auto cbid = std::get<2>(api_key);
  const char* name;
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    name = runtime_cbid_to_string(cbid);
  } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    name = driver_cbid_to_string(cbid);
  } else {
    name = "UNKNOWN";
  }
  return std::make_tuple(tid, name);
}

std::unique_ptr<iml::CUDAAPIPhaseStatsProto> CUDAAPIProfilerState::AsProto() {
  std::unique_ptr<iml::CUDAAPIPhaseStatsProto> proto(new iml::CUDAAPIPhaseStatsProto);
  proto->set_process_name(_process_name);
  proto->set_machine_name(_machine_name);
  proto->set_phase(_phase_name);
  for (auto const& pair : _api_stats) {
    auto tid_name = _GetTidApiName(pair.first);
    auto tid = std::get<0>(tid_name);
    auto name = std::get<1>(tid_name);

    auto thread_stats = proto->add_stats();
    thread_stats->set_tid(tid);
    thread_stats->set_api_name(name);
    thread_stats->set_total_time_us(pair.second.total_api_time_usec);
    thread_stats->set_num_calls(pair.second.n_calls);
  }

  for (auto const& event : _events) {
    auto tid_name = _GetTidApiName(event.api_key);
    auto tid = std::get<0>(tid_name);
    auto name = std::get<1>(tid_name);

    auto event_proto = proto->add_events();
    event_proto->set_tid(tid);
    event_proto->set_api_name(name);
    event_proto->set_start_time_us(event.start_us);
    event_proto->set_duration_us(event.duration_us);
    event_proto->set_active_operation(event.active_operation);
  }

  return proto;
}

std::map<CUpti_CallbackId, std::string> CUDAAPIProfiler::RuntimeCallbackIDToName() {
    std::map<CUpti_CallbackId, std::string> cbid_to_name;
    for (int i = CUPTI_RUNTIME_TRACE_CBID_INVALID + 1; i < CUPTI_RUNTIME_TRACE_CBID_SIZE; i++) {
        cbid_to_name[i] = runtime_cbid_to_string(i);
    }
    return cbid_to_name;
}

std::map<CUpti_CallbackId, std::string> CUDAAPIProfiler::DriverCallbackIDToName() {
    std::map<CUpti_CallbackId, std::string> cbid_to_name;
    for (int i = CUPTI_DRIVER_TRACE_CBID_INVALID + 1; i < CUPTI_DRIVER_TRACE_CBID_SIZE; i++) {
        cbid_to_name[i] = driver_cbid_to_string(i);
    }
    return cbid_to_name;
}

void CUDAAPIProfiler::Print(std::ostream& out, int indent) {
    mutex_lock l(_mu);
    PrintIndent(out, indent);
    out << "CUDAAPIProfiler: size = " << _state._api_stats.size();
    for (auto const& pair : _state._api_stats) {
        auto tid = std::get<0>(pair.first);
        auto domain = std::get<1>(pair.first);
        auto cbid = std::get<2>(pair.first);
        const char* name;
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            name = runtime_cbid_to_string(cbid);
        } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            name = driver_cbid_to_string(cbid);
        } else {
            name = "UNKNOWN";
        }

        out << "\n";
        PrintIndent(out, indent + 1);
        out << "(tid=" << tid << ", api=" << name << "):";

        out << "\n";
        PrintIndent(out, indent + 1);
        out << "total_api_time_usec = " << pair.second.total_api_time_usec;

        out << "\n";
        PrintIndent(out, indent + 1);
        out << "n_calls = " << pair.second.n_calls;
    }
}

void CUDAAPIProfiler::EnableFuzzing() {
    _state._fuzzing = true;
}

void CUDAAPIProfiler::EnableEventRecording() {
  _state._event_recording = true;
}

void CUDAAPIProfiler::ApiCallback(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
//        const void *cbdata
    CUpti_ApiCallbackSite cb_site
) {
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API || domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
//        auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);

//    std::map<CUDAAPIProfilerState::APIKey, CUDAAPIProfilerState::TimeUsec>* timestamp_map;
//    if (cb_site == CUPTI_API_ENTER) {
//      timestamp_map = &_state._start_t;
//    } else {
//      timestamp_map = &_state._end_t;
//    }

    auto api_key = std::make_tuple(gettid(), domain, cbid);


    // This code will capture as much "profile-book-keeping overhead" added by this code as possible.
    // If we want to subtract from CUDA API events, we need those CUDA API events to capture the total overhead.
    {
      mutex_lock l(_mu);
      if (cb_site == CUPTI_API_EXIT) {
        // After cudaLaunchKernel
        auto const& active_operation = _op_stack.ActiveOperation();
        if (active_operation == "") {
          return;
        }

        // _op_stack.RecordOverheadEvent("cuda_api_interception", 1);
        auto start_us = _state._start_t.at(api_key);
        if (_state._event_recording) {
          // auto duration_us = end_us - start_us;
          _state._events.emplace_back(api_key, start_us, -1, active_operation);
        }
        auto& api_stats = _state._api_stats[api_key];
        auto& end_t = _state._end_t[api_key];
        // Capture as much profiling book-keeping overhead as we can before getting now_us.
        // - Looking up stuff in hash-maps.
        // - Allocating list-entries.
        auto now_us = Env::Default()->NowMicros();
        if (_state._event_recording) {
          // Now that we have now_us, fixup the event we recorded with the correct duration_us.
          auto last = _state._events.end();
          last--;
          auto end_us = now_us;
          auto duration_us = end_us - start_us;
          DCHECK(last->duration_us == -1);
          last->duration_us = duration_us;
        }
        auto end_us = now_us;
        // auto end_us = _state._end_t.at(api_key);
        // _state._api_stats[api_key].AddCall(start_us, end_us);
        api_stats.AddCall(start_us, end_us);
        // _state._end_t[api_key] = now_us;
        end_t = now_us;

        // PROBLEM: we need to dump only state belonging to the current thread;
        // otherwise we will dump a start time before the end time arrives...
        // Q: Can we JUST dump _events and leave the start_t/end_t dictionary in place...?
        // A: Yes I think so.
        _MaybeDump();

      } else if (cb_site == CUPTI_API_ENTER) {
        // Before cudaLaunchKernel
        _state._start_t[api_key] = Env::Default()->NowMicros();
      }
    }

  }
}

CUDAAPIProfiler::CUDAAPIProfiler(OpStack& op_stack) :
    _pool("CUDAAPIProfiler.pool", /*num_threads=*/4),
    _op_stack(op_stack)
{
}

bool CUDAAPIProfilerState::CanDump() {
  return _process_name != "" &&
         _machine_name != "" &&
         _phase_name != "" &&
         _directory != "" &&
         _api_stats.size() > 0;
}

std::string CUDAAPIProfilerState::DumpPath(int trace_id) {
  DCHECK(_directory != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
  DCHECK(_phase_name != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
  DCHECK(_process_name != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";

  std::stringstream ss;

  ss << _directory << path_separator();

  ss << "process" << path_separator() << _process_name << path_separator();
  ss << "phase" << path_separator() << _phase_name << path_separator();

  if (_fuzzing) {
    // $ iml-prof --fuzz-cuda-api
    ss << "fuzz_cuda_api_stats";
  } else {
    // $ iml-prof --cuda-api-calls
    ss << "cuda_api_stats";
  }

  ss << ".trace_" << trace_id;

  ss << ".proto";

  return ss.str();
}

void CUDAAPIProfiler::SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  mutex_lock lock(_mu);
  if (_state.CanDump()) {
    _AsyncDump();
  }
  VLOG(1) << "CUDAAPIProfiler." << __func__
          << " " << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  _state._directory = directory;
  _state._process_name = process_name;
  _state._machine_name = machine_name;
  _state._phase_name = phase_name;
}

//void CUDAAPIProfiler::SetProcessName(const std::string& process_name) {
//  mutex_lock lock(_mu);
//  if (_state.CanDump()) {
//    _AsyncDump();
//  }
//  _state._process_name = process_name;
//}
//
//void CUDAAPIProfiler::SetPhaseName(const std::string& phase_name) {
//  mutex_lock lock(_mu);
//  if (_state.CanDump()) {
//    _AsyncDump();
//  }
//  _state._phase_name = phase_name;
//}
//
//void CUDAAPIProfiler::SetMachineName(const std::string& machine_name) {
//  mutex_lock lock(_mu);
//  if (_state.CanDump()) {
//    _AsyncDump();
//  }
//  _state._machine_name = machine_name;
//}

void CUDAAPIProfiler::_MaybeDump() {
  if (_state.ShouldDump() && _state.CanDump()) {
    VLOG(1) << "CUDAAPIProfiler saw more than " << CUDA_API_PROFILER_MAX_RECORDS_PER_DUMP << " event records; "
            << "triggering async dump.";
    auto dump_state = _state.DumpState();
    _AsyncDumpWithState(std::move(dump_state));
  }
}

void CUDAAPIProfiler::AsyncDump() {
  mutex_lock lock(_mu);
  _AsyncDump();
}

bool CUDAAPIProfilerState::ShouldDump() {
  return
    // Number of records is larger than some threshold (~ ... MB).
      ( _events.size() ) >= CUDA_API_PROFILER_MAX_RECORDS_PER_DUMP;
}

void CUDAAPIProfiler::_AsyncDump() {
  if (_state.CanDump()) {
    CUDAAPIProfilerState dump_state;
    dump_state = _state.DumpState();
    _AsyncDumpWithState(std::move(dump_state));
  }
}

void CUDAAPIProfiler::_AsyncDumpWithState(CUDAAPIProfilerState&& dump_state) {
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

void CUDAAPIProfiler::AwaitDump() {
  _pool.AwaitAll();
}

CUDAAPIProfilerPrinter::CUDAAPIProfilerPrinter(CUDAAPIProfiler &profiler, float every_sec) :
    _profiler(profiler),
    _every_sec(every_sec)
{
}

void CUDAAPIProfilerPrinter::_Run() {
  _event_handler.RegisterFunc([this]() {
    this->_EverySec();
  }, _every_sec);

  _event_handler.EventLoop([this]() {
    return _should_stop.HasBeenNotified();
  });
}

void CUDAAPIProfilerPrinter::_EverySec() {
  if (VLOG_IS_ON(1)) {
    DECLARE_LOG_INFO(info);
    _profiler.Print(info, 0);
  }
}

void CUDAAPIProfilerPrinter::Start() {
    if (_printer_thread) {
        return;
    }
    _printer_thread.reset(new std::thread([this] {
        this->_Run();
    }));
}

void CUDAAPIProfilerPrinter::Stop() {
  if (!_should_stop.HasBeenNotified()) {
    _should_stop.Notify();
    if (_printer_thread) {
      VLOG(1) << "Wait for CUDA API stat printer thread to stop...";
      _printer_thread->join();
      VLOG(1) << "... printer thread done";
    }
//    if (VLOG_IS_ON(1)) {
//      DECLARE_LOG_INFO(info);
//      _profiler.Print(info, 0);
//    }
  }
}

CUDAAPIProfilerPrinter::~CUDAAPIProfilerPrinter() {
    Stop();
}

}
