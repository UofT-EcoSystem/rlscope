//
// Created by jagle on 8/2/2019.
//


#include "common_util.h"

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

//#include "rlscope/protobuf/rlscope_prof.pb.h"
#include "rlscope_prof.pb.h"

#include "cuda_api_profiler/op_stack.h"
#include "cuda_api_profiler/get_env_var.h"
#include "cuda_api_profiler/defines.h"
#include "cuda_api_profiler/cupti_logging.h"
#include "common/util.h"

#include <mutex>

namespace rlscope {

OpStack::~OpStack() {
//  if (VLOG_IS_ON(1)) {
//    DECLARE_LOG_INFO(info);
//    this->Print(info, 0);
//  }
  AwaitDump();
  DCHECK(!_state.CanDump())
    << "Looks like you forgot to dump OpStack state: "
    << _state.size()  << " records";
  if (_state.size() > 0 and !_state.CanDump()) {
    LOG(WARNING) << "There were " << _state.size() << " OpStack records left over, "
                 << "but we are not able to dump them; did you forget a call to rlscope_api.set_metadata(...)?";
  }
}

size_t OpStackState::size() {
  return _overhead_events.size();
}

OpStackState OpStackState::DumpState() {
  OpStackState state;
  state._directory = _directory;
  state._process_name = _process_name;
  state._machine_name = _machine_name;
  state._phase_name = _phase_name;
  state._trace_id = _next_trace_id;
  _next_trace_id += 1;

  state._overhead_events = std::move(_overhead_events);
  _overhead_events.clear();

  DCHECK(size() == 0);

  return state;
}


std::unique_ptr<rlscope::OpStackProto> OpStackState::AsProto() {
  std::unique_ptr<rlscope::OpStackProto> proto(new rlscope::OpStackProto);
  proto->set_process_name(_process_name);
  proto->set_machine_name(_machine_name);
  proto->set_phase(_phase_name);

  // overhead-type -> phase-name -> operation-name -> # of overhead events
  for (auto const& overhead_type_pair : _overhead_events) {
    auto const& overhead_type = overhead_type_pair.first;
    for (auto const& phase_pair : overhead_type_pair.second) {
      auto const& phase_name = phase_pair.first;
      for (auto const& op_pair : phase_pair.second) {
        auto const& operation = op_pair.first;
        auto const& overhead_event = op_pair.second;

        rlscope::OperationOverheadEvents& overhead_events_proto = (*(*proto->mutable_overhead_events())[overhead_type].mutable_phase_overhead_events())[phase_name];
        overhead_events_proto.set_operation_name(operation);
        overhead_events_proto.set_num_overhead_events(overhead_event.num_events);
      }
    }
  }

  return proto;
}

void OpStack::Print(std::ostream& out, int indent) {
  std::unique_lock<std::mutex> l(_mu);
  PrintIndent(out, indent);
  out << "OpStack: size = " << _state.size();

  // overhead-type -> phase-name -> operation-name -> # of overhead events
  for (auto const& overhead_type_pair : _state._overhead_events) {
    auto const& overhead_type = overhead_type_pair.first;
    out << "\n";
    PrintIndent(out, indent + 1);
    out << "OverheadType = " << overhead_type;
    for (auto const& phase_pair : overhead_type_pair.second) {
      auto const& phase_name = phase_pair.first;
      out << "\n";
      PrintIndent(out, indent + 2);
      out << "Phase: " << phase_name;
      for (auto const& op_pair : phase_pair.second) {
        auto const& operation = op_pair.first;
        auto const& overhead_event = op_pair.second;
        out << "\n";
        PrintIndent(out, indent + 3);
        out << "NumOverheadEvents[op=" << operation << "] = " << overhead_event.num_events;
      }
    }
  }

}

OpStack::OpStack() :
    _pool("OpStack.pool", /*num_threads=*/2)
{
}

bool OpStackState::CanDump() {
  return _process_name != "" &&
         _machine_name != "" &&
         _phase_name != "" &&
         _directory != "" &&
         size() > 0;
}

std::string OpStackState::DumpPath(int trace_id) {
  DCHECK(_directory != "") << "You forgot to call OpStack.SetMetadata";
  DCHECK(_phase_name != "") << "You forgot to call OpStack.SetMetadata";
  DCHECK(_process_name != "") << "You forgot to call OpStack.SetMetadata";

  std::stringstream ss;

  ss << _directory << path_separator();

  ss << "process" << path_separator() << _process_name << path_separator();
  ss << "phase" << path_separator() << _phase_name << path_separator();

  ss << "op_stack";

  ss << ".trace_" << trace_id;

  ss << ".proto";

  return ss.str();
}

void OpStack::SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  std::unique_lock<std::mutex> lock(_mu);
  if (_state.CanDump()) {
    _AsyncDump();
  }
  VLOG(1) << "OpStack." << __func__
          << " " << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  _state._directory = directory;
  _state._process_name = process_name;
  _state._machine_name = machine_name;
  _state._phase_name = phase_name;
}

void OpStack::AsyncDump() {
  std::unique_lock<std::mutex> lock(_mu);
  _AsyncDump();
}

void OpStack::_AsyncDump() {
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

void OpStack::AwaitDump() {
  _pool.AwaitAll();
}

std::string OpStack::ActiveOperation() {
//  VLOG(1) << "OpStack." << __func__ << ", size() == " << _state._operation_stack.size();
  std::unique_lock<std::mutex> lock(_mu);
  if (_state._operation_stack.size() == 0) {
    // It's possible (though strange) to run GPU kernels without an operation active.
    return OPERATION_UNKNOWN;
  }
  DCHECK(_state._operation_stack.size() > 0);
  auto const& operation = _state._operation_stack.back();
  return operation;
}
void OpStack::PushOperation(const std::string& operation) {
//  VLOG(1) << "OpStack." << __func__ << ", operation=" << operation;
  std::unique_lock<std::mutex> lock(_mu);
  _state._operation_stack.push_back(operation);
}
void OpStack::RecordOverheadEvent(
    const std::string& overhead_type,
    int64_t num_events) {
//  VLOG(1) << "OpStack." << __func__ << ", overhead_type=" << overhead_type << ", num_events = " << num_events;
  std::unique_lock<std::mutex> lock(_mu);
  std::string operation;
  if (_state._operation_stack.size() == 0) {
    operation = OPERATION_UNKNOWN;
  } else {
    operation = _state._operation_stack.back();
  }
//  DCHECK(_state._operation_stack.size() > 0);
  DCHECK(_state._phase_name != "");
  _state._overhead_events[overhead_type][_state._phase_name][operation].num_events += num_events;
}
void OpStack::RecordOverheadEventForOperation(
    const std::string& overhead_type,
    const std::string& operation,
    int64_t num_events) {
//  VLOG(1) << "OpStack." << __func__ << ", overhead_type = " << overhead_type << ", operation=" << operation << ", num_events = " << num_events;
  std::unique_lock<std::mutex> lock(_mu);
  DCHECK(_state._phase_name != "");
  _state._overhead_events[overhead_type][_state._phase_name][operation].num_events += num_events;
}

void OpStack::PopOperation() {
//  VLOG(1) << "OpStack." << __func__ << ", size() == " << _state._operation_stack.size();
  std::unique_lock<std::mutex> lock(_mu);
  DCHECK(_state._operation_stack.size() > 0);
  _state._operation_stack.pop_back();
}

}
