//
// Created by jagle on 8/2/2019.
//

#ifndef DNN_TENSORFLOW_CPP_OP_STACK_H
#define DNN_TENSORFLOW_CPP_OP_STACK_H

#include "iml_profiler/protobuf/iml_prof.pb.h"

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

struct OverheadEvent {
  int64 num_events;
  OverheadEvent() : num_events(0) {
  }
};

struct OpStackState {
  using OverheadType = std::string;
  using OperationName = std::string;
  using PhaseName = std::string;

  std::string _directory;
  std::string _process_name;
  std::string _machine_name;
  std::string _phase_name;
  int _next_trace_id;
  int _trace_id;

  std::list<OperationName> _operation_stack;

  // overhead-type -> phase-name -> operation-name -> # of overhead events
  std::map<OverheadType,
      std::map<PhaseName,
          std::map<OperationName,
              OverheadEvent>>> _overhead_events;

  // WARNING: if you add a member here, don't forget to copy the field in DumpState()!
  OpStackState() :
      _next_trace_id(0),
      _trace_id(-1)
  {
  }

  size_t size();

  bool CanDump();
  std::string DumpPath(int trace_id);
  OpStackState DumpState();
  std::unique_ptr<iml::OpStackProto> AsProto();

};

class OpStack {
public:
  ThreadPoolWrapper _pool;
  mutex _mu;
  OpStackState _state;

  OpStack();
  ~OpStack();

  void Print(std::ostream& out, int indent);

  void SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name);
  void AsyncDump();
  void _AsyncDump();
  void AwaitDump();

  std::string ActiveOperation();
  void PushOperation(const std::string& operation);
  void RecordOverheadEvent(
      const std::string& overhead_type,
      int64 num_events);
  void RecordOverheadEventForOperation(
      const std::string& overhead_type,
      const std::string& operation,
      int64 num_events);
  void PopOperation();

};

}

#endif //DNN_TENSORFLOW_CPP_OP_STACK_H
