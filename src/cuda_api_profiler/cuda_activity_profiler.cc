//
// Created by jagle on 8/23/2019.
//

#include "cuda_api_profiler/cuda_activity_profiler.h"
#include "cuda_api_profiler/get_env_var.h"
//#include "iml_profiler/protobuf/iml_prof.pb.h"
#include "iml_prof.pb.h"
#include "cuda_api_profiler/util.h"
#include "cuda_api_profiler/cupti_logging.h"

#include "tensorflow/core/platform/env.h"

#define CONFIG_TRACE_STATS

#include <fstream>

namespace tensorflow {

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(FATAL) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)


bool CUDAActivityProfilerState::CanDump() {
  return _process_name != "" &&
         _machine_name != "" &&
         _phase_name != "" &&
         _directory != "" && (
                                 kernel_records_.size() > 0 ||
                                 memcpy_records_.size() > 0
                             );
}

//#define CUDA_ACTIVITY_PROFILER_MAX_RECORDS_PER_DUMP 10000
// 100 -> 1.6kb per proto.
// x   -> 20mb
//
// 100 / 1.6 = x / (20*1024)
// x = (20*1024) * (100 / 1.6)
// x = 1 280 000
//
// 100 / 1.6 = 10000 / y

//#define CUDA_ACTIVITY_PROFILER_MAX_RECORDS_PER_DUMP 100
#define CUDA_ACTIVITY_PROFILER_MAX_RECORDS_PER_DUMP 1280000

bool CUDAActivityProfilerState::ShouldDump() {
    return
      // Number of records is larger than some thresold (~ ... MB).
      ( kernel_records_.size() + memcpy_records_.size() ) >= CUDA_ACTIVITY_PROFILER_MAX_RECORDS_PER_DUMP;
}

std::string CUDAActivityProfilerState::DumpPath(int trace_id) {
  DCHECK(_directory != "") << "You forgot to call CUDAActivityProfiler.SetMetadata";
  DCHECK(_phase_name != "") << "You forgot to call CUDAActivityProfiler.SetMetadata";
  DCHECK(_process_name != "") << "You forgot to call CUDAActivityProfiler.SetMetadata";

  std::stringstream ss;

  ss << _directory << path_separator();

  ss << "process" << path_separator() << _process_name << path_separator();
  ss << "phase" << path_separator() << _phase_name << path_separator();

  ss << "cuda_device_events";

  ss << ".trace_" << trace_id;

  ss << ".proto";

  return ss.str();
}

CUDAActivityProfilerState CUDAActivityProfilerState::DumpState() {
  CUDAActivityProfilerState state;
  state._directory = _directory;
  state._process_name = _process_name;
  state._machine_name = _machine_name;
  state._phase_name = _phase_name;
  state._trace_id = _next_trace_id;
  _next_trace_id += 1;

  state.correlations_ = std::move(correlations_);
  correlations_.clear();
  state.kernel_records_ = std::move(kernel_records_);
  kernel_records_.clear();
  state.memcpy_records_ = std::move(memcpy_records_);
  memcpy_records_.clear();

  state.start_walltime_us_ = start_walltime_us_;
  state.end_walltime_us_ = end_walltime_us_;
  state.start_timestamp_ = start_timestamp_;
  state.end_timestamp_ = end_timestamp_;

  DCHECK(kernel_records_.size() == 0);
  DCHECK(memcpy_records_.size() == 0);

  return state;
}

template <class RecordType>
static void _AddEvent(
    CUDAActivityProfilerState& self,
    iml::CudaEventType cuda_event_type,
    iml::MachineDevsEventsProto* proto,
    const std::string& device_name, const RecordType& rec) {
  bool has_dev_events = proto->dev_events().find(device_name) != proto->dev_events().end();
  auto &dev_events = (*proto->mutable_dev_events())[device_name];
  if (!has_dev_events) {
    dev_events.set_device_name(device_name);
  }

  auto start_us = self.start_walltime_us_ + ((rec.start_timestamp - self.start_timestamp_) / 1000);
  auto duration_us = std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);

  auto* event = dev_events.add_events();
  // Ideally we would record the operator name.
  // However, for now we have disabled collecting this for now since it requires enabling
  // libcupti runtime API callbacks.
//  event->name = ...;
  event->set_cuda_event_type(cuda_event_type);
  event->set_start_time_us(start_us);
  event->set_duration_us(duration_us);
}

std::unique_ptr<iml::MachineDevsEventsProto> CUDAActivityProfilerState::AsProto() {
  std::unique_ptr<iml::MachineDevsEventsProto> proto(new iml::MachineDevsEventsProto);

  proto->set_process_name(_process_name);
  proto->set_machine_name(_machine_name);
  proto->set_phase(_phase_name);

  const string prefix = "";
  // TODO: only works with gpu:0.
  const int id = 0;
  const string stream_device =
      strings::StrCat(prefix, "/device:GPU:", id, "/stream:");
  const string memcpy_device =
      strings::StrCat(prefix, "/device:GPU:", id, "/memcpy");

  for (auto const& rec : kernel_records_) {
    auto device_name = strings::StrCat(stream_device, "all");
    // TODO: TF also saves this under "${stream_device}:${rec.stream_id}"
    _AddEvent<KernelRecord>(*this, iml::CudaEventType::KERNEL, proto.get(), device_name, rec);
  }

  for (auto const& rec : memcpy_records_) {
    auto device_name = strings::StrCat(stream_device, "all");
    // TODO: TF also saves this under "${stream_device}:${rec.stream_id}"
    _AddEvent<MemcpyRecord>(*this, iml::CudaEventType::MEMCPY, proto.get(), memcpy_device, rec);
  }

  return proto;
}

void CUDAActivityProfiler::Print(std::ostream& out, int indent) {
  mutex_lock l(_mu);
  PrintIndent(out, indent);
  out << "CUDAActivityProfiler: "
      << "kernel_records.size() = " << _state.kernel_records_.size()
      << ", " << "memcpy_records.size() = " << _state.memcpy_records_.size();
//  for (auto const& rec : _state.kernel_records_) {
//    out << "\n";
//    PrintIndent(out, indent + 1);
//    out << "(tid=" << tid << ", api=" << name << "):";
//
//    out << "\n";
//    PrintIndent(out, indent + 1);
//    out << "total_api_time_usec = " << pair.second.total_api_time_usec;
//
//    out << "\n";
//    PrintIndent(out, indent + 1);
//    out << "n_calls = " << pair.second.n_calls;
//  }
}

void CUDAActivityProfiler::AsyncDump() {
  mutex_lock lock(_mu);
  {
    mutex_lock lock(_trace_mu);
    _AsyncDump();
  }
}

void CUDAActivityProfiler::_AsyncDumpWithState(CUDAActivityProfilerState&& dump_state) {
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

void CUDAActivityProfiler::_AsyncDump() {
  if (_state.CanDump()) {
    CUDAActivityProfilerState dump_state;
    dump_state = _state.DumpState();
    _AsyncDumpWithState(std::move(dump_state));
  }
}

void CUDAActivityProfiler::AwaitDump() {
  _pool.AwaitAll();
}

void CUDAActivityProfiler::SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  mutex_lock lock(_mu);
  {
    mutex_lock lock(_trace_mu);
    if (_state.CanDump()) {
      _AsyncDump();
    }
  }
  VLOG(1) << "CUDAActivityProfiler." << __func__
          << " " << "directory = " << directory
          << ", " << "process_name = " << process_name
          << ", " << "machine_name = " << machine_name
          << ", " << "phase_name = " << phase_name;
  _state._directory = directory;
  _state._process_name = process_name;
  _state._machine_name = machine_name;
  _state._phase_name = phase_name;
}

// NOTE: CUPTIManager calls into this.
void CUDAActivityProfiler::ActivityCallback(const CUpti_Activity &record) {
  VLOG(2) << "ActivityCallback " << record.kind;
  if (is_yes("TF_CUPTI_PRINT_ACTIVITY", false)) {
    printActivity(&record);
  }

  // Protect against shared access to _trace_mu:
  // - _state.DumpState()
  // - This callback
  {
  mutex_lock l(_trace_mu);
  switch (record.kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      if (_state.memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy *>(&record);
      _state.memcpy_records_.emplace_back(
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
//          memcpy->correlationId,
          memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      if (_state.memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy2 *>(&record);
      _state.memcpy_records_.emplace_back(
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
//          memcpy->correlationId,
          memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes);
      break;
    }

      // IML TODO: record contribution of libcupti overhead to profiling time.
      // Overhead could be CUPTI, DRIVER, COMPILER, etc; documentation:
      //
      //   CUPTI_ACTIVITY_OVERHEAD_UNKNOWN = 0
      //     The overhead kind is not known.
      //   CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER = 1
      //     Compiler(JIT) overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH = 1<<16
      //     Activity buffer flush overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION = 2<<16
      //     CUPTI instrumentation overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE = 3<<16
      //     CUPTI resource creation and destruction overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_FORCE_INT = 0x7fffffff


    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
//          if (is_yes("TF_CUPTI_RECORD_PROFILING_OVERHEAD", true)) {
//            printActivity(&record);
//          }
      if (_state.kernel_records_.size() >= kMaxRecords) return;
      auto *kernel = reinterpret_cast<const CUpti_ActivityKernel3 *>(&record);
      _state.kernel_records_.emplace_back(
          kernel->start, kernel->end,
          kernel->deviceId, kernel->streamId
//          kernel->correlationId
          );
      break;
    }

//    case CUPTI_ACTIVITY_KIND_OVERHEAD:
//      if (!is_yes("TF_CUPTI_RECORD_PROFILING_OVERHEAD", true)) {
//        break;
//      }
//      printActivity(&record);
//      // LOG(INFO) << "libcupti: CUPTI_ACTIVITY_KIND_OVERHEAD event: ";
//      break;

    default:
      VLOG(1) << "ActivityCallback unhandled kind";
      if (VLOG_IS_ON(1)) {
        printActivity(&record);
      }
      break;
  }

  _MaybeDump();

  }

}

void CUDAActivityProfiler::_MaybeDump() {
  if (_state.ShouldDump() && _state.CanDump()) {
    VLOG(1) << "CUDAActivityProfiler saw more than " << CUDA_ACTIVITY_PROFILER_MAX_RECORDS_PER_DUMP << " records; "
            << "triggering async dump.";
    auto dump_state = _state.DumpState();
    _AsyncDumpWithState(std::move(dump_state));
  }
}

Status CUDAActivityProfiler::Start() {
  mutex_lock lock(_mu);
  // NOTE: This registers ActivityCallback to be called, until DisableTrace is called.
  VLOG(1) << "CUDAActivityProfiler: " << __func__ << ", call EnableTrace";
  TF_RETURN_IF_ERROR(cupti_manager_->EnableTrace(this));
  CUPTI_CALL(cuptiGetTimestamp(&_state.start_timestamp_));
  _state.start_walltime_us_ = Env::Default()->NowMicros();
  return Status::OK();
}

Status CUDAActivityProfiler::Stop() {
  // VLOG(1) << "CUDAActivityProfiler." << __func__ << ": Grab mutex";
  mutex_lock lock(_mu);
  // VLOG(1) << "CUDAActivityProfiler." << __func__ << ": Mutex grabbed";
  Status status;
  status = cupti_manager_->DisableTrace();
//  TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace());
  MAYBE_LOG_ERROR(LOG(FATAL), __func__, status);
  _state.end_walltime_us_ = Env::Default()->NowMicros();
  CUPTI_CALL(cuptiGetTimestamp(&_state.end_timestamp_));
  return Status::OK();
}

//void CUDAActivityProfiler::ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) {
//  mutex_lock lock(_mu);
//  VLOG(2) << "ActivityBufferCallback";
//  // We're running on the main-thread;
//  // we don't want to delay until Collect is called, since we'll end up keeping the libcupti buffer allocated,
//  // which is especially bad if the buffer came from an Arena.
//
////  if (is_yes("TF_CUPTI_BUFFER_ARENA", false)) {
////    LOG(FATAL) << "Cannot use TF_CUPTI_BUFFER_ARENA=yes AND delay gathering of events from libcupti activity buffer, since arena will grow too large";
////  }
//
//  _state.activity_buffers_.push_back(std::move(activity_buffer));
//}

//class ProfileProtoBuilder {
//public:
//  ProfileProtoBuilder(
//      const std::string process_name,
//      const std::string phase,
//      const std::string machine_name)
//      : process_name_(process_name),
//        phase_(phase),
//        machine_name_(machine_name),
//        next_node_id_(0)
//  {
//    profile_proto_.set_process_name(process_name);
//    profile_proto_.set_phase(phase);
//    profile_proto_.set_machine_name(machine_name);
//    LOG(INFO) << "> ProfileProtoBuilder: "
//              << "process_name = " << process_name
//              << ", phase = " << phase
//              << ", machine_name = " << machine_name;
//  }
//
//  size_t SizeBytes() {
//    return profile_proto_.ByteSizeLong();
//  }
//
//  void Dump(const std::string& path) {
//    std::ofstream f;
//    f.open(path);
//    profile_proto_.SerializeToOstream(&f);
//  }
//
//  void AddRunMeta(int step, const RunMetadata& run_meta) {
//
//    auto IsGPUTime = [](const std::string& device) {
//      // Q: Does TF count the memcpy_records as GPU time?  From this regex, it doesn't seem like it does.
//      std::regex re(R"(stream:all)");
//      std::smatch match;
//      return std::regex_search(device, match, re);
//    };
//
//    auto IsCPUTime = [](const std::string& device) {
//      std::regex re(R"(.*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\d+)");
//      std::smatch match;
//      return std::regex_search(device, match, re);
//    };
//
//    if (std::find(profile_proto_.steps().begin(),
//                  profile_proto_.steps().end(),
//                  step) == profile_proto_.steps().end()) {
//      profile_proto_.add_steps(step);
//      CHECK(profile_proto_.steps_size() > 0);
//    }
//
//    for (auto const& dev_stat : run_meta.step_stats().dev_stats()) {
//      std::string dev = dev_stat.device();
//      std::transform(dev.begin(), dev.end(), dev.begin(), ::tolower);
//      for (auto const& node_stat : dev_stat.node_stats()) {
//        std::string name = node_stat.node_name();
//        std::regex re(R"((.*):)");
//        std::smatch match;
//        if (std::regex_search(name, match, re) && match.size() > 1) {
//          name = match.str(1);
//        }
//        auto name_to_id_it = name_to_id_.find(name);
//        int node_id;
//        if (name_to_id_it != name_to_id_.end()) {
//          node_id = name_to_id_it->second;
//        } else {
//          node_id = next_node_id_;
//          next_node_id_ += 1;
//          name_to_id_[name] = node_id;
//        }
//
//        bool has_node = profile_proto_.nodes().find(node_id) != profile_proto_.nodes().end();
//        auto& profile_node = (*profile_proto_.mutable_nodes())[node_id];
//        if (!has_node) {
//          profile_node.set_name(name);
//        }
//        if (node_stat.all_start_micros() > 0) {
//          auto op_end_rel_micros = std::max(static_cast<::google::protobuf::int64>(1), node_stat.op_end_rel_micros());
//
//          auto start_us = node_stat.all_start_micros();
//          auto end_us = op_end_rel_micros;
//
//          auto& exec_profile = (*profile_node.mutable_execs())[step];
//
//          tfprof::ExecTime* exec_time = nullptr;
//          if (IsGPUTime(dev)) {
//            exec_time = &(*exec_profile.mutable_accelerator_execs())[dev];
//          } else {
//            CHECK(IsCPUTime(dev));
//            exec_time = &(*exec_profile.mutable_cpu_execs())[dev];
//          }
//
//          auto tupl = exec_time->add_times();
//          tupl->add_int64_values(start_us);
//          tupl->add_int64_values(end_us);
//        }
//
//      }
//    }
//  }
//  const std::string process_name_;
//  const std::string phase_;
//  const std::string machine_name_;
//  tfprof::ProfileProto profile_proto_;
//  int next_node_id_;
//  std::map<const std::string, int> name_to_id_;
//};


//void ActivityBuffer::RecordActivitiesFromBuffer() {
//  if (is_yes("TF_CUPTI_ASYNC_RECORD_ACTIVITY_DEBUG", false)) {
//    LOG(INFO) << "RecordActivitiesFromBuffer";
//  }
//  if (_validSize > 0) {
//    CUptiResult status;
//    CUpti_Activity *record = nullptr;
//    do {
////      status = cupti_wrapper_->ActivityGetNextRecord(_buffer, _validSize, &record);
//      status = cuptiActivityGetNextRecord(_buffer, _validSize, &record);
//      if (status == CUPTI_SUCCESS) {
////          client_->ActivityCallback(*record);
////          this->_record_activity_callback(*record);
//        _manager->RecordActivityCallback(_client, *record);
//      } else {
//        break;
//      }
//    } while (1);
//
//    // report any records dropped from the queue
//    size_t dropped;
//    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(_ctx, _streamId, &dropped));
//    if (dropped != 0) {
//      LOG(WARNING) << "Dropped " << dropped << " activity records";
//    }
//  }
//  // All done recorded activities from libcupti buffer; free it now.
//  FreeBuffer();
//}
//
//void ActivityBuffer::FreeBuffer() {
//  if (_buffer) {
//    _manager->FreeBufferCallback(_buffer);
//    _buffer = nullptr;
//  }
//}
//
//ActivityBuffer::~ActivityBuffer() {
//  if (_buffer != nullptr && !is_yes("TF_CUPTI_EMPTY_TRACING_CALLBACKS", false)) {
//    LOG(WARNING) << "Looks like we forgot to record some GPU-time event data.  Make Sure RecordActivitiesFromBuffer gets called!";
//  }
//  FreeBuffer();
//}


//void CUDAActivityProfilerState::RecordActivityBuffers() {
//  for (auto& activity_buffer : activity_buffers_) {
//    activity_buffer->RecordActivitiesFromBuffer();
//  }
//  activity_buffers_.clear();
//}

CUDAActivityProfiler::CUDAActivityProfiler(CUPTIManager* cupti_manager) :
    _pool("CUDAActivityProfiler.pool", /*num_threads=*/5)
    , _enabled(false)
    , cupti_manager_(cupti_manager)
{
}

//void CUDAActivityProfiler::AddCorrelationId(uint32 correlation_id,
//                                        const string &name) {
//  VLOG(2) << correlation_id << " : " << name;
//  mutex_lock l(mu_);
//  if (_state.correlations_.size() >= kMaxRecords) return;
//  _state.correlations_.emplace(correlation_id, name);
//}

}
