//
// Created by jagle on 8/23/2019.
//

#ifndef IML_CUDA_ACTIVITY_PROFILER_H
#define IML_CUDA_ACTIVITY_PROFILER_H

#include <stdlib.h>

#include <driver_types.h>
#include <cupti_target.h>
#include <cupti.h>

#include <cuda.h>
//#include <driver_types.h>
//#include <cupti.h>
//#include <cupti_host.h>

#include <map>
#include <vector>
#include <memory>

#include <mutex>

//#include "iml_profiler/protobuf/iml_prof.pb.h"
#include "iml_prof.pb.h"

#include "cuda_api_profiler/cupti_manager.h"
#include "cuda_api_profiler/thread_pool_wrapper.h"

namespace rlscope {

// Internal struct to record kernel launches.
struct KernelRecord {
  KernelRecord(
      uint64_t start_timestamp_,
      uint64_t end_timestamp_,
      uint32_t device_id_,
      uint32_t stream_id_
//      uint32_t correlation_id_,
  ) :
      start_timestamp(start_timestamp_)
      , end_timestamp(end_timestamp_)
      , device_id(device_id_)
      , stream_id(stream_id_)
//      , correlation_id(correlation_id_)
  {
  }
  uint64_t start_timestamp;
  uint64_t end_timestamp;
  uint32_t device_id;
  uint32_t stream_id;
//  uint32_t correlation_id;
};
// Internal struct to record memcpy operations.
struct MemcpyRecord {
  MemcpyRecord(
      uint64_t start_timestamp_,
      uint64_t end_timestamp_,
      uint32_t device_id_,
      uint32_t stream_id_,
//      uint32_t correlation_id_,
      uint8_t copyKind_,
      uint8_t srcKind_,
      uint8_t dstKind_,
      uint64_t bytes_) :
      start_timestamp(start_timestamp_)
      , end_timestamp(end_timestamp_)
      , device_id(device_id_)
      , stream_id(stream_id_)
//      , correlation_id(correlation_id_)
      , copyKind(copyKind_)
      , srcKind(srcKind_)
      , dstKind(dstKind_)
      , bytes(bytes_)
  {
  }
  uint64_t start_timestamp;
  uint64_t end_timestamp;
  uint32_t device_id;
  uint32_t stream_id;
//  uint32_t correlation_id;
  uint8_t copyKind;
  uint8_t srcKind;
  uint8_t dstKind;
  uint64_t bytes;
};

static constexpr size_t kMaxRecords = 1024 * 1024;
struct CUDAActivityProfilerState {
  //
  std::string _directory;
  std::string _process_name;
  std::string _machine_name;
  std::string _phase_name;
  int _next_trace_id;
  int _trace_id;

  //
  std::map<uint32_t, std::string> correlations_;
  std::vector<KernelRecord> kernel_records_;
  std::vector<MemcpyRecord> memcpy_records_;

//  std::vector<std::unique_ptr<ActivityBuffer>> activity_buffers_;

  //
  int64_t start_walltime_us_;
  int64_t end_walltime_us_;
  uint64_t start_timestamp_;
  uint64_t end_timestamp_;

  // WARNING: if you add a member here, don't forget to copy the field in DumpState()!
  CUDAActivityProfilerState() :

      _next_trace_id(0)
      , _trace_id(-1)

      , start_walltime_us_(0)
      , end_walltime_us_(0)
      , start_timestamp_(0)
      , end_timestamp_(0)

  {
  }

  bool ShouldDump();
  bool CanDump();
  std::string DumpPath(int trace_id);
  CUDAActivityProfilerState DumpState();
  std::unique_ptr<iml::MachineDevsEventsProto> AsProto();

  void RecordActivityBuffers();
};

class CUDAActivityProfiler : public CUPTIClient {
public:
  ThreadPoolWrapper _pool;
  // - Protects start/end-tracing timestamps
  // - Monitor lock used to serialize all the DeviceTracerImpl method calls (Start/Stop/Collect).
  std::mutex _mu;
  // - Protects kernel_records, memcpy_records, correlations
  // - Grabbed during:
  //   - Activity callbacks
  //   - When collecting records
  std::mutex _trace_mu;
//  CUDAActivityProfilerState _state;
  CUDAActivityProfilerState _state;
//  std::vector<std::unique_ptr<ActivityBuffer>> activity_buffers_;
  bool _enabled;
  CUPTIManager* cupti_manager_;
  CUDAActivityProfiler(CUPTIManager* cupti_manager);

  void Print(std::ostream& out, int indent);

  void SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name);
  void AsyncDump();
  void _AsyncDump();
  void _AsyncDumpWithState(CUDAActivityProfilerState&& dump_state);
  void _MaybeDump();
  void AwaitDump();

  MyStatus Start();
  MyStatus Stop();

  void ActivityCallback(const CUpti_Activity &activity) override;
//  void ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) override;
};

//class ActivityBuffer {
//public:
////  using RecordActivityCallback = std::function<void(const CUpti_Activity &activity)>;
//  ActivityBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize,
//                 CUPTIManager* manager, CUPTIClient* client) :
////      RecordActivityCallback record_activity_callback) :
//      _ctx(ctx)
//      , _streamId(streamId)
//      , _buffer(buffer)
//      , _size(size)
//      , _validSize(validSize)
//      , _manager(manager)
//      , _client(client)
////      , _record_activity_callback(record_activity_callback)
//  {
////    cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
//  }
//  CUcontext _ctx;
//  uint32_t _streamId;
//  uint8_t *_buffer;
//  size_t _size;
//  size_t _validSize;
//  CUPTIManager* _manager;
//  CUPTIClient* _client;
////  RecordActivityCallback _record_activity_callback;
////  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;
//
//  void RecordActivitiesFromBuffer();
//
//  void FreeBuffer();
//
//  ~ActivityBuffer();
//
//};

//class TracedStepData {
//public:
//  struct StepData {
//    std::unique_ptr<DeviceTracer> tracer;
//    std::unique_ptr<StepStatsCollector> collector;
//    bool processed;
//
//    StepData() {}
//
//    StepData(
//        std::unique_ptr<DeviceTracer> tracer_
//        , std::unique_ptr<StepStatsCollector> collector_
//    ) :
//        tracer(std::move(tracer_)),
//        collector(std::move(collector_)),
//        processed(false)
//    {
//    }
//
//  };
//  std::unordered_map<int64_t, StepData> step_data;
//  std::unique_ptr<TraceDataProto> processed_step_data;
//
//  TracedStepData();
//  StepStats* GetNewStepStats(int64_t tracer_step);
//  void AddStep(
//      int64_t tracer_step,
//      std::unique_ptr<DeviceTracer> tracer,
//      std::unique_ptr<StepStatsCollector> collector);
//  MyStatus ProcessSteps();
//  MyStatus StopTracer(int64_t step);
//  MyStatus ProcessStep(int64_t step);
//  std::vector<int64_t> Steps() const;
//  void Clear();
//  std::unique_ptr<TraceDataProto> GetTraceData();
//};

//struct TraceDump {
//  std::unique_ptr<TraceDataProto> trace_data;
//  const std::string dump_path;
//  const std::string process_name;
//  const std::string phase;
//  const std::string machine_name;
//  TraceDump();
//  TraceDump(
//      std::unique_ptr<TraceDataProto> _trace_data,
//      const std::string _dump_path,
//      const std::string _process_name,
//      const std::string _phase,
//      const std::string _machine_name) :
//      trace_data(std::move(_trace_data)),
//      dump_path(_dump_path),
//      process_name(_process_name),
//      phase(_phase),
//      machine_name(_machine_name)
//  {
//  }
//};
//// Singleton class for dumping TraceDataProto asynchronously to avoid blocking python-side.
//class AsyncTraceDumper {
//public:
//  AsyncTraceDumper();
//  void DumpTraceDataAsync(
//      std::unique_ptr<TraceDataProto> trace_data,
//      const std::string dump_path,
//      const std::string process_name,
//      const std::string phase,
//      const std::string machine_name);
//  void AwaitTraceDataDumps();
//private:
//  void _ResetNotification();
//  void _DumpTraceDataSync(TraceDump& trace_dump);
//  void _SerializeTraceDump(TraceDump& trace_dump);
//
//  thread::ThreadPool async_dump_pool_;
//  std::unique_ptr<Notification> all_done_;
//  std::mutex mu_;
//  int dumps_scheduled_;
//  int waiters_;
//};
//
//extern std::unique_ptr<AsyncTraceDumper> _async_trace_dumper;
//AsyncTraceDumper* GetAsyncTraceDumper();

}

#endif //IML_CUDA_ACTIVITY_PROFILER_H
