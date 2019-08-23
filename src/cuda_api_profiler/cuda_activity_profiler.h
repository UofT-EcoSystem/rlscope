//
// Created by jagle on 8/23/2019.
//

#ifndef IML_CUDA_ACTIVITY_PROFILER_H
#define IML_CUDA_ACTIVITY_PROFILER_H

class TracedStepData {
public:
  struct StepData {
    std::unique_ptr<DeviceTracer> tracer;
    std::unique_ptr<StepStatsCollector> collector;
    bool processed;

    StepData() {}

    StepData(
        std::unique_ptr<DeviceTracer> tracer_
        , std::unique_ptr<StepStatsCollector> collector_
    ) :
        tracer(std::move(tracer_)),
        collector(std::move(collector_)),
        processed(false)
    {
    }

  };
  std::unordered_map<int64, StepData> step_data;
  std::unique_ptr<TraceDataProto> processed_step_data;

  TracedStepData();
  StepStats* GetNewStepStats(int64 tracer_step);
  void AddStep(
      int64 tracer_step,
      std::unique_ptr<DeviceTracer> tracer,
      std::unique_ptr<StepStatsCollector> collector);
  Status ProcessSteps();
  Status StopTracer(int64 step);
  Status ProcessStep(int64 step);
  std::vector<int64> Steps() const;
  void Clear();
  std::unique_ptr<TraceDataProto> GetTraceData();
};

struct TraceDump {
  std::unique_ptr<TraceDataProto> trace_data;
  const std::string dump_path;
  const std::string process_name;
  const std::string phase;
  const std::string machine_name;
  TraceDump();
  TraceDump(
      std::unique_ptr<TraceDataProto> _trace_data,
      const std::string _dump_path,
      const std::string _process_name,
      const std::string _phase,
      const std::string _machine_name) :
      trace_data(std::move(_trace_data)),
      dump_path(_dump_path),
      process_name(_process_name),
      phase(_phase),
      machine_name(_machine_name)
  {
  }
};
// Singleton class for dumping TraceDataProto asynchronously to avoid blocking python-side.
class AsyncTraceDumper {
public:
  AsyncTraceDumper();
  void DumpTraceDataAsync(
      std::unique_ptr<TraceDataProto> trace_data,
      const std::string dump_path,
      const std::string process_name,
      const std::string phase,
      const std::string machine_name);
  void AwaitTraceDataDumps();
private:
  void _ResetNotification();
  void _DumpTraceDataSync(TraceDump& trace_dump);
  void _SerializeTraceDump(TraceDump& trace_dump);

  thread::ThreadPool async_dump_pool_;
  std::unique_ptr<Notification> all_done_;
  mutex mu_;
  int dumps_scheduled_ GUARDED_BY(mu_);
  int waiters_ GUARDED_BY(mu_);
};

extern std::unique_ptr<AsyncTraceDumper> _async_trace_dumper;
AsyncTraceDumper* GetAsyncTraceDumper();

#endif //IML_CUDA_ACTIVITY_PROFILER_H
