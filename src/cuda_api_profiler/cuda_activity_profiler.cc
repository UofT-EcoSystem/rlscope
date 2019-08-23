//
// Created by jagle on 8/23/2019.
//

#include "cuda_api_profiler/cuda_activity_profiler.h"

TracedStepData::TracedStepData() :
    processed_step_data(new TraceDataProto())
{
}

StepStats* TracedStepData::GetNewStepStats(int64 tracer_step) {
  auto run_meta = (*processed_step_data->mutable_traced_steps())[tracer_step].add_traces();
  return run_meta->mutable_step_stats();
}

void TracedStepData::AddStep(
    int64 tracer_step,
    std::unique_ptr<DeviceTracer> tracer,
    std::unique_ptr<StepStatsCollector> collector) {

  if (tracer == nullptr) {
    VLOG(0) << "Tracer was NULL for step=" << tracer_step;
    assert(tracer != nullptr);
  }

  if (collector == nullptr) {
    VLOG(0) << "Collector was NULL for step=" << tracer_step;
    assert(collector != nullptr);
  }

  Status s = tracer->Stop();
  if (!s.ok()) {
    VLOG(0) << s.error_message();
    assert(s.ok());
  }

  assert(step_data.find(tracer_step) == step_data.end());
#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "AddStep step=" << tracer_step;
#endif
  step_data[tracer_step] = std::move(StepData(
      std::move(tracer), std::move(collector)));
}

Status TracedStepData::ProcessSteps() {
  for (const auto& it : step_data) {
    auto status = ProcessStep(it.first);
    if (!status.ok()) {
      VLOG(0) << "Failed to ProcessSteps; step=" << it.first << " not processed: " << status.error_message();
      assert(status.ok());
      return status;
    }
  }
  return Status::OK();
}

Status TracedStepData::StopTracer(int64 step) {
  StepData& data = step_data.at(step);
  Status s = data.tracer->Stop();
  if (!s.ok()) {
    VLOG(0) << s.error_message();
    assert(s.ok());
  }
  return s;
}

Status TracedStepData::ProcessStep(int64 step) {
#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "ProcessStep step=" << step << "...";
#endif
  StepData& data = step_data.at(step);

  assert(data.collector != nullptr);
  if (data.processed) {
    return Status::OK();
  }
  assert(data.tracer != nullptr);

#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "ProcessStep step=" << step << "...Processing...";
#endif

  print_timestamp("Start Collect stats");
  TF_RETURN_IF_ERROR(data.tracer->Collect(data.collector.get()));
  // Cannot register two tracers at once.
  data.tracer.reset(nullptr);
  print_timestamp("End Collect stats");

  print_timestamp("Start StepStatsCollector::Finalize");
  data.collector->Finalize();
  print_timestamp("End StepStatsCollector::Finalize");

  data.processed = true;
#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "Processed step=" << step;
#endif

  return Status::OK();
}

std::vector<int64> TracedStepData::Steps() const {
  std::vector<int64> steps;
  steps.reserve(step_data.size());
  for (const auto& it : step_data) {
    steps.push_back(it.first);
  }
  return steps;
}

void TracedStepData::Clear() {
  // Check that we only clear stuff that's been "processed" (i.e. read via c_api_util.get_trace_stats(sess).
  // NOTE: We don't want this anymore with async trace dumping.
//#ifndef NDEBUG
//  for (const auto& it : step_data) {
//    bool failed = false;
//    if (!it.second.processed) {
//      VLOG(0) << "Failed to clear; step=" << it.first << " not processed";
//      failed = failed || !(it.second.processed);
//    }
//    assert(!failed);
//  }
//#endif
  step_data.clear();
  processed_step_data.reset(new TraceDataProto());
}
std::unique_ptr<TraceDataProto> TracedStepData::GetTraceData() {
  ProcessSteps();
  return std::move(processed_step_data);
}

AsyncTraceDumper::AsyncTraceDumper() :
    async_dump_pool_(Env::Default(), ThreadOptions(), "AsyncTraceDumper.async_dump_pool_", 5, false),
    all_done_(new Notification()),
    dumps_scheduled_(0),
    waiters_(0)
{
}

void AsyncTraceDumper::DumpTraceDataAsync(
    std::unique_ptr<TraceDataProto> trace_data,
    const std::string dump_path,
    const std::string process_name,
    const std::string phase,
    const std::string machine_name) {
  {
    mutex_lock l(mu_);
    dumps_scheduled_ += 1;
  }

  // C++11 doesn't support std::move(...) lambda capture.
  // Instead, convert unique_ptr to shared_ptr which allows copy-constructor.
  // NOTE: C++14 does it though....
  //
  // https://stackoverflow.com/questions/8640393/move-capture-in-lambda
  auto trace_dump = std::make_shared<TraceDump>(
      std::move(trace_data),
      dump_path,
      process_name,
      phase,
      machine_name);

  auto fn = [this, trace_dump] () {
    this->_DumpTraceDataSync(*trace_dump);
  };
  async_dump_pool_.Schedule(fn);
}

void AsyncTraceDumper::AwaitTraceDataDumps() {
  {
    mutex_lock l(mu_);
    if (dumps_scheduled_ == 0) {
      return;
    }
    waiters_ += 1;
  }
  all_done_->WaitForNotification();
  {
    mutex_lock l(mu_);
    // The main python thread is the only one that can schedule
    // new dumps, and wait for dumps to finish.
    //
    // No other thread should be adding dumps while we were waiting.
    // So the only reason we wake up is if all dumps are finished.
    CHECK(dumps_scheduled_ == 0);
    waiters_ -= 1;
    if (waiters_ == 0) {
      _ResetNotification();
    }
  }
}

void AsyncTraceDumper::_ResetNotification() {
  CHECK(waiters_ == 0);
  CHECK(dumps_scheduled_ == 0);
  all_done_.reset(new Notification());
}

class ProfileProtoBuilder {
public:
  ProfileProtoBuilder(
      const std::string process_name,
      const std::string phase,
      const std::string machine_name)
      : process_name_(process_name),
        phase_(phase),
        machine_name_(machine_name),
        next_node_id_(0)
  {
    profile_proto_.set_process_name(process_name);
    profile_proto_.set_phase(phase);
    profile_proto_.set_machine_name(machine_name);
    LOG(INFO) << "> ProfileProtoBuilder: "
              << "process_name = " << process_name
              << ", phase = " << phase
              << ", machine_name = " << machine_name;
  }

  size_t SizeBytes() {
    return profile_proto_.ByteSizeLong();
  }

  void Dump(const std::string& path) {
    std::ofstream f;
    f.open(path);
    profile_proto_.SerializeToOstream(&f);
  }

  void AddRunMeta(int step, const RunMetadata& run_meta) {

    auto IsGPUTime = [](const std::string& device) {
      std::regex re(R"(stream:all)");
      std::smatch match;
      return std::regex_search(device, match, re);
    };

    auto IsCPUTime = [](const std::string& device) {
      std::regex re(R"(.*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\d+)");
      std::smatch match;
      return std::regex_search(device, match, re);
    };

    if (std::find(profile_proto_.steps().begin(),
                  profile_proto_.steps().end(),
                  step) == profile_proto_.steps().end()) {
      profile_proto_.add_steps(step);
      CHECK(profile_proto_.steps_size() > 0);
    }

    for (auto const& dev_stat : run_meta.step_stats().dev_stats()) {
      std::string dev = dev_stat.device();
      std::transform(dev.begin(), dev.end(), dev.begin(), ::tolower);
      for (auto const& node_stat : dev_stat.node_stats()) {
        std::string name = node_stat.node_name();
        std::regex re(R"((.*):)");
        std::smatch match;
        if (std::regex_search(name, match, re) && match.size() > 1) {
          name = match.str(1);
        }
        auto name_to_id_it = name_to_id_.find(name);
        int node_id;
        if (name_to_id_it != name_to_id_.end()) {
          node_id = name_to_id_it->second;
        } else {
          node_id = next_node_id_;
          next_node_id_ += 1;
          name_to_id_[name] = node_id;
        }

        bool has_node = profile_proto_.nodes().find(node_id) != profile_proto_.nodes().end();
        auto& profile_node = (*profile_proto_.mutable_nodes())[node_id];
        if (!has_node) {
          profile_node.set_name(name);
        }
        if (node_stat.all_start_micros() > 0) {
          auto op_end_rel_micros = std::max(static_cast<::google::protobuf::int64>(1), node_stat.op_end_rel_micros());

          auto start_us = node_stat.all_start_micros();
          auto end_us = op_end_rel_micros;

          auto& exec_profile = (*profile_node.mutable_execs())[step];

          tfprof::ExecTime* exec_time = nullptr;
          if (IsGPUTime(dev)) {
            exec_time = &(*exec_profile.mutable_accelerator_execs())[dev];
          } else {
            CHECK(IsCPUTime(dev));
            exec_time = &(*exec_profile.mutable_cpu_execs())[dev];
          }

          auto tupl = exec_time->add_times();
          tupl->add_int64_values(start_us);
          tupl->add_int64_values(end_us);
        }

      }
    }
  }
  const std::string process_name_;
  const std::string phase_;
  const std::string machine_name_;
  tfprof::ProfileProto profile_proto_;
  int next_node_id_;
  std::map<const std::string, int> name_to_id_;
};

void AsyncTraceDumper::_SerializeTraceDump(TraceDump& trace_dump) {
  auto byte_size = trace_dump.trace_data->ByteSize();

  // Q: Why is TraceDataProto.size so large in comparison to the ProfileProto that gets stored...?
  // If this is really how small it is, we should try to avoid storing so much data in the
  // first place to avoid CPU profiling overhead.
  LOG(INFO) << "> TraceDataProto.size = " << byte_size << " bytes, path = " << trace_dump.dump_path;

  auto profile_proto_builder = ProfileProtoBuilder(trace_dump.process_name, trace_dump.phase, trace_dump.machine_name);
  for (auto const& step_traces : trace_dump.trace_data->traced_steps()) {
    auto step = step_traces.first;
    for (auto const& run_meta : step_traces.second.traces()) {
      profile_proto_builder.AddRunMeta(step, run_meta);
    }
  }

  if (trace_dump.trace_data->traced_steps().size() == 0) {
    LOG(INFO) << "> tfprof didn't capture any session.run(...) calls!\n"
              << "Maybe try setting --iml-start-measuring-call lower (e.g. 0)?";
  }

  auto size_bytes = profile_proto_builder.SizeBytes();
  LOG(INFO) << "> Dump tfprof (" << size_bytes << " bytes) to: " << trace_dump.dump_path;
  profile_proto_builder.Dump(trace_dump.dump_path);
}

void AsyncTraceDumper::_DumpTraceDataSync(TraceDump& trace_dump) {
  // Dump TraceDataProto to file.
  _SerializeTraceDump(trace_dump);

  // Notify python-thread waiting for async dumping to finish.
  {
    mutex_lock l(mu_);

    dumps_scheduled_ -= 1;
    if (dumps_scheduled_ == 0) {
      all_done_->Notify();
    }

    if (waiters_ == 0 && dumps_scheduled_ == 0) {
      _ResetNotification();
    }
  }
}

std::unique_ptr<AsyncTraceDumper> _async_trace_dumper;
AsyncTraceDumper* GetAsyncTraceDumper() {
  if (!_async_trace_dumper) {
    _async_trace_dumper.reset(new AsyncTraceDumper());
  }
  return _async_trace_dumper.get();
}


std::unique_ptr<TraceDataProto> DirectSession::GetTraceData() {
#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "GetTraceData";
#endif
  return traced_step_data_.GetTraceData();
}

void DirectSession::DumpTraceDataAsync(
    const std::string& dump_path,
    const std::string& process_name,
    const std::string& phase,
    const std::string& machine_name) {
#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "DumpTraceDataAsync";
#endif
  auto async_trace_dumper = GetAsyncTraceDumper();
  LOG(INFO) << "Gather trace-data for async thread for path=" << dump_path << "...";
  auto trace_data = traced_step_data_.GetTraceData();
  LOG(INFO) << "... done";
  async_trace_dumper->DumpTraceDataAsync(
      std::move(trace_data),
      dump_path, process_name, phase, machine_name);
  traced_step_data_.Clear();
}

void DirectSession::AwaitTraceDataDumps() {
#ifdef CONFIG_DEBUG_TRACE_STATS
  VLOG(0) << "AwaitTraceDataDumps";
#endif
  auto async_trace_dumper = GetAsyncTraceDumper();
  async_trace_dumper->AwaitTraceDataDumps();
}
Status DirectSession::RunInternal(int64 step_id, const RunOptions& run_options,
                                  CallFrameInterface* call_frame,
                                  ExecutorsAndKeys* executors_and_keys,
                                  RunMetadata* run_metadata) {
  const uint64 start_time_usecs = Env::Default()->NowMicros();
  string session_id_meta = strings::StrCat("SessionRun #id=", step_id, "#");
  tracing::ScopedActivity activity(session_id_meta);

  const int64 executor_step_count = executors_and_keys->step_count.fetch_add(1);

#ifdef CONFIG_TRACE_STATS
  const bool do_trace = !is_yes("TF_DISABLE_CPP_TFPROF", false) && (run_options.trace_level() > RunOptions::NO_TRACE);

  uint64
      end_run_internal_us = 0, start_run_internal_us = 0,
      end_collect_stats_us = 0, start_collect_stats_us = 0,
      end_finalize_us = 0, start_finalize_us = 0,
      end_wait_for_notify_t = 0, start_wait_for_notify_t = 0;

  start_run_internal_us = Env::Default()->NowMicros();
  if (do_trace) {
    print_timestamp("Start RunInternal");
  }
#endif // CONFIG_TRACE_STATS

  std::unique_ptr<DebuggerStateInterface> debugger_state;
  if (!run_options.debug_options().debug_tensor_watch_opts().empty()) {
    TF_RETURN_IF_ERROR(
        CreateDebuggerState(executors_and_keys->callable_options,
                            run_options.debug_options().global_step(), step_id,
                            executor_step_count, &debugger_state));
  }

  // Create a run state and start execution.
  RunState run_state(step_id, &devices_);
  run_state.rendez = new IntraProcessRendezvous(device_mgr_.get());
#ifndef __ANDROID__
  // Set up for collectives if ExecutorsAndKeys declares a key.
  if (executors_and_keys->collective_graph_key !=
      BuildGraphOptions::kNoCollectiveGraphKey) {
    if (run_options.experimental().collective_graph_key() !=
        BuildGraphOptions::kNoCollectiveGraphKey) {
      // If a collective_graph_key was specified in run_options, ensure that it
      // matches what came out of GraphExecutionState::BuildGraph().
      if (run_options.experimental().collective_graph_key() !=
          executors_and_keys->collective_graph_key) {
        return errors::Internal(
            "collective_graph_key in RunOptions ",
            run_options.experimental().collective_graph_key(),
            " should match collective_graph_key from optimized graph ",
            executors_and_keys->collective_graph_key);
      }
    }
    if (!collective_executor_mgr_) {
      std::unique_ptr<DeviceResolverInterface> drl(
          new DeviceResolverLocal(device_mgr_.get()));
      std::unique_ptr<ParamResolverInterface> cprl(
          new CollectiveParamResolverLocal(device_mgr_.get(), drl.get(),
                                           "/job:localhost/replica:0/task:0"));
      collective_executor_mgr_.reset(new CollectiveExecutorMgr(
          options_.config, device_mgr_.get(), std::move(drl), std::move(cprl)));
    }
    run_state.collective_executor.reset(new CollectiveExecutor::Handle(
        collective_executor_mgr_->FindOrCreate(step_id), true /*inherit_ref*/));
  }
#endif

  // Start parallel Executors.
  const size_t num_executors = executors_and_keys->items.size();
  ExecutorBarrier* barrier = new ExecutorBarrier(
      num_executors, run_state.rendez, [&run_state](const Status& ret) {
        {
          mutex_lock l(run_state.mu_);
          run_state.status.Update(ret);
        }
        run_state.executors_done.Notify();
      });

  Executor::Args args;
  args.step_id = step_id;
  args.call_frame = call_frame;
  args.rendezvous = run_state.rendez;
  args.collective_executor =
      (run_state.collective_executor ? run_state.collective_executor->get()
                                     : nullptr);
  CancellationManager step_cancellation_manager;
  args.cancellation_manager = &step_cancellation_manager;
  args.session_state = &session_state_;
  args.tensor_store = &run_state.tensor_store;
  args.step_container = &run_state.step_container;
  args.sync_on_finish = sync_on_finish_;

#ifndef CONFIG_TRACE_STATS
  const bool do_trace = (run_options.trace_level() > RunOptions::NO_TRACE);
#endif // !CONFIG_TRACE_STATS

  bool update_cost_model = false;
  if (options_.config.graph_options().build_cost_model() > 0) {
    const int64 build_cost_model_every =
        options_.config.graph_options().build_cost_model();
    const int64 build_cost_model_after =
        options_.config.graph_options().build_cost_model_after();
    int64 measure_step_count = executor_step_count - build_cost_model_after;
    if (measure_step_count >= 0) {
      update_cost_model =
          ((measure_step_count + 1) % build_cost_model_every == 0);
    }
  }
  if (do_trace || update_cost_model ||
      run_options.report_tensor_allocations_upon_oom()) {
    run_state.collector.reset(
#ifdef CONFIG_TRACE_STATS
        new StepStatsCollector(traced_step_data_.GetNewStepStats(tracer_step_)
#else
        new StepStatsCollector(run_metadata->mutable_step_stats()
#endif
            ));
    args.stats_collector = run_state.collector.get();
  }

#ifdef CONFIG_TRACE_STATS
  // Either they're both set, or neither are set.
  //
  // i.e. If we preallocate a tracer, then the next call should be session.run(FULL_TRACE)
  //
  // Allow tracer_ to be set when TF_DISABLE_CPP_TFPROF=yes.
  // For performance debugging to allow us to quanity C++ profile-book-keeping overhead.
  if (do_trace || tracer_) {
    assert(do_trace || is_yes("TF_DISABLE_CPP_TFPROF", false));
    assert(tracer_);
  }
#else
  std::unique_ptr<DeviceTracer> tracer;
  if (run_options.trace_level() >= RunOptions::HARDWARE_TRACE) {
    tracer = CreateDeviceTracer();
    // tracer may be NULL on platforms without accelerators.
    if (tracer) {
      Status s = tracer->Start();
      if (!s.ok()) {
        run_state.executors_done.Notify();
        delete barrier;
        return s;
      }
    }
  }
#endif

  if (run_options.inter_op_thread_pool() < -1 ||
      run_options.inter_op_thread_pool() >=
          static_cast<int32>(thread_pools_.size())) {
    run_state.executors_done.Notify();
    delete barrier;
    return errors::InvalidArgument("Invalid inter_op_thread_pool: ",
                                   run_options.inter_op_thread_pool());
  }

  // Register this step with session's cancellation manager, so that
  // `Session::Close()` will cancel the step.
  const CancellationToken cancellation_token =
      cancellation_manager_->get_cancellation_token();
  const bool already_cancelled = !cancellation_manager_->RegisterCallback(
      cancellation_token, [&step_cancellation_manager]() {
        step_cancellation_manager.StartCancel();
      });
  if (already_cancelled) {
    // NOTE(mrry): If we don't explicitly notify
    // `run_state.executors_done`, the RunState destructor would
    // block on this notification.
    run_state.executors_done.Notify();
    delete barrier;
    return errors::Cancelled("Run call was cancelled");
  }

  thread::ThreadPool* pool =
      run_options.inter_op_thread_pool() >= 0
          ? thread_pools_[run_options.inter_op_thread_pool()].first
          : nullptr;

  if (pool == nullptr) {
    // We allow using the caller thread only when having a single executor
    // specified.
    if (executors_and_keys->items.size() > 1) {
      pool = thread_pools_[0].first;
    } else {
      VLOG(1) << "Executing Session::Run() synchronously!";
    }
  }

  std::unique_ptr<RunHandler> handler;
  if (ShouldUseRunHandlerPool(run_options) &&
      run_options.experimental().use_run_handler_pool()) {
    VLOG(1) << "Using RunHandler to scheduler inter-op closures.";
    handler = GetOrCreateRunHandlerPool(options_)->Get();
  }
  auto* handler_ptr = handler.get();

  Executor::Args::Runner default_runner = nullptr;

  if (pool == nullptr) {
    default_runner = [](Executor::Args::Closure c) { c(); };
  } else if (handler_ptr != nullptr) {
    default_runner = [handler_ptr](Executor::Args::Closure c) {
      handler_ptr->ScheduleInterOpClosure(std::move(c));
    };
  } else {
    default_runner = [this, pool](Executor::Args::Closure c) {
      SchedClosure(pool, std::move(c));
    };
  }

  for (const auto& item : executors_and_keys->items) {
    // TODO(azaks): support partial run.
    // TODO(azaks): if the device picks its own threadpool, we need to assign
    //     less threads to the main compute pool by default.
    thread::ThreadPool* device_thread_pool =
        item.device->tensorflow_device_thread_pool();
    // TODO(crk): Investigate usage of RunHandlerPool when using device specific
    // thread pool(s).
    if (!device_thread_pool) {
      args.runner = default_runner;
    } else {
      args.runner = [this, device_thread_pool](Executor::Args::Closure c) {
        SchedClosure(device_thread_pool, std::move(c));
      };
    }
    item.executor->RunAsync(args, barrier->Get());
  }

#ifdef CONFIG_TRACE_STATS
  start_wait_for_notify_t = Env::Default()->NowMicros();
  if (do_trace) {
    print_timestamp("Start wait for notification");
  }
#endif // CONFIG_TRACE_STATS
  WaitForNotification(&run_state, &step_cancellation_manager,
                      run_options.timeout_in_ms() > 0
                          ? run_options.timeout_in_ms()
                          : operation_timeout_in_ms_);
#ifdef CONFIG_TRACE_STATS
  if (do_trace) {
    print_timestamp("End wait for notification");
  }
  end_wait_for_notify_t = Env::Default()->NowMicros();
#endif // CONFIG_TRACE_STATS

  if (!cancellation_manager_->DeregisterCallback(cancellation_token)) {
    // The step has been cancelled: make sure we don't attempt to receive the
    // outputs as this would make it block forever.
    mutex_lock l(run_state.mu_);
    run_state.status.Update(errors::Cancelled("Run call was cancelled"));
  }

#ifdef CONFIG_TRACE_STATS
  if (tracer_ && !is_yes("TF_DISABLE_CPP_TFPROF", false)) {
    traced_step_data_.AddStep(
        /* NOTE: we DON'T use step_id, since step_id is a global step counter across all DirectSession's.
         * Hence, it's easier to keep tfprof step_id's in-sync with pyprof step_id's if we use this instead.
         */
        tracer_step_,
        std::move(tracer_),
        std::move(run_state.collector)
        );
  }
#else // CONFIG_TRACE_STATS
  if (tracer) {
    TF_RETURN_IF_ERROR(tracer->Stop());
    TF_RETURN_IF_ERROR(tracer->Collect(run_state.collector.get()));
  }
#endif // CONFIG_TRACE_STATS

  {
    mutex_lock l(run_state.mu_);
    TF_RETURN_IF_ERROR(run_state.status);
  }

  // Save the output tensors of this run we choose to keep.
  if (!run_state.tensor_store.empty()) {
    TF_RETURN_IF_ERROR(run_state.tensor_store.SaveTensors(
        {executors_and_keys->callable_options.fetch().begin(),
         executors_and_keys->callable_options.fetch().end()},
        &session_state_));
  }

#ifndef CONFIG_TRACE_STATS
  if (run_state.collector) {
    run_state.collector->Finalize();
  }
#endif // !CONFIG_TRACE_STATS

  // Build and return the cost model as instructed.
#ifdef CONFIG_TRACE_STATS
  // JAMES NOTE: don't bother with this code path; I don't want to bother with delaying calling this during profiling
  assert(!update_cost_model);
#endif
  if (update_cost_model) {
    // Build the cost model
    std::unordered_map<string, const Graph*> device_to_graph;
    for (const PerPartitionExecutorsAndLib& partition :
         executors_and_keys->items) {
      const Graph* graph = partition.graph;
      const string device = partition.flib->device()->name();
      device_to_graph[device] = graph;
    }

    mutex_lock l(executor_lock_);
    run_state.collector->BuildCostModel(&cost_model_manager_, device_to_graph);

    // annotate stats onto cost graph.
    CostGraphDef* cost_graph = run_metadata->mutable_cost_graph();
    for (const auto& item : executors_and_keys->items) {
      TF_RETURN_IF_ERROR(
          cost_model_manager_.AddToCostGraphDef(item.graph, cost_graph));
    }
  }

  // If requested via RunOptions, output the partition graphs.
  if (run_options.output_partition_graphs()) {
    protobuf::RepeatedPtrField<GraphDef>* partition_graph_defs =
        run_metadata->mutable_partition_graphs();
    for (const PerPartitionExecutorsAndLib& exec_and_lib :
         executors_and_keys->items) {
      GraphDef* partition_graph_def = partition_graph_defs->Add();
      exec_and_lib.graph->ToGraphDef(partition_graph_def);
    }
  }

#ifdef CONFIG_TRACE_STATS
  if (do_trace) {
    print_timestamp("End RunInternal");
  }
  end_run_internal_us = Env::Default()->NowMicros();

  uint64 time_us;

  uint64 run_internal_us = end_run_internal_us - start_run_internal_us;
  print_total_usec("RunInternal", end_run_internal_us - start_run_internal_us);

  uint64 profile_corrected_us = run_internal_us;

  time_us = end_collect_stats_us - start_collect_stats_us;
  profile_corrected_us -= time_us;
  print_total_usec("Collect stats", time_us);

  time_us = end_finalize_us - start_finalize_us;
  profile_corrected_us -= time_us;
  print_total_usec("StepStatsCollector::Finalize", time_us);

  time_us = end_wait_for_notify_t - start_wait_for_notify_t;
  print_total_usec("Wait for notification", time_us);

  print_total_usec("Profiling-corrected RunInternal", profile_corrected_us);

  // Let diff_us = [No-profiling RunInternal] - [Profiling-corrected RunInternal]
  // NOTE: [No-profiling RunInternal] = [RunInternal] when profiling is NOT running.
  //
  // If diff_us is SMALL, then we have accounted for all the profiling related overheads in this function.
  //
#endif // CONFIG_TRACE_STATS

  UpdateGraphExecTime(Env::Default()->NowMicros() - start_time_usecs);

  return Status::OK();
}
