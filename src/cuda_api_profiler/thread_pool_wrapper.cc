//
// Created by jagle on 8/6/2019.
//

#include "cuda_api_profiler/thread_pool_wrapper.h"

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

ThreadPoolWrapper::ThreadPoolWrapper(const string& name, int num_threads) :
        async_dump_pool_(Env::Default(), ThreadOptions(), name, num_threads, false),
        all_done_(new Notification()),
        fns_scheduled_(0),
        waiters_(0)
{
}

void ThreadPoolWrapper::Schedule(Func fn) {
    {
        mutex_lock l(mu_);
        fns_scheduled_ += 1;
    }
    async_dump_pool_.Schedule(fn);
}

void ThreadPoolWrapper::AwaitAll() {
    {
        mutex_lock l(mu_);
        if (fns_scheduled_ == 0) {
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
        CHECK(fns_scheduled_ == 0);
        waiters_ -= 1;
        if (waiters_ == 0) {
            _ResetNotification();
        }
    }
}

void ThreadPoolWrapper::_ResetNotification() {
    CHECK(waiters_ == 0);
    CHECK(fns_scheduled_ == 0);
    all_done_.reset(new Notification());
}

//void ThreadPoolWrapper::_SerializeTraceDump(TraceDump& trace_dump) {
//    auto byte_size = trace_dump.trace_data->ByteSize();
//
//    // Q: Why is TraceDataProto.size so large in comparison to the ProfileProto that gets stored...?
//    // If this is really how small it is, we should try to avoid storing so much data in the
//    // first place to avoid CPU profiling overhead.
//    LOG(INFO) << "> TraceDataProto.size = " << byte_size << " bytes, path = " << trace_dump.dump_path;
//
//    auto profile_proto_builder = ProfileProtoBuilder(trace_dump.process_name, trace_dump.phase, trace_dump.machine_name);
//    for (auto const& step_traces : trace_dump.trace_data->traced_steps()) {
//        auto step = step_traces.first;
//        for (auto const& run_meta : step_traces.second.traces()) {
//            profile_proto_builder.AddRunMeta(step, run_meta);
//        }
//    }
//
//    if (trace_dump.trace_data->traced_steps().size() == 0) {
//        LOG(INFO) << "> tfprof didn't capture any session.run(...) calls!\n"
//                  << "Maybe try setting --iml-start-measuring-call lower (e.g. 0)?";
//    }
//
//    auto size_bytes = profile_proto_builder.SizeBytes();
//    LOG(INFO) << "> Dump tfprof (" << size_bytes << " bytes) to: " << trace_dump.dump_path;
//    profile_proto_builder.Dump(trace_dump.dump_path);
//}
//
//void ThreadPoolWrapper::_DumpTraceDataSync(TraceDump& trace_dump) {
//    // Dump TraceDataProto to file.
//    _SerializeTraceDump(trace_dump);
//
//    // Notify python-thread waiting for async dumping to finish.
//    {
//        mutex_lock l(mu_);
//
//        fns_scheduled_ -= 1;
//        if (fns_scheduled_ == 0) {
//            all_done_->Notify();
//        }
//
//        if (waiters_ == 0 && fns_scheduled_ == 0) {
//            _ResetNotification();
//        }
//    }
//}

//std::unique_ptr<ThreadPoolWrapper> _async_trace_dumper;
//ThreadPoolWrapper* GetThreadPoolWrapper() {
//    if (!_async_trace_dumper) {
//        _async_trace_dumper.reset(new ThreadPoolWrapper());
//    }
//    return _async_trace_dumper.get();
//}

}
