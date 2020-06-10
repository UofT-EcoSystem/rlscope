/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/platform/logging.h"

#include "cuda_api_profiler/op_stack.h"
#include "cuda_api_profiler/cuda_api_profiler.h"
#include "cuda_api_profiler/cupti_manager.h"
#include "cuda_api_profiler/cuda_api_profiler.h"
#include "cuda_api_profiler/get_env_var.h"
#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/cuda_stream_monitor.h"
#include "cuda_api_profiler/cupti_api_wrapper.h"
#include "cuda_api_profiler/event_profiler.h"
#ifdef WITH_CUDA_LD_PRELOAD
#include "cuda_api_profiler/cuda_ld_preload.h"
#include "cuda_api_profiler/registered_handle.h"
#endif

//#include "iml_profiler/protobuf/iml_prof.pb.h"
#include "iml_prof.pb.h"

#include "cuda_api_profiler/cuda_activity_profiler.h"

#include <cuda.h>
#include <cupti.h>

#include <absl/memory/memory.h>

#include <atomic>
#include <map>
//#include <mutex>
#include <vector>

#ifdef CONFIG_TRACE_STATS

// Record memcpy/kernel launch timings.
//
// NOTE: Even if this is disabled, the callbacks for recording
// these statistics ARE still triggered.
//#define ENABLE_GPU_TRACING (true)

// Register CUDA API callbacks, but do nothing inside of them.
//#define TF_CUPTI_EMPTY_TRACING_CALLBACKS (false)

// Skip registering CUDA API callbacks altogether.
//#define TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS (false)

#endif // CONFIG_TRACE_STATS

#include <stdlib.h>
#include <memory>
#include <algorithm>
#include <list>
#include <cassert>

//#include <google/protobuf/arena.h>

//#include "tensorflow/core/common_runtime/step_stats_collector.h"
//#include "tensorflow/core/framework/step_stats.pb.h"
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/lib/strings/strcat.h"
//#include "tensorflow/core/lib/strings/stringprintf.h"
//#include "tensorflow/core/platform/cupti_wrapper.h"
#include "tensorflow/core/platform/env.h"
//#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "cuda_api_profiler/cupti_logging.h"

//#define mutex std::mutex
//#define mutex_lock std::lock_guard<std::mutex>

// Print time of operations that occur during sample_cuda_api.disable_tracing() / DeviceTracerImpl::Stop()
// that are on the critical-path of the script exiting during Profiler.finish().
//#define DEBUG_CRITICAL_PATH

namespace {


}  // namespace

namespace rlscope {

class CUDAAPIProfiler;

namespace devicetracer {

// Used by ActivityBuffer and DeviceTracerImpl
#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = call;                                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      const char *errstr;                                           \
      cuptiGetResultString(_status, &errstr);                       \
      LOG(FATAL) << "libcupti call " << #call << " failed with " << errstr; \
    }                                                               \
  } while (0)

#ifdef _MSC_VER
#define __thread __declspec(thread)
#endif

// TODO(pbar) Move this to platform specific header file?
// Static thread local variable for POD types.
#define TF_STATIC_THREAD_LOCAL_POD(_Type_, _var_)                  \
  static __thread _Type_ s_obj_##_var_;                            \
  namespace {                                                      \
  class ThreadLocal_##_var_ {                                      \
   public:                                                         \
    ThreadLocal_##_var_() {}                                       \
    void Init() {}                                                 \
    inline _Type_ *pointer() const { return &s_obj_##_var_; }      \
    inline _Type_ *safe_pointer() const { return &s_obj_##_var_; } \
    _Type_ &get() const { return s_obj_##_var_; }                  \
    bool is_native_tls() const { return true; }                    \
                                                                   \
   private:                                                        \
    TF_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);              \
  } _var_;                                                         \
  }  // namespace

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the CUPTI API callback.
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_annotation);

class TraceCollectorImpl : public tracing::TraceCollector {
public:
  TraceCollectorImpl() :
      active_trace_session_(false)
  {
    tracing::SetTraceCollector(this);
  }

  ~TraceCollectorImpl() override {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
  }

  // Note the method can be called after a call to Stop().
  virtual std::unique_ptr<Handle> CreateAnnotationHandle(
      StringPiece name_part1, StringPiece name_part2) const {
    struct Impl : public tracing::TraceCollector::Handle {
      string annotation;
      explicit Impl(string &&name_scope) : annotation(name_scope) {
        VLOG(2) << "CreateAnnotationHandle " << annotation;
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation.get() = annotation.c_str();
      }
      ~Impl() override { tls_current_annotation.get() = nullptr; }
    };
    return std::unique_ptr<Handle>(
        new Impl{ConcatenateNames(name_part1, name_part2)});
  }

  virtual std::unique_ptr<Handle> CreateActivityHandle(StringPiece, StringPiece,
                                                       bool) const {
    // We don't do anything with 'Activities' yet.
    return nullptr;
  }

  bool IsEnabledForAnnotations() const override {
    return active_trace_session_.load(std::memory_order_relaxed);
  }

  bool IsEnabledForActivities(bool is_expensive) const override {
    // We don't do anything with 'Activities' so we are never 'enabled'.
    return false;
  }

  void Start() {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
    active_trace_session_ = true;
  }

  void Stop() {
    DCHECK(active_trace_session_) << "No active trace session detected. ";
    active_trace_session_ = false;
  }

 private:
  std::atomic<bool> active_trace_session_;
};


TraceCollectorImpl *GlobalDefaultTraceCollector() {
  static auto *instance = new TraceCollectorImpl();
  return instance;
}


class DeviceTracerImpl : public DeviceTracer {
 public:

//  google::protobuf::Arena cupti_protobuf_arena;
  DeviceTracerImpl(CUPTIManager *cupti_manager);
  ~DeviceTracerImpl() override;

  // DeviceTracer interface:
  Status Start() override;
  Status Stop() override;
  Status Print() override;
//  Status Collect() override;
  Status SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) override;
  Status AsyncDump() override;
  Status AwaitDump() override;
  Status RecordEvent(
      const char* category,
      int64 start_us,
      int64 duration_us,
      const char* name) override;

  Status PushOperation(const char* operation) override;
  Status RecordOverheadEvent(
      const char* overhead_type,
      int64 num_events) override;
  Status RecordOverheadEventForOperation(
      const char* overhead_type,
      const char* operation,
      int64 num_events) override;
  Status PopOperation() override;

  void _Register_LD_PRELOAD_Callbacks();
  void _RegisterCUDAAPICallbacks();
  void _EnableSomeCUDAAPICallbacks();
  void _EnableAllCUDAAPICallbacks();

#ifdef CONFIG_TRACE_STATS
  bool IsEnabled() override;
#endif

 protected:
  // This callback is used exclusively by CUPTIManager.
  friend class CUPTIManager;
//  void ActivityCallback(const CUpti_Activity &activity) override;
//  void ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) override;

  // CudaStreamMonitor callback.
  void PollStreamsCallback(const std::vector<PollStreamResult>& poll_stream_results);

 private:
//  // Internal struct to record kernel launches.
//  struct KernelRecord {
//    KernelRecord(
//        uint64_t start_timestamp_,
//        uint64_t end_timestamp_,
//        uint32 device_id_,
//        uint32 stream_id_,
//        uint32 correlation_id_) :
//        start_timestamp(start_timestamp_)
//        , end_timestamp(end_timestamp_)
//        , device_id(device_id_)
//        , stream_id(stream_id_)
//        , correlation_id(correlation_id_)
//    {
//    }
//    uint64_t start_timestamp;
//    uint64_t end_timestamp;
//    uint32 device_id;
//    uint32 stream_id;
//    uint32 correlation_id;
//  };
//  // Internal struct to record memcpy operations.
//  struct MemcpyRecord {
//    MemcpyRecord(
//        uint64_t start_timestamp_,
//        uint64_t end_timestamp_,
//        uint32 device_id_,
//        uint32 stream_id_,
//        uint32 correlation_id_,
//        uint8 copyKind_,
//        uint8 srcKind_,
//        uint8 dstKind_,
//        uint64 bytes_) :
//        start_timestamp(start_timestamp_)
//        , end_timestamp(end_timestamp_)
//        , device_id(device_id_)
//        , stream_id(stream_id_)
//        , correlation_id(correlation_id_)
//        , copyKind(copyKind_)
//        , srcKind(srcKind_)
//        , dstKind(dstKind_)
//        , bytes(bytes_)
//    {
//    }
//    uint64_t start_timestamp;
//    uint64_t end_timestamp;
//    uint32 device_id;
//    uint32 stream_id;
//    uint32 correlation_id;
//    uint8 copyKind;
//    uint8 srcKind;
//    uint8 dstKind;
//    uint64 bytes;
//  };

  // This is the subscriber callback which is invoked directly by CUPTI.
  // The 'userdata' argument will be a pointer to the active 'DeviceTracerImpl'.
  static void CUPTIAPI __ApiCallback(
      void *userdata, CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid, const void *cbdata);
//  void _ApiCallback(
//      CUpti_CallbackDomain domain,
//      CUpti_CallbackId cbid, const void *cbdata);

  // Records the mapping between correlation ID and kernel name.
//  void AddCorrelationId(uint32 correlation_id, const string &name);

  // Returns the current system time in microseconds.
  inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

  std::vector<std::unique_ptr<ActivityBuffer>> activity_buffers_;
  OpStack _op_stack;
  CUDAAPIProfiler _api_profiler;
  CUDAAPIProfilerPrinter api_printer_;
  CUDAActivityProfiler _activity_profiler;
  std::shared_ptr<CudaStreamMonitor> stream_monitor_;
  EventProfiler _event_profiler;
  RegisteredFunc::FuncId _cuda_stream_monitor_cbid;
  std::shared_ptr<CuptiAPI> _cupti_api;
  RegisteredHandle<CuptiCallback::FuncId> _cupti_cb_handle;

#ifdef WITH_CUDA_LD_PRELOAD
  std::vector<RegisteredHandleInterface> _cuda_api_api_profiler_cbs;
//  CudaAPIRegisteredFuncHandle<cudaLaunchKernel_callback> _cuda_api_cudaLaunchKernel_api_profiler_cb;
#endif

  CUPTIManager *cupti_manager_;
//  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;
  CUpti_SubscriberHandle subscriber_;

  mutex trace_mu_;
  static constexpr size_t kMaxRecords = 1024 * 1024;

//  std::map<uint32, string> correlations_ GUARDED_BY(trace_mu_);
//  std::vector<KernelRecord> kernel_records_ GUARDED_BY(trace_mu_);
//  std::vector<MemcpyRecord> memcpy_records_ GUARDED_BY(trace_mu_);

  mutex mu_;
  bool enabled_ GUARDED_BY(mu_);
//  int64 start_walltime_us_ GUARDED_BY(mu_);
//  int64 end_walltime_us_ GUARDED_BY(mu_);
//  uint64_t start_timestamp_ GUARDED_BY(mu_);
//  uint64_t end_timestamp_ GUARDED_BY(mu_);

  std::string _directory GUARDED_BY(mu_);
  std::string _process_name GUARDED_BY(mu_);
  std::string _machine_name GUARDED_BY(mu_);
  std::string _phase_name GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceTracerImpl);
};


DeviceTracerImpl::DeviceTracerImpl(CUPTIManager *cupti_manager)
    :
        _api_profiler(_op_stack),
        api_printer_(_api_profiler, get_TF_CUDA_API_PRINT_EVERY_SEC(0)),
        _activity_profiler(cupti_manager),
        _cuda_stream_monitor_cbid(-1),
        _cupti_api(CuptiAPI::GetCuptiAPI()),
//        _cupti_api_device_tracer_callback_id(-1),
        cupti_manager_(cupti_manager)
    {
  VLOG(1) << "DeviceTracer created.";
//  cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
  if (is_yes("TF_CUPTI_PROTOBUF_ARENA", false)) {
//    kernel_records_.reserve(kMaxRecords);
//    memcpy_records_.reserve(kMaxRecords);
//    cupti_protobuf_arena = std::move(google::protobuf::Arena(google::protobuf::ArenaOptions{
//    }))
  }
  enabled_ = false;
  if (is_yes("IML_STREAM_SAMPLING", false)) {
    // --stream-sampling
    VLOG(0) << "Enabling CUDA stream sampling";
    stream_monitor_ = CudaStreamMonitor::GetCudaStreamMonitor();
    _cuda_stream_monitor_cbid = stream_monitor_->RegisterPollStreamsCallback([this] (const std::vector<PollStreamResult>& poll_stream_results) {
      this->PollStreamsCallback(poll_stream_results);
    });
  }
}
void DeviceTracerImpl::PollStreamsCallback(const std::vector<PollStreamResult>& poll_stream_results) {
  // Do nothing for now...
  // TODO: record state inside of sample-state.
  // TODO: if any results have is_valid unset, discard them (during driver shutdown)
//  if (VLOG_IS_ON(1)) {
//    std::stringstream ss("PollStreamsCallback: ");
//    ss << "size = " << poll_stream_results.size();
//    int i = 0;
//    for (auto const& result : poll_stream_results) {
//      ss << "\n  [" << i << "]: is_active = " << result.is_active;
//      i += 1;
//    }
//    VLOG(INFO) << ss.str();
//  }
}

DeviceTracerImpl::~DeviceTracerImpl() {
  // Unregister the CUPTI callbacks if needed to prevent them from accessing
  // freed memory.
  Stop().IgnoreError();
  if (stream_monitor_) {
    DCHECK(_cuda_stream_monitor_cbid != -1);
    stream_monitor_->UnregisterCallback(_cuda_stream_monitor_cbid);
  }
}

void DeviceTracerImpl::_Register_LD_PRELOAD_Callbacks() {
#ifdef WITH_CUDA_LD_PRELOAD
  VLOG(1) << "Register LD_PRELOAD CUDA API callbacks";

#define REGISTER_CUDA_API_CB(domain, cbid, funcname, RetType, ...) \
  _cuda_api_api_profiler_cbs.emplace_back( \
      GetCudaLibrary()->_cuda_api.funcname ## _cbs.RegisterCallback( \
          /*start_cb=*/[this](__VA_ARGS__) { \
            VLOG(1) << "CUDA_API_INTERCEPT.ENTER: " << #funcname; \
            this->_api_profiler.ApiCallback( \
                domain, \
                cbid, \
                CUPTI_API_ENTER); \
          }, \
          /*exit_cb=*/[this](__VA_ARGS__, RetType ret) { \
            VLOG(1) << "CUDA_API_INTERCEPT.EXIT: " << #funcname; \
            this->_api_profiler.ApiCallback( \
                domain, \
                cbid, \
                CUPTI_API_EXIT); \
          }));

  REGISTER_CUDA_API_CB(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000,
      cudaLaunchKernel,
      cudaError_t,
      const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);

  REGISTER_CUDA_API_CB(
      CUPTI_CB_DOMAIN_DRIVER_API,
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
      cuLaunchKernel,
      CUresult,
      CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

  REGISTER_CUDA_API_CB(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000,
      cudaLaunchKernel_ptsz,
      cudaError_t,
      const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);

  REGISTER_CUDA_API_CB(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020,
      cudaMemcpyAsync,
      cudaError_t,
      void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

  REGISTER_CUDA_API_CB(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020,
      cudaMalloc,
      cudaError_t,
      void **devPtr, size_t size);

  REGISTER_CUDA_API_CB(
      CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020,
      cudaFree,
      cudaError_t,
      void *devPtr);

#else
  DCHECK(false) << "LD_PRELOAD callbacks cannot be enabled unless librlscope.so is compiled with WITH_CUDA_LD_PRELOAD=ON in CMakeLists.txt";
#endif // WITH_CUDA_LD_PRELOAD
}

void DeviceTracerImpl::_RegisterCUDAAPICallbacks() {
  _cupti_cb_handle = _cupti_api->RegisterCallback(
      [this] (CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
        if (
            domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API
            ) {
          auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
          auto callback_site = cbInfo->callbackSite;
          this->_api_profiler.ApiCallback(domain, cbid, callback_site);
        }
      });
}

void DeviceTracerImpl::_EnableSomeCUDAAPICallbacks() {

  CUPTI_CALL(_cupti_api->EnableCallback(
      /*enable=*/1,
      // subscriber_,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

  // cudaMemcpy isn't asynchronous, so time spent in the API call will vary depending on the size of the copy.
  // Q: Not sure why, but we still end up tracing cudaMemcpy calls even though we said to trace only cudaMemcpyAsync.
//  CUPTI_CALL(_cupti_api->EnableCallback(
//      /*enable=*/1,
//      // subscriber_,
//                 CUPTI_CB_DOMAIN_RUNTIME_API,
//                 CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020));

  CUPTI_CALL(_cupti_api->EnableCallback(
      /*enable=*/1,
      // subscriber_,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));
  CUPTI_CALL(_cupti_api->EnableCallback(
      /*enable=*/1,
      // subscriber_,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020));
  CUPTI_CALL(_cupti_api->EnableCallback(
      /*enable=*/1,
      // subscriber_,
                 CUPTI_CB_DOMAIN_RUNTIME_API,
                 CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020));

}

void DeviceTracerImpl::_EnableAllCUDAAPICallbacks() {
  // Q: Should we also trace driver API?  What's the difference between the driver API and runtime API?
  // _cupti_api->EnableAllDomains(/*enable=*/1);
  _cupti_api->EnableDomain(/*enable=*/1, CUPTI_CB_DOMAIN_RUNTIME_API);
}

Status DeviceTracerImpl::Start() {
  VLOG(1) << "DeviceTracer::Start";
  mutex_lock l(mu_);
  if (VLOG_IS_ON(1)) {
    api_printer_.Start();
  }
  if (stream_monitor_) {
    stream_monitor_->Start();
  }
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is already enabled.");
  }
  if (!is_yes("TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS", false)) {
    VLOG(1) << "TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS is not set";

    if (!is_yes("TF_CUPTI_SKIP_REGISTER_API_CALLBACKS", false)) {
      VLOG(1) << "TF_CUPTI_SKIP_REGISTER_API_CALLBACKS is not set";
      // There can only be one CUPTI subscriber.  If we can't create one then
      // there is another trace in progress (possibly by external code).
      CUptiResult ret;
//      ret = cupti_wrapper_->Subscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);

// IML NOTE: disable CUDA API callbacks for giving labels to kernel execution times.
// Not needed for IML metrics, and adds runtime overhead.
//      if (!is_yes("IML_DISABLE", false)) {
//        VLOG(1) << "Register DeviceTracer CUDA API calls callback";
//        _cupti_api_device_tracer_callback_id = _cupti_api->RegisterCallback(
//            [this](CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
//              this->_ApiCallback(domain, cbid, cbdata);
//            });
//      }

      DCHECK(!(
          is_yes("IML_FUZZ_CUDA_API", false) &&
          is_yes("IML_CUDA_API_CALLS", false) ))
        << "Can only run iml-prof with --fuzz-cuda-api or --cuda-api-calls, not both";

      if (is_yes("IML_CUDA_API_CALLS", false)) {
#ifdef WITH_CUDA_LD_PRELOAD
        _Register_LD_PRELOAD_Callbacks();
#else
        _RegisterCUDAAPICallbacks();
        _EnableSomeCUDAAPICallbacks();
#endif
      }

      if (is_yes("IML_FUZZ_CUDA_API", false)) {
        this->_api_profiler.EnableFuzzing();
        _RegisterCUDAAPICallbacks();
        _EnableAllCUDAAPICallbacks();
      }

      if (is_yes("IML_CUDA_API_EVENTS", false)) {
        // Record raw start/end timestamps of CUDA API calls.
        // Required during instrumented runs.
        VLOG(1) << "CUDAAPIProfiler: enable event recording (IML_CUDA_API_EVENTS=yes)";
        this->_api_profiler.EnableEventRecording();
      }

//      VLOG(1) << "cuptiSubscribe";
//      ret = cuptiSubscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
//      if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
//        VLOG(1) << "Fail 1";
//        return errors::Unavailable("CUPTI subcriber limit reached.");
//      } else if (ret != CUPTI_SUCCESS) {
//        VLOG(1) << "Fail 2";
//        const char *errstr;
//        cuptiGetResultString(ret, &errstr);
//        return errors::Internal("Failed to create CUPTI subcriber: ", errstr);
//      }

      // Register as a TraceEngine to receive ScopedAnnotations.
      GlobalDefaultTraceCollector()->Start();

      // Intercept launch and memcpy calls to capture the Op name annotation.
      // TODO(pbar) Add callbacks for memcpy variants.
//      if (!is_yes("IML_DISABLE", false)) {
//        // NOTE: runtime API callbacks cause significant slow-downs.
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
////      if (!is_yes("IML_DISABLE", false)) {
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_RUNTIME_API,
//                       CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020));
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_RUNTIME_API,
//                       CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));
//
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2));
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2));
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2));
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2));
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2));
//        CUPTI_CALL(_cupti_api->EnableCallback(
//            /*enable=*/1,
//            // subscriber_,
//                       CUPTI_CB_DOMAIN_DRIVER_API,
//                       CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2));
//      }
//      }
    }

//    VLOG(1) << "Run EnableTrace";
//    TF_RETURN_IF_ERROR(cupti_manager_->EnableTrace(this));

//    CUPTI_CALL(cuptiGetTimestamp(&start_timestamp_));
  }

//  start_walltime_us_ = NowInUsec();

  if (is_yes("IML_CUDA_ACTIVITIES", false)) {
    VLOG(1) << "Start CUDAActivityProfiler";
    _activity_profiler.Start();
  }

  enabled_ = true;
  return Status::OK();
}

Status DeviceTracerImpl::Print() {
  mutex_lock l(mu_);
  DECLARE_LOG_INFO(info);
  PrintIndent(info, 0);
  info << "DeviceTracerImpl:\n";
  _op_stack.Print(info, 1);
  info << "\n";
  _api_profiler.Print(info, 1);
  if (is_yes("IML_CUDA_ACTIVITIES", false)) {
    info << "\n";
    _activity_profiler.Print(info, 1);
  }
  if (stream_monitor_) {
    info << "\n";
    stream_monitor_->Print(info, 1);
  }
  // EventProfiler output can get REALLY big...
  // internally we limit printing to 15 events (per category).
  info << "\n";
  _event_profiler.Print(info, 1);
  return Status::OK();
}


Status DeviceTracerImpl::Stop() {
  VLOG(1) << "DeviceTracer::Stop";
  mutex_lock l(mu_);
  // VLOG(1) << "DeviceTracerImpl." << __func__ << ": api_printer.Stop()";
  SimpleTimer timer("DeviceTracerImpl.Stop");
  timer.ResetStartTime();
  api_printer_.Stop();
  timer.EndOperation("api_printer_.Stop");
  if (is_yes("IML_CUDA_ACTIVITIES", false)) {
    // VLOG(1) << "DeviceTracerImpl." << __func__ << ": activity_profiler.Stop()";
    _activity_profiler.Stop();
    timer.EndOperation("_activity_profiler.Stop");
    // VLOG(1) << "DeviceTracerImpl." << __func__ << ": activity_profiler.Stop() done";
  }
  if (stream_monitor_) {
    // VLOG(1) << "DeviceTracerImpl." << __func__ << ": stream_monitor.Stop()";
    stream_monitor_->Stop();
    timer.EndOperation("stream_monitor_.Stop");
    // VLOG(1) << "DeviceTracerImpl." << __func__ << ": stream_monitor.Stop() done";
  }
  if (!enabled_) {
    return Status::OK();
  }
#ifdef CONFIG_TRACE_STATS
  // VLOG(1) << "DeviceTracerImpl." << __func__ << ": GlobalDefaultTraceCollector.Stop()";
  GlobalDefaultTraceCollector()->Stop();
  timer.EndOperation("GlobalDefaultTraceCollector()->Stop()");
  // VLOG(1) << "DeviceTracerImpl." << __func__ << ": GlobalDefaultTraceCollector.Stop() done";

//  TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace());
//  end_walltime_us_ = NowInUsec();
//  CUPTI_CALL(cuptiGetTimestamp(&end_timestamp_));
#else
  CUPTI_CALL(cuptiUnsubscribe(subscriber_));
  GlobalDefaultTraceCollector()->Stop();

  TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace());
  end_walltime_us_ = NowInUsec();
  CUPTI_CALL(cuptiGetTimestamp(&end_timestamp_));
#endif // CONFIG_TRACE_STATS
  enabled_ = false;
#ifdef DEBUG_CRITICAL_PATH
  {
    DECLARE_LOG_INFO(info);
    timer.Print(info, 0);
  }
#endif // DEBUG_CRITICAL_PATH
  VLOG(1) << "DeviceTracerImpl." << __func__ << ": done";
  return Status::OK();
}


///*static*/ void DeviceTracerImpl::__ApiCallback(void *userdata,
//                                                CUpti_CallbackDomain domain,
//                                                CUpti_CallbackId cbid,
//                                                const void *cbdata) {
//  auto *self = reinterpret_cast<DeviceTracerImpl *>(userdata);
//  self->_ApiCallback(domain, cbid, cbdata);
//}

// CorrelationID is used for associating information gathered during a CUDA API runtime callback
// with information gathered from a GPU-activity-record.
// This functionality is used by TensorFlow to label each GPU activity record with a string-identifier
// (an annotation in the TensorFlow source-code at the site of the cudaLaunch call).
//
// IML: since we create our own CUDA API wrappers separate from libcupti, we DON'T use this.
//void DeviceTracerImpl::_ApiCallback(
//    CUpti_CallbackDomain domain,
//    CUpti_CallbackId cbid,
//    const void *cbdata) {
//  auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
//  if (!is_yes("TF_CUPTI_EMPTY_TRACING_CALLBACKS", false)) {
//      VLOG(2) << "ApiCallback " << domain << ":" << cbid
//          << " func: " << cbInfo->functionName;
//
//      // API callbacks are invoked synchronously on the thread making the
//      // CUDA API call.  If this pointer is non-null then the ScopedAnnotation
//      // must be valid.
//      const char *tls_annotation = tls_current_annotation.get();
//
//      if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
//              (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
//          if (cbInfo->callbackSite == CUPTI_API_ENTER) {
//              auto *params = reinterpret_cast<const cuLaunchKernel_params *>(
//                      cbInfo->functionParams);
//              if (VLOG_IS_ON(2)) {
//                  VLOG(2) << "LAUNCH stream " << params->hStream << " correllation "
//                      << cbInfo->correlationId << " kernel " << cbInfo->symbolName;
//              }
//              const string annotation =
//                  tls_annotation ? tls_annotation : cbInfo->symbolName;
//              this->AddCorrelationId(cbInfo->correlationId, annotation);
//          }
//      } else if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API) &&
//              (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
//               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) {
//          if (cbInfo->callbackSite == CUPTI_API_ENTER) {
//              if (VLOG_IS_ON(2)) {
//                  auto *funcParams = reinterpret_cast<const cudaMemcpy_v3020_params *>(
//                          cbInfo->functionParams);
//                  size_t count = funcParams->count;
//                  enum cudaMemcpyKind kind = funcParams->kind;
//                  VLOG(2) << "MEMCPY count " << count << " kind " << kind;
//              }
//              if (tls_annotation) {
//                  const string annotation = tls_annotation;
//                  this->AddCorrelationId(cbInfo->correlationId, annotation);
//              }
//          }
//      } else if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
//              (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
//               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
//               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 ||
//               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 ||
//               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 ||
//               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2)) {
//          if (cbInfo->callbackSite == CUPTI_API_EXIT && tls_annotation) {
//              const string annotation = tls_annotation;
//              this->AddCorrelationId(cbInfo->correlationId, annotation);
//          }
//      } else {
//          VLOG(1) << "Unhandled API Callback for " << domain << " " << cbid;
//      }
//  }
//}



#ifdef CONFIG_TRACE_STATS
bool DeviceTracerImpl::IsEnabled() {
  mutex_lock l(mu_);
  return enabled_;
}
#endif // CONFIG_TRACE_STATS

Status DeviceTracerImpl::SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) {
  mutex_lock l(mu_);
  _op_stack.SetMetadata(directory, process_name, machine_name, phase_name);
  _activity_profiler.SetMetadata(directory, process_name, machine_name, phase_name);
  _directory = directory;
  _process_name = process_name;
  _machine_name = machine_name;
  _phase_name = phase_name;
  _api_profiler.SetMetadata(directory, process_name, machine_name, phase_name);
  _event_profiler.SetMetadata(directory, process_name, machine_name, phase_name);
  return Status::OK();
}

Status DeviceTracerImpl::AsyncDump() {
  mutex_lock l(mu_);
  _op_stack.AsyncDump();
  if (is_yes("IML_CUDA_ACTIVITIES", false)) {
    _activity_profiler.AsyncDump();
  }
  _api_profiler.AsyncDump();
  _event_profiler.AsyncDump();
  return Status::OK();
}
Status DeviceTracerImpl::AwaitDump() {
  // Q: Do we need to grab this...?
  mutex_lock l(mu_);
  _op_stack.AwaitDump();
  if (is_yes("IML_CUDA_ACTIVITIES", false)) {
    _activity_profiler.AwaitDump();
  }
  _api_profiler.AwaitDump();
  _event_profiler.AwaitDump();
  return Status::OK();
}

Status DeviceTracerImpl::RecordEvent(
    const char* category,
    int64 start_us,
    int64 duration_us,
    const char* name) {
  // NOTE: don't grab the lock, don't need to.
  // mutex_lock l(mu_);
  _event_profiler.RecordEvent(
      category,
      start_us,
      duration_us,
      name);

  return Status::OK();
}

Status DeviceTracerImpl::PushOperation(const char* operation) {
  _op_stack.PushOperation(operation);
  return Status::OK();
}
Status DeviceTracerImpl::RecordOverheadEvent(
    const char* overhead_type,
    int64 num_events) {
  _op_stack.RecordOverheadEvent(
      overhead_type,
      num_events);
  return Status::OK();
}
Status DeviceTracerImpl::RecordOverheadEventForOperation(
    const char* overhead_type,
    const char* operation,
    int64 num_events) {
  _op_stack.RecordOverheadEventForOperation(
      overhead_type,
      operation,
      num_events);
  return Status::OK();
}
Status DeviceTracerImpl::PopOperation() {
  _op_stack.PopOperation();
  return Status::OK();
}

//Status DeviceTracerImpl::Collect() {
//  mutex_lock l(mu_);
//  if (enabled_) {
//    return errors::FailedPrecondition("DeviceTracer is still enabled.");
//  }
//
//  // Collect is called from an async thread; now it's safe to take events out of the libcupti activity buffer without
//  // blocking the main-thread.
//  for (auto& activity_buffer : activity_buffers_) {
//    activity_buffer->RecordActivitiesFromBuffer();
//  }
//  activity_buffers_.clear();
//
//  bool use_arena = is_yes("TF_CUPTI_PROTOBUF_ARENA", false);
//
//  // TODO(pbar) Handle device IDs and prefix properly.
//  const string prefix = "";
//  const int id = 0;
//  const string stream_device =
//      strings::StrCat(prefix, "/device:GPU:", id, "/stream:");
//  const string memcpy_device =
//      strings::StrCat(prefix, "/device:GPU:", id, "/memcpy");
//
//  mutex_lock l2(trace_mu_);
//
//  // TODO: pass kernel_records and memcpy_records to a class that handles all this stuff OFF the critical path.
//
//  // SKIP until we start dumping to our own protobuf.
//  for (const auto &rec : kernel_records_) {
//    auto it = correlations_.find(rec.correlation_id);
//    const string name = (it != correlations_.cend()) ? it->second : "unknown";
//    NodeExecStats *ns = new NodeExecStats;
//    ns->set_all_start_micros(start_walltime_us_ +
//                             ((rec.start_timestamp - start_timestamp_) / 1000));
//    ns->set_op_start_rel_micros(0);
//    auto elapsed_us =
//        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
//    ns->set_op_end_rel_micros(elapsed_us);
//    ns->set_all_end_rel_micros(elapsed_us);
//    ns->set_node_name(name);
//    // TODO(pbar) Generate details based on the kernel activity record.
//    // ns->set_timeline_label(details);
//    auto nscopy = new NodeExecStats;
//    *nscopy = *ns;
//    collector->Save(strings::StrCat(stream_device, "all"), ns);
//    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
//  }
//  for (const auto &rec : memcpy_records_) {
//    auto it = correlations_.find(rec.correlation_id);
//    const string name = (it != correlations_.cend()) ? it->second : "unknown";
//    NodeExecStats *ns = new NodeExecStats;
//    ns->set_all_start_micros(start_walltime_us_ +
//                             ((rec.start_timestamp - start_timestamp_) / 1000));
//    ns->set_op_start_rel_micros(0);
//    auto elapsed_us =
//        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
//    ns->set_op_end_rel_micros(elapsed_us);
//    ns->set_all_end_rel_micros(elapsed_us);
//    auto copyKind = static_cast<CUpti_ActivityMemcpyKind>(rec.copyKind);
//    auto srcKind = static_cast<CUpti_ActivityMemoryKind>(rec.srcKind);
//    auto dstKind = static_cast<CUpti_ActivityMemoryKind>(rec.dstKind);
//    const string details = strings::Printf(
//        "MEMCPY%s %llu bytes (%s to %s)", getMemcpyKindString(copyKind),
//        rec.bytes, getMemoryKindString(srcKind), getMemoryKindString(dstKind));
//    ns->set_node_name(
//        strings::StrCat(name, ":MEMCPY", getMemcpyKindString(copyKind)));
//    ns->set_timeline_label(details);
//    auto nscopy = new NodeExecStats;
//    *nscopy = *ns;
//    collector->Save(memcpy_device, ns);
//    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
//  }
//
//  return Status::OK();
//}

}  // namespace devicetracer

std::unique_ptr<DeviceTracer> CreateDeviceTracer() {
  CUPTIManager *cupti_manager = GetCUPTIManager();
  if (cupti_manager == nullptr) {
    return nullptr;
  }
  std::unique_ptr<DeviceTracer> tracer(
      new devicetracer::DeviceTracerImpl(cupti_manager));
  return tracer;
}




}  // namespace rlscope

