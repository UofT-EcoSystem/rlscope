//
// Created by jagle on 8/16/2019.
//

#ifndef IML_CUDA_STREAM_MONITOR_H
#define IML_CUDA_STREAM_MONITOR_H

#include <cuda.h>
#include <cupti.h>

#include <list>
#include <vector>
#include <thread>
#include <memory>
#include <functional>
#include <mutex>

#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/cupti_api_wrapper.h"

#include "common_util.h"

namespace rlscope {

struct PollStreamResult {
  PollStreamResult(cudaStream_t stream, bool is_active, bool is_valid) :
      stream(stream),
      is_active(is_active),
      is_valid(is_valid) {
  }

  cudaStream_t stream;
  bool is_active;
  bool is_valid;

  std::ostream& Print(std::ostream& out, int indent);
};

struct PollStreamSummary {
  PollStreamSummary(cudaStream_t stream) :
      stream(stream),
      num_samples_is_active(0),
      num_samples_is_inactive(0)
  {
  }
  cudaStream_t stream;
  uint64_t num_samples_is_active;
  uint64_t num_samples_is_inactive;

  std::ostream& Print(std::ostream& out, int indent);

  void AddPollStreamResult(const PollStreamResult& poll_stream);
};

class CudaStreamMonitor {
public:
  using PollStreamsCallback = std::function<void(const std::vector<PollStreamResult>&)>;
  std::list<cudaStream_t> _active_streams;
  std::vector<PollStreamsCallback> _callbacks;
  std::vector<PollStreamSummary> _poll_stream_summaries;
  std::mutex mu_;
  EventHandler _event_handler;
  Notification _should_stop;
  std::unique_ptr<std::thread> _polling_thread;
  CUpti_SubscriberHandle subscriber_;
  std::vector<CUpti_runtime_api_trace_cbid> _callback_ids;
  float _sample_every_sec;
  std::shared_ptr<CuptiAPI> _cupti_api;
  RegisteredHandle<CuptiCallback::FuncId> _cupti_api_callback_id;

  // Singleton.
  static std::shared_ptr<CudaStreamMonitor> GetCudaStreamMonitor();


  // Streams get added during a call to cudaStreamCreate.
  void AddStream(cudaStream_t stream);
  // Streams get removed during a call to cudaStreamDestroy.
  void RemoveStream(cudaStream_t stream);
  std::vector<PollStreamResult> PollStreams();
  int RegisterPollStreamsCallback(
      PollStreamsCallback callback);
  void UnregisterCallback(RegisteredFunc::FuncId func_id);


  template <class CudaStreamParams>
  void _HandleRemoveStream(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata);
  template <class CudaStreamParams>
  void _HandleAddStream(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata);

//  static void CUPTIAPI _ApiCallback(void *userdata, CUpti_CallbackDomain domain,
//                                    CUpti_CallbackId cbid, const void *cbdata);
  void ApiCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata);

  void _RunPollingThread();
  void Start();
  void Stop();
  std::ostream& Print(std::ostream& out, int indent);
  CudaStreamMonitor();
  ~CudaStreamMonitor();

  void _RegisterCUPTICallbacks();
};

}

#endif //IML_CUDA_STREAM_MONITOR_H
