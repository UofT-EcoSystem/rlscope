//
// Created by jagle on 8/16/2019.
//

#include "cuda_api_profiler/cuda_stream_monitor.h"
#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/get_env_var.h"
#include "cuda_api_profiler/cupti_logging.h"
#include "cuda_api_profiler/cupti_api_wrapper.h"

#include "common_util.h"

#include <cuda.h>
#include <cupti_target.h>
#include <cupti.h>

#include <memory>
#include <thread>
#include <mutex>

namespace rlscope {

// Sample every millisecond.
// There are a lot of tiny kernels that execute... (look at q-forward trace from powerpoint)... so I suspect it's unlikely
// that we will obtain super accurate results...it's highly dependent on our sampling frequency.
#define SAMPLE_STREAM_EVERY_SEC (1/((float)1e3))

CudaStreamMonitor::CudaStreamMonitor() :
    _callback_ids({
                      CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020,
                      CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000,
                      CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050,
                      CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020,
                      CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050
                  }),
    _sample_every_sec(SAMPLE_STREAM_EVERY_SEC),
//    _sample_every_sec(get_IML_SAMPLE_EVERY_SEC(0)),
    _cupti_api(CuptiAPI::GetCuptiAPI())
{
  _RegisterCUPTICallbacks();
  // Add default stream; there's no call to cudaStreamCreate or cudaStreamDestroy for the default stream.
  cudaStream_t default_stream = 0;
  AddStream(default_stream);
}
RegisteredFunc::FuncId CudaStreamMonitor::RegisterPollStreamsCallback(
    CudaStreamMonitor::PollStreamsCallback callback)
{
  return _event_handler.RegisterFunc([this, callback]() {
    auto poll_stream_results = this->PollStreams();
    callback(poll_stream_results);
  }, _sample_every_sec);
}

void CudaStreamMonitor::UnregisterCallback(
    RegisteredFunc::FuncId func_id)
{
  _event_handler.UnregisterFunc(func_id);
}

void CudaStreamMonitor::_RunPollingThread() {
  _event_handler.EventLoop([this]() {
    return this->_should_stop.HasBeenNotified();
  });
}

CudaStreamMonitor::~CudaStreamMonitor() {
  Stop();
//  if (VLOG_IS_ON(1)) {
//    DECLARE_LOG_INFO(info);
//    this->Print(info, 0);
//  }
}

// NOTE: If anyone calls GetCudaStreamMonitor, CUPTI callbacks will immediately be registered.
// Basically, if we want to run with --iml-disable, we need to be careful not to call this.
// PROBLEM: it's entirely possible TensorFlow internally creates Stream's during import before
// python has a chance to parse --iml-disable.
// SOLUTION: --iml-disable runs simply shouldn't run with rls-prof.
static std::shared_ptr<CudaStreamMonitor> _cuda_stream_monitor;
/* static */ std::shared_ptr<CudaStreamMonitor> CudaStreamMonitor::GetCudaStreamMonitor() {
  if (!_cuda_stream_monitor) {
    _cuda_stream_monitor.reset(new CudaStreamMonitor());
  }
  return _cuda_stream_monitor;
}

///* static */ void CUPTIAPI CudaStreamMonitor::ApiCallback(void *userdata, CUpti_CallbackDomain domain,
//                                                           CUpti_CallbackId cbid, const void *cbdata) {
//  CudaStreamMonitor *self = reinterpret_cast<CudaStreamMonitor*>(userdata);
//  return self->ApiCallback(domain, cbid, cbdata);
//}
void CudaStreamMonitor::ApiCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
//  const char* cb_string = "";
//  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
//    cb_string = runtime_cbid_to_string(cbid);
//  } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
//    cb_string = driver_cbid_to_string(cbid);
//  }
//  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
//    VLOG(1) << "CudaStreamMonitor::ApiCallback, domain=" << domain << ", cbid=" << cb_string << " (" << cbid << ")";
//  }

  switch (cbid) {

    // Handle cudaStreamCreate:
    // - At the END of cudaStreamCreate/cudaStreamCreateWithFlags/cudaStreamCreateWithPriority, add stream to CudaStreamMonitor.
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
      _HandleAddStream<cudaStreamCreate_v3020_params>(domain, cbid, cbdata);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000:
      _HandleAddStream<cudaStreamCreateWithFlags_v5000_params>(domain, cbid, cbdata);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050:
      _HandleAddStream<cudaStreamCreateWithPriority_v5050_params>(domain, cbid, cbdata);
      break;

    // Handle cudaStreamDestroy:
    // - At the START of cudaStreamDestroy, remove stream from CudaStreamMonitor.
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050:
      _HandleRemoveStream<cudaStreamDestroy_v5050_params>(domain, cbid, cbdata);
      break;

    default:
      break;

  }

//  if (0) {
//
//  // Handle cudaStreamCreate:
//  // - At the END of cudaStreamCreate/cudaStreamCreateWithFlags/cudaStreamCreateWithPriority, add stream to CudaStreamMonitor.
//  } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020) {
//    _HandleAddStream<cudaStreamCreate_v3020_params>(domain, cbid, cbdata);
//  } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000) {
//    _HandleAddStream<cudaStreamCreateWithFlags_v5000_params>(domain, cbid, cbdata);
//  } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050) {
//    _HandleAddStream<cudaStreamCreateWithPriority_v5050_params>(domain, cbid, cbdata);
//
//  // Handle cudaStreamDestroy:
//  // - At the START of cudaStreamDestroy, remove stream from CudaStreamMonitor.
//
//  // cudaStreamDestroy_v3020_params is MISSING from CUDA 10.1.
//  // } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020) {
//  //  _HandleRemoveStream<cudaStreamDestroy_v3020_params>(domain, cbid, cbdata);
//  } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050) {
//    _HandleRemoveStream<cudaStreamDestroy_v5050_params>(domain, cbid, cbdata);
//
//  }

}

template <class CudaStreamParams>
void CudaStreamMonitor::_HandleRemoveStream(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
  // - At the START of cudaStreamDestroy, remove stream from CudaStreamMonitor.
  auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    const auto params = ((CudaStreamParams *) (cbInfo->functionParams));
    cudaStream_t stream = params->stream;
    this->RemoveStream(stream);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    DCHECK(false) << "Saw cudaStreamCreate.CUPTI_API_EXIT, but we shouldn't be registered to receive those.";
  } else {
    DCHECK(false) << "Shouldn't reach here";
  }
}

template <class CudaStreamParams>
void CudaStreamMonitor::_HandleAddStream(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
  // - At the END of cudaStreamCreate/cudaStreamCreateWithFlags/cudaStreamCreateWithPriority, add stream to CudaStreamMonitor.
  auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
  if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    const auto params = ((CudaStreamParams *)(cbInfo->functionParams));
    cudaStream_t stream = *(params->pStream);
    this->AddStream(stream);
  } else if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    DCHECK(false) << "Saw cudaStreamCreate.CUPTI_API_ENTER, but we shouldn't be registered to receive those.";
  } else {
    DCHECK(false) << "Shouldn't reach here";
  }
}

void CudaStreamMonitor::_RegisterCUPTICallbacks() {
//  CUpti_SubscriberHandle subscriber;
  CUresult cuerr;
  CUptiResult cuptierr;

  // NOTE This fails; CUPTI only allows a single subscriber handle.
//  CUpti_SubscriberHandle sub1;
//  CUpti_SubscriberHandle sub2;
//  cuptierr = cuptiSubscribe(&sub1, CudaStreamMonitor::_ApiCallback, this);
//  CHECK_CUPTI_ERROR(LOG(FATAL), cuptierr, "cuptiSubscribe sub1");
//  cuptierr = cuptiSubscribe(&sub2, CudaStreamMonitor::_ApiCallback, this);
//  CHECK_CUPTI_ERROR(LOG(FATAL), cuptierr, "cuptiSubscribe sub2");
//  exit(0);

//  CUptiResult ret;
//      ret = cupti_wrapper_->Subscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
//static_cast<CUpti_CallbackFunc>(ApiCallback)
//  VLOG(1) << "cuptiSubscribe";
  _cupti_api_callback_id = _cupti_api->RegisterCallback([this] (CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
    this->ApiCallback(domain, cbid, cbdata);
  });
//  cuptierr = cuptiSubscribe(&subscriber_, CudaStreamMonitor::_ApiCallback, this);
//  CHECK_CUPTI_ERROR(LOG(FATAL), cuptierr, "cuptiSubscribe");
//  if (cuptierr == CUPTI_ERROR_MAX_LIMIT_REACHED) {
//    VLOG(1) << "Fail 1";
//    return errors::Unavailable("CUPTI subcriber limit reached.");
//  } else if (cuptierr != CUPTI_SUCCESS) {
//    VLOG(1) << "Fail 2";
//    const char *errstr;
//    cuptiGetResultString(cuptierr, &errstr);
//    return errors::Internal("Failed to create CUPTI subcriber: ", errstr);
//  }

  // Q: If we call this multiple times, is it safe, or will a "double-call" to it erase
  // previously initialized stuff...?  I know there's some cudaDeviceReset() function
  // that has this effect.
//  cuerr = cuInit(0);
//  CHECK_CU_ERROR(LOG(FATAL), cuerr, "cuInit");

//  cuerr = cuCtxCreate(&context, 0, device);
//  CHECK_CU_ERROR(LOG(FATAL), cuerr, "cuCtxCreate");

//  cuptierr = cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)getTimestampCallback , &trace);
//  CHECK_CUPTI_ERROR(LOG(FATAL), cuptierr, "cuptiSubscribe");

//  cuptierr = cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
//  CHECK_CUPTI_ERROR(LOG(FATAL), cuptierr, "cuptiEnableDomain");

  for (auto callback_id : _callback_ids) {
    cuptierr = _cupti_api->EnableCallback(1, CUPTI_CB_DOMAIN_RUNTIME_API, callback_id);
//    cuptierr = cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API, callback_id);
    CHECK_CUPTI_ERROR(LOG(FATAL), cuptierr, "cuptiEnableDomain");
  }

}

void CudaStreamMonitor::Start() {
  if (_polling_thread) {
    return;
  }

  // Register CUPTI callbacks to monitor stream creation/deletion...
  // No, we must do that as early as possible, even before we have started sampling state.
  // i.e. in the constructor of CudaStreamMonitor.

  _polling_thread.reset(new std::thread([this] {
    this->_RunPollingThread();
  }));
}

void CudaStreamMonitor::Stop() {
  if (!_should_stop.HasBeenNotified()) {
    _should_stop.Notify();
    if (_polling_thread) {
      VLOG(1) << "Wait for CUDA Stream polling thread to stop...";
      _polling_thread->join();
      VLOG(1) << "... CUDA Stream polling done";
    }
    if (VLOG_IS_ON(1)) {
      DECLARE_LOG_INFO(info);
      this->Print(info, 0);
    }
  }
}

void CudaStreamMonitor::AddStream(cudaStream_t stream) {
  std::unique_lock<std::mutex> lock(mu_);
  VLOG(1) << "AddStream = " << reinterpret_cast<void*>(stream);
  _active_streams.push_back(stream);
  _poll_stream_summaries.emplace_back(stream);
}

void CudaStreamMonitor::RemoveStream(cudaStream_t stream) {
  std::unique_lock<std::mutex> lock(mu_);
  VLOG(1) << "RemoveStream = " << reinterpret_cast<void*>(stream);
  std::remove_if(_active_streams.begin(), _active_streams.end(),
                 [stream](cudaStream_t s) { return stream == s; });
  //  std::remove_if(_poll_stream_summaries.begin(), _poll_stream_summaries.end(),
  //    [stream](const PollStreamSummary& smry) { return stream == smry.stream; });
}

std::vector<PollStreamResult> CudaStreamMonitor::PollStreams() {
//  VLOG(1) << "CudaStreamMonitor::PollStreams";
  std::unique_lock<std::mutex> lock(mu_);
  std::vector<PollStreamResult> results;
  results.reserve(_active_streams.size());
  DCHECK(_active_streams.size() == _poll_stream_summaries.size());
  size_t i = 0;
  for (auto stream : _active_streams) {
    VLOG(1) << "  CudaStreamMonitor::PollStreams: stream = " << reinterpret_cast<void*>(stream);
    cudaError_t ret = cudaStreamQuery(stream);
    bool is_active;
    bool is_valid = true;
    if (ret == cudaErrorNotReady) {
      is_active = true;
    } else if (ret == cudaSuccess) {
      is_active = false;
    } else if (ret == cudaErrorCudartUnloading) {
      is_active = false;
      is_valid = false;
    } else {
      LOG(FATAL) << "CUDA API cudaStreamQuery returned unexpected return-code: " << cudaGetErrorString(ret);
      is_active = false;
    }
    results.emplace_back(stream, is_active, is_valid);
    _poll_stream_summaries[i].AddPollStreamResult(results[i]);
//    {
//      DECLARE_LOG_INFO(info);
//      _poll_stream_summaries[i].Print(info, 0);
//    }
    i += 1;
  }
  {
    DECLARE_LOG_INFO(info);
    this->Print(info, 0);
  }
  return results;
}

std::ostream& CudaStreamMonitor::Print(std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << "CudaStreamMonitor: ";
  for (auto& poll_stream_summary : _poll_stream_summaries) {
    out << "\n";
    poll_stream_summary.Print(out, indent + 1);
  }
  return out;
}

std::ostream& PollStreamResult::Print(std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << "PollStreamResult: "
      << "stream = " << reinterpret_cast<void*>(stream)
      << ", " << "is_active = " << is_active;
  return out;
}

std::ostream& PollStreamSummary::Print(std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << "PollStreamSummary: "
      << "stream = " << reinterpret_cast<void*>(stream)
      << ", " << "num_samples_is_active = " << num_samples_is_active
      << ", " << "num_samples_is_inactive = " << num_samples_is_inactive;
  return out;
}

void PollStreamSummary::AddPollStreamResult(const PollStreamResult& poll_stream) {
//  VLOG(1) << "PollStreamSummary::AddPollStreamResult, is_valid = " << poll_stream.is_valid << ", is_active = " << poll_stream.is_active;
  DCHECK(poll_stream.stream == stream);
  if (poll_stream.is_valid) {
    if (poll_stream.is_active) {
      num_samples_is_active += 1;
    } else {
      num_samples_is_inactive += 1;
    }
  }
//  VLOG(1) << "PollStreamSummary::AddPollStreamResult, is_valid = " << poll_stream.is_valid << ", is_active = " << poll_stream.is_active;
//  VLOG(1) << "PollStreamSummary::AddPollStreamResult, num_samples_is_active = " << num_samples_is_active << ", num_samples_is_inactive = " << num_samples_is_inactive;
}

}
