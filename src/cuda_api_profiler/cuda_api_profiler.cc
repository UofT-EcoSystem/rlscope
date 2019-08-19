//
// Created by jagle on 8/2/2019.
//


#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"

#include <map>

//#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <list>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include "cuda_api_profiler/cuda_api_profiler.h"
#include "cuda_api_profiler/defines.h"
#include "cuda_api_profiler/cupti_logging.h"

namespace tensorflow {

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = cupti_wrapper_->call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

CUDAAPIProfiler::~CUDAAPIProfiler() {
    if (VLOG_IS_ON(1)) {
        Print(LOG(INFO));
    }
}

std::map<CUpti_CallbackId, std::string> CUDAAPIProfiler::RuntimeCallbackIDToName() {
    std::map<CUpti_CallbackId, std::string> cbid_to_name;
    for (int i = CUPTI_RUNTIME_TRACE_CBID_INVALID + 1; i < CUPTI_RUNTIME_TRACE_CBID_SIZE; i++) {
        cbid_to_name[i] = runtime_cbid_to_string(i);
    }
    return cbid_to_name;
}

std::map<CUpti_CallbackId, std::string> CUDAAPIProfiler::DriverCallbackIDToName() {
    std::map<CUpti_CallbackId, std::string> cbid_to_name;
    for (int i = CUPTI_DRIVER_TRACE_CBID_INVALID + 1; i < CUPTI_DRIVER_TRACE_CBID_SIZE; i++) {
        cbid_to_name[i] = driver_cbid_to_string(i);
    }
    return cbid_to_name;
}

template <class Stream>
void CUDAAPIProfiler::Print(Stream&& out) {
    mutex_lock l(_mu);
    out << "CUDAAPIProfiler: size = " << _api_stats.size() << "\n";
    for (auto const& pair : _api_stats) {
        auto tid = std::get<0>(pair.first);
        auto domain = std::get<1>(pair.first);
        auto cbid = std::get<2>(pair.first);
        const char* name;
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            name = runtime_cbid_to_string(cbid);
        } else if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            name = driver_cbid_to_string(cbid);
        } else {
            name = "UNKNOWN";
        }
        out << "  " << "(tid=" << tid << ", api=" << name << "):\n"
            << "    " << "total_api_time_usec = " << pair.second.total_api_time_usec << "\n"
            << "    " << "n_calls = " << pair.second.n_calls << "\n";
    }
}

void CUDAAPIProfiler::ApiCallback(
        CUpti_CallbackDomain domain,
        CUpti_CallbackId cbid,
        const void *cbdata) {
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API || domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
        std::map<APIKey, TimeUsec>* timestamp_map;
        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            timestamp_map = &_start_t;
        } else {
            timestamp_map = &_end_t;
        }
        auto api_key = std::make_tuple(gettid(), domain, cbid);

        {
            mutex_lock l(_mu);

            (*timestamp_map)[api_key] = Env::Default()->NowMicros();
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                _api_stats[api_key].AddCall(_start_t.at(api_key), _end_t.at(api_key));
            }
        }

    }
}




CUDAAPIProfilerPrinter::CUDAAPIProfilerPrinter(CUDAAPIProfiler &profiler, float every_sec) :
    _profiler(profiler),
    _every_sec(every_sec)
{
}

void CUDAAPIProfilerPrinter::_Run() {
  _event_handler.RegisterFunc([this]() {
    this->_EverySec();
  }, _every_sec);

  _event_handler.EventLoop([this]() {
    return _should_stop.HasBeenNotified();
  });
}

void CUDAAPIProfilerPrinter::_EverySec() {
  if (VLOG_IS_ON(1)) {
    _profiler.Print(LOG(INFO));
  }
}

void CUDAAPIProfilerPrinter::Start() {
    if (_printer_thread) {
        return;
    }
    _printer_thread.reset(new std::thread([this] {
        this->_Run();
    }));
}

void CUDAAPIProfilerPrinter::Stop() {
  if (!_should_stop.HasBeenNotified()) {
    _should_stop.Notify();
    if (_printer_thread) {
      VLOG(1) << "Wait for CUDA API stat printer thread to stop...";
      _printer_thread->join();
      VLOG(1) << "... printer thread done";
    }
  }
}

CUDAAPIProfilerPrinter::~CUDAAPIProfilerPrinter() {
    Stop();
}

}
