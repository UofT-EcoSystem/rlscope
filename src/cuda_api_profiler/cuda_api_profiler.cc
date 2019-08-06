//
// Created by jagle on 8/2/2019.
//

#include "cuda_api_profiler/cuda_api_profiler.h"

CUDAAPIProfiler::~CUDAAPIProfiler() {
    if (VLOG_IS_ON(INFO)) {
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
//  std::stringstream ss;
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
//  out << ss.str();
}

void CUDAAPIProfiler::ApiCallback(
        void *userdata,
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
        (*timestamp_map)[api_key] = Env::Default()->NowMicros();
        if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            _api_stats[api_key].AddCall(_start_t.at(api_key), _end_t.at(api_key));
        }
    }
}
