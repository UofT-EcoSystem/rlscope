//
// Created by jagle on 8/16/2019.
//

#include <memory>
#include <algorithm>

#include "common_util.h"

#include "cuda_api_profiler/cupti_logging.h"
#include "cuda_api_profiler/cupti_api_wrapper.h"

#include <cupti_target.h>
#include <cupti.h>
#include <cuda.h>

#include <mutex>

namespace rlscope {

static std::shared_ptr<CuptiAPI> _cupti_api;

/* static */ std::shared_ptr<CuptiAPI> CuptiAPI::GetCuptiAPI() {
  if (!_cupti_api) {
    _cupti_api.reset(new CuptiAPI());
  }
  return _cupti_api;
}

RegisteredHandle<CuptiCallback::FuncId> CuptiAPI::RegisterCallback(CuptiCallback::Callback callback) {
  std::unique_lock<std::mutex> lock(_mu);
  CUptiResult ret;
  if (_next_func_id == 0) {
//    ret = cuptiSubscribe(&_subscriber, CuptiAPI::__RunCUPTICallbacks, this);
//    CHECK_CUPTI_ERROR(LOG(FATAL), ret, "cuptiSubscribe");
    CUPTI_API_CALL_MAYBE_EXIT(cuptiSubscribe(&_subscriber, CuptiAPI::__RunCUPTICallbacks, this));
//    std::stringstream ss;
//    ss << "cuptiSubscribe: CuptiAPI::__RunCUPTICallbacks"
    RLS_LOG("CUPTI", "cuptiSubscribe: CuptiAPI::__RunCUPTICallbacks, subscriber={}", reinterpret_cast<void*>(_subscriber));
  }
  auto func_id = _next_func_id;
  RegisteredHandle<CuptiCallback::FuncId> handle(func_id, [this] (CuptiCallback::FuncId func_id) {
    this->UnregisterCallback(func_id);
  });
  _cupti_subscribe_callbacks.emplace_back(func_id, callback);
  _next_func_id += 1;
  return handle;
}

/* static */ void CUPTIAPI CuptiAPI::__RunCUPTICallbacks(
    void *userdata, CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid, const void *cbdata) {
  auto *self = reinterpret_cast<CuptiAPI*>(userdata);
  self->_RunCUPTICallbacks(domain, cbid, cbdata);
}

void CuptiAPI::_RunCUPTICallbacks(
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid, const void *cbdata) {
  for (auto const& callback : _cupti_subscribe_callbacks) {
    callback.callback(domain, cbid, cbdata);
  }
}

void CuptiAPI::UnregisterCallback(CuptiCallback::FuncId func_id) {
  std::unique_lock<std::mutex> lock(_mu);
  std::remove_if(_cupti_subscribe_callbacks.begin(), _cupti_subscribe_callbacks.end(),
                 [func_id](const CuptiCallback& cb) { return cb.func_id == func_id; });
}

CUptiResult CuptiAPI::EnableCallback(
    uint32_t enable,
    // CUpti_SubscriberHandle subscriber,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid) {
  CUptiResult ret;
  _CheckCallbacksRegistered();
  ret = cuptiEnableCallback(enable, _subscriber, domain, cbid);
  return ret;
}

void CuptiAPI::_CheckCallbacksRegistered() {
  DCHECK(_next_func_id > 0) << "You should call CuptiAPI::RegisterCallback before EnableCallback.";
  assert(_subscriber != nullptr);
}

CUptiResult CuptiAPI::EnableDomain(
    uint32_t enable,
    // CUpti_SubscriberHandle subscriber,
    CUpti_CallbackDomain domain)
{
  CUptiResult ret;
  _CheckCallbacksRegistered();
  ret = cuptiEnableDomain(enable, _subscriber, domain);
  return ret;
}

void CuptiAPI::ClearCallbacks() {
  if (_next_func_id > 0) {
    {
      std::unique_lock<std::mutex> lock(_mu);
      RLS_LOG("CUPTI", "cuptiEnableAllDomains: DISABLE, subscriber={}", reinterpret_cast<void*>(_subscriber));
      CUPTI_API_CALL_MAYBE_EXIT(cuptiEnableAllDomains(/*enable=*/0, _subscriber));
    }
//    CUptiResult ret;
//    ret = cuptiUnsubscribe(_subscriber);
//    CHECK_CUPTI_ERROR(LOG(FATAL), ret, "cuptiUnsubscribe");
    assert(_subscriber != nullptr);
    RLS_LOG("CUPTI", "cuptiUnsubscribe: CuptiAPI::__RunCUPTICallbacks, subscriber={}", reinterpret_cast<void*>(_subscriber));
    CUPTI_API_CALL_MAYBE_EXIT(cuptiUnsubscribe(_subscriber));
    _subscriber = nullptr;
    _next_func_id = 0;
    _cupti_subscribe_callbacks.clear();
  }

}

CuptiAPI::~CuptiAPI() {
  ClearCallbacks();
//  if (_next_func_id > 0) {
//    {
//      std::unique_lock<std::mutex> lock(_mu);
//      RLS_LOG("CUPTI", "cuptiEnableAllDomains: DISABLE, subscriber={}", reinterpret_cast<void*>(_subscriber));
//      CUPTI_API_CALL_MAYBE_EXIT(cuptiEnableAllDomains(/*enable=*/0, _subscriber));
//    }
////    CUptiResult ret;
////    ret = cuptiUnsubscribe(_subscriber);
////    CHECK_CUPTI_ERROR(LOG(FATAL), ret, "cuptiUnsubscribe");
//    assert(_subscriber != nullptr);
//    RLS_LOG("CUPTI", "cuptiUnsubscribe: CuptiAPI::__RunCUPTICallbacks, subscriber={}", reinterpret_cast<void*>(_subscriber));
//    CUPTI_API_CALL_MAYBE_EXIT(cuptiUnsubscribe(_subscriber));
//  }
}

}
