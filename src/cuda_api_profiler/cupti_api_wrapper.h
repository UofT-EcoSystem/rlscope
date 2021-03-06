//
// Created by jagle on 8/16/2019.
//

#ifndef RLSCOPE_CUPTI_API_WRAPPER_H
#define RLSCOPE_CUPTI_API_WRAPPER_H

#include <cupti_target.h>
#include <cupti.h>
#include <cuda.h>

#include <vector>
#include <functional>
#include <memory>

#include "cuda_api_profiler/registered_handle.h"

#include <mutex>

namespace rlscope {

// There are some singleton instances that get managed by the CUPTI API.
// For example, only ONE call to cuptiSubscribe(&handle, callback) is allowed.
// To better manage CUPTI function registration across classes, use a singleton instance
// that manages singleton resources like this.

struct CuptiCallback {
  using Callback = std::function<void(
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid,
      const void *cbdata)>;
  using FuncId = int;

  CuptiCallback(FuncId func_id, Callback callback) :
      func_id(func_id),
      callback(callback) {
  }

  FuncId func_id;
  Callback callback;
};

class CuptiAPI {
public:
  static std::shared_ptr<CuptiAPI> GetCuptiAPI();

  using FuncId = int;
  std::mutex _mu;

  CUpti_SubscriberHandle _subscriber{nullptr};
  FuncId _next_func_id{0};

  std::vector<CuptiCallback> _cupti_subscribe_callbacks;

  CuptiAPI() = default;
  ~CuptiAPI();
  void ClearCallbacks();

  static void CUPTIAPI __RunCUPTICallbacks(
      void *userdata, CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid, const void *cbdata);
  void _RunCUPTICallbacks(
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid, const void *cbdata);

  RegisteredHandle<CuptiCallback::FuncId> RegisterCallback(CuptiCallback::Callback callback);

  CUptiResult EnableCallback(
      uint32_t enable,
      // CUpti_SubscriberHandle subscriber,
      CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid);

  CUptiResult EnableDomain(
      uint32_t enable,
      // CUpti_SubscriberHandle subscriber,
      CUpti_CallbackDomain domain);

  void _CheckCallbacksRegistered();

  void UnregisterCallback(CuptiCallback::FuncId func_id);

};

}

#endif //RLSCOPE_CUPTI_API_WRAPPER_H
