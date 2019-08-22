//
// Created by jagle on 8/19/2019.
//

#ifndef IML_CUDA_LD_PRELOAD_H
#define IML_CUDA_LD_PRELOAD_H

#include <memory>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_api_profiler/registered_handle.h"

#include "tensorflow/core/platform/mutex.h"

#include <vector>
#include <functional>

typedef cudaError_t (*cudaLaunchKernel_func)(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);

using CudaAPIFuncId = int;
enum {
  CUDA_API_FUNC_ID_UNUSED = -1,
  CUDA_API_FUNC_ID_FIRST = 0
};

template <typename StartCallback, typename ExitCallback>
struct CudaAPICallback {
  CudaAPICallback(CudaAPIFuncId func_id, StartCallback start_cb, ExitCallback exit_cb) :
      func_id(func_id),
      start_cb(start_cb),
      exit_cb(exit_cb)
  {
  }
  CudaAPIFuncId func_id;
  StartCallback start_cb;
  ExitCallback exit_cb;
};

//
// Define CUDA API callbacks using the function types of CUDA runtime functions (cuda_runtime_api.h).
//

using cudaLaunchKernel_start_cb = std::function<void (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)>;
using cudaLaunchKernel_exit_cb = std::function<void (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream, cudaError_t ret)>;
using cudaLaunchKernel_callback = CudaAPICallback<cudaLaunchKernel_start_cb, cudaLaunchKernel_exit_cb>;

// Forward decl.
template <class CudaCallbackType> struct CudaAPIRegisteredFuncHandle;

// e.g. CudaCallbackType = cudaLaunchKernel_callback
template <class CudaCallbackType>
struct CudaAPICallbacks {
  std::vector<CudaCallbackType> callbacks;
  CudaAPIFuncId _next_func_id;
  CudaAPICallbacks() :
      _next_func_id(CUDA_API_FUNC_ID_FIRST)
  {
  }
  // For some reason, just the act of adding this data-memmber makes cudaLanch time go from  1:26:35 -> 1:29:13...
  // I doubt this is something we should care about... (likely miniscule caching effect we cannot control anyways.)
//  tensorflow::mutex _mu;
  // e.g.
  // start_cb = cudaLaunchKernel_start_cb
  // exit_cb = cudaLaunchKernel_exit_cb
  template <typename StartCallback, typename ExitCallback>
  tensorflow::RegisteredHandle<CudaAPIFuncId> RegisterCallback(StartCallback start_cb, ExitCallback exit_cb) {
//    tensorflow::mutex_lock lock(_mu);
    auto func_id = _next_func_id;
    _next_func_id += 1;
    callbacks.emplace_back(func_id, start_cb, exit_cb);
    tensorflow::RegisteredHandle<CudaAPIFuncId> handle(func_id, /*unregister_cb=*/[this] (CudaAPIFuncId func_id) {
      this->UnregisterCallback(func_id);
    });
    VLOG(1) << "Register func_id = " << func_id;
    return handle;
  }

  void UnregisterCallback(CudaAPIFuncId func_id) {
//    tensorflow::mutex_lock lock(_mu);
    std::remove_if(callbacks.begin(), callbacks.end(),
                   [func_id](const CudaCallbackType& cb) { return cb.func_id == func_id; });
  }

  template <typename ...Params>
  void StartCallbacks(Params&&... params) {
    for (auto const& cb : callbacks) {
      cb.start_cb(std::forward<Params>(params)...);
    }
  }

  template <typename ...Params>
  void ExitCallbacks(Params&&... params) {
    for (auto const& cb : callbacks) {
      cb.exit_cb(std::forward<Params>(params)...);
    }
  }

};


struct CudaAPI {

  cudaLaunchKernel_func cudaLaunchKernel;
  CudaAPICallbacks<cudaLaunchKernel_callback> cudaLaunchKernel_cbs;
//  RegisteredHandle<CudaAPIFuncId> cudaLaunchKernel_cb_empty_01;
//  RegisteredHandle<CudaAPIFuncId> cudaLaunchKernel_cb_empty_02;

  CudaAPI();
};

class CudaLibrary {
public:
  CudaLibrary();
  ~CudaLibrary();
  void DLOpen();
  void DLClose();
  const char* DLError();
  void* _lib_handle;
  CudaAPI _cuda_api;
};

//std::shared_ptr<CudaLibrary> GetCudaLibrary();
CudaLibrary* GetCudaLibrary();

// Use LD_PRELOAD trick to intercept CUDA runtime API function calls.
extern "C" {
extern __host__ cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
}

#endif //IML_CUDA_LD_PRELOAD_H
