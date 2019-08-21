//
// Created by jagle on 8/19/2019.
//

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"

#include "cuda_api_profiler/cuda_ld_preload.h"
#include "cuda_ld_preload_export.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <functional>

//#error "CUDA LD_PRELOAD file was included"

using namespace tensorflow;

CudaAPI::CudaAPI() {
  cudaLaunchKernel = nullptr;

  cudaLaunchKernel_cb_empty_01 = std::move(cudaLaunchKernel_cbs.RegisterCallback(
      /*start_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
        // pass
      },
      /*end_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream, cudaError_t ret) {
        // pass
      }));
  VLOG(1) << "Register func_id = " << cudaLaunchKernel_cb_empty_01.func_id;

  cudaLaunchKernel_cb_empty_02 = std::move(cudaLaunchKernel_cbs.RegisterCallback(
      /*start_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
        // pass
      },
      /*end_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream, cudaError_t ret) {
        // pass
      }));
  VLOG(1) << "Register func_id = " << cudaLaunchKernel_cb_empty_02.func_id;

}

CudaLibrary::CudaLibrary() : _lib_handle(nullptr) {
}
const char* CudaLibrary::DLError() {
  char* error = dlerror();
  if (error == nullptr) {
    return "Unknown";
  }
  return error;
}
void CudaLibrary::DLClose() {
  if (_lib_handle) {
    dlclose(_lib_handle);
  }
}

void CudaLibrary::DLOpen() {
  const char *error;
  if (_lib_handle) {
    return;
  }
  _lib_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
  if (_lib_handle == nullptr) {
    error = DLError();
    LOG(FATAL) << "dlopen(\"libcudart.so\") failed: " << error;
  }
  _cuda_api.cudaLaunchKernel = reinterpret_cast<cudaLaunchKernel_func>(dlsym(_lib_handle, "cudaLaunchKernel"));
  if (_cuda_api.cudaLaunchKernel == nullptr) {
    error = DLError();
    LOG(FATAL) << "dlsym(\"cudaLaunchKernel\") failed: " << error;
  }
}

CudaLibrary::~CudaLibrary() {
  DLClose();
}

CudaLibrary _cuda_library;

CudaLibrary* GetCudaLibrary() {
  if (!_cuda_library._lib_handle) {
    _cuda_library.DLOpen();
  }
  return &_cuda_library;
}

extern "C" {

cudaError_t __host__ CUDA_LD_PRELOAD_EXPORT cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  _cuda_library._cuda_api.cudaLaunchKernel_cbs.StartCallbacks(func, gridDim, blockDim, args, sharedMem, stream);
  auto ret = _cuda_library._cuda_api.cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  _cuda_library._cuda_api.cudaLaunchKernel_cbs.ExitCallbacks(func, gridDim, blockDim, args, sharedMem, stream, ret);
  return ret;
}

}
