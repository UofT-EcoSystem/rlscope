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

//  cudaLaunchKernel_cb_empty_01 = std::move(cudaLaunchKernel_cbs.RegisterCallback(
//      /*start_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
//        // pass
//      },
//      /*end_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream, cudaError_t ret) {
//        // pass
//      }));
//  VLOG(1) << "Register func_id = " << cudaLaunchKernel_cb_empty_01.func_id;
//
//  cudaLaunchKernel_cb_empty_02 = std::move(cudaLaunchKernel_cbs.RegisterCallback(
//      /*start_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
//        // pass
//      },
//      /*end_cb=*/ [] (const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream, cudaError_t ret) {
//        // pass
//      }));
//  VLOG(1) << "Register func_id = " << cudaLaunchKernel_cb_empty_02.func_id;

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

#define LOAD_SYM(funcname) \
  _cuda_api.funcname = reinterpret_cast<funcname ##_func>(dlsym(_lib_handle, #funcname)); \
  if (_cuda_api.funcname == nullptr) { \
    error = DLError(); \
    LOG(FATAL) << "dlsym(\"" << #funcname << "\") failed: " << error; \
  }

  LOAD_SYM(cudaLaunchKernel);
  LOAD_SYM(cudaMemcpyAsync);
  LOAD_SYM(cudaMalloc);
  LOAD_SYM(cudaFree);

//  _cuda_api.cudaLaunchKernel = reinterpret_cast<cudaLaunchKernel_func>(dlsym(_lib_handle, "cudaLaunchKernel"));
//  if (_cuda_api.cudaLaunchKernel == nullptr) {
//    error = DLError();
//    LOG(FATAL) << "dlsym(\"cudaLaunchKernel\") failed: " << error;
//  }

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

//cudaDeviceGetAttribute
//cudaEventCreateWithFlags
//cudaFree
//cudaGetDevice
//cudaGetDeviceProperties
//cudaGetLastError
//cudaLaunchKernel
//cudaMalloc
//cudaMemcpy
//cudaMemcpyAsync

__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  _cuda_library._cuda_api.cudaLaunchKernel_cbs.StartCallbacks(func, gridDim, blockDim, args, sharedMem, stream);
  auto ret = _cuda_library._cuda_api.cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  _cuda_library._cuda_api.cudaLaunchKernel_cbs.ExitCallbacks(
      func, gridDim, blockDim, args, sharedMem, stream,
      ret);
  return ret;
}

//extern __host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  _cuda_library._cuda_api.cudaMemcpyAsync_cbs.StartCallbacks(dst, src, count, kind, stream);
  auto ret = _cuda_library._cuda_api.cudaMemcpyAsync(dst, src, count, kind, stream);
  _cuda_library._cuda_api.cudaMemcpyAsync_cbs.ExitCallbacks(
      dst, src, count, kind, stream,
      ret);
  return ret;
}

//extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaMalloc(void **devPtr, size_t size) {
  _cuda_library._cuda_api.cudaMalloc_cbs.StartCallbacks(devPtr, size);
  auto ret = _cuda_library._cuda_api.cudaMalloc(devPtr, size);
  _cuda_library._cuda_api.cudaMalloc_cbs.ExitCallbacks(
      devPtr, size,
      ret);
  return ret;
}

//extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaFree(void *devPtr) {
  _cuda_library._cuda_api.cudaFree_cbs.StartCallbacks(devPtr);
  auto ret = _cuda_library._cuda_api.cudaFree(devPtr);
  _cuda_library._cuda_api.cudaFree_cbs.ExitCallbacks(
      devPtr,
      ret);
  return ret;
}

}
