//
// Created by jagle on 8/19/2019.
//

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"

#include "cuda_api_profiler/cuda_ld_preload.h"
#include "cuda_ld_preload_export.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>

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

CudaLibrary::CudaLibrary() {
}
const char* CudaLibrary::DLError() {
  char* error = dlerror();
  if (error == nullptr) {
    return "Unknown";
  }
  return error;
}
void CudaLibrary::DLClose() {
  _libcudart.Close();
  _libcuda.Close();
}

#define DEFAULT_DLOPEN_FLAGS (RTLD_NOW | RTLD_LOCAL)
//#define DEFAULT_DLOPEN_FLAGS (RTLD_NOW | RTLD_GLOBAL)

LibHandle::LibHandle() :
    _lib_handle(nullptr),
    _flags(DEFAULT_DLOPEN_FLAGS)
{
}

LibHandle::LibHandle(const std::string& so_path, int flags) :
    _so_path(so_path),
    _lib_handle(nullptr),
    _flags(flags)
{
}
LibHandle::LibHandle(const std::string& so_path) :
    _so_path(so_path),
    _lib_handle(nullptr),
    _flags(DEFAULT_DLOPEN_FLAGS)
{
}
const char* LibHandle::_DLError() {
  const char* error = dlerror();
  if (error == nullptr) {
    return "Unknown";
  }
  return error;
}
bool LibHandle::Opened() const {
  return _lib_handle != nullptr;
}
void LibHandle::Open() {
  const char *error;
  if (_lib_handle) {
    return;
  }
  _lib_handle = dlopen(_so_path.c_str(), _flags);
  if (_lib_handle == nullptr) {
    error = _DLError();
    LOG(FATAL) << "dlopen(\"" << _so_path << "\") failed: " << error;
  }
}
void LibHandle::Close() {
  if (_lib_handle) {
    dlclose(_lib_handle);
    _lib_handle = nullptr;
  }
}
void* LibHandle::LoadSym(const std::string& funcname) {
  DCHECK(Opened());
  void* sym = dlsym(_lib_handle, funcname.c_str());
  if (sym == nullptr) {
    const char* error = _DLError();
    LOG(FATAL) << "dlsym(lib=\"" << _so_path << "\", func=\"" << funcname << "\") failed: " << error;
  }
  return sym;
}

LibHandle& LibHandle::operator=(LibHandle&& other) {
  if (this != &other) {
    // Move assignment operator: this is initialized, need to free existing resources first.
    this->Close();
  }
  this->_so_path = other._so_path;
  this->_lib_handle = other._lib_handle;
  this->_flags = other._flags;
  // Prevent double-calls to dlclose(...).
  other._lib_handle = nullptr;
  return *this;
}
LibHandle::LibHandle( LibHandle&& other ) {
  // Move constructor: this is uninitialized, no need to free existing resources.
  this->_so_path = other._so_path;
  this->_lib_handle = other._lib_handle;
  this->_flags = other._flags;
  // Prevent double-calls to dlclose(...).
  other._lib_handle = nullptr;
}

LibHandle::~LibHandle() {
  Close();
}

void CudaLibrary::DLOpen() {
//  const char *error;
  if (_libcudart.Opened()) {
    return;
  }
  _libcudart = LibHandle("libcudart.so");
  _libcudart.Open();

  DCHECK(!_libcuda.Opened());
  _libcuda = LibHandle("libcuda.so");
  _libcuda.Open();

#define LOAD_CUDART_SYM(funcname) \
  _cuda_api.funcname = reinterpret_cast<funcname ##_func>(_libcudart.LoadSym(#funcname));

#define LOAD_CUDA_SYM(funcname) \
  _cuda_api.funcname = reinterpret_cast<funcname ##_func>(_libcuda.LoadSym(#funcname));

  LOAD_CUDART_SYM(cudaLaunchKernel_ptsz);
  LOAD_CUDART_SYM(cudaLaunchKernel);
  LOAD_CUDART_SYM(cudaMemcpyAsync);
  LOAD_CUDART_SYM(cudaMalloc);
  LOAD_CUDART_SYM(cudaFree);

  LOAD_CUDA_SYM(cuLaunchKernel);

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

bool CudaLibrary::Opened() const {
  return _libcudart.Opened() && _libcuda.Opened();
}

CudaLibrary* GetCudaLibrary() {
  if (!_cuda_library.Opened()) {
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

CUresult CUDA_LD_PRELOAD_EXPORT cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  VLOG(1) << "cuda_ld_preload." << __func__;
  _cuda_library._cuda_api.cuLaunchKernel_cbs.StartCallbacks(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
  auto ret = _cuda_library._cuda_api.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
  _cuda_library._cuda_api.cuLaunchKernel_cbs.ExitCallbacks(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra,
      ret);
  return ret;
}

__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaLaunchKernel_ptsz(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  VLOG(1) << "cuda_ld_preload." << __func__;
  _cuda_library._cuda_api.cudaLaunchKernel_ptsz_cbs.StartCallbacks(func, gridDim, blockDim, args, sharedMem, stream);
  auto ret = _cuda_library._cuda_api.cudaLaunchKernel_ptsz(func, gridDim, blockDim, args, sharedMem, stream);
  _cuda_library._cuda_api.cudaLaunchKernel_ptsz_cbs.ExitCallbacks(
      func, gridDim, blockDim, args, sharedMem, stream,
      ret);
  return ret;
}

__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  VLOG(1) << "cuda_ld_preload." << __func__;
  _cuda_library._cuda_api.cudaLaunchKernel_cbs.StartCallbacks(func, gridDim, blockDim, args, sharedMem, stream);
  auto ret = _cuda_library._cuda_api.cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  _cuda_library._cuda_api.cudaLaunchKernel_cbs.ExitCallbacks(
      func, gridDim, blockDim, args, sharedMem, stream,
      ret);
  return ret;
}

//extern __host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
  VLOG(1) << "cuda_ld_preload." << __func__;
  _cuda_library._cuda_api.cudaMemcpyAsync_cbs.StartCallbacks(dst, src, count, kind, stream);
  auto ret = _cuda_library._cuda_api.cudaMemcpyAsync(dst, src, count, kind, stream);
  _cuda_library._cuda_api.cudaMemcpyAsync_cbs.ExitCallbacks(
      dst, src, count, kind, stream,
      ret);
  return ret;
}

//extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaMalloc(void **devPtr, size_t size) {
  VLOG(1) << "cuda_ld_preload." << __func__;
  _cuda_library._cuda_api.cudaMalloc_cbs.StartCallbacks(devPtr, size);
  auto ret = _cuda_library._cuda_api.cudaMalloc(devPtr, size);
  _cuda_library._cuda_api.cudaMalloc_cbs.ExitCallbacks(
      devPtr, size,
      ret);
  return ret;
}

//extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
__host__ cudaError_t CUDA_LD_PRELOAD_EXPORT cudaFree(void *devPtr) {
  VLOG(1) << "cuda_ld_preload." << __func__;
  _cuda_library._cuda_api.cudaFree_cbs.StartCallbacks(devPtr);
  auto ret = _cuda_library._cuda_api.cudaFree(devPtr);
  _cuda_library._cuda_api.cudaFree_cbs.ExitCallbacks(
      devPtr,
      ret);
  return ret;
}

}
