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

#define USEC_IN_MS (1000)
#define USEC_IN_SEC (USEC_IN_MS*1000)

#include "cuda_api_profiler/cuda_api_profiler.h"

#define XSTR(a) STR(a)
#define STR(a) #a

namespace tensorflow {


void EventHandler::EventLoop(std::function<bool()> should_stop) {
  int64 sleep_for_usec = round(EVENT_LOOP_EVERY_SEC * USEC_IN_SEC);
  while (not should_stop()) {
    Env::Default()->SleepForMicroseconds(sleep_for_usec);
    RunFuncs();
  }
}

bool RegisteredFunc::ShouldRun(uint64 now_usec) {
  DCHECK(last_run_usec <= now_usec);
  bool ret = last_run_usec == 0 or
             (now_usec - last_run_usec) >= (every_sec*USEC_IN_SEC);
  if (ret) {
    VLOG(1) << "now_usec = " << now_usec
            << ", last_run_usec = " << last_run_usec
            << ", time_usec since last ran = " << now_usec - last_run_usec
            << ", run every usec = " << every_sec*USEC_IN_SEC;
  }
  return ret;
}


void RegisteredFunc::Run(uint64 now_usec) {
  last_run_usec = now_usec;
  func();
}

void EventHandler::RegisterFunc(RegisteredFunc::Func func, float every_sec) {
  _funcs.emplace_back(RegisteredFunc(func, every_sec));
}

void EventHandler::RunFuncs() {
  auto now_usec = Env::Default()->NowMicros();
  for (auto& func : _funcs) {
    if (func.ShouldRun(now_usec)) {
      func.Run(now_usec);
    }
  }
}

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = cupti_wrapper_->call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

const char* runtime_cbid_to_string(CUpti_CallbackId cbid);
const char* driver_cbid_to_string(CUpti_CallbackId cbid);

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

        {
            mutex_lock l(_mu);

            (*timestamp_map)[api_key] = Env::Default()->NowMicros();
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                _api_stats[api_key].AddCall(_start_t.at(api_key), _end_t.at(api_key));
            }
        }

    }
}

const char* runtime_cbid_to_string(CUpti_CallbackId cbid) {

#define CUPTI_RUNTIME_CASE(apiName, version) \
    case CUPTI_RUNTIME_TRACE_CBID_ ## apiName ## _ ## version: \
        return STR(apiName);

    switch (cbid) {
        CUPTI_RUNTIME_CASE(cudaDriverGetVersion, v3020)
        CUPTI_RUNTIME_CASE(cudaRuntimeGetVersion, v3020)
        CUPTI_RUNTIME_CASE(cudaGetDeviceCount, v3020)
        CUPTI_RUNTIME_CASE(cudaGetDeviceProperties, v3020)
        CUPTI_RUNTIME_CASE(cudaChooseDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGetChannelDesc, v3020)
        CUPTI_RUNTIME_CASE(cudaCreateChannelDesc, v3020)
        CUPTI_RUNTIME_CASE(cudaConfigureCall, v3020)
        CUPTI_RUNTIME_CASE(cudaSetupArgument, v3020)
        CUPTI_RUNTIME_CASE(cudaGetLastError, v3020)
        CUPTI_RUNTIME_CASE(cudaPeekAtLastError, v3020)
        CUPTI_RUNTIME_CASE(cudaGetErrorString, v3020)
        CUPTI_RUNTIME_CASE(cudaLaunch, v3020)
        CUPTI_RUNTIME_CASE(cudaFuncSetCacheConfig, v3020)
        CUPTI_RUNTIME_CASE(cudaFuncGetAttributes, v3020)
        CUPTI_RUNTIME_CASE(cudaSetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaSetValidDevices, v3020)
        CUPTI_RUNTIME_CASE(cudaSetDeviceFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaMalloc, v3020)
        CUPTI_RUNTIME_CASE(cudaMallocPitch, v3020)
        CUPTI_RUNTIME_CASE(cudaFree, v3020)
        CUPTI_RUNTIME_CASE(cudaMallocArray, v3020)
        CUPTI_RUNTIME_CASE(cudaFreeArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMallocHost, v3020)
        CUPTI_RUNTIME_CASE(cudaFreeHost, v3020)
        CUPTI_RUNTIME_CASE(cudaHostAlloc, v3020)
        CUPTI_RUNTIME_CASE(cudaHostGetDevicePointer, v3020)
        CUPTI_RUNTIME_CASE(cudaHostGetFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaMemGetInfo, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2D, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyToArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DToArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DFromArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyArrayToArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DArrayToArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyToSymbol, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromSymbol, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyToArrayAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromArrayAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DToArrayAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DFromArrayAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyToSymbolAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromSymbolAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemset, v3020)
        CUPTI_RUNTIME_CASE(cudaMemset2D, v3020)
        CUPTI_RUNTIME_CASE(cudaMemsetAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemset2DAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaGetSymbolAddress, v3020)
        CUPTI_RUNTIME_CASE(cudaGetSymbolSize, v3020)
        CUPTI_RUNTIME_CASE(cudaBindTexture, v3020)
        CUPTI_RUNTIME_CASE(cudaBindTexture2D, v3020)
        CUPTI_RUNTIME_CASE(cudaBindTextureToArray, v3020)
        CUPTI_RUNTIME_CASE(cudaUnbindTexture, v3020)
        CUPTI_RUNTIME_CASE(cudaGetTextureAlignmentOffset, v3020)
        CUPTI_RUNTIME_CASE(cudaGetTextureReference, v3020)
        CUPTI_RUNTIME_CASE(cudaBindSurfaceToArray, v3020)
        CUPTI_RUNTIME_CASE(cudaGetSurfaceReference, v3020)
        CUPTI_RUNTIME_CASE(cudaGLSetGLDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGLRegisterBufferObject, v3020)
        CUPTI_RUNTIME_CASE(cudaGLMapBufferObject, v3020)
        CUPTI_RUNTIME_CASE(cudaGLUnmapBufferObject, v3020)
        CUPTI_RUNTIME_CASE(cudaGLUnregisterBufferObject, v3020)
        CUPTI_RUNTIME_CASE(cudaGLSetBufferObjectMapFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaGLMapBufferObjectAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaGLUnmapBufferObjectAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaWGLGetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsGLRegisterImage, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsGLRegisterBuffer, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsUnregisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsResourceSetMapFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsMapResources, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsUnmapResources, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsResourceGetMappedPointer, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsSubResourceGetMappedArray, v3020)
        CUPTI_RUNTIME_CASE(cudaVDPAUGetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaVDPAUSetVDPAUDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsVDPAURegisterVideoSurface, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsVDPAURegisterOutputSurface, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D11GetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D11GetDevices, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D11SetDirect3DDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsD3D11RegisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10GetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10GetDevices, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10SetDirect3DDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsD3D10RegisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10RegisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10UnregisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10MapResources, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10UnmapResources, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10ResourceSetMapFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10ResourceGetSurfaceDimensions, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10ResourceGetMappedArray, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10ResourceGetMappedPointer, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10ResourceGetMappedSize, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10ResourceGetMappedPitch, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9GetDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9GetDevices, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9SetDirect3DDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9GetDirect3DDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaGraphicsD3D9RegisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9RegisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9UnregisterResource, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9MapResources, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9UnmapResources, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9ResourceSetMapFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9ResourceGetSurfaceDimensions, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9ResourceGetMappedArray, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9ResourceGetMappedPointer, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9ResourceGetMappedSize, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9ResourceGetMappedPitch, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9Begin, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9End, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9RegisterVertexBuffer, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9UnregisterVertexBuffer, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9MapVertexBuffer, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D9UnmapVertexBuffer, v3020)
        CUPTI_RUNTIME_CASE(cudaThreadExit, v3020)
        CUPTI_RUNTIME_CASE(cudaSetDoubleForDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaSetDoubleForHost, v3020)
        CUPTI_RUNTIME_CASE(cudaThreadSynchronize, v3020)
        CUPTI_RUNTIME_CASE(cudaThreadGetLimit, v3020)
        CUPTI_RUNTIME_CASE(cudaThreadSetLimit, v3020)
        CUPTI_RUNTIME_CASE(cudaStreamCreate, v3020)
        CUPTI_RUNTIME_CASE(cudaStreamDestroy, v3020)
        CUPTI_RUNTIME_CASE(cudaStreamSynchronize, v3020)
        CUPTI_RUNTIME_CASE(cudaStreamQuery, v3020)
        CUPTI_RUNTIME_CASE(cudaEventCreate, v3020)
        CUPTI_RUNTIME_CASE(cudaEventCreateWithFlags, v3020)
        CUPTI_RUNTIME_CASE(cudaEventRecord, v3020)
        CUPTI_RUNTIME_CASE(cudaEventDestroy, v3020)
        CUPTI_RUNTIME_CASE(cudaEventSynchronize, v3020)
        CUPTI_RUNTIME_CASE(cudaEventQuery, v3020)
        CUPTI_RUNTIME_CASE(cudaEventElapsedTime, v3020)
        CUPTI_RUNTIME_CASE(cudaMalloc3D, v3020)
        CUPTI_RUNTIME_CASE(cudaMalloc3DArray, v3020)
        CUPTI_RUNTIME_CASE(cudaMemset3D, v3020)
        CUPTI_RUNTIME_CASE(cudaMemset3DAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy3D, v3020)
        CUPTI_RUNTIME_CASE(cudaMemcpy3DAsync, v3020)
        CUPTI_RUNTIME_CASE(cudaThreadSetCacheConfig, v3020)
        CUPTI_RUNTIME_CASE(cudaStreamWaitEvent, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D11GetDirect3DDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaD3D10GetDirect3DDevice, v3020)
        CUPTI_RUNTIME_CASE(cudaThreadGetCacheConfig, v3020)
        CUPTI_RUNTIME_CASE(cudaPointerGetAttributes, v4000)
        CUPTI_RUNTIME_CASE(cudaHostRegister, v4000)
        CUPTI_RUNTIME_CASE(cudaHostUnregister, v4000)
        CUPTI_RUNTIME_CASE(cudaDeviceCanAccessPeer, v4000)
        CUPTI_RUNTIME_CASE(cudaDeviceEnablePeerAccess, v4000)
        CUPTI_RUNTIME_CASE(cudaDeviceDisablePeerAccess, v4000)
        CUPTI_RUNTIME_CASE(cudaPeerRegister, v4000)
        CUPTI_RUNTIME_CASE(cudaPeerUnregister, v4000)
        CUPTI_RUNTIME_CASE(cudaPeerGetDevicePointer, v4000)
        CUPTI_RUNTIME_CASE(cudaMemcpyPeer, v4000)
        CUPTI_RUNTIME_CASE(cudaMemcpyPeerAsync, v4000)
        CUPTI_RUNTIME_CASE(cudaMemcpy3DPeer, v4000)
        CUPTI_RUNTIME_CASE(cudaMemcpy3DPeerAsync, v4000)
        CUPTI_RUNTIME_CASE(cudaDeviceReset, v3020)
        CUPTI_RUNTIME_CASE(cudaDeviceSynchronize, v3020)
        CUPTI_RUNTIME_CASE(cudaDeviceGetLimit, v3020)
        CUPTI_RUNTIME_CASE(cudaDeviceSetLimit, v3020)
        CUPTI_RUNTIME_CASE(cudaDeviceGetCacheConfig, v3020)
        CUPTI_RUNTIME_CASE(cudaDeviceSetCacheConfig, v3020)
        CUPTI_RUNTIME_CASE(cudaProfilerInitialize, v4000)
        CUPTI_RUNTIME_CASE(cudaProfilerStart, v4000)
        CUPTI_RUNTIME_CASE(cudaProfilerStop, v4000)
        CUPTI_RUNTIME_CASE(cudaDeviceGetByPCIBusId, v4010)
        CUPTI_RUNTIME_CASE(cudaDeviceGetPCIBusId, v4010)
        CUPTI_RUNTIME_CASE(cudaGLGetDevices, v4010)
        CUPTI_RUNTIME_CASE(cudaIpcGetEventHandle, v4010)
        CUPTI_RUNTIME_CASE(cudaIpcOpenEventHandle, v4010)
        CUPTI_RUNTIME_CASE(cudaIpcGetMemHandle, v4010)
        CUPTI_RUNTIME_CASE(cudaIpcOpenMemHandle, v4010)
        CUPTI_RUNTIME_CASE(cudaIpcCloseMemHandle, v4010)
        CUPTI_RUNTIME_CASE(cudaArrayGetInfo, v4010)
        CUPTI_RUNTIME_CASE(cudaFuncSetSharedMemConfig, v4020)
        CUPTI_RUNTIME_CASE(cudaDeviceGetSharedMemConfig, v4020)
        CUPTI_RUNTIME_CASE(cudaDeviceSetSharedMemConfig, v4020)
        CUPTI_RUNTIME_CASE(cudaCreateTextureObject, v5000)
        CUPTI_RUNTIME_CASE(cudaDestroyTextureObject, v5000)
        CUPTI_RUNTIME_CASE(cudaGetTextureObjectResourceDesc, v5000)
        CUPTI_RUNTIME_CASE(cudaGetTextureObjectTextureDesc, v5000)
        CUPTI_RUNTIME_CASE(cudaCreateSurfaceObject, v5000)
        CUPTI_RUNTIME_CASE(cudaDestroySurfaceObject, v5000)
        CUPTI_RUNTIME_CASE(cudaGetSurfaceObjectResourceDesc, v5000)
        CUPTI_RUNTIME_CASE(cudaMallocMipmappedArray, v5000)
        CUPTI_RUNTIME_CASE(cudaGetMipmappedArrayLevel, v5000)
        CUPTI_RUNTIME_CASE(cudaFreeMipmappedArray, v5000)
        CUPTI_RUNTIME_CASE(cudaBindTextureToMipmappedArray, v5000)
        CUPTI_RUNTIME_CASE(cudaGraphicsResourceGetMappedMipmappedArray, v5000)
        CUPTI_RUNTIME_CASE(cudaStreamAddCallback, v5000)
        CUPTI_RUNTIME_CASE(cudaStreamCreateWithFlags, v5000)
        CUPTI_RUNTIME_CASE(cudaGetTextureObjectResourceViewDesc, v5000)
        CUPTI_RUNTIME_CASE(cudaDeviceGetAttribute, v5000)
        CUPTI_RUNTIME_CASE(cudaStreamDestroy, v5050)
        CUPTI_RUNTIME_CASE(cudaStreamCreateWithPriority, v5050)
        CUPTI_RUNTIME_CASE(cudaStreamGetPriority, v5050)
        CUPTI_RUNTIME_CASE(cudaStreamGetFlags, v5050)
        CUPTI_RUNTIME_CASE(cudaDeviceGetStreamPriorityRange, v5050)
        CUPTI_RUNTIME_CASE(cudaMallocManaged, v6000)
        CUPTI_RUNTIME_CASE(cudaOccupancyMaxActiveBlocksPerMultiprocessor, v6000)
        CUPTI_RUNTIME_CASE(cudaStreamAttachMemAsync, v6000)
        CUPTI_RUNTIME_CASE(cudaGetErrorName, v6050)
        CUPTI_RUNTIME_CASE(cudaOccupancyMaxActiveBlocksPerMultiprocessor, v6050)
        CUPTI_RUNTIME_CASE(cudaLaunchKernel, v7000)
        CUPTI_RUNTIME_CASE(cudaGetDeviceFlags, v7000)
        CUPTI_RUNTIME_CASE(cudaLaunch_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaLaunchKernel_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2D_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyToArray_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DToArray_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromArray_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DFromArray_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyArrayToArray_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DArrayToArray_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyToSymbol_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromSymbol_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyToArrayAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromArrayAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DToArrayAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy2DFromArrayAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyToSymbolAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpyFromSymbolAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemset_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemset2D_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemsetAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemset2DAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamGetPriority_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamGetFlags_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamSynchronize_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamQuery_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamAttachMemAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaEventRecord_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemset3D_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemset3DAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy3D_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy3DAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamWaitEvent_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaStreamAddCallback_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy3DPeer_ptds, v7000)
        CUPTI_RUNTIME_CASE(cudaMemcpy3DPeerAsync_ptsz, v7000)
        CUPTI_RUNTIME_CASE(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, v7000)
        CUPTI_RUNTIME_CASE(cudaMemPrefetchAsync, v8000)
        CUPTI_RUNTIME_CASE(cudaMemPrefetchAsync_ptsz, v8000)
        CUPTI_RUNTIME_CASE(cudaMemAdvise, v8000)
        CUPTI_RUNTIME_CASE(cudaDeviceGetP2PAttribute, v8000)
        CUPTI_RUNTIME_CASE(cudaGraphicsEGLRegisterImage, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamConsumerConnect, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamConsumerDisconnect, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamConsumerAcquireFrame, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamConsumerReleaseFrame, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamProducerConnect, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamProducerDisconnect, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamProducerPresentFrame, v7000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamProducerReturnFrame, v7000)
        CUPTI_RUNTIME_CASE(cudaGraphicsResourceGetMappedEglFrame, v7000)
        CUPTI_RUNTIME_CASE(cudaMemRangeGetAttribute, v8000)
        CUPTI_RUNTIME_CASE(cudaMemRangeGetAttributes, v8000)
        CUPTI_RUNTIME_CASE(cudaEGLStreamConsumerConnectWithFlags, v7000)
        CUPTI_RUNTIME_CASE(cudaLaunchCooperativeKernel, v9000)
        CUPTI_RUNTIME_CASE(cudaLaunchCooperativeKernel_ptsz, v9000)
        CUPTI_RUNTIME_CASE(cudaEventCreateFromEGLSync, v9000)
        CUPTI_RUNTIME_CASE(cudaLaunchCooperativeKernelMultiDevice, v9000)
        CUPTI_RUNTIME_CASE(cudaFuncSetAttribute, v9000)
        CUPTI_RUNTIME_CASE(cudaImportExternalMemory, v10000)
        CUPTI_RUNTIME_CASE(cudaExternalMemoryGetMappedBuffer, v10000)
        CUPTI_RUNTIME_CASE(cudaExternalMemoryGetMappedMipmappedArray, v10000)
        CUPTI_RUNTIME_CASE(cudaDestroyExternalMemory, v10000)
        CUPTI_RUNTIME_CASE(cudaImportExternalSemaphore, v10000)
        CUPTI_RUNTIME_CASE(cudaSignalExternalSemaphoresAsync, v10000)
        CUPTI_RUNTIME_CASE(cudaSignalExternalSemaphoresAsync_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaWaitExternalSemaphoresAsync, v10000)
        CUPTI_RUNTIME_CASE(cudaWaitExternalSemaphoresAsync_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaDestroyExternalSemaphore, v10000)
        CUPTI_RUNTIME_CASE(cudaLaunchHostFunc, v10000)
        CUPTI_RUNTIME_CASE(cudaLaunchHostFunc_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphCreate, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphKernelNodeGetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphKernelNodeSetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddKernelNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddMemcpyNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphMemcpyNodeGetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphMemcpyNodeSetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddMemsetNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphMemsetNodeGetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphMemsetNodeSetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddHostNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphHostNodeGetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddChildGraphNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphChildGraphNodeGetGraph, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddEmptyNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphClone, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphNodeFindInClone, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphNodeGetType, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphGetRootNodes, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphNodeGetDependencies, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphNodeGetDependentNodes, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphAddDependencies, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphRemoveDependencies, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphDestroyNode, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphInstantiate, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphLaunch, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphLaunch_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphExecDestroy, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphDestroy, v10000)
        CUPTI_RUNTIME_CASE(cudaStreamBeginCapture, v10000)
        CUPTI_RUNTIME_CASE(cudaStreamBeginCapture_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaStreamIsCapturing, v10000)
        CUPTI_RUNTIME_CASE(cudaStreamIsCapturing_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaStreamEndCapture, v10000)
        CUPTI_RUNTIME_CASE(cudaStreamEndCapture_ptsz, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphHostNodeSetParams, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphGetNodes, v10000)
        CUPTI_RUNTIME_CASE(cudaGraphGetEdges, v10000)
        default:
            return "INVALID";
    }
#undef CUPTI_RUNTIME_CASE
}


const char* driver_cbid_to_string(CUpti_CallbackId cbid) {

#define CUPTI_DRIVER_CASE(apiName) \
    case CUPTI_DRIVER_TRACE_CBID_ ## apiName: \
        return STR(apiName);

    switch (cbid) {
        CUPTI_DRIVER_CASE(cuInit)
        CUPTI_DRIVER_CASE(cuDriverGetVersion)
        CUPTI_DRIVER_CASE(cuDeviceGet)
        CUPTI_DRIVER_CASE(cuDeviceGetCount)
        CUPTI_DRIVER_CASE(cuDeviceGetName)
        CUPTI_DRIVER_CASE(cuDeviceComputeCapability)
        CUPTI_DRIVER_CASE(cuDeviceTotalMem)
        CUPTI_DRIVER_CASE(cuDeviceGetProperties)
        CUPTI_DRIVER_CASE(cuDeviceGetAttribute)
        CUPTI_DRIVER_CASE(cuCtxCreate)
        CUPTI_DRIVER_CASE(cuCtxDestroy)
        CUPTI_DRIVER_CASE(cuCtxAttach)
        CUPTI_DRIVER_CASE(cuCtxDetach)
        CUPTI_DRIVER_CASE(cuCtxPushCurrent)
        CUPTI_DRIVER_CASE(cuCtxPopCurrent)
        CUPTI_DRIVER_CASE(cuCtxGetDevice)
        CUPTI_DRIVER_CASE(cuCtxSynchronize)
        CUPTI_DRIVER_CASE(cuModuleLoad)
        CUPTI_DRIVER_CASE(cuModuleLoadData)
        CUPTI_DRIVER_CASE(cuModuleLoadDataEx)
        CUPTI_DRIVER_CASE(cuModuleLoadFatBinary)
        CUPTI_DRIVER_CASE(cuModuleUnload)
        CUPTI_DRIVER_CASE(cuModuleGetFunction)
        CUPTI_DRIVER_CASE(cuModuleGetGlobal)
        CUPTI_DRIVER_CASE(cu64ModuleGetGlobal)
        CUPTI_DRIVER_CASE(cuModuleGetTexRef)
        CUPTI_DRIVER_CASE(cuMemGetInfo)
        CUPTI_DRIVER_CASE(cu64MemGetInfo)
        CUPTI_DRIVER_CASE(cuMemAlloc)
        CUPTI_DRIVER_CASE(cu64MemAlloc)
        CUPTI_DRIVER_CASE(cuMemAllocPitch)
        CUPTI_DRIVER_CASE(cu64MemAllocPitch)
        CUPTI_DRIVER_CASE(cuMemFree)
        CUPTI_DRIVER_CASE(cu64MemFree)
        CUPTI_DRIVER_CASE(cuMemGetAddressRange)
        CUPTI_DRIVER_CASE(cu64MemGetAddressRange)
        CUPTI_DRIVER_CASE(cuMemAllocHost)
        CUPTI_DRIVER_CASE(cuMemFreeHost)
        CUPTI_DRIVER_CASE(cuMemHostAlloc)
        CUPTI_DRIVER_CASE(cuMemHostGetDevicePointer)
        CUPTI_DRIVER_CASE(cu64MemHostGetDevicePointer)
        CUPTI_DRIVER_CASE(cuMemHostGetFlags)
        CUPTI_DRIVER_CASE(cuMemcpyHtoD)
        CUPTI_DRIVER_CASE(cu64MemcpyHtoD)
        CUPTI_DRIVER_CASE(cuMemcpyDtoH)
        CUPTI_DRIVER_CASE(cu64MemcpyDtoH)
        CUPTI_DRIVER_CASE(cuMemcpyDtoD)
        CUPTI_DRIVER_CASE(cu64MemcpyDtoD)
        CUPTI_DRIVER_CASE(cuMemcpyDtoA)
        CUPTI_DRIVER_CASE(cu64MemcpyDtoA)
        CUPTI_DRIVER_CASE(cuMemcpyAtoD)
        CUPTI_DRIVER_CASE(cu64MemcpyAtoD)
        CUPTI_DRIVER_CASE(cuMemcpyHtoA)
        CUPTI_DRIVER_CASE(cuMemcpyAtoH)
        CUPTI_DRIVER_CASE(cuMemcpyAtoA)
        CUPTI_DRIVER_CASE(cuMemcpy2D)
        CUPTI_DRIVER_CASE(cuMemcpy2DUnaligned)
        CUPTI_DRIVER_CASE(cuMemcpy3D)
        CUPTI_DRIVER_CASE(cu64Memcpy3D)
        CUPTI_DRIVER_CASE(cuMemcpyHtoDAsync)
        CUPTI_DRIVER_CASE(cu64MemcpyHtoDAsync)
        CUPTI_DRIVER_CASE(cuMemcpyDtoHAsync)
        CUPTI_DRIVER_CASE(cu64MemcpyDtoHAsync)
        CUPTI_DRIVER_CASE(cuMemcpyDtoDAsync)
        CUPTI_DRIVER_CASE(cu64MemcpyDtoDAsync)
        CUPTI_DRIVER_CASE(cuMemcpyHtoAAsync)
        CUPTI_DRIVER_CASE(cuMemcpyAtoHAsync)
        CUPTI_DRIVER_CASE(cuMemcpy2DAsync)
        CUPTI_DRIVER_CASE(cuMemcpy3DAsync)
        CUPTI_DRIVER_CASE(cu64Memcpy3DAsync)
        CUPTI_DRIVER_CASE(cuMemsetD8)
        CUPTI_DRIVER_CASE(cu64MemsetD8)
        CUPTI_DRIVER_CASE(cuMemsetD16)
        CUPTI_DRIVER_CASE(cu64MemsetD16)
        CUPTI_DRIVER_CASE(cuMemsetD32)
        CUPTI_DRIVER_CASE(cu64MemsetD32)
        CUPTI_DRIVER_CASE(cuMemsetD2D8)
        CUPTI_DRIVER_CASE(cu64MemsetD2D8)
        CUPTI_DRIVER_CASE(cuMemsetD2D16)
        CUPTI_DRIVER_CASE(cu64MemsetD2D16)
        CUPTI_DRIVER_CASE(cuMemsetD2D32)
        CUPTI_DRIVER_CASE(cu64MemsetD2D32)
        CUPTI_DRIVER_CASE(cuFuncSetBlockShape)
        CUPTI_DRIVER_CASE(cuFuncSetSharedSize)
        CUPTI_DRIVER_CASE(cuFuncGetAttribute)
        CUPTI_DRIVER_CASE(cuFuncSetCacheConfig)
        CUPTI_DRIVER_CASE(cuArrayCreate)
        CUPTI_DRIVER_CASE(cuArrayGetDescriptor)
        CUPTI_DRIVER_CASE(cuArrayDestroy)
        CUPTI_DRIVER_CASE(cuArray3DCreate)
        CUPTI_DRIVER_CASE(cuArray3DGetDescriptor)
        CUPTI_DRIVER_CASE(cuTexRefCreate)
        CUPTI_DRIVER_CASE(cuTexRefDestroy)
        CUPTI_DRIVER_CASE(cuTexRefSetArray)
        CUPTI_DRIVER_CASE(cuTexRefSetAddress)
        CUPTI_DRIVER_CASE(cu64TexRefSetAddress)
        CUPTI_DRIVER_CASE(cuTexRefSetAddress2D)
        CUPTI_DRIVER_CASE(cu64TexRefSetAddress2D)
        CUPTI_DRIVER_CASE(cuTexRefSetFormat)
        CUPTI_DRIVER_CASE(cuTexRefSetAddressMode)
        CUPTI_DRIVER_CASE(cuTexRefSetFilterMode)
        CUPTI_DRIVER_CASE(cuTexRefSetFlags)
        CUPTI_DRIVER_CASE(cuTexRefGetAddress)
        CUPTI_DRIVER_CASE(cu64TexRefGetAddress)
        CUPTI_DRIVER_CASE(cuTexRefGetArray)
        CUPTI_DRIVER_CASE(cuTexRefGetAddressMode)
        CUPTI_DRIVER_CASE(cuTexRefGetFilterMode)
        CUPTI_DRIVER_CASE(cuTexRefGetFormat)
        CUPTI_DRIVER_CASE(cuTexRefGetFlags)
        CUPTI_DRIVER_CASE(cuParamSetSize)
        CUPTI_DRIVER_CASE(cuParamSeti)
        CUPTI_DRIVER_CASE(cuParamSetf)
        CUPTI_DRIVER_CASE(cuParamSetv)
        CUPTI_DRIVER_CASE(cuParamSetTexRef)
        CUPTI_DRIVER_CASE(cuLaunch)
        CUPTI_DRIVER_CASE(cuLaunchGrid)
        CUPTI_DRIVER_CASE(cuLaunchGridAsync)
        CUPTI_DRIVER_CASE(cuEventCreate)
        CUPTI_DRIVER_CASE(cuEventRecord)
        CUPTI_DRIVER_CASE(cuEventQuery)
        CUPTI_DRIVER_CASE(cuEventSynchronize)
        CUPTI_DRIVER_CASE(cuEventDestroy)
        CUPTI_DRIVER_CASE(cuEventElapsedTime)
        CUPTI_DRIVER_CASE(cuStreamCreate)
        CUPTI_DRIVER_CASE(cuStreamQuery)
        CUPTI_DRIVER_CASE(cuStreamSynchronize)
        CUPTI_DRIVER_CASE(cuStreamDestroy)
        CUPTI_DRIVER_CASE(cuGraphicsUnregisterResource)
        CUPTI_DRIVER_CASE(cuGraphicsSubResourceGetMappedArray)
        CUPTI_DRIVER_CASE(cuGraphicsResourceGetMappedPointer)
        CUPTI_DRIVER_CASE(cu64GraphicsResourceGetMappedPointer)
        CUPTI_DRIVER_CASE(cuGraphicsResourceSetMapFlags)
        CUPTI_DRIVER_CASE(cuGraphicsMapResources)
        CUPTI_DRIVER_CASE(cuGraphicsUnmapResources)
        CUPTI_DRIVER_CASE(cuGetExportTable)
        CUPTI_DRIVER_CASE(cuCtxSetLimit)
        CUPTI_DRIVER_CASE(cuCtxGetLimit)
        CUPTI_DRIVER_CASE(cuD3D10GetDevice)
        CUPTI_DRIVER_CASE(cuD3D10CtxCreate)
        CUPTI_DRIVER_CASE(cuGraphicsD3D10RegisterResource)
        CUPTI_DRIVER_CASE(cuD3D10RegisterResource)
        CUPTI_DRIVER_CASE(cuD3D10UnregisterResource)
        CUPTI_DRIVER_CASE(cuD3D10MapResources)
        CUPTI_DRIVER_CASE(cuD3D10UnmapResources)
        CUPTI_DRIVER_CASE(cuD3D10ResourceSetMapFlags)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedArray)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedPointer)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedSize)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedPitch)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetSurfaceDimensions)
        CUPTI_DRIVER_CASE(cuD3D11GetDevice)
        CUPTI_DRIVER_CASE(cuD3D11CtxCreate)
        CUPTI_DRIVER_CASE(cuGraphicsD3D11RegisterResource)
        CUPTI_DRIVER_CASE(cuD3D9GetDevice)
        CUPTI_DRIVER_CASE(cuD3D9CtxCreate)
        CUPTI_DRIVER_CASE(cuGraphicsD3D9RegisterResource)
        CUPTI_DRIVER_CASE(cuD3D9GetDirect3DDevice)
        CUPTI_DRIVER_CASE(cuD3D9RegisterResource)
        CUPTI_DRIVER_CASE(cuD3D9UnregisterResource)
        CUPTI_DRIVER_CASE(cuD3D9MapResources)
        CUPTI_DRIVER_CASE(cuD3D9UnmapResources)
        CUPTI_DRIVER_CASE(cuD3D9ResourceSetMapFlags)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetSurfaceDimensions)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedArray)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedPointer)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedSize)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedPitch)
        CUPTI_DRIVER_CASE(cuD3D9Begin)
        CUPTI_DRIVER_CASE(cuD3D9End)
        CUPTI_DRIVER_CASE(cuD3D9RegisterVertexBuffer)
        CUPTI_DRIVER_CASE(cuD3D9MapVertexBuffer)
        CUPTI_DRIVER_CASE(cuD3D9UnmapVertexBuffer)
        CUPTI_DRIVER_CASE(cuD3D9UnregisterVertexBuffer)
        CUPTI_DRIVER_CASE(cuGLCtxCreate)
        CUPTI_DRIVER_CASE(cuGraphicsGLRegisterBuffer)
        CUPTI_DRIVER_CASE(cuGraphicsGLRegisterImage)
        CUPTI_DRIVER_CASE(cuWGLGetDevice)
        CUPTI_DRIVER_CASE(cuGLInit)
        CUPTI_DRIVER_CASE(cuGLRegisterBufferObject)
        CUPTI_DRIVER_CASE(cuGLMapBufferObject)
        CUPTI_DRIVER_CASE(cuGLUnmapBufferObject)
        CUPTI_DRIVER_CASE(cuGLUnregisterBufferObject)
        CUPTI_DRIVER_CASE(cuGLSetBufferObjectMapFlags)
        CUPTI_DRIVER_CASE(cuGLMapBufferObjectAsync)
        CUPTI_DRIVER_CASE(cuGLUnmapBufferObjectAsync)
        CUPTI_DRIVER_CASE(cuVDPAUGetDevice)
        CUPTI_DRIVER_CASE(cuVDPAUCtxCreate)
        CUPTI_DRIVER_CASE(cuGraphicsVDPAURegisterVideoSurface)
        CUPTI_DRIVER_CASE(cuGraphicsVDPAURegisterOutputSurface)
        CUPTI_DRIVER_CASE(cuModuleGetSurfRef)
        CUPTI_DRIVER_CASE(cuSurfRefCreate)
        CUPTI_DRIVER_CASE(cuSurfRefDestroy)
        CUPTI_DRIVER_CASE(cuSurfRefSetFormat)
        CUPTI_DRIVER_CASE(cuSurfRefSetArray)
        CUPTI_DRIVER_CASE(cuSurfRefGetFormat)
        CUPTI_DRIVER_CASE(cuSurfRefGetArray)
        CUPTI_DRIVER_CASE(cu64DeviceTotalMem)
        CUPTI_DRIVER_CASE(cu64D3D10ResourceGetMappedPointer)
        CUPTI_DRIVER_CASE(cu64D3D10ResourceGetMappedSize)
        CUPTI_DRIVER_CASE(cu64D3D10ResourceGetMappedPitch)
        CUPTI_DRIVER_CASE(cu64D3D10ResourceGetSurfaceDimensions)
        CUPTI_DRIVER_CASE(cu64D3D9ResourceGetSurfaceDimensions)
        CUPTI_DRIVER_CASE(cu64D3D9ResourceGetMappedPointer)
        CUPTI_DRIVER_CASE(cu64D3D9ResourceGetMappedSize)
        CUPTI_DRIVER_CASE(cu64D3D9ResourceGetMappedPitch)
        CUPTI_DRIVER_CASE(cu64D3D9MapVertexBuffer)
        CUPTI_DRIVER_CASE(cu64GLMapBufferObject)
        CUPTI_DRIVER_CASE(cu64GLMapBufferObjectAsync)
        CUPTI_DRIVER_CASE(cuD3D11GetDevices)
        CUPTI_DRIVER_CASE(cuD3D11CtxCreateOnDevice)
        CUPTI_DRIVER_CASE(cuD3D10GetDevices)
        CUPTI_DRIVER_CASE(cuD3D10CtxCreateOnDevice)
        CUPTI_DRIVER_CASE(cuD3D9GetDevices)
        CUPTI_DRIVER_CASE(cuD3D9CtxCreateOnDevice)
        CUPTI_DRIVER_CASE(cu64MemHostAlloc)
        CUPTI_DRIVER_CASE(cuMemsetD8Async)
        CUPTI_DRIVER_CASE(cu64MemsetD8Async)
        CUPTI_DRIVER_CASE(cuMemsetD16Async)
        CUPTI_DRIVER_CASE(cu64MemsetD16Async)
        CUPTI_DRIVER_CASE(cuMemsetD32Async)
        CUPTI_DRIVER_CASE(cu64MemsetD32Async)
        CUPTI_DRIVER_CASE(cuMemsetD2D8Async)
        CUPTI_DRIVER_CASE(cu64MemsetD2D8Async)
        CUPTI_DRIVER_CASE(cuMemsetD2D16Async)
        CUPTI_DRIVER_CASE(cu64MemsetD2D16Async)
        CUPTI_DRIVER_CASE(cuMemsetD2D32Async)
        CUPTI_DRIVER_CASE(cu64MemsetD2D32Async)
        CUPTI_DRIVER_CASE(cu64ArrayCreate)
        CUPTI_DRIVER_CASE(cu64ArrayGetDescriptor)
        CUPTI_DRIVER_CASE(cu64Array3DCreate)
        CUPTI_DRIVER_CASE(cu64Array3DGetDescriptor)
        CUPTI_DRIVER_CASE(cu64Memcpy2D)
        CUPTI_DRIVER_CASE(cu64Memcpy2DUnaligned)
        CUPTI_DRIVER_CASE(cu64Memcpy2DAsync)
        CUPTI_DRIVER_CASE(cuCtxCreate_v2)
        CUPTI_DRIVER_CASE(cuD3D10CtxCreate_v2)
        CUPTI_DRIVER_CASE(cuD3D11CtxCreate_v2)
        CUPTI_DRIVER_CASE(cuD3D9CtxCreate_v2)
        CUPTI_DRIVER_CASE(cuGLCtxCreate_v2)
        CUPTI_DRIVER_CASE(cuVDPAUCtxCreate_v2)
        CUPTI_DRIVER_CASE(cuModuleGetGlobal_v2)
        CUPTI_DRIVER_CASE(cuMemGetInfo_v2)
        CUPTI_DRIVER_CASE(cuMemAlloc_v2)
        CUPTI_DRIVER_CASE(cuMemAllocPitch_v2)
        CUPTI_DRIVER_CASE(cuMemFree_v2)
        CUPTI_DRIVER_CASE(cuMemGetAddressRange_v2)
        CUPTI_DRIVER_CASE(cuMemHostGetDevicePointer_v2)
        CUPTI_DRIVER_CASE(cuMemcpy_v2)
        CUPTI_DRIVER_CASE(cuMemsetD8_v2)
        CUPTI_DRIVER_CASE(cuMemsetD16_v2)
        CUPTI_DRIVER_CASE(cuMemsetD32_v2)
        CUPTI_DRIVER_CASE(cuMemsetD2D8_v2)
        CUPTI_DRIVER_CASE(cuMemsetD2D16_v2)
        CUPTI_DRIVER_CASE(cuMemsetD2D32_v2)
        CUPTI_DRIVER_CASE(cuTexRefSetAddress_v2)
        CUPTI_DRIVER_CASE(cuTexRefSetAddress2D_v2)
        CUPTI_DRIVER_CASE(cuTexRefGetAddress_v2)
        CUPTI_DRIVER_CASE(cuGraphicsResourceGetMappedPointer_v2)
        CUPTI_DRIVER_CASE(cuDeviceTotalMem_v2)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedPointer_v2)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedSize_v2)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetMappedPitch_v2)
        CUPTI_DRIVER_CASE(cuD3D10ResourceGetSurfaceDimensions_v2)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetSurfaceDimensions_v2)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedPointer_v2)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedSize_v2)
        CUPTI_DRIVER_CASE(cuD3D9ResourceGetMappedPitch_v2)
        CUPTI_DRIVER_CASE(cuD3D9MapVertexBuffer_v2)
        CUPTI_DRIVER_CASE(cuGLMapBufferObject_v2)
        CUPTI_DRIVER_CASE(cuGLMapBufferObjectAsync_v2)
        CUPTI_DRIVER_CASE(cuMemHostAlloc_v2)
        CUPTI_DRIVER_CASE(cuArrayCreate_v2)
        CUPTI_DRIVER_CASE(cuArrayGetDescriptor_v2)
        CUPTI_DRIVER_CASE(cuArray3DCreate_v2)
        CUPTI_DRIVER_CASE(cuArray3DGetDescriptor_v2)
        CUPTI_DRIVER_CASE(cuMemcpyHtoD_v2)
        CUPTI_DRIVER_CASE(cuMemcpyHtoDAsync_v2)
        CUPTI_DRIVER_CASE(cuMemcpyDtoH_v2)
        CUPTI_DRIVER_CASE(cuMemcpyDtoHAsync_v2)
        CUPTI_DRIVER_CASE(cuMemcpyDtoD_v2)
        CUPTI_DRIVER_CASE(cuMemcpyDtoDAsync_v2)
        CUPTI_DRIVER_CASE(cuMemcpyAtoH_v2)
        CUPTI_DRIVER_CASE(cuMemcpyAtoHAsync_v2)
        CUPTI_DRIVER_CASE(cuMemcpyAtoD_v2)
        CUPTI_DRIVER_CASE(cuMemcpyDtoA_v2)
        CUPTI_DRIVER_CASE(cuMemcpyAtoA_v2)
        CUPTI_DRIVER_CASE(cuMemcpy2D_v2)
        CUPTI_DRIVER_CASE(cuMemcpy2DUnaligned_v2)
        CUPTI_DRIVER_CASE(cuMemcpy2DAsync_v2)
        CUPTI_DRIVER_CASE(cuMemcpy3D_v2)
        CUPTI_DRIVER_CASE(cuMemcpy3DAsync_v2)
        CUPTI_DRIVER_CASE(cuMemcpyHtoA_v2)
        CUPTI_DRIVER_CASE(cuMemcpyHtoAAsync_v2)
        CUPTI_DRIVER_CASE(cuMemAllocHost_v2)
        CUPTI_DRIVER_CASE(cuStreamWaitEvent)
        CUPTI_DRIVER_CASE(cuCtxGetApiVersion)
        CUPTI_DRIVER_CASE(cuD3D10GetDirect3DDevice)
        CUPTI_DRIVER_CASE(cuD3D11GetDirect3DDevice)
        CUPTI_DRIVER_CASE(cuCtxGetCacheConfig)
        CUPTI_DRIVER_CASE(cuCtxSetCacheConfig)
        CUPTI_DRIVER_CASE(cuMemHostRegister)
        CUPTI_DRIVER_CASE(cuMemHostUnregister)
        CUPTI_DRIVER_CASE(cuCtxSetCurrent)
        CUPTI_DRIVER_CASE(cuCtxGetCurrent)
        CUPTI_DRIVER_CASE(cuMemcpy)
        CUPTI_DRIVER_CASE(cuMemcpyAsync)
        CUPTI_DRIVER_CASE(cuLaunchKernel)
        CUPTI_DRIVER_CASE(cuProfilerStart)
        CUPTI_DRIVER_CASE(cuProfilerStop)
        CUPTI_DRIVER_CASE(cuPointerGetAttribute)
        CUPTI_DRIVER_CASE(cuProfilerInitialize)
        CUPTI_DRIVER_CASE(cuDeviceCanAccessPeer)
        CUPTI_DRIVER_CASE(cuCtxEnablePeerAccess)
        CUPTI_DRIVER_CASE(cuCtxDisablePeerAccess)
        CUPTI_DRIVER_CASE(cuMemPeerRegister)
        CUPTI_DRIVER_CASE(cuMemPeerUnregister)
        CUPTI_DRIVER_CASE(cuMemPeerGetDevicePointer)
        CUPTI_DRIVER_CASE(cuMemcpyPeer)
        CUPTI_DRIVER_CASE(cuMemcpyPeerAsync)
        CUPTI_DRIVER_CASE(cuMemcpy3DPeer)
        CUPTI_DRIVER_CASE(cuMemcpy3DPeerAsync)
        CUPTI_DRIVER_CASE(cuCtxDestroy_v2)
        CUPTI_DRIVER_CASE(cuCtxPushCurrent_v2)
        CUPTI_DRIVER_CASE(cuCtxPopCurrent_v2)
        CUPTI_DRIVER_CASE(cuEventDestroy_v2)
        CUPTI_DRIVER_CASE(cuStreamDestroy_v2)
        CUPTI_DRIVER_CASE(cuTexRefSetAddress2D_v3)
        CUPTI_DRIVER_CASE(cuIpcGetMemHandle)
        CUPTI_DRIVER_CASE(cuIpcOpenMemHandle)
        CUPTI_DRIVER_CASE(cuIpcCloseMemHandle)
        CUPTI_DRIVER_CASE(cuDeviceGetByPCIBusId)
        CUPTI_DRIVER_CASE(cuDeviceGetPCIBusId)
        CUPTI_DRIVER_CASE(cuGLGetDevices)
        CUPTI_DRIVER_CASE(cuIpcGetEventHandle)
        CUPTI_DRIVER_CASE(cuIpcOpenEventHandle)
        CUPTI_DRIVER_CASE(cuCtxSetSharedMemConfig)
        CUPTI_DRIVER_CASE(cuCtxGetSharedMemConfig)
        CUPTI_DRIVER_CASE(cuFuncSetSharedMemConfig)
        CUPTI_DRIVER_CASE(cuTexObjectCreate)
        CUPTI_DRIVER_CASE(cuTexObjectDestroy)
        CUPTI_DRIVER_CASE(cuTexObjectGetResourceDesc)
        CUPTI_DRIVER_CASE(cuTexObjectGetTextureDesc)
        CUPTI_DRIVER_CASE(cuSurfObjectCreate)
        CUPTI_DRIVER_CASE(cuSurfObjectDestroy)
        CUPTI_DRIVER_CASE(cuSurfObjectGetResourceDesc)
        CUPTI_DRIVER_CASE(cuStreamAddCallback)
        CUPTI_DRIVER_CASE(cuMipmappedArrayCreate)
        CUPTI_DRIVER_CASE(cuMipmappedArrayGetLevel)
        CUPTI_DRIVER_CASE(cuMipmappedArrayDestroy)
        CUPTI_DRIVER_CASE(cuTexRefSetMipmappedArray)
        CUPTI_DRIVER_CASE(cuTexRefSetMipmapFilterMode)
        CUPTI_DRIVER_CASE(cuTexRefSetMipmapLevelBias)
        CUPTI_DRIVER_CASE(cuTexRefSetMipmapLevelClamp)
        CUPTI_DRIVER_CASE(cuTexRefSetMaxAnisotropy)
        CUPTI_DRIVER_CASE(cuTexRefGetMipmappedArray)
        CUPTI_DRIVER_CASE(cuTexRefGetMipmapFilterMode)
        CUPTI_DRIVER_CASE(cuTexRefGetMipmapLevelBias)
        CUPTI_DRIVER_CASE(cuTexRefGetMipmapLevelClamp)
        CUPTI_DRIVER_CASE(cuTexRefGetMaxAnisotropy)
        CUPTI_DRIVER_CASE(cuGraphicsResourceGetMappedMipmappedArray)
        CUPTI_DRIVER_CASE(cuTexObjectGetResourceViewDesc)
        CUPTI_DRIVER_CASE(cuLinkCreate)
        CUPTI_DRIVER_CASE(cuLinkAddData)
        CUPTI_DRIVER_CASE(cuLinkAddFile)
        CUPTI_DRIVER_CASE(cuLinkComplete)
        CUPTI_DRIVER_CASE(cuLinkDestroy)
        CUPTI_DRIVER_CASE(cuStreamCreateWithPriority)
        CUPTI_DRIVER_CASE(cuStreamGetPriority)
        CUPTI_DRIVER_CASE(cuStreamGetFlags)
        CUPTI_DRIVER_CASE(cuCtxGetStreamPriorityRange)
        CUPTI_DRIVER_CASE(cuMemAllocManaged)
        CUPTI_DRIVER_CASE(cuGetErrorString)
        CUPTI_DRIVER_CASE(cuGetErrorName)
        CUPTI_DRIVER_CASE(cuOccupancyMaxActiveBlocksPerMultiprocessor)
        CUPTI_DRIVER_CASE(cuCompilePtx)
        CUPTI_DRIVER_CASE(cuBinaryFree)
        CUPTI_DRIVER_CASE(cuStreamAttachMemAsync)
        CUPTI_DRIVER_CASE(cuPointerSetAttribute)
        CUPTI_DRIVER_CASE(cuMemHostRegister_v2)
        CUPTI_DRIVER_CASE(cuGraphicsResourceSetMapFlags_v2)
        CUPTI_DRIVER_CASE(cuLinkCreate_v2)
        CUPTI_DRIVER_CASE(cuLinkAddData_v2)
        CUPTI_DRIVER_CASE(cuLinkAddFile_v2)
        CUPTI_DRIVER_CASE(cuOccupancyMaxPotentialBlockSize)
        CUPTI_DRIVER_CASE(cuGLGetDevices_v2)
        CUPTI_DRIVER_CASE(cuDevicePrimaryCtxRetain)
        CUPTI_DRIVER_CASE(cuDevicePrimaryCtxRelease)
        CUPTI_DRIVER_CASE(cuDevicePrimaryCtxSetFlags)
        CUPTI_DRIVER_CASE(cuDevicePrimaryCtxReset)
        CUPTI_DRIVER_CASE(cuGraphicsEGLRegisterImage)
        CUPTI_DRIVER_CASE(cuCtxGetFlags)
        CUPTI_DRIVER_CASE(cuDevicePrimaryCtxGetState)
        CUPTI_DRIVER_CASE(cuEGLStreamConsumerConnect)
        CUPTI_DRIVER_CASE(cuEGLStreamConsumerDisconnect)
        CUPTI_DRIVER_CASE(cuEGLStreamConsumerAcquireFrame)
        CUPTI_DRIVER_CASE(cuEGLStreamConsumerReleaseFrame)
        CUPTI_DRIVER_CASE(cuMemcpyHtoD_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyDtoH_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyDtoD_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyDtoA_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyAtoD_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyHtoA_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyAtoH_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyAtoA_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpy2D_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpy2DUnaligned_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpy3D_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpy_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyPeer_ptds)
        CUPTI_DRIVER_CASE(cuMemcpy3DPeer_ptds)
        CUPTI_DRIVER_CASE(cuMemsetD8_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemsetD16_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemsetD32_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemsetD2D8_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemsetD2D16_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemsetD2D32_v2_ptds)
        CUPTI_DRIVER_CASE(cuGLMapBufferObject_v2_ptds)
        CUPTI_DRIVER_CASE(cuMemcpyAsync_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpyHtoAAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpyAtoHAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpyHtoDAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpyDtoHAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpyDtoDAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpy2DAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpy3DAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpyPeerAsync_ptsz)
        CUPTI_DRIVER_CASE(cuMemcpy3DPeerAsync_ptsz)
        CUPTI_DRIVER_CASE(cuMemsetD8Async_ptsz)
        CUPTI_DRIVER_CASE(cuMemsetD16Async_ptsz)
        CUPTI_DRIVER_CASE(cuMemsetD32Async_ptsz)
        CUPTI_DRIVER_CASE(cuMemsetD2D8Async_ptsz)
        CUPTI_DRIVER_CASE(cuMemsetD2D16Async_ptsz)
        CUPTI_DRIVER_CASE(cuMemsetD2D32Async_ptsz)
        CUPTI_DRIVER_CASE(cuStreamGetPriority_ptsz)
        CUPTI_DRIVER_CASE(cuStreamGetFlags_ptsz)
        CUPTI_DRIVER_CASE(cuStreamWaitEvent_ptsz)
        CUPTI_DRIVER_CASE(cuStreamAddCallback_ptsz)
        CUPTI_DRIVER_CASE(cuStreamAttachMemAsync_ptsz)
        CUPTI_DRIVER_CASE(cuStreamQuery_ptsz)
        CUPTI_DRIVER_CASE(cuStreamSynchronize_ptsz)
        CUPTI_DRIVER_CASE(cuEventRecord_ptsz)
        CUPTI_DRIVER_CASE(cuLaunchKernel_ptsz)
        CUPTI_DRIVER_CASE(cuGraphicsMapResources_ptsz)
        CUPTI_DRIVER_CASE(cuGraphicsUnmapResources_ptsz)
        CUPTI_DRIVER_CASE(cuGLMapBufferObjectAsync_v2_ptsz)
        CUPTI_DRIVER_CASE(cuEGLStreamProducerConnect)
        CUPTI_DRIVER_CASE(cuEGLStreamProducerDisconnect)
        CUPTI_DRIVER_CASE(cuEGLStreamProducerPresentFrame)
        CUPTI_DRIVER_CASE(cuGraphicsResourceGetMappedEglFrame)
        CUPTI_DRIVER_CASE(cuPointerGetAttributes)
        CUPTI_DRIVER_CASE(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
        CUPTI_DRIVER_CASE(cuOccupancyMaxPotentialBlockSizeWithFlags)
        CUPTI_DRIVER_CASE(cuEGLStreamProducerReturnFrame)
        CUPTI_DRIVER_CASE(cuDeviceGetP2PAttribute)
        CUPTI_DRIVER_CASE(cuTexRefSetBorderColor)
        CUPTI_DRIVER_CASE(cuTexRefGetBorderColor)
        CUPTI_DRIVER_CASE(cuMemAdvise)
        CUPTI_DRIVER_CASE(cuStreamWaitValue32)
        CUPTI_DRIVER_CASE(cuStreamWaitValue32_ptsz)
        CUPTI_DRIVER_CASE(cuStreamWriteValue32)
        CUPTI_DRIVER_CASE(cuStreamWriteValue32_ptsz)
        CUPTI_DRIVER_CASE(cuStreamBatchMemOp)
        CUPTI_DRIVER_CASE(cuStreamBatchMemOp_ptsz)
        CUPTI_DRIVER_CASE(cuNVNbufferGetPointer)
        CUPTI_DRIVER_CASE(cuNVNtextureGetArray)
        CUPTI_DRIVER_CASE(cuNNSetAllocator)
        CUPTI_DRIVER_CASE(cuMemPrefetchAsync)
        CUPTI_DRIVER_CASE(cuMemPrefetchAsync_ptsz)
        CUPTI_DRIVER_CASE(cuEventCreateFromNVNSync)
        CUPTI_DRIVER_CASE(cuEGLStreamConsumerConnectWithFlags)
        CUPTI_DRIVER_CASE(cuMemRangeGetAttribute)
        CUPTI_DRIVER_CASE(cuMemRangeGetAttributes)
        CUPTI_DRIVER_CASE(cuStreamWaitValue64)
        CUPTI_DRIVER_CASE(cuStreamWaitValue64_ptsz)
        CUPTI_DRIVER_CASE(cuStreamWriteValue64)
        CUPTI_DRIVER_CASE(cuStreamWriteValue64_ptsz)
        CUPTI_DRIVER_CASE(cuLaunchCooperativeKernel)
        CUPTI_DRIVER_CASE(cuLaunchCooperativeKernel_ptsz)
        CUPTI_DRIVER_CASE(cuEventCreateFromEGLSync)
        CUPTI_DRIVER_CASE(cuLaunchCooperativeKernelMultiDevice)
        CUPTI_DRIVER_CASE(cuFuncSetAttribute)
        CUPTI_DRIVER_CASE(cuDeviceGetUuid)
        CUPTI_DRIVER_CASE(cuStreamGetCtx)
        CUPTI_DRIVER_CASE(cuStreamGetCtx_ptsz)
        CUPTI_DRIVER_CASE(cuImportExternalMemory)
        CUPTI_DRIVER_CASE(cuExternalMemoryGetMappedBuffer)
        CUPTI_DRIVER_CASE(cuExternalMemoryGetMappedMipmappedArray)
        CUPTI_DRIVER_CASE(cuDestroyExternalMemory)
        CUPTI_DRIVER_CASE(cuImportExternalSemaphore)
        CUPTI_DRIVER_CASE(cuSignalExternalSemaphoresAsync)
        CUPTI_DRIVER_CASE(cuSignalExternalSemaphoresAsync_ptsz)
        CUPTI_DRIVER_CASE(cuWaitExternalSemaphoresAsync)
        CUPTI_DRIVER_CASE(cuWaitExternalSemaphoresAsync_ptsz)
        CUPTI_DRIVER_CASE(cuDestroyExternalSemaphore)
        CUPTI_DRIVER_CASE(cuStreamBeginCapture)
        CUPTI_DRIVER_CASE(cuStreamBeginCapture_ptsz)
        CUPTI_DRIVER_CASE(cuStreamEndCapture)
        CUPTI_DRIVER_CASE(cuStreamEndCapture_ptsz)
        CUPTI_DRIVER_CASE(cuStreamIsCapturing)
        CUPTI_DRIVER_CASE(cuStreamIsCapturing_ptsz)
        CUPTI_DRIVER_CASE(cuGraphCreate)
        CUPTI_DRIVER_CASE(cuGraphAddKernelNode)
        CUPTI_DRIVER_CASE(cuGraphKernelNodeGetParams)
        CUPTI_DRIVER_CASE(cuGraphAddMemcpyNode)
        CUPTI_DRIVER_CASE(cuGraphMemcpyNodeGetParams)
        CUPTI_DRIVER_CASE(cuGraphAddMemsetNode)
        CUPTI_DRIVER_CASE(cuGraphMemsetNodeGetParams)
        CUPTI_DRIVER_CASE(cuGraphMemsetNodeSetParams)
        CUPTI_DRIVER_CASE(cuGraphNodeGetType)
        CUPTI_DRIVER_CASE(cuGraphGetRootNodes)
        CUPTI_DRIVER_CASE(cuGraphNodeGetDependencies)
        CUPTI_DRIVER_CASE(cuGraphNodeGetDependentNodes)
        CUPTI_DRIVER_CASE(cuGraphInstantiate)
        CUPTI_DRIVER_CASE(cuGraphLaunch)
        CUPTI_DRIVER_CASE(cuGraphLaunch_ptsz)
        CUPTI_DRIVER_CASE(cuGraphExecDestroy)
        CUPTI_DRIVER_CASE(cuGraphDestroy)
        CUPTI_DRIVER_CASE(cuGraphAddDependencies)
        CUPTI_DRIVER_CASE(cuGraphRemoveDependencies)
        CUPTI_DRIVER_CASE(cuGraphMemcpyNodeSetParams)
        CUPTI_DRIVER_CASE(cuGraphKernelNodeSetParams)
        CUPTI_DRIVER_CASE(cuGraphDestroyNode)
        CUPTI_DRIVER_CASE(cuGraphClone)
        CUPTI_DRIVER_CASE(cuGraphNodeFindInClone)
        CUPTI_DRIVER_CASE(cuGraphAddChildGraphNode)
        CUPTI_DRIVER_CASE(cuGraphAddEmptyNode)
        CUPTI_DRIVER_CASE(cuLaunchHostFunc)
        CUPTI_DRIVER_CASE(cuLaunchHostFunc_ptsz)
        CUPTI_DRIVER_CASE(cuGraphChildGraphNodeGetGraph)
        CUPTI_DRIVER_CASE(cuGraphAddHostNode)
        CUPTI_DRIVER_CASE(cuGraphHostNodeGetParams)
        CUPTI_DRIVER_CASE(cuDeviceGetLuid)
        CUPTI_DRIVER_CASE(cuGraphHostNodeSetParams)
        CUPTI_DRIVER_CASE(cuGraphGetNodes)
        CUPTI_DRIVER_CASE(cuGraphGetEdges)
        default:
            return "INVALID";
    }
#undef CUPTI_DRIVER_CASE
}

CUDAAPIProfilerPrinter::CUDAAPIProfilerPrinter(CUDAAPIProfiler &profiler, float every_sec) :
    _event_handler(0.5),
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
