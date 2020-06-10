//
// Created by jagle on 8/23/2019.
//

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/mem.h"


#include "cuda_api_profiler/cupti_manager.h"
#include "cuda_api_profiler/get_env_var.h"

#include <cuda.h>
#include <cupti.h>

#include <absl/memory/memory.h>
#include <memory>

namespace rlscope {

// Used by ActivityBuffer and DeviceTracerImpl
#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = call;                                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      const char *errstr;                                           \
      cuptiGetResultString(_status, &errstr);                       \
      LOG(FATAL) << "libcupti call " << #call << " failed with " << errstr; \
    }                                                               \
  } while (0)

//
// SimpleByteAllocationPool
//

SimpleByteAllocationPool::SimpleByteAllocationPool(int max_num_allocations, size_t allocation_size, size_t allocation_alignment) :
    _max_num_allocations(max_num_allocations)
    , _allocation_size(allocation_size)
    , _allocation_alignment(allocation_size)
    , _num_allocated(0)
    , _max_num_allocated(0)
{
  _free_buffers.reserve(_max_num_allocations);
  for (int i = 0; i < _max_num_allocations; i++) {
    char* buffer = _NewBuffer();
    _free_buffers.emplace_back(buffer);
  }
}

char* SimpleByteAllocationPool::_NewBuffer() {
  void* ptr = nullptr;
  int err = posix_memalign(&ptr, _allocation_alignment, _allocation_size);
  if (err != 0) {
    LOG(FATAL) << "posix_memalign failed with err = " << err << ": " << strerror(err);
  }
  return reinterpret_cast<char*>(ptr);
}

SimpleByteAllocationPool::~SimpleByteAllocationPool() {
  if (_num_allocated != 0) {
    LOG(FATAL) << "SimpleByteAllocationPool: you forgot to call FreeBuffer; saw " << _num_allocated
               << " allocations outstanding";
  }
  LOG(INFO) << "SimpleByteAllocationPool: allocated at most " << _max_num_allocated
            << " buffers each of size " << _allocation_size
            << "bytes, totalling " << (_allocation_size*_max_num_allocated) << " bytes";
}

void SimpleByteAllocationPool::FreeBuffer(char* buffer) {
  if (_num_allocated == 0) {
    LOG(FATAL) << "SimpleByteAllocationPool: tried to call FreeBuffer but there are no buffers allocated";
  }
  _num_allocated -= 1;
}

char* SimpleByteAllocationPool::AllocateBuffer() {
  if (_num_allocated >= _free_buffers.size()) {
    LOG(FATAL) << "SimpleByteAllocationPool: reached allocation limit of " << _free_buffers.size() << " buffers";
  }
  char* buffer = _free_buffers[_num_allocated].get();
  _num_allocated += 1;
  if (_num_allocated > _max_num_allocated) {
    _max_num_allocated.store(_num_allocated.load());
  }
//    _max_num_allocated = std::max(_max_num_allocated, _num_allocated);
  return buffer;
}

//
// CUPTIManager
//

CUPTIManager::CUPTIManager() :
    allocation_pool_(MAX_CUPTI_BUFFERS_ALLOCATED, kBufferSize, kBufferAlignment)
{
//    cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
  void* ptr = nullptr;
  int err = posix_memalign(&ptr, kBufferAlignment, kBufferSize);
  if (err != 0) {
    LOG(FATAL) << "posix_memalign failed with err = " << err << ": " << strerror(err);
  }
//    global_libcupti_buffer_.reset(reinterpret_cast<char*>(ptr));
}

void CUPTIManager::FreeBufferCallback(uint8_t *_buffer) {
  // If buffer came from an Arena, we don't need to free it; it will get freed
  // whenever the Arena it belongs to is freed.
  bool use_arena = is_yes("TF_CUPTI_BUFFER_ARENA", false);
  if (use_arena) {
    allocation_pool_.FreeBuffer(reinterpret_cast<char*>(_buffer));
  }
  if (!use_arena) {
    port::AlignedFree(_buffer);
  }
}

void CUPTIManager::RecordActivityCallback(CUPTIClient* client, const CUpti_Activity &record) {
  client->ActivityCallback(record);
}

/* static */ CUPTIManager *CUPTIManager::Create() {
  auto manager = absl::make_unique<CUPTIManager>();
//    CUptiResult status = manager->cupti_wrapper_->ActivityRegisterCallbacks(
//        BufferRequested, BufferCompleted);
  if (!is_yes("IML_CUDA_ACTIVITIES", false)) {
    VLOG(1) << "SKIP: activity callbacks using cuptiActivityRegisterCallbacks";
  } else {
    // NOTE: Even just calling cuptiActivityRegisterCallbacks without enabling any activities with cuptiActivityEnable
    // causes 7% runtime overhead.
    VLOG(1) << "Enable activity callbacks using cuptiActivityRegisterCallbacks";
    CUptiResult status = cuptiActivityRegisterCallbacks(
        BufferRequested,
        BufferCompleted);
    if (status != CUPTI_SUCCESS) {
      LOG(ERROR) << "Failed to initialize CUPTI: " << status;
      return nullptr;
    }
  }
  return manager.release();
}

/* static */ void CUPTIManager::BufferRequested(uint8_t **buffer, size_t *size,
                                          size_t *maxNumRecords) {
  // VLOG(1) << "CUPTIManager." << __func__;
  GetCUPTIManager()->InternalBufferRequested(buffer, size, maxNumRecords);
}

/* static */ void CUPTIManager::BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                                          size_t size, size_t validSize) {
  GetCUPTIManager()->InternalBufferCompleted(ctx, streamId, buffer, size,
                                             validSize);
}

Status CUPTIManager::_EnablePCSampling() {
  VLOG(0) << "Enabling PC sampling";

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

  CUpti_ActivityPCSamplingConfig configPC;
  CUcontext cuCtx;
  configPC.size = sizeof(CUpti_ActivityPCSamplingConfig);
  // REALLY slow.
  configPC.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
//  configPC.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX;
  configPC.samplingPeriod2 = 0;
  cuCtxGetCurrent(&cuCtx);

  CUPTI_CALL(cuptiActivityConfigurePCSampling(cuCtx, &configPC));

  return Status::OK();
}

Status CUPTIManager::_DisablePCSampling() {
  VLOG(1) << "CUPTIManager." << __func__ << ": disable PC sampling";
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
  return Status::OK();
}

Status CUPTIManager::Flush() {
  VLOG(1) << "CUPTIManager." << __func__ << ": flush cupti activities";
  CUPTI_CALL(cuptiActivityFlushAll(0));
  return Status::OK();
}

Status CUPTIManager::EnableTrace(CUPTIClient *client) {
  mutex_lock l(mu_);
  VLOG(0) << __func__;
  // TODO(pbar) Work out the minimal set to trace.
  // We can currently manage without driver/runtime tracing.
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // These might be useful for annotations but require NVTX API.
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
  if (!is_yes("TF_CUPTI_SKIP_REGISTER_ACTIVITY", false)) {
    if (is_yes("IML_PC_SAMPLING", false)) {
      _EnablePCSampling();
    }
  }

  if (is_yes("IML_CUDA_ACTIVITIES", false)) {
    VLOG(1) << "Enable GPU activities (IML_CUDA_ACTIVITIES=yes)";
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
//      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
//      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  }

  client_ = client;
  return Status::OK();
}

Status CUPTIManager::DisableTrace() {
  // We turn off all tracing regardless.
  if (is_yes("IML_PC_SAMPLING", false)) {
    _DisablePCSampling();
  }
  // VLOG(1) << "CUPTIManager." << __func__ << ": disable cupti activities";
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
//  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2));
//  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  // VLOG(1) << "CUPTIManager." << __func__ << ": flush cupti activities";
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  // VLOG(1) << "CUPTIManager." << __func__ << ": grab lock to nullify client";
  {
    // Don't acquire this lock until Flush returns, since Flush
    // will potentially cause callbacks into BufferCompleted.
    mutex_lock l(mu_);
    client_ = nullptr;
  }
  return Status::OK();
}

void CUPTIManager::InternalBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {
  // VLOG(1) << "CUPTIManager." << __func__;
  // IML TODO: does this callback's malloc call add large overhead?
  // What if we register for fewer ACTIVITY_KIND's (see EnableTrace)?
  bool use_arena = is_yes("TF_CUPTI_BUFFER_ARENA", false);
  if (use_arena) {
//    void *p = google::protobuf::Arena::CreateArray<char>(&cupti_buffer_arena, kBufferSize);
//    *size = kBufferSize;
//    *buffer = reinterpret_cast<uint8_t *>(p);

    // To see if overhead is related to Arena, just return a one-time allocated global array.
    // ERROR: causes program to hang... I guess more than 1 buffer is needed by libcupti
//    void *p = global_libcupti_buffer_.get();
//    *size = kBufferSize;
//    *buffer = reinterpret_cast<uint8_t *>(p);

    // Use really simple non-thread-safe pooling allocator.
    void *p = allocation_pool_.AllocateBuffer();
    *size = kBufferSize;
    *buffer = reinterpret_cast<uint8_t *>(p);

  } else {
    // VLOG(1) << "CUPTIManager." << __func__ << ": alloc " << kBufferSize << " bytes";
    void *p = port::AlignedMalloc(kBufferSize, kBufferAlignment);
    *size = kBufferSize;
    *buffer = reinterpret_cast<uint8_t *>(p);
  }
  *maxNumRecords = 0;
  if (is_yes("TF_CUPTI_DEBUG", false)) {
    VLOG(0) << "CUPTIManager." << __func__ << ": BufferRequested, buffer=" << ((void*)*buffer) << ", size=" << *size << ", arena = " << use_arena;
  }
}

void CUPTIManager::InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {
  if (is_yes("TF_CUPTI_DEBUG", false)) {
    VLOG(0) << "CUPTIManager." << __func__ << ": BufferCompleted, buffer=" << ((void *) buffer) << ", size=" << validSize;
  }
  CUptiResult status;
  CUpti_Activity *record = nullptr;
  // Q: If we return from this function ASAP by handing off work to a
  // separate thread, will this help...?
  // A: Potentially, yes.  Basically, we are FORCING the main thread to wait
  // for all the records to be recorded in DeviceTraceImpl when it calls
  // device_tracer->Stop().  Really, we just want to hand it the buffer, and
  // have device_tracer->Collect() append the records in an async thread.

//  auto client = client_;
//                                                           [client] (const CUpti_Activity &record) {
//                                                             client->ActivityCallback(record);
//                                                           });

//  if (!is_yes("TF_CUPTI_EMPTY_TRACING_CALLBACKS", false) && client_ && validSize > 0) {
//    mutex_lock l(mu_);  // Hold mu_ while using client_.
//    if (is_yes("TF_CUPTI_ASYNC_RECORD_ACTIVITY", false)) {
//      auto activity_buffer = absl::make_unique<ActivityBuffer>(ctx, streamId, buffer, size, validSize,
//                                                               this, client_);
//      if (is_yes("TF_CUPTI_ASYNC_RECORD_ACTIVITY_DEBUG", false)) {
//        LOG(INFO) << "Delay handling libcupti activity buffer (TF_CUPTI_ASYNC_RECORD_ACTIVITY=yes)";
//      }
//      client_->ActivityBufferCallback(std::move(activity_buffer));
//    } else {
//      ActivityBuffer activity_buffer(ctx, streamId, buffer, size, validSize, this, client_);
//      /* Pull activity records out of libcupti activity buffer directly on critical path
//       * (not desirable, but this is the default TensorFlow behaviour).
//       */
//      activity_buffer.RecordActivitiesFromBuffer();
//    }
//  } else {
//    ActivityBuffer activity_buffer(ctx, streamId, buffer, size, validSize, this, client_);
//    // Don't record anything, just free the buffer.
//  }


  if (client_ && validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        client_->ActivityCallback(*record);
      } else {
        break;
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG(WARNING) << "Dropped " << dropped << " activity records";
    }
  }
  bool use_arena = is_yes("TF_CUPTI_BUFFER_ARENA", false);
  if (!use_arena) {
    port::AlignedFree(buffer);
  }

}

CUPTIManager *GetCUPTIManager() {
  static CUPTIManager *manager = CUPTIManager::Create();
  return manager;
}

}
