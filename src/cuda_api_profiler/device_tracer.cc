/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/platform/logging.h"

#include "cuda_api_profiler/cuda_api_profiler.h"

#include <cuda.h>
#include <cupti.h>

#include <absl/memory/memory.h>

#include <atomic>
#include <map>
//#include <mutex>
#include <vector>

#ifdef CONFIG_TRACE_STATS

// Record memcpy/kernel launch timings.
//
// NOTE: Even if this is disabled, the callbacks for recording
// these statistics ARE still triggered.
//#define ENABLE_GPU_TRACING (true)

// Register CUDA API callbacks, but do nothing inside of them.
//#define TF_CUPTI_EMPTY_TRACING_CALLBACKS (false)

// Skip registering CUDA API callbacks altogether.
//#define TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS (false)

#endif // CONFIG_TRACE_STATS

#include <stdlib.h>
#include <memory>
#include <algorithm>
#include <list>
#include <cassert>

//#include <google/protobuf/arena.h>

//#include "tensorflow/core/common_runtime/step_stats_collector.h"
//#include "tensorflow/core/framework/step_stats.pb.h"
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/lib/strings/strcat.h"
//#include "tensorflow/core/lib/strings/stringprintf.h"
//#include "tensorflow/core/platform/cupti_wrapper.h"
#include "tensorflow/core/platform/env.h"
//#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"

//#define mutex std::mutex
//#define mutex_lock std::lock_guard<std::mutex>

namespace {

bool is_yes(const char* env_var, bool default_value);
bool is_no(const char* env_var, bool default_value);

bool is_yes(const char* env_var, bool default_value) {
  if (getenv(env_var) == nullptr) {
    return default_value;
  }
  return strcmp("yes", getenv(env_var)) == 0;
}
bool is_no(const char* env_var, bool default_value) {
  if (getenv(env_var) == nullptr) {
    return default_value;
  }
  return strcmp("no", getenv(env_var)) == 0;
}

const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind);
const char *getMemoryKindString(CUpti_ActivityMemoryKind kind);
const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind);
uint32_t getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id);
const char * getActivityObjectKindString(CUpti_ActivityObjectKind kind);
const char * getComputeApiKindString(CUpti_ActivityComputeApiKind kind);


void
printActivity(const CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
      printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
             "multiprocessors %u, clock %u MHz\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor,
             (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
             (unsigned int) (device->globalMemorySize / 1024 / 1024),
             device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
      break;
    }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
      printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
             attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
      break;
    }
  case CUPTI_ACTIVITY_KIND_CONTEXT:
    {
      CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
      printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
             context->contextId, context->deviceId,
             getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
             (int) context->nullStreamId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
             getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
//             (unsigned long long) (memcpy->start - startTimestamp),
//             (unsigned long long) (memcpy->end - startTimestamp),
             (unsigned long long) (memcpy->start),
             (unsigned long long) (memcpy->end),
             memcpy->deviceId, memcpy->contextId, memcpy->streamId,
             memcpy->correlationId, memcpy->runtimeCorrelationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset *memset = (CUpti_ActivityMemset *) record;
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             memset->value,
//             (unsigned long long) (memset->start - startTimestamp),
//             (unsigned long long) (memset->end - startTimestamp),
             (unsigned long long) (memset->start),
             (unsigned long long) (memset->end),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *) record;
      printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
//             (unsigned long long) (kernel->start - startTimestamp),
//             (unsigned long long) (kernel->end - startTimestamp),
             (unsigned long long) (kernel->start),
             (unsigned long long) (kernel->end),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
//             (unsigned long long) (api->start - startTimestamp),
//             (unsigned long long) (api->end - startTimestamp),
             (unsigned long long) (api->start),
             (unsigned long long) (api->end),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
//             (unsigned long long) (api->start - startTimestamp),
//             (unsigned long long) (api->end - startTimestamp),
             (unsigned long long) (api->start),
             (unsigned long long) (api->end),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_NAME:
    {
      CUpti_ActivityName *name = (CUpti_ActivityName *) record;
      switch (name->objectKind)
      {
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        printf("NAME  %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      case CUPTI_ACTIVITY_OBJECT_STREAM:
        printf("NAME %s %u %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      default:
        printf("NAME %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               name->name);
        break;
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER:
    {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
      printf("MARKER id %u [ %llu ], name %s, domain %s\n",
             marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA:
    {
      CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
      printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
             marker->id, marker->color, marker->category,
             (unsigned long long) marker->payload.metricValueUint64,
             marker->payload.metricValueDouble);
      break;
    }
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
      printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
             getActivityOverheadKindString(overhead->overheadKind),
//             (unsigned long long) overhead->start - startTimestamp,
//             (unsigned long long) overhead->end - startTimestamp,
             (unsigned long long) overhead->start,
             (unsigned long long) overhead->end,
             getActivityObjectKindString(overhead->objectKind),
             getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
      break;
    }
  default:
    printf("  <unknown>\n");
    break;
  }
}

const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      return "PROCESS";
    case CUPTI_ACTIVITY_OBJECT_THREAD:
      return "THREAD";
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
      return "DEVICE";
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      return "CONTEXT";
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      return "STREAM";
    default:
      break;
  }

  return "<unknown>";
}

uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
    case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
      return "CUDA";
    case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
      return "CUDA_MPS";
    default:
      break;
  }

  return "<unknown>";
}

// Maps a MemcpyKind enum to a const string.
const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

// Maps a MemoryKind enum to a const string.
const char *getMemoryKindString(CUpti_ActivityMemoryKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    default:
      break;
  }
  return "<unknown>";
}

// Maps an OverheadKind enum to a const string.
const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }
  return "<unknown>";
}

}  // namespace

namespace tensorflow {

class CUDAAPIProfiler;

namespace devicetracer {

// Used by ActivityBuffer and DeviceTracerImpl
#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = call;                                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

// Forward declaration.
class CUPTIManager;
class CUPTIClient;

class ActivityBuffer {
public:
//  using RecordActivityCallback = std::function<void(const CUpti_Activity &activity)>;
  ActivityBuffer(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize,
      CUPTIManager* manager, CUPTIClient* client) :
//      RecordActivityCallback record_activity_callback) :
      _ctx(ctx)
      , _streamId(streamId)
      , _buffer(buffer)
      , _size(size)
      , _validSize(validSize)
      , _manager(manager)
      , _client(client)
//      , _record_activity_callback(record_activity_callback)
  {
//    cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
  }
  CUcontext _ctx;
  uint32_t _streamId;
  uint8_t *_buffer;
  size_t _size;
  size_t _validSize;
  CUPTIManager* _manager;
  CUPTIClient* _client;
//  RecordActivityCallback _record_activity_callback;
//  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;

  void RecordActivitiesFromBuffer();

  void FreeBuffer();

  ~ActivityBuffer();

};


// Returns a pointer to the CUPTIManager singleton.
CUPTIManager *GetCUPTIManager();

// Callback interface for consumers of CUPTI tracing.
class CUPTIClient {
 public:
  virtual ~CUPTIClient() {}

  // Invoked for each CUPTI activity reported.
  virtual void ActivityCallback(const CUpti_Activity &activity) = 0;
  virtual void ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) = 0;
};

class SimpleByteAllocationPool {
public:
  SimpleByteAllocationPool(int max_num_allocations, size_t allocation_size, size_t allocation_alignment) :
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
  char* _NewBuffer() {
    void* ptr = nullptr;
    int err = posix_memalign(&ptr, _allocation_alignment, _allocation_size);
    if (err != 0) {
      LOG(FATAL) << "posix_memalign failed with err = " << err << ": " << strerror(err);
    }
    return reinterpret_cast<char*>(ptr);
  }
  char* AllocateBuffer() {
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
  void FreeBuffer(char* buffer) {
    if (_num_allocated == 0) {
      LOG(FATAL) << "SimpleByteAllocationPool: tried to call FreeBuffer but there are no buffers allocated";
    }
    _num_allocated -= 1;
  }
  int _max_num_allocations;
  size_t _allocation_size;
  size_t _allocation_alignment;
//  int _num_allocated;
  // I'm not sure if this makes it thread-safe to be honest...
  std::atomic<std::size_t> _num_allocated;
//  int _max_num_allocated;
  // NOTE: This variable is not thread-safe...it might be wrong, but that's ok since it's just a debug statistic.
  std::atomic<std::size_t> _max_num_allocated;
  std::vector<std::unique_ptr<char>> _free_buffers;

  ~SimpleByteAllocationPool() {
    if (_num_allocated != 0) {
      LOG(FATAL) << "SimpleByteAllocationPool: you forgot to call FreeBuffer; saw " << _num_allocated
                 << " allocations outstanding";
    }
    LOG(INFO) << "SimpleByteAllocationPool: allocated at most " << _max_num_allocated
              << " buffers each of size " << _allocation_size
              << "bytes, totalling " << (_allocation_size*_max_num_allocated) << " bytes";
  }
};

// 32MB
// PROBLEM:
// - we delay processing libcupti buffers so they won't be freed until much later...
// - allocation pool needs to be thread-safe.
#define MAX_CUPTI_BUFFERS_ALLOCATED 1024

// Singleton class to manage registration of CUPTI callbacks.
class CUPTIManager {
 public:
//  google::protobuf::Arena cupti_buffer_arena;
//  std::unique_ptr<char> global_libcupti_buffer_;
  SimpleByteAllocationPool allocation_pool_;

  CUPTIManager() :
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

  void FreeBufferCallback(uint8_t *_buffer) {
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
  void RecordActivityCallback(CUPTIClient* client, const CUpti_Activity &record) {
    client->ActivityCallback(record);
  }

  static CUPTIManager *Create() {
    auto manager = absl::make_unique<CUPTIManager>();
//    CUptiResult status = manager->cupti_wrapper_->ActivityRegisterCallbacks(
//        BufferRequested, BufferCompleted);
    CUptiResult status = cuptiActivityRegisterCallbacks(
            BufferRequested,
            BufferCompleted);
    if (status != CUPTI_SUCCESS) {
      LOG(ERROR) << "Failed to initialize CUPTI: " << status;
      return nullptr;
    }
    return manager.release();
  }

  // Enables tracing and delivers event callbacks to 'client'.
  // Does not take ownership of client.  Client's lifetime must persist
  // until tracing is disabled.
  Status EnableTrace(CUPTIClient *client);

  // Disable tracing.  No further events will be delivered to 'client'.
  Status DisableTrace();

 private:
  // Static functions which we can use as CUPTI callbacks.
  static void BufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
    GetCUPTIManager()->InternalBufferRequested(buffer, size, maxNumRecords);
  }
  static void BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
    GetCUPTIManager()->InternalBufferCompleted(ctx, streamId, buffer, size,
                                               validSize);
  }
  // These methods are called by the static stubs above.
  void InternalBufferRequested(uint8_t **buffer, size_t *size,
                               size_t *maxNumRecords);
  void InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                               uint8_t *buffer, size_t size, size_t validSize);

  // Size of buffers used for CUPTI tracing.
  static constexpr size_t kBufferSize = 32 * 1024;
  // Required alignment of CUPTI buffers.
  static constexpr size_t kBufferAlignment = 8;

  mutex mu_;
  CUPTIClient *client_ GUARDED_BY(mu_);
//  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;

  TF_DISALLOW_COPY_AND_ASSIGN(CUPTIManager);
};

void ActivityBuffer::RecordActivitiesFromBuffer() {
  if (is_yes("TF_CUPTI_ASYNC_RECORD_ACTIVITY_DEBUG", false)) {
    LOG(INFO) << "RecordActivitiesFromBuffer";
  }
  if (_validSize > 0) {
    CUptiResult status;
    CUpti_Activity *record = nullptr;
    do {
//      status = cupti_wrapper_->ActivityGetNextRecord(_buffer, _validSize, &record);
      status = cuptiActivityGetNextRecord(_buffer, _validSize, &record);
      if (status == CUPTI_SUCCESS) {
//          client_->ActivityCallback(*record);
//          this->_record_activity_callback(*record);
        _manager->RecordActivityCallback(_client, *record);
      } else {
        break;
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(_ctx, _streamId, &dropped));
    if (dropped != 0) {
      LOG(WARNING) << "Dropped " << dropped << " activity records";
    }
  }
  // All done recorded activities from libcupti buffer; free it now.
  FreeBuffer();
}

void ActivityBuffer::FreeBuffer() {
  if (_buffer) {
    _manager->FreeBufferCallback(_buffer);
    _buffer = nullptr;
  }
}

ActivityBuffer::~ActivityBuffer() {
  if (_buffer != nullptr && !is_yes("TF_CUPTI_EMPTY_TRACING_CALLBACKS", false)) {
    LOG(WARNING) << "Looks like we forgot to record some GPU-time event data.  Make Sure RecordActivitiesFromBuffer gets called!";
  }
  FreeBuffer();
}

Status CUPTIManager::EnableTrace(CUPTIClient *client) {
  mutex_lock l(mu_);
  // TODO(pbar) Work out the minimal set to trace.
  // We can currently manage without driver/runtime tracing.
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // These might be useful for annotations but require NVTX API.
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));

  if (!is_yes("TF_CUPTI_SKIP_REGISTER_ACTIVITY", false)) {
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  }
  client_ = client;
  return Status::OK();
}

Status CUPTIManager::DisableTrace() {
  // We turn off all tracing regardless.
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
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
    void *p = port::AlignedMalloc(kBufferSize, kBufferAlignment);
    *size = kBufferSize;
    *buffer = reinterpret_cast<uint8_t *>(p);
  }
  *maxNumRecords = 0;
  if (is_yes("TF_CUPTI_DEBUG", false)) {
    VLOG(0) << "BufferRequested, buffer=" << ((void*)*buffer) << ", size=" << *size << ", arena = " << use_arena;
  }
}

void CUPTIManager::InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {
  if (is_yes("TF_CUPTI_DEBUG", false)) {
    VLOG(0) << "BufferCompleted, buffer=" << ((void *) buffer) << ", size=" << validSize;
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

  if (!is_yes("TF_CUPTI_EMPTY_TRACING_CALLBACKS", false) && client_ && validSize > 0) {
    mutex_lock l(mu_);  // Hold mu_ while using client_.
    if (is_yes("TF_CUPTI_ASYNC_RECORD_ACTIVITY", false)) {
      auto activity_buffer = absl::make_unique<ActivityBuffer>(ctx, streamId, buffer, size, validSize,
                                                               this, client_);
      if (is_yes("TF_CUPTI_ASYNC_RECORD_ACTIVITY_DEBUG", false)) {
        LOG(INFO) << "Delay handling libcupti activity buffer (TF_CUPTI_ASYNC_RECORD_ACTIVITY=yes)";
      }
      client_->ActivityBufferCallback(std::move(activity_buffer));
    } else {
      ActivityBuffer activity_buffer(ctx, streamId, buffer, size, validSize, this, client_);
      /* Pull activity records out of libcupti activity buffer directly on critical path
       * (not desirable, but this is the default TensorFlow behaviour).
       */
      activity_buffer.RecordActivitiesFromBuffer();
    }
  } else {
    ActivityBuffer activity_buffer(ctx, streamId, buffer, size, validSize, this, client_);
    // Don't record anything, just free the buffer.
  }


//  if (client_ && validSize > 0) {
//    do {
//      status =
//          cupti_wrapper_->ActivityGetNextRecord(buffer, validSize, &record);
//      if (status == CUPTI_SUCCESS) {
//        client_->ActivityCallback(*record);
//      } else {
//        break;
//      }
//    } while (1);
//
//    // report any records dropped from the queue
//    size_t dropped;
//    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
//    if (dropped != 0) {
//      LOG(WARNING) << "Dropped " << dropped << " activity records";
//    }
//  }
//  bool use_arena = is_yes("TF_CUPTI_BUFFER_ARENA", false);
//  if (!use_arena) {
//    port::AlignedFree(buffer);
//  }

}

CUPTIManager *GetCUPTIManager() {
  static CUPTIManager *manager = CUPTIManager::Create();
  return manager;
}

#ifdef _MSC_VER
#define __thread __declspec(thread)
#endif

// TODO(pbar) Move this to platform specific header file?
// Static thread local variable for POD types.
#define TF_STATIC_THREAD_LOCAL_POD(_Type_, _var_)                  \
  static __thread _Type_ s_obj_##_var_;                            \
  namespace {                                                      \
  class ThreadLocal_##_var_ {                                      \
   public:                                                         \
    ThreadLocal_##_var_() {}                                       \
    void Init() {}                                                 \
    inline _Type_ *pointer() const { return &s_obj_##_var_; }      \
    inline _Type_ *safe_pointer() const { return &s_obj_##_var_; } \
    _Type_ &get() const { return s_obj_##_var_; }                  \
    bool is_native_tls() const { return true; }                    \
                                                                   \
   private:                                                        \
    TF_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);              \
  } _var_;                                                         \
  }  // namespace

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the CUPTI API callback.
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_annotation);

class TraceCollectorImpl : public tracing::TraceCollector {
 public:
  TraceCollectorImpl() { tracing::SetTraceCollector(this); }

  ~TraceCollectorImpl() override {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
  }

  // Note the method can be called after a call to Stop().
  virtual std::unique_ptr<Handle> CreateAnnotationHandle(
      StringPiece name_part1, StringPiece name_part2) const {
    struct Impl : public tracing::TraceCollector::Handle {
      string annotation;
      explicit Impl(string &&name_scope) : annotation(name_scope) {
        VLOG(2) << "CreateAnnotationHandle " << annotation;
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation.get() = annotation.c_str();
      }
      ~Impl() override { tls_current_annotation.get() = nullptr; }
    };
    return std::unique_ptr<Handle>(
        new Impl{ConcatenateNames(name_part1, name_part2)});
  }

  virtual std::unique_ptr<Handle> CreateActivityHandle(StringPiece, StringPiece,
                                                       bool) const {
    // We don't do anything with 'Activities' yet.
    return nullptr;
  }

  bool IsEnabledForAnnotations() const override {
    return active_trace_session_.load(std::memory_order_relaxed);
  }

  bool IsEnabledForActivities(bool is_expensive) const override {
    // We don't do anything with 'Activities' so we are never 'enabled'.
    return false;
  }

  void Start() {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
    active_trace_session_ = true;
  }

  void Stop() {
    DCHECK(active_trace_session_) << "No active trace session detected. ";
    active_trace_session_ = false;
  }

 private:
  std::atomic<bool> active_trace_session_;
};


TraceCollectorImpl *GlobalDefaultTraceCollector() {
  static auto *instance = new TraceCollectorImpl();
  return instance;
}


class DeviceTracerImpl : public DeviceTracer, public CUPTIClient {
 public:
  std::vector<std::unique_ptr<ActivityBuffer>> activity_buffers_;
  CUDAAPIProfiler api_profiler_;
  CUDAAPIProfilerPrinter api_printer_;
//  google::protobuf::Arena cupti_protobuf_arena;
  DeviceTracerImpl(CUPTIManager *cupti_manager);
  ~DeviceTracerImpl() override;

  // DeviceTracer interface:
  Status Start() override;
  Status Stop() override;
  Status Collect() override;

#ifdef CONFIG_TRACE_STATS
  bool IsEnabled() override;
#endif

 protected:
  // This callback is used exclusively by CUPTIManager.
  friend class CUPTIManager;
  void ActivityCallback(const CUpti_Activity &activity) override;
  void ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) override;

 private:
  // Internal struct to record kernel launches.
  struct KernelRecord {
    KernelRecord(
        uint64_t start_timestamp_,
        uint64_t end_timestamp_,
        uint32 device_id_,
        uint32 stream_id_,
        uint32 correlation_id_) :
        start_timestamp(start_timestamp_)
        , end_timestamp(end_timestamp_)
        , device_id(device_id_)
        , stream_id(stream_id_)
        , correlation_id(correlation_id_)
    {
    }
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint32 device_id;
    uint32 stream_id;
    uint32 correlation_id;
  };
  // Internal struct to record memcpy operations.
  struct MemcpyRecord {
    MemcpyRecord(
        uint64_t start_timestamp_,
        uint64_t end_timestamp_,
        uint32 device_id_,
        uint32 stream_id_,
        uint32 correlation_id_,
        uint8 copyKind_,
        uint8 srcKind_,
        uint8 dstKind_,
        uint64 bytes_) :
        start_timestamp(start_timestamp_)
        , end_timestamp(end_timestamp_)
        , device_id(device_id_)
        , stream_id(stream_id_)
        , correlation_id(correlation_id_)
        , copyKind(copyKind_)
        , srcKind(srcKind_)
        , dstKind(dstKind_)
        , bytes(bytes_)
    {
    }
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint32 device_id;
    uint32 stream_id;
    uint32 correlation_id;
    uint8 copyKind;
    uint8 srcKind;
    uint8 dstKind;
    uint64 bytes;
  };

  // This is the subscriber callback which is invoked directly by CUPTI.
  // The 'userdata' argument will be a pointer to the active 'DeviceTracerImpl'.
  static void CUPTIAPI ApiCallback(void *userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void *cbdata);

  // Records the mapping between correlation ID and kernel name.
  void AddCorrelationId(uint32 correlation_id, const string &name);

  // Returns the current system time in microseconds.
  inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

  CUPTIManager *cupti_manager_;
//  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;
  CUpti_SubscriberHandle subscriber_;

  mutex trace_mu_;
  static constexpr size_t kMaxRecords = 1024 * 1024;
  std::map<uint32, string> correlations_ GUARDED_BY(trace_mu_);
  std::vector<KernelRecord> kernel_records_ GUARDED_BY(trace_mu_);
  std::vector<MemcpyRecord> memcpy_records_ GUARDED_BY(trace_mu_);

  mutex mu_;
  bool enabled_ GUARDED_BY(mu_);
  int64 start_walltime_us_ GUARDED_BY(mu_);
  int64 end_walltime_us_ GUARDED_BY(mu_);
  uint64_t start_timestamp_ GUARDED_BY(mu_);
  uint64_t end_timestamp_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceTracerImpl);
};

int ParseFloat(const char* str, size_t size) {
    // Ideally we would use env_var / safe_strto64, but it is
    // hard to use here without pulling in a lot of dependencies,
    // so we use std:istringstream instead
    string integer_str(str, size);
    std::istringstream ss(integer_str);
    float val = 0;
    ss >> val;
    return val;
}

static const float TF_CUDA_API_PRINT_EVERY_SEC_DEFAULT = 5.0;
float PrintCUDAAPIEverySec(float user_value) {
    if (user_value != 0) {
        return user_value;
    }
    const char* TF_CUDA_API_PRINT_EVERY_SEC = getenv("TF_CUDA_API_PRINT_EVERY_SEC");
    if (TF_CUDA_API_PRINT_EVERY_SEC == nullptr) {
        return TF_CUDA_API_PRINT_EVERY_SEC_DEFAULT;
    }
    return ParseFloat(TF_CUDA_API_PRINT_EVERY_SEC, strlen(TF_CUDA_API_PRINT_EVERY_SEC));
}

DeviceTracerImpl::DeviceTracerImpl(CUPTIManager *cupti_manager)
    :
        api_printer_(api_profiler_, PrintCUDAAPIEverySec(0)),
        cupti_manager_(cupti_manager)
    {
  VLOG(1) << "DeviceTracer created.";
//  cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
  if (is_yes("TF_CUPTI_PROTOBUF_ARENA", false)) {
    kernel_records_.reserve(kMaxRecords);
    memcpy_records_.reserve(kMaxRecords);
//    cupti_protobuf_arena = std::move(google::protobuf::Arena(google::protobuf::ArenaOptions{
//    }))
  }
  enabled_ = false;
}

DeviceTracerImpl::~DeviceTracerImpl() {
  // Unregister the CUPTI callbacks if needed to prevent them from accessing
  // freed memory.
  Stop().IgnoreError();
}

Status DeviceTracerImpl::Start() {
  VLOG(1) << "DeviceTracer::Start";
  mutex_lock l(mu_);
  api_printer_.Start();
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is already enabled.");
  }
#ifdef CONFIG_TRACE_STATS
  if (!is_yes("TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS", false)) {

    if (!is_yes("TF_CUPTI_SKIP_REGISTER_API_CALLBACKS", false)) {
      // There can only be one CUPTI subscriber.  If we can't create one then
      // there is another trace in progress (possibly by external code).
      CUptiResult ret;
//      ret = cupti_wrapper_->Subscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
      ret = cuptiSubscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
      if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        return errors::Unavailable("CUPTI subcriber limit reached.");
      } else if (ret != CUPTI_SUCCESS) {
        return errors::Internal("Failed to create CUPTI subcriber.");
      }

      // Register as a TraceEngine to receive ScopedAnnotations.
      GlobalDefaultTraceCollector()->Start();

      // Intercept launch and memcpy calls to capture the Op name annotation.
      // TODO(pbar) Add callbacks for memcpy variants.
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_RUNTIME_API,
                                           CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020));
      CUPTI_CALL(cuptiEnableCallback(
          /*enable=*/1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));

      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2));
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2));
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2));
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2));
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2));
      CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                                           CUPTI_CB_DOMAIN_DRIVER_API,
                                           CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2));
    }

    TF_RETURN_IF_ERROR(cupti_manager_->EnableTrace(this));

    CUPTI_CALL(cuptiGetTimestamp(&start_timestamp_));
  }
#else
  // There can only be one CUPTI subscriber.  If we can't create one then
  // there is another trace in progress (possibly by external code).
  CUptiResult ret;
//  ret = cupti_wrapper_->Subscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
  ret = cuptiSubscribe(&subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
  if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
    return errors::Unavailable("CUPTI subcriber limit reached.");
  } else if (ret != CUPTI_SUCCESS) {
    return errors::Internal("Failed to create CUPTI subcriber.");
  }

  // Register as a TraceEngine to receive ScopedAnnotations.
  GlobalDefaultTraceCollector()->Start();

  // Intercept launch and memcpy calls to capture the Op name annotation.
  // TODO(pbar) Add callbacks for memcpy variants.
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020));
  CUPTI_CALL(cuptiEnableCallback(
      /*enable=*/1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));

  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2));
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2));
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2));
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2));
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2));
  CUPTI_CALL(cuptiEnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2));

  TF_RETURN_IF_ERROR(cupti_manager_->EnableTrace(this));

  CUPTI_CALL(cuptiGetTimestamp(&start_timestamp_));
#endif // CONFIG_TRACE_STATS

  start_walltime_us_ = NowInUsec();
  enabled_ = true;
  return Status::OK();
}

Status DeviceTracerImpl::Stop() {
  VLOG(1) << "DeviceTracer::Stop";
  mutex_lock l(mu_);
  api_printer_.Stop();
  if (!enabled_) {
    return Status::OK();
  }
#ifdef CONFIG_TRACE_STATS
  if (!is_yes("TF_CUPTI_SKIP_REGISTER_CUPTI_CALLBACKS", false)) {
    if (!is_yes("TF_CUPTI_SKIP_REGISTER_API_CALLBACKS", false)) {
      CUPTI_CALL(cuptiUnsubscribe(subscriber_));
      GlobalDefaultTraceCollector()->Stop();
    }

    TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace());
    end_walltime_us_ = NowInUsec();
    CUPTI_CALL(cuptiGetTimestamp(&end_timestamp_));
  }
#else
  CUPTI_CALL(cuptiUnsubscribe(subscriber_));
  GlobalDefaultTraceCollector()->Stop();

  TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace());
  end_walltime_us_ = NowInUsec();
  CUPTI_CALL(cuptiGetTimestamp(&end_timestamp_));
#endif // CONFIG_TRACE_STATS
  enabled_ = false;
  return Status::OK();
}

void DeviceTracerImpl::AddCorrelationId(uint32 correlation_id,
                                        const string &name) {
  VLOG(2) << correlation_id << " : " << name;
  mutex_lock l(trace_mu_);
  if (correlations_.size() >= kMaxRecords) return;
  correlations_.emplace(correlation_id, name);
}



/*static*/ void DeviceTracerImpl::ApiCallback(void *userdata,
                                              CUpti_CallbackDomain domain,
                                              CUpti_CallbackId cbid,
                                              const void *cbdata) {
  auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
#ifdef CONFIG_TRACE_STATS
  if (is_yes("TF_CUPTI_TRACE_API_CALLS", false)) {
    DeviceTracerImpl *tracer = reinterpret_cast<DeviceTracerImpl *>(userdata);
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API || domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
      tracer->api_profiler_.ApiCallback(userdata, domain, cbid, cbdata);
    }
  }

  if (!is_yes("TF_CUPTI_EMPTY_TRACING_CALLBACKS", false)) {
      DeviceTracerImpl *tracer = reinterpret_cast<DeviceTracerImpl *>(userdata);
      VLOG(2) << "ApiCallback " << domain << ":" << cbid
          << " func: " << cbInfo->functionName;

      // API callbacks are invoked synchronously on the thread making the
      // CUDA API call.  If this pointer is non-null then the ScopedAnnotation
      // must be valid.
      const char *tls_annotation = tls_current_annotation.get();

      if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
              (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
          if (cbInfo->callbackSite == CUPTI_API_ENTER) {
              auto *params = reinterpret_cast<const cuLaunchKernel_params *>(
                      cbInfo->functionParams);
              if (VLOG_IS_ON(2)) {
                  VLOG(2) << "LAUNCH stream " << params->hStream << " correllation "
                      << cbInfo->correlationId << " kernel " << cbInfo->symbolName;
              }
              const string annotation =
                  tls_annotation ? tls_annotation : cbInfo->symbolName;
              tracer->AddCorrelationId(cbInfo->correlationId, annotation);
          }
      } else if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API) &&
              (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
               cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) {
          if (cbInfo->callbackSite == CUPTI_API_ENTER) {
              if (VLOG_IS_ON(2)) {
                  auto *funcParams = reinterpret_cast<const cudaMemcpy_v3020_params *>(
                          cbInfo->functionParams);
                  size_t count = funcParams->count;
                  enum cudaMemcpyKind kind = funcParams->kind;
                  VLOG(2) << "MEMCPY count " << count << " kind " << kind;
              }
              if (tls_annotation) {
                  const string annotation = tls_annotation;
                  tracer->AddCorrelationId(cbInfo->correlationId, annotation);
              }
          }
      } else if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
              (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 ||
               cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2)) {
          if (cbInfo->callbackSite == CUPTI_API_EXIT && tls_annotation) {
              const string annotation = tls_annotation;
              tracer->AddCorrelationId(cbInfo->correlationId, annotation);
          }
      } else {
          VLOG(1) << "Unhandled API Callback for " << domain << " " << cbid;
      }
  }
#else
  DeviceTracerImpl *tracer = reinterpret_cast<DeviceTracerImpl *>(userdata);
  VLOG(2) << "ApiCallback " << domain << ":" << cbid
          << " func: " << cbInfo->functionName;

  // API callbacks are invoked synchronously on the thread making the
  // CUDA API call.  If this pointer is non-null then the ScopedAnnotation
  // must be valid.
  const char *tls_annotation = tls_current_annotation.get();

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
      (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      auto *params = reinterpret_cast<const cuLaunchKernel_params *>(
          cbInfo->functionParams);
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "LAUNCH stream " << params->hStream << " correllation "
                << cbInfo->correlationId << " kernel " << cbInfo->symbolName;
      }
      const string annotation =
          tls_annotation ? tls_annotation : cbInfo->symbolName;
      tracer->AddCorrelationId(cbInfo->correlationId, annotation);
    }
  } else if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API) &&
             (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
              cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      if (VLOG_IS_ON(2)) {
        auto *funcParams = reinterpret_cast<const cudaMemcpy_v3020_params *>(
            cbInfo->functionParams);
        size_t count = funcParams->count;
        enum cudaMemcpyKind kind = funcParams->kind;
        VLOG(2) << "MEMCPY count " << count << " kind " << kind;
      }
      if (tls_annotation) {
        const string annotation = tls_annotation;
        tracer->AddCorrelationId(cbInfo->correlationId, annotation);
      }
    }
  } else if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
             (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2)) {
    if (cbInfo->callbackSite == CUPTI_API_EXIT && tls_annotation) {
      const string annotation = tls_annotation;
      tracer->AddCorrelationId(cbInfo->correlationId, annotation);
    }
  } else {
    VLOG(1) << "Unhandled API Callback for " << domain << " " << cbid;
  }
#endif // CONFIG_TRACE_STATS
}

void DeviceTracerImpl::ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) {
  VLOG(2) << "ActivityBufferCallback";
  // We're running on the main-thread;
  // we don't want to delay until Collect is called, since we'll end up keeping the libcupti buffer allocated,
  // which is especially bad if the buffer came from an Arena.

//  if (is_yes("TF_CUPTI_BUFFER_ARENA", false)) {
//    LOG(FATAL) << "Cannot use TF_CUPTI_BUFFER_ARENA=yes AND delay gathering of events from libcupti activity buffer, since arena will grow too large";
//  }

  activity_buffers_.push_back(std::move(activity_buffer));
}

void DeviceTracerImpl::ActivityCallback(const CUpti_Activity &record) {
  VLOG(2) << "ActivityCallback " << record.kind;
#ifdef CONFIG_TRACE_STATS
  if (is_yes("TF_CUPTI_PRINT_ACTIVITY", false)) {
    printActivity(&record);
  }

  mutex_lock l(trace_mu_);
  switch (record.kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      if (memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy *>(&record);
      memcpy_records_.emplace_back(
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      if (memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy2 *>(&record);
      memcpy_records_.emplace_back(
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes);
      break;
    }

      // IML TODO: record contribution of libcupti overhead to profiling time.
      // Overhead could be CUPTI, DRIVER, COMPILER, etc; documentation:
      //
      //   CUPTI_ACTIVITY_OVERHEAD_UNKNOWN = 0
      //     The overhead kind is not known.
      //   CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER = 1
      //     Compiler(JIT) overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH = 1<<16
      //     Activity buffer flush overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION = 2<<16
      //     CUPTI instrumentation overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE = 3<<16
      //     CUPTI resource creation and destruction overhead.
      //   CUPTI_ACTIVITY_OVERHEAD_FORCE_INT = 0x7fffffff


//        case CUPTI_ACTIVITY_KIND_OVERHEAD:
//          break;

    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
//          if (is_yes("TF_CUPTI_RECORD_PROFILING_OVERHEAD", true)) {
//            printActivity(&record);
//          }
      if (kernel_records_.size() >= kMaxRecords) return;
      auto *kernel = reinterpret_cast<const CUpti_ActivityKernel3 *>(&record);
      kernel_records_.emplace_back(
          kernel->start, kernel->end,
          kernel->deviceId, kernel->streamId,
          kernel->correlationId);
      break;
    }

    case CUPTI_ACTIVITY_KIND_OVERHEAD:
      if (!is_yes("TF_CUPTI_RECORD_PROFILING_OVERHEAD", true)) {
        break;
      }
      printActivity(&record);
      // LOG(INFO) << "libcupti: CUPTI_ACTIVITY_KIND_OVERHEAD event: ";
      break;

    default:
      VLOG(1) << "ActivityCallback unhandled kind";
      break;
  }
#else
  mutex_lock l(trace_mu_);
  switch (record.kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      if (memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy *>(&record);
      memcpy_records_.push_back(MemcpyRecord{
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes});
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      if (memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy2 *>(&record);
      memcpy_records_.push_back(MemcpyRecord{
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes});
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      if (kernel_records_.size() >= kMaxRecords) return;
      auto *kernel = reinterpret_cast<const CUpti_ActivityKernel3 *>(&record);
      kernel_records_.push_back(KernelRecord{kernel->start, kernel->end,
                                             kernel->deviceId, kernel->streamId,
                                             kernel->correlationId});
      break;
    }
    default:
      VLOG(1) << "ActivityCallback unhandled kind";
      break;
  }
#endif // CONFIG_TRACE_STATS
}

#ifdef CONFIG_TRACE_STATS
bool DeviceTracerImpl::IsEnabled() {
  mutex_lock l(mu_);
  return enabled_;
}
#endif // CONFIG_TRACE_STATS

Status DeviceTracerImpl::Collect() {
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is still enabled.");
  }

  // Collect is called from an async thread; now it's safe to take events out of the libcupti activity buffer without
  // blocking the main-thread.
  for (auto& activity_buffer : activity_buffers_) {
    activity_buffer->RecordActivitiesFromBuffer();
  }
  activity_buffers_.clear();

  bool use_arena = is_yes("TF_CUPTI_PROTOBUF_ARENA", false);

  // TODO(pbar) Handle device IDs and prefix properly.
  const string prefix = "";
  const int id = 0;
  const string stream_device =
      strings::StrCat(prefix, "/device:GPU:", id, "/stream:");
  const string memcpy_device =
      strings::StrCat(prefix, "/device:GPU:", id, "/memcpy");

  mutex_lock l2(trace_mu_);

  // SKIP until we start dumping to our own protobuf.
//  for (const auto &rec : kernel_records_) {
//    auto it = correlations_.find(rec.correlation_id);
//    const string name = (it != correlations_.cend()) ? it->second : "unknown";
//    NodeExecStats *ns = new NodeExecStats;
//    ns->set_all_start_micros(start_walltime_us_ +
//                             ((rec.start_timestamp - start_timestamp_) / 1000));
//    ns->set_op_start_rel_micros(0);
//    auto elapsed_us =
//        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
//    ns->set_op_end_rel_micros(elapsed_us);
//    ns->set_all_end_rel_micros(elapsed_us);
//    ns->set_node_name(name);
//    // TODO(pbar) Generate details based on the kernel activity record.
//    // ns->set_timeline_label(details);
//    auto nscopy = new NodeExecStats;
//    *nscopy = *ns;
//    collector->Save(strings::StrCat(stream_device, "all"), ns);
//    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
//  }
//  for (const auto &rec : memcpy_records_) {
//    auto it = correlations_.find(rec.correlation_id);
//    const string name = (it != correlations_.cend()) ? it->second : "unknown";
//    NodeExecStats *ns = new NodeExecStats;
//    ns->set_all_start_micros(start_walltime_us_ +
//                             ((rec.start_timestamp - start_timestamp_) / 1000));
//    ns->set_op_start_rel_micros(0);
//    auto elapsed_us =
//        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
//    ns->set_op_end_rel_micros(elapsed_us);
//    ns->set_all_end_rel_micros(elapsed_us);
//    auto copyKind = static_cast<CUpti_ActivityMemcpyKind>(rec.copyKind);
//    auto srcKind = static_cast<CUpti_ActivityMemoryKind>(rec.srcKind);
//    auto dstKind = static_cast<CUpti_ActivityMemoryKind>(rec.dstKind);
//    const string details = strings::Printf(
//        "MEMCPY%s %llu bytes (%s to %s)", getMemcpyKindString(copyKind),
//        rec.bytes, getMemoryKindString(srcKind), getMemoryKindString(dstKind));
//    ns->set_node_name(
//        strings::StrCat(name, ":MEMCPY", getMemcpyKindString(copyKind)));
//    ns->set_timeline_label(details);
//    auto nscopy = new NodeExecStats;
//    *nscopy = *ns;
//    collector->Save(memcpy_device, ns);
//    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
//  }

  return Status::OK();
}

}  // namespace devicetracer

std::unique_ptr<DeviceTracer> CreateDeviceTracer() {
  devicetracer::CUPTIManager *cupti_manager = devicetracer::GetCUPTIManager();
  if (cupti_manager == nullptr) {
    return nullptr;
  }
  std::unique_ptr<DeviceTracer> tracer(
      new devicetracer::DeviceTracerImpl(cupti_manager));
  return tracer;
}




}  // namespace tensorflow

