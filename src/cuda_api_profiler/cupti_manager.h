//
// Created by jagle on 8/23/2019.
//

#ifndef IML_CUPTI_MANAGER_H
#define IML_CUPTI_MANAGER_H

#include "cuda_api_profiler/registered_handle.h"
#include "cuda_api_profiler/cupti_manager.h"

#include <cuda.h>
#include <cupti_target.h>
#include <cupti.h>

#include <mutex>

namespace rlscope {

class CUPTIManager;

class SimpleByteAllocationPool {
public:
  SimpleByteAllocationPool(int max_num_allocations, size_t allocation_size, size_t allocation_alignment);
  char* _NewBuffer();
  char* AllocateBuffer();
  void FreeBuffer(char* buffer);
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

  ~SimpleByteAllocationPool();
};

// Returns a pointer to the CUPTIManager singleton.
CUPTIManager *GetCUPTIManager();

// Callback interface for consumers of CUPTI tracing.
class CUPTIClient {
public:
  virtual ~CUPTIClient() {}

  // Invoked for each CUPTI activity reported.
  virtual void ActivityCallback(const CUpti_Activity &activity) = 0;
//  virtual void ActivityBufferCallback(std::unique_ptr<ActivityBuffer> activity_buffer) = 0;
};

// 32MB
// PROBLEM:
// - we delay processing libcupti buffers so they won't be freed until much later...
// - allocation pool needs to be thread-safe.
#define MAX_CUPTI_BUFFERS_ALLOCATED 1024

// Singleton class to manage registration of CUPTI callbacks,
// and managing cupti memory buffers.
class CUPTIManager {
public:
  SimpleByteAllocationPool allocation_pool_;

  CUPTIManager();

  void FreeBufferCallback(uint8_t *_buffer);
  void RecordActivityCallback(CUPTIClient* client, const CUpti_Activity &record);

  static CUPTIManager *Create();

  // Enables tracing and delivers event callbacks to 'client'.
  // Does not take ownership of client.  Client's lifetime must persist
  // until tracing is disabled.
  MyStatus EnableTrace(CUPTIClient *client);
  MyStatus _EnablePCSampling();
  MyStatus _DisablePCSampling();
  MyStatus Flush();

  // Disable tracing.  No further events will be delivered to 'client'.
  MyStatus DisableTrace();

private:
  // Static functions which we can use as CUPTI callbacks.
  static void BufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords);
  static void BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize);
  // These methods are called by the static stubs above.
  void InternalBufferRequested(uint8_t **buffer, size_t *size,
                               size_t *maxNumRecords);
  void InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                               uint8_t *buffer, size_t size, size_t validSize);

  // Size of buffers used for CUPTI tracing.
  static constexpr size_t kBufferSize = 32 * 1024;
  // Required alignment of CUPTI buffers.
  static constexpr size_t kBufferAlignment = 8;

  std::mutex mu_;
  CUPTIClient *client_;
//  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;

  TF_DISALLOW_COPY_AND_ASSIGN(CUPTIManager);
};

}

#endif //IML_CUPTI_MANAGER_H
