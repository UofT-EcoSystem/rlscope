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

#ifndef TENSORFLOW_CORE_PLATFORM_DEVICE_TRACER_H_
#define TENSORFLOW_CORE_PLATFORM_DEVICE_TRACER_H_

#include <memory>

#include "common_util.h"

#include <cuda.h>
#include <cupti_target.h>
#include <cupti.h>

#define CONFIG_TRACE_STATS

namespace rlscope {

class StepStatsCollector;

// 'DeviceTracer' is an interface for collecting low-level execution timings
// of hardware accelerator (e.g. GPU) computation and DMA transfers.
//
// Typical usage pattern is as follows:
//
// DeviceTracer* tracer = CreateDeviceTracer();
// if (tracer) {
//   tracer->Start();
//
//   ... perform some computations on a hardware accelerator.
//
//   tracer->Stop();
//
//   StepStats stats;
//   StepStatsCollector collector(&stats);
//   tracer->Collect(&collector);
// }
//
// Notes:
// Tracing is not supported on all plaforms.  On platforms
// with no tracing support, 'CreateDeviceTracer' will return 'nullptr'.
// On most plaforms, hardware tracing will be a system-wide activity and
// a single 'DeviceTracer' will collect activity from all devices.
// It is also common that only a single tracer may be active at any
// given time.  The 'Start' method will return an error if tracing is
// already in progress elsewhere.
//
class DeviceTracer {
 public:
  virtual ~DeviceTracer() {}

  // Start device tracing.
  // Note that only a single trace can be active, in which case this
  // methods will return an 'Unavailable' error.
  virtual MyStatus Start() = 0;

  // Stop device tracing.
  // It is safe to call 'Stop' on a tracer which is not enabled.
  virtual MyStatus Stop() = 0;
  virtual MyStatus DisableGpuHW() = 0;

  virtual MyStatus Print() = 0;
  virtual MyStatus SetMetadata(const char* directory, const char* process_name, const char* machine_name, const char* phase_name) = 0;
  virtual MyStatus AsyncDump() = 0;
  virtual MyStatus AwaitDump() = 0;
  virtual MyStatus RecordEvent(
      const char* category,
      int64_t start_us,
      int64_t duration_us,
      const char* name) = 0;

  virtual MyStatus StartPass() = 0;
  virtual MyStatus EndPass() = 0;
  virtual MyStatus HasNextPass(bool* has_next_pass) = 0;
  virtual MyStatus PushOperation(const char* operation) = 0;
  virtual MyStatus RecordOverheadEvent(
      const char* overhead_type,
      int64_t num_events) = 0;
  virtual MyStatus RecordOverheadEventForOperation(
      const char* overhead_type,
      const char* operation,
      int64_t num_events) = 0;
  virtual MyStatus PopOperation() = 0;
  virtual MyStatus SetMaxOperations(const char* operation, int num_pushes) = 0;

  // Collect trace results.  Results are added to the specified
  // StepStatsCollector.  Does not clear any existing stats.
  // It is an error to call 'Collect' while a trace is running.
//  virtual MyStatus Collect() = 0;

#ifdef CONFIG_TRACE_STATS
  virtual bool IsEnabled() = 0;
#endif
};

// Creates a platform-specific DeviceTracer.
// Returns 'nullptr' on platforms where tracing is not supported.
std::unique_ptr<DeviceTracer> CreateDeviceTracer();

}  // namespace rlscope

#endif  // TENSORFLOW_CORE_PLATFORM_DEVICE_TRACER_H_
