//
// Created by jagle on 8/12/2019.
//

#include <unistd.h>
#include <cmath>
#include <list>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <sstream>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"

#include "cuda_api_profiler/pc_sampling.h"
#include "cuda_api_profiler/defines.h"

namespace rlscope {

using namespace iml;

//#define VECTOR_FOR_EACH_SAMPLE(data, list_member)
//    data->list_member.resize(list_member.size());
//    for (size_t i = 0; i < list_member.size(); i++) {
//      data->list_member[i].Sample(&list_member[i]);
//    }

#define LIST_FOR_EACH_SAMPLE(data, list_member) do { \
    data->list_member.resize(list_member.size()); \
    auto data_it = data->list_member.begin(); \
    auto member_it = list_member.begin(); \
    while (data_it != data->list_member.end() && \
           member_it != list_member.end()) { \
        (*data_it).Sample(&(*member_it)); \
        data_it++; \
        member_it++; \
    } \
} while (0);

#define VECTOR_FOR_EACH_SAMPLE(data, list_member) LIST_FOR_EACH_SAMPLE(data, list_member)

thread_local ThreadInfo tinfo;

SampleAPI sample_api;

//https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-output-of-command-within-c-using-posix
#define MAX_CMD_SIZE 256
std::string run_cmd(const char* cmd) {
  DCHECK(strlen(cmd) < MAX_CMD_SIZE);
  std::array<char, MAX_CMD_SIZE> buffer;

  std::stringstream ss;
//  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
//    result += buffer.data();
    ss << buffer.data();
  }
  return ss.str();
//  return result;
}

std::string get_cpu_name() {
  return run_cmd("cat /proc/cpuinfo | grep 'model name' | head -n 1 | sed 's|^\\s*model name\\s*:\\s*||' | tr -d '\\n'");
}

void SampleEvents::SetAttrs(const std::string& process_name, const std::string& phase, const std::string& machine_name) {
  this->process_name = process_name;
  this->phase = phase;
  this->machine_name = machine_name;
}

SampleEventsProto SampleEvents::AsProto() const {
  SampleEventsProto proto;
  proto.set_process_name(process_name);
  proto.set_phase(phase);
  proto.set_machine_name(machine_name);

  auto event_proto = proto.add_events();
  auto event_as_proto = event.AsProto();
  event_proto->Swap(&event_as_proto);

//  for (auto const& sample_event : events) {
//    auto event_proto = proto.add_events();
//    auto event_as_proto = sample_event.AsProto();
//    event_proto->Swap(&event_as_proto);
//  }

  return proto;
}


AnnotationProto Annotation::AsProto() const {
  AnnotationProto proto;
  proto.set_name(name);
  proto.set_category(category);
  return proto;
}

int SampleAPI::SetThreadID() {
  mutex_lock lock(_mu);
  return _SetThreadID();
}
int SampleAPI::_SetThreadID() {
  auto id = _next_thread_id;
  pid_t tid = gettid();
  _tid_to_id[tid] = id;
  _next_thread_id += 1;

  DCHECK(_sample_events.event.cpu_sample_state.size() == static_cast<size_t>(id));
  _sample_events.event.cpu_sample_state.emplace_back(tid, id);

  return id;
}

int SampleAPI::SetGPUID(const std::string& device_name) {
  mutex_lock lock(_mu);
  return _SetGPUID(device_name);
}
int SampleAPI::_SetGPUID(const std::string& device_name) {
  auto id = _next_gpu_id;
  _dev_to_dev_id[device_name] = id;
  _next_gpu_id += 1;

  DCHECK(_sample_events.event.gpu_sample_state.size() == static_cast<size_t>(id));
  _sample_events.event.gpu_sample_state.emplace_back(device_name);

  return id;
}

void SampleAPI::MarkGPUActive(const std::string& device_name) {
  mutex_lock lock(_mu);

  auto it = _dev_to_dev_id.find(device_name);
  int dev_id;
  if (it == _dev_to_dev_id.end()) {
    dev_id = _SetGPUID(device_name);
  } else {
    dev_id = it->second;
  }

  _sample_events.event.gpu_sample_state[dev_id].MarkGPUActive();
}

SampleEventProto SampleEvent::AsProto() const {
  SampleEventProto proto;
  proto.set_sample_time_us(sample_time_us);

  for (auto const& sample_state : cpu_sample_state) {
    auto sample_state_proto = proto.add_cpu_sample_state();
    auto sample_state_as_proto = sample_state.AsProto();
    sample_state_proto->Swap(&sample_state_as_proto);
  }

  for (auto const& sample_state : gpu_sample_state) {
    auto sample_state_proto = proto.add_gpu_sample_state();
    auto sample_state_as_proto = sample_state.AsProto();
    sample_state_proto->Swap(&sample_state_as_proto);
  }

  return proto;
}

CPUSampleState::CPUSampleState(pid_t tid, int64 thread_id) :
    tid(tid),
    thread_id(thread_id),
    device_name(get_cpu_name())
{
}

void CPUSampleState::Push(const std::string& name, const std::string& category) {
  annotations.emplace_back(name, category);
}

void CPUSampleState::Pop() {
  annotations.pop_back();
}

CPUSampleStateProto CPUSampleState::AsProto() const {
  CPUSampleStateProto proto;
  proto.set_device_name(device_name);
  proto.set_tid(tid);
  proto.set_thread_id(thread_id);

  for (auto const& annotation : annotations) {
    auto annotation_proto = proto.add_annotations();
    auto annotation_as_proto = annotation.AsProto();
    annotation_proto->Swap(&annotation_as_proto);
  }

  return proto;
}

GPUSampleStateProto GPUSampleState::AsProto() const {
  GPUSampleStateProto proto;
  proto.set_device_name(device_name);
  proto.set_is_gpu_active(is_gpu_active);

  return proto;
}

void SampleEvents::Sample(SampleEvents* data) {
  data->process_name = process_name;
  data->phase = phase;
  data->machine_name = machine_name;

  data->event.Sample(&event);
//  FOR_EACH_SAMPLE(data, events);

}

void Annotation::Sample(Annotation* data) {
  data->name = name;
  data->category = category;
}

void SampleEvent::Sample(SampleEvent* data) {
  data->sample_time_us = sample_time_us;
  VECTOR_FOR_EACH_SAMPLE(data, cpu_sample_state);
  VECTOR_FOR_EACH_SAMPLE(data, gpu_sample_state);
}

void CPUSampleState::Sample(CPUSampleState* data) {
  data->tid = tid;
  data->thread_id = thread_id;
  data->device_name = device_name;
  LIST_FOR_EACH_SAMPLE(data, annotations);
}

void GPUSampleState::Sample(GPUSampleState* data) {
  data->device_name = device_name;
  data->is_gpu_active = is_gpu_active;
}

SampleEvents SampleAPI::Sample() {
  mutex_lock lock(_mu);
  SampleEvents sample_events;
  sample_events.Sample(&_sample_events);
  return sample_events;
}

void SampleAPI::MaybeSetThreadID(ThreadInfo* tinfo) {
  if (tinfo->thread_id == THREAD_ID_UNSET) {
    mutex_lock lock(_mu);
    tinfo->thread_id = sample_api._SetThreadID();
  }
}

void SampleAPI::_MaybeSetThreadID(ThreadInfo* tinfo) {
  if (tinfo->thread_id == THREAD_ID_UNSET) {
    tinfo->thread_id = sample_api._SetThreadID();
  }
}

ThreadInfo* SampleAPI::_CurrentThreadInfo() {
  sample_api._MaybeSetThreadID(&tinfo);
  return &tinfo;
}

void SampleAPI::Push(const std::string& name, const std::string& operation) {
  mutex_lock lock(_mu);
  auto tinfo = _CurrentThreadInfo();
  _sample_events.event.cpu_sample_state[tinfo->thread_id].Push(name, operation);
}

void SampleAPI::Pop() {
  mutex_lock lock(_mu);
  auto tinfo = _CurrentThreadInfo();
  _sample_events.event.cpu_sample_state[tinfo->thread_id].Pop();
}

void SampleAPI::_RunMakeGPUInactive(GPUSampleState* gpu_sample_state) {
  mutex_lock lock(_mu);
//  _mu.lock();
  while (true) {
    while (!gpu_sample_state->is_gpu_active && !gpu_sample_state->_sync_state->_should_stop.HasBeenNotified()) {
      gpu_sample_state->_sync_state->_cv_is_gpu_active.wait(lock);
    }
    if (gpu_sample_state->_sync_state->_should_stop.HasBeenNotified()) {
      // Program is exiting; break.
//      _mu.unlock();
      break;
    }
    if (gpu_sample_state->is_gpu_active) {
      auto last_marked_active_usec_before_sleep = sample_api._sample_events.event.sample_time_us;
      // TODO: we want to sleep for at least the sampling interval, plus some fudge factor.
      // Currently we just sleep for a sampling-interval and a half.
      // However, currently the sampling interval determines how accurate our "GPU active"
      // reading is...
      // Ideally, the CUDA API would inform us IMMEDIATELY when the GPU becomes inactive...
      int64 sleep_for_usec = round(1.5*_sample_every_sec * USEC_IN_SEC);
      _mu.unlock();
      Env::Default()->SleepForMicroseconds(sleep_for_usec);
      _mu.lock();
      auto last_marked_active_usec_after_sleep = sample_api._sample_events.event.sample_time_us;
      if (last_marked_active_usec_after_sleep == last_marked_active_usec_before_sleep) {
        // We slept through the sampling-interval time, but the GPU wasn't marked as active again.
        // Assume it's inactive.
        gpu_sample_state->is_gpu_active = false;
        sample_api._sample_events.event.sample_time_us = Env::Default()->NowMicros();
      }
    }
  }
}

void GPUSampleState::MarkGPUActive() {
  is_gpu_active = true;
  _sync_state->_cv_is_gpu_active.notify_all();
}

GPUSampleState::GPUSampleState(const std::string& device_name) :
    device_name(device_name)
    , is_gpu_active(false)
{
  GPUSampleState* self = this;
  _sync_state.reset(new SyncState);
  _sync_state->_make_gpu_inactive_thread.reset(new std::thread([self] {
    sample_api._RunMakeGPUInactive(self);
  }));
}


GPUSampleState::SyncState::~SyncState() {
  if (_make_gpu_inactive_thread) {
    if (!_should_stop.HasBeenNotified()) {
      _should_stop.Notify();
    }
    _make_gpu_inactive_thread->join();
  }
}

}
