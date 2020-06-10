//
// Created by jagle on 8/16/2019.
//

#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/defines.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"

#include <algorithm>

namespace rlscope {

// Sleep for at MOST 0.5 seconds.
// We need to be able to terminate quickly, so this sleep time affects that.
// In particular, in the worst case, the call to sample_cuda_api.disable_tracing()
// during Profiler.finish() will stall for this time.
#define MAX_SLEEP_FOR_USEC 5e6

void EventHandler::EventLoop(std::function<bool()> should_stop) {
  while (not should_stop()) {
    // GOAL:
    // - sleep until the next event has to run
    // - next_event_to_run = func such that func.time_until_next_run is minimized
    // - if time is negative, RunFuncs and skip sleeping.
    auto now_usec = Env::Default()->NowMicros();
    RunFuncs(now_usec);
    uint64_t sleep_for_us;
    if (_funcs.size() > 0) {
      auto next_func = std::min_element(
          _funcs.begin(), _funcs.end(),
          [now_usec] (const RegisteredFunc& f1, const RegisteredFunc& f2) {
            return f1.TimeUsecUntilNextRun(now_usec) < f2.TimeUsecUntilNextRun(now_usec); });
      auto us_until_next_func = next_func->TimeUsecUntilNextRun(now_usec);
      // Sleep for at MOST 0.5 seconds.
      // We need to be able to terminate quickly, so this sleep time affects that.
      const uint64_t max_sleep_for_usec = MAX_SLEEP_FOR_USEC;
      sleep_for_us = std::min(max_sleep_for_usec, us_until_next_func);
    } else {
      sleep_for_us = MAX_SLEEP_FOR_USEC;
    }
    Env::Default()->SleepForMicroseconds(sleep_for_us);
  }
}

uint64_t RegisteredFunc::TimeUsecUntilNextRun(uint64 now_usec) const {
  // every_sec seconds into the future, we would like to run this function.
  uint64_t next_run_usec = last_run_usec + every_sec*USEC_IN_SEC;
  if (next_run_usec >= now_usec) {
    // There's still some time to wait until this function is "due" to run.
    return next_run_usec - now_usec;
  }
  // This function is past-due to be run.
  // In this case, we return 0 meaning:
  //   "we should wait 0 seconds until running this function; run it now!"
  return 0;
}

bool RegisteredFunc::ShouldRun(uint64 now_usec) {
  DCHECK(last_run_usec <= now_usec);
  bool ret = last_run_usec == 0 or
             (now_usec - last_run_usec) >= (every_sec*USEC_IN_SEC);
//  if (ret) {
//    VLOG(1) << "now_usec = " << now_usec
//            << ", last_run_usec = " << last_run_usec
//            << ", time_usec since last ran = " << now_usec - last_run_usec
//            << ", run every usec = " << every_sec*USEC_IN_SEC;
//  }
  return ret;
}


void RegisteredFunc::Run(uint64 now_usec) {
  last_run_usec = now_usec;
  func();
}

RegisteredFunc::FuncId EventHandler::RegisterFunc(RegisteredFunc::Func func, float every_sec) {
  auto func_id = _next_func_id;
  _next_func_id += 1;
  _funcs.emplace_back(RegisteredFunc(func_id, func, every_sec));
  return func_id;
}

void EventHandler::UnregisterFunc(RegisteredFunc::FuncId func_id) {
  std::remove_if(_funcs.begin(), _funcs.end(),
                 [func_id](const RegisteredFunc& f) { return f.func_id == func_id; });
}

void EventHandler::RunFuncs(uint64 now_usec) {
//  auto now_usec = Env::Default()->NowMicros();
  for (auto& func : _funcs) {
    if (func.ShouldRun(now_usec)) {
      func.Run(now_usec);
    }
  }
}

}
