//
// Created by jagle on 8/16/2019.
//

#include "cuda_api_profiler/event_handler.h"
#include "cuda_api_profiler/defines.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

void EventHandler::EventLoop(std::function<bool()> should_stop) {
  while (not should_stop()) {
    // GOAL:
    // - sleep until the next event has to run
    // - next_event_to_run = func such that func.time_until_next_run is minimized
    // - if time is negative, RunFuncs and skip sleeping.
    auto now_usec = Env::Default()->NowMicros();
    RunFuncs(now_usec);
    auto next_func = std::min_element(
        _funcs.begin(), _funcs.end(),
        [now_usec] (const RegisteredFunc& f1, const RegisteredFunc& f2) {
          return f1.TimeUsecUntilNextRun(now_usec) < f2.TimeUsecUntilNextRun(now_usec); });
    auto sleep_for_us = next_func->TimeUsecUntilNextRun(now_usec);
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
