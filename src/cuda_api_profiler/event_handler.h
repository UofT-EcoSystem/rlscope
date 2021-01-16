//
// Created by jagle on 8/16/2019.
//

#ifndef RLSCOPE_EVENT_HANDLER_H
#define RLSCOPE_EVENT_HANDLER_H

#include <list>
#include <functional>

namespace rlscope {

struct RegisteredFunc {
  using FuncId = int;
  using Func = std::function<void()>;

  FuncId func_id;
  Func func;
  uint64_t last_run_usec;
  float every_sec;
  RegisteredFunc(FuncId func_id, Func func, float every_sec) :
      func_id(func_id)
      , func(func)
      , last_run_usec(0)
      , every_sec(every_sec)
  {
  }
  bool ShouldRun(uint64_t now_usec);
  void Run(uint64_t now_usec);
  uint64_t TimeUsecUntilNextRun(uint64_t now_usec) const;
};

class EventHandler {
public:
  EventHandler() :
      _next_func_id(0)
  {
  }
  RegisteredFunc::FuncId RegisterFunc(RegisteredFunc::Func func, float every_sec);
  void UnregisterFunc(RegisteredFunc::FuncId func_id);
  void RunFuncs(uint64_t now_usec);
  void EventLoop(std::function<bool()> should_stop);
  std::list<RegisteredFunc> _funcs;
  RegisteredFunc::FuncId _next_func_id;
};

}

#endif //RLSCOPE_EVENT_HANDLER_H
