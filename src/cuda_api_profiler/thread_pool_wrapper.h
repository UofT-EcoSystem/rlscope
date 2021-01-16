//
// Created by jagle on 8/6/2019.
//

#ifndef RLSCOPE_THREAD_POOL_WRAPPER_H
#define RLSCOPE_THREAD_POOL_WRAPPER_H

#include <string>
#include <mutex>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

namespace rlscope {

// Singleton class for dumping TraceDataProto asynchronously to avoid blocking python-side.
class ThreadPoolWrapper {
public:
  using Func = std::function<void()>;
  ThreadPoolWrapper(const std::string& name, int num_threads);
  void Schedule(Func fn);
  void AwaitAll();
private:
  void _ResetNotification();

  boost::asio::thread_pool async_dump_pool_;

  std::mutex mu_;
//  int fns_scheduled_;
//  int waiters_;
  std::string name_;
};

}

#endif //RLSCOPE_THREAD_POOL_WRAPPER_H
