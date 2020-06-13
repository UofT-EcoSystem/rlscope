//
// Created by jagle on 8/6/2019.
//

#include <string>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include "cuda_api_profiler/thread_pool_wrapper.h"

namespace rlscope {

ThreadPoolWrapper::ThreadPoolWrapper(const std::string& name, int num_threads) :
    async_dump_pool_(num_threads),
//    fns_scheduled_(0),
//    waiters_(0),
    name_(name)
{
}

void ThreadPoolWrapper::Schedule(Func fn) {

//  boost::asio::post(async_dump_pool_, [] () mutable {
//    MyStatus ret = MyStatus::OK();
//    ret = proto_state.DumpSync();
//    if (!ret.ok()) {
//      std::stringstream ss;
//      ss << "Failed to dump GPU hw sampling state " << proto_state.DumpPath() << " asynchronously";
//      IF_BAD_STATUS_EXIT(ss.str(), ret);
//    }
//  });

//  {
//    std::lock_guard<std::mutex> l(mu_);
//    fns_scheduled_ += 1;
//    VLOG(1) << "ThreadPoolWrapper.name = " << name_ << ", fns_scheduled = " << fns_scheduled_;
//  }
  boost::asio::post(async_dump_pool_, fn);

//  boost::asio::post(async_dump_pool_, [fn, this] () {
//    // Run the function.
//    fn();
//
//    {
//      // Notify python-thread waiting for async dumping to finish.
//      std::lock_guard<std::mutex> l(mu_);
//
//      fns_scheduled_ -= 1;
//      if (fns_scheduled_ == 0) {
//        // We were the last scheduled function to run;
//        // awake anyone that's waiting for us to finish.
//        all_done_->Notify();
//      }
//
//      if (waiters_ == 0 && fns_scheduled_ == 0) {
//        // Notification is one-time use; we need to re-allocate it.
//        _ResetNotification();
//      }
//    }
//  });

}

void ThreadPoolWrapper::AwaitAll() {
  async_dump_pool_.join();
//    {
//        std::lock_guard<std::mutex> l(mu_);
//        if (fns_scheduled_ == 0) {
//            VLOG(1) << "ThreadPoolWrapper.name = " << name_ << ", fns_scheduled = 0, return";
//            return;
//        }
//        waiters_ += 1;
//        VLOG(1) << "ThreadPoolWrapper.name = " << name_ << ", fns_scheduled = " << fns_scheduled_ << ", waiters = " << waiters_;
//    }
//    all_done_->WaitForNotification();
//    {
//        std::lock_guard<std::mutex> l(mu_);
//        // The main python thread is the only one that can schedule
//        // new dumps, and wait for dumps to finish.
//        //
//        // No other thread should be adding dumps while we were waiting.
//        // So the only reason we wake up is if all dumps are finished.
//        CHECK(fns_scheduled_ == 0);
//        waiters_ -= 1;
//        if (waiters_ == 0) {
//            _ResetNotification();
//        }
//    }
}

//void ThreadPoolWrapper::_ResetNotification() {
//    CHECK(waiters_ == 0);
//    CHECK(fns_scheduled_ == 0);
//    all_done_.reset(new Notification());
//}

}
