//
// Created by jagle on 8/6/2019.
//

#ifndef IML_THREAD_POOL_WRAPPER_H
#define IML_THREAD_POOL_WRAPPER_H

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// Singleton class for dumping TraceDataProto asynchronously to avoid blocking python-side.
class ThreadPoolWrapper {
public:
    using Func = std::function<void()>;
    ThreadPoolWrapper(const string& name, int num_threads);
    void Schedule(Func fn);
    void AwaitAll();
private:
    void _ResetNotification();

    thread::ThreadPool async_dump_pool_;
    std::unique_ptr<Notification> all_done_;
    mutex mu_;
    int fns_scheduled_ GUARDED_BY(mu_);
    int waiters_ GUARDED_BY(mu_);
};

}

#endif //IML_THREAD_POOL_WRAPPER_H
