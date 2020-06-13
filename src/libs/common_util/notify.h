#pragma once
// FROM TENSORLFOW

#include <mutex>
#include <condition_variable>
#include <cassert>
#include <memory>
#include <atomic>

namespace rlscope {

class Notification {
public:
  Notification() : notified_(0) {}
  ~Notification() {
    // In case the notification is being used to synchronize its own deletion,
    // force any prior notifier to leave its critical section before the object
    // is destroyed.
    std::unique_lock<std::mutex> l(mu_);
  }

  void Notify() {
    std::unique_lock<std::mutex> l(mu_);
    assert(!HasBeenNotified());
    notified_.store(true, std::memory_order_release);
    cv_.notify_all();
  }

  bool HasBeenNotified() const {
    return notified_.load(std::memory_order_acquire);
  }

  void WaitForNotification() {
    if (!HasBeenNotified()) {
      std::unique_lock<std::mutex> l(mu_);
      while (!HasBeenNotified()) {
        cv_.wait(l);
      }
    }
  }

private:
  friend bool WaitForNotificationWithTimeout(Notification* n,
                                             int64_t timeout_in_us);
  bool WaitForNotificationWithTimeout(int64_t timeout_in_us) {
    bool notified = HasBeenNotified();
    if (!notified) {
      std::unique_lock<std::mutex> l(mu_);
      do {
        notified = HasBeenNotified();
      } while (!notified &&
               cv_.wait_for(l, std::chrono::microseconds(timeout_in_us)) !=
               std::cv_status::timeout);
    }
    return notified;
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<bool> notified_;
};

inline bool WaitForNotificationWithTimeout(Notification* n,
                                           int64_t timeout_in_us) {
  return n->WaitForNotificationWithTimeout(timeout_in_us);
}

} // namespace rlscope
