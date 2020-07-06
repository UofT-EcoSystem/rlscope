//
// Created by jgleeson on 2020-07-03.
//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#pragma GCC diagnostic pop

#include <iostream>
#include <cassert>

#include "generic_logging.h"

#include "concurrency.h"

namespace rlscope {

InterProcessBarrier::InterProcessBarrier(size_t num_threads) :
    num_threads(num_threads),
    n_threads_left(num_threads),
    generation(0) {
}

void InterProcessBarrier::arrive_and_wait(const std::string &reason) {
  boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(mutex);
  auto gen = generation;
  assert(n_threads_left >= 1);

  if (n_threads_left == num_threads) {
    assert(reason.size() < BARRIER_NAME_MAX_LEN);
    strncpy(barrier_reason, reason.c_str(), reason.size());
    barrier_reason[reason.size()] = '\0';
  } else {
    if (reason != barrier_reason) {
      std::cerr << "ERROR: saw thread arriving at barrier with reason=\"" << reason << "\", but there are already "
                << (num_threads - n_threads_left) << " threads waiting at barrier with reason=\"" << barrier_reason
                << "\""
                << std::endl;
      assert(reason == barrier_reason);
    }
  }

  n_threads_left -= 1;
  if (n_threads_left == 0) {
    n_threads_left = num_threads;
    generation += 1;
    barrier_limit_break.notify_all();
  } else {
    barrier_limit_break.wait(lock, [this, gen] {
      return gen != this->generation;
    });
  }
}

}
