//
// Created by jgleeson on 2020-07-03.
//

#pragma once

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

namespace rlscope {

//#define BARRIER_NAME_MAX_LEN 256
int constexpr BARRIER_NAME_MAX_LEN = 256;

struct InterProcessBarrier {
  boost::interprocess::interprocess_mutex mutex;
  boost::interprocess::interprocess_condition barrier_limit_break;

  // Current generation's barrier_reason.
  char barrier_reason[BARRIER_NAME_MAX_LEN];

  size_t num_threads;
  // Number of threads that need to call arrive_and_wait before all threads are awoken,
  // for the CURRENT <generation>.
  size_t n_threads_left;

  // https://stackoverflow.com/a/27118537
  // Need this to make the barrier "multi-use".
  size_t generation;

  InterProcessBarrier(size_t num_threads);

  void arrive_and_wait(const std::string &reason);

  template<typename OStream>
  void Print(OStream &out, int indent) const {
    PrintIndent(out, indent);
    out << "InterProcessBarrier(n_threads_left=" << n_threads_left << ", num_threads=" << num_threads << ", generation="
        << generation << ")";
  }

  template<typename OStream>
  friend OStream &operator<<(OStream &os, const InterProcessBarrier &obj) {
    obj.Print(os, 0);
    return os;
  }

};

}
