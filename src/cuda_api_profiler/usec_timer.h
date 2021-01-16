//
// Created by jagle on 11/13/2019.
//

#ifndef RLSCOPE_USEC_TIMER_H
#define RLSCOPE_USEC_TIMER_H

#include <cstdint>
#include <time.h>

//static constexpr uint64_t kMicrosToNanos = 1000ULL;
//static constexpr uint64_t kMillisToMicros = 1000ULL;
//static constexpr uint64_t kMillisToNanos = 1000ULL * 1000ULL;
//static constexpr uint64_t kSecondsToMillis = 1000ULL;
//static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
//static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

#define MY_kMicrosToNanos (1000ULL)
#define MY_kMillisToMicros (1000ULL)
#define MY_kMillisToNanos (1000ULL * 1000ULL)
#define MY_kSecondsToMillis (1000ULL)
#define MY_kSecondsToMicros (1000ULL * 1000ULL)
#define MY_kSecondsToNanos (1000ULL * 1000ULL * 1000ULL)

namespace rlscope {

static uint64_t TimeNowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * MY_kSecondsToNanos +
          static_cast<uint64_t>(ts.tv_nsec));
}

static uint64_t TimeNowMicros() {
  return TimeNowNanos() / MY_kMicrosToNanos;
}

static uint64_t TimeNowSeconds() {
  return TimeNowNanos() / MY_kSecondsToNanos;
}

}

#endif //RLSCOPE_USEC_TIMER_H
