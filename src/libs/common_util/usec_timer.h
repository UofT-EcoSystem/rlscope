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

static void SleepForMicroseconds(int64_t micros) {
  while (micros > 0) {
    timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 0;

    if (micros >= 1e6) {
      sleep_time.tv_sec =
          std::min<int64_t>(micros / 1e6, std::numeric_limits<time_t>::max());
      micros -= static_cast<int64_t>(sleep_time.tv_sec) * 1e6;
    }
    if (micros < 1e6) {
      sleep_time.tv_nsec = 1000 * micros;
      micros = 0;
    }
    while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
      // Ignore signals and wait for the full interval to elapse.
    }
  }
}

}

#endif //RLSCOPE_USEC_TIMER_H
