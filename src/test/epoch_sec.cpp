//
// Created by jagle on 2/19/2019.
//

#include <stdio.h>
#include <assert.h>
#include <time.h>

#include <gflags/gflags.h>
#include <iostream>
#include "src/common/util.h"

#include "src/common/debug.h"
//#include "src/common/util.h"

DEFINE_bool(debug, false, "Debug");
DEFINE_int32(repetitions, 10, "repetitions");

inline struct timespec get_monotonic_time(clockid_t clk, bool debug = false);

inline double monotonic_as_sec(struct timespec a) {
  double sec;
  sec = a.tv_sec + ( ((double)a.tv_nsec)/( (double)1e9 ) );
  return sec;
}

// If time doesn't change between iterations, ignore it (clock isn't precise).
// Else, report the time difference.
void print_clock_resolution(clockid_t clk, const char* clk_name, int repetitions) {
  printf("> Clock = %s\n", clk_name);
  std::vector<double> sec_diff;
  int i = 0;
  double prevtime = 0;
  double curtime;
  while (i < repetitions) {
    auto curtime_spec = get_monotonic_time(clk);
    curtime = monotonic_as_sec(curtime_spec);
    if (prevtime != 0) {
      auto difftime = curtime - prevtime;
//      assert(difftime > 0);
      if (difftime == 0) {
        LOG(INFO) << "WARNING: detected difftime == 0 for clk=" << clk_name;
      }
      sec_diff.push_back(difftime);
      i += 1;
    }
    prevtime = curtime;
  }
//  for (int i = 0; i < repetitions; ++i) {
//  }

  auto avg_sec_diff = Average(sec_diff);
  auto sd_sec_diff = Stdev(sec_diff);
  LOG(INFO) << "clk = "
      << clk_name
      << ", avg clock resolution = " << avg_sec_diff << " +/- " << sd_sec_diff
      << " sec ";
}

inline struct timespec get_monotonic_time(clockid_t clk, bool debug) {
  struct timespec t;
  double time_epoch_sec;
  clock_gettime(clk, &t);
  time_epoch_sec = monotonic_as_sec(t);
  if (debug) {
    printf("> %s:%i @ %s: "
           "EPOCH: "
           "t.tv_sec = %ld"
           ", t.tv_nsec = %ld"
           ", time_epoch_sec = %f"
           "\n"
        , __FILE__, __LINE__, __func__,
           t.tv_sec, t.tv_nsec, time_epoch_sec);
  }
  return t;
//    double time_sec = t.tv_sec + ((double)t.tv_nsec)/((double)1e9);
//    return time_sec;
}

int main(int argc, char *argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  print_clock_resolution(CLOCK_REALTIME, "CLOCK_REALTIME", FLAGS_repetitions);
  print_clock_resolution(CLOCK_MONOTONIC, "CLOCK_MONOTONIC", FLAGS_repetitions);
//  print_clock_resolution(CLOCK_MONOTONIC_RAW, "CLOCK_MONOTONIC_RAW", FLAGS_repetitions);

  std::cout << "> CLOCK_REALTIME:" << std::endl;
  get_monotonic_time(CLOCK_REALTIME, true);

  std::cout << "> CLOCK_MONOTONIC:" << std::endl;
  get_monotonic_time(CLOCK_MONOTONIC, true);

//  double sec = monotonic_as_sec(t);
  return 0;
}

