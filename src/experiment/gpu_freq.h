//
// Created by jgleeson on 2020-01-23.
//

#ifndef IML_GPU_FREQ_H
#define IML_GPU_FREQ_H

#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include "analysis/my_status.h"

//#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (10)
#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (5)
#define GPU_CLOCK_REPETITIONS (3)
#define GPU_CLOCK_INIT_GPU_SLEEP_CYCLES (1000)
#define GPU_CLOCK_MIN_GUESS_TIME_SEC (1)

namespace tensorflow {

template <class Container>
double Average(Container& buffer) {
  double summation = 0;
  int n = 0;
  for (auto x : buffer) {
    summation += x;
    n += 1;
  }
  return summation / ((double) n);
}

template <class Container>
double Std(Container& buffer) {
  auto avg = Average(buffer);
  double summation = 0;
  int n = 0;
  for (auto x : buffer) {
    summation += pow(x - avg, 2);
    n += 1;
  }
  return sqrt(summation / ((double) n));
}


//struct GPUClockResult {
//  double avg_mhz;
//  double std_mhz;
//};
class GPUClockFreq {
public:
  using clock_value_t = long long;
// https://www.softwariness.com/articles/monotonic-clocks-windows-and-posix/
//  using std::chrono::steady_clock;
  using time_type = std::chrono::steady_clock::time_point;

  std::vector<double> _time_secs;
  std::vector<double> _freq_mhz;
  double _avg_mhz;
  double _std_mhz;
//  std::vector<clock_value_t> time_cycles;
  // Number of cycles to sleep for on the GPU.
  clock_value_t _sleep_cycles;
  int _repetitions;
//  GPUClockResult _result;
  std::string _directory;

  GPUClockFreq(int repetitions, const std::string& directory) :
      _avg_mhz(0.)
      , _std_mhz(0.)
      , _sleep_cycles(GPU_CLOCK_INIT_GPU_SLEEP_CYCLES)
      , _repetitions(repetitions)
      , _directory(directory)
  {
  }

  void guess_cycles();

  static time_type now();

  static double elapsed_sec(time_type start, time_type stop);

  void iter(time_type *start_t, time_type *end_t);

  double freq_mhz(double time_sec);

  static double gpu_sleep(clock_value_t sleep_cycles);
  void run();

  MyStatus dump_json() const;
  MyStatus load_json(const std::string &path);
  std::string json_path() const;
  std::string json_basename() const;
};

} // namespace tensorflow

#endif //IML_GPU_FREQ_H
