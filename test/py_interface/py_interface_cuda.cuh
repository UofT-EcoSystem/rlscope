//
// Created by jagle on 11/27/2018.
//

#ifndef DNN_TENSORFLOW_CPP_PY_INTERFACE_CUDA_H
#define DNN_TENSORFLOW_CPP_PY_INTERFACE_CUDA_H

#include <cmath>
#include <chrono>
#include <vector>

//#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (10)
#define GPU_CLOCK_MIN_SAMPLE_TIME_SEC (5)
#define GPU_CLOCK_REPETITIONS (3)
#define GPU_CLOCK_INIT_GPU_SLEEP_CYCLES (1000)
#define GPU_CLOCK_MIN_GUESS_TIME_SEC (1)

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


struct GPUClockResult {
  double avg_mhz;
  double std_mhz;
};
class GPUClockFreq {
public:
  using clock_value_t = long long;
// https://www.softwariness.com/articles/monotonic-clocks-windows-and-posix/
//  using std::chrono::steady_clock;
  using time_type = std::chrono::steady_clock::time_point;

  std::vector<double> _time_secs;
  std::vector<double> _freq_mhz;
  std::vector<clock_value_t> time_cycles;
  // Number of cycles to sleep for on the GPU.
  clock_value_t _sleep_cycles;
  int _repetitions;
  GPUClockResult _result;

  GPUClockFreq(int repetitions) :
      _sleep_cycles(GPU_CLOCK_INIT_GPU_SLEEP_CYCLES)
      , _repetitions(repetitions)
  {
  }

  void guess_cycles();

  static time_type now();

  static double elapsed_sec(time_type start, time_type stop);

  void iter(time_type *start_t, time_type *end_t);

  double freq_mhz(double time_sec);

  static void gpu_sleep(clock_value_t sleep_cycles);
  GPUClockResult run();
};

#endif //DNN_TENSORFLOW_CPP_PY_INTERFACE_CUDA_H
