//
// Created by jagle on 11/27/2018.
//

#include "test/py_interface/py_interface_src.h"
#include "test/py_interface/py_interface.h"
#include "test/py_interface/py_interface_cuda.cuh"

#include "tensorflow/core/platform/logging.h"

#include <iostream>
#include <cassert>
#include <cmath>

void LibHandle::call_c() {
  std::cout << "Hello world from C++ (called from python)" << std::endl;
}

double LibHandle::guess_gpu_freq_mhz() {
  std::cout << "Guessing GPU clock frequency." << std::endl;
  int repetitions = GPU_CLOCK_REPETITIONS;
  GPUClockFreq clock_freq(repetitions);
  GPUClockResult result = clock_freq.run();
  _clock_result = result;
  _has_clock_result = true;
  return result.avg_mhz;
}

void LibHandle::gpu_sleep(double seconds) {
  assert(_has_clock_result);
  // cycles = cycles/sec * seconds
  GPUClockFreq::clock_value_t cycles = ceil(_clock_result.avg_mhz * seconds);
  std::cout << "> Sleeping on GPU for " << seconds << " seconds (" << cycles << " cycles @ " << _clock_result.avg_mhz << " MHz)." << std::endl;
  auto start_t = GPUClockFreq::now();
  GPUClockFreq::gpu_sleep(cycles);
  auto end_t = GPUClockFreq::now();
  auto time_sec = GPUClockFreq::elapsed_sec(start_t, end_t);
  GPUClockFreq::gpu_sleep(cycles);
  std::cout << "> Slept on GPU for " << time_sec << " seconds." << std::endl;
}

void LibHandle::run_cpp(double seconds) {
  std::cout << "> Running inside C++ for " << seconds << " seconds" << std::endl;
  auto start_t = GPUClockFreq::now();
  while (true) {
    auto end_t = GPUClockFreq::now();
    auto time_sec = GPUClockFreq::elapsed_sec(start_t, end_t);
    if (time_sec >= seconds) {
      break;
    }
  }
}
void LibHandle::set_gpu_freq_mhz(double mhz) {
  _has_clock_result = true;
  _clock_result = GPUClockResult{.avg_mhz=mhz, .std_mhz=0};
}
