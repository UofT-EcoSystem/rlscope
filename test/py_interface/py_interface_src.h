//
// Created by jagle on 11/27/2018.
//

#ifndef DNN_TENSORFLOW_CPP_PY_INTERFACE_SRC_H
#define DNN_TENSORFLOW_CPP_PY_INTERFACE_SRC_H

#include "test/py_interface/py_interface_cuda.cuh"

class LibHandle {
public:
  GPUClockResult _clock_result;
  bool _has_clock_result;
  LibHandle() : _has_clock_result(false) {
  }
  void call_c();
  double guess_gpu_freq_mhz();
  double gpu_sleep(double seconds);
  double run_cpp(double seconds);
  void set_gpu_freq_mhz(double mhz);
};

#endif //DNN_TENSORFLOW_CPP_PY_INTERFACE_SRC_H
