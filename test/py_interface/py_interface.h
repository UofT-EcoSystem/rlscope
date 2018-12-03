//
// Created by jagle on 11/27/2018.
//

#ifndef DNN_TENSORFLOW_CPP_PY_INTERFACE_H
#define DNN_TENSORFLOW_CPP_PY_INTERFACE_H

#endif //DNN_TENSORFLOW_CPP_PY_INTERFACE_H

#include "test/py_interface/py_interface_src.h"

extern "C" {

LibHandle *NewLibHandle() {
  return new LibHandle();
}
void call_c(LibHandle *lib){
  lib->call_c();
}

double guess_gpu_freq_mhz(LibHandle *lib){
  return lib->guess_gpu_freq_mhz();
}

double gpu_sleep(LibHandle *lib, double seconds){
  return lib->gpu_sleep(seconds);
}

double run_cpp(LibHandle *lib, double seconds){
  return lib->run_cpp(seconds);
}

void set_gpu_freq_mhz(LibHandle *lib, double mhz){
  lib->set_gpu_freq_mhz(mhz);
}

};
