//
// Created by jagle on 8/19/2019.
//

#ifndef IML_GLOBALS_H
#define IML_GLOBALS_H

#include "tensorflow/core/platform/device_tracer.h"

#include "cuda_api_profiler/cuda_ld_preload.h"

namespace tensorflow {

class Globals {
public:
  Globals();
  ~Globals();

#ifdef WITH_CUDA_LD_PRELOAD
//  std::shared_ptr<CudaLibrary> cuda_library;
  CudaLibrary* cuda_library;
#endif
  std::unique_ptr<tensorflow::DeviceTracer> device_tracer;
};

extern Globals globals;

}

#endif //IML_GLOBALS_H
