//
// Created by jagle on 8/19/2019.
//

#ifndef IML_GLOBALS_H
#define IML_GLOBALS_H

#include <boost/process.hpp>

#include "cuda_api_profiler/device_tracer.h"

#include "cuda_api_profiler/cuda_ld_preload.h"

namespace rlscope {

class Globals {
public:
  Globals();
  ~Globals();

  void TraceAtStart();
  void StartUtilSampler();
  boost::process::environment PatchedEnviron();
  bool env_is_yes(const std::string& var) const;
  bool env_is_no(const std::string& var) const;
  void CheckAvailGpus() const;
  void DeleteOldTraceFiles() const;

  std::string IMLConfigPath() const;
  void DumpIMLConfig() const;

#ifdef WITH_CUDA_LD_PRELOAD
//  std::shared_ptr<CudaLibrary> cuda_library;
  CudaLibrary* cuda_library;
#endif
  std::unique_ptr<rlscope::DeviceTracer> device_tracer;
  boost::process::child _util_sampler;
  std::string _directory;
  std::string _process_name;
  std::string _machine_name;
  std::string _phase_name;

};

extern Globals globals;

}

#endif //IML_GLOBALS_H
