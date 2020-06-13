//
// Created by jgleeson on 2020-06-11.
//

#include <string>
#include <sstream>
#include <cassert>

#include "common_util.h"

namespace rlscope {


std::string DumpDirectory(
    const std::string& directory,
    const std::string& phase_name,
    const std::string& process_name) {
//  DCHECK(directory != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
//  DCHECK(phase_name != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";
//  DCHECK(process_name != "") << "You forgot to call CUDAAPIProfiler.SetMetadata";

  assert(directory != "");
  assert(phase_name != "");
  assert(process_name != "");

  std::stringstream ss;

  ss << directory;
  ss << path_separator() << "process" << path_separator() << process_name;
  ss << path_separator() << "phase" << path_separator() << phase_name;
  return ss.str();

}

void AddTraceIDSuffix(std::stringstream& ss, int trace_id) {
  ss << ".trace_" << trace_id;
}


} // namespace rlscope
