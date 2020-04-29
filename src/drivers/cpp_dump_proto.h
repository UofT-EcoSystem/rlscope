//
// Created by jgleeson on 2020-04-13.
//

#ifndef IML_GPU_UTIL_EXPERIMENT_H
#define IML_GPU_UTIL_EXPERIMENT_H

#include <boost/optional.hpp>
#include <boost/process.hpp>

#include "cuda_api_profiler/generic_logging.h"

namespace tensorflow {

struct RLSAnalyzeArgs {
//   Environment variables
//  boost::optional<std::string> IML_PROCESS_NAME;

  // Flags
  boost::optional<bool> FLAGS_debug;
  boost::optional<bool> FLAGS_ignore_memcpy;
  boost::optional<bool> FLAGS_cross_process;
  boost::optional<std::string> FLAGS_proto;
  boost::optional<std::string> FLAGS_iml_directory;
  boost::optional<std::string> FLAGS_mode;

  boost::optional<std::string> FLAGS_cupti_overhead_json;
  boost::optional<std::string> FLAGS_LD_PRELOAD_overhead_json;
  boost::optional<std::string> FLAGS_python_annotation_json;
  boost::optional<std::string> FLAGS_python_clib_interception_tensorflow_json;
  boost::optional<std::string> FLAGS_python_clib_interception_simulator_json;
  boost::optional<std::string> FLAGS_nvprof_process_regex;

  boost::optional<int64_t> FLAGS_polling_interval_us;

  static RLSAnalyzeArgs FromFlags();


  template <typename OStream, class FlagVar>
  void _PrintFlag(OStream& out, int indent, const std::string& flag_name, FlagVar FLAGS_var) const {
    out << "\n";
    PrintIndent(out, indent + 1);
    out << flag_name << " = ";
    if (!FLAGS_var.has_value()) {
      out << "None";
    } else {
      out << FLAGS_var.get();
    }
  }


  template <typename OStream>
  void Print(OStream& out, int indent) const {
    PrintIndent(out, indent);
    out << "RLSAnalyzeArgs:";

#define PRINT_FLAG(FLAGS_var) \
    _PrintFlag(out, indent, #FLAGS_var, FLAGS_var);

    PRINT_FLAG(FLAGS_debug);
    PRINT_FLAG(FLAGS_ignore_memcpy);
    PRINT_FLAG(FLAGS_cross_process);
    PRINT_FLAG(FLAGS_proto);
    PRINT_FLAG(FLAGS_iml_directory);
    PRINT_FLAG(FLAGS_mode);
    PRINT_FLAG(FLAGS_cupti_overhead_json);
    PRINT_FLAG(FLAGS_LD_PRELOAD_overhead_json);
    PRINT_FLAG(FLAGS_python_annotation_json);
    PRINT_FLAG(FLAGS_python_clib_interception_tensorflow_json);
    PRINT_FLAG(FLAGS_python_clib_interception_simulator_json);
    PRINT_FLAG(FLAGS_nvprof_process_regex);
    PRINT_FLAG(FLAGS_polling_interval_us);
#undef PRINT_FLAG

//    PrintValue();
  }

  template <typename OStream>
  friend OStream &operator<<(OStream &os, const RLSAnalyzeArgs &obj)
  {
    obj.Print(os, 0);
    return os;
  }

};

}

#endif //IML_GPU_UTIL_EXPERIMENT_H
