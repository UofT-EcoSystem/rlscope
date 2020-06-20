//
// Created by jagle on 8/12/2019.
//

#include <string>
#include <cstring>
#include <sstream>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <memory>

#include <boost/optional.hpp>

#include "common_util.h"
#include "range_sampling.h"

#include "get_env_var.h"

namespace rlscope {

float get_IML_SAMPLE_EVERY_SEC(boost::optional<float> user_value) {
  return ParseEnvOrDefault("float", "IML_SAMPLE_EVERY_SEC", user_value, IML_SAMPLE_EVERY_SEC_DEFAULT);
}

int get_IML_GPU_HW_CONFIG_PASSES(boost::optional<int> user_value) {
  return ParseEnvOrDefault("integer", "IML_GPU_HW_CONFIG_PASSES", user_value, IML_GPU_HW_CONFIG_PASSES_DEFAULT);
}

std::vector<std::string> get_IML_GPU_HW_METRICS(boost::optional<std::string> user_value) {
  std::string dflt = rlscope::DEFAULT_METRICS_STR;
  std::string env_name = "IML_GPU_HW_METRICS";
  std::string value;

  MyStatus my_status = MyStatus::OK();
  if (user_value.has_value()) {
    value = user_value.get();
  } else {
    const char* env_val = getenv(env_name.c_str());
    if (env_val == nullptr) {
      value = dflt;
    } else {
      value = env_val;
    }
  }
  return StringSplit(value, ",");
}

float get_TF_CUDA_API_PRINT_EVERY_SEC(boost::optional<float> user_value) {
  return ParseEnvOrDefault("float", "TF_CUDA_API_PRINT_EVERY_SEC", user_value, TF_CUDA_API_PRINT_EVERY_SEC_DEFAULT);
}

}
