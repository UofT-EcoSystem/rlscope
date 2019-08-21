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

#include "get_env_var.h"

namespace tensorflow {

int ParseFloat(const char* str, size_t size) {
  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string integer_str(str, size);
  std::istringstream ss(integer_str);
  float val = 0;
  ss >> val;
  return val;
}

float get_IML_SAMPLE_EVERY_SEC(float user_value) {
  return ParseEnvFloatOrDefault("IML_SAMPLE_EVERY_SEC", user_value, IML_SAMPLE_EVERY_SEC_DEFAULT);
}

float get_TF_CUDA_API_PRINT_EVERY_SEC(float user_value) {
  return ParseEnvFloatOrDefault("TF_CUDA_API_PRINT_EVERY_SEC", user_value, TF_CUDA_API_PRINT_EVERY_SEC_DEFAULT);
}

float ParseEnvFloatOrDefault(const char* env_name, float user_value, float dflt) {
  if (user_value != 0) {
    return user_value;
  }
  const char* env_val = getenv(env_name);
  if (env_val == nullptr) {
    return dflt;
  }
  return ParseFloat(env_val, strlen(env_val));
}

bool env_is_on(const char* var, bool dflt, bool debug) {
  const char* val = getenv(var);
  if (val == nullptr) {
//    VLOG(0) << "Return dflt = " << dflt << " for " << var;
    return dflt;
  }
  std::string val_str(val);
  std::transform(
      val_str.begin(), val_str.end(), val_str.begin(),
      [](unsigned char c){ return std::tolower(c); });
  bool ret =
      val_str == "on"
      || val_str == "1"
      || val_str == "true"
      || val_str == "yes";
//  VLOG(0) << "val_str = \"" << val_str << "\", " << " ret = " << ret << " for " << var;
  return ret;
}

}
