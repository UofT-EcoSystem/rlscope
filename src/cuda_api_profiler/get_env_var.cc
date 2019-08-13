//
// Created by jagle on 8/12/2019.
//

#include <string>
#include <cstring>
#include <sstream>

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

}
