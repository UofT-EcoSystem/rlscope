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

#include "get_env_var.h"

namespace rlscope {

template <typename T>
MyStatus ParseValue(const char* type_name, const char* str, size_t size, T* value) {
  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string integer_str(str, size);
  std::istringstream ss(integer_str);
  ss >> *value;
  if (ss.fail()) {
    std::stringstream err_ss;
    err_ss << "Failed to parse " << type_name << " from \"" << str << "\"";
    return MyStatus(error::INVALID_ARGUMENT, err_ss.str());
  }
  return MyStatus::OK();
}

template <typename T>
T ParseEnvOrDefault(const char* type_name, const char* env_name, boost::optional<T> user_value, float dflt) {
  MyStatus my_status = MyStatus::OK();
  if (user_value.has_value()) {
    return user_value.get();
  }
  const char* env_val = getenv(env_name);
  if (env_val == nullptr) {
    return dflt;
  }
  T value;
  my_status = ParseValue(type_name, env_val, strlen(env_val), &value);
  if (!my_status.ok()) {
    LOG(FATAL) << "Failed to parse env variable " << env_name << ": " << my_status.error_message();
  }
  return value;
}

float get_IML_SAMPLE_EVERY_SEC(boost::optional<float> user_value) {
  return ParseEnvOrDefault("float", "IML_SAMPLE_EVERY_SEC", user_value, IML_SAMPLE_EVERY_SEC_DEFAULT);
}

int get_IML_GPU_HW_CONFIG_PASSES(boost::optional<int> user_value) {
  return ParseEnvOrDefault("integer", "IML_GPU_HW_CONFIG_PASSES", user_value, IML_GPU_HW_CONFIG_PASSES_DEFAULT);
}

std::vector<std::string> get_IML_GPU_HW_METRICS(boost::optional<std::string> user_value) {
  // return ParseEnvOrDefault("integer", "IML_GPU_HW_METRICS", user_value, IML_GPU_HW_METRICS_DEFAULT);
  std::string dflt = IML_GPU_HW_METRICS_DEFAULT;
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

bool is_yes(const char* env_var, bool default_value) {
  if (getenv(env_var) == nullptr) {
    return default_value;
  }
  return strcmp("yes", getenv(env_var)) == 0;
}
bool is_no(const char* env_var, bool default_value) {
  if (getenv(env_var) == nullptr) {
    return default_value;
  }
  return strcmp("no", getenv(env_var)) == 0;
}

}
