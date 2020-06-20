//
// Created by jgleeson on 2020-06-15.
//

#pragma once

#include <boost/optional.hpp>

#include "my_status.h"

namespace rlscope {

bool env_is_on(const char* var, bool dflt, bool debug);

bool is_yes(const char* env_var, bool default_value);
bool is_no(const char* env_var, bool default_value);

template <typename T>
MyStatus ParseEnvValue(const char* type_name, const char* str, size_t size, T* value) {
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
struct ParseEnvOrDefaultImpl {
    static T ParseEnvOrDefault(const char* type_name, const char* env_name, boost::optional<T> user_value, float dflt) {
      MyStatus my_status = MyStatus::OK();
      if (user_value.has_value()) {
        return user_value.get();
      }
      const char* env_val = getenv(env_name);
      if (env_val == nullptr) {
        return dflt;
      }
      T value;
      my_status = ParseEnvValue(type_name, env_val, strlen(env_val), &value);
      if (!my_status.ok()) {
        // LOG(FATAL) << "Failed to parse env variable " << env_name << ": " << my_status.error_message();
        std::stringstream ss;
        ss << "Failed to parse env variable " << env_name;
        my_status.PrependMsg(ss.str());
        IF_BAD_STATUS_EXIT_WITH(my_status);
      }
      return value;
    }
};
template <>
struct ParseEnvOrDefaultImpl<bool> {
  static bool ParseEnvOrDefault(const char* type_name, const char* env_name, boost::optional<bool> user_value, float dflt) {
    MyStatus my_status = MyStatus::OK();
    if (user_value.has_value()) {
      return user_value.get();
    }
    const char* env_val = getenv(env_name);
    if (env_val == nullptr) {
      return dflt;
    }
    std::string val_str(env_val);
    std::transform(
        val_str.begin(), val_str.end(), val_str.begin(),
        [](unsigned char c){ return std::tolower(c); });
    bool value =
        val_str == "on"
        || val_str == "1"
        || val_str == "true"
        || val_str == "yes";
    return value;
  }
};
// Specializing function templates requires an extra level of indirection.
// http://www.gotw.ca/publications/mill17.htm
template <typename T>
T ParseEnvOrDefault(const char* type_name, const char* env_name, boost::optional<T> user_value, float dflt) {
  return ParseEnvOrDefaultImpl<T>::ParseEnvOrDefault(type_name, env_name, user_value, dflt);
}

//template <typename T>
//T ParseEnvOrDefault(const char* type_name, const char* env_name, boost::optional<T> user_value, float dflt) {
//  MyStatus my_status = MyStatus::OK();
//  if (user_value.has_value()) {
//    return user_value.get();
//  }
//  const char* env_val = getenv(env_name);
//  if (env_val == nullptr) {
//    return dflt;
//  }
//  T value;
//  my_status = ParseEnvValue(type_name, env_val, strlen(env_val), &value);
//  if (!my_status.ok()) {
//    // LOG(FATAL) << "Failed to parse env variable " << env_name << ": " << my_status.error_message();
//    std::stringstream ss;
//    ss << "Failed to parse env variable " << env_name;
//    my_status.PrependMsg(ss.str());
//    IF_BAD_STATUS_EXIT_WITH(my_status);
//  }
//  return value;
//}

//template <>
//bool ParseEnvOrDefault(const char* type_name, const char* env_name, boost::optional<bool> user_value, float dflt) {
//  MyStatus my_status = MyStatus::OK();
//  if (user_value.has_value()) {
//    return user_value.get();
//  }
//  const char* env_val = getenv(env_name);
//  if (env_val == nullptr) {
//    return dflt;
//  }
//  std::string val_str(env_val);
//  std::transform(
//      val_str.begin(), val_str.end(), val_str.begin(),
//      [](unsigned char c){ return std::tolower(c); });
//  bool value =
//      val_str == "on"
//      || val_str == "1"
//      || val_str == "true"
//      || val_str == "yes";
//  return value;
//}

} // namespace rlscope
