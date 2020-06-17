//
// Created by jgleeson on 2020-06-15.
//

#include <cstdlib>
#include <string>
#include <algorithm>
#include <cstring>

#include "env_var.h"

namespace rlscope {

bool env_is_on(const char* var, bool dflt, bool debug) {
  const char* val = getenv(var);
  if (val == nullptr) {
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

} // namespace rlscope
