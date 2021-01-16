//
// Created by jgleeson on 2020-01-23.
//

#ifndef RLSCOPE_JSON_H
#define RLSCOPE_JSON_H

#include "my_status.h"

#include <string>
#include <nlohmann/json.hpp>

namespace rlscope {

MyStatus ReadJson(std::string path, nlohmann::json *j);
MyStatus WriteJson(std::string path, const nlohmann::json &j);

} // namespace rlscope

#endif //RLSCOPE_JSON_H
