//
// Created by jgleeson on 2020-01-23.
//

#ifndef IML_JSON_H
#define IML_JSON_H

#include "common/my_status.h"

#include <string>
#include <nlohmann/json.hpp>

namespace tensorflow {

MyStatus ReadJson(std::string path, nlohmann::json *j);
MyStatus WriteJson(std::string path, const nlohmann::json &j);

} // namespace tensorflow

#endif //IML_JSON_H
