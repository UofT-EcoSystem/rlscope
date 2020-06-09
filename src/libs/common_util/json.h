//
// Created by jgleeson on 2020-01-23.
//

#ifndef IML_JSON_H
#define IML_JSON_H

#include "my_status.h"

#include <string>
#include <nlohmann/json.hpp>

namespace CuptiSamples {

MyStatus ReadJson(std::string path, nlohmann::json *j);
MyStatus WriteJson(std::string path, const nlohmann::json &j);

} // namespace CuptiSamples

#endif //IML_JSON_H
