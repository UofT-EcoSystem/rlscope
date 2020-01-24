//
// Created by jgleeson on 2020-01-23.
//

#include "analysis/my_status.h"

#include <string>
#include <sstream>
#include <iomanip>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <boost/filesystem.hpp>

#include "error_codes.pb.h"

namespace tensorflow {

MyStatus ReadJson(std::string path, json* j) {
  boost::filesystem::path file(path);

  if(!boost::filesystem::exists(file)) {
    std::stringstream ss;
    ss << "Couldn't find json file @ path=" << path;
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  // read a JSON file
  std::ifstream inp(path);
  try {
    inp >> *j;
  } catch (nlohmann::detail::parse_error e) {
    std::stringstream ss;
    ss << "Failed to parse json file @ path=" << path << ":\n";
    ss << e.what();
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }

  return MyStatus::OK();
}

MyStatus WriteJson(std::string path, const nlohmann::json& j) {
  boost::filesystem::path file(path);
  auto parent = file.parent_path();
  if(!boost::filesystem::exists(parent)) {
    std::stringstream ss;
    ss << "Couldn't find directory of json file @ directory=" << parent;
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  // read a JSON file
  std::ofstream out(path);
  try {
    // j.dump()
    out << std::setw(4) << j;
  } catch (nlohmann::detail::parse_error e) {
    std::stringstream ss;
    ss << "Failed to write json file @ path=" << path << ":\n";
    ss << e.what();
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }

  return MyStatus::OK();
}

} // namespace tensorflow
