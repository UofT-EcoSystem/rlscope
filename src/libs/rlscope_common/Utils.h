//
// Created by jgleeson on 2020-06-11.
//

#ifndef IML_UTILS_H
#define IML_UTILS_H

#include <string>
#include <sstream>

namespace rlscope {

std::string DumpDirectory(
    const std::string& directory,
    const std::string& phase_name,
    const std::string& process_name);

void AddTraceIDSuffix(std::stringstream& ss, int trace_id);

} // namespace rlscope

#endif //IML_UTILS_H
