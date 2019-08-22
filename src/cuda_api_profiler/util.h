//
// Created by jagle on 8/21/2019.
//

#ifndef IML_UTIL_H
#define IML_UTIL_H

#include <sys/stat.h>
#include <sys/types.h>

#include <string>
#include <cstring>

namespace tensorflow {

void mkdir_p(const std::string& dir,
    bool exist_ok = true);
void mkdir_p_with_mode(const std::string& dir,
    bool exist_ok = true,
    // Read/write/search permissions for owner and group,
    // and with read/search permissions for others.
    mode_t mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

std::string os_dirname(const std::string& path);
std::string os_basename(const std::string& path);

}

#endif //IML_UTIL_H
