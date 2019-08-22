//
// Created by jagle on 8/21/2019.
//

#include "tensorflow/core/platform/logging.h"

#include "cuda_api_profiler/util.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <libgen.h>

#include <memory>
#include <string>
#include <cstring>
#include <cstdio>

namespace tensorflow {

void mkdir_p(const std::string& dir, bool exist_ok) {
  return mkdir_p_with_mode(dir, exist_ok);
}

static void _mkdir(const char* dir, bool exist_ok, mode_t mode) {
  int ret = 0;
  if (mkdir(dir, mode)) {
    ret = errno;
    if (ret == EEXIST && exist_ok) {
      // pass
    } else {
      LOG(FATAL) << "Failed to create directory: " << dir << " ; " << strerror(ret);
    }
  }
}

void mkdir_p_with_mode(const std::string& dir, bool exist_ok, mode_t mode) {
  int ret = 0;
  size_t len = dir.size();
  DCHECK(len > 0);
  std::unique_ptr<char[]> tmp(new char[len + 1]);
  char *p = nullptr;

  ret = snprintf(tmp.get(), len, "%s", dir.c_str());
  DCHECK(ret > 0);
  if (tmp[len - 1] == '/') {
    tmp[len - 1] = 0;
  }
  for(p = tmp.get() + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      _mkdir(tmp.get(), exist_ok, mode);
      *p = '/';
    }
  }
  _mkdir(tmp.get(), exist_ok, mode);
}

std::string os_dirname(const std::string& path) {
  std::string path_copy(path);
  std::unique_ptr<char> c_path (new char [path.length()+1]);
  std::strcpy(c_path.get(), path.c_str());
  auto dir = ::dirname(c_path.get());
  return std::string(dir);
}

std::string os_basename(const std::string& path) {
  std::string path_copy(path);
  std::unique_ptr<char> c_path (new char [path.length()+1]);
  std::strcpy(c_path.get(), path.c_str());
  auto dir = ::basename(c_path.get());
  return std::string(dir);
}

}

