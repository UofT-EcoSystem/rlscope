/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/load_library.h"

#include <dlfcn.h>

#include "tensorflow/core/lib/core/errors.h"

#include "tensorflow/core/platform/env.h"

namespace rlscope {

namespace internal {

#define DEBUG_LOAD_LIBRARY

bool is_yes(const char* env_var, bool default_value);
bool is_no(const char* env_var, bool default_value);

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

Status LoadLibrary(const char* library_filename, void** handle) {

#ifdef DEBUG_LOAD_LIBRARY
  uint64 start_us, end_us;
  bool TF_DEBUG_LOAD_LIBRARY = is_yes("TF_DEBUG_LOAD_LIBRARY", false);
  if (TF_DEBUG_LOAD_LIBRARY) {
    start_us = Env::Default()->NowMicros();
  }
#endif // DEBUG_LOAD_LIBRARY

  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);

#ifdef DEBUG_LOAD_LIBRARY
  if (TF_DEBUG_LOAD_LIBRARY) {
    end_us = Env::Default()->NowMicros();
#define USEC_IN_MS (1000)
#define USEC_IN_SEC (USEC_IN_MS*1000)
    auto total_sec = (end_us - start_us)/float(USEC_IN_SEC);
    VLOG(0) << "> LoadLibrary library=" << library_filename << " took " << total_sec << " sec";
  }
#endif // DEBUG_LOAD_LIBRARY

  if (!*handle) {
    return errors::NotFound(dlerror());
  }
  return Status::OK();
}

Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol) {
  *symbol = dlsym(handle, symbol_name);
  if (!*symbol) {
    return errors::NotFound(dlerror());
  }
  return Status::OK();
}

string FormatLibraryFileName(const string& name, const string& version) {
  string filename;
#if defined(__APPLE__)
  if (version.size() == 0) {
    filename = "lib" + name + ".dylib";
  } else {
    filename = "lib" + name + "." + version + ".dylib";
  }
#else
  if (version.empty()) {
    filename = "lib" + name + ".so";
  } else {
    filename = "lib" + name + ".so" + "." + version;
  }
#endif
  return filename;
}

}  // namespace internal

}  // namespace rlscope
