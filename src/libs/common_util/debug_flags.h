//
// Created by jagle on 11/26/2019.
//

#ifndef IML_DEBUG_FLAGS_H
#define IML_DEBUG_FLAGS_H

#include <bitset>
#include <set>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

#include <backward.hpp>

#include <spdlog/spdlog.h>

//#include <sys/types.h>
// Must be included in order operator<< to work with spd logging.
// https://github.com/gabime/spdlog#user-defined-types
#include "spdlog/fmt/ostr.h"

//#include "env_var.h"
#include "my_status.h"

#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

namespace rlscope {

// If defined, print every CUDA/CUPTI/NVPW API call to stderr.
//#define SHOULD_PRINT_CUDA_API_CALLS

#define DBG_LOG(fmt, ...) SPDLOG_DEBUG("pid={} @ {}: " fmt, gettid(), __func__, __VA_ARGS__)
#define DBG_WARN(fmt, ...) SPDLOG_WARN("pid={} @ {}: " fmt, gettid(), __func__, __VA_ARGS__)

#define RLS_INFO(flag_name, fmt, ...) SPDLOG_INFO("[{}] pid={} @ {}: " fmt, flag_name, gettid(), __func__, __VA_ARGS__)
#define RLS_LOG(flag_name, fmt, ...) SPDLOG_DEBUG("[{}] pid={} @ {}: " fmt, flag_name, gettid(), __func__, __VA_ARGS__)
#define RLS_WARN(flag_name, fmt, ...) SPDLOG_WARN("[{}] pid={} @ {}: " fmt, flag_name, gettid(), __func__, __VA_ARGS__)

// NOTE: this is the only variation of bit-flags I saw the compiler successfully "optimize out" of my program.
// Attempting to use constexpr in combination with std::bitset or even just plain uint64_t FAILS.
// To test things, I did something like this:
// if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//   std::cout << "BANANAS" << std::endl;
// }
// $ strings a.out | grep BANANAS
// <OUPTPUT>
//
constexpr bool FEATURE_OVERLAP = 0;
constexpr bool FEATURE_OVERLAP_META = 0;
constexpr bool FEATURE_LOAD_DATA = 1;
constexpr bool FEATURE_SAVE_JS = 1;
constexpr bool FEATURE_PREPROCESS_DATA = 0;
constexpr bool FEATURE_OVERHEAD_CORRECTION = 0;
constexpr bool FEATURE_GPU_CLOCK_FREQ = 1;
constexpr bool FEATURE_GPU_UTIL_CUDA_CONTEXT = 0;
constexpr bool FEATURE_GPU_UTIL_SYNC = 0;
constexpr bool FEATURE_GPU_UTIL_KERNEL_TIME = 0;
constexpr bool FEATURE_GPU_HW = 1;
constexpr bool FEATURE_RLSCOPE_LIB_TRACE = 0;
constexpr bool FEATURE_GPU_HW_TRACE = 0;
constexpr bool FEATURE_ANY =
    FEATURE_OVERLAP
    || FEATURE_OVERLAP_META
    || FEATURE_LOAD_DATA
    || FEATURE_SAVE_JS
    || FEATURE_PREPROCESS_DATA
    || FEATURE_OVERHEAD_CORRECTION
    || FEATURE_GPU_CLOCK_FREQ
    || FEATURE_GPU_UTIL_CUDA_CONTEXT
    || FEATURE_GPU_UTIL_SYNC
    || FEATURE_GPU_UTIL_KERNEL_TIME
    || FEATURE_GPU_HW
    || FEATURE_GPU_HW_TRACE;
#define SHOULD_DEBUG(feature) ((SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG) && feature)

constexpr bool FEATURE_BREAKPOINT_DUMP_STACK = 0;

// Just do overlap computation on CATEGORY_OPERATION's.
constexpr bool FEATURE_JUST_OPERATIONS = 0;

#define RAISE_NOT_IMPLEMENTED(msg) \
  throw NotImplementedException(msg, __FILE__, __LINE__)

class NotImplementedException : public std::logic_error {
public:
  std::string message;
  NotImplementedException(const std::string& msg, const char* filename, int lineno) :
      std::logic_error("") {
    std::stringstream ss;
    ss << "Function not yet implemented @ " << filename << ":" << lineno << " :: " << msg;
    message = ss.str();
  }

  virtual char const * what() const noexcept {
    return message.c_str();
  }
};

void dbg_breakpoint(const std::string& name, const char* file, int lineno);
void _dbg_breakpoint(const std::string& name, const char* file, int lineno);
void _dbg_breakpoint_with_stacktrace(const std::string& name, const char* file, int lineno);

#define DBG_BREAKPOINT(name) rlscope::_dbg_breakpoint(name, __FILE__, __LINE__)
#define DBG_BREAKPOINT_STACKTRACE(name) rlscope::_dbg_breakpoint_with_stacktrace(name, __FILE__, __LINE__)
#define PRINT_AND_DBG_BREAKPOINT(name, status) \
    std::cerr << "ERROR: " << status << std::endl; \
    DBG_BREAKPOINT(name);

} // namespace rlscope

#endif //IML_DEBUG_FLAGS_H
