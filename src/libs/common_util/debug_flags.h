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

#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

namespace rlscope {

// If defined, print every CUDA/CUPTI/NVPW API call to stderr.
#define SHOULD_PRINT_CUDA_API_CALLS

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
constexpr bool FEATURE_GPU_HW_TRACE = 1;
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

static inline void DumpStacktrace(size_t skip_frames, bool snippet) {
  backward::StackTrace st;
  const size_t MAX_STACKFRAMES = 32;
  // Skip stackframes added by this callframe.
  skip_frames += 3;
  st.load_here(MAX_STACKFRAMES);
// Last 4 frames are always related to backward.hpp or logging.cc.
// Skip those frames; make the latest frame the LOG(FAIL) or DCHECK failure.
  size_t idx;
  if (st.size() < skip_frames) {
// Print the whole thing.
    idx = 0;
  } else {
// Skip the last 4 frames.
    idx = skip_frames;
  }
  st.load_from(st[idx].addr, MAX_STACKFRAMES);
  backward::Printer p;
  p.snippet = snippet;
  p.print(st);
}

static void __attribute_noinline__ dbg_breakpoint(const std::string& name, const char* file, int lineno) {
  std::cout << "";
}
static void __attribute_noinline__ _dbg_breakpoint(const std::string& name, const char* file, int lineno) {
  if (FEATURE_BREAKPOINT_DUMP_STACK) {
    DumpStacktrace(1, true);
  }
  std::cout << "[ HIT BREAKPOINT \"" << name << "\" @ " << file << ":" << lineno << " ]" << std::endl;
  dbg_breakpoint(name, file, lineno);
}
#define DBG_BREAKPOINT(name) rlscope::_dbg_breakpoint(name, __FILE__, __LINE__)

} // namespace rlscope

#endif //IML_DEBUG_FLAGS_H
