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

#include <spdlog/spdlog.h>
//#include <sys/types.h>

#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

namespace tensorflow {

#define DBG_LOG(fmt, ...) SPDLOG_DEBUG("pid={} @ {}: " fmt, gettid(), __func__, __VA_ARGS__)

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
constexpr bool FEATURE_LOAD_DATA = 0;
constexpr bool FEATURE_SAVE_JS = 1;
#define SHOULD_DEBUG(feature) ((SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG) && feature)

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

};

#endif //IML_DEBUG_FLAGS_H
