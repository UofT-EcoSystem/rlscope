//
// Created by jagle on 11/26/2019.
//

#ifndef IML_DEBUG_FLAGS_H
#define IML_DEBUG_FLAGS_H

#include <bitset>
#include <set>

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
#define SHOULD_DEBUG(feature) ((SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG) && feature)

// This allows making a block of code pertain to multiple features, but that's not very useful anyways, and this
// has a limit of 64 features (maximum shift width).
// If debug statements are optimized out, we expect NOT to find "BANANAS" in <OUPTPUT>
//#define MAX_NUM_DEBUG_FEATURES 64
//#define MK_DEBUG_FEATURE(i) (1 << i)
//#define FEATURE_OVERLAP MK_DEBUG_FEATURE(0)
//#define FEATURE_LOAD_DATA MK_DEBUG_FEATURE(1)
//#define FEATURE_TOO_BIG MK_DEBUG_FEATURE(65)
//#define DEBUG_FEATURES ( FEATURE_OVERLAP | FEATURE_TOO_BIG )
//#define SHOULD_DEBUG(features) ( DEBUG_FEATURES & features )

//#define MAX_NUM_DEBUG_FEATURES 64
//using DebugBitset = std::bitset<MAX_NUM_DEBUG_FEATURES>;
//constexpr DebugBitset MK_DEBUG_FEATURE(size_t i) {
//  // static_assert(i < MAX_NUM_DEBUG_FEATURES, "Number of debug features cannot exceed MAX_NUM_DEBUG_FEATURES (default 64)");
//  return DebugBitset(1 << i);
//}
//constexpr DebugBitset FEATURE_OVERLAP = MK_DEBUG_FEATURE(0);
//constexpr DebugBitset FEATURE_LOAD_DATA = MK_DEBUG_FEATURE(1);
//constexpr DebugBitset DEBUG_FEATURES = (
//  FEATURE_OVERLAP
//);
//constexpr bool SHOULD_DEBUG(DebugBitset& features) {
//  // FAILS: "call to non-constexpr function any()"
//  return (DEBUG_FEATURES & features).any();
//}

//#define MAX_NUM_FEATURES 64
//#define MK_DEBUG_FEATURE(i) std::bitset<MAX_NUM_FEATURES>(1 << i)
//#define FEATURE_OVERLAP MK_DEBUG_FEATURE(0)
//#define FEATURE_LOAD_DATA MK_DEBUG_FEATURE(1)
//#define DEBUG_FEATURES (
//  FEATURE_OVERLAP
//)
//#define SHOULD_DEBUG(DEBUG_FEATURE) ((DEBUG_FEATURES & DEBUG_FEATURE).any())

//#define MAX_NUM_DEBUG_FEATURES 64
//using DebugBitset = uint64_t;
//constexpr DebugBitset MK_DEBUG_FEATURE(size_t i) {
//  // static_assert(i < MAX_NUM_DEBUG_FEATURES, "Number of debug features cannot exceed MAX_NUM_DEBUG_FEATURES (default 64)");
//  return (1 << i);
//}
//constexpr DebugBitset FEATURE_OVERLAP = MK_DEBUG_FEATURE(0);
//constexpr DebugBitset FEATURE_LOAD_DATA = MK_DEBUG_FEATURE(1);
//constexpr DebugBitset DEBUG_FEATURES = (
//  FEATURE_OVERLAP
//);
//constexpr bool SHOULD_DEBUG(DebugBitset features) {
//  // FAILS: "call to non-constexpr function any()"
//  return DEBUG_FEATURES & features;
//}


//#define MAX_NUM_DEBUG_FEATURES 64
//using DebugBitset = std::set<int>;
//constexpr int MK_DEBUG_FEATURE(size_t i) {
//  // static_assert(i < MAX_NUM_DEBUG_FEATURES, "Number of debug features cannot exceed MAX_NUM_DEBUG_FEATURES (default 64)");
//  return i;
//}
//constexpr int FEATURE_OVERLAP = MK_DEBUG_FEATURE(0);
//constexpr int FEATURE_LOAD_DATA = MK_DEBUG_FEATURE(1);
//constexpr DebugBitset DEBUG_FEATURES{
//  FEATURE_OVERLAP
////  , FEATURE_LOAD_DATA
//};
//constexpr bool SHOULD_DEBUG(int feature) {
//  // FAILS: "call to non-constexpr function any()"
//  return DEBUG_FEATURES.find(feature) != DEBUG_FEATURES.end();
//}

};

#endif //IML_DEBUG_FLAGS_H
