//
// Created by jagle on 11/20/2019.
//

#include <limits.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iostream>
#include <bitset>

#include "analysis/trace_file_parser.h"

using namespace Eigen;

namespace rlscope {

template <typename T>
void PrintGot(const T& got) {
  std::cout << "GOT:\n";
  got.Print(std::cout, 1);
  std::cout << "\n";
}

template <typename T>
void PrintExpect(const T& expect) {
  std::cout << "EXPECT:\n";
  expect.Print(std::cout, 1);
  std::cout << "\n";
}

template <typename T>
void PrintIfFail(bool success, const T& got, const T& expect) {
  if (success) {
    return;
  }
  std::cout << "Failed test-case;\n";
  std::cout << "GOT:\n";
  got.Print(std::cout, 1);
  std::cout << "\n";
  std::cout << "EXPECT:\n";
  expect.Print(std::cout, 1);
  std::cout << "\n";
}

struct TestEvent {
  TimeUsec start_us;
  TimeUsec end_us;
  TestEvent(TimeUsec start_us_, TimeUsec end_us_) :
      start_us(start_us_)
      , end_us(end_us_)
  {
  }
};
using TestEventData = std::map<std::string, std::vector<TestEvent>>;
using OverlapData = std::map<std::set<std::string>, TimeUsec>;
using RegionData = std::map<std::set<std::string>, TestEvent>;
//using T = TestEvent;

TestEvent T(int64_t start_sec, int64_t end_sec) {
  auto start_us = start_sec*USEC_IN_SEC;
  auto end_us = end_sec*USEC_IN_SEC;
  return TestEvent(start_us, end_us);
}

TimeUsec sec(int64_t seconds) {
  return seconds*USEC_IN_SEC;
}

static const std::string process = "process";

CategoryKey AsCategoryKey(const std::string& key) {
  CategoryKey category_key;
  category_key.ops.insert(key);
  category_key.procs.insert(process);
  return category_key;
}

CategoryTimesCount CountCategoryTimes(const TestEventData& event_data) {
  CategoryTimesCount count;
  for (const auto& pair : event_data) {
    const auto& category_key = AsCategoryKey(pair.first);
    auto n_events = pair.second.size();
    count.Add(category_key, n_events);
  }
  return count;
}

CategoryTimes CategoryTimesFrom(const TestEventData& event_data) {
  auto count = CountCategoryTimes(event_data);
  CategoryTimes category_times(process, count);

  for (const auto& pair : event_data) {
    const auto category_key = AsCategoryKey(pair.first);
    EOEvents& eo_events = category_times.eo_times.at(category_key);
    for (const auto& event : pair.second) {
      auto start_us = event.start_us;
      auto end_us = event.end_us;
      eo_events.AppendEvent(boost::none, start_us, end_us);
    }
  }

  return category_times;
}


OverlapResult OverlapFrom(const OverlapComputer& overlap_computer, const OverlapData& overlap_data,
    const RegionData& region_data = {},
    bool debug=false) {
  OverlapResult r(overlap_computer.category_times);
  r.idx_map = overlap_computer.ctimes.idx_map;

  auto as_bitset = [debug, &r] (const std::set<std::string>& ops) -> CategoryKeyBitset {
    CategoryKeyBitset bitset = CategoryKeyBitset::EmptySet(r.idx_map);
    assert(bitset.idx_map != nullptr);
    bitset.debug = debug;
    std::set<size_t> indices;
    std::vector<CategoryKey> category_keys;
    for (auto const& key : ops) {
      const auto category_key = AsCategoryKey(key);
      category_keys.push_back(category_key);
      auto category_idx = r.idx_map->Idx(category_key);
      indices.insert(category_idx);
      bitset.Add(category_idx);
    }
    return bitset;
  };

  for (const auto& pair : overlap_data) {
    auto bitset = as_bitset(pair.first);
    r.overlap[bitset] = pair.second;
  }

  for (const auto& pair : region_data) {
    const auto& bitset = as_bitset(pair.first);
    const auto& event = pair.second;
    const auto& region_meta = RegionMetadata(bitset, event.start_us, event.end_us);
    r.meta.regions[bitset] = region_meta;
  }
  for (const auto& pair : r.meta.regions) {
    const auto& bitset = pair.first;
    assert(bitset.idx_map != nullptr);
  }
  return r;
}

TEST(TextIdxMap, Test_BitsetAdd) {
  using Bitset = std::bitset<MAX_CATEGORY_KEYS>;
  Bitset bitset;
  bitset[1] = 1;
  // SPDLOG_DEBUG("bitset[1] => {}", bitset.to_string());
  EXPECT_EQ(
      bitset.to_string(),
      "0000000000000000000000000000000000000000000000000000000000000010");
}

TEST(TextIdxMap, Test_OverlapData) {
  TestEventData event_data = {
      {"c1", {T(3, 7), T(8, 10)}},
      {"c2", {T(1, 4), T(6, 9)}},
      {"c3", {T(2, 5), T(7, 8), T(11, 12)}},
  };
  OverlapData overlap_data = {

      {{"c1"}, sec(2)},
      {{"c2"}, sec(1)},
      {{"c3"}, sec(1)},

      {{"c1", "c2"}, sec(2)},
      {{"c1", "c3"}, sec(1)},
      {{"c2", "c3"}, sec(2)},

      {{"c1", "c2", "c3"}, sec(1)},

  };
  auto category_times = CategoryTimesFrom(event_data);
  OverlapComputer overlap_computer(category_times);

  auto expect_r = OverlapFrom(overlap_computer, overlap_data);

  EXPECT_EQ(overlap_data.size(), expect_r.overlap.size());
}

TEST(TestIdxMap, Test_01) {
  using IndexMap = IdxMaps<size_t, std::string>;
  using Bitset = std::bitset<3>;
  using KeySet = std::set<std::string>;
  KeySet keys = {"c1", "c2", "c3"};
  auto idx_map = IndexMap::From(keys);
  // idx_map.debug = true;

  Bitset bitset;
  bitset[0] = 1;
  bitset[2] = 1;
  auto got = idx_map.KeySetFrom(bitset);
  KeySet expect = {"c1", "c3"};
  EXPECT_EQ(got, expect);
}

TEST(TestComputeOverlap, Test_01_Complete) {
  TestEventData event_data = {
      {"A", {T(3, 7), T(8, 10)}},
      {"B", {T(1, 4), T(6, 9)}},
      {"C", {T(2, 5), T(7, 8), T(11, 12)}},
  };
  //       1   2   3   4   5   6   7   8   9   10  11  12  13
  // 0: A:         [               ]   [       ]
  // 1: B: [           ]       [           ]
  // 2: C:     [           ]       [   ]           [   ]
  //       |   |   |   |   |   |   |   |   |   |   |   |
  //       B   BC  ABC AC  A   AB  BC  AB  A   0   C   0
  //
  // 7 events.
  //
  // Expect FEATURE_OVERLAP_META:
  //   NOTE: AB = {0, 1}
  //     {1} 1..2
  //     {1, 2} 2..3
  //     {0, 1, 2} 3..4
  //     {0, 2} 4..5
  //     {0} 5..6
  //     {0, 1} 6..7
  //     {1, 2} 7..8
  //     {0, 1} 8..9
  //     {0} 9..10
  //     {2} 11..12
  //
  //     Tally:
  //       {0} 5..6
  //       {0} 9..10
  //       --------------
  //       {0} 5..10
  //
  //       {0, 1} 6..7
  //       {0, 1} 8..9
  //       --------------
  //       {0, 1} 6..9
  //
  //       {0, 1, 2} 3..4
  //       --------------
  //       {0, 1, 2} 3..4
  //
  //       {0, 2} 4..5
  //       --------------
  //       {0, 2} 4..5
  //
  //       {1} 1..2
  //       --------------
  //       {1} 1..2
  //
  //       {1, 2} 2..3
  //       {1, 2} 7..8
  //       --------------
  //       {1, 2} 2..8
  //
  //       {2} 11..12
  //       --------------
  //       {2} 11..12
  //
  // NOTE: our way of counting num_events is totally WRONG...
  // it will double-count events:
  //   A:     [   ]
  //   B: [           ]
  //      |   |   |   |
  //      B   AB  B   0
  //
  //   AB, 2, 3, 1
  //   ------------
  //   AB, 2, 3, 1
  //
  //   B, 1, 2, 1
  //   B, 3, 4, 1
  //   ------------
  //   B, 1, 4, 2
  //
  //   GOT: 3 num_events, ACTUAL = 2
  //
  //   num_events represents the "number of overlap regions" that make up an overlap.
  OverlapData overlap_data = {

      {{"A"}, sec(2)},
      {{"B"}, sec(1)},
      {{"C"}, sec(1)},

      {{"A", "B"}, sec(2)},
      {{"A", "C"}, sec(1)},
      {{"B", "C"}, sec(2)},

      {{"A", "B", "C"}, sec(1)},

  };
  RegionData region_data = {
      {{"A"}, T(5, 10)},
      {{"A", "B"}, T(6, 9)},
      {{"A", "B", "C"}, T(3, 4)},
      {{"A", "C"}, T(4, 5)},
      {{"B"}, T(1, 2)},
      {{"B", "C"}, T(2, 8)},
      {{"C"}, T(11, 12)},
  };
  auto category_times = CategoryTimesFrom(event_data);

//  std::cout << "CategoryTimes:\n";
//  category_times.Print(std::cout, 1);
//  std::cout << "\n";

  OverlapComputer overlap_computer(category_times);
  overlap_computer.debug = true;

  auto expect_r = OverlapFrom(overlap_computer, overlap_data, region_data);
  auto got_r = overlap_computer.ComputeOverlap();
  // TODO: lookup how to use custom printers with gtest for got/expect when test fails.

  bool overlap_eq = (got_r.overlap == expect_r.overlap);
  PrintIfFail(overlap_eq, got_r, expect_r);
  EXPECT_TRUE(overlap_eq);

  bool meta_eq = (got_r.meta == expect_r.meta);
  PrintIfFail(meta_eq, got_r.meta, expect_r.meta);
  EXPECT_TRUE(meta_eq);

}

TEST(TestEachMerged, Test_01_Merge) {
  // ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
  std::vector<std::vector<std::string>> vec_of_xs{
      {     "b", "c",                          "i"},
      {"a", "b",                               "i"},
      {"a", "b",      "d", "e", "f", "g", "h", "i"},
  };


  std::vector<std::string> expect{
    "a",
    "a",
    "b",
    "b",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "i",
    "i",
  };
  std::vector<std::string> merged;
  auto func = [&merged] (const std::vector<std::string>& letters, size_t i) {
    EXPECT_LT(i, letters.size());
    merged.push_back(letters[i]);
  };
  auto key_func = [] (const std::vector<std::string>& letters, size_t i) {
    EXPECT_LT(i, letters.size());
    return letters[i];
  };
  EachMerged<std::vector<std::string>, std::string>(vec_of_xs, func, key_func);
  EXPECT_EQ(merged, expect);
}

#define MAX_BITS 64
// using IthBitType = size_t;
using IthBitType = uint64_t;


//
// WARNING regarding std::bitset and signed/unsigned integers.
//
// The expressions:
// (1) 1 << 31
// (2) 1U << 31
// (3) 1UL << 31
// (4) static_cast<uint64_t>(1) << 31
// These are all fundamentally different things.
//
// In (1), "1" has type int which is a 2's complement integer that has only 31 bits to shift within;
// "1 << 31" will cause the "<<" operator to "overflow" leading to undefined behaviour.
//
// In (2), "1U" is an unsigned int.  So, we have 32 bits to shift within.  So, "1 << 31" is ok,
// BUT "1 << 32" will overflow silently.
//
// In (3), "1UL" is an unsigned long int.  So, we have 64 bits to shift within.  So, "1 << 31" is ok,
// and so is "1 << 32".
//
// In (4), we can guarantee the underlying type of "static_cast<uint64_t>(1)" and unsigned and has 64 bits of width.
// This avoids the unexpected size differences of "unsigned long" on 32bit vs 64bit platforms.
// So, this is the most sane option for our purposes.
//
// int sizes:
//   sizeof(int) = 4
//   sizeof(unsigned int) = 4
//   sizeof(unsigned long int) = 8
//   sizeof(unsigned long long int) = 8
//   sizeof(uint64_t) = 8
//
// https://stackoverflow.com/questions/22904670/c-c-left-shift-unsigned-vs-signed
//

#define _64BIT_1 static_cast<uint64_t>(1)
//#define _64BIT_1 1UL
//#define _64BIT_1 1ULL
//#define _64BIT_1 1U
//#define _64BIT_1 1

TEST(TestBitset, Test_01_30_bits) {
  IthBitType ith_bit = 30;
  std::bitset<MAX_BITS> bitset(_64BIT_1 << ith_bit);

  // int sizes:
  //   sizeof(int) = 4
  //   sizeof(unsigned int) = 4
  //   sizeof(unsigned long int) = 8
  //   sizeof(unsigned long long int) = 8
  //   sizeof(uint64_t) = 8
  std::cout << "int sizes:" << std::endl;
  std::cout << "  sizeof(int) = " << sizeof(int) << std::endl;
  std::cout << "  sizeof(unsigned int) = " << sizeof(unsigned int) << std::endl;
  std::cout << "  sizeof(unsigned long int) = " << sizeof(unsigned long int) << std::endl;
  std::cout << "  sizeof(unsigned long long int) = " << sizeof(unsigned long long int) << std::endl;
  std::cout << "  sizeof(uint64_t) = " << sizeof(uint64_t) << std::endl;

  auto bitset_ullong = bitset.to_ullong();
  // NOTE: Compiler will warn us here that "_64BIT_1 = 1" is an "int" type
  // and hence "_64BIT_1 << ith_bit" is an int type.
  EXPECT_EQ(bitset_ullong, _64BIT_1 << ith_bit);

  auto bitset_string = bitset.to_string();
  auto expect_bitset_string = "0000000000000000000000000000000001000000000000000000000000000000";
  EXPECT_EQ(bitset_string, expect_bitset_string);
}

TEST(TestBitset, Test_01_31_bits) {
  IthBitType ith_bit = 31;
  std::bitset<MAX_BITS> bitset(_64BIT_1 << ith_bit);

  auto bitset_ullong = bitset.to_ullong();
  EXPECT_EQ(bitset_ullong, _64BIT_1 << ith_bit);

  auto bitset_string = bitset.to_string();
  auto expect_bitset_string = "0000000000000000000000000000000010000000000000000000000000000000";
  EXPECT_EQ(bitset_string, expect_bitset_string);
}

TEST(TestBitset, Test_01_31_bit_from_string) {
  IthBitType ith_bit = 31;
  auto expect_bitset_string = "0000000000000000000000000000000010000000000000000000000000000000";
  std::bitset<MAX_BITS> bitset(expect_bitset_string);

  auto bitset_string = bitset.to_string();
  EXPECT_EQ(bitset_string, expect_bitset_string);

  auto bitset_ullong = bitset.to_ullong();
  EXPECT_EQ(bitset_ullong, _64BIT_1 << ith_bit);
}

TEST(TestBitset, Test_01_32_bits) {
  IthBitType ith_bit = 32;
  std::bitset<MAX_BITS> bitset(_64BIT_1 << ith_bit);

  auto bitset_ullong = bitset.to_ullong();
  EXPECT_EQ(bitset_ullong, _64BIT_1 << ith_bit);

  auto bitset_string = bitset.to_string();
  auto expect_bitset_string = "0000000000000000000000000000000100000000000000000000000000000000";
  EXPECT_EQ(bitset_string, expect_bitset_string);
}

}
