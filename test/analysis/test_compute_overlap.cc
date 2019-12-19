//
// Created by jagle on 11/20/2019.
//

#include <limits.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <iostream>

#include "analysis/trace_file_parser.h"

using namespace Eigen;

namespace tensorflow {

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

struct Event {
  TimeUsec start_us;
  TimeUsec end_us;
  Event(TimeUsec start_us_, TimeUsec end_us_) :
      start_us(start_us_)
      , end_us(end_us_)
  {
  }
};
using EventData = std::map<std::string, std::vector<Event>>;
using OverlapData = std::map<std::set<std::string>, TimeUsec>;
using RegionData = std::map<std::set<std::string>, Event>;
//using T = Event;

Event T(int64_t start_sec, int64_t end_sec) {
  auto start_us = start_sec*USEC_IN_SEC;
  auto end_us = end_sec*USEC_IN_SEC;
  return Event(start_us, end_us);
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

CategoryTimesCount CountCategoryTimes(const EventData& event_data) {
  CategoryTimesCount count;
  for (const auto& pair : event_data) {
    const auto& category_key = AsCategoryKey(pair.first);
    auto n_events = pair.second.size();
    count.Add(category_key, n_events);
  }
  return count;
}

CategoryTimes CategoryTimesFrom(const EventData& event_data) {
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
  OverlapResult r;
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
  EventData event_data = {
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
  EventData event_data = {
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

}
