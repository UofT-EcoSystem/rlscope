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
      eo_events.AppendEvent(start_us, end_us);
    }
  }

  return category_times;
}


OverlapResult OverlapFrom(const OverlapComputer& overlap_computer, const OverlapData& overlap_data,
    bool debug=false) {
  OverlapResult r;
  r.idx_map = overlap_computer.ctimes.idx_map;

  for (const auto& pair : overlap_data) {
    CategoryKeyBitset bitset = CategoryKeyBitset::EmptySet(r.idx_map);
    bitset.debug = debug;
    std::set<size_t> indices;
    std::vector<CategoryKey> category_keys;
    for (auto const& key : pair.first) {
      const auto category_key = AsCategoryKey(key);
      category_keys.push_back(category_key);
      auto category_idx = r.idx_map->Idx(category_key);
      indices.insert(category_idx);
      bitset.Add(category_idx);
    }

    if (debug) {
      std::stringstream ss;
      ss << "\n";

      ss << "set = ";
      PrintValue(ss, pair.first);
      ss << "\n";

      ss << "indices = ";
      PrintValue(ss, indices);
      ss << "\n";

      ss << "category_keys = ";
      PrintValue(ss, category_keys);
      ss << "\n";

      bitset.Print(ss, 1);
      ss << "\n";

      PrintIndent(ss, 1);
      ss << "TimeUs = " << pair.second << "\n";

      SPDLOG_DEBUG("{}", ss.str());
    }

    r.overlap[bitset] = pair.second;
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

//  std::cout << "CategoryTimes:\n";
//  category_times.Print(std::cout, 1);
//  std::cout << "\n";

  OverlapComputer overlap_computer(category_times);
  // overlap_computer.debug = true;

  auto expect_r = OverlapFrom(overlap_computer, overlap_data);
  auto got_r = overlap_computer.ComputeOverlap();
  // TODO: lookup how to use custom printers with gtest for got/expect when test fails.

  bool overlap_eq = (got_r.overlap == expect_r.overlap);
  PrintIfFail(overlap_eq, got_r, expect_r);
//  if (overlap_eq) {
//    PrintGot(got_r);
//    PrintExpect(expect_r);
//  }
  EXPECT_TRUE(overlap_eq);

}

}
