//
// Created by jagle on 11/20/2019.
//

#include <limits.h>
#include <gtest/gtest.h>

#include "analysis/trace_file_parser.h"

namespace tensorflow {

//TEST(FailingTest, OneEqZero) {
// EXPECT_EQ(1, 0);
//}

TEST(TestCategoryKey, MapContainsKey) {
  std::map<CategoryKey, int> cmap;
  auto process = "process";
  auto category = "category";
  auto category_key = CategoryKey::FromCategory(process, category);
  cmap[category_key] = 1;
  bool contains = (cmap.find(category_key) != cmap.end());
  EXPECT_EQ(contains, true);
}

TEST(TestCategoryKey, MapContainsKeyCopy) {
  std::map<CategoryKey, int> cmap;
  auto process = "process";
  auto category = "category";
  auto category_key = CategoryKey::FromCategory(process, category);
  cmap[category_key] = 1;
  bool contains = (cmap.find(category_key) != cmap.end());
  EXPECT_EQ(contains, true);


  auto process_copy = "process";
  auto category_copy = "category";
  auto category_key_copy = CategoryKey::FromCategory(process_copy, category_copy);
  contains = (cmap.find(category_key_copy) != cmap.end());
  EXPECT_EQ(contains, true);
}

}
