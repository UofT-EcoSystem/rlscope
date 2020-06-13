//
// Created by jgleeson on 2020-06-13.
//

#include <gtest/gtest.h>

#include "common_util.h"

using namespace rlscope;

TEST(TestCommonUtil, Test_StringSplit_01_One) {
  auto got = StringSplit("one", ",");
  std::vector<std::string> expect = {"one"};
  EXPECT_EQ(got, expect);
}

TEST(TestCommonUtil, Test_StringSplit_02_Empy) {
  auto got = StringSplit("", ",");
  std::vector<std::string> expect = {""};
  EXPECT_EQ(got, expect);
}
