//
// Created by jagle on 11/21/2019.
//
#include <spdlog/spdlog.h>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
