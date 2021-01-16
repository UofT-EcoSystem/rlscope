//
// Created by jgleeson on 2020-01-21.
//

#ifndef RLSCOPE_SAMPLE_PERIODS_H
#define RLSCOPE_SAMPLE_PERIODS_H

#include "analysis/trace_file_parser.h"

#include <nlohmann/json.hpp>

namespace rlscope {

class PollingUtil {
public:
  const CategoryTimes &category_times;
  TimeUsec _polling_interval_us;
  std::string _rlscope_directory;

  PollingUtil(const CategoryTimes &category_times, TimeUsec polling_interval_us, const std::string& rlscope_directory) :
      category_times(category_times),
      _polling_interval_us(polling_interval_us),
      _rlscope_directory(rlscope_directory) {
  }

  std::string JSPath() const;

  std::string JSBasename() const;

  nlohmann::json Compute() const;

};

} // namespace rlscope

#endif //RLSCOPE_SAMPLE_PERIODS_H
