//
// Created by jgleeson on 2020-01-21.
//

#ifndef IML_SAMPLE_PERIODS_H
#define IML_SAMPLE_PERIODS_H

#include "analysis/trace_file_parser.h"

#include <nlohmann/json.hpp>

namespace tensorflow {

class SamplePeriods {
public:
  const CategoryTimes &category_times;
  TimeUsec _polling_interval_us;
  std::string _iml_directory;

  SamplePeriods(const CategoryTimes &category_times, TimeUsec polling_interval_us, const std::string& iml_directory) :
      category_times(category_times),
      _polling_interval_us(polling_interval_us),
      _iml_directory(iml_directory) {
  }

  std::string JSPath() const;

  std::string JSBasename() const;

  nlohmann::json Compute() const;

};

} // namespace tensorflow

#endif //IML_SAMPLE_PERIODS_H
