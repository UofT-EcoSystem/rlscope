//
// Created by jagle on 11/13/2019.
//

#include "analysis/trace_file_parser.h"
#include "cuda_api_profiler/generic_logging.h"

#include <assert.h>

namespace tensorflow {


void EOEvents::Print(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "EOEvents: size = " << _n_events;
  for (size_t i = 0; i < _n_events; i++) {
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    auto start_us = _events[start_idx] / PSEC_IN_USEC;
    auto end_us = _events[end_idx] / PSEC_IN_USEC;
    auto dur_us = end_us - start_us;

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Event[" << i << "] = (start=" << start_us << " us, dur=" << dur_us << " us)";

  }
}

void PrintCategoryTimes(const CategoryTimes& category_times, std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << "CategoryTimes: size = " << category_times.size();
  size_t category_idx = 0;
  for (const auto& pair : category_times.eo_times) {
    const auto& category = pair.first;
    const auto& eo_times = pair.second;

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Category[" << category_idx << "] = " << category;

    out << "\n";
    eo_times.Print(out, indent + 2);

    category_idx += 1;
  }
}


MyStatus CategoryEventsParser::_CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const iml::CategoryEventsProto& proto) {
  for (const auto& pair : proto.category_events()) {
    const auto& category = pair.first;
    size_t n_events = pair.second.events().size();
    count->Add(category, n_events);
  }
  return MyStatus::OK();
}
MyStatus CategoryEventsParser::_AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const iml::CategoryEventsProto& proto) {
  MyStatus status = MyStatus::OK();
  for (const auto& pair : proto.category_events()) {
    const auto& category = pair.first;
    EOEvents& eo_events = out_category_times->eo_times.at(category);
    status = _AppendCategory(category, proto, &eo_events);
    IF_BAD_STATUS_RETURN(status);
  }
  return MyStatus::OK();
}
MyStatus CategoryEventsParser::_AppendCategory(const Category& category, const iml::CategoryEventsProto& proto, EOEvents* eo_events) {
  const auto& events = proto.category_events().at(category).events();
  size_t n_events = events.size();
  for (const auto& event : proto.category_events().at(category).events()) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    eo_events->AppendEvent(start_us, end_us);
  }
  return MyStatus::OK();
}

MyStatus CUDAAPIStatsParser::_CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const ProtoKlass& proto) {
  const std::string category = CATEGORY_CUDA_API_CPU;
  size_t n_events = proto.events().size();
  count->Add(category, n_events);
  return MyStatus::OK();
}
MyStatus CUDAAPIStatsParser::_AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const ProtoKlass& proto) {
  const std::string category = CATEGORY_CUDA_API_CPU;
  EOEvents& eo_events = out_category_times->eo_times.at(category);
  for (const auto& event : proto.events()) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    eo_events.AppendEvent(start_us, end_us);
  }
  return MyStatus::OK();
}

MyStatus CUDADeviceEventsParser::_CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const CUDADeviceEventsParser::ProtoKlass& proto) {
  const std::string category = CATEGORY_GPU;
  for (const auto& dev_events_pair : proto.dev_events()) {
    const auto& dev = dev_events_pair.first;
    const auto& events = dev_events_pair.second.events();
    size_t n_events = events.size();
    count->Add(category, n_events);
  }
  return MyStatus::OK();
}
MyStatus CUDADeviceEventsParser::_AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const CUDADeviceEventsParser::ProtoKlass& proto) {
  const std::string category = CATEGORY_GPU;
  EOEvents& eo_events = out_category_times->eo_times.at(category);
  for (const auto& dev_events_pair : proto.dev_events()) {
    const auto& dev = dev_events_pair.first;
    const auto& events = dev_events_pair.second.events();
    for (const auto& event : events) {
      auto start_us = event.start_time_us();
      auto end_us = event.start_time_us() + event.duration_us();
      eo_events.AppendEvent(start_us, end_us);
    }
  }
  return MyStatus::OK();
}

MyStatus FindRLSFiles(const std::string& iml_directory, std::list<std::string>* paths) {
  return RecursiveFindFiles(paths, iml_directory, [] (const boost::filesystem::path& bpath) {
    if (!boost::filesystem::is_regular_file(bpath)) {
      return false;
    }
    auto file_type = GetRLSFileType(bpath.string());
    return file_type != RLSFileType::UNKNOWN_FILE;
  });
}

CategoryTimes::CategoryTimes(const CategoryTimesCount& count) {
  // Use count to preallocate space.
  for (const auto& pair : count.num_events) {
    auto const& category = pair.first;
    auto const n_events = pair.second;
    eo_times[category] = EOEvents(n_events);
  }
}

bool isRLSFile(RLSFileType file_type, const std::string& path) {
  boost::filesystem::path bpath(path);

  auto matches_regex = [&bpath] (const std::string& regex) {
    std::regex file_regex(regex);
    return std::regex_match(bpath.filename().string(), file_regex);
  };

  switch (file_type) {
    case RLSFileType::CUDA_API_STATS_FILE:
      return matches_regex(CUDA_API_STATS_REGEX);
      break;
    case RLSFileType::CATEGORY_EVENTS_FILE:
      return matches_regex(CATEGORY_EVENTS_REGEX);
      break;
    case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
      return matches_regex(CUDA_DEVICE_EVENTS_REGEX);
      break;
    case RLSFileType::UNKNOWN_FILE:
      return false;
  }
  // Not sure how to handle this file type.
  assert(false);
  return false;
}

RLSFileType GetRLSFileType(const std::string& path) {
  boost::filesystem::path bpath(path);

  auto matches_regex = [&bpath] (const std::string& regex) {
    std::regex file_regex(regex);
    return std::regex_match(bpath.filename().string(), file_regex);
  };

  if (matches_regex(CATEGORY_EVENTS_REGEX)) {
    return RLSFileType::CATEGORY_EVENTS_FILE;
  }
  if (matches_regex(CUDA_API_STATS_REGEX)) {
    return RLSFileType::CUDA_API_STATS_FILE;
  }
  if (matches_regex(CUDA_DEVICE_EVENTS_REGEX)) {
    return RLSFileType::CUDA_DEVICE_EVENTS_FILE;
  }

  return RLSFileType::UNKNOWN_FILE;

#undef IF_MATCH_RETURN_TYPE
}

MyStatus GetRLSEventParser(const std::string& path, std::unique_ptr<IEventFileParser>* parser) {
  auto file_type = GetRLSFileType(path);
  switch (file_type) {
    case RLSFileType::CUDA_API_STATS_FILE:
      parser->reset(new CUDAAPIStatsParser());
      break;
    case RLSFileType::CATEGORY_EVENTS_FILE:
      parser->reset(new CategoryEventsParser());
      break;
    case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
      parser->reset(new CUDADeviceEventsParser());
      break;
    default:
      assert(false);
  }
  return MyStatus::OK();
}

std::list<std::unique_ptr<IEventFileParser>> AllRLSParsers() {
  std::list<std::unique_ptr<IEventFileParser>> parsers;
  parsers.emplace_back(new CategoryEventsParser());
  parsers.emplace_back(new CUDAAPIStatsParser());
  parsers.emplace_back(new CUDADeviceEventsParser());
  return parsers;
}

} // namespace tensorflow

