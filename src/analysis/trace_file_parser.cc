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


MyStatus CategoryEventsParser::CountCategoryTimes(const std::string& path, CategoryTimesCount* count) {
  MyStatus status = MyStatus::OK();
  iml::CategoryEventsProto proto;
  status = _ReadProto(path, &proto);
  IF_BAD_STATUS_RETURN(status);
  return _CountCategoryTimes(path, count, proto);
}

MyStatus CategoryEventsParser::_CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const iml::CategoryEventsProto& proto) {
  for (const auto& pair : proto.category_events()) {
    const auto& category = pair.first;
    size_t n_events = pair.second.events().size();
    count->Add(category, n_events);
  }
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::ReadFile(const std::string& path, CategoryTimes* out_category_times) {
  MyStatus status = MyStatus::OK();

  iml::CategoryEventsProto proto;
  status = _ReadProto(path, &proto);
  IF_BAD_STATUS_RETURN(status);

  CategoryTimesCount count;
  status = _CountCategoryTimes(path, &count, proto);
  IF_BAD_STATUS_RETURN(status);

  *out_category_times = std::move(CategoryTimes(count));
  status = _AppendCategoryTimes(path, out_category_times, proto);
  IF_BAD_STATUS_RETURN(status);

  return MyStatus::OK();
}

MyStatus CategoryEventsParser::AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times) {
  MyStatus status = MyStatus::OK();
  iml::CategoryEventsProto proto;
  status = _ReadProto(path, &proto);
  IF_BAD_STATUS_RETURN(status);
  return _AppendCategoryTimes(path, out_category_times, proto);
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

MyStatus CategoryEventsParser::_ReadProto(const std::string& path, iml::CategoryEventsProto* proto) {
  MyStatus status = ParseProto("category_events", path, proto);
  IF_BAD_STATUS_RETURN(status);
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

MyStatus FindRLSFiles(const std::string& iml_directory, RLSFileType rls_file_type, std::list<std::string>* paths) {
  return RecursiveFindFiles(paths, iml_directory, [rls_file_type] (const boost::filesystem::path& path) {
    if (!boost::filesystem::is_regular_file(path)) {
      return false;
    }
    switch (rls_file_type) {
      case RLSFileType::CUDA_API_STATS_FILE:
        assert(false);
        break;
      case RLSFileType::CATEGORY_EVENTS_FILE:
        return CategoryEventsParser::IsFile(path.string());
        break;
      case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
        assert(false);
        break;
      default:
        assert(false);
    }
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

} // namespace tensorflow

