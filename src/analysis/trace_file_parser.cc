//
// Created by jagle on 11/13/2019.
//

#include "analysis/trace_file_parser.h"
#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/debug_flags.h"

#include <Eigen/Dense>

//#include <boost/compute/algorithm/reduce.hpp>

// NOTE: not until gcc 7.1 (7.4.0 in Ubuntu 18.04); use boost::optional instead.
// #include <optional>
#include <boost/optional.hpp>
#include <boost/utility/string_view.hpp>

//#include <string_view>
#include <iomanip>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <assert.h>
#include <numeric>
#include <iostream>
#include <spdlog/spdlog.h>
#include <limits>
#include <regex>

using namespace Eigen;

namespace tensorflow {

const std::regex PROCESS_OPERATION_REGEX = std::regex(R"(\[.*\])");

const std::vector<RLSFileType> RLS_FILE_TYPES = {
    CUDA_API_STATS_FILE,
    CATEGORY_EVENTS_FILE,
    CUDA_DEVICE_EVENTS_FILE,
};

const std::set<Category> CATEGORIES_C_EVENTS = std::set<Category>{
    CATEGORY_TF_API,
    CATEGORY_SIMULATOR_CPP,
};

const std::set<Category> CATEGORIES_PROF = std::set<Category>{
    CATEGORY_PROF_CUPTI,
    CATEGORY_PROF_LD_PRELOAD,
    CATEGORY_PROF_PYTHON_ANNOTATION,
    CATEGORY_PROF_PYTHON_CLIB_INTERCEPTION_TENSORFLOW,
    CATEGORY_PROF_PYTHON_CLIB_INTERCEPTION_SIMULATOR,
};

const std::set<Category> CATEGORIES_CPU = std::set<Category>{
    CATEGORY_TF_API,
    CATEGORY_PYTHON,
    CATEGORY_CUDA_API_CPU,
    CATEGORY_SIMULATOR_CPP,
    CATEGORY_PYTHON_PROFILER,
    // NOTE: need to have this here to handle overhead subtraction correctly.
    CATEGORY_CPU,
};

const std::set<Category> CATEGORIES_GPU = std::set<Category>{
    CATEGORY_GPU,
};

const std::set<OverlapType> OVERLAP_TYPES = std::set<OverlapType>{
    "ResourceOverlap",
    "CategoryOverlap",
    "OperationOverlap",
    "ResourceSubplot",
    "ProfilingOverhead",
};

template <typename Func, class EventListIter>
MyStatus EachEvent(const EventListIter& events, Func func) {
  MyStatus status = MyStatus::OK();
  for (const auto& event : events) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    auto const& name = event.name();
    status = func(name, start_us, end_us);
    IF_BAD_STATUS_RETURN(status);
  }
  return MyStatus::OK();
}

MyStatus ReadJson(std::string path, json* j) {
  boost::filesystem::path file(path);

  if(!boost::filesystem::exists(file)) {
    std::stringstream ss;
    ss << "Couldn't find json file @ path=" << path;
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  // read a JSON file
  std::ifstream inp(path);
  try {
    inp >> *j;
  } catch (nlohmann::detail::parse_error e) {
    std::stringstream ss;
    ss << "Failed to parse json file @ path=" << path << ":\n";
    ss << e.what();
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }

  return MyStatus::OK();
}

MyStatus WriteJson(std::string path, const nlohmann::json& j) {
  boost::filesystem::path file(path);
  auto parent = file.parent_path();
  if(!boost::filesystem::exists(parent)) {
    std::stringstream ss;
    ss << "Couldn't find directory of json file @ directory=" << parent;
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  // read a JSON file
  std::ofstream out(path);
  try {
    // j.dump()
    out << std::setw(4) << j;
  } catch (nlohmann::detail::parse_error e) {
    std::stringstream ss;
    ss << "Failed to write json file @ path=" << path << ":\n";
    ss << e.what();
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }

  return MyStatus::OK();
}

void EOEvents::Print(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "EOEvents: size = " << this->size() << ", capacity = " << this->capacity();
  for (size_t i = 0; i < this->size(); i++) {
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    auto start_us = (*_events)[start_idx] / PSEC_IN_USEC;
    auto end_us = (*_events)[end_idx] / PSEC_IN_USEC;
    auto dur_us = end_us - start_us;

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Event[" << i << "] = (start=" << start_us << " us, dur=" << dur_us << " us)";

  }
}
void EOEvents::PrintSummary(std::ostream& out, int indent) const {
  double total_sec = 0;
  for (size_t i = 0; i < this->size(); i++) {
    total_sec += ((double)this->DurationUsec(i)) / ((double)USEC_IN_SEC);
  }
  PrintIndent(out, indent);
  out << "EOEvents: size = " << this->size() << ", capacity = " << this->capacity() << ", duration = " << total_sec << " sec";
}
void EOEvents::CheckIntegrity(std::ostream& out, int indent) const {
  for (size_t i = 0; i < this->size(); i++) {
    assert((*_events)[EVENT_START_IDX(i)] <= (*_events)[EVENT_END_IDX(i)]);
    if (i > 0) {
      assert((*_events)[EVENT_END_IDX(i-1)] <= (*_events)[EVENT_START_IDX(i)]);
    }
  }
}

//MyStatus CategoryEventsParser::_CountCategoryTimes(
//    CategoryTimesCount* count,
//    ProtoKlass* proto,
//    boost::optional<const TraceFileMeta&> next_meta) {
//  MyStatus status = MyStatus::OK();
//  for (const auto& pair : proto->category_events()) {
//    size_t number_of_events = pair.second.events().size();
//    const auto& category = pair.first;
//    if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//      DBG_LOG("Count category={}, n_events={}", category, number_of_events);
//    }
//    if (category == CATEGORY_OPERATION) {
//      status = _CountCategoryTimesOperation(count, proto, next_meta);
//      IF_BAD_STATUS_RETURN(status);
//    } else if (!FEATURE_JUST_OPERATIONS) {
//      auto category_key = CategoryKey::FromCategory(get_process(), category);
//      size_t n_events = pair.second.events().size();
//      // Q: is this a requirement...?
//      assert(n_events > 0);
//      count->Add(category_key, n_events);
//    }
//  }
//  return MyStatus::OK();
//}

TimeUsec StartOperationTimeUsec(const TraceFileMeta& meta) {
//  return meta.parser_meta["start_operation_usec"];
  return meta.parser_meta["start_usec"][CATEGORY_OPERATION];
}

//MyStatus CategoryEventsParser::_CountCategoryTimesOperation(
//    CategoryTimesCount* count,
//    ProtoKlass* proto,
//    boost::optional<const TraceFileMeta&> next_meta) {
//  MyStatus status = MyStatus::OK();
//  auto const& process = get_process();
//  auto const& events = proto->category_events().at(CATEGORY_OPERATION).events();
////  DBG_LOG("CategoryEventsParser::ParserMeta.num_CATEGORY_OPERATION = {}", events.size());
//  count->AddExtra(CategoryKey::FromCategory(process, CATEGORY_EXTRA_OPERATION), events.size());
//
//  auto add_op = [&process, count] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
//    auto category_key = CategoryKey::FromOpEvent(process, op);
//    count->Add(category_key, 1);
//  };
//  auto skip_func = [] (const auto& event) -> bool {
//    return CategoryEventsProtoReader::SkipOperationEvent(event.name());
//  };
//
//  if (next_meta) {
//    _event_flattener.ProcessUntil(events, StartOperationTimeUsec(next_meta.get()), add_op, skip_func);
//  } else {
//    _event_flattener.ProcessUntilFinish(events, add_op, skip_func);
//  }
//
////  EventFlattener<iml::Event>::EachOpEvent(
////      events,
////      [&process, count] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
////        auto category_key = CategoryKey::FromOpEvent(process, op);
////        count->Add(category_key, 1);
////      });
//
//  return MyStatus::OK();
//}

//MyStatus CategoryEventsParser::AppendAllCategory(CategoryTimes* out_category_times, const Category& category, const EventList& events) {
//  MyStatus status = MyStatus::OK();
//  CategoryKey category_key = CategoryKey::FromCategory(get_process(), category);
//  EOEvents& eo_events = out_category_times->MutableEvents(category_key);
//  status = EachEvent(events, [&eo_events] (const EventName& name, TimeUsec start_us, TimeUsec end_us) {
//    eo_events.AppendEvent(name, start_us, end_us);
//    return MyStatus::OK();
//  });
//  IF_BAD_STATUS_RETURN(status);
//}
//MyStatus CategoryEventsParser::_AppendCategoryTimes(
//    CategoryTimes* out_category_times,
//    ProtoKlass* proto,
//    boost::optional<const TraceFileMeta&> next_meta) {
//  MyStatus status = MyStatus::OK();
//  for (const auto& pair : proto->category_events()) {
//    const auto& category = pair.first;
//    if (category == CATEGORY_OPERATION) {
//      status = _AppendCategoryOperation(category, proto, out_category_times, next_meta);
//      IF_BAD_STATUS_RETURN(status);
//    } else if (!FEATURE_JUST_OPERATIONS) {
//      CategoryKey category_key = CategoryKey::FromCategory(get_process(), category);
//      EOEvents& eo_events = out_category_times->MutableEvents(category_key);
//      status = _AppendCategory(category, proto, &eo_events);
//      IF_BAD_STATUS_RETURN(status);
//    }
//  }
//  return MyStatus::OK();
//}

MyStatus CategoryEventsParser::AppendAllCategory(
    const EntireTraceMeta& entire_meta,
    const Category& category,
    const std::vector<typename ProtoReader::EventKlass>& events,
    CategoryTimes* out_category_times) {
  auto status = MyStatus::OK();
  if (category == CATEGORY_OPERATION) {
    status = _AppendCategoryOperation(
        entire_meta,
        category,
        events,
        out_category_times);
    IF_BAD_STATUS_RETURN(status);
  } else if (!FEATURE_JUST_OPERATIONS) {
    status = DefaultAppendAllCategory(
        entire_meta,
        category,
        events,
        out_category_times);
    IF_BAD_STATUS_RETURN(status);
  }
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::_AppendCategoryOperation(
    const EntireTraceMeta& entire_meta,
    const Category& category,
    const std::vector<typename ProtoReader::EventKlass>& events,
    CategoryTimes* out_category_times) {
  MyStatus status = MyStatus::OK();
  auto const& process = get_process();
  assert(category == CATEGORY_OPERATION);

  auto append_event = [this, &process, out_category_times] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
    auto category_key = CategoryKey::FromOpEvent(process, op);
    auto& eo_events = out_category_times->MutableEvents(category_key);
    eo_events.AppendEvent(op, start_us, end_us);
  };
  auto skip_func = [] (const auto& event) -> bool {
    return CategoryEventsProtoReader::SkipOperationEvent(event.name());
  };
  CategoryTimesCount count;
  auto add_op = [&process, &count] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
    auto category_key = CategoryKey::FromOpEvent(process, op);
    count.Add(category_key, 1);
  };

  status = DefaultAppendAllCategoryExtra(entire_meta, CATEGORY_EXTRA_OPERATION, events, out_category_times);
  IF_BAD_STATUS_RETURN(status);

//  // auto const& events = proto->category_events().at(category).events();
//  auto extra_op_category_key = CategoryKey::FromCategory(process, CATEGORY_EXTRA_OPERATION);
//  out_category_times->PreallocateExtra(extra_op_category_key, events.size());
//  auto& extra_op_events = out_category_times->MutableEventsExtra(extra_op_category_key);
//  status = EachEvent(events, [&extra_op_events] (const EventName& name, TimeUsec start_us, TimeUsec end_us) {
//    extra_op_events.AppendEvent(name, start_us, end_us);
//    return MyStatus::OK();
//  });
//  IF_BAD_STATUS_RETURN(status);
//  // Watch out for accidental copy-construction.
//  assert(out_category_times->extra_eo_times.at(extra_op_category_key).size() > 0);

  _event_flattener.ProcessUntilFinish(events, add_op, skip_func);
  out_category_times->Preallocate(count);

//  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//    std::stringstream ss;
//    ss << "Count of events for adding flattened CATEGORY_OPERATION's:";
//    ss << "\n";
//    count.Print(ss, 1);
//    DBG_LOG("{}", ss.str());
//  }

  _event_flattener.ProcessUntilFinish(events, append_event, skip_func);

  return MyStatus::OK();
}

//MyStatus CategoryEventsParser::_AppendCategory(const Category& category, ProtoKlass* proto, EOEvents* eo_events) {
//  MyStatus status = MyStatus::OK();
//  const auto& events = proto->category_events().at(category).events();
//  status = EachEvent(events, [eo_events] (const EventName& name, TimeUsec start_us, TimeUsec end_us) {
//    eo_events->AppendEvent(name, start_us, end_us);
//    return MyStatus::OK();
//  });
//  IF_BAD_STATUS_RETURN(status);
//  return MyStatus::OK();
//}

MyStatus CUDAAPIStatsParser::_CountCategoryTimes(CategoryTimesCount* count, ProtoKlass* proto) {
  if (FEATURE_JUST_OPERATIONS) {
    return MyStatus::OK();
  }
  const std::string category = CATEGORY_CUDA_API_CPU;
  size_t n_events = proto->events().size();
  auto category_key = CategoryKey::FromCategory(get_process(), category);
  count->Add(category_key, n_events);
  return MyStatus::OK();
}
MyStatus CUDAAPIStatsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, ProtoKlass* proto) {
  if (FEATURE_JUST_OPERATIONS) {
    return MyStatus::OK();
  }
  MyStatus status = MyStatus::OK();
  const std::string category = CATEGORY_CUDA_API_CPU;
  auto category_key = CategoryKey::FromCategory(get_process(), category);
  EOEvents& eo_events = out_category_times->MutableEvents(category_key);
  const auto& events = proto->events();



  for (const auto& event : events) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    auto const &name = event.api_name();
    eo_events.AppendEvent(name, start_us, end_us);
  }
  return MyStatus::OK();
}

MyStatus CUDADeviceEventsParser::_CountCategoryTimes(CategoryTimesCount* count, ProtoKlass* proto) {
  if (FEATURE_JUST_OPERATIONS) {
    return MyStatus::OK();
  }
  const std::string category = CATEGORY_GPU;
  for (const auto& dev_events_pair : proto->dev_events()) {
    const auto& dev = dev_events_pair.first;
    const auto& events = dev_events_pair.second.events();
    size_t n_events = events.size();
    auto category_key = CategoryKey::FromCategory(get_process(), category);
    count->Add(category_key, n_events);
  }
  return MyStatus::OK();
}
MyStatus CUDADeviceEventsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, ProtoKlass* proto) {
  if (FEATURE_JUST_OPERATIONS) {
    return MyStatus::OK();
  }
  const std::string category = CATEGORY_GPU;
  auto category_key = CategoryKey::FromCategory(get_process(), category);
  EOEvents& eo_events = out_category_times->MutableEvents(category_key);
  for (const auto& dev_events_pair : proto->dev_events()) {
    const auto& dev = dev_events_pair.first;
    const auto& events = dev_events_pair.second.events();
    for (const auto& event : events) {
      auto start_us = event.start_time_us();
      auto end_us = event.start_time_us() + event.duration_us();
      eo_events.AppendEvent(event.name(), start_us, end_us);
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

bool CategoryShouldKeepNames(const CategoryKey& key) {
  return key.non_ops.find(CATEGORY_CUDA_API_CPU) != key.non_ops.end();
}

size_t CategoryTimes::TotalEvents() const {
  size_t n_events = 0;
  for (const auto& pair : eo_times) {
    auto const& category_key = pair.first;
    auto const& eo_events = pair.second;
    n_events += eo_events.size();
  }
  return n_events;
}

void CategoryTimes::_Preallocate(EOTimes* eo_times, const CategoryKey& category_key, size_t n_events) {
  bool keep_names = CategoryShouldKeepNames(category_key);
  (*eo_times)[category_key] = EOEvents(n_events, keep_names);
}
void CategoryTimes::Preallocate(const CategoryTimesCount& count) {
  for (const auto& pair : count.num_events) {
    const auto& category_key = pair.first;
    auto n_events = pair.second;
    Preallocate(category_key, n_events);
  }
}
void CategoryTimes::Preallocate(const CategoryKey& category_key, size_t n_events) {
  _Preallocate(&eo_times, category_key, n_events);
}
void CategoryTimes::PreallocateExtra(const CategoryKey& category_key, size_t n_events) {
  _Preallocate(&extra_eo_times, category_key, n_events);
}

const EOEvents& CategoryTimes::Events(const CategoryKey& category_key) {
  return _Events(&eo_times, category_key);
}
EOEvents& CategoryTimes::MutableEvents(const CategoryKey& category_key) {
  return _MutableEvents(&eo_times, category_key);
}
boost::optional<const EOEvents&> CategoryTimes::MaybeEvents(const CategoryKey& category_key) const {
  return _MaybeEvents(eo_times, category_key);
}
boost::optional<EOEvents&> CategoryTimes::MaybeMutableEvents(const CategoryKey& category_key) {
  return _MaybeMutableEvents(&eo_times, category_key);
}


const EOEvents& CategoryTimes::EventsExtra(const CategoryKey& category_key) {
  return _Events(&extra_eo_times, category_key);
}
EOEvents& CategoryTimes::MutableEventsExtra(const CategoryKey& category_key) {
  return _MutableEvents(&extra_eo_times, category_key);
}
boost::optional<const EOEvents&> CategoryTimes::MaybeEventsExtra(const CategoryKey& category_key) const {
  return _MaybeEvents(extra_eo_times, category_key);
}
boost::optional<EOEvents&> CategoryTimes::MaybeMutableEventsExtra(const CategoryKey& category_key) {
  return _MaybeMutableEvents(&extra_eo_times, category_key);
}

const EOEvents& CategoryTimes::_Events(EOTimes* eo_times, const CategoryKey& category_key) {
  return (*eo_times)[category_key];
}
EOEvents& CategoryTimes::_MutableEvents(EOTimes* eo_times, const CategoryKey& category_key) {
  return (*eo_times)[category_key];
}
boost::optional<const EOEvents&> CategoryTimes::_MaybeEvents(const EOTimes& eo_times, const CategoryKey& category_key) const {
  auto it = eo_times.find(category_key);
  boost::optional<const EOEvents&> events;
  if (it != eo_times.end()) {
    events = it->second;
  }
  return events;
}
boost::optional<EOEvents&> CategoryTimes::_MaybeMutableEvents(EOTimes* eo_times, const CategoryKey& category_key) {
  auto it = eo_times->find(category_key);
  boost::optional<EOEvents&> events;
  if (it != eo_times->end()) {
    events = it->second;
  }
  return events;
}



size_t CategoryTimes::_Count(const EOTimes& eo_times, const CategoryKey& category_key) const {
  auto it = eo_times.find(category_key);
  if (it == eo_times.end()) {
    return 0;
  }
  return it->second.size();
}
size_t CategoryTimes::Count(const CategoryKey& category_key) const {
  return _Count(eo_times, category_key);
}
size_t CategoryTimes::CountExtra(const CategoryKey& category_key) const {
  return _Count(extra_eo_times, category_key);
}

CategoryTimes::CategoryTimes(const Process& process_, const CategoryTimesCount& count) :
    process(process_) {
  // Use count to preallocate space.
  auto preallocate = [] (const std::string& name, EOTimes* eo_times, const CategoryTimesCount::CountMap& cmap) {
    for (const auto& pair : cmap) {
      auto const& category_key = pair.first;
      auto const n_events = pair.second;
      bool keep_names = CategoryShouldKeepNames(category_key);
      if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
        DBG_LOG("Preallocate({}) category={}, n_events={}",
            name,
            category_key,
            n_events);
      }
      (*eo_times)[category_key] = EOEvents(n_events, keep_names);
    }
  };
  preallocate("eo_times", &eo_times, count.num_events);
  preallocate("extra_eo_times", &extra_eo_times, count.extra_num_events);
//  for (const auto& pair : count.num_events) {
//    auto const& category_key = pair.first;
//    auto const n_events = pair.second;
//    bool keep_names = CategoryShouldKeepNames(category_key);
//    eo_times[category_key] = EOEvents(n_events, keep_names);
//  }
}
template <class Map>
void CategoryTimesPrint(std::string name, const Map& eo_times, std::ostream& out, int indent) {
  PrintIndent(out, indent);
  // e.g. name = CategoryTimes
  out << name << ": size = " << eo_times.size();
  size_t category_idx = 0;
  for (const auto& pair : eo_times) {
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
template <class Map>
void CategoryTimesPrintSummary(std::string name, const Map& eo_times, std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << name << ": size = " << eo_times.size();
  size_t category_idx = 0;
  for (const auto& pair : eo_times) {
    const auto& category = pair.first;
    const auto& eo_times = pair.second;

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Category[" << category_idx << "] = " << category;

    out << "\n";
    eo_times.PrintSummary(out, indent + 2);

    category_idx += 1;
  }
}
void CategoryTimes::Print(std::ostream& out, int indent) const {
  CategoryTimesPrint("CategoryTime", this->eo_times, out, indent);
  if (this->extra_eo_times.size() > 0) {
    out << "\n";
    CategoryTimesPrint("CategoryTimeExtra", this->extra_eo_times, out, indent);
  }
}
void CategoryTimes::PrintSummary(std::ostream& out, int indent) const {
  CategoryTimesPrintSummary("CategoryTime", this->eo_times, out, indent);
  if (this->extra_eo_times.size() > 0) {
    out << "\n";
    CategoryTimesPrintSummary("CategoryTimeExtra", this->extra_eo_times, out, indent);
  }
}
void CategoryTimesBitset::Print(std::ostream& out, int indent) const {
  CategoryTimesPrint("CategoryTimeBitset", this->eo_times, out, indent);
}
void CategoryTimesBitset::PrintSummary(std::ostream& out, int indent) const {
  CategoryTimesPrintSummary("CategoryTimeBitset", this->eo_times, out, indent);
}

void CategoryTimesCount::Print(std::ostream& out, int indent) const {
  _Print(this->num_events, "CategoryTimesCount", out, indent);
  if (this->extra_num_events.size() > 0) {
    out << "\n";
    _Print(this->extra_num_events, "CategoryTimesCountExtra", out, indent);
  }
}
void CategoryTimesCount::_Print(const CountMap& cmap, const std::string& name, std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  // e.g. CategoryTimesCount
  out << name << ": size = " << cmap.size();
  size_t category_idx = 0;
  for (const auto& pair : cmap) {
    const auto& category_key = pair.first;
    const auto& count = pair.second;

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Category[" << category_idx << "] = " << category_key << ", n_events = " << count;

    category_idx += 1;
  }
}
void CategoryTimes::CheckIntegrity(std::ostream& out, int indent) const {
  for (const auto& pair : eo_times) {
    auto const &category_key = pair.first;
    auto const &eo_events = pair.second;
    eo_events.CheckIntegrity(out, indent);
  }
}

bool isRLSFileWithType(RLSFileType file_type, const std::string& path) {
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

bool isRLSFile(const std::string& path) {
  auto file_type = GetRLSFileType(path);
  return file_type != RLSFileType::UNKNOWN_FILE;
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

}
const char* RLSFileTypeString(RLSFileType file_type) {
  switch (file_type) {
    case CUDA_API_STATS_FILE:
      return "CUDA_API_STATS_FILE";
    case CATEGORY_EVENTS_FILE:
      return "CATEGORY_EVENTS_FILE";
    case CUDA_DEVICE_EVENTS_FILE:
      return "CUDA_DEVICE_EVENTS_FILE";
    case UNKNOWN_FILE:
      return "UNKNOWN_FILE";
  }
  assert(false);
  return "UNKNOWN_RLS_FILE_TYPE";
}

//template <class ParserKlass>
//typename ParserKlass::ParserMeta* GenericGetParserMetaDerived(const ParserKlass& self) {
//  assert(self._file_type == ParserKlass::FILE_TYPE);
//  assert(self._parser_meta->file_type == ParserKlass::FILE_TYPE);
//  typename ParserKlass::ParserMeta* parser_meta = reinterpret_cast<typename ParserKlass::ParserMeta*>(self._parser_meta.get());
//  return parser_meta;
//}

//template <class ParserKlass>
//typename ParserKlass::ParserMeta* AsDerivedParserMeta(const std::shared_ptr<IParserMeta>& parser_meta) {
//  assert(parser_meta->file_type == ParserKlass::FILE_TYPE);
//  typename ParserKlass::ParserMeta* pm = reinterpret_cast<typename ParserKlass::ParserMeta*>(parser_meta.get());
//  return pm;
//}

#define DEFINE_GET_PARSER_META(ParserKlass) \
  ParserKlass::ParserMeta* ParserKlass::GetParserMetaDerived() const { \
    return GenericGetParserMetaDerived<ParserKlass>(*this); \
  }

//DEFINE_GET_PARSER_META(CUDAAPIStatsParser)
//DEFINE_GET_PARSER_META(CUDADeviceEventsParser)
//DEFINE_GET_PARSER_META(CategoryEventsParser)

MyStatus GetRLSEventParser(const std::string& path, TraceParserMeta parser_meta, std::unique_ptr<IEventFileParser>* parser) {
  auto file_type = GetRLSFileType(path);
  auto status = GetRLSEventParserFromType(file_type, parser_meta, parser);
  return status;
}

MyStatus GetRLSEventParserFromType(RLSFileType file_type, TraceParserMeta parser_meta, std::unique_ptr<IEventFileParser>* parser) {
  switch (file_type) {
    case RLSFileType::CUDA_API_STATS_FILE:
      parser->reset(new CUDAAPIStatsParser(std::move(parser_meta)));
      break;
    case RLSFileType::CATEGORY_EVENTS_FILE:
      parser->reset(new CategoryEventsParser(std::move(parser_meta)));
      break;
    case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
      parser->reset(new CUDADeviceEventsParser(std::move(parser_meta)));
      break;
    default:
      assert(false);
  }
  return MyStatus::OK();
}

MyStatus GetTraceProtoReader(const std::string& path, std::unique_ptr<ITraceProtoReader>* reader) {
  auto file_type = GetRLSFileType(path);
  switch (file_type) {
    case RLSFileType::CUDA_API_STATS_FILE:
      reader->reset(new CUDAAPIStatsProtoReader(path));
      break;
    case RLSFileType::CATEGORY_EVENTS_FILE:
      reader->reset(new CategoryEventsProtoReader(path));
      break;
    case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
      reader->reset(new CUDADeviceEventsProtoReader(path));
      break;
    default:
      assert(false);
  }
  return MyStatus::OK();
}

void CategoryTimesCount::_AddToCategoryTimes(const CategoryTimesCount& ctimes) {
  auto add_to = [] (const CountMap& from_map, CountMap* to_map) {
    for (auto const& pair : from_map) {
      auto const& category = pair.first;
      auto n_events = pair.second;
      (*to_map)[category] += n_events;
    }
  };
  add_to(ctimes.num_events, &this->num_events);
  add_to(ctimes.extra_num_events, &this->extra_num_events);
}
CategoryTimesCount operator+(const CategoryTimesCount& left, const CategoryTimesCount& right) {
  CategoryTimesCount added;
  added._AddToCategoryTimes(left);
  added._AddToCategoryTimes(right);
  return added;
}
CategoryTimesCount& CategoryTimesCount::operator+=(const CategoryTimesCount& rhs) {
  this->_AddToCategoryTimes(rhs);
  return *this;
}


MyStatus TraceFileMeta::Init() {
  assert(!initialized);
  MyStatus status = MyStatus::OK();

  status = GetTraceID(path, &trace_id);
  IF_BAD_STATUS_RETURN(status);

//  std::unique_ptr<IEventFileParser> parser;
//  status = GetRLSEventParser(path, &parser);
//  IF_BAD_STATUS_RETURN(status);
//
//  status = parser->CountCategoryTimes(&count);
//  IF_BAD_STATUS_RETURN(status);

  std::unique_ptr<ITraceProtoReader> reader;
  status = GetTraceProtoReader(path, &reader);
  IF_BAD_STATUS_RETURN(status);
  status = reader->ReadMeta(&parser_meta);
  IF_BAD_STATUS_RETURN(status);
  status = reader->ReadTraceFileMeta(this);
  IF_BAD_STATUS_RETURN(status);

  // this->parser_meta = parser._parser_meta;
//  this->parser_meta = parser->GetParserMeta();
//  DBG_LOG("TraceFileMeta::Init() : file_type = {}, parser_meta = {}, Parser::this = {}"
//  , file_type
//  , reinterpret_cast<void*>(this->parser_meta.get())
//  , reinterpret_cast<void*>(this)
//  );

  machine = reader->get_machine();
  process = reader->get_process();
  phase = reader->get_phase();

  initialized = true;
  return MyStatus::OK();
}

MyStatus TraceFileWalker::ReadMeta(const std::string& path, TraceFileMeta* meta) {
  MyStatus status = MyStatus::OK();
  auto it = _path_to_meta.find(path);
  if (it == _path_to_meta.end()) {
    TraceFileMeta new_meta(path);
    status = new_meta.Init();
    IF_BAD_STATUS_RETURN(status);
    *meta = new_meta;
    _path_to_meta[path] = new_meta;
    _meta
      [new_meta.get_machine()]
      [new_meta.get_process()]
      [new_meta.get_phase()]
      [new_meta.get_file_type()]
      [new_meta.get_trace_id()] = new_meta;
  } else {
    *meta = it->second;
  }
  return MyStatus::OK();
}
std::set<RLSFileType> TraceFileWalker::FileTypes() const {
  std::set<RLSFileType> file_types;
  for (auto const& machine_pair : _meta) {
    for (auto const& process_pair : machine_pair.second) {
      for (auto const& phase_pair : process_pair.second) {
        for (auto const& file_type_pair : phase_pair.second) {
          file_types.insert(file_type_pair.first);
        }
      }
    }
  }
  return file_types;
}
std::list<Machine> TraceFileWalker::Machines() const {
  std::list<Machine> machines;
  for (auto const& pair : _meta) {
    machines.push_back(pair.first);
  }
  return machines;
}
std::list<Process> TraceFileWalker::Processes(const Machine& machine) const {
  std::list<Process> process;
  for (auto const& pair : _meta.at(machine)) {
    process.push_back(pair.first);
  }
  return process;
}
std::list<Phase> TraceFileWalker::Phases(const Machine& machine, const Process& process) const {
  std::list<Phase> phase;
  for (auto const& pair : _meta.at(machine).at(process)) {
    phase.push_back(pair.first);
  }
  return phase;
}

MyStatus TraceFileWalker::TraceMetas(RLSFileType file_type, const Machine& machine, const Process& process, const Phase& phase, std::vector<TraceFileMeta>* metas) {
  if (
      _meta.find(machine) == _meta.end()
      || _meta[machine].find(process) == _meta[machine].end()
      || _meta[machine][process].find(phase) == _meta[machine][process].end()
      || _meta[machine][process][phase].find(file_type) == _meta[machine][process][phase].end()
    // || _meta[machine][process][phase].find(trace_id) == _meta[machine][process][phase].end()
      ) {
    // No trace files for this machine/process/phase.
    // Return empty list.
    return MyStatus::OK();
  }

  for (const auto& pair : _meta.at(machine).at(process).at(phase).at(file_type)) {
    auto trace_id = pair.first;
    auto const& meta = pair.second;
    metas->push_back(meta);
  }

  return MyStatus::OK();
}

MyStatus TraceFileWalker::Init() {
  MyStatus status = MyStatus::OK();
  std::list<std::string> paths;
  status = FindRLSFiles(_iml_directory, &paths);
  IF_BAD_STATUS_RETURN(status);
  for (auto const& path : paths) {
    if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
      DBG_LOG("TraceFileWalker saw path={}", path);
    }
    TraceFileMeta meta;
    // Cache meta-data for trace-file.
    status = ReadMeta(path, &meta);
    IF_BAD_STATUS_RETURN(status);
  }

  if (timer) {
    std::stringstream ss;
    ss << "TraceFileWalker.Init()";
    timer->EndOperation(ss.str());
  } else {
    assert(false);
  }

  return MyStatus::OK();
}

MyStatus RawTraceParser::Init() {
  MyStatus status = MyStatus::OK();

  _has_calibration_files = (
      _cupti_overhead_json_path != ""
      && _LD_PRELOAD_overhead_json_path != ""
      && _python_clib_interception_tensorflow_json != ""
      && _python_clib_interception_simulator_json != ""
  );

  if (_has_calibration_files) {
    status = ReadJson(_cupti_overhead_json_path, &_cupti_overhead_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(_LD_PRELOAD_overhead_json_path, &_LD_PRELOAD_overhead_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(_python_annotation_json_path, &_python_annotation_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(_python_clib_interception_tensorflow_json_path, &_python_clib_interception_tensorflow_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(_python_clib_interception_simulator_json_path, &_python_clib_interception_simulator_json);
    IF_BAD_STATUS_RETURN(status);
  }

  status = _walker.Init();
  IF_BAD_STATUS_RETURN(status);

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    std::stringstream ss;
    ss << "After TraceFileWalker.Init()\n";
    _walker.Print(ss, 1);
    DBG_LOG("{}", ss.str());
  }

  return MyStatus::OK();
}

//MyStatus RawTraceParser::_ReadOneFileSequential(
//    const Machine& machine,
//    const Process& process,
//    const Phase& phase,
//    CategoryTimes *category_times,
//    EntireTraceMeta* entire_meta,
//    const std::map<RLSFileType, std::vector<TraceFileMeta>>& meta_map,
//    const std::map<RLSFileType, std::unique_ptr<IEventFileParser>>& parser_map) {
//  MyStatus status = MyStatus::OK();
//
//  for (auto rls_file_type : RLS_FILE_TYPES) {
//    auto const &metas = meta_map.at(rls_file_type);
//    for (size_t i = 1; i < metas.size(); i++) {
//      auto const &last_meta = metas[i - 1];
//      auto const &meta = metas[i];
//      if (
//          meta.parser_meta.find("start_usec") != meta.parser_meta.end()
//          && last_meta.parser_meta.find("start_usec") != last_meta.parser_meta.end()
//          ) {
//        std::set<Category> categories;
//        for (const auto &pair : meta.parser_meta["start_usec"].items()) {
//          categories.insert(pair.key());
//        }
//        for (const auto &pair : last_meta.parser_meta["start_usec"].items()) {
//          categories.insert(pair.key());
//        }
//        for (const auto &category : categories) {
//          TimeUsec last_start_us = last_meta.parser_meta["start_usec"][category];
//          const std::string &last_start_us_name = last_meta.parser_meta["start_usec_name"][category];
//          TimeUsec start_us = meta.parser_meta["start_usec"][category];
//          const std::string &start_us_name = meta.parser_meta["start_usec_name"][category];
//          if (!(last_start_us <= start_us)) {
//            std::stringstream ss;
//            ss << "Saw start_us of category=\"" << category << "\" unordered between consecutive trace-id files:\n";
//            ss << "  file 1 @ " << last_meta.get_path() << "\n";
//            ss << "    start_us = " << last_start_us << " us\n";
//            ss << "    name = " << last_start_us_name << " us\n";
//            ss << "  file 2 @ " << meta.get_path() << "\n";
//            ss << "    start_us = " << start_us << " us\n";
//            ss << "    name = " << start_us_name << " us\n";
//            DBG_LOG("{}", ss.str());
//            assert(last_start_us <= start_us);
//          }
//        }
//      }
//    }
//  }
//
//  for (auto rls_file_type : RLS_FILE_TYPES) {
//    const auto &parser = parser_map.at(rls_file_type);
//    status = parser->CountCategoryTimes(meta_map.at(rls_file_type));
//    IF_BAD_STATUS_RETURN(status);
//  }
//
//  CategoryTimesCount count;
//  for (auto const &pair : parser_map) {
//    auto const &parser = pair.second;
//    count += parser->GetCount();
//  }
//
//
//  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//    std::stringstream ss;
//    ss << "\n";
//    count.Print(ss, 1);
//    DBG_LOG("Total count: {}", ss.str());
//  }
//
//  // Preallocate space for eo_times for this (machine, process, phase).
//  *category_times = std::move(CategoryTimes(process, count));
//
//  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//    std::stringstream ss;
//    ss << "\n";
//    category_times->PrintSummary(ss, 1);
//    DBG_LOG("Preallocated: {}", ss.str());
//  }
//
//  for (auto rls_file_type : RLS_FILE_TYPES) {
//    const auto &parser = parser_map.at(rls_file_type);
//    status = parser->AppendCategoryTimes(meta_map.at(rls_file_type), category_times);
//    IF_BAD_STATUS_RETURN(status);
//  }
//
//  return MyStatus::OK();
//}

MyStatus RawTraceParser::_ReadMergeSorted(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times,
    EntireTraceMeta* entire_meta,
    const std::map<RLSFileType, std::vector<TraceFileMeta>>& meta_map,
    const std::map<RLSFileType, std::unique_ptr<IEventFileParser>>& parser_map) {
  MyStatus status = MyStatus::OK();

  for (auto rls_file_type : RLS_FILE_TYPES) {
    const auto &parser = parser_map.at(rls_file_type);
    status = parser->AppendAllCategoryTimes(*entire_meta, meta_map.at(rls_file_type), category_times);
    if (timer) {
      std::stringstream ss;
      ss << "_ReadMergeSorted(machine=" << machine << ", process=" << process << ", phase=" << phase << ", file_type=" << RLSFileTypeString(rls_file_type) << ")";
      timer->EndOperation(ss.str());
    }
    IF_BAD_STATUS_RETURN(status);
  }

  return MyStatus::OK();
}

MyStatus RawTraceParser::ReadEntireTrace(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times,
    EntireTraceMeta* entire_meta) {
  MyStatus status = MyStatus::OK();

  *entire_meta = EntireTraceMeta(machine, process, phase);

  std::map<RLSFileType, std::vector<TraceFileMeta>> meta_map;

  for (auto rls_file_type : RLS_FILE_TYPES) {
    meta_map[rls_file_type] = {};
    status = _walker.TraceMetas(rls_file_type, machine, process, phase, &meta_map[rls_file_type]);
    IF_BAD_STATUS_RETURN(status);
  }

  std::map<RLSFileType, std::unique_ptr<IEventFileParser>> parser_map;
  for (auto rls_file_type : RLS_FILE_TYPES) {
    TraceParserMeta parser_meta(machine, process, phase);
    status = GetRLSEventParserFromType(rls_file_type, parser_meta, &parser_map[rls_file_type]);
    IF_BAD_STATUS_RETURN(status);
  }

//  status = _ReadOneFileSequential(
//      machine, process, phase, category_times, entire_meta,
//      meta_map, parser_map);
//  IF_BAD_STATUS_RETURN(status);

  status = _ReadMergeSorted(
      machine, process, phase, category_times, entire_meta,
      meta_map, parser_map);
  IF_BAD_STATUS_RETURN(status);



  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    std::stringstream ss;
    ss << "\n";
    category_times->PrintSummary(ss, 1);
    DBG_LOG("After adding events: {}", ss.str());
  }

  if (timer) {
    std::stringstream ss;
    ss << "ReadEntireTrace(machine=" << machine << ", process=" << process << ", phase=" << phase << ")";
    timer->EndOperation(ss.str());
  }

  if (_has_calibration_files) {
    // Use calibration files to subtract overhead by "injecting" overhead events.
    status = _AppendOverheadEvents(
        machine,
        process,
        phase,
        category_times);
    IF_BAD_STATUS_RETURN(status);
    if (timer) {
      std::stringstream ss;
      ss << "AppendOverheadEvents(machine=" << machine << ", process=" << process << ", phase=" << phase << ")";
      timer->EndOperation(ss.str());
    }
  }

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    std::stringstream ss;
    ss << "\n";
    category_times->PrintSummary(ss, 1);
    DBG_LOG("After adding overhead events: {}", ss.str());
  }

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    std::stringstream ss;
    category_times->CheckIntegrity(ss, 0);
//    std::stringstream ss;
//    ss << "\n";
//    category_times->PrintSummary(ss, 1);
//    DBG_LOG("After adding overhead events: {}", ss.str());
  }

  return MyStatus::OK();
}

MyStatus RawTraceParser::_AppendOverheadEvents(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times) {
  MyStatus status = MyStatus::OK();
  status = _AppendOverhead_CUPTI_and_LD_PRELOAD(
      machine,
      process,
      phase,
      category_times);
  IF_BAD_STATUS_RETURN(status);

  status = _AppendOverhead_PYTHON_INTERCEPTION(
      machine,
      process,
      phase,
      category_times,
      CATEGORY_PROF_PYTHON_CLIB_INTERCEPTION_TENSORFLOW,
      {CATEGORY_TF_API},
      _python_clib_interception_tensorflow_json["mean_pyprof_interception_overhead_per_call_us"]);
  IF_BAD_STATUS_RETURN(status);
  status = _AppendOverhead_PYTHON_INTERCEPTION(
      machine,
      process,
      phase,
      category_times,
      CATEGORY_PROF_PYTHON_CLIB_INTERCEPTION_SIMULATOR,
      {CATEGORY_SIMULATOR_CPP},
      _python_clib_interception_simulator_json["mean_pyprof_interception_overhead_per_call_us"]);
  IF_BAD_STATUS_RETURN(status);

  status = _AppendOverhead_PYTHON_ANNOTATION(
      machine,
      process,
      phase,
      category_times);
  IF_BAD_STATUS_RETURN(status);
  return MyStatus::OK();
}
MyStatus RawTraceParser::_AppendOverhead_CUPTI_and_LD_PRELOAD(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times) {
  double per_LD_PRELOAD_interception_us = _LD_PRELOAD_overhead_json["mean_interception_overhead_per_call_us"];
  auto LD_PRELOAD_category_key = CategoryKey::FromCategory(process, CATEGORY_PROF_LD_PRELOAD);
  auto CUPTI_category_key = CategoryKey::FromCategory(process, CATEGORY_PROF_CUPTI);
  auto CUDA_API_category_key = CategoryKey::FromCategory(process, CATEGORY_CUDA_API_CPU);
  auto n_cuda_api_events = category_times->Count(CUDA_API_category_key);
  category_times->Preallocate(LD_PRELOAD_category_key, n_cuda_api_events);
  category_times->Preallocate(CUPTI_category_key, n_cuda_api_events);
  auto& CUDA_API_events = category_times->MutableEvents(CUDA_API_category_key);
  auto& LD_PRELOAD_events = category_times->MutableEvents(LD_PRELOAD_category_key);
  auto& CUPTI_events = category_times->MutableEvents(CUPTI_category_key);

  // cuda_api_id => mean_time_us
  std::map<EventNameID, double> cupti_overhead_ps;
  std::map<EventNameID, size_t> missing_cupti_overhead_cuda_api_calls;
  for (auto const& pair : _cupti_overhead_json.items()) {
    auto const& cuda_api_name = pair.key();
    // NOTE: all the CUDA API calls we see in the json file SHOULD exist in the trace-file...
    // If they don't, add them as a key, or just skip them.
    if (!CUDA_API_events.HasEventID(cuda_api_name)) {
      continue;
    }
    auto cuda_api_id = CUDA_API_events.AsEventID(cuda_api_name);
    auto const& js = pair.value();
    double mean_cupti_overhead_per_call_us = js["mean_cupti_overhead_per_call_us"];
    cupti_overhead_ps[cuda_api_id] = mean_cupti_overhead_per_call_us * PSEC_IN_USEC;
  }

  for (size_t i = 0; i < CUDA_API_events.size(); i++) {
    // CATEGORY_PROF_CUPTI
    {
      auto id = CUDA_API_events.GetEventNameID(i);
      auto it = cupti_overhead_ps.find(id);
      if (it == cupti_overhead_ps.end()) {
        missing_cupti_overhead_cuda_api_calls[id] += 1;
      } else {
        auto mean_cupti_overhead_ps = it->second;
        TimePsec start_ps = CUDA_API_events.StartPsec(i);
        TimePsec end_ps = start_ps + mean_cupti_overhead_ps;
        OptionalString name;
        if (CUPTI_events.KeepNames()) {
          name = CUDA_API_events.GetEventName(i);
        }
        CUPTI_events.AppendEventPsec(name, start_ps, end_ps);
      }
    }
    // CATEGORY_PROF_LD_PRELOAD
    {
//      TimePsec cuda_api_duration_us = CUDA_API_events.DurationUsec(i);
//      if (cuda_api_duration_us <= per_LD_PRELOAD_interception_us) {
//        SPDLOG_WARN("CUDA_API_events[{}, {}]:  cuda_api_duration_us ({} us) > ({} us) per_LD_PRELOAD_interception_us",
//            i, CUDA_API_events.GetEventName(i),
//            cuda_api_duration_us,
//            per_LD_PRELOAD_interception_us);
//      }
      TimePsec start_ps = CUDA_API_events.EndPsec(i);
      TimePsec end_ps = start_ps + (per_LD_PRELOAD_interception_us * PSEC_IN_USEC);
      OptionalString name;
      if (LD_PRELOAD_events.KeepNames()) {
        name = CUDA_API_events.GetEventName(i);
      }
      LD_PRELOAD_events.AppendEventPsec(name, start_ps, end_ps);
    }
  }

  if (missing_cupti_overhead_cuda_api_calls.size() > 0) {
    std::stringstream ss;
//    ss << "Saw CUDA API calls that we didn't have calibrated CUPTI overheads for overheads for in " << meta.path << ":\n";
    ss << "WARNING: Saw CUDA API calls that we didn't have calibrated CUPTI overheads for overheads for:\n";
    PrintIndent(ss, 1);
    for (auto const& pair : missing_cupti_overhead_cuda_api_calls) {
      auto id = pair.first;
      auto n_events = pair.second;
      auto cuda_api_name = CUDA_API_events.AsEventName(id);
      ss << "  cuda_api=" << cuda_api_name << ": " << n_events << " events\n";
    }
    ss << "\n";
    std::cout << ss.str();
  }
  return MyStatus::OK();
}

MyStatus RawTraceParser::_AppendOverhead_PYTHON_INTERCEPTION(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times,
    const Category& category_prof,
    const std::set<Category>& c_events_categories,
    double per_python_clib_interception_us) {
  auto category_key = CategoryKey::FromCategory(process, category_prof);
  size_t n_events = 0;
  for (auto const& category : c_events_categories) {
    n_events += category_times->Count(CategoryKey::FromCategory(process, category));
  }
  // DBG_LOG("_AppendOverhead_PYTHON_INTERCEPTION: n_events = {}", n_events);
  category_times->Preallocate(category_key, n_events);
  auto& prof_events = category_times->eo_times.at(category_key);
  std::vector<EOEvents> clib_eo_events;
  for (auto const& category : c_events_categories) {
    auto it = category_times->eo_times.find(CategoryKey::FromCategory(process, category));
    if (it != category_times->eo_times.end()) {
      clib_eo_events.push_back(it->second);
    }
  }
  auto func = [per_python_clib_interception_us, &prof_events] (const EOEvents& events, size_t i) {
    TimePsec start_ps = events.EndPsec(i) + (per_python_clib_interception_us * PSEC_IN_USEC);
    TimePsec end_ps = start_ps + (per_python_clib_interception_us * PSEC_IN_USEC);
    OptionalString name;
    if (prof_events.KeepNames()) {
      name = events.GetEventName(i);
    }
    prof_events.AppendEventPsec(name, start_ps, end_ps);
  };
  auto key_func = [] (const EOEvents& events, size_t i) -> TimeUsec {
    return events.StartUsec(i);
  };
  EachMerged<EOEvents, TimeUsec>(clib_eo_events, func, key_func);
  return MyStatus::OK();
}

MyStatus RawTraceParser::_AppendOverhead_PYTHON_ANNOTATION(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times) {
  double per_pyprof_annotation_overhead_us = _python_annotation_json["mean_pyprof_annotation_overhead_per_call_us"];
  auto category_key = CategoryKey::FromCategory(process, CATEGORY_PROF_PYTHON_ANNOTATION);
  auto ops_key = CategoryKey::FromCategory(process, CATEGORY_EXTRA_OPERATION);
  auto const& ops_events = category_times->MutableEventsExtra(ops_key);
  assert(ops_events.size() > 0);
  category_times->Preallocate(category_key, ops_events.size());
  auto& prof_events = category_times->MutableEvents(category_key);
  for (size_t i = 0; i < ops_events.size(); i++) {
    TimePsec start_ps = ops_events.StartPsec(i);
    TimePsec end_ps = start_ps + (per_pyprof_annotation_overhead_us * PSEC_IN_USEC);
    OptionalString name;
    if (prof_events.KeepNames()) {
      name = ops_events.GetEventName(i);
    }
    // PROBLEM: we only want to keep names of events if we HAVE to.
    // Solution: optionally provide name.  If
    prof_events.AppendEventPsec(name, start_ps, end_ps);
  }
  return MyStatus::OK();
}

MyStatus GetTraceID(const std::string& path, TraceID* trace_id) {
  RLSFileType file_type = GetRLSFileType(path);
  if (file_type == RLSFileType::UNKNOWN_FILE) {
    std::stringstream ss;
    ss << "Couldn't find trace_id in path=" << path;
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  std::smatch match;
  boost::filesystem::path bpath(path);
  std::regex re(TRACE_SUFFIX_RE);
  if (!std::regex_search(bpath.filename().string(), match, re)) {
    std::stringstream ss;
    ss << "Couldn't find trace_id in path=" << path << " (it doesn't match *.trace_id_\\d+)";
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  const auto& trace_str = match.str(1);
  auto parsed_trace_id =  StringToNumber<TraceID>(trace_str);
  *trace_id = parsed_trace_id;
  return MyStatus::OK();
}

void CategoryKeyBitset::Print(std::ostream& out, int indent) const {
 // auto keys = idx_map.KeySetFrom(bitset);
  auto keys = idx_map->KeySetFrom(bitset);

  // Print a single CategoryKey all merged together.
  auto merged_key = AsCategoryKey();
  merged_key.Print(out, indent);

  // Print each CategoryKey individually.
//  PrintIndent(out, indent);
//  out << "CategoryKeyBitset: bits = " << bitset.to_string() << ", size = " << keys.size();
//  for (const auto& key : keys) {
//    out << "\n";
//    key.Print(out, indent + 1);
//  }

}

std::set<CategoryKey> CategoryKeyBitset::Keys() const {
  auto keys = idx_map->KeySetFrom(bitset);
  return keys;
}

std::set<size_t> CategoryKeyBitset::Indices() const {
  auto indices = idx_map->IndexSetFrom(bitset);
  return indices;
}


CategoryKeyBitset CategoryKeyBitset::EmptySet(std::shared_ptr<const CategoryIdxMap> idx_map) {
  CategoryKeyBitset ctimes(idx_map);
  return ctimes;
}

#define IS_START_IDX(i) (i % 2 == 0)
#define IS_END_IDX(i) (i % 2 == 1)
OverlapResult OverlapComputer::ComputeOverlap(bool keep_empty_time) const {
  OverlapResult r;
  r.idx_map = ctimes.idx_map;

  // NOTE: eigen uses int for indexing rows/columns, not size_t...
  // https://stackoverflow.com/questions/33993918/eigenmatrix-why-does-eigen-expect-an-int-and-not-size-t
  // using IndexType = size_t;
  using IndexType = int;

  using IdxArray = Array<size_t, Dynamic, 1>;
  size_t k = ctimes.size();
  IdxArray index = IdxArray::Zero(k);

  // std::vector<EOEvents> times;
  std::vector<const TimeUsec*> times;
  times.reserve(k);
  for (const auto& pair : ctimes.eo_times) {
    const auto& eo_events = pair.second;
    times.push_back(eo_events.RawPtr());
  }

  IdxArray lengths = IdxArray(k);
  {
    IndexType i = 0;
    for (const auto& pair : ctimes.eo_times) {
      lengths(i) = 2 * pair.second.size();
      i += 1;
    }
  }

  TimeUsec min_time_value = std::numeric_limits<TimeUsec>::min();
  TimeUsec last_time = min_time_value;
  CategoryKeyBitset cur_cat = CategoryKeyBitset::EmptySet(ctimes.idx_map);
  auto const& all_ops_set = CategoryKeyBitset::Ops(cur_cat);

  while ((index < lengths).any()) {
    // Find the non-empty category with the next minimum start/end time.
    IndexType min_cat = 0;
    TimeUsec min_time = std::numeric_limits<TimeUsec>::max();
    for (IndexType i = 0; i < index.size(); i++) {
      // Check we haven't exhausted the intervals in the category.
      if (index(i) < lengths(i)) {
        // Non-empty category.
        if (times[i][index(i)] < min_time) {
          if (debug && SHOULD_DEBUG(FEATURE_OVERLAP)) {
            std::stringstream ss;
            auto left = times[i][index(i)];
            auto right = min_time;
            ss << "\n";
            PrintIndent(ss, 1);
            ss << "(" << left << ") times[i][index[i]] <= min_time (" << right  << ")" << "\n";
            PrintIndent(ss, 2);
            ss << "min_cat = " << i << "\n";
            ss << "index = " << index << "\n";
            ss << "lengths = " << lengths;
            DBG_LOG("{}", ss.str());
          }
          min_cat = i;
          min_time = times[i][index(i)];
        }
      }
    }

//    // Verbose: print entire category key.
//    if (debug && SHOULD_DEBUG(FEATURE_OVERLAP)) {
//      std::stringstream ss;
//
//      CategoryKeyBitset min_cat_key(min_cat, ctimes.idx_map);
//
//      ss << "\n";
//      ss << "min_cat:";
//      ss << "\n";
//      min_cat_key.Print(ss, 1);
//
//      ss << "\n";
//      ss << "min_time = " << min_time;
//
//      ss << "\n";
//      ss << "cur_cat:";
//      ss << "\n";
//      cur_cat.Print(ss, 1);
//
//      ss << "\n";
//      if (index(min_cat) % 2 == 0) {
//        ss << "time_type = start";
//      } else {
//        ss << "time_type = end";
//      }
//
//      DBG_LOG("{}", ss.str());
//    }

    // Less verbose: just print category index.
    // start {i} @ {time} => {new_set}
    if (debug && SHOULD_DEBUG(FEATURE_OVERLAP)) {
      std::stringstream ss;
      ss << "\n";
      PrintIndent(ss, 1);
      if (index(min_cat) % 2 == 0) {
        ss << "start";
      } else {
        ss << "end  ";
      }
      ss << " " << min_cat << " @ " << min_time;

      ss << " => ";
      // CategoryKeyBitset min_cat_key(min_cat, ctimes.idx_map);
      auto indices = cur_cat.Indices();
      PrintValue(ss, indices);

      // Print Event(start_us, end_us)
      TimeUsec start_us;
      TimeUsec end_us;
      if (IS_START_IDX(index(min_cat))) {
        start_us = times[min_cat][index(min_cat)];
        end_us = times[min_cat][index(min_cat) + 1];
      } else {
        start_us = times[min_cat][index(min_cat) - 1];
        end_us = times[min_cat][index(min_cat)];
      }
      ss << "\n";
      PrintIndent(ss, 1);
      ss << "Event(" << start_us << ", " << end_us << ")";

      DBG_LOG("{}", ss.str());
    }


    if ((index(min_cat) % 2) == 0 and min_time == times[min_cat][index(min_cat)+1]) {
      index(min_cat) += 2;
      continue;
    }

    auto time_chunk = min_time - last_time;
    bool non_zero_time = time_chunk > 0;
    if (non_zero_time) {
      // NOTE: std::map<Key, Number> defaults to 0 if the key doesn't exist.
      r.overlap[cur_cat] += time_chunk;
    }

    // Update current list of active categories.
    bool is_start = (index(min_cat) % 2 == 0);
    if (is_start) {
      if (non_zero_time) {
        TimeUsec start_time_usec = last_time;
        TimeUsec end_time_usec = min_time;
        r.meta.AddEvent(cur_cat, start_time_usec, end_time_usec);
      }
      cur_cat.Add(min_cat);
    } else {
      if (non_zero_time) {
        TimeUsec start_time_usec = last_time;
        TimeUsec end_time_usec = min_time;
        if (debug && SHOULD_DEBUG(FEATURE_OVERLAP_META)) {
          std::stringstream ss;

          ss << "meta.AddRegion";

          const auto& indices = cur_cat.Indices();
          ss << "\n";
          PrintIndent(ss, 2);
          ss << "cat = ";
          PrintValue(ss, indices);

          ss << "\n";
          PrintIndent(ss, 2);
          ss << "start_us = " << start_time_usec << " us";

          ss << "\n";
          PrintIndent(ss, 2);
          ss << "end_us = " << end_time_usec << " us";

          DBG_LOG("{}", ss.str());
        }
        r.meta.AddEvent(cur_cat, start_time_usec, end_time_usec);
      }

      cur_cat.Remove(min_cat);
    }

    if (SHOULD_DEBUG(FEATURE_JUST_OPERATIONS)) {
      auto const& ops = all_ops_set.Intersection(cur_cat);
      if (!(ops.size() <= 1)) {
        std::stringstream ss;
        ss << "Saw " << ops.size() << " ops, but expected <= 1";
        ss << "\n";
        ops.Print(ss, 1);
        DBG_LOG("{}", ss.str());
        assert(ops.size() <= 1);
      }
    }

    // Q: Can we use this information to merge overlap results from different machines...?
//    // Can have multiple categories entering and leaving, so just make sure we keep things correct
//    if (last_time == min_time) {
//      // Start of new interval which is the same as the previous interval.
//      // output_cats[cur_output-1, min_cat] = is_start;
//      output_cats[cur_output-1] = bitset_add(output_cats[cur_output-1], min_cat);
//    } else {
//      // Normal case:
//      // Insert event if there is a change from last time
//      outputs[cur_output] = min_time;
//      // output_cats[cur_output, :] = cur_cat;
//      output_cats[cur_output] = cur_cat;
//      cur_output += 1;
//      // last_time = min_time;
//    }

    last_time = min_time;
    index(min_cat) += 1;

  }

  if (!keep_empty_time) {
    // Delete empty keys; these result from blank space between events. e.g.
    //   frozenset(): 1000000
    assert(r.overlap.size() == r.meta.size());
    std::set<CategoryKeyBitset> del_keys;
    for (const auto& pair : r.overlap) {
      const auto& bitset = pair.first;
      if (bitset.IsEmpty()) {
        del_keys.insert(bitset);
      }
    }
    for (const auto& bitset : del_keys) {
      r.overlap.erase(bitset);
    }
    for (const auto& bitset : del_keys) {
      r.meta.regions.erase(bitset);
    }
  }

  r.ConvertPsecToUsec();

  return r;
}
#undef IS_START_IDX
#undef IS_END_IDX

std::unique_ptr<OverlapTypeReducerInterface> GetJSDumper(const OverlapType& overlap_type) {
  std::unique_ptr<OverlapTypeReducerInterface> ret;
  if (overlap_type == "ResourceOverlap") {
    ret = std::make_unique<ResourceOverlapType>();
  } else if (overlap_type == "CategoryOverlap") {
    ret = std::make_unique<CategoryOverlapType>();
  } else if (overlap_type == "OperationOverlap") {
    ret = std::make_unique<OperationOverlapType>();
  } else if (overlap_type == "ResourceSubplot") {
    ret = std::make_unique<ResourceSubplotOverlapType>();
  } else if (overlap_type == "ProfilingOverhead") {
    ret = std::make_unique<ProfilingOverheadOverlapType>();
  } else {
    // Not sure what overlap_type is.
    assert(false);
    std::stringstream ss;
    ss << "Not sure how to handle overlap_type = " << overlap_type;
    RAISE_NOT_IMPLEMENTED(ss.str());
  }
  return ret;
}


void OverlapResult::DumpVennJS(const std::string& directory,
                               const Machine& machine,
                               const Process& process,
                               const Phase& phase) const {
  for (auto const& overlap_type : OVERLAP_TYPES) {

    auto const& overlap_reducer = OverlapResultReducer::ReduceToCategoryKey(*this);
    if (SHOULD_DEBUG(FEATURE_SAVE_JS)) {
      std::stringstream ss;
      ss << "ReduceToCategoryKey overlap_reducer: overlap_type = " << overlap_type;
      ss << "\n";
      overlap_reducer.Print(ss, 1);
      DBG_LOG("{}", ss.str());
    }

    auto js_dumper = GetJSDumper(overlap_type);
    auto const& reduced_overlap = js_dumper->PostReduceCategoryKey(overlap_reducer);
    if (SHOULD_DEBUG(FEATURE_SAVE_JS)) {
      std::stringstream ss;
      ss << "PostReduceCategoryKey reduced_overlap: overlap_type = " << overlap_type;
      ss << "\n";
      reduced_overlap.Print(ss, 1);
      DBG_LOG("{}", ss.str());
    }

    js_dumper->DumpOverlapJS(
        directory,
        machine,
        process,
        phase,
        reduced_overlap);

  }
}

void OverlapResult::Print(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "OverlapResult:";

  out << "\n";
  PrintIndent(out, indent + 1);
  out << "Overlap: size = " << overlap.size();
  size_t i = 0;
  for (const auto& pair : overlap) {
    auto const& bitset = pair.first;
    auto time_us = pair.second;
    double time_sec = ((double)time_us) / ((double)USEC_IN_SEC);

    out << "\n";
    PrintIndent(out, indent + 2);
    out << "Overlap[" << i << "]: duration = " << time_sec << " sec";
    out << "\n";
    bitset.Print(out, indent + 3);
    i += 1;
  }

  out << "\n";
  meta.Print(out, indent + 1);

}

//DEFINE_PRINT_OPERATOR(CategoryKey)
DEFINE_PRINT_OPERATOR(CategoryKeyBitset)
DEFINE_PRINT_OPERATOR(EOEvents)
DEFINE_PRINT_OPERATOR(CategoryTimesCount)
DEFINE_PRINT_OPERATOR(CategoryTimes)
DEFINE_PRINT_OPERATOR(CategoryTimesBitset)
DEFINE_PRINT_OPERATOR(OverlapResult)

DEFINE_PRINT_DEBUG(CategoryKey)
DEFINE_PRINT_DEBUG(CategoryKeyBitset)
DEFINE_PRINT_DEBUG(EOEvents)
DEFINE_PRINT_DEBUG(CategoryTimesCount)
DEFINE_PRINT_DEBUG(CategoryTimes)
DEFINE_PRINT_DEBUG(CategoryTimesBitset)
DEFINE_PRINT_DEBUG(OverlapResult)

} // namespace tensorflow

