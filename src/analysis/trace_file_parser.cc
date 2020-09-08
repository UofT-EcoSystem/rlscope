//
// Created by jagle on 11/13/2019.
//

#include "analysis/trace_file_parser.h"
//#include "cuda_api_profiler/generic_logging.h"
#include "common_util.h"
#include "trace_file_parser.h"

#include <algorithm>

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
//#include <sys/types.h>
// Must be included in order operator<< to work with spd logging.
// https://github.com/gabime/spdlog#user-defined-types
#include "spdlog/fmt/ostr.h"
#include <limits>
#include <regex>

using namespace Eigen;

#define CSV_DELIM_STR ","
#define CSV_ESCAPE_CHAR '\\'
#define CSV_QUOTE_CHAR '"'

namespace rlscope {

const std::regex PROCESS_OPERATION_REGEX = std::regex(R"(\[.*\])");

const std::set<RLSFileType> RLS_FILE_TYPES = {
    CUDA_API_STATS_FILE,
    CATEGORY_EVENTS_FILE,
    CUDA_DEVICE_EVENTS_FILE,
    NVPROF_GPU_TRACE_CSV_FILE,
    NVPROF_API_TRACE_CSV_FILE,
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



const std::string& EOEvent::name() const {
  return _eo_events->GetEventName(_i);
}

TimeUsec EOEvent::start_time_us() const {
  return _eo_events->StartUsec(_i);
}

TimeUsec EOEvent::end_time_us() const {
  return _eo_events->EndUsec(_i);
}

TimeUsec EOEvent::duration_us() const {
  return _eo_events->DurationUsec(_i);
}

const IEventMetadata* EOEvent::metadata() const {
  return _eo_events->GetEventMetadata(_i);
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

EOEvents EOEvents::Preallocate(const CategoryKey& category_key, size_t n_events, bool keep_event_metadata) {
  bool keep_names = CategoryShouldKeepNames(category_key);
  EOEvents eo_events(n_events, keep_names, keep_event_metadata);
  return eo_events;
}

EOEvents EOEvents::PreallocateEvents(const CategoryKey& category_key, size_t n_events, bool keep_names, bool keep_event_metadata) {
  EOEvents eo_events(n_events, keep_names, keep_event_metadata);
  return eo_events;
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
  auto const& process = entire_meta.process;
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

void CategoryTimes::MoveInto(CategoryTimes& category_times) {
  for (auto& pair : category_times.eo_times) {
    const auto& category_key = pair.first;
    EOEvents& eo_events = pair.second;
    if (eo_times.find(category_key) == eo_times.end()) {
      MoveEventsInto(category_key, eo_events);
    } else {
      MergeEventsInto(category_key, eo_events);
      assert(false);
    }
  }
}
void CategoryTimes::MoveEventsInto(const CategoryKey& category_key, EOEvents& eo_events) {
  assert(eo_times.find(category_key) == eo_times.end());
  eo_times[category_key] = std::move(eo_events);
}

void CategoryTimes::MergeEventsInto(const CategoryKey& category_key, EOEvents& eo_events) {
  assert(eo_times.find(category_key) != eo_times.end());
  eo_times[category_key] = EOEvents::Merge(eo_times[category_key], eo_events);
}
void CategoryTimes::_Preallocate(EOTimes* eo_times, const CategoryKey& category_key, size_t n_events) {
  bool keep_names = CategoryShouldKeepNames(category_key);
  (*eo_times)[category_key] = EOEvents(n_events, keep_names, /*keep_event_metadata=*/false);
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
      (*eo_times)[category_key] = EOEvents(n_events, keep_names, false);
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
    case RLSFileType::NVPROF_GPU_TRACE_CSV_FILE:
      return matches_regex(NVPROF_GPU_TRACE_CSV_REGEX);
      break;
    case RLSFileType::NVPROF_API_TRACE_CSV_FILE:
      return matches_regex(NVPROF_API_TRACE_CSV_REGEX);
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
  // TODO: Read the header line of the nvprof csv file to determine the RLS file type?
  if (matches_regex(NVPROF_GPU_TRACE_CSV_REGEX)) {
    return RLSFileType::NVPROF_GPU_TRACE_CSV_FILE;
  }
  if (matches_regex(NVPROF_API_TRACE_CSV_REGEX)) {
    return RLSFileType::NVPROF_API_TRACE_CSV_FILE;
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
    case NVPROF_GPU_TRACE_CSV_FILE:
      return "NVPROF_GPU_TRACE_CSV_FILE";
    case NVPROF_API_TRACE_CSV_FILE:
      return "NVPROF_API_TRACE_CSV_FILE";
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

MyStatus GetRLSEventParser(const RLSAnalyzeArgs& args, const std::string& path, TraceParserMeta parser_meta, std::unique_ptr<IEventFileParser>* parser, const EntireTraceSelector& selector) {
  auto file_type = GetRLSFileType(path);
  auto status = GetRLSEventParserFromType(args, file_type, parser_meta, parser, selector);
  return status;
}

MyStatus GetRLSEventParserFromType(const RLSAnalyzeArgs& args, RLSFileType file_type, TraceParserMeta parser_meta, std::unique_ptr<IEventFileParser>* parser, const EntireTraceSelector& selector) {
  switch (file_type) {
    case RLSFileType::CUDA_API_STATS_FILE:
      parser->reset(new CUDAAPIStatsParser(std::move(parser_meta)));
      break;
    case RLSFileType::CATEGORY_EVENTS_FILE:
      parser->reset(new CategoryEventsParser(std::move(parser_meta)));
      break;
    case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
      parser->reset(new CUDADeviceEventsParser(
          std::move(parser_meta),
          selector));
      break;
    case RLSFileType::NVPROF_API_TRACE_CSV_FILE:
    case RLSFileType::NVPROF_GPU_TRACE_CSV_FILE:
      parser->reset(new NvprofCSVParser(args, std::move(parser_meta), file_type));
      break;
    default:
      assert(false);
  }
  return MyStatus::OK();
}

MyStatus GetTraceFileReader(const RLSAnalyzeArgs& args, const std::string& path, std::unique_ptr<ITraceFileReader>* reader) {
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
    case RLSFileType::NVPROF_GPU_TRACE_CSV_FILE:
    case RLSFileType::NVPROF_API_TRACE_CSV_FILE:
      reader->reset(new NvprofTraceFileReader(args, path, file_type));
      break;
    default:
      reader->reset(nullptr);
      return MyStatus(error::NOT_FOUND, "Not a protobuf parser");
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
//  status = GetRLSEventParser(args, path, &parser);
//  IF_BAD_STATUS_RETURN(status);
//
//  status = parser->CountCategoryTimes(&count);
//  IF_BAD_STATUS_RETURN(status);

  std::unique_ptr<ITraceFileReader> reader;
  status = GetTraceFileReader(args, path, &reader);
  if (status.code() == MyStatus::OK().code()) {
    IF_BAD_STATUS_RETURN(status);
    status = reader->ReadMeta(&parser_meta);
    IF_BAD_STATUS_RETURN(status);
    status = reader->ReadTraceFileMeta(this);
    IF_BAD_STATUS_RETURN(status);

    machine = reader->get_machine();
    process = reader->get_process();
    phase = reader->get_phase();
  } else if (status.code() == error::NOT_FOUND) {
    // Fallback behaviour: it's not a recognized proto file, instead just use the
    // basename of the file for machine/process/phase name.
    boost::filesystem::path bpath = path;
    auto basename = bpath.filename().string();
    machine = basename;
    process = basename;
    phase = basename;
  } else {
    return status;
  }

  // this->parser_meta = parser._parser_meta;
//  this->parser_meta = parser->GetParserMeta();
//  DBG_LOG("TraceFileMeta::Init() : file_type = {}, parser_meta = {}, Parser::this = {}"
//  , file_type
//  , reinterpret_cast<void*>(this->parser_meta.get())
//  , reinterpret_cast<void*>(this)
//  );

  initialized = true;
  return MyStatus::OK();
}

MyStatus TraceFileWalker::ReadMeta(const std::string& path, TraceFileMeta* meta) {
  MyStatus status = MyStatus::OK();
  auto it = _path_to_meta.find(path);
  if (it == _path_to_meta.end()) {
    TraceFileMeta new_meta(args, path);
    status = new_meta.Init();
    IF_BAD_STATUS_RETURN(status);
    if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
      DBG_LOG("Add {}", new_meta);
    }
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
      args.FLAGS_cupti_overhead_json.has_value()
      && args.FLAGS_LD_PRELOAD_overhead_json.has_value()
      && args.FLAGS_python_clib_interception_tensorflow_json.has_value()
      && args.FLAGS_python_clib_interception_simulator_json.has_value()
  );

  if (_has_calibration_files) {
    status = ReadJson(args.FLAGS_cupti_overhead_json.value(), &_cupti_overhead_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(args.FLAGS_LD_PRELOAD_overhead_json.value(), &_LD_PRELOAD_overhead_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(args.FLAGS_python_annotation_json.value(), &_python_annotation_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(args.FLAGS_python_clib_interception_tensorflow_json.value(), &_python_clib_interception_tensorflow_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(args.FLAGS_python_clib_interception_simulator_json.value(), &_python_clib_interception_simulator_json);
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

using EachReadMergeSortedFunc = std::function<MyStatus(RLSFileType rls_file_type)>;
MyStatus RawTraceParser::_ReadMergeSorted(
    const std::set<RLSFileType>& file_types,
    CategoryTimes *category_times,
    EntireTraceMeta* entire_meta,
    const std::map<RLSFileType, std::vector<TraceFileMeta>>& meta_map,
    const std::map<RLSFileType, std::unique_ptr<IEventFileParser>>& parser_map,
    EachReadMergeSortedFunc func) {
  MyStatus status = MyStatus::OK();

  for (auto rls_file_type : file_types) {
    const auto &parser = parser_map.at(rls_file_type);
    const auto& metas = meta_map.at(rls_file_type);
    status = parser->AppendAllCategoryTimes(*entire_meta, meta_map.at(rls_file_type), category_times);
    IF_BAD_STATUS_RETURN(status);
    status = func(rls_file_type, metas);
    IF_BAD_STATUS_RETURN(status);
  }

  return MyStatus::OK();
}

MyStatus RawTraceParser::ReadEntireTrace(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    const std::set<RLSFileType>& file_types,
    CategoryTimes *category_times,
    EntireTraceMeta* entire_meta,
    const EntireTraceSelector& selector) {
  MyStatus status = MyStatus::OK();

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    DBG_LOG("ReadEntireTrace(machine={}, process={}, phase={})", machine, process, phase);
  }

  *entire_meta = EntireTraceMeta(machine, process, phase);

  std::map<RLSFileType, std::vector<TraceFileMeta>> meta_map;

  for (auto rls_file_type : file_types) {
    meta_map[rls_file_type] = {};
    status = _walker.TraceMetas(rls_file_type, machine, process, phase, &meta_map[rls_file_type]);
    IF_BAD_STATUS_RETURN(status);
  }

  std::map<RLSFileType, std::unique_ptr<IEventFileParser>> parser_map;
  for (auto rls_file_type : file_types) {
    TraceParserMeta parser_meta(machine, process, phase);
    status = GetRLSEventParserFromType(args, rls_file_type, parser_meta, &parser_map[rls_file_type], selector);
    IF_BAD_STATUS_RETURN(status);
  }

//  status = _ReadOneFileSequential(
//      machine, process, phase, category_times, entire_meta,
//      meta_map, parser_map);
//  IF_BAD_STATUS_RETURN(status);

  status = _ReadMergeSorted(
      file_types,
      category_times, entire_meta,
      meta_map, parser_map, [this, &machine, &process, &phase] (RLSFileType rls_file_type, const auto& metas) {
        if (timer && metas.size() > 0) {
          std::stringstream ss;
          ss << "_ReadMergeSorted(machine=" << machine << ", process=" << process << ", phase=" << phase << ", file_type=" << RLSFileTypeString(rls_file_type) << ")";
          timer->EndOperation(ss.str());
        }
        return MyStatus::OK();
      });
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
    {
      std::stringstream ss;
      category_times->CheckIntegrity(ss, 0);
    }
//    std::stringstream ss;
//    ss << "\n";
//    category_times->PrintSummary(ss, 1);
//    DBG_LOG("After adding overhead events: {}", ss.str());
    if (timer) {
      std::stringstream ss;
      ss << "CheckIntegrity(machine=" << machine << ", process=" << process << ", phase=" << phase << ")";
      timer->EndOperation(ss.str());
    }
  }

  return MyStatus::OK();
}

MyStatus RawTraceParser::CrossProcessReadEntireTrace(
    const std::set<RLSFileType>& file_types,
    CategoryTimes *category_times,
    EntireTraceMeta* entire_meta,
    const EntireTraceSelector& selector) {
  auto status = MyStatus::OK();

  auto const& machines = this->Machines();
  for (auto const& machine : machines) {
    entire_meta->machines.insert(machine);
    auto const& processes = this->Processes(machine);
    for (auto const& process : processes) {
      entire_meta->processes.insert(process);
      auto const& phases = this->Phases(machine, process);
      for (auto const &phase : phases) {
        entire_meta->phases.insert(phase);
        EntireTraceMeta meta;
        std::unique_ptr<CategoryTimes> partial_category_times(new CategoryTimes());
        status = this->ReadEntireTrace(
            machine, process, phase,
            file_types,
            partial_category_times.get(), &meta, selector);
        IF_BAD_STATUS_RETURN(status);
        // NOTE: this is inefficient IF category_times keys overlapped...but they should since we should be reading from separate processes?
        // If we have all the files we can preallocate enough space for each CategoryKey(proc, non_op) ahead of time.
        // We just need to make it so the EventParser can read from multiple proc's at once
        // (currently assumes 1 proc at at time when creating CategoryKey for proto files...)
        category_times->MoveInto(*partial_category_times);
        partial_category_times.reset(nullptr);
        if (timer) {
          std::stringstream ss;
          ss << "category_times.MoveInto(machine=" << machine << ", process=" << process << ", phase=" << phase << ")";
          timer->EndOperation(ss.str());
        }
      }
    }
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

// NOTE: sometimes calibration can end up with negative durations for overhead events when overhead is negligible.
// In that case, just using a duration of zero for the overhead event.
TimePsec as_overhead_duration_ps(TimePsec overhead_duration_ps) {
  return std::max(static_cast<TimePsec>(0), overhead_duration_ps);
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
        TimePsec end_ps = start_ps + as_overhead_duration_ps(mean_cupti_overhead_ps);
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
    TimePsec start_ps = events.EndPsec(i) + as_overhead_duration_ps((per_python_clib_interception_us * PSEC_IN_USEC));
    TimePsec end_ps = start_ps + as_overhead_duration_ps((per_python_clib_interception_us * PSEC_IN_USEC));
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
    TimePsec end_ps = start_ps + as_overhead_duration_ps((per_pyprof_annotation_overhead_us * PSEC_IN_USEC));
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

  if (file_type == NVPROF_API_TRACE_CSV_FILE || file_type == NVPROF_GPU_TRACE_CSV_FILE) {
    // No trace_id's for nvprof csv files.
    // There's only one .nvprof file for each process.
    // Default to 0.
    *trace_id = 0;
    return MyStatus::OK();
  }

  if (file_type == RLSFileType::UNKNOWN_FILE) {
    std::stringstream ss;
    ss << "Couldn't find trace_id in path=" << path;
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  std::smatch match;
  boost::filesystem::path bpath(path);
  std::regex re(TRACE_SUFFIX_RE);
  std::string filename = bpath.filename().string();
  if (!std::regex_search(filename, match, re)) {
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

//  // Print a single CategoryKey all merged together.
//  auto merged_key = AsCategoryKey();
//  merged_key.Print(out, indent);

  // Print each CategoryKey individually.
  PrintIndent(out, indent);
  out << "CategoryKeyBitset: set_id = " << SetID() << ", bits = " << bitset.to_string() << ", size = " << keys.size();
  for (const auto& key : keys) {
    out << "\n";
    key.Print(out, indent + 1);
  }

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

MyStatus OverlapInterval::EachEvent(const OverlapResult& result, EachEventFunc func) const {
  auto status = MyStatus::OK();
  // Iterate over the non-zero categories, extract EOEvent's.

  using IndexType = int;
  OverlapInterval::IdxArray index = start_events;

  assert(start_events.size() == end_events.size());
  assert(result.ctimes.eo_times.size() == static_cast<size_t>(start_events.size()));
  for (auto const& pair : result.ctimes.eo_times) {
    auto const& category_key = pair.first;
    auto const& eo_events = pair.second;
    const auto i = result.idx_map->Idx(category_key);
    auto end_events_size = end_events.size();
    assert(i <= static_cast<size_t>(end_events_size));
    auto end_events_i = end_events(i);
    assert(end_events_i <= eo_events.size());
  }

  for (IndexType i = 0; i < index.size(); i++) {
    const auto& category_key = result.idx_map->Key(i);
    if (!result.ctimes.Count(category_key)) {
      continue;
    }
    const auto& eo_events = result.ctimes.MaybeEvents(category_key).value();

    auto eo_events_size = eo_events.size();
    auto end_events_i = end_events(i);
    // assert(end_events_i <= eo_events_size); // FAIL
    if (!(end_events_i <= eo_events_size)) {
      DBG_LOG("FAIL: end_events_i ({}) <= ({}) eo_events_size\n{}",
              end_events_i, eo_events_size,
              *this);
      assert(end_events_i <= eo_events_size); // FAIL
    }

    while (index(i) < end_events(i)) {

      auto idx = index(i);
      assert(idx < eo_events_size);

      auto event_idx = index(i);
      auto event = eo_events.Event(event_idx);
      status = func(category_key, event);
      if (status.code() != error::OK) {
        return status;
      }
      index(i) += 1;
    }
  }
  return MyStatus::OK();
}

#define IS_START_IDX(i) (i % 2 == 0)
#define IS_END_IDX(i) (i % 2 == 1)
OverlapResult OverlapComputer::ComputeOverlap(bool keep_empty_time, bool keep_intervals) const {
  OverlapResult r(category_times);
  r.idx_map = ctimes.idx_map;

  // NOTE: eigen uses int for indexing rows/columns, not size_t...
  // https://stackoverflow.com/questions/33993918/eigenmatrix-why-does-eigen-expect-an-int-and-not-size-t
  // using IndexType = size_t;
  using IndexType = int;

  using IdxArray = Array<size_t, Dynamic, 1>;
  size_t k = ctimes.size();
  assert(ctimes.size() == category_times.size());
  IdxArray index = IdxArray::Zero(k);

  // std::vector<EOEvents> times;
  std::vector<const TimeUsec*> times;
  std::vector<size_t> n_events;
  times.reserve(k);
  for (const auto& pair : ctimes.eo_times) {
    const auto& eo_events = pair.second;
    times.push_back(eo_events.RawPtr());
    n_events.push_back(eo_events.size());
  }

  // Q: Can we establish how many intervals there will be ahead of time... or an upper bound?
  //    With two events we could have 2 intervals (the events), or 3 (overlap between the two).
  //    With three events, we could have anywhere between 3 and 5 events.
  //    I suspect the most overlap happens when we consider adjacent events partially overlapping each other,
  //    leading to an upper bound of (2*N - 1) intervals for N events... but I'm not sure how to show this.
  //    But I'm not sure how to show this.
  size_t N = 0;
  for (const auto& pair : ctimes.eo_times) {
    const auto& eo_events = pair.second;
    N += eo_events.size();
  }
  if (keep_intervals) {
    r.interval_meta.intervals.reserve(2*N - 1);
  }

  IdxArray lengths = IdxArray(k);
  {
    IndexType i = 0;
    // NOTE: index order of lengths(...) is determined by iteration order of ctimes.eo_times...
    for (const auto& pair : ctimes.eo_times) {
      lengths(i) = 2 * pair.second.size();
      i += 1;
    }
  }

  TimeUsec min_time_value = std::numeric_limits<TimeUsec>::min();
  TimeUsec last_time = min_time_value;
  CategoryKeyBitset cur_cat = CategoryKeyBitset::EmptySet(ctimes.idx_map);
  CategoryKeyBitset last_cat = cur_cat;
  auto const& all_ops_set = CategoryKeyBitset::Ops(cur_cat);

  // If we just STARTED a new event:
  //   // Time to record the LAST interval.
  //   // We DON'T include the new event in the next interval.
  //   // But we might be in the middle of existing events.
  //   start_events = last_interval_end_events
  //   end_events = (index + 1)/2
  //   last_interval_end_events = end_events
  //   start_us = last_time
  //   end_us = min_time
  // If we just ENDED a new event:
  //   // Time to record the LAST interval.
  //   // We DO include the ending event in the next interval.
  //   // But we might be in the middle of existing events.
  //   start_events = last_interval_end_events
  //   end_events = (index + 1)/2
  //   last_interval_end_events = end_events
  //   start_us = last_time
  //   end_us = min_time
  //
  // // Update indices
  // index(min_event) += 1
  //
  // // NOTE: As long as we record interval information BEFORE updating the index,
  // // we can use the same code to record interval info when adding/removing events.

  size_t next_interval_id = 0;
  auto zero_indices = IdxArray::Zero(k);
  IdxArray last_interval_end_events = zero_indices;
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
      // Use last_cat where there was non_zero_time, and cur_cat where there is non_zero_time.
      r.category_trans_counts[std::make_tuple(last_cat, cur_cat)] += 1;
      last_cat = cur_cat;

      if (keep_intervals) {
        auto end_events = (index + 1)/2;
//        assert(n_events.size() == static_cast<size_t>(end_events.size()));
//        for (int i = 0; i < end_events.size(); i++) {
//          auto end_events_i = end_events(i);
//          auto n_events_i = n_events[i];
//          assert(end_events_i <= n_events_i);
//          auto start_events_i = last_interval_end_events[i];
//          assert(start_events_i <= end_events_i);
//        }
        OverlapInterval overlap_interval(
            next_interval_id,
            last_interval_end_events,
            end_events,
            last_time,
            time_chunk);

//        assert(overlap_interval.start_events.size() == overlap_interval.end_events.size());
//        assert(category_times.eo_times.size() == static_cast<size_t>(overlap_interval.start_events.size())); // FAIL

//        for (int i = 0; i < end_events.size(); i++) {
//          auto end_events_i = overlap_interval.end_events(i);
//          auto n_events_i = n_events[i];
//          assert(end_events_i <= n_events_i);
//          auto start_events_i = overlap_interval.start_events(i);
//          assert(start_events_i <= end_events_i);
//        }


        next_interval_id += 1;
        last_interval_end_events = overlap_interval.end_events;
        // ASSUME: number of intervals is <= 2*N - 1
        // I'm not sure if this is the case, but it seems to be the upper-bound trend for examples with 2/3/4 events.
        assert(overlap_interval.interval_id < r.interval_meta.intervals.capacity());
        r.interval_meta.intervals.push_back(std::move(overlap_interval));
      }
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

  size_t k_end = ctimes.size();
  assert(k_end == k);
  DBG_LOG("ctimes.size at start = {}, ctimes.size at end = {}", k, k_end);

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

template<class OStream, typename Arg>
OStream & WriteCSVCol(OStream& os, const std::string& delimiter, const char escape, Arg&& arg) {
  std::stringstream ss;
  ss << std::forward<Arg>(arg);
  os << std::quoted(ss.str(), CSV_QUOTE_CHAR, escape);
  return os;
}


template<class OStream, typename Arg>
OStream & WriteCSVCols(OStream& os, const std::string& delimiter, const char escape, Arg&& arg) {
  return WriteCSVCol(os, delimiter, escape, std::forward<Arg>(arg));
}

template<class OStream, typename Arg, typename ...Args>
OStream & WriteCSVCols(OStream& os, const std::string& delimiter, const char escape, Arg&& arg, Args&&... args)
{
  WriteCSVCol(os, delimiter, escape, std::forward<Arg>(arg));
  os << delimiter;
  return WriteCSVCols(os, delimiter, escape, std::forward<Args>(args)...);
}

template<class OStream, typename Iterable>
OStream & WriteCSVColsForIterable(OStream& os, const std::string& delimiter, const char escape, const Iterable& xs)
{
  size_t i = 0;
  for (auto const& x : xs) {
    if (i != 0) {
      os << delimiter;
    }
    os << std::quoted(x, CSV_QUOTE_CHAR, escape);
    i += 1;
  }
  return os;
}

template<class OStream, typename T>
OStream & WriteCSVCols(OStream& os, const std::string& delimiter, const char escape, const std::vector<T>& xs)
{
  return WriteCSVColsForIterable(os, delimiter, escape, xs);
}
template<class OStream, typename T>
OStream & WriteCSVCols(OStream& os, const std::string& delimiter, const char escape, const std::list<T>& xs)
{
  return WriteCSVColsForIterable(os, delimiter, escape, xs);
}
template<class OStream, typename T>
OStream & WriteCSVCols(OStream& os, const std::string& delimiter, const char escape, const std::set<T>& xs)
{
  return WriteCSVColsForIterable(os, delimiter, escape, xs);
}

//template <class OStream, typename Iterable>
//void WriteCSVRow(OStream& os, Iterable xs, const std::string& delimiter, const std::string& escape) {
//  size_t i = 0;
//  for (auto const& x : xs) {
//    if (i != 0) {
//      os << delimiter;
//    }
//    os << std::quoted(x, delimiter, escape);
//  }
//  os << std::endl;
//}

MyStatus IntervalMeta::EachInterval(const OverlapResult& this_result, EachIntervalFunc func) const {
  auto status = MyStatus::OK();
  for (const auto& interval : intervals) {
    status = func(interval);
    if (status.code() != error::OK) {
      return status;
    }
  }
  return status;
}

const std::vector<std::string>& NvprofEventMetadata::GetHeader() const {
  return *header;
}
const std::vector<std::string>& NvprofEventMetadata::GetRow() const {
  return row;
}
IEventMetadata* NvprofEventMetadata::clone() const {
  auto obj = new NvprofEventMetadata(header, row);
  return obj;
}

MyStatus OverlapResult::DumpIntervalCSV(const std::string& base_path) const {
  boost::filesystem::path bbase_path(base_path);
  auto csv_path = bbase_path.string() + (std::string(".Intervals") + ".csv");
  std::ofstream out(csv_path);
  if (!out) {
    std::stringstream ss;
    ss << "Failed to write to " << csv_path << ": " << strerror(errno);
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  size_t csv_line = 0;
  return interval_meta.EachInterval(*this, [&] (const OverlapInterval& interval) {
    return interval.EachEvent(*this, [&] (const CategoryKey& category_key, const EOEvent& event) {
      // NOTE: this COULD be an operation event...
      auto event_metadata = event.metadata();

      if (csv_line == 0) {
        // Output header
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, "interval_id", "start_us", "duration_us", "num_events");
        out << std::endl;
      }

      //
      // Output data line
      //

      // Data: interval_id, event_name, start_us, duration_us
      auto num_events = (interval.end_events - interval.start_events).sum();
      // I don't think we should have any empty intervals...?
      assert(num_events > 0);
      WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, interval.interval_id, interval.start_us, interval.duration_us, num_events);
      out << std::endl;

      csv_line += 1;
      return MyStatus::OK();
    });
  });
}

MyStatus OverlapResult::DumpCSVFiles(const std::string& base_path) const {
  auto status = MyStatus::OK();
  status = DumpIntervalCSV(base_path);
  IF_BAD_STATUS_RETURN(status);
  status = DumpIntervalEventsCSV(base_path);
  IF_BAD_STATUS_RETURN(status);
  return MyStatus::OK();
}

MyStatus OverlapResult::DumpIntervalEventsCSV(const std::string& base_path) const {
  boost::filesystem::path bbase_path(base_path);
  auto csv_path = bbase_path.string() + (std::string(".Events") + ".csv");
  std::ofstream out(csv_path);
  if (!out) {
    std::stringstream ss;
    ss << "Failed to write to " << csv_path << ": " << strerror(errno);
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  size_t csv_line = 0;
  int header_size = -1;
  return interval_meta.EachInterval(*this, [&] (const OverlapInterval& interval) {
    return interval.EachEvent(*this, [&] (const CategoryKey& category_key, const EOEvent& event) {
      // NOTE: this COULD be an operation event...
      auto event_metadata = event.metadata();

      if (csv_line == 0) {
        // Output header

        // Header: interval_id, event_name, start_us, duration_us
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, "interval_id", "event_name", "start_us", "duration_us");
        // Header: category
        out << CSV_DELIM_STR;
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, "category");
        // Header: process
        out << CSV_DELIM_STR;
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, "process");
        if (event_metadata) {
          // Header: --nvprof_keep_column_names
          out << CSV_DELIM_STR;
          WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, event_metadata->GetHeader());
          header_size = static_cast<int>(event_metadata->GetHeader().size());
        }
        out << std::endl;

      }

      //
      // Output data line
      //

      // Data: interval_id, event_name, start_us, duration_us
      WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, interval.interval_id, event.name(), event.start_time_us(), event.duration_us());

      // Data: category
      assert(category_key.non_ops.size() == 1);
      auto const& category = *(category_key.non_ops.begin());
      out << CSV_DELIM_STR;
      WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, category);

      // Data: process
      out << CSV_DELIM_STR;
      if (category_key.procs.size() > 0) {
        assert(category_key.procs.size() == 1);
        auto const& process = *(category_key.procs.begin());
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, process);
      } else {
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, "");
      }

      // Data: --nvprof_keep_column_names
      if (event_metadata) {
        out << CSV_DELIM_STR;
        WriteCSVCols(out, CSV_DELIM_STR, CSV_ESCAPE_CHAR, event_metadata->GetRow());
        if (header_size != -1) {
          assert(event_metadata->GetRow().size() == static_cast<size_t>(header_size));
        }
      }

      out << std::endl;

      csv_line += 1;
      return MyStatus::OK();
    });
  });
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
  PrintIndent(out, indent + 1);
  out << "CategoryTransitionCount: size = " << category_trans_counts.size();
  i = 0;
  for (const auto& pair : category_trans_counts) {
    auto const& from_bitset = std::get<0>(pair.first);
    auto const& to_bitset = std::get<1>(pair.first);
    auto const& trans_count = pair.second;

    out << "\n";
    PrintIndent(out, indent + 2);
    out << "CategoryTransitionCount[" << i << "]: count = " << trans_count;
    out << "\n";
    PrintIndent(out, indent + 3);
    out << "From Category:\n";
    from_bitset.Print(out, indent + 4);
    out << "\n";
    PrintIndent(out, indent + 3);
    out << "To Category:\n";
    to_bitset.Print(out, indent + 4);
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


MyStatus NvprofTraceFileReader::Init() {
  MyStatus status;

  if (_initialized) {
    return MyStatus::OK();
  }

  switch (_file_type) {
    case NVPROF_GPU_TRACE_CSV_FILE:
    case NVPROF_API_TRACE_CSV_FILE:
      break;
    default:
      assert(false);
  }

  status = _ReadProcess();
  IF_BAD_STATUS_RETURN(status);
  status = _ReadCSVMeta();
  IF_BAD_STATUS_RETURN(status);

  _initialized = true;
  return MyStatus::OK();
}
MyStatus NvprofTraceFileReader::_ReadProcess() {
  std::string match_group;
  boost::filesystem::path bpath(_path);
  if (!args.FLAGS_nvprof_process_regex.has_value()) {
    match_group = bpath.filename().string();
  } else {
    std::regex nvprof_process_regex(args.FLAGS_nvprof_process_regex.value());
    std::string filename = bpath.filename().string();
    std::smatch m;
    if (!std::regex_search(filename, m, nvprof_process_regex)) {
      std::stringstream ss;
      ss << "--nvprof-process-regex=\"" << args.FLAGS_nvprof_process_regex.value()
         << "\" doesn't match " << bpath.filename().string()
         << " for path = " << _path;
      return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
    if (m.size() > 1) {
      match_group = m.str(1);
    } else {
      match_group = m.str(0);
    }
  }

  // Not sure how to figure out what machine we ran on from nvprof output.
  _machine = "machine";
  _process = match_group;
  _phase = _process;

  return MyStatus::OK();
}
void NvprofTraceFileReader::Clear() {
}
MyStatus NvprofTraceFileReader::ReadMeta(nlohmann::json* meta) {
  // Default: no extra initialization.
  return MyStatus::OK();
}
MyStatus NvprofTraceFileReader::ReadTraceFileMeta(TraceFileMeta* meta) {
  MyStatus status;
  status = Init();
  IF_BAD_STATUS_RETURN(status);
  auto category = _nvprof_file_type->category;
  meta->n_events[category] += _num_data_lines;
  meta->categories.insert(category);
  return MyStatus::OK();
}
std::vector<std::string> ParseCSVRow(const std::string& line) {
  std::vector<std::string> cols;
  using namespace boost;
  tokenizer<escaped_list_separator<char> > tk(line, escaped_list_separator<char>('\\', ',', '\"'));
  for (tokenizer<escaped_list_separator<char> >::iterator it(tk.begin()); it != tk.end(); ++it) {
    cols.push_back(*it);
  }
  return cols;
}
MyStatus NvprofTraceFileReader::_ReadCSVMeta() {
  assert(!_initialized);
  auto status = MyStatus::OK();

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    DBG_LOG("Reading csv={}", _path);
  }

  // File format looks like:
#if 0
  // ==> profile.process_15562.nvprof.api_trace.csv <==
// ======== Profiling result:
// "Start","Duration","Name","Correlation_ID"
// us,us,,
// 0.000000,2.866000,"cuDeviceGetPCIBusId",1
// 35863.518000,0.952000,"cuDeviceGetCount",2
//
// ==> profile.process_15562.nvprof.gpu_trace.csv <==
// ======== Profiling result:
// "Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","SrcMemType","DstMemType","Device","Context","Stream","Name","Correlation_ID"
// us,us,,,,,,,,KB,KB,KB,GB/s,,,,,,,
// 0.000000,1.888000,,,,,,,,,,1.003906,0.507097,"Device",,"GeForce RTX 2080 Ti (0)","1","7","[CUDA memset]",272
// 4643.346000,1.536000,,,,,,,,,,0.125000,0.077610,"Pinned","Device","GeForce RTX 2080 Ti (0)","1","22","[CUDA memcpy HtoD]",316
#endif

  // NOTE: we assume csv files have been read ahead-of-time by iml-analyze.
  // Parse _header and _units.
  {
    std::ifstream infile(_path);
    if (!infile) {
      std::stringstream ss;
      ss << "Failed to open file " << _path << " with error " << errno << "; " << strerror(errno);
      return MyStatus(error::INVALID_ARGUMENT, ss.str());
    } else {
      std::string line;
      size_t lineno = 1;
      const std::regex comment_regex(R"(^=+\s+(.*))");
      const std::regex ignore_line_regex(R"(^no kernels were profiled)", std::regex_constants::icase);
      while (std::getline(infile, line)) {
        std::smatch m;
        if (std::regex_search(line, m, comment_regex)) {
          if (std::regex_search(m.str(1), std::regex("warning", std::regex_constants::icase))) {
            DBG_WARN("Saw WARNING in {} at line {}:\n{}", _path, lineno);
          }
        } else if (std::regex_search(line, m, ignore_line_regex)) {
          // pass
        } else {
          if (_num_other_lines == 0) {
            if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
              DBG_LOG("Parse nvprof csv header from line {}:\n  {}", lineno, line);
            }
            _header = ParseCSVRow(line);
          } else if (_num_other_lines == 1) {
            if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
              DBG_LOG("Parse nvprof csv units from line {}:\n  {}", lineno, line);
            }
            _units = ParseCSVRow(line);
          } else {
            // We've encountered the first data line.
            break;
          }
          _num_other_lines += 1;
        }
        _num_skip_lines += 1;
        lineno += 1;

      }

      while (std::getline(infile, line)) {
        _num_data_lines += 1;
        lineno += 1;
      }
    }
  }

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    DBG_LOG("Read metadata for nvprof csv file {};  _num_data_lines = {}, _num_skip_lines = {}, _num_other_lines = {}",
            _path, _num_data_lines, _num_skip_lines, _num_other_lines);
  }

  _nvprof_file_type = nullptr;
  status = GetNvprofFileType(_header, _file_type, &_nvprof_file_type);
  if (status.code() != MyStatus::OK().code()) {
    std::stringstream ss;
    ss << "Failed to parse nvprof csv file " << _path << ": " << status.error_message();
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  if (_num_data_lines > 0 && _header.size() > 0) {
    if (_nvprof_file_type->HeaderMatches(_header)) {
      _header_meta = _nvprof_file_type->ParseHeaderMeta(args, _header);
    } else {
      std::stringstream ss;
      ss << "Not sure how to parse nvprof csv file " << _path << " with header that looks like:\n"
         << "  ";
      PrintValue(ss, _header);
      return MyStatus(error::INVALID_ARGUMENT, ss.str());
    }
  }

  if (
      (_nvprof_file_type->file_type == NVPROF_CSV_API_TRACE && _file_type != NVPROF_API_TRACE_CSV_FILE) ||
      (_nvprof_file_type->file_type == NVPROF_CSV_GPU_TRACE && _file_type != NVPROF_GPU_TRACE_CSV_FILE)
      ) {
    std::stringstream ss;
    ss << "Failed to parse nvprof csv file " << _path << ": " << "file name extension didn't match file content based on its csv header:\n"
       << "  header = ";
    PrintValue(ss, _header);
    ss << "\n";
    ss << "Use .nvprof.api_trace.csv for [nvprof --print-api-trace] and \n";
    ss << "    .nvprof.gpu_trace.csv for [nvprof --print-gpu-trace]\n";
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  return MyStatus::OK();

}
MyStatus NvprofTraceFileReader::EachEvent(const Category& category, EachEventFunc func) const {
  assert(_initialized);
  if (_num_data_lines == 0) {
    return MyStatus::OK();
  }
  // Read data.
  {
    std::ifstream infile(_path);
    size_t lineno = 1;
    for (size_t i = 0; i < _num_skip_lines + _num_other_lines; i++) {
      std::string line;
      std::getline(infile, line);
      lineno += 1;
    }
    std::string line;
    size_t i = 0;
    TimeUsec last_start_us = 0;
    while (std::getline(infile, line)) {
      auto row = ParseCSVRow(line);
      NvprofFileType::EventRow event_row;
      auto status = _nvprof_file_type->ParseRowEvent(_header_meta, row, &event_row);
      if (_file_type == NVPROF_GPU_TRACE_CSV_FILE && args.FLAGS_nvprof_keep_column_names.has_value()) {
        assert(event_row.event_metadata.size() > 0);
      }
      if (status.code() != error::OK) {
        std::stringstream ss;
        ss << "Failed to parse nvprof CSV file = " << _path << ": " << status.error_message();
        return MyStatus(status.code(), ss.str());
      }
      auto start_us = event_row.start_us;
      auto end_us = event_row.end_us;
      if (i != 0) {
        // ASSUMPTION: "nvprof [--print-gpu-trace|--print-api-trace]" outputs events sorted by start time.
        // Q: is this true for multi-threaded traces?
        // If this fails, it just means we need to sort the events before recording them.
        assert(last_start_us <= start_us);
      }
//      OptionalString name;
//      name = event_row.name;
      if (end_us < start_us) {
        DBG_BREAKPOINT("negative event duration");
        DBG_LOG("BUG: skip negative duration Event(name=\"{}\", start_us={}, duration_us={} us)",
                event_row.name, start_us, end_us - start_us);
        // Just insert a zero-length event since we've already preallocated space for it
        // (they're effectively be ignored during overlap).
        end_us = start_us;
      }
      // func(name, start_us, end_us);
      func(event_row);

      lineno += 1;
      i += 1;
      last_start_us = start_us;
    }
  }

  return MyStatus::OK();
}
std::set<Category> NvprofTraceFileReader::Categories() const {
  return {_nvprof_file_type->category};
}
const Machine& NvprofTraceFileReader::get_machine() const {
  assert(_initialized);
  return _machine;
}
const Process& NvprofTraceFileReader::get_process() const {
  assert(_initialized);
  return _process;
}
const Phase& NvprofTraceFileReader::get_phase() const {
  assert(_initialized);
  return _phase;
}

MyStatus NvprofCSVParser::_ReadEOEvents(
    const Category& category,
    const TraceFileMeta& meta,
    EOEvents* eo_events) {
  auto status = MyStatus::OK();
  auto cat_key = CategoryKey::FromCategory(meta.process, category);
  auto n_events = meta.n_events.at(category);
  *eo_events = EOEvents::PreallocateEvents(cat_key, n_events,
      /*keep_names=*/true,
      /*keep_event_metadata=*/true);
  NvprofTraceFileReader reader(args, meta.get_path(), _file_type);
  status = reader.Init();
  IF_BAD_STATUS_RETURN(status);
  status = reader.EachEvent(category, [this, eo_events] (const NvprofFileType::EventRow& event_row) {
    std::unique_ptr<IEventMetadata> event_metadata;
    if (args.FLAGS_nvprof_keep_column_names.has_value()) {
      event_metadata = std::make_unique<NvprofEventMetadata>(_event_metadata_header, event_row.event_metadata);
    }
    eo_events->AppendEvent(event_row.name, event_row.start_us, event_row.end_us, std::move(event_metadata));
    return MyStatus::OK();
  });
  IF_BAD_STATUS_RETURN(status);
  return MyStatus::OK();
}
MyStatus NvprofCSVParser::AppendAllCategoryTimes(
    const EntireTraceMeta& entire_meta,
    const std::vector<TraceFileMeta>& metas,
    CategoryTimes* out_category_times) {
  auto status = MyStatus::OK();

  // Preallocate space for ALL metas (csv files).
  // For each csv file, do merge inplace algorithm to read all events into one big flat vector.
  // Convert big flat vector to EOEvents.
  // OR: implement merge inplace for EOEvents (not sure how to do that)
  for (const auto& meta : metas) {
    for (const auto& category : meta.categories) {
      auto cat_key = CategoryKey::FromCategory(meta.process, category);

      EOEvents eo_events;
      status = _ReadEOEvents(category, meta, &eo_events);
      IF_BAD_STATUS_RETURN(status);
      if (!out_category_times->Count(cat_key)) {
//        out_category_times->MoveEventsInto(cat_key, std::move(eo_events));
        out_category_times->MoveEventsInto(cat_key, eo_events);
      } else {
        // NOTE: this assume eo_events is already sorted.
        out_category_times->MergeEventsInto(cat_key, eo_events);
      }

    }
  }

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    std::stringstream ss;
    for (const auto& meta : metas) {
      ss << "\n";
      meta.Print(ss, 1);
    }
    DBG_LOG("Reading csv from {} metas:\n{}", metas.size(), ss.str());
  }

  return MyStatus::OK();
}

std::map<std::set<CategoryKey>, TimeUsec> OverlapResult::AsOverlapMap() const {
  std::map<std::set<CategoryKey>, TimeUsec> overlap_map;

  for (const auto& pair : overlap) {
    auto const& bitset = pair.first;
    auto time_us = pair.second;
    auto keys = bitset.Keys();
    assert(overlap_map.count(keys) == 0);
    overlap_map[keys] = time_us;
  }
  return overlap_map;
}

std::map<std::tuple<std::set<CategoryKey>, std::set<CategoryKey>>, size_t> OverlapResult::AsCategoryTransCountsMap() const {
  std::map<std::tuple<std::set<CategoryKey>, std::set<CategoryKey>>, size_t> counts_map;

  for (const auto& pair : category_trans_counts) {
    auto const& from_bitset = std::get<0>(pair.first);
    auto const& to_bitset = std::get<1>(pair.first);
    auto count = pair.second;
    auto key = std::make_tuple(from_bitset.Keys(), to_bitset.Keys());
    assert(counts_map.count(key) == 0);
    counts_map[key] = count;
  }
  return counts_map;
}

bool NvprofFileType::HeaderMatches(const std::vector<std::string>& row) const {
  // row contains all the columns present in nvprof file-type header.
  // NOTE: row may contain extra columns we don't care about.
  // ==> header subsetof row
  std::set<std::string> row_set;
  for (const auto& field : row) {
    row_set.insert(field);
  }
  for (const auto& field : header) {
    if (!row_set.count(field)) {
      return false;
    }
  }
  return true;
}
std::map<std::string, size_t> NvprofFileType::ParseColIdxMap(const std::vector<std::string>& row) const {
  std::map<std::string, size_t> col_idx_map;
  for (size_t i = 0; i < row.size(); i++) {
    col_idx_map[row[i]] = i;
  }
  return col_idx_map;
}

template <class Self>
NvprofFileType::HeaderMeta CommonParseHeaderMeta(const Self* self, const RLSAnalyzeArgs& args, const std::vector<std::string>& row) {
  NvprofFileType::HeaderMeta header_meta;
  header_meta.col_idx_map = self->ParseColIdxMap(row);
  header_meta.start_idx = header_meta.col_idx_map.at("Start");
  header_meta.duration_idx = header_meta.col_idx_map.at("Duration");
  header_meta.name_idx = header_meta.col_idx_map.at("Name");
  if (args.FLAGS_nvprof_keep_column_names.has_value()) {
    header_meta.event_metadata_cols = args.FLAGS_nvprof_keep_column_names.value();
  }
  return header_meta;
}

NvprofFileType::HeaderMeta NvprofAPITraceFileType::ParseHeaderMeta(const RLSAnalyzeArgs& args, const std::vector<std::string>& row) const {
  return CommonParseHeaderMeta(this, args, row);
}
NvprofFileType::HeaderMeta NvprofGPUTraceFileType::ParseHeaderMeta(const RLSAnalyzeArgs& args, const std::vector<std::string>& row) const {
  return CommonParseHeaderMeta(this, args, row);
}

template <class Self>
MyStatus CommonParseRowEvent(
    const Self* self, const NvprofFileType::HeaderMeta& header_meta, const std::vector<std::string>& row,
    NvprofFileType::EventRow* event_row) {
  double start_us_dbl = atof(row[header_meta.start_idx].c_str());
  double duration_us_dbl = atof(row[header_meta.duration_idx].c_str());
  const std::string& name = row[header_meta.name_idx];
  TimeUsec start_us = static_cast<TimeUsec>(round(start_us_dbl));
  TimeUsec end_us = static_cast<TimeUsec>(round(start_us_dbl + duration_us_dbl));
  for (auto const& fieldname : header_meta.event_metadata_cols) {
    auto it = header_meta.col_idx_map.find(fieldname);
    if (it == header_meta.col_idx_map.end()) {
      event_row->event_metadata.push_back("");
      continue;
    }
//    if (it == header_meta.col_idx_map.end()) {
//      std::stringstream ss;
//      std::vector<std::string> choices;
//      for (const auto& pair : header_meta.col_idx_map) {
//        choices.push_back(pair.first);
//      }
//      ss << "Didn't see colname=" << fieldname << " in csv file; choices are: ";
//      PrintValue(ss, choices);
//      return MyStatus(error::INVALID_ARGUMENT, ss.str());
//    }
    auto idx = it->second;
    event_row->event_metadata.push_back(row[idx]);
  }
  event_row->name = name;
  event_row->start_us = start_us;
  event_row->end_us = end_us;
  return MyStatus::OK();
}

MyStatus NvprofAPITraceFileType::ParseRowEvent(
    const HeaderMeta& header_meta, const std::vector<std::string>& row,
    NvprofFileType::EventRow* event_row) const {
  return CommonParseRowEvent(this, header_meta, row, event_row);
//  double start_us_dbl = atof(row[header_meta.start_idx].c_str());
//  double duration_us_dbl = atof(row[header_meta.duration_idx].c_str());
//  const std::string& name = row[header_meta.name_idx];
//  TimeUsec start_us = static_cast<TimeUsec>(round(start_us_dbl));
//  TimeUsec end_us = static_cast<TimeUsec>(round(start_us_dbl + duration_us_dbl));
//  EventRow event_row {name, start_us, end_us};
//  return event_row;
}
MyStatus NvprofGPUTraceFileType::ParseRowEvent(
    const HeaderMeta& header_meta, const std::vector<std::string>& row,
    NvprofFileType::EventRow* event_row) const {
  return CommonParseRowEvent(this, header_meta, row, event_row);
//  double start_us_dbl = atof(row[header_meta.start_idx].c_str());
//  double duration_us_dbl = atof(row[header_meta.duration_idx].c_str());
//  const std::string& name = row[header_meta.name_idx];
//  TimeUsec start_us = static_cast<TimeUsec>(round(start_us_dbl));
//  TimeUsec end_us = static_cast<TimeUsec>(round(start_us_dbl + duration_us_dbl));
//  EventRow event_row {name, start_us, end_us};
//  return event_row;
}


} // namespace rlscope

