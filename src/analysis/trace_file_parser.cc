//
// Created by jagle on 11/13/2019.
//

#include "analysis/trace_file_parser.h"
#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/debug_flags.h"

#include <Eigen/Dense>

#include <boost/compute/algorithm/reduce.hpp>

// NOTE: not until gcc 7.1 (7.4.0 in Ubuntu 18.04); use boost::optional instead.
// #include <optional>
#include <boost/optional.hpp>
#include <boost/utility/string_view.hpp>

//#include <string_view>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <assert.h>
#include <numeric>
#include <iostream>
#include <spdlog/spdlog.h>
#include <limits>

using namespace Eigen;

namespace tensorflow {

const std::set<Category> CATEGORIES_C_EVENTS = std::set<Category>{
    CATEGORY_TF_API,
    CATEGORY_SIMULATOR_CPP,
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

MyStatus CategoryEventsParser::_CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) {
  MyStatus status = MyStatus::OK();
  for (const auto& pair : proto.category_events()) {
    size_t number_of_events = pair.second.events().size();
    const auto& category = pair.first;
//    DBG_LOG("CategoryEventsParser::_CountCategoryTimes category={}, n_events={}", category, number_of_events);
    if (category == CATEGORY_OPERATION) {
      status = _CountCategoryTimesOperation(count, proto);
      IF_BAD_STATUS_RETURN(status);
    } else {
      auto category_key = CategoryKey::FromCategory(get_process(), category);
      size_t n_events = pair.second.events().size();
      count->Add(category_key, n_events);
    }
  }
  return MyStatus::OK();
}

// PSEUDOCODE:
// # NOTE: to construct oe_times, we must run this twice; # once to determine the number of events,
// # and another to fill in the events.
// def each_op_event():
// 	ops = []
// 	last_time = None
// 	while i < len(events) or len(ops) > 0:
//    if len(ops) == 0:
//      last_time = events[i].start
//      ops.push(events[i])
// 		  i += 1
//      continue
//
// 		# Skip empty events
// 		if i < len(events) and events[i].start == events[i].end:
// 		  i += 1
// 		  continue
//
// 		if i < len(events) and ops[-1].subsumes(events[i]):
// 		  yield Op(
// 		    op=ops[-1].name,
// 		    start=last_time,
// 		    end=events[i].start)
// 		   last_time = events[i].start
// 		  ops.push(events[i])
// 		  i += 1
// 		else:
// 	  	# assert:
// 	   	#   ops[-1] ends before events[i] begins
// 		  yield Op(
// 		    op=ops[-1].name,
// 		    start=last_time,
// 		    end=ops[-1].end_time)
// 		  ops.pop()
//      last_time = ops[-1].end_time
template <typename EventProto, typename EventProtoList, typename Func>
MyStatus EachOpEvent(const EventProtoList& event_protos, Func func) {
  std::list<const EventProto*> ops;
  TimeUsec last_time = std::numeric_limits<TimeUsec>::max();
  auto max_time = std::numeric_limits<TimeUsec>::max();
  size_t i = 0;
  // For some reason, protobuf uses "int" for the size() of its repeated fields.
  assert(event_protos.size() >= 0);
  size_t events_size = static_cast<size_t>(event_protos.size());
  auto get_end_time = [] (const EventProto* A) {
    return A->start_time_us() + A->duration_us();
  };
  auto subsumes = [get_end_time] (const EventProto* A, const EventProto* B) {
    //     [ B ]
    // [     A     ]
    // return A->start_time_us() <= B->start_time_us() <= B->end_time_us() <= A->end_time_us();
    // ===
    // return A->start_time_us() <= B->start_time_us() <= A->end_time_us();
    return A->start_time_us() <= B->start_time_us() &&
                                 B->start_time_us() <= get_end_time(A);
  };
  while (i < events_size || ops.size() > 0) {
    if (ops.size() == 0) {
      last_time = event_protos[i].start_time_us();
      ops.push_back(&event_protos[i]);
      i += 1;
      continue;
    }

    // Skip empty events:
    if (i < events_size && event_protos[i].start_time_us() == get_end_time(&event_protos[i])) {
      i += 1;
      continue;
    }

    if (i < events_size && subsumes(ops.back(), &event_protos[i])) {
      auto op = ops.back();
      auto start_time_us = last_time;
      auto end_time_us = event_protos[i].start_time_us();
      func(op->name(), start_time_us, end_time_us);
      ops.push_back(&event_protos[i]);
      last_time = end_time_us;
      i += 1;
    } else {
      auto op = ops.back();
      auto start_time_us = last_time;
      auto end_time_us = get_end_time(op);
      func(op->name(), start_time_us, end_time_us);
      ops.pop_back();
      last_time = end_time_us;
    }
  }
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::_CountCategoryTimesOperation(CategoryTimesCount* count, const ProtoKlass& proto) {
  MyStatus status = MyStatus::OK();
  auto const& process = get_process();
  auto const& events = proto.category_events().at(CATEGORY_OPERATION).events();
//  DBG_LOG("CategoryEventsParser::ParserMeta.num_CATEGORY_OPERATION = {}", events.size());
  count->AddExtra(CategoryKey::FromCategory(process, CATEGORY_EXTRA_OPERATION), events.size());

  EachOpEvent<iml::Event>(
      events,
      [&process, count] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
        auto category_key = CategoryKey::FromOpEvent(process, op);
        count->Add(category_key, 1);
      });
  return MyStatus::OK();
}
MyStatus CategoryEventsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) {
  MyStatus status = MyStatus::OK();
  for (const auto& pair : proto.category_events()) {
    const auto& category = pair.first;
    if (category == CATEGORY_OPERATION) {
      status = _AppendCategoryOperation(category, proto, out_category_times);
      IF_BAD_STATUS_RETURN(status);
    } else {
      CategoryKey category_key = CategoryKey::FromCategory(get_process(), category);
//      if (out_category_times->eo_times.find(category_key) == out_category_times->eo_times.end()) {
//        std::stringstream ss;
//        ss << "FAIL:\n";
//        out_category_times->PrintSummary(ss, 0);
//        ss << "\n";
//        category_key.Print(ss, 0);
//        ss << "\n";
//        SPDLOG_DEBUG(ss.str());
//
//        for (const auto& eo_times_pair : out_category_times->eo_times) {
//          if (out_category_times->eo_times.find(eo_times_pair.first) != out_category_times->eo_times.end()) {
//            assert(out_category_times->eo_times.find(eo_times_pair.first) != out_category_times->eo_times.end());
//          }
//        }
//
//        assert(false);
//      }
      EOEvents& eo_events = out_category_times->eo_times.at(category_key);
      status = _AppendCategory(category, proto, &eo_events);
      IF_BAD_STATUS_RETURN(status);
    }
  }
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::_AppendCategoryOperation(const Category& category, const ProtoKlass& proto, CategoryTimes* out_category_times) {
  // for (start, end, op_name) in _EachEvent():
  //   category_key = CategoryKey::FromOpEvent(process, op_name)
  //   out_category_times->eo_times.at(category_key).AppendEvent(start, end)
  MyStatus status = MyStatus::OK();
  auto const& process = get_process();
  assert(category == CATEGORY_OPERATION);
  auto const& events = proto.category_events().at(category).events();
//  auto* parser_meta = GetParserMetaDerived();
//  parser_meta->eo_CATEGORY_OPERATION = std::move(EOEvents(parser_meta->num_CATEGORY_OPERATION, true));
  auto extra_op_category_key = CategoryKey::FromCategory(process, CATEGORY_EXTRA_OPERATION);
  auto& extra_op_events = out_category_times->extra_eo_times.at(extra_op_category_key);
  status = EachEvent(events, [&extra_op_events] (const EventName& name, TimeUsec start_us, TimeUsec end_us) {
    extra_op_events.AppendEvent(name, start_us, end_us);
    return MyStatus::OK();
  });
  IF_BAD_STATUS_RETURN(status);
  // Watch out for accidental copy-construction.
  assert(out_category_times->extra_eo_times.at(extra_op_category_key).size() > 0);

  EachOpEvent<iml::Event>(
      events,
      [&process, out_category_times] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
        auto category_key = CategoryKey::FromOpEvent(process, op);
        out_category_times->eo_times.at(category_key).AppendEvent(op, start_us, end_us);
      });
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::_AppendCategory(const Category& category, const ProtoKlass& proto, EOEvents* eo_events) {
  MyStatus status = MyStatus::OK();
  const auto& events = proto.category_events().at(category).events();
  status = EachEvent(events, [eo_events] (const EventName& name, TimeUsec start_us, TimeUsec end_us) {
    eo_events->AppendEvent(name, start_us, end_us);
    return MyStatus::OK();
  });
  IF_BAD_STATUS_RETURN(status);
  return MyStatus::OK();
}

MyStatus CUDAAPIStatsParser::_CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) {
  const std::string category = CATEGORY_CUDA_API_CPU;
  size_t n_events = proto.events().size();
  auto category_key = CategoryKey::FromCategory(get_process(), category);
  count->Add(category_key, n_events);
  return MyStatus::OK();
}
MyStatus CUDAAPIStatsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) {
  MyStatus status = MyStatus::OK();
  const std::string category = CATEGORY_CUDA_API_CPU;
  auto category_key = CategoryKey::FromCategory(get_process(), category);
  EOEvents& eo_events = out_category_times->eo_times.at(category_key);
  const auto& events = proto.events();

  for (const auto& event : events) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    auto const &name = event.api_name();
    eo_events.AppendEvent(name, start_us, end_us);
  }
  return MyStatus::OK();
}

MyStatus CUDADeviceEventsParser::_CountCategoryTimes(CategoryTimesCount* count, const CUDADeviceEventsParser::ProtoKlass& proto) {
  const std::string category = CATEGORY_GPU;
  for (const auto& dev_events_pair : proto.dev_events()) {
    const auto& dev = dev_events_pair.first;
    const auto& events = dev_events_pair.second.events();
    size_t n_events = events.size();
    auto category_key = CategoryKey::FromCategory(get_process(), category);
    count->Add(category_key, n_events);
  }
  return MyStatus::OK();
}
MyStatus CUDADeviceEventsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, const CUDADeviceEventsParser::ProtoKlass& proto) {
  const std::string category = CATEGORY_GPU;
  auto category_key = CategoryKey::FromCategory(get_process(), category);
  EOEvents& eo_events = out_category_times->eo_times.at(category_key);
  for (const auto& dev_events_pair : proto.dev_events()) {
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
void CategoryTimes::Preallocate(const CategoryKey& category_key, size_t n_events) {
  _Preallocate(&eo_times, category_key, n_events);
}
void CategoryTimes::PreallocateExtra(const CategoryKey& category_key, size_t n_events) {
  _Preallocate(&extra_eo_times, category_key, n_events);
}

size_t CategoryTimes::_Count(const EOTimes& eo_times, const CategoryKey& category_key) const {
  return eo_times.at(category_key).size();
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
  auto preallocate = [] (EOTimes* eo_times, const CategoryTimesCount::CountMap& cmap) {
    for (const auto& pair : cmap) {
      auto const& category_key = pair.first;
      auto const n_events = pair.second;
      bool keep_names = CategoryShouldKeepNames(category_key);
      (*eo_times)[category_key] = EOEvents(n_events, keep_names);
    }
  };
  preallocate(&eo_times, count.num_events);
  preallocate(&extra_eo_times, count.extra_num_events);
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

MyStatus GetRLSEventParser(const std::string& path, std::unique_ptr<IEventFileParser>* parser) {
  auto file_type = GetRLSFileType(path);
  switch (file_type) {
    case RLSFileType::CUDA_API_STATS_FILE:
      parser->reset(new CUDAAPIStatsParser(path));
      break;
    case RLSFileType::CATEGORY_EVENTS_FILE:
      parser->reset(new CategoryEventsParser(path));
      break;
    case RLSFileType::CUDA_DEVICE_EVENTS_FILE:
      parser->reset(new CUDADeviceEventsParser(path));
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

  std::unique_ptr<IEventFileParser> parser;
  status = GetRLSEventParser(path, &parser);
  IF_BAD_STATUS_RETURN(status);

  // Initialize:
  //   CategoryTimesCount count;
  //   Machine machine;
  //   Process process;
  //   Phase phase;

  status = parser->CountCategoryTimes(&count);
  IF_BAD_STATUS_RETURN(status);

  // this->parser_meta = parser._parser_meta;
//  this->parser_meta = parser->GetParserMeta();
//  DBG_LOG("TraceFileMeta::Init() : file_type = {}, parser_meta = {}, Parser::this = {}"
//  , file_type
//  , reinterpret_cast<void*>(this->parser_meta.get())
//  , reinterpret_cast<void*>(this)
//  );

  parser->Init();
  machine = parser->get_machine();
  process = parser->get_process();
  phase = parser->get_phase();

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
    _meta[new_meta.get_machine()][new_meta.get_process()][new_meta.get_phase()][new_meta.get_file_type()][new_meta.get_trace_id()] = new_meta;
  } else {
    *meta = it->second;
  }
  return MyStatus::OK();
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

MyStatus TraceFileWalker::TraceMetas(const Machine& machine, const Process& process, const Phase& phase, std::list<TraceFileMeta>* metas) {
  if (
      _meta.find(machine) == _meta.end()
      || _meta[machine].find(process) == _meta[machine].end()
      || _meta[machine][process].find(phase) == _meta[machine][process].end()
    // || _meta[machine][process][phase].find(trace_id) == _meta[machine][process][phase].end()
      ) {
    // No trace files for this machine/process/phase.
    // Return empty list.
    return MyStatus::OK();
  }
  auto const& file_type_to_meta = _meta.at(machine).at(process).at(phase);
  // NOTE: std::map iterates in sorted order of its key (trace_id).
  for (auto const& pair_01 : _meta.at(machine).at(process).at(phase)) {
    auto file_type = pair_01.first;
    for (auto const& pair_02 : pair_01.second) {
      auto trace_id = pair_02.first;
      auto const &meta = pair_02.second;
      metas->push_back(meta);
    }
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
  return MyStatus::OK();
}

MyStatus RawTraceParser::Init() {
  MyStatus status = MyStatus::OK();

  _has_calibration_files = (
      _cupti_overhead_json_path != "" &&
      _LD_PRELOAD_overhead_json_path != "" &&
      _pyprof_overhead_json_path != ""
  );

  if (_has_calibration_files) {
    status = ReadJson(_cupti_overhead_json_path, &_cupti_overhead_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(_LD_PRELOAD_overhead_json_path, &_LD_PRELOAD_overhead_json);
    IF_BAD_STATUS_RETURN(status);
    status = ReadJson(_pyprof_overhead_json_path, &_pyprof_overhead_json);
    IF_BAD_STATUS_RETURN(status);
  }

  status = _walker.Init();
  IF_BAD_STATUS_RETURN(status);

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

  std::list<TraceFileMeta> metas;
  status = _walker.TraceMetas(machine, process, phase, &metas);
  IF_BAD_STATUS_RETURN(status);

  CategoryTimesCount count;
  for (auto const &meta : metas) {
    count += meta.get_count();
  }

  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
    std::stringstream ss;
    ss << "\n";
    count.Print(ss, 1);
    DBG_LOG("Total count: {}", ss.str());
  }

  // Preallocate space for eo_times for this (machine, process, phase).
  *category_times = std::move(CategoryTimes(process, count));

//  if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//    std::stringstream ss;
//    ss << "\n";
//    category_times->PrintSummary(ss, 1);
//    DBG_LOG("Preallocated: {}", ss.str());
//  }

  for (auto const &meta : metas) {
    std::unique_ptr<IEventFileParser> parser;
    if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
      DBG_LOG("read path = {}", meta.get_path());
    }
    status = GetRLSEventParser(meta.get_path(), &parser);
    IF_BAD_STATUS_RETURN(status);
    // TODO: cache read proto-files to avoid re-reading 20MB files...maybe.
    status = parser->AppendCategoryTimes(category_times);
    IF_BAD_STATUS_RETURN(status);

//    if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
//      DBG_LOG("After {}", meta.path);
//      category_times->PrintSummary(std::cout, 1);
//      std::cout << "\n";
//    }

  }

  if (_has_calibration_files) {
    // Use calibration files to subtract overhead by "injecting" overhead events.
    status = _AppendOverheadEvents(
        machine,
        process,
        phase,
        category_times);
    IF_BAD_STATUS_RETURN(status);
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
      category_times);
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
  auto& CUDA_API_events = category_times->eo_times.at(CUDA_API_category_key);
  auto& LD_PRELOAD_events = category_times->eo_times.at(LD_PRELOAD_category_key);
  auto& CUPTI_events = category_times->eo_times.at(CUPTI_category_key);

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
    CategoryTimes *category_times) {
//      auto* parser_meta = AsDerivedParserMeta<CUDAAPIStatsParser>(parser->GetParserMeta());
  double per_pyprof_interception_us = _pyprof_overhead_json["mean_pyprof_interception_overhead_per_call_us"];
  auto category_key = CategoryKey::FromCategory(process, CATEGORY_PROF_PYTHON_INTERCEPTION);
  size_t n_events = 0;
  for (auto const& category : CATEGORIES_C_EVENTS) {
    n_events += category_times->Count(CategoryKey::FromCategory(process, category));
  }
  // DBG_LOG("_AppendOverhead_PYTHON_INTERCEPTION: n_events = {}", n_events);
  category_times->Preallocate(category_key, n_events);
  auto& prof_events = category_times->eo_times.at(category_key);
  std::vector<EOEvents> clib_eo_events;
  for (auto const& category : CATEGORIES_C_EVENTS) {
    clib_eo_events.push_back(category_times->eo_times[CategoryKey::FromCategory(process, category)]);
  }
  auto func = [per_pyprof_interception_us, &prof_events] (const EOEvents& events, size_t i) {
    TimePsec start_ps = events.EndPsec(i) + (per_pyprof_interception_us * PSEC_IN_USEC);
    TimePsec end_ps = start_ps + (per_pyprof_interception_us * PSEC_IN_USEC);
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
  double per_pyprof_annotation_overhead_us = _pyprof_overhead_json["mean_pyprof_annotation_overhead_per_call_us"];
  auto category_key = CategoryKey::FromCategory(process, CATEGORY_PROF_PYTHON_ANNOTATION);
  auto ops_key = CategoryKey::FromCategory(process, CATEGORY_EXTRA_OPERATION);
  auto const& ops_events = category_times->extra_eo_times.at(ops_key);
  assert(ops_events.size() > 0);
  category_times->Preallocate(category_key, ops_events.size());
  auto& prof_events = category_times->eo_times.at(category_key);
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

//MyStatus ReadEntireOverhead_CATEGORY_PROF_PYTHON_ANNOTATION(
//    const Machine& machine,
//    const Process& process,
//    const Phase& phase,
//    CategoryTimes* category_times,
//    CategoryTimesCount* count,
//    EntireTraceMeta* entire_meta) {
//  *entire_meta = EntireTraceMeta(machine, process, phase);
//  assert(!count->Contains(CATEGORY_PROF_PYTHON_ANNOTATION));
//  // TODO: we want to count the number of operation events BEFORE op-stack processing...
//  // Q: Is it equivalent to use the post-processed one?
//  // NO; this INCREASES the number of "operation end times" in the trace.
//  // maybe it's easier to just "walk through" the files again...?
//  auto n_events = count->Count(CATEGORY_OPERATION);
//  count->Add(CATEGORY_PROF_PYTHON_ANNOTATION, n_events);
//
//  return MyStatus::OK();
//}
//
//MyStatus RawTraceParser::ReadEntireOverhead_CATEGORY_PROF_PYTHON_ANNOTATION(
//    const Machine& machine,
//    const Process& process,
//    const Phase& phase,
//    CategoryTimes* category_times,
//    CategoryTimesCount* count,
//    EntireTraceMeta* entire_meta) {
//  CategoryTimesCount count;
//  if (_cupti_overhead_json != "") {
//    CategoryTimesCount count;
//    // Preallocate
//    // Insert: Python annotations
//    // PSEUDOCODE:
//    //     op_events = SQL:
//    //         select all from Event where e.category == CATEGORY_OPERATION
//    //     for op in op_events:
//    //         insert Event(
//    //             start=op.end,
//    //             duration=mean_pyprof_annotation_overhead_us,
//    //             category=CATEGORY_PROF_PYTHON_ANNOTATION)
//
//    // Preallocate: count[CATEGORY_PROF_PYTHON_ANNOTATION] = count[CATEGORY_PROF_PYTHON_ANNOTATION]
//    // Preallocate: count[CATEGORY_PROF_PYTHON_INTERCEPTION] = count[CATEGORIES_C_EVENTS]
//    // Preallocate: count[CATEGORY_PROF_CUPTI] = count[CATEGORY_CUDA_API_CPU] (upper bound)
//    // Preallocate: count[CATEGORY_PROF_LD_PRELOAD] = count[CATEGORY_CUDA_API_CPU]
//
//  }
//}

template <class ParserKlass, class ProtoKlass>
MyStatus GenericInitFromProto(ParserKlass& self, const ProtoKlass& proto) {
  self._machine = proto.machine_name();
  self._process = proto.process_name();
  self._phase = proto.phase();
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::_InitFromProto(const ProtoKlass& proto) {
  // TODO:
  // - count number of operation events.
  // - record their end times (convience: record both start/end times in EOEvents)
  return GenericInitFromProto(*this, proto);
}
MyStatus CUDAAPIStatsParser::_InitFromProto(const ProtoKlass& proto) {
  return GenericInitFromProto(*this, proto);
}
MyStatus CUDADeviceEventsParser::_InitFromProto(const ProtoKlass& proto) {
  return GenericInitFromProto(*this, proto);
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

void CategoryKey::Print(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "CategoryKey(procs=";

  PrintValue(out, this->procs);

  out << ", ";
  out << "ops=";
  PrintValue(out, this->ops);

  out << ", ";
  out << "non_ops=";
  PrintValue(out, this->non_ops);

  out << ")";
}

void CategoryKeyBitset::Print(std::ostream& out, int indent) const {
 // auto keys = idx_map.KeySetFrom(bitset);
  auto keys = idx_map->KeySetFrom(bitset);
  PrintIndent(out, indent);

  out << "CategoryKeyBitset: bits = " << bitset.to_string() << ", size = " << keys.size();
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

      DBG_LOG("{}", ss.str());
    }

    if ((index(min_cat) % 2) == 0 and min_time == times[min_cat][index(min_cat)+1]) {
      index(min_cat) += 2;
      continue;
    }

    auto time_chunk = min_time - last_time;
    if (last_time != min_time_value and time_chunk > 0) {
      // NOTE: std::map<Key, Number> defaults to 0 if the key doesn't exist.
      r.overlap[cur_cat] += time_chunk;
    }

    // Update current list of active categories.
    bool is_start = (index(min_cat) % 2 == 0);
    if (is_start) {
      cur_cat.Add(min_cat);
    } else {
      TimeUsec start_time_usec = times[min_cat][index(min_cat)-1];
      TimeUsec end_time_usec = min_time;
      r.meta.AddEvent(cur_cat, start_time_usec, end_time_usec);
      cur_cat.Remove(min_cat);
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
  }

  for (auto& pair : r.overlap) {
    // Change overlap from psec to usec.
    pair.second = pair.second / PSEC_IN_USEC;
  }

  return r;
}

void OverlapResult::Print(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "OverlapResult: size = " << overlap.size();
  size_t i = 0;
  for (const auto& pair : overlap) {
    auto const& bitset = pair.first;
    auto time_us = pair.second;
    double time_sec = ((double)time_us) / ((double)USEC_IN_SEC);

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Overlap[" << i << "]: duration = " << time_sec << " sec";
    out << "\n";
    bitset.Print(out, indent + 2);

    i += 1;
  }
}

DEFINE_PRINT_OPERATOR(CategoryKey)
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

