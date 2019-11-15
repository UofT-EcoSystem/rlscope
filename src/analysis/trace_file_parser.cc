//
// Created by jagle on 11/13/2019.
//

#include "analysis/trace_file_parser.h"
#include "cuda_api_profiler/generic_logging.h"

#include <boost/compute/algorithm/reduce.hpp>

#include <assert.h>
#include <numeric>
#include <iostream>
#include <spdlog/spdlog.h>

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
void EOEvents::PrintSummary(std::ostream& out, int indent) const {
  double total_sec = 0;
  for (size_t i = 0; i < _n_events; i++) {
    total_sec += ((double)this->DurationUsec(i)) / ((double)USEC_IN_SEC);
  }
  PrintIndent(out, indent);
  out << "EOEvents: size = " << _n_events << ", duration = " << total_sec << " sec";
}

MyStatus CategoryEventsParser::_CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) {
  for (const auto& pair : proto.category_events()) {
    const auto& category = pair.first;
    size_t n_events = pair.second.events().size();
    count->Add(category, n_events);
  }
  return MyStatus::OK();
}
MyStatus CategoryEventsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) {
  MyStatus status = MyStatus::OK();
  for (const auto& pair : proto.category_events()) {
    const auto& category = pair.first;
    EOEvents& eo_events = out_category_times->eo_times.at(category);
    status = _AppendCategory(category, proto, &eo_events);
    IF_BAD_STATUS_RETURN(status);
  }
  return MyStatus::OK();
}
MyStatus CategoryEventsParser::_AppendCategory(const Category& category, const ProtoKlass& proto, EOEvents* eo_events) {
  const auto& events = proto.category_events().at(category).events();
  size_t n_events = events.size();
  for (const auto& event : proto.category_events().at(category).events()) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    eo_events->AppendEvent(start_us, end_us);
  }
  return MyStatus::OK();
}

MyStatus CUDAAPIStatsParser::_CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) {
  const std::string category = CATEGORY_CUDA_API_CPU;
  size_t n_events = proto.events().size();
  count->Add(category, n_events);
  return MyStatus::OK();
}
MyStatus CUDAAPIStatsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) {
  const std::string category = CATEGORY_CUDA_API_CPU;
  EOEvents& eo_events = out_category_times->eo_times.at(category);
  for (const auto& event : proto.events()) {
    auto start_us = event.start_time_us();
    auto end_us = event.start_time_us() + event.duration_us();
    eo_events.AppendEvent(start_us, end_us);
  }
  return MyStatus::OK();
}

MyStatus CUDADeviceEventsParser::_CountCategoryTimes(CategoryTimesCount* count, const CUDADeviceEventsParser::ProtoKlass& proto) {
  const std::string category = CATEGORY_GPU;
  for (const auto& dev_events_pair : proto.dev_events()) {
    const auto& dev = dev_events_pair.first;
    const auto& events = dev_events_pair.second.events();
    size_t n_events = events.size();
    count->Add(category, n_events);
  }
  return MyStatus::OK();
}
MyStatus CUDADeviceEventsParser::_AppendCategoryTimes(CategoryTimes* out_category_times, const CUDADeviceEventsParser::ProtoKlass& proto) {
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
void CategoryTimes::Print(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "CategoryTimes: size = " << this->size();
  size_t category_idx = 0;
  for (const auto& pair : this->eo_times) {
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
void CategoryTimes::PrintSummary(std::ostream& out, int indent) const {
  PrintIndent(out, indent);
  out << "CategoryTimes: size = " << this->size();
  size_t category_idx = 0;
  for (const auto& pair : this->eo_times) {
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

#undef IF_MATCH_RETURN_TYPE
}

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
  for (auto const& pair : ctimes.num_events) {
    auto const& category = pair.first;
    auto n_events = pair.second;
    this->num_events[category] += n_events;
  }
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
    SPDLOG_DEBUG("TraceFileWalker saw path={}", path);
    TraceFileMeta meta;
    // Cache meta-data for trace-file.
    status = ReadMeta(path, &meta);
    IF_BAD_STATUS_RETURN(status);
  }
  return MyStatus::OK();
}

MyStatus RawTraceParser::Init() {
  MyStatus status = MyStatus::OK();
  status = _walker.Init();
  IF_BAD_STATUS_RETURN(status);
  return MyStatus::OK();
}

MyStatus RawTraceParser::ReadEntireTrace(
    const Machine& machine,
    const Process& process,
    const Phase& phase,
    CategoryTimes *category_times) {
  MyStatus status = MyStatus::OK();
  std::list<TraceFileMeta> metas;
  status = _walker.TraceMetas(machine, process, phase, &metas);
  IF_BAD_STATUS_RETURN(status);

  CategoryTimesCount count;
  for (auto const& meta : metas) {
    count += meta.get_count();
  }

  // Preallocate space for eo_times for this (machine, process, phase).
  *category_times = std::move(CategoryTimes(count));
  for (auto const& meta : metas) {
    std::unique_ptr<IEventFileParser> parser;
    SPDLOG_DEBUG("read path = {}", meta.get_path());
    status = GetRLSEventParser(meta.get_path(), &parser);
    IF_BAD_STATUS_RETURN(status);
    // TODO: cache read proto-files to avoid re-reading 20MB files...maybe.
    status = parser->AppendCategoryTimes(category_times);
    IF_BAD_STATUS_RETURN(status);
  }
  return MyStatus::OK();
}

template <class ParserKlass, class ProtoKlass>
MyStatus GenericInitFromProto(ParserKlass& self, const ProtoKlass& proto) {
  self._machine = proto.machine_name();
  self._process = proto.process_name();
  self._phase = proto.phase();
  return MyStatus::OK();
}

MyStatus CategoryEventsParser::_InitFromProto(const ProtoKlass& proto) {
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

} // namespace tensorflow

