//
// Created by jagle on 11/13/2019.
//

#ifndef IML_TRACE_FILE_PARSER_H
#define IML_TRACE_FILE_PARSER_H

#include "iml_prof.pb.h"
#include "pyprof.pb.h"

#include <assert.h>

#include <boost/filesystem.hpp>

//#include "cuda_api_profiler/cupti_logging.h"

#include <memory>
#include <map>
#include <regex>
#include <fstream>
#include <list>

#include <sys/types.h>
//#include <sys/stat.h>

//#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "error_codes.pb.h"
//#include "tensorflow/core/lib/core/status.h"
#include "analysis/my_status.h"

#define PSEC_IN_USEC (1000)
#define USEC_IN_SEC (1000000)

#define IF_BAD_STATUS_RETURN(status)  \
      if (status.code() != MyStatus::OK().code()) { \
        return status; \
      }


namespace tensorflow {

template <typename T>
T StringToNumber(const std::string& s) {
  std::stringstream ss(s);
  T x = 0;
  ss >> x;
  return x;
}

enum RLSFileType {
  UNKNOWN_FILE = 0,
  CUDA_API_STATS_FILE = 1,
  CATEGORY_EVENTS_FILE = 2,
  CUDA_DEVICE_EVENTS_FILE = 3,
};
//const std::vector<RLSFileType>& AllRLSFileTypes() {
//  static const std::vector<RLSFileType> RLSFileTypeVector = std::vector<RLSFileType>{
//      CUDA_API_STATS_FILE ,
//      CATEGORY_EVENTS_FILE ,
//      CUDA_DEVICE_EVENTS_FILE ,
//  };
//  return RLSFileTypeVector;
//}


// Copied from: iml_profiler/parser/common.py
#define CATEGORY_TF_API "Framework API C"
#define CATEGORY_PYTHON "Python"
#define CATEGORY_PYTHON_PROFILER "Python profiler"
#define CATEGORY_CUDA_API_CPU "CUDA API CPU"
#define CATEGORY_UNKNOWN "Unknown"
#define CATEGORY_GPU "GPU"
#define CATEGORY_DUMMY_EVENT "Dummy event"
#define CATEGORY_OPERATION "Operation"
#define CATEGORY_SIMULATOR_CPP "Simulator C"


//#define TRACE_SUFFIX_RE (R"((?:\.trace_(?P<trace_id>\d+))?)")
#define TRACE_SUFFIX_RE R"((?:\.trace_(\d+))?)"

//#define CUDA_API_STATS_REGEX (R"(^cuda_api_stats{trace}\.proto)")
#define CUDA_API_STATS_REGEX (R"(^cuda_api_stats)" TRACE_SUFFIX_RE R"(\.proto)")

//#define CATEGORY_EVENTS_REGEX (R"(category_events{trace}\.proto)")
#define CATEGORY_EVENTS_REGEX (R"(category_events)" TRACE_SUFFIX_RE R"(\.proto)")

//#define CUDA_DEVICE_EVENTS_REGEX (R"(cuda_device_events{trace}\.proto)")
#define CUDA_DEVICE_EVENTS_REGEX (R"(cuda_device_events)" TRACE_SUFFIX_RE R"(\.proto)")

using Category = std::string;
using Machine = std::string;
using Process = std::string;
using Phase = std::string;
using TraceID = uint64_t;
//using CategoryTimes = std::map<Category, EOEvents>;

bool isRLSFileWithType(RLSFileType file_type, const std::string& path);
bool isRLSFile(const std::string& path);
RLSFileType GetRLSFileType(const std::string& path);

template <typename KeepFunc>
MyStatus RecursiveFindFiles(std::list<std::string>* paths, const std::string& root, KeepFunc func) {
    boost::filesystem::path root_path(root); //
  // https://rosettacode.org/wiki/Walk_a_directory/Recursively#C.2B.2B
  if (!boost::filesystem::is_directory(root_path)) {
    std::stringstream ss;
    ss << "Couldn't search recursively for files rooted at path=" << root << "; not a directory";
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  for (boost::filesystem::recursive_directory_iterator iter(root_path), end;
       iter != end;
       ++iter)
  {
    auto path = iter->path();
    if (func(path)) {
      paths->push_back(iter->path().string());
    }
  }
  return MyStatus::OK();
}

MyStatus FindRLSFiles(const std::string& iml_directory, std::list<std::string>* paths);

//enum RawTraceSplitLocationType {
//  START = 0
//  , END = 1
//};
//struct RawTraceSplitLocation {
//  // Is this the START or the END of a split?
//  RawTraceSplitLocationType location_type;
//};
//struct TraceSplit {
//  int64_t start_us;
//  int64_t end_us;
//  RawTraceSplitLocation start_location;
//  RawTraceSplitLocation end_location;
//};

#define EVENT_START_IDX(i) (2*i)
#define EVENT_END_IDX(i) (2*i + 1)
class EOEvents {
public:
  // Number of events
  size_t _n_events;
  size_t _next_event_to_set;
  // [e1.start, e1.end, e2.start, e2.end, ...]
  std::unique_ptr<int64_t[]> _events;

  EOEvents() :
      _n_events(0),
      _next_event_to_set(0) {
  }
  EOEvents(size_t n_events) :
      _n_events(n_events),
      _next_event_to_set(0),
      // 2*n_events: For each event, we need the (start, end) time.
      _events(new int64_t[2*n_events]) {
  }

  void Print(std::ostream& out, int indent) const;
  void PrintSummary(std::ostream& out, int indent) const;

  inline void SetEvent(size_t i, int64_t start_us, int64_t end_us) {
    assert(i < _n_events);
    assert(i == _next_event_to_set);
    assert(start_us <= end_us);
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    _events[start_idx] = start_us * PSEC_IN_USEC;
    _events[end_idx] = end_us * PSEC_IN_USEC;
    if (i > 0) {
      assert(_events[EVENT_END_IDX(i - 1)] <= _events[EVENT_START_IDX(i)]);
    }
    _next_event_to_set += 1;
  }

  inline int64_t DurationUsec(size_t i) const {
    assert(i < _n_events);
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    auto start_us = _events[start_idx] / PSEC_IN_USEC;
    auto end_us = _events[end_idx] / PSEC_IN_USEC;
    return end_us - start_us;
  }

  inline void AppendEvent(int64_t start_us, int64_t end_us) {
    SetEvent(_next_event_to_set, start_us, end_us);
  }

  inline size_t size() const {
    return _n_events;
  }

};

//bool IsFile(const std::string& path) {
//  struct stat info;
//  int ret = stat(path.c_str(), &info);
//  if (ret != 0)
//    return false;
//  // Q: What is they have a symlink to a directory?
//  return S_ISREG(info.st_mode) || S_ISLNK(info.st_mode);
//}

template <class ProtoType>
MyStatus ParseProto(const std::string& file_type, const std::string& path, ProtoType* proto) {
  boost::filesystem::path bpath(path);
  if (!boost::filesystem::is_regular_file(path)) {
    std::stringstream ss;
    ss << "Failed to read " << file_type << " file from path=" << path << "; didn't find a file at that location.";
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  std::fstream input(path, std::ios::in | std::ios::binary);
  if (!proto->ParseFromIstream(&input)) {
    std::stringstream ss;
    ss << "Failed to read " << file_type << " file from path=" << path << "; protobuf library failed to read the file";
    return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
  return MyStatus::OK();
}

class CategoryTimesCount {
public:
  std::map<Category, size_t> num_events;

  inline void Add(const Category& category, size_t n_events) {
    // NOTE: if num_events[category] will default to zero if its not inside the map.
    num_events[category] += n_events;
  }

  void _AddToCategoryTimes(const CategoryTimesCount& ctimes);

  friend CategoryTimesCount operator+(const CategoryTimesCount& left, const CategoryTimesCount& right);
  CategoryTimesCount& operator+=(const CategoryTimesCount& rhs);

};

class CategoryTimes {
public:
  std::map<Category, EOEvents> eo_times;

  CategoryTimes() = default;
  CategoryTimes(const CategoryTimesCount& count);
  inline size_t size() const {
    return eo_times.size();
  }

  void Print(std::ostream& out, int indent) const;
  void PrintSummary(std::ostream& out, int indent) const;
};

void PrintCategoryTimes(const CategoryTimes& category_times, std::ostream& out, int indent);

//class EOTimes {
//public:
//  CategoryTimes _category_times;
//};

// Navigate trace-files for a particular (machine, process, phase) in the --iml-directory in time-stamp order.
// Used by RawTraceParser for reading events into eo_times in the correct order.
struct TraceFileMeta {
  std::string path;

  CategoryTimesCount count;
  Machine machine;
  Process process;
  Phase phase;
  TraceID trace_id;
  RLSFileType file_type;

  bool initialized;

  TraceFileMeta() :
      trace_id(0),
      file_type(RLSFileType::UNKNOWN_FILE),
      initialized(false)
  {
  }
  TraceFileMeta(std::string const& path) :
      path(path),
      trace_id(0),
      file_type(GetRLSFileType(path)),
      initialized(false)
  {
    assert(isRLSFile(path));
  }

  inline const std::string& get_path() const {
    return path;
  }

  inline const Machine& get_machine() const {
    assert(initialized);
    return machine;
  }
  inline const Process& get_process() const {
    assert(initialized);
    return process;
  }
  inline const Phase& get_phase() const {
    assert(initialized);
    return phase;
  }
  inline RLSFileType get_file_type() const {
    assert(initialized);
    return file_type;
  }
  inline const TraceID& get_trace_id() const {
    assert(initialized);
    return trace_id;
  }
  inline const CategoryTimesCount& get_count() const {
    assert(initialized);
    return count;
  }

  MyStatus Init();

};
class TraceFileWalker {
public:
  // RLSFilePath -> TraceFileMeta
  std::map<std::string, TraceFileMeta> _path_to_meta;
  // [machine_name][process_name][phase_name][trace_id] -> TraceFileMeta
  std::map<Machine, std::map<Process, std::map<Phase, std::map<RLSFileType, std::map<TraceID, TraceFileMeta>>>>> _meta;
  std::string _iml_directory;

  TraceFileWalker(const std::string& iml_directory) :
      _iml_directory(iml_directory)
  {
  }

  MyStatus ReadMeta(const std::string& path, TraceFileMeta* meta);
  MyStatus TraceMetas(const Machine& machine, const Process& process, const Phase& phase, std::list<TraceFileMeta>* metas);

  std::list<Machine> Machines() const;
  std::list<Process> Processes(const Machine& machine) const;
  std::list<Phase> Phases(const Machine& machine, const Process& process) const;

  MyStatus Init();
};

class RawTraceParser {
public:
  std::string _iml_directory;
  TraceFileWalker _walker;

  RawTraceParser(const std::string& iml_directory) :
      _iml_directory(iml_directory),
      _walker(_iml_directory)
  {
  }
  // ASSUMPTION: all events i have category.events[i].start_us <= category.events[i+1].start_us
  // Q: What do we do if this assumption FAILS?
  // A: This assumption should hold with a thread...if it fails split by thread-id?
  //
  // Basic: Single eo_times:
  // - Create an array of int64 and read usec->psec for each category.
  // - Pass it to Python, OR, implement event overlap in C++.
  //
  // Advanced: Splitting event trace:
  // - obtain N splits of eo_times format with <= E events in total per split (across all categories).
  // - would be nice if we could iterate through each split quickly so we can divide up work between threads;
  //   for each split, provide precise information about START and END of split.
  //   Q: What determines the location of a split?
  //   e.g. location of the START of a split[split.start_us, split.end_us]:
  //   - split.end_us and split.start_us
  //   - Machine, process, phase
  //   - Which FILE (file-type, trace index)
  //     - category_events:
  //       - category -> [Event]
  //         START index into EACH category whose duration overlaps with split.start_us:
  //         ASSUMPTION: all events i have category.events[i].start_us <= category.events[i+1].start_us
  //         Q: What do we do if this assumption FAILS?
  //         A: This assumption should hold with a thread...if it fails split by thread-id?
  //         [ category.events[i] ]
  //            split.start_us
  //     - cuda_api_stats
  //       category="CUDA API call" -> [Events]
  //       Same as category_events (index into event list)
  //     - cuda_device_events
  //       category="GPU" -> [Events]
  //
  //
  //

  MyStatus Init();

  inline std::list<Machine> Machines() const {
    return _walker.Machines();
  }
  inline std::list<Process> Processes(const Machine& machine) const {
    return _walker.Processes(machine);
  }
  inline std::list<Phase> Phases(const Machine& machine, const Process& process) const {
    return _walker.Phases(machine, process);
  }

  MyStatus ReadEntireTrace(
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      CategoryTimes *category_times);

//  CategoryTimes RawTraceParser::ReadEntireTrace(const std::string& iml_directory) {
//  }
//  // E = each split should have <= E events across all categories.
//  std::list<TraceSplit> FindSplits(const std::string& iml_directory, size_t E);
//  // Read CategoryTimes belonging to this split.
//  CategoryTimes ReadSplit(const std::string& iml_directory, const TraceSplit& split);

};

class IEventFileParser {
public:

  // Implemented in IEventFileProtoParser<ProtoKlass>
  virtual bool IsFile(const std::string& path) const = 0;
  virtual MyStatus ReadFile(CategoryTimes* out_category_times) = 0;
  virtual MyStatus CountCategoryTimes(CategoryTimesCount* count) = 0;
  virtual MyStatus AppendCategoryTimes(CategoryTimes* out_category_times) = 0;
  virtual MyStatus Init() = 0;

  virtual const Machine& get_machine() const = 0;
  virtual const Process& get_process() const = 0;
  virtual const Phase& get_phase() const = 0;


};

template <class ProtoKlass>
class IEventFileProtoParser : public IEventFileParser {
public:
  std::string _path;
  RLSFileType _file_type;
  std::string _proto_nickname;

  Machine _machine;
  Process _process;
  Phase _phase;

  bool _initialized;

  IEventFileProtoParser(const std::string& path, RLSFileType file_type, const std::string& proto_nickname) :
      _path(path),
      _file_type(file_type),
      _proto_nickname(proto_nickname),
      _initialized(false)
  {
  }

  virtual const Machine& get_machine() const {
    assert(_initialized);
    return _machine;
  }
  virtual const Process& get_process() const {
    assert(_initialized);
    return _process;
  }
  virtual const Phase& get_phase() const {
    assert(_initialized);
    return _phase;
  }

  virtual bool IsFile(const std::string& path) const {
    return isRLSFileWithType(_file_type, path);
  }

  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) = 0;
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) = 0;

  virtual MyStatus _InitFromProto(const ProtoKlass& proto) = 0;

  virtual MyStatus Init() {
    if (_initialized) {
      return MyStatus::OK();
    }
    // Initialization happens in _ReadProto.
    MyStatus status = MyStatus::OK();
    ProtoKlass proto;
    status = _ReadProto(_path, &proto);
    IF_BAD_STATUS_RETURN(status);
    assert(_initialized);
    return MyStatus::OK();
  }

  MyStatus AppendCategoryTimes(CategoryTimes* out_category_times) {
    MyStatus status = MyStatus::OK();
    ProtoKlass proto;
    status = _ReadProto(_path, &proto);
    IF_BAD_STATUS_RETURN(status);
    return _AppendCategoryTimes(out_category_times, proto);
  }

  MyStatus _ReadProto(const std::string& path, ProtoKlass* proto) {
    MyStatus status = ParseProto(_proto_nickname, path, proto);
    IF_BAD_STATUS_RETURN(status);
    if (!_initialized) {
      _initialized;
      status = _InitFromProto(*proto);
      IF_BAD_STATUS_RETURN(status);
      _initialized = true;
    }
    return MyStatus::OK();
  }

  virtual MyStatus CountCategoryTimes(CategoryTimesCount* count) {
    MyStatus status = MyStatus::OK();
    ProtoKlass proto;
    status = _ReadProto(_path, &proto);
    IF_BAD_STATUS_RETURN(status);
    return _CountCategoryTimes(count, proto);
  }


  virtual MyStatus ReadFile(CategoryTimes* out_category_times) {
    MyStatus status = MyStatus::OK();

    ProtoKlass proto;
    status = this->_ReadProto(_path, &proto);
    IF_BAD_STATUS_RETURN(status);

    CategoryTimesCount count;
    status = this->_CountCategoryTimes(&count, proto);
    IF_BAD_STATUS_RETURN(status);

    *out_category_times = std::move(CategoryTimes(count));
    status = this->_AppendCategoryTimes(out_category_times, proto);
    IF_BAD_STATUS_RETURN(status);

    return MyStatus::OK();
  }
};


// GOAL:
//
// for each trace file:
//   - take path, add to counts (don't care about underlying parser).
//   AddEventFileCounts(path, &counts)
//
// CategoryTimes category_times(counts)
//
// for each trace file in trace_id order:
//   - append events from trace file to eo_times
//   AppendEventFile(path, &category_times)

bool isRLSFileWithType(RLSFileType file_type, const std::string& path);

MyStatus GetRLSEventParser(const std::string& path, std::unique_ptr<IEventFileParser>* parser);

class CategoryEventsParser : public IEventFileProtoParser<iml::CategoryEventsProto> {
public:
  using ProtoKlass = iml::CategoryEventsProto;
  CategoryEventsParser(const std::string& path) :
      IEventFileProtoParser<iml::CategoryEventsProto>(path, RLSFileType::CATEGORY_EVENTS_FILE, "category_events")
  {
  }


  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) override;
  virtual MyStatus _InitFromProto(const ProtoKlass& proto) override;

  MyStatus _AppendCategory(const Category& category, const ProtoKlass& proto, EOEvents* eo_events);

  // PROBLEM:
  // - we need to determine ahead of time how many events we need to read so we can preallocate an array.
  // - to do this, we need to load each protobuf file.
  // - our "ReadCategory" function should instead become "AppendCategory", and we should return an error or Assert if its too big.
};

class CUDAAPIStatsParser : public IEventFileProtoParser<iml::CUDAAPIPhaseStatsProto> {
public:
  using ProtoKlass = iml::CUDAAPIPhaseStatsProto;
  CUDAAPIStatsParser(const std::string& path) :
      IEventFileProtoParser<ProtoKlass>(path, RLSFileType::CUDA_API_STATS_FILE, "cuda_api_stats")
  {
  }


  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) override;
  virtual MyStatus _InitFromProto(const ProtoKlass& proto) override;
};

class CUDADeviceEventsParser : public IEventFileProtoParser<iml::MachineDevsEventsProto> {
public:
  using ProtoKlass = iml::MachineDevsEventsProto;
  CUDADeviceEventsParser(const std::string& path) :
      IEventFileProtoParser<ProtoKlass>(path, RLSFileType::CUDA_DEVICE_EVENTS_FILE, "cuda_device_events")
  {
  }

  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) override;
  virtual MyStatus _InitFromProto(const ProtoKlass& proto) override;
};

MyStatus GetTraceID(const std::string& path, TraceID* trace_id);


}

#endif //IML_TRACE_FILE_PARSER_H
