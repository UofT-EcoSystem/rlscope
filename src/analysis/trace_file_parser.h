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

#define IF_BAD_STATUS_RETURN(status)  \
      if (status.code() != MyStatus::OK().code()) { \
        return status; \
      }


namespace tensorflow {

enum RLSFileType {
  UNKNOWN_FILE = 0,
  CUDA_API_STATS_FILE = 1,
  CATEGORY_EVENTS_FILE = 2,
  CUDA_DEVICE_EVENTS_FILE = 3,
};


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
//using CategoryTimes = std::map<Category, EOEvents>;

bool isRLSFile(RLSFileType file_type, const std::string& path);
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
};

class CategoryTimes {
public:
  std::map<Category, EOEvents> eo_times;

  CategoryTimes() = default;
  CategoryTimes(const CategoryTimesCount& count);
  inline size_t size() const {
    return eo_times.size();
  }
};

void PrintCategoryTimes(const CategoryTimes& category_times, std::ostream& out, int indent);

//class EOTimes {
//public:
//  CategoryTimes _category_times;
//};

class RawTraceParser {
public:
  RawTraceParser();
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
//  CategoryTimes ReadEntireTrace(const std::string& iml_directory);

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
  virtual MyStatus ReadFile(const std::string& path, CategoryTimes* out_category_times) = 0;
  virtual MyStatus CountCategoryTimes(const std::string& path, CategoryTimesCount* count) = 0;
  virtual MyStatus AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times) = 0;

};

template <class ProtoKlass>
class IEventFileProtoParser : public IEventFileParser {
public:
  RLSFileType _file_type;
  std::string _proto_nickname;
  IEventFileProtoParser(RLSFileType file_type, const std::string& proto_nickname) :
      _file_type(file_type),
      _proto_nickname(proto_nickname) {
  }

  virtual bool IsFile(const std::string& path) const {
    return isRLSFile(_file_type, path);
  }

  virtual MyStatus _CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const ProtoKlass& proto) = 0;
  virtual MyStatus _AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const ProtoKlass& proto) = 0;

  MyStatus AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times) {
    MyStatus status = MyStatus::OK();
    ProtoKlass proto;
    status = _ReadProto(path, &proto);
    IF_BAD_STATUS_RETURN(status);
    return _AppendCategoryTimes(path, out_category_times, proto);
  }


  MyStatus _ReadProto(const std::string& path, ProtoKlass* proto) {
    MyStatus status = ParseProto(_proto_nickname, path, proto);
    IF_BAD_STATUS_RETURN(status);
    return MyStatus::OK();
  }

  virtual MyStatus CountCategoryTimes(const std::string& path, CategoryTimesCount* count) {
    MyStatus status = MyStatus::OK();
    ProtoKlass proto;
    status = _ReadProto(path, &proto);
    IF_BAD_STATUS_RETURN(status);
    return _CountCategoryTimes(path, count, proto);
  }


  virtual MyStatus ReadFile(const std::string& path, CategoryTimes* out_category_times) {
    MyStatus status = MyStatus::OK();

    ProtoKlass proto;
    status = this->_ReadProto(path, &proto);
    IF_BAD_STATUS_RETURN(status);

    CategoryTimesCount count;
    status = this->_CountCategoryTimes(path, &count, proto);
    IF_BAD_STATUS_RETURN(status);

    *out_category_times = std::move(CategoryTimes(count));
    status = this->_AppendCategoryTimes(path, out_category_times, proto);
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

bool isRLSFile(RLSFileType file_type, const std::string& path);

MyStatus GetRLSEventParser(const std::string& path, std::unique_ptr<IEventFileParser>* parser);

class CategoryEventsParser : public IEventFileProtoParser<iml::CategoryEventsProto> {
public:
  CategoryEventsParser() :
      IEventFileProtoParser<iml::CategoryEventsProto>(RLSFileType::CATEGORY_EVENTS_FILE, "category_events")
  {
  }


  virtual MyStatus _CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const iml::CategoryEventsProto& proto) override;
  virtual MyStatus _AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const iml::CategoryEventsProto& proto) override;

  MyStatus _AppendCategory(const Category& category, const iml::CategoryEventsProto& proto, EOEvents* eo_events);

  // PROBLEM:
  // - we need to determine ahead of time how many events we need to read so we can preallocate an array.
  // - to do this, we need to load each protobuf file.
  // - our "ReadCategory" function should instead become "AppendCategory", and we should return an error or Assert if its too big.
};

class CUDAAPIStatsParser : public IEventFileProtoParser<iml::CUDAAPIPhaseStatsProto> {
public:
  using ProtoKlass = iml::CUDAAPIPhaseStatsProto;
  CUDAAPIStatsParser() :
      IEventFileProtoParser<ProtoKlass>(RLSFileType::CUDA_API_STATS_FILE, "cuda_api_stats")
  {
  }


  virtual MyStatus _CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const ProtoKlass& proto) override;
};

class CUDADeviceEventsParser : public IEventFileProtoParser<iml::MachineDevsEventsProto> {
public:
  using ProtoKlass = iml::MachineDevsEventsProto;
  CUDADeviceEventsParser() :
      IEventFileProtoParser<ProtoKlass>(RLSFileType::CUDA_DEVICE_EVENTS_FILE, "cuda_device_events")
  {
  }

  virtual MyStatus _CountCategoryTimes(const std::string& path, CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times, const ProtoKlass& proto) override;
};

std::list<std::unique_ptr<IEventFileParser>> AllRLSParsers();


}

#endif //IML_TRACE_FILE_PARSER_H
