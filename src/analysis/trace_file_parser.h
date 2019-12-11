//
// Created by jagle on 11/13/2019.
//

#ifndef IML_TRACE_FILE_PARSER_H
#define IML_TRACE_FILE_PARSER_H

#include "iml_prof.pb.h"
#include "pyprof.pb.h"

#include <Eigen/Dense>

#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/defines.h"

#include <assert.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <boost/filesystem.hpp>

// NOTE: not until gcc 7.1 (7.4.0 in Ubuntu 18.04); use boost::optional instead.
// #include <optional>
#include <boost/optional.hpp>
#include <boost/utility/string_view.hpp>

#include <nlohmann/json.hpp>
// using json = nlohmann::json;

#include <iostream>
#include <algorithm>
//#include <string_view>

//#include "cuda_api_profiler/cupti_logging.h"

#include <memory>
#include <map>
#include <regex>
#include <fstream>
#include <list>
#include <map>

#include <sys/types.h>
//#include <sys/stat.h>

//#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "error_codes.pb.h"
//#include "tensorflow/core/lib/core/status.h"
#include "analysis/my_status.h"

#include <set>
#include <cuda_api_profiler/debug_flags.h>

//#define PSEC_IN_USEC (1000)
//#define USEC_IN_SEC (1000000)

#define IF_BAD_STATUS_RETURN(status)  \
      if (status.code() != MyStatus::OK().code()) { \
        return status; \
      }

#define DEFINE_PRINT_OPERATOR(Klass) \
  std::ostream& operator<<(std::ostream& os, const Klass& obj) { \
    obj.Print(os, 0); \
    return os; \
  }

#define DECLARE_PRINT_OPERATOR(Klass) \
  friend std::ostream& operator<<(std::ostream& os, const Klass& obj);


#define DECLARE_PRINT_DEBUG \
  void Pr() const;

#define DEFINE_PRINT_DEBUG(Klass) \
  void Klass::Pr() const { \
    this->Print(std::cout, 0); \
    std::cout.flush(); \
  }

#define DEFINE_PRINT_DEBUG_HEADER \
  void Pr() const { \
    this->Print(std::cout, 0); \
    std::cout.flush(); \
  }


#define DECLARE_GET_PARSER_META \
  ParserMeta* GetParserMetaDerived() const;

namespace tensorflow {


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

#define CATEGORY_EXTRA_OPERATION "Operation: original"
#define CATEGORY_TOTAL "Total"
// Not a category used during tracing;
// represents a group of categories.
#define CATEGORY_CPU "CPU"

#define CATEGORY_PROF_CUPTI "Profiling: CUPTI"
#define CATEGORY_PROF_LD_PRELOAD "Profiling: LD_PRELOAD"
#define CATEGORY_PROF_PYTHON_ANNOTATION "Profiling: Python annotation"
#define CATEGORY_PROF_PYTHON_INTERCEPTION "Profiling: Python interception"
//CATEGORIES_PROF = {
//    CATEGORY_PROF_CUPTI,
//    CATEGORY_PROF_LD_PRELOAD,
//    CATEGORY_PROF_PYTHON_ANNOTATION,
//    CATEGORY_PROF_PYTHON_INTERCEPTION,
//}


//#define TRACE_SUFFIX_RE (R"((?:\.trace_(?P<trace_id>\d+))?)")
#define TRACE_SUFFIX_RE R"((?:\.trace_(\d+))?)"

//#define CUDA_API_STATS_REGEX (R"(^cuda_api_stats{trace}\.proto)")
#define CUDA_API_STATS_REGEX (R"(^cuda_api_stats)" TRACE_SUFFIX_RE R"(\.proto)")

//#define CATEGORY_EVENTS_REGEX (R"(category_events{trace}\.proto)")
#define CATEGORY_EVENTS_REGEX (R"(category_events)" TRACE_SUFFIX_RE R"(\.proto)")

//#define CUDA_DEVICE_EVENTS_REGEX (R"(cuda_device_events{trace}\.proto)")
#define CUDA_DEVICE_EVENTS_REGEX (R"(cuda_device_events)" TRACE_SUFFIX_RE R"(\.proto)")

// NOT supported until C++17 Ubuntu 18.04
//using OptionalString = boost::optional<std::string_view>;
// Doesn't allow automatic conversion of std::string arguments.
// using OptionalString = boost::optional<boost::string_view>;
// Works?
using OptionalString = boost::optional<const std::string&>;
// Doesn't allow automatic conversion of std::string arguments.
// using OptionalString = boost::optional<const std::string*>;
using Category = std::string;
using OverlapType = std::string;
using EventNameID = uint32_t;
using Machine = std::string;
using Process = std::string;
using Operation = std::string;
using EventName = std::string;
using Phase = std::string;
using TraceID = uint64_t;
using TimeUsec = int64_t;
using TimePsec = int64_t;
#define MAX_CATEGORY_KEYS 64
//using CategoryTimes = std::map<Category, EOEvents>;
using OverlapMap = std::map<std::set<std::string>, TimeUsec>;
using Metadata = nlohmann::json;

extern const std::set<Category> CATEGORIES_C_EVENTS;
extern const std::set<Category> CATEGORIES_PROF;
extern const std::set<Category> CATEGORIES_CPU;
extern const std::set<Category> CATEGORIES_GPU;

extern const std::set<OverlapType> OVERLAP_TYPES;

template <typename OStream>
void _AddSuffix(OStream& ss, std::string name, std::string key, const Metadata& md) {
  if (md.find(key) != md.end()) {
    std::string value = md[key];
    ss << "." << name << "_" << value;
  }
}

template <typename OStream>
void _AddSuffixList(OStream& ss, std::string name, std::string key, const Metadata& md) {
  if (md.find(key) != md.end()) {
    ss << "." << name;
    for (auto const& js_value : md[key]) {
      std::string value = js_value;
      ss << "_" << value;
    }
  }
}

template <typename OStream>
void AddPhaseSuffix(OStream& ss, const Metadata& md) {
  _AddSuffix(ss, "phase", "phase", md);
}

template <typename OStream>
void AddMachineSuffix(OStream& ss, const Metadata& md) {
  _AddSuffix(ss, "machine", "machine", md);
}

template <typename OStream>
void AddOverlapTitle(OStream& ss, const Metadata& md) {
  std::string overlap_type = md["overlap_type"];
  ss << overlap_type;
}

template <typename OStream>
void AddProcessSuffix(OStream& ss, const Metadata& md) {
  _AddSuffix(ss, "process", "process", md);
}

template <typename OStream>
void AddResourcesSuffix(OStream& ss, const Metadata& md) {
  _AddSuffixList(ss, "resources", "resource_overlap", md);
}

template <typename OStream>
void AddOpsSuffix(OStream& ss, const Metadata& md) {
  _AddSuffixList(ss, "ops", "operation", md);
}

template <typename T>
T StringToNumber(const std::string& s) {
  std::stringstream ss(s);
  T x = 1;
  ss >> x;
  return x;
}

template <typename Number, typename K>
struct IdxMaps {
  std::map<K, Number> to_idx;
  std::map<Number, K> from_idx;

  bool debug{false};

  void Print(std::ostream& out, int indent) const {
    PrintIndent(out, indent);
    out << "IdxMaps: size = " << to_idx.size();
    for (const auto& pair : to_idx) {
      auto const& key = pair.first;
      auto const idx = pair.second;
      auto time_us = pair.second;
      double time_sec = ((double)time_us) / ((double)USEC_IN_SEC);

      out << "\n";
      PrintIndent(out, indent + 1);
      out << "Idx: [" << idx << "] -> Key: [ " << key << " ]";
    }
  }
  DEFINE_PRINT_DEBUG_HEADER

  template <typename Num, typename Key>
  friend std::ostream& operator<<(std::ostream& os, const IdxMaps<Num, Key>& obj);

  inline Number Idx(const K& k) const {
    if (to_idx.find(k) == to_idx.end()) {
      std::stringstream ss;
      ss << "\n";
      this->Print(ss, 0);
      ss << "\n";
      k.Print(ss, 0);
      DBG_LOG("{}", ss.str());
      assert(false);
    }
    return to_idx.at(k);
  }

  inline K Key(Number idx) const {
    return from_idx.at(idx);
  }

  template <std::size_t N>
  std::set<K> KeySetFrom(const std::bitset<N>& bitset) const {
    std::set<K> keys;

    assert(N >= to_idx.size());
    for (Number idx = 0; idx < to_idx.size(); idx++) {
      if (debug && SHOULD_DEBUG(FEATURE_OVERLAP)) {
        DBG_LOG("Trying idx = {}", idx);
      }
      if (bitset[idx]) {
        auto const& key = Key(idx);
        if (debug && SHOULD_DEBUG(FEATURE_OVERLAP)) {
          std::stringstream ss;
          ss << key;
          DBG_LOG("Insert idx={}, key = {}", idx, ss.str());
        }
        keys.insert(Key(idx));
      }
    }

    return keys;
  }

  template <std::size_t N>
  std::set<Number> IndexSetFrom(const std::bitset<N>& bitset) const {
    std::set<Number> indices;

    assert(N >= to_idx.size());
    for (Number idx = 0; idx < to_idx.size(); idx++) {
      if (bitset[idx]) {
        indices.insert(idx);
      }
    }

    return indices;
  }

  template <typename Iterable>
  static IdxMaps From(const Iterable& xs) {
    IdxMaps m;
    Number i = 0;
    for (const auto& x : xs) {
      m.to_idx[x] = i;
      m.from_idx[i] = x;
      i += 1;
    }
    return m;
  }
};

template <typename Number, typename K>
std::ostream& operator<<(std::ostream& os, const IdxMaps<Number, K>& obj) {
  obj.Print(os, 0);
  return os;
}

template <typename Number, class Key>
class KeyVector {
  public:
  std::vector<Number> ids;
  Number next_id;
  using IdToKey = std::map<Number, Key>;
  IdToKey id_to_key;
  using KeyToId = std::map<Key, Number>;
  KeyToId key_to_id;
  std::set<Key> keys;

  KeyVector() :
      next_id(0)
  {
  }

  inline void reserve(size_t n) {
    ids.reserve(n);
  }

  inline size_t size() const {
    return ids.size();
  }

  inline void push_back(const Key& key) {

    // https://stackoverflow.com/questions/97050/stdmap-insert-or-stdmap-find
    Number id;
    auto lb = key_to_id.lower_bound(key);
    if (lb != key_to_id.end() && !(key_to_id.key_comp()(key, lb->first))) {
        // key already exists
        // update lb->second if you care to
        id = lb->second;
    } else {
        // The key does not exist in the map; add it to the map
        id = next_id;
        next_id += 1;
        // Use lb as a hint to insert, so it can avoid another lookup.
        key_to_id.insert(lb, typename KeyToId::value_type(key, id));
        id_to_key[id] = key;
        keys.insert(key);
        assert(keys.size() < std::numeric_limits<Number>::max());
    }
    ids.push_back(id);

  }

  inline const Key& operator[](int index) const {
    assert(static_cast<size_t>(index) < ids.size());
    auto id = ids[index];
    auto const& key = id_to_key.at(id);
    return key;
  }

  inline Number ID(int index) const {
    assert(static_cast<size_t>(index) < ids.size());
    auto id = ids[index];
    return id;
  }

  inline Number AsID(const Key& key) const {
    auto id = key_to_id.at(key);
    return id;
  }

  inline const Key& AsKey(Number id) const {
    return id_to_key.at(id);
  }

  inline Number HasID(const Key& key) const {
    return key_to_id.find(key) != key_to_id.end();
  }

//  inline const Key& Key(size_t i) {
//    assert(i < ids.size());
//    auto number = ids[i];
//    auto it = id_to_key.find(number);
//    assert(it != id_to_key.end());
//    return it->second;
//  }

  inline const std::set<Key>& Keys() {
    return keys;
  }

};

class CategoryKey {
public:
  std::set<Process> procs;
  std::set<Operation> ops;
  std::set<Category> non_ops;

  CategoryKey() = default;

  CategoryKey(
      std::set<Process> procs_,
      std::set<Operation> ops_,
      std::set<Category> non_ops_) :
      procs(std::move(procs_)),
      ops(std::move(ops_)),
      non_ops(std::move(non_ops_)) {
  }


  static CategoryKey FromCategory(const Process& proc, const Category& category) {
    CategoryKey category_key;
    category_key.procs.insert(proc);
    category_key.non_ops.insert(category);
    return category_key;
  }

  static CategoryKey FromOpEvent(const Process& proc, const Operation& op) {
    CategoryKey category_key;
    category_key.procs.insert(proc);
    category_key.ops.insert(op);
    return category_key;
  }

  bool operator<(const CategoryKey& rhs) const {
    auto const& lhs = *this;
    // https://en.cppreference.com/w/cpp/utility/tuple/operator_cmp
    // Here's how you implement operator< for two tuples lhs and rhs.
    //   (bool)(std::get<0>(lhs) < std::get<0>(rhs)) || (!(bool)(std::get<0>(rhs) < std::get<0>(lhs)) && lhstail < rhstail),
    //
    // Either, the left-most element of lhs is < the right-most element of rhs, OR
    // its NOT the case that the left-most element of rhs is less than the left-most element of lhs (that would make rhs < lhs)
    // AND the lhstail < rhstail (we evaluate the remaining elements if the first element of lhs and rhs are equal)
    // return std::make_tuple(lhs.procs, lhs.ops, lhs.non_ops) <
    //        std::make_tuple(rhs.procs, rhs.ops, rhs.non_ops);
    return std::tie(lhs.procs, lhs.ops, lhs.non_ops) <
           std::tie(rhs.procs, rhs.ops, rhs.non_ops);
  }

  bool operator==(const CategoryKey& rhs) const {
    auto const& lhs = *this;
    return (lhs.procs == rhs.procs)
           && (lhs.ops == rhs.ops)
           && (lhs.non_ops == rhs.non_ops);
  }

  DECLARE_PRINT_DEBUG
  // DECLARE_PRINT_OPERATOR(CategoryKey)

  template <typename OStream>
  void Print(OStream& out, int indent) const {
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


  template <typename OStream>
  friend OStream &operator<<(OStream &os, const CategoryKey &obj)
  {
    obj.Print(os, 0);
    return os;
  }

};
using CategoryIdxMap = IdxMaps<size_t, CategoryKey>;

class CategoryKeyBitset {
public:
  bool debug{false};
  std::bitset<MAX_CATEGORY_KEYS> bitset;
  // Keep this for debugging purposes, so we can convert a bitset into a string.
  std::shared_ptr<const CategoryIdxMap> idx_map;
  CategoryKeyBitset() : idx_map(nullptr) {
  }
  CategoryKeyBitset(size_t category_idx, std::shared_ptr<const CategoryIdxMap> idx_map) :
      bitset(1 << category_idx),
      idx_map(idx_map) {
    // Each CategoryKey is a unique bit in the bitset.
    // This allows us to do set operations.
    assert(category_idx < MAX_CATEGORY_KEYS);
  }
  CategoryKeyBitset(std::shared_ptr<const CategoryIdxMap> idx_map) :
      idx_map(idx_map) {
    // Empty set.
  }

  std::set<CategoryKey> Keys() const;
  std::set<size_t> Indices() const;

  inline void Add(size_t category_idx) {
    assert(category_idx < MAX_CATEGORY_KEYS);
    bitset[category_idx] = 1;
  }

  inline void Remove(size_t category_idx) {
    assert(category_idx < MAX_CATEGORY_KEYS);
    bitset[category_idx] = 0;
  }

  inline bool IsEmpty() const {
    return bitset.count() == 0;
    // return bitset.to_ullong() == 0;
  }

  static CategoryKeyBitset EmptySet(std::shared_ptr<const CategoryIdxMap> idx_map);

  bool operator<(const CategoryKeyBitset& rhs) const {
    auto const& lhs = *this;

    static_assert(
        MAX_CATEGORY_KEYS <= 8*sizeof(unsigned long long),
        "We assume < 64 CategoryKey's, if you need more, then change CategoryKeyBitset::operator< to support arbitrary length bitsets (needed for std::map)");
    return lhs.bitset.to_ullong() < rhs.bitset.to_ullong();

    // Arbitrary length bitsets:
    // return BitsetLessThan(lhs.bitset, rhs.bitset);

  }

  bool operator==(const CategoryKeyBitset& rhs) const {
    auto const& lhs = *this;
    return lhs.bitset == rhs.bitset;
  }

  void Print(std::ostream& out, int indent) const;
  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(CategoryKeyBitset)

  // NOTE: Use this for operator< for arbitrary length bitsets.
  // https://stackoverflow.com/questions/21245139/fastest-way-to-compare-bitsets-operator-on-bitsets
  //
  // template <std::size_t N>
  // inline bool BitsetLessThan(const std::bitset<N>& x, const std::bitset<N>& y)
  // {
  //     for (int i = N-1; i >= 0; i--) {
  //         if (x[i] ^ y[i]) return y[i];
  //     }
  //     return false;
  // }
};
using Overlap = std::map<CategoryKeyBitset, TimeUsec>;

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
  size_t _max_events;
  // size_t _n_events;
  size_t _next_event_to_set;
  // [e1.start, e1.end, e2.start, e2.end, ...]

  // NOTE: operator[] isn't supported for shared_ptr<T[]>...
  // TODO: if this slows stuff down, just make our own array template that avoids double dereference.
  // https://stackoverflow.com/questions/8947579/why-isnt-there-a-stdshared-ptrt-specialisation
  std::shared_ptr<std::vector<TimeUsec>> _events;

  // Assumption:
  // The number of unique event-names belonging to a single category is < 2^32.
  // NOTE: It's not clear to me what a reasonable value to use here really is...
  KeyVector<EventNameID, std::string> _names;
  bool _keep_names;

  EOEvents() :
      _max_events(0),
      _next_event_to_set(0),
      _events(new std::vector<TimeUsec>()),
      _keep_names(false)
  {
  }
  EOEvents(size_t n_events, bool keep_names) :
      _max_events(n_events)
      , _next_event_to_set(0)
      // 2*n_events: For each event, we need the (start, end) time.
      // , _events(new std::vector<TimeUsec>(2*n_events))
      , _events(new std::vector<TimeUsec>())
      , _keep_names(keep_names)
  {
    _events->reserve(2*n_events);
  }

  inline bool KeepNames() const {
    return _keep_names;
  }

  // Return read-only raw pointer to array of times.
  // NOTE: be careful, since the lifetime of this pointer is still managed by std::shared_ptr!
  inline const TimeUsec* RawPtr() const {
    // NOTE: if you get a segfault here, it's because you've default-constructed EOEvents() instead of calling
    // EOEvents(size_t n_events).
    // Perhaps you accidentally inserted a map key when calling eo_times[key]?
    return _events->data();
  }

  void Print(std::ostream& out, int indent) const;
  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(EOEvents)
  void PrintSummary(std::ostream& out, int indent) const;

  inline void SetEvent(OptionalString name, size_t i, TimeUsec start_us, TimeUsec end_us) {
    SetEventPsec(name, i, start_us*PSEC_IN_USEC, end_us*PSEC_IN_USEC);
  }

  inline void SetEventPsec(OptionalString name, size_t i, TimePsec start_ps, TimePsec end_ps) {
    assert(i < _max_events);
    assert(i == _next_event_to_set);
    assert(start_ps <= end_ps);
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    if (_keep_names) {
      assert(name.has_value());
      _names.push_back(name.value());
    }
    (*_events)[start_idx] = start_ps;
    (*_events)[end_idx] = end_ps;
    if (i > 0) {
      assert((*_events)[EVENT_END_IDX(i - 1)] <= (*_events)[EVENT_START_IDX(i)]);
    }
    _next_event_to_set += 1;
  }

  inline TimeUsec DurationUsec(size_t i) const {
    assert(i < _max_events);
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    auto start_us = (*_events)[start_idx] / PSEC_IN_USEC;
    auto end_us = (*_events)[end_idx] / PSEC_IN_USEC;
    return end_us - start_us;
  }

  inline TimeUsec StartUsec(size_t i) const {
    return StartPsec(i)/PSEC_IN_USEC;
  }

  inline TimePsec StartPsec(size_t i) const {
    assert(i < _max_events);
    auto start_idx = EVENT_START_IDX(i);
    auto start_ps = (*_events)[start_idx];
    return start_ps;
  }

  inline TimePsec EndPsec(size_t i) const {
    assert(i < _max_events);
    auto end_idx = EVENT_END_IDX(i);
    auto end_ps = (*_events)[end_idx];
    return end_ps;
  }

  inline const std::string& GetEventName(size_t i) const {
    assert(i < _max_events);
    assert(_keep_names);
    return _names[i];
  }

  inline EventNameID GetEventNameID(size_t i) const {
    assert(i < _max_events);
    return _names.ID(i);
  }

  inline EventNameID AsEventID(const std::string& name) const {
    return _names.AsID(name);
  }

  inline const EventName& AsEventName(EventNameID id) const {
    return _names.AsKey(id);
  }

  inline bool HasEventID(const std::string& name) const {
    return _names.HasID(name);
  }

  inline void AppendEvent(OptionalString name, TimeUsec start_us, TimeUsec end_us) {
    SetEvent(name, _next_event_to_set, start_us, end_us);
  }

  inline void AppendEventPsec(OptionalString name, TimePsec start_ps, TimePsec end_ps) {
    SetEventPsec(name, _next_event_to_set, start_ps, end_ps);
  }

  inline size_t size() const {
    return _next_event_to_set;
  }

  inline size_t capacity() const {
    return _max_events;
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


//class OverheadCounter {
//  public:
//    // 1) Read JSON.
//    // 2) Generate overhead events.
//    //   a) Simple event types: only need EOTimes to insert, don't need event_name
//    //      (i.e. DON'T need to re-read raw files)
//    //      - CATEGORY_PROF_PYTHON_ANNOTATION: Python annotations
//    //      - CATEGORY_PROF_PYTHON_INTERCEPTION: Python -> C-library interception
//    //      - CATEGORY_PROF_LD_PRELOAD: LD_PRELOAD
//    //   b) Complex event types: need event_name, so we need to re-read raw files,
//    //      OR provide an option to record operation event-names;
//    //      we can encode unique operation names as integer id's (bitset if we want to be hardcore).
//    //      This will make storing way more efficient.
//    //      - CATEGORY_PROF_CUPTI: CUPTI
//};


class CategoryTimesCount {
public:
  using CountMap = std::map<CategoryKey, size_t>;
  CountMap num_events;
  CountMap extra_num_events;

  inline void _Add(CountMap* cmap, const CategoryKey& category_key, size_t n_events) {
    // NOTE: if num_events[category_key] will default to zero if its not inside the map.
    (*cmap)[category_key] += n_events;
  }
  inline void Add(const CategoryKey& category_key, size_t n_events) {
    _Add(&num_events, category_key, n_events);
  }
  inline void AddExtra(const CategoryKey& category_key, size_t n_events) {
    _Add(&extra_num_events, category_key, n_events);
  }

  inline size_t _Count(const CountMap& cmap, const CategoryKey& category_key) const {
    return cmap.at(category_key);
  }
  inline size_t Count(const CategoryKey& category_key) const {
    return _Count(num_events, category_key);
  }
  inline size_t CountExtra(const CategoryKey& category_key) const {
    return _Count(extra_num_events, category_key);
  }

  inline bool _Contains(const CountMap& cmap, const CategoryKey& category_key) const {
    return cmap.count(category_key) == 0;
  }
  inline bool Contains(const CategoryKey& category_key) const {
    return _Contains(num_events, category_key);
  }
  inline bool ContainsExtra(const CategoryKey& category_key) const {
    return _Contains(extra_num_events, category_key);
  }

  void _AddToCategoryTimes(const CategoryTimesCount& ctimes);

  friend CategoryTimesCount operator+(const CategoryTimesCount& left, const CategoryTimesCount& right);
  CategoryTimesCount& operator+=(const CategoryTimesCount& rhs);
  void _Print(const CountMap& cmap, const std::string& name, std::ostream& out, int indent) const;
  void Print(std::ostream& out, int indent) const;
  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(CategoryTimesCount)

};


template <typename Iterable,
//    typename Func, typename KeyFunc,
    typename Key>
void EachMerged(const std::vector<Iterable>& vec_of_xs
    // Func func,

    , std::function<void(const Iterable&, size_t)> func
    , std::function<Key(const Iterable&, size_t)> key_func

//    , Func func
//    , KeyFunc key_func

    ) {
  using namespace Eigen;

  using IdxArray = Array<size_t, Dynamic, 1>;
  size_t k = vec_of_xs.size();
  IdxArray index = IdxArray::Zero(k);

  IdxArray lengths = IdxArray(k);
  {
    int i = 0;
    for (const auto& xs : vec_of_xs) {
      lengths(i) = xs.size();
      i += 1;
    }
  }

  while ((index < lengths).any()) {
    // Find the non-empty category with the next minimum start/end time.
    int min_i = -1;
    Key min_key;
    for (int i = 0; i < index.size(); i++) {
      // Check we haven't exhausted the intervals in the category.
      if (index(i) < lengths(i)) {
        // Non-empty category.
        // auto x = get_func(vec_of_xs[i], index(i));
        Key key = key_func(vec_of_xs[i], index(i));
        if (min_i == -1 || key < min_key) {
          min_i = i;
          min_key = key;
        }
      }
    }

    assert(min_i != -1);
    func(vec_of_xs[min_i], index(min_i));
    index(min_i) += 1;
  }
}

class CategoryTimes {
public:
  using EOTimes = std::map<CategoryKey, EOEvents>;
  EOTimes eo_times;
  // Extra EOEvents data that WON'T be used during overlap computation.
  // Useful for storing additional data needed for inserting overhead events.
  EOTimes extra_eo_times;
  Process process;

  CategoryTimes() = default;
  CategoryTimes(const Process& process, const CategoryTimesCount& count);
  inline size_t size() const {
    return eo_times.size();
  }
  size_t TotalEvents() const;

  void Preallocate(const CategoryKey& category_key, size_t n_events);
  void PreallocateExtra(const CategoryKey& category_key, size_t n_events);
  void _Preallocate(EOTimes* eo_times, const CategoryKey& category_key, size_t n_events);

  // // Writable events, create if it doesn't exist.
  // EOEvents& eo_events = category_times->MutableEvents(key);
  // // Read-only events, create if it doesn't exist.
  // const EOEvents& eo_events = category_times->Events(key);
  //
  // // Same as above, but don't assume it exists, and don't create it if it doesn't.
  // boost::optional<EOEvents&> eo_events = category_times->MaybeMutableEvents(key);
  // boost::optional<const EOEvents&> eo_events = category_times->MaybeEvents(key);
  const EOEvents& Events(const CategoryKey& category_key);
  EOEvents& MutableEvents(const CategoryKey& category_key);
  boost::optional<const EOEvents&> MaybeEvents(const CategoryKey& category_key) const;
  boost::optional<EOEvents&> MaybeMutableEvents(const CategoryKey& category_key);

  const EOEvents& EventsExtra(const CategoryKey& category_key);
  EOEvents& MutableEventsExtra(const CategoryKey& category_key);
  boost::optional<const EOEvents&> MaybeEventsExtra(const CategoryKey& category_key) const;
  boost::optional<EOEvents&> MaybeMutableEventsExtra(const CategoryKey& category_key);

  const EOEvents& _Events(EOTimes* eo_times, const CategoryKey& category_key);
  EOEvents& _MutableEvents(EOTimes* eo_times, const CategoryKey& category_key);
  boost::optional<const EOEvents&> _MaybeEvents(const EOTimes& eo_times, const CategoryKey& category_key) const;
  boost::optional<EOEvents&> _MaybeMutableEvents(EOTimes* eo_times, const CategoryKey& category_key);

  size_t _Count(const EOTimes& eo_times, const CategoryKey& category_key) const;
  size_t Count(const CategoryKey& category_key) const;
  size_t CountExtra(const CategoryKey& category_key) const;

  void Print(std::ostream& out, int indent) const;
  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(CategoryTimes)
  void PrintSummary(std::ostream& out, int indent) const;
};

class CategoryTimesBitset {
public:
  std::map<CategoryKeyBitset, EOEvents> eo_times;
  Process process;

  std::shared_ptr<CategoryIdxMap> idx_map;

  CategoryTimesBitset(const CategoryTimes& category_times) {
    // Copy plain fields.
    process = category_times.process;

    // Copy eo_times (NOTE: float array is a shared_ptr).
    std::vector<CategoryKey> keys;
    for (auto const& pair : category_times.eo_times) {
      keys.push_back(pair.first);
    }
    idx_map.reset(new CategoryIdxMap());
    *idx_map = CategoryIdxMap::From(keys);
    for (auto const& pair : category_times.eo_times) {
      auto idx = idx_map->Idx(pair.first);
      CategoryKeyBitset category(idx, idx_map);
      eo_times[category] = pair.second;
    }
  }
  CategoryTimesBitset() = default;
  inline size_t size() const {
    return eo_times.size();
  }

  void Print(std::ostream& out, int indent) const;
  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(CategoryTimesBitset)
  void PrintSummary(std::ostream& out, int indent) const;
};

//class EOTimes {
//public:
//  CategoryTimes _category_times;
//};

// Navigate trace-files for a particular (machine, process, phase) in the --iml-directory in time-stamp order.
// Used by RawTraceParser for reading events into eo_times in the correct order.
struct EntireTraceMeta {
  Machine machine;
  Process process;
  Phase phase;

  EntireTraceMeta() = default;
  EntireTraceMeta(
      const Machine& machine_,
      const Process& process_,
      const Phase& phase_) :
      machine(machine_)
      , process(process_)
      , phase(phase_)
  {
  }
};

//struct IParserMeta {
//  RLSFileType file_type;
//  IParserMeta() : file_type(UNKNOWN_FILE) {
//  }
//  IParserMeta(RLSFileType file_type_) : file_type(file_type_) {
//  }
//};

struct TraceFileMeta {
  std::string path;

  CategoryTimesCount count;
  Machine machine;
  Process process;
  Phase phase;
  TraceID trace_id;
  RLSFileType file_type;

//  std::shared_ptr<IParserMeta> parser_meta;

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
  std::shared_ptr<SimpleTimer> timer;
  std::string _iml_directory;
  TraceFileWalker _walker;

  std::string _cupti_overhead_json_path;
  std::string _LD_PRELOAD_overhead_json_path;
  std::string _pyprof_overhead_json_path;

  nlohmann::json _cupti_overhead_json;
  nlohmann::json _LD_PRELOAD_overhead_json;
  nlohmann::json _pyprof_overhead_json;

  bool _has_calibration_files;

  RawTraceParser(const std::string& iml_directory,
      const std::string& cupti_overhead_json,
      const std::string& LD_PRELOAD_overhead_json,
      const std::string& pyprof_overhead_json) :
      _iml_directory(iml_directory),
      _walker(_iml_directory),
      _cupti_overhead_json_path(cupti_overhead_json),
      _LD_PRELOAD_overhead_json_path(LD_PRELOAD_overhead_json),
      _pyprof_overhead_json_path(pyprof_overhead_json),
      _has_calibration_files(false)
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

  template <typename Func>
  MyStatus EachEntireTrace(Func func) {
    MyStatus status = MyStatus::OK();
    for (auto const& machine : this->Machines()) {
      for (auto const& process : this->Processes(machine)) {
        for (auto const &phase : this->Phases(machine, process)) {
          CategoryTimes category_times;
          EntireTraceMeta meta;
          status = this->ReadEntireTrace(machine, process, phase,
                                          &category_times, &meta);
          IF_BAD_STATUS_RETURN(status);
          status = func(category_times, meta);
          IF_BAD_STATUS_RETURN(status);
        }
      }
    }
    return MyStatus::OK();
  }

  MyStatus ReadEntireTrace(
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      CategoryTimes* category_times,
      EntireTraceMeta* entire_meta);

  MyStatus _AppendOverheadEvents(
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      CategoryTimes *category_times);
  MyStatus _AppendOverhead_CUPTI_and_LD_PRELOAD(
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      CategoryTimes *category_times);
  MyStatus _AppendOverhead_PYTHON_INTERCEPTION(
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      CategoryTimes *category_times);
  MyStatus _AppendOverhead_PYTHON_ANNOTATION(
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
//  virtual std::shared_ptr<IParserMeta> GetParserMeta() = 0;

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

//  std::shared_ptr<IParserMeta> _parser_meta;

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

//  virtual std::shared_ptr<IParserMeta> GetParserMeta() override {
//    return _parser_meta;
//  }

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

    if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
      std::stringstream ss;
      ss << "\n";
      count.Print(ss, 1);
      DBG_LOG("{}", ss.str());
    }

    *out_category_times = std::move(CategoryTimes(get_process(), count));
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
  static const RLSFileType FILE_TYPE = RLSFileType::CATEGORY_EVENTS_FILE;
  using ProtoKlass = iml::CategoryEventsProto;

//  struct ParserMeta : IParserMeta {
//    // How many operation events are there BEFORE we apply EachOpEvent.
//    size_t num_CATEGORY_OPERATION;
//    // eo_events for CATEGORY_OPERATION BEFORE we apply EachOpEvent.
//    EOEvents eo_CATEGORY_OPERATION;
//    ParserMeta() :
//        IParserMeta(FILE_TYPE),
//        num_CATEGORY_OPERATION(0)
//    {
//    }
//  };


  CategoryEventsParser(const std::string& path) :
      IEventFileProtoParser<iml::CategoryEventsProto>(path, RLSFileType::CATEGORY_EVENTS_FILE, "category_events")
  {
  }

  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) override;
  MyStatus _CountCategoryTimesOperation(CategoryTimesCount* count, const ProtoKlass& proto);
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) override;
  MyStatus _AppendCategoryOperation(const Category& category, const ProtoKlass& proto, CategoryTimes* out_category_times);
  virtual MyStatus _InitFromProto(const ProtoKlass& proto) override;

  MyStatus _AppendCategory(const Category& category, const ProtoKlass& proto, EOEvents* eo_events);

//  DECLARE_GET_PARSER_META

  // PROBLEM:
  // - we need to determine ahead of time how many events we need to read so we can preallocate an array.
  // - to do this, we need to load each protobuf file.
  // - our "ReadCategory" function should instead become "AppendCategory", and we should return an error or Assert if its too big.
};

class CUDAAPIStatsParser : public IEventFileProtoParser<iml::CUDAAPIPhaseStatsProto> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CUDA_API_STATS_FILE;
//  struct ParserMeta : IParserMeta {
//    ParserMeta() :
//        IParserMeta(FILE_TYPE)
//    {
//    }
//  };

  using ProtoKlass = iml::CUDAAPIPhaseStatsProto;
  CUDAAPIStatsParser(const std::string& path) :
      IEventFileProtoParser<ProtoKlass>(path, RLSFileType::CUDA_API_STATS_FILE, "cuda_api_stats")
  {
  }


  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) override;
  virtual MyStatus _InitFromProto(const ProtoKlass& proto) override;

//  DECLARE_GET_PARSER_META

};

class CUDADeviceEventsParser : public IEventFileProtoParser<iml::MachineDevsEventsProto> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CUDA_DEVICE_EVENTS_FILE;
//  struct ParserMeta : IParserMeta {
//    ParserMeta() :
//        IParserMeta(FILE_TYPE) {
//    }
//  };
  using ProtoKlass = iml::MachineDevsEventsProto;
  CUDADeviceEventsParser(const std::string& path) :
      IEventFileProtoParser<ProtoKlass>(path, RLSFileType::CUDA_DEVICE_EVENTS_FILE, "cuda_device_events")
  {
  }

  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, const ProtoKlass& proto) override;
  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, const ProtoKlass& proto) override;
  virtual MyStatus _InitFromProto(const ProtoKlass& proto) override;

//  DECLARE_GET_PARSER_META

};

MyStatus GetTraceID(const std::string& path, TraceID* trace_id);

template <typename Elem, typename Iterable, typename BoolUnaryOp, typename BoolBinaryOp>
Elem KeepExtreme(
    const Iterable& xs,
    Elem dflt,
    BoolBinaryOp should_keep,
    BoolUnaryOp should_skip) {
  Elem value = dflt;
  bool is_dflt = true;
  for (auto const& x: xs) {
    if (should_skip(x)) {
      continue;
    }

    if (is_dflt) {
      value = x;
      is_dflt = false;
    } else if (should_keep(x, value)) {
      value = x;
    }
  }
  return value;
}

struct RegionMetadataReducer {
  CategoryKey category_key;
  TimeUsec start_time_usec;
  TimeUsec end_time_usec;
  size_t num_events;
  RegionMetadataReducer() = default;
  RegionMetadataReducer(const CategoryKey& category_key) :
      category_key(category_key),
      start_time_usec(0),
      end_time_usec(0),
      num_events(0) {
  }

  RegionMetadataReducer(
      const CategoryKey& category_key_,
      TimeUsec start_time_usec_,
      TimeUsec end_time_usec_,
      size_t num_events_) :
      category_key(category_key_),
      start_time_usec(start_time_usec_),
      end_time_usec(end_time_usec_),
      num_events(num_events_) {
  }

  void MergeInplace(const RegionMetadataReducer& region2) {
    RegionMetadataReducer& region1 = *this;

    region1.start_time_usec = KeepExtreme(
        std::vector<TimeUsec>{region1.start_time_usec, region2.start_time_usec},
        0,
        /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
          return x < min_x;
        },
        /*should_skip=*/[] (TimeUsec x) {
          return x == 0;
        });

    region1.end_time_usec = KeepExtreme(
        std::vector<TimeUsec>{region1.end_time_usec, region2.end_time_usec},
        0,
        /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
          return x > min_x;
        },
        /*should_skip=*/[] (TimeUsec x) {
          return x == 0;
        });

    region1.num_events = region1.num_events + region2.num_events;
  }

};
struct RegionMetadata {
  CategoryKeyBitset category_key;
  TimeUsec start_time_usec;
  TimeUsec end_time_usec;
  size_t num_events;
  RegionMetadata() = default;
  RegionMetadata(const CategoryKeyBitset& category_key) :
      category_key(category_key),
      start_time_usec(0),
      end_time_usec(0),
      num_events(0) {
  }

  inline void AddEvent(TimeUsec start_us, TimeUsec end_us) {
    if (this->start_time_usec == 0 || start_us < this->start_time_usec) {
      this->start_time_usec = start_us;
    }

    if (this->end_time_usec == 0 || end_us > this->end_time_usec) {
      this->end_time_usec = end_us;
    }

    this->num_events += 1;
  }


};
struct OverlapMetadata {
  std::map<CategoryKeyBitset, RegionMetadata> regions;

  void AddEvent(const CategoryKeyBitset& category_key, TimeUsec start_us, TimeUsec end_us) {
    if (regions.find(category_key) == regions.end()) {
      regions[category_key] = RegionMetadata(category_key);
    }
    regions[category_key].AddEvent(start_us, end_us);
  }
};
struct OverlapMetadataReducer {
  std::map<CategoryKey, RegionMetadataReducer> regions;

  inline OverlapMetadataReducer OnlyKey(const CategoryKey& category_key) const {
    OverlapMetadataReducer r;
    r.regions[category_key] = this->regions.at(category_key);
    return r;
  }

  void MergeRegion(const CategoryKey& category_key, const RegionMetadataReducer& region) {
    if (regions.find(category_key) == regions.end()) {
      regions[category_key] = RegionMetadataReducer(category_key);
    }
    regions[category_key].MergeInplace(region);
  }

  void AddRegion(const CategoryKey& category_key, const RegionMetadata& region_metadata) {
    assert(regions.find(category_key) == regions.end());
    regions[category_key] = RegionMetadataReducer(
        category_key,
        region_metadata.start_time_usec,
        region_metadata.end_time_usec,
        region_metadata.num_events);
  }

};
struct OverlapResult {
  Overlap overlap;
  OverlapMetadata meta;
  std::shared_ptr<CategoryIdxMap> idx_map;

  inline const RegionMetadata& GetMeta(const CategoryKeyBitset& bitset) const {
    return meta.regions.at(bitset);
  }

  void DumpVennJS(
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase) const;

  void Print(std::ostream& out, int indent) const;
  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(OverlapResult)
};
class OverlapResultReducer {
public:
  std::map<CategoryKey, TimeUsec> overlap;
  OverlapMetadataReducer regions;
  bool debug{false};

  OverlapResultReducer() = default;

  inline const RegionMetadataReducer& GetMeta(const CategoryKey& category_key) const {
    return regions.regions.at(category_key);
  }

  inline OverlapResultReducer OnlyKey(const CategoryKey& category_key) const {
    OverlapResultReducer r;
    r.overlap[category_key] = this->overlap.at(category_key);
    r.regions = this->regions.OnlyKey(category_key);
    return r;
  }

  inline const TimeUsec& GetTimeUsec(const CategoryKey& category_key) const {
    return overlap.at(category_key);
  }

  static OverlapResultReducer ReduceToCategoryKey(const OverlapResult& result) {
    OverlapResultReducer ov;
    for (auto const& pair : result.overlap) {

      const auto& bitset = pair.first;
      auto category_keys = bitset.Keys();
      auto time_usec = pair.second;

      CategoryKey new_key(
          /*procs=*/{},
          /*ops=*/{},
          /*non_ops=*/{});
      for (const auto& category_key : category_keys) {
        assert(category_key.procs.size() == 1);
        new_key.ops.insert(category_key.ops.begin(), category_key.ops.end());
        new_key.non_ops.insert(category_key.non_ops.begin(), category_key.non_ops.end());
        new_key.procs.insert(category_key.procs.begin(), category_key.procs.end());
      }

      // We don't yet support overlap across processes.
      assert(new_key.procs.size() == 1);

      // If involves BOTH execution: {CPU, GPU, CPU/GPU},
      //    AND an operation: {q_forward, q_backward, ...}
      //   Keep.
      if (new_key.ops.size() == 0 || new_key.non_ops.size() == 0) {
        continue;
      }

      assert(ov.overlap.find(new_key) == ov.overlap.end());
      ov.overlap[new_key] = time_usec;
      ov.regions.AddRegion(new_key, result.GetMeta(bitset));
    }
    return ov;
  }

  static bool IsEmptyKey(const CategoryKey& category_key) {
    return category_key.ops.size() == 0 ||
           category_key.non_ops.size() == 0;
  }

  static CategoryKey NoOverheadKey(const CategoryKey& key) {
    // maybe_remove_overhead
    CategoryKey new_key;
    new_key.ops = key.ops;
    new_key.procs = key.procs;
    new_key.non_ops = key.non_ops;

    std::set<Category> prof_categories;
    std::set_intersection(
        key.non_ops.begin(), key.non_ops.end(),
        CATEGORIES_PROF.begin(), CATEGORIES_PROF.end(),
        std::inserter(prof_categories, prof_categories.begin()));
    if (prof_categories.size() > 0) {
        // Discard CPU-time that is due to profiling overhead.
        // NOTE: CATEGORY_GPU won't get discarded.
        new_key.non_ops.erase(CATEGORIES_CPU.begin(), CATEGORIES_CPU.end());
        // Q: Should we remove the profiling category as well...? I think so yes.
        new_key.non_ops.erase(CATEGORIES_PROF.begin(), CATEGORIES_PROF.end());
    }
    return new_key;
  }
  static CategoryKey ReduceCategoryKey(const CategoryKey& key, bool as_cpu_gpu) {
    // Modular function to bin_events for "reducing" events to CPU/GPU BEFORE OverlapComputation.
    // Also, allow ability to "filter-out" events (e.g. category=GPU; needed for CategoryOverlap).
    CategoryKey new_key(
        /*procs=*/key.procs,
        /*ops=*/key.ops,
        /*non_ops=*/{});
    for (auto const& category : key.non_ops) {

      if (CATEGORIES_CPU.count(category) != 0) {
        if (as_cpu_gpu) {
          // NOTE: profiling types are treated as fine-grained CPU categories.
          new_key.non_ops.insert(CATEGORY_CPU);
        } else {
          new_key.non_ops.insert(category);
        }
      } else if (CATEGORIES_GPU.count(category) != 0) {
        if (as_cpu_gpu) {
          new_key.non_ops.insert(CATEGORY_GPU);
        } else {
          new_key.non_ops.insert(category);
        }
      } else if (CATEGORIES_PROF.count(category) != 0) {
        new_key.non_ops.insert(category);
      } else {
        // Not sure how to categorize key.
        assert(false);
      }

    }
    return new_key;

  }

  void AddOverlapWithKey(
      const CategoryKey& old_key,
      const CategoryKey& new_key,
      const OverlapResultReducer& old_reducer) {


    const auto& no_overhead_key = OverlapResultReducer::NoOverheadKey(new_key);

    if (debug) {
      std::stringstream ss;
      ss << "AddOverlapWithKey:";

      ss << "\n";
      ss << "old_key = ";
      old_key.Print(ss, 1);

      ss << "\n";
      ss << "new_key = ";
      new_key.Print(ss, 1);

      ss << "\n";
      ss << "no_overhead_key = ";
      no_overhead_key.Print(ss, 1);

      DBG_LOG("{}", ss.str());
    }


    if (OverlapResultReducer::IsEmptyKey(no_overhead_key)) {
      return;
    }
    auto time_usec = old_reducer.GetTimeUsec(old_key);
    overlap[no_overhead_key] += time_usec;
    regions.MergeRegion(no_overhead_key, old_reducer.GetMeta(old_key));
  }

};
class OverlapComputer {
public:
  bool debug{false};
  const CategoryTimes& category_times;
  CategoryTimesBitset ctimes;
  void _CategoryToIdxMaps();
  OverlapComputer(const CategoryTimes& category_times_) :
      category_times(category_times_),
      ctimes(category_times)
  {
  }
  OverlapResult ComputeOverlap(bool keep_empty_time = false) const;
};

MyStatus ReadJson(std::string path, nlohmann::json* j);
MyStatus WriteJson(std::string path, const nlohmann::json& j);


#define DECLARE_DumpOverlapJS \
  virtual void DumpOverlapJS( \
        const std::string& directory, \
        const Machine& machine, \
        const Process& process, \
        const Phase& phase, \
        const OverlapResultReducer& reducer) = 0;

#define DEFINE_DumpOverlapJS(OverlapTypeClass) \
  virtual void DumpOverlapJS( \
      const std::string& directory, \
      const Machine& machine, \
      const Process& process, \
      const Phase& phase, \
      const OverlapResultReducer& reducer) { \
    _GenericDumpOverlapJS<OverlapTypeClass>( \
        directory, \
        machine, \
        process, \
        phase, \
        reducer); \
  }

class OverlapTypeReducerInterface {
public:
  virtual OverlapType GetOverlapType() const = 0;
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const = 0;

  DECLARE_DumpOverlapJS

  template <class OverlapTypeClass>
  void _GenericDumpOverlapJS(
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      const OverlapResultReducer& reducer) const {
    typename OverlapTypeClass::OverlapJSDumper js_dumper(
        GetOverlapType(),
        directory,
        machine,
        process,
        phase,
        reducer);
    js_dumper.DumpJSONs();
  }

};

class OverlapMapToVennJSConverter {
public:
  std::map<std::string, TimeUsec> _ComputeSetSizes(const OverlapMap& overlap_map) const {
    std::map<std::string, TimeUsec> set_to_size;
    for (const auto& pair :overlap_map) {
      const auto& overlap_region = pair.first;
      auto time_usec = pair.second;
      for (const auto& set_region : overlap_region) {
        set_to_size[set_region] += time_usec;
      }
    }
    return set_to_size;
  }
  nlohmann::json convert(const OverlapMap& overlap_map) const {
    nlohmann::json venn_js;
    auto const& set_to_size = _ComputeSetSizes(overlap_map);

    std::map<std::string, size_t> label_to_id;
    std::set<std::string> labels;
    for (const auto& pair : overlap_map) {
      auto const& overlap = pair.first;
      labels.insert(overlap.begin(), overlap.end());
    }
    size_t i = 0;
    for (const auto& label : labels) {
      label_to_id[label] = 0;
    }

    auto as_sets = [&label_to_id] (const std::set<std::string>& overlap) {
      std::set<size_t> set_ids;
      for (auto const& category : overlap) {
        auto ident = label_to_id[category];
        set_ids.insert(ident);
      }
      return set_ids;
    };

    for (const auto& pair : set_to_size) {
      const auto& label = pair.first;
      const auto size = pair.second;
      nlohmann::json venn_set = {
          {"sets", as_sets({label})},
          {"size", size},
          {"label", label},
      };
      venn_js.push_back(venn_set);
    }

    for (const auto& pair : overlap_map) {
      const auto& overlap = pair.first;
      auto size = pair.second;
      if (overlap.size() == 1) {
        // Single "set region" doesn't include overlap with other sets.
        // "Set region" is handled in for-loop above this one.
        continue;
      }
      nlohmann::json venn_set = {
          {"sets", as_sets(overlap)},
          {"size", size},
      };
      venn_js.push_back(venn_set);
    }

    // venn_js.sort(key=lambda venn_set: (len(venn_set['sets']), venn_set['sets']))

    // Make the shorter (in particular, single-element) venn_sets appear first.
    // venn_sets within the same length are ordered based on lexicographic order.
#define SORT_KEY(venn_set) std::make_tuple(venn_set["sets"].size(), venn_set["sets"])
    std::sort(
        venn_js.begin(), venn_js.end(),
        [] (const nlohmann::json& lhs, const nlohmann::json& rhs) {
      return SORT_KEY(lhs) < SORT_KEY(rhs);
    });
#undef SORT_KEY

    return venn_js;
  }
};

class BaseOverlapJSDumper {
public:

  OverlapType overlap_type;
  std::string directory;
  Machine machine;
  Process process;
  Phase phase;

  std::map<std::set<Operation>,
      std::map<std::set<Category>,
          OverlapResultReducer>> reducer_map;

  BaseOverlapJSDumper(
      const OverlapType& overlap_type_,
      const std::string& directory_,
      const Machine& machine_,
      const Process& process_,
      const Phase& phase_,
      const OverlapResultReducer& reducer) :
      overlap_type(overlap_type_),
      directory(directory_),
      machine(machine_),
      process(process_),
      phase(phase_)
  {
    for (auto const& pair : reducer.overlap) {
      const auto& category_key = pair.first;
      assert(
          ( reducer_map.find(category_key.ops) == reducer_map.end() ) ||
          ( reducer_map[category_key.ops].find(category_key.non_ops) == reducer_map[category_key.ops].end() ));
      reducer_map[category_key.ops][category_key.non_ops] = reducer.OnlyKey(category_key);
    }
  }

  void DumpJSONs() const {
    for (auto const& ops_pair : reducer_map) {
      const auto& ops = ops_pair.first;
      for (auto const& non_ops_pair : ops_pair.second) {
        const auto& non_ops = non_ops_pair.first;
        const auto& reducer = non_ops_pair.second;

        OverlapMap overlap_map;
        for (auto const& category_key_pair : reducer.overlap) {
          auto const& category_key = category_key_pair.first;
          auto time_usec = category_key_pair.second;

          auto const& strings = CategoryKeyToStrings(category_key);
          overlap_map[strings] = time_usec;

          nlohmann::json js;

          Metadata md;
          md["machine"] = machine;
          md["process"] = process;
          md["phase"] = phase;
          md["overlap_type"] = overlap_type;

          auto const& meta = reducer.GetMeta(category_key);
          md["start_time_usec"] = meta.start_time_usec;
          md["end_time_usec"] = meta.end_time_usec;
          md["num_events"] = meta.num_events;

          AddMetadataFields(&md, category_key);
          js["metadata"] = md;

          OverlapMapToVennJSConverter converter;
          auto const& venn_js = converter.convert(overlap_map);

          js["venn"] = venn_js;
          const auto& venn_js_path = VennJSPath(md);
          if (SHOULD_DEBUG(FEATURE_SAVE_JS)) {
            DBG_LOG("Write json to path = {}", venn_js_path);
          }

          boost::filesystem::path parent = boost::filesystem::path(venn_js_path).parent_path();
          boost::filesystem::create_directories(parent.string());

          MyStatus status = MyStatus::OK();
          status = WriteJson(venn_js_path, js);
          assert(status.code() == MyStatus::OK().code());
        }
      }
    }
  }

  std::string VennJSPath(const Metadata& md) const {
    boost::filesystem::path direc(directory);
    boost::filesystem::path base = VennJSBasename(md);
    return (direc / base).string();
  }

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const = 0;
  virtual std::string VennJSBasename(const Metadata& md) const = 0;
  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const = 0;

};

class ResourceJSDumper : public BaseOverlapJSDumper {
public:

  ResourceJSDumper(
      const OverlapType& overlap_type,
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      const OverlapResultReducer& reducer) :
      BaseOverlapJSDumper(
          overlap_type,
          directory,
          machine,
          process,
          phase,
          reducer)
  { }

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const {
    // set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
    assert(category_key.ops.size() == 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const {
    // pass
  }

};
class ResourceOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = ResourceJSDumper;
  DEFINE_DumpOverlapJS(ResourceOverlapType)

  virtual OverlapType GetOverlapType() const {
    return "ResourceOverlap";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    OverlapResultReducer r;
//    r.debug = true;
    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;
      if (old_key.ops.size() > 1) {
        // Operations can only overlap cross-process, not within a single-process
        assert(old_key.procs.size() > 1);
      }
      CategoryKey new_key(
          /*procs=*/{},
          /*ops=*/{},
          /*non_ops=*/old_key.non_ops);
      auto cpu_gpu_key = OverlapResultReducer::ReduceCategoryKey(new_key, /*as_cpu_gpu=*/true);
      r.AddOverlapWithKey(old_key, cpu_gpu_key, old_reducer);
    }
    return r;
  }

};

class ResourceSubplotJSDumper : public BaseOverlapJSDumper {
public:

  ResourceSubplotJSDumper(
      const OverlapType& overlap_type,
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      const OverlapResultReducer& reducer) :
      BaseOverlapJSDumper(
          overlap_type,
          directory,
          machine,
          process,
          phase,
          reducer)
  { }

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const {
    // set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
    assert(category_key.ops.size() == 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const {
    // pass
  }

};
class ResourceSubplotOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = ResourceSubplotJSDumper;
  DEFINE_DumpOverlapJS(ResourceSubplotOverlapType)

  virtual OverlapType GetOverlapType() const {
    return "ResourceSubplot";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    OverlapResultReducer r;
    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;

      if (old_key.ops.size() > 1) {
        // Operations can only overlap cross-process, not within a single-process
        assert(old_key.procs.size() > 1);
      }
      assert(old_key.non_ops.size() > 0);

      // Just {CPU}
      // Add time to CPU, add time to Total.
      //
      // Just {GPU}
      // Add time to GPU, add time to Total.
      //
      // Just {CPU, GPU}
      // Add time to CPU, add time to GPU, add time to Total.

      CategoryKey new_key(
          /*procs=*/{},
          /*ops=*/{},
          /*non_ops=*/{CATEGORY_TOTAL});
      r.AddOverlapWithKey(old_key, new_key, old_reducer);

      auto cpu_gpu_key = OverlapResultReducer::ReduceCategoryKey(old_key, /*as_cpu_gpu=*/true);
      for (auto const& resource_type : cpu_gpu_key.non_ops) {
        // NOTE: This is sort of hacky;
        // we AREN'T outputting disjoint overlap regions here;
        // instead we are outputting an entire "set" including its overlaps:
        // i.e.
        // CPU   = [CPU only time] + [CPU overlapped with GPU time]
        // GPU   = [GPU only time] + [GPU overlapped with GPU time]
        // Total = [CPU only time] + [GPU only time] + [CPU overlapped with GPU time]
        CategoryKey add_key(
            /*procs=*/{},
            /*ops=*/{},
            /*non_ops=*/{resource_type});
        r.AddOverlapWithKey(old_key, add_key, old_reducer);
      }

    }
    return r;
  }

};


class OperationJSDumper : public BaseOverlapJSDumper {
public:

  OperationJSDumper(
      const OverlapType& overlap_type,
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      const OverlapResultReducer& reducer) :
      BaseOverlapJSDumper(
          overlap_type,
          directory,
          machine,
          process,
          phase,
          reducer)
  { }

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const {
    // set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec
    assert(category_key.ops.size() > 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    AddResourcesSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const {
    (*md)["resource_overlap"] = category_key.non_ops;
  }

};
class OperationOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = OperationJSDumper;
  DEFINE_DumpOverlapJS(OperationOverlapType)

  virtual OverlapType GetOverlapType() const {
    return "OperationOverlap";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    // reduce_overlap_resource_operation

    OverlapResultReducer r;
    r.debug = true;

    if (r.debug) {
      DBG_LOG("{}", "OperationOverlapType.PostReduceCategoryKey");
    }

    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;

      if (old_key.ops.size() > 1) {
        // Operations can only overlap cross-process, not within a single-process
        assert(old_key.procs.size() > 1);
      }
      assert(old_key.non_ops.size() > 0);

      CategoryKey new_key(
          /*procs=*/{},
          /*ops=*/old_key.ops,
          /*non_ops=*/old_key.non_ops);
      new_key = OverlapResultReducer::ReduceCategoryKey(
          new_key,
          /*as_cpu_gpu=*/true);
      r.AddOverlapWithKey(old_key, new_key, old_reducer);

    }

    return r;
  }

};

class CategoryJSDumper : public BaseOverlapJSDumper {
public:

  CategoryJSDumper(
      const OverlapType& overlap_type,
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      const OverlapResultReducer& reducer) :
      BaseOverlapJSDumper(
          overlap_type,
          directory,
          machine,
          process,
          phase,
          reducer)
  { }

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const {
    // set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec
    assert(category_key.ops.size() > 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    AddOpsSuffix(ss, md);
    AddResourcesSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  struct CPUAndGPUCategories {
    std::set<Category> cpus;
    std::set<Category> gpus;
  };
  CPUAndGPUCategories SplitCpuGpuCategories(const CategoryKey& category_key) const {
    CPUAndGPUCategories cats;
    for (auto const& category : category_key.non_ops) {
      if (CATEGORIES_CPU.count(category) > 0) {
        cats.cpus.insert(category);
      } else if (CATEGORIES_GPU.count(category) > 0) {
        cats.gpus.insert(category);
      } else {
        // "Not sure how to categorize category={cat} as CPU vs GPU"
        assert(false);
      }
    }
    return cats;
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const {

    // add md["operation"]
    assert(category_key.ops.size() == 1);
    (*md)["operation"] = category_key.ops;

    // add md["resource_overlap"]
    auto const& cpus_gpus = SplitCpuGpuCategories(category_key);
    std::set<Category> resource_key;
    if (cpus_gpus.cpus.size() > 0) {
      resource_key.insert(CATEGORY_CPU);
    } else if (cpus_gpus.gpus.size() > 0) {
      resource_key.insert(CATEGORY_GPU);
    }
    (*md)["resource_overlap"] = resource_key;

  }

};
class CategoryOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = CategoryJSDumper;
  DEFINE_DumpOverlapJS(CategoryOverlapType)

  virtual OverlapType GetOverlapType() const {
    return "CategoryOverlap";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    // reduce_overlap_resource_operation

    OverlapResultReducer r;

    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;

      if (old_key.ops.size() > 1) {
        // Operations can only overlap cross-process, not within a single-process
        assert(old_key.procs.size() > 1);
      }
      assert(old_key.procs.size() == 1);
      assert(old_key.ops.size() == 1);
      assert(old_key.non_ops.size() >= 1);

      CategoryKey new_key(
          /*procs=*/{},
          /*ops=*/old_key.ops,
          /*non_ops=*/old_key.non_ops);
      new_key = OverlapResultReducer::ReduceCategoryKey(
          new_key,
          /*as_cpu_gpu=*/false);
      r.AddOverlapWithKey(old_key, new_key, old_reducer);

    }

    return r;
  }

};


}

#endif //IML_TRACE_FILE_PARSER_H
