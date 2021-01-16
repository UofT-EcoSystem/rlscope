//
// Created by jagle on 11/13/2019.
//

#ifndef RLSCOPE_TRACE_FILE_PARSER_H
#define RLSCOPE_TRACE_FILE_PARSER_H

#include "rlscope_prof.pb.h"
#include "pyprof.pb.h"

#include <future>

#include <ctpl.h>
#include <eigen3/Eigen/Dense>

#include "cuda_api_profiler/defines.h"

#include <assert.h>

#include <spdlog/spdlog.h>
//#include <sys/types.h>
// Must be included in order operator<< to work with spd logging.
// https://github.com/gabime/spdlog#user-defined-types
#include "spdlog/fmt/ostr.h"
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

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


#include <set>
#include "common_util.h"
#include <drivers/cpp_dump_proto.h>

//#define PSEC_IN_USEC (1000)
//#define USEC_IN_SEC (1000000)

#define RLSCOPE_VERSION 1

#define IS_BAD_STATUS(status) (status.code() != MyStatus::OK().code())

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

namespace rlscope {

// forward decls
class IEventFileParser;

enum RLSFileType {
  UNKNOWN_FILE = 0,
  CUDA_API_STATS_FILE = 1,
  CATEGORY_EVENTS_FILE = 2,
  CUDA_DEVICE_EVENTS_FILE = 3,
  NVPROF_GPU_TRACE_CSV_FILE = 4,
  NVPROF_API_TRACE_CSV_FILE = 5,
};

extern const std::regex PROCESS_OPERATION_REGEX;

extern const std::set<RLSFileType> RLS_FILE_TYPES;
//const std::set<RLSFileType>& AllRLSFileTypes() {
//  static const std::set<RLSFileType> RLSFileTypeVector = std::set<RLSFileType>{
//      CUDA_API_STATS_FILE ,
//      CATEGORY_EVENTS_FILE ,
//      CUDA_DEVICE_EVENTS_FILE ,
//  };
//  return RLSFileTypeVector;
//}


// Copied from: rlscope/parser/common.py
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
#define CATEGORY_CORRECTED_TRAINING_TIME "Corrected training time"
// Not a category used during tracing;
// represents a group of categories.
#define CATEGORY_CPU "CPU"

#define CATEGORY_PROF_CUPTI "Profiling: CUPTI"
#define CATEGORY_PROF_LD_PRELOAD "Profiling: LD_PRELOAD"
#define CATEGORY_PROF_PYTHON_ANNOTATION "Profiling: Python annotation"
#define CATEGORY_PROF_PYTHON_CLIB_INTERCEPTION_TENSORFLOW "Profiling: Python->C interception TensorFlow"
#define CATEGORY_PROF_PYTHON_CLIB_INTERCEPTION_SIMULATOR "Profiling: Python->C interception Simulator"

//#define TRACE_SUFFIX_RE (R"((?:\.trace_(?P<trace_id>\d+))?)")
// WARNING: using non-capturing groups (?:) causes match strings to become empty.
#define TRACE_SUFFIX_RE R"(\.trace_(\d+)?)"

//#define CUDA_API_STATS_REGEX (R"(^cuda_api_stats{trace}\.proto)")
#define CUDA_API_STATS_REGEX (R"(^cuda_api_stats)" TRACE_SUFFIX_RE R"(\.proto)")

//#define CATEGORY_EVENTS_REGEX (R"(category_events{trace}\.proto)")
#define CATEGORY_EVENTS_REGEX (R"(category_events)" TRACE_SUFFIX_RE R"(\.proto)")

//#define CUDA_DEVICE_EVENTS_REGEX (R"(cuda_device_events{trace}\.proto)")
#define CUDA_DEVICE_EVENTS_REGEX (R"(cuda_device_events)" TRACE_SUFFIX_RE R"(\.proto)")

// e.g.
// profile.process_15562.nvprof.gpu_trace.csv
//  profile.process_15562.nvprof.api_trace.csv
#define NVPROF_API_TRACE_CSV_REGEX (R"(.*\.nvprof.*\.api_trace.*\.csv)")
#define NVPROF_GPU_TRACE_CSV_REGEX (R"(.*\.nvprof.*\.gpu_trace.*\.csv)")

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

template <class Elem>
bool IsSubset(const std::set<Elem>& A, const std::set<Elem>& B) {
  // Return true iff A issubset-of B.
  // i.e. for all elements a in A, a is in B
  for (auto const& a : A) {
    if (B.find(a) == B.end()) {
      return false;
    }
  }
  return true;
}

template <typename EventList>
void SortEvents(EventList* events) {
  std::sort(
      events->begin(),
      events->end(),
      [] (const auto& lhs, const auto& rhs) {
        return lhs.start_time_us() < rhs.start_time_us();
      });
}

template <typename Events>
void CheckEventsSorted(const Category& category, const Events& events) {
  size_t failed = 0;
  for (int i = 0; i < events.size(); i++) {
    if (i > 0) {
      if (!(events[i-1].start_time_us() <= events[i].start_time_us())) {
        failed += 1;
      }
      // assert(events[i-1].start_time_us() <= events[i].start_time_us());
    }
  }
  if (failed > 0) {
    DBG_LOG("WARNING: saw {} unordered events for category = \"{}\"", failed, category);
  }
}


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

  inline size_t size() const {
    assert(from_idx.size() == to_idx.size());
    return from_idx.size();
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

  inline const std::set<Key>& Keys() const {
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

  void MergeInplace(const CategoryKey& category_key) {
    procs.insert(category_key.procs.begin(), category_key.procs.end());
    ops.insert(category_key.ops.begin(), category_key.ops.end());
    non_ops.insert(category_key.non_ops.begin(), category_key.non_ops.end());
  }

  static CategoryKey FromCategory(const Process& proc, const Category& category) {
    CategoryKey category_key;
    category_key.procs.insert(proc);
    category_key.non_ops.insert(category);
    return category_key;
  }

  static CategoryKey FromCategory(const std::set<Process>& procs, const Category& category) {
    CategoryKey category_key;
    for (const auto& proc : procs) {
      category_key.procs.insert(proc);
    }
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

using OverlapKey = std::set<CategoryKey>;
using CategoryIdxMap = IdxMaps<size_t, CategoryKey>;

class CategoryKeyBitset {
public:
  bool debug{false};
  std::bitset<MAX_CATEGORY_KEYS> bitset;
  // Keep this for debugging purposes, so we can convert a bitset into a std::string.
  std::shared_ptr<const CategoryIdxMap> idx_map;
  CategoryKeyBitset() : idx_map(nullptr) {
  }
  CategoryKeyBitset(size_t category_idx, std::shared_ptr<const CategoryIdxMap> idx_map) :
      // NOTE: refer to "WARNING regarding std::bitset" to understand why we need the static_cast<uint64_t>(1).
      // TLDR: want to guarantee 64-bit width, and prevent unexpected (compiler silent!) signed values; bare "1" is an "int" (32-bit signed!)
      bitset(static_cast<uint64_t>(1) << category_idx),
      idx_map(idx_map) {
    // Each CategoryKey is a unique bit in the bitset.
    // This allows us to do set operations.
    assert(category_idx < MAX_CATEGORY_KEYS);
  }
  CategoryKeyBitset(std::shared_ptr<const CategoryIdxMap> idx_map) :
      idx_map(idx_map) {
    // Empty set.
  }

  // TODO: this doesn't make sense for multi-process overlap; should return OverlapKey instead.
  CategoryKey AsCategoryKey() const {
    auto keys = Keys();
    CategoryKey merged_key;
    for (const auto& key : keys) {
      merged_key.MergeInplace(key);
    }
    return merged_key;
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

  inline size_t size() const {
    return bitset.count();
  }

  inline CategoryKeyBitset Intersection(const CategoryKeyBitset& rhs) const {
    const auto& lhs = *this;
    CategoryKeyBitset intersect;
    intersect.bitset = lhs.bitset & rhs.bitset;
    intersect.idx_map = lhs.idx_map;
    intersect.debug = lhs.debug;
    return intersect;
  }

  static CategoryKeyBitset Ops(const CategoryKeyBitset& category_set) {
    CategoryKeyBitset ops;
    ops.idx_map = category_set.idx_map;
    ops.debug = category_set.debug;
    for (const auto& pair : category_set.idx_map->to_idx) {
      auto const& category_key = pair.first;
      auto ident = pair.second;
      if (category_key.ops.size() > 0) {
        ops.Add(ident);
      }
    }
    return ops;
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

  unsigned long long SetID() const {
    return bitset.to_ullong();
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
// (From: from_category, To: to_category) -> count
using CategoryTransitionCounts = std::map<std::tuple<CategoryKeyBitset, CategoryKeyBitset>, size_t>;

bool isRLSFileWithType(RLSFileType file_type, const std::string& path);
bool isRLSFile(const std::string& path);
RLSFileType GetRLSFileType(const std::string& path);
const char* RLSFileTypeString(RLSFileType file_type);

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

MyStatus FindRLSFiles(const std::string& rlscope_directory, std::list<std::string>* paths);

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

class EOEvents;

bool CategoryShouldKeepNames(const CategoryKey& key);

class IEventMetadata {
public:
  // Internally, we don't care how parser stores event metadata.
  // However, we do require that we can output the metadata to a standard format (csv).
  virtual const std::vector<std::string>& GetHeader() const = 0;
  virtual const std::vector<std::string>& GetRow() const = 0;
  virtual IEventMetadata* clone() const = 0;
};

class EOEvent {
public:
  const EOEvents* _eo_events;
  size_t _i;

  EOEvent(const EOEvents* eo_events, size_t i) :
      _eo_events(eo_events),
      _i(i) {
  }

  const std::string& name() const;
  TimeUsec start_time_us() const;
  TimeUsec end_time_us() const;
  TimeUsec duration_us() const;
  const IEventMetadata* metadata() const;
};

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
  TimeUsec _min_start_ps;
  TimeUsec _max_end_ps;

  // Assumption:
  // The number of unique event-names belonging to a single category is < 2^32.
  // NOTE: It's not clear to me what a reasonable value to use here really is...
  KeyVector<EventNameID, std::string> _names;
  bool _keep_names;

  bool _keep_event_metadata;
  std::shared_ptr<std::vector<std::unique_ptr<IEventMetadata>>> _event_metadata;

//  typedef A allocator_type;
  typedef EOEvent value_type;
  typedef EOEvent& reference;
//  typedef typename A::const_reference const_reference;
  typedef size_t difference_type;
  typedef size_t size_type;
  typedef size_t pointer;

  class iterator {
  public:
    typedef iterator self_type;
    typedef EOEvents::value_type value_type;
    typedef EOEvents::value_type& reference;
    typedef EOEvents::pointer pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef EOEvents::difference_type difference_type;
    iterator(const EOEvents* eo_events, size_type i) :
        eo_events_(eo_events),
        i_(i) {
    }
    // PREFIX
    self_type operator++() {
      i_++;
      return *this;
    }
    // POSTFIX
    self_type operator++(int junk) {
      self_type old_self = *this;
      i_++;
      return old_self;
    }
    value_type operator*() {
      return EOEvent(eo_events_, i_);
    }
    pointer operator->() { return i_; }
    bool operator==(const self_type& rhs) {
      auto const& lhs = *this;
      return lhs.eo_events_ == rhs.eo_events_ &&
             lhs.i_ == rhs.i_;
    }
    bool operator!=(const self_type& rhs) {
      auto const& lhs = *this;
      return lhs.eo_events_ != rhs.eo_events_ ||
             lhs.i_ != rhs.i_;
    }
  private:
    const EOEvents* eo_events_;
    size_type i_;
  };

  iterator begin() const
  {
    return iterator(this, 0);
  }

  iterator end() const
  {
    return iterator(this, this->size());
  }

  inline EOEvent Event(size_t i) const {
    assert(i < _next_event_to_set);
    return EOEvent(this, i);
  }

//  class const_iterator {
//  public:
//    typedef const_iterator self_type;
//    typedef T value_type;
//    typedef T& reference;
//    typedef T* pointer;
//    typedef int difference_type;
//    typedef std::forward_iterator_tag iterator_category;
//    const_iterator(pointer ptr) : ptr_(ptr) { }
//    self_type operator++() { self_type i = *this; ptr_++; return i; }
//    self_type operator++(int junk) { ptr_++; return *this; }
//    const reference operator*() { return *ptr_; }
//    const pointer operator->() { return ptr_; }
//    bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
//    bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
//  private:
//    pointer ptr_;
//  };

  static EOEvents Merge(const EOEvents& lhs, const EOEvents& rhs) {
    if (lhs._keep_names) {
      assert(rhs._keep_names);
    }
    EOEvents merged(
        lhs._max_events + rhs._max_events,
        lhs._keep_names,
        lhs._keep_event_metadata);

    bool keep_names = merged.KeepNames();
    bool keep_event_metadata = merged.KeepEventMetadata();
    auto append_event = [&merged, keep_names, keep_event_metadata] (const EOEvents& eo_events, size_t* i) {
      OptionalString name;
      if (keep_names) {
        name = eo_events.GetEventName(*i);
      }
      std::unique_ptr<IEventMetadata> event_metadata;
      if (keep_event_metadata) {
        event_metadata.reset(event_metadata->clone());
      }
      merged.AppendEvent(name, eo_events.StartUsec(*i), eo_events.EndUsec(*i), std::move(event_metadata));
      *i += 1;
    };

    size_t lhs_i = 0;
    size_t rhs_i = 0;
    while (lhs_i < lhs.size() && rhs_i < rhs.size()) {
      if (lhs.StartUsec(lhs_i) <= rhs.StartUsec(rhs_i)) {
        append_event(lhs, &lhs_i);
      } else {
        append_event(rhs, &rhs_i);
      }
    }
    while (lhs_i < lhs.size()) {
      append_event(lhs, &lhs_i);
    }
    while (rhs_i < rhs.size()) {
      append_event(rhs, &rhs_i);
    }

    return merged;
  }

  static EOEvents Preallocate(const CategoryKey& category_key, size_t n_events, bool keep_event_metadata = false);
  static EOEvents PreallocateEvents(const CategoryKey& category_key, size_t n_events, bool keep_names, bool keep_event_metadata);

  EOEvents() :
      _max_events(0),
      _next_event_to_set(0),
      _events(new std::vector<TimeUsec>()),
      _min_start_ps(0),
      _max_end_ps(0),
      _keep_names(false),
      _keep_event_metadata(false)
  {
  }
  EOEvents(size_t n_events, bool keep_names, bool keep_event_metadata) :
      _max_events(n_events)
      , _next_event_to_set(0)
      // 2*n_events: For each event, we need the (start, end) time.
      // , _events(new std::vector<TimeUsec>(2*n_events))
      , _events(new std::vector<TimeUsec>())
      , _min_start_ps(0)
      , _max_end_ps(0)
      , _keep_names(keep_names)
      , _keep_event_metadata(keep_event_metadata)
      , _event_metadata(new std::vector<std::unique_ptr<IEventMetadata>>())
  {
    _events->reserve(2*n_events);
    if (_keep_names) {
      _names.reserve(n_events);
    }
    if (_keep_event_metadata) {
      (*_event_metadata).reserve(n_events);
    }
  }

  inline bool KeepNames() const {
    return _keep_names;
  }

  inline bool KeepEventMetadata() const {
    return _keep_event_metadata;
  }

  inline const std::set<std::string>& UniqueNames() const {
    return _names.Keys();
  }

  inline const IEventMetadata* GetEventMetadata(size_t i) const {
    assert(i < _next_event_to_set);
    // assert(_keep_event_metadata);
    if (!_keep_event_metadata) {
      return nullptr;
    }
    return (*_event_metadata)[i].get();
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
  void CheckIntegrity(std::ostream& out, int indent) const;

  inline void SetEvent(OptionalString name, size_t i, TimeUsec start_us, TimeUsec end_us, std::unique_ptr<IEventMetadata> event_metadata = nullptr) {
    SetEventPsec(name, i, start_us*PSEC_IN_USEC, end_us*PSEC_IN_USEC, std::move(event_metadata));
  }

  inline void SetEventPsec(OptionalString name, size_t i, TimePsec start_ps, TimePsec end_ps, std::unique_ptr<IEventMetadata> event_metadata = nullptr) {

//    if (start_ps == 1571629741190258000 && end_ps == 1571629774546157000) {
//      DBG_BREAKPOINT("A");
//    }
//    if (start_ps == 1571629741190322000 && end_ps == 1571629741195270000) {
//      DBG_BREAKPOINT("B");
//    }

    assert(i < _max_events);
    assert(i == _next_event_to_set);
    assert(start_ps <= end_ps);
    auto start_idx = EVENT_START_IDX(i);
    auto end_idx = EVENT_END_IDX(i);
    if (_keep_names) {
      assert(name.has_value());
      _names.push_back(name.value());
    }
    if (_keep_event_metadata) {
      // assert(event_metadata != nullptr);
      (*_event_metadata).push_back(std::move(event_metadata));
    }
    if (_next_event_to_set == 0 || start_ps < _min_start_ps) {
      _min_start_ps = start_ps;
    }
    if (_next_event_to_set == 0 || end_ps > _max_end_ps) {
      _max_end_ps = end_ps;
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

  inline TimeUsec MinStartUsec() const {
    return _min_start_ps/PSEC_IN_USEC;
  }

  inline TimeUsec MaxEndUsec() const {
    return _max_end_ps/PSEC_IN_USEC;
  }

  inline TimeUsec StartUsec(size_t i) const {
    return StartPsec(i)/PSEC_IN_USEC;
  }

  inline TimeUsec EndUsec(size_t i) const {
    return EndPsec(i)/PSEC_IN_USEC;
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

  inline void AppendEvent(OptionalString name, TimeUsec start_us, TimeUsec end_us, std::unique_ptr<IEventMetadata> event_metadata = nullptr) {
    SetEvent(name, _next_event_to_set, start_us, end_us, std::move(event_metadata));
  }

  inline void AppendEventPsec(OptionalString name, TimePsec start_ps, TimePsec end_ps, std::unique_ptr<IEventMetadata> event_metadata = nullptr) {
    SetEventPsec(name, _next_event_to_set, start_ps, end_ps, std::move(event_metadata));
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
    return cmap.count(category_key) > 0;
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

  template <typename Func>
  void RemapKeysInplace(Func remap_key_func) {
    auto remap_eo_times = [remap_key_func] (const EOTimes& times) -> EOTimes {
      EOTimes remapped_eo_times;
      for (auto const& pair : times) {
        const auto& old_key = pair.first;
        auto new_key = remap_key_func(old_key);
        remapped_eo_times[new_key] = pair.second;
      }
      return remapped_eo_times;
    };
    auto remapped_eo_times = remap_eo_times(this->eo_times);
    auto remapped_extra_eo_times = remap_eo_times(this->extra_eo_times);
    eo_times = remapped_eo_times;
    extra_eo_times = remapped_extra_eo_times;
  }

  void MoveInto(CategoryTimes& category_times);
  void MoveEventsInto(const CategoryKey& category_key, EOEvents& eo_events);

  void MergeEventsInto(const CategoryKey& category_key, EOEvents& eo_events);

  void Preallocate(const CategoryTimesCount& count);
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

  void CheckIntegrity(std::ostream& out, int indent) const;

  static EOTimes _EOTimesMergeAll(std::list<const EOTimes*> all_eo_times) {
    EOTimes merged;
    for (const auto& eo_times : all_eo_times) {
      for (const auto& pair : *eo_times) {
        auto it = merged.find(pair.first);
        if (it != merged.end()) {
          merged[pair.first] = EOEvents::Merge(it->second, pair.second);
        } else {
          merged[pair.first] = pair.second;
        }
      }
    }
    return merged;
  }

  static CategoryTimes MergeAll(std::list<const CategoryTimes*> all_category_times) {
    CategoryTimes merged;
    if (all_category_times.size() == 0) {
      return merged;
    }
    std::list<const EOTimes*> all_eo_times;
    std::list<const EOTimes*> all_extra_eo_times;
    for (auto const& category_times : all_category_times) {
      all_eo_times.push_back(&category_times->eo_times);
      all_extra_eo_times.push_back(&category_times->extra_eo_times);
    }
    merged.eo_times = _EOTimesMergeAll(all_eo_times);
    merged.extra_eo_times = _EOTimesMergeAll(all_extra_eo_times);
    // Should we make it "", since it contais multiple processes?
    // Or, make a member "processes" that contains all of them as a set?
    merged.process = (*all_category_times.begin())->process;

    return merged;
  }

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
      assert(eo_times.find(category) == eo_times.end());
      eo_times[category] = pair.second;
    }
    assert(category_times.size() == this->size());
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

// Navigate trace-files for a particular (machine, process, phase) in the --rlscope-directory in time-stamp order.
// Used by RawTraceParser for reading events into eo_times in the correct order.
struct EntireTraceMeta {
  Machine machine;
  Process process;
  Phase phase;

  std::set<Machine> machines;
  std::set<Process> processes;
  std::set<Phase> phases;

  EntireTraceMeta() = default;
  EntireTraceMeta(
      const Machine& machine_,
      const Process& process_,
      const Phase& phase_) :
      machine(machine_)
      , process(process_)
      , phase(phase_)
  {
    machines.insert(machine);
    processes.insert(process);
    phases.insert(phase);
  }
  template <typename OStream, typename Value>
  void _PrintField(OStream& out, int indent, const std::string& name, const Value& value) const {
    out << "\n";
    PrintIndent(out, indent);
    out << name << " = ";
    PrintValue(out, value);
  }
  template <typename OStream>
  void Print(OStream& out, int indent) const {
    PrintIndent(out, indent);

    out << "EntireTraceMeta:";
    _PrintField(out, indent + 1, "machines", machines);
    _PrintField(out, indent + 1, "processes", processes);
    _PrintField(out, indent + 1, "phases", processes);
  }

};

// Parameters that determine how to parse proto files.
struct EntireTraceSelector {
  bool ignore_memcpy;
  EntireTraceSelector() :
      ignore_memcpy(false) {
  }

  template <typename OStream, typename Value>
  void _PrintField(OStream& out, int indent, const std::string& name, const Value& value) const {
    out << "\n";
    PrintIndent(out, indent);
    out << name << " = " << value;
  }
  template <typename OStream>
  void Print(OStream& out, int indent) const {
    PrintIndent(out, indent);

    out << "EntireTraceSelector:";
    _PrintField(out, indent + 1, "ignore_memcpy", ignore_memcpy);

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
  RLSAnalyzeArgs args;
  std::string path;

  CategoryTimesCount count;
  Machine machine;
  Process process;
  Phase phase;
  TraceID trace_id;
  RLSFileType file_type;

  std::map<Category, size_t> n_events;
  std::set<Category> categories;

//  std::shared_ptr<IParserMeta> parser_meta;

  nlohmann::json parser_meta;

  bool initialized;

  TraceFileMeta() :
      trace_id(0),
      file_type(RLSFileType::UNKNOWN_FILE),
      initialized(false)
  {
  }
  TraceFileMeta(RLSAnalyzeArgs args, std::string const& path) :
      args(std::move(args)),
      path(path),
      trace_id(0),
      file_type(GetRLSFileType(path)),
      initialized(false)
  {
    assert(isRLSFile(path));
  }

  template <typename OStream, typename Value>
  void _PrintField(OStream& out, int indent, const std::string& name, const Value& value) const {
    out << "\n";
    PrintIndent(out, indent);
    out << name << " = " << value;
  }
  template <typename OStream>
  void Print(OStream& out, int indent) const {
    PrintIndent(out, indent);

    out << "TraceFileMeta:";
    _PrintField(out, indent + 1, "path", path);
    _PrintField(out, indent + 1, "file_type", RLSFileTypeString(file_type));
    _PrintField(out, indent + 1, "machine", machine);
    _PrintField(out, indent + 1, "process", process);
    _PrintField(out, indent + 1, "phase", phase);
    _PrintField(out, indent + 1, "trace_id", trace_id);

  }

  template <typename OStream>
  friend OStream& operator<<(OStream& os, const TraceFileMeta& obj) {
    os << "TraceFileMeta("
       << "path=" << obj.path
       << ", file_type=" << RLSFileTypeString(obj.file_type)
       << ", machine=" << obj.machine
       << ", process=" << obj.process
       << ", phase=" << obj.phase
       << ", trace_id=" << obj.trace_id
       << ")";
    return os;
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
  RLSAnalyzeArgs args;
  std::shared_ptr<SimpleTimer> timer;
  // RLSFilePath -> TraceFileMeta
  std::map<std::string, TraceFileMeta> _path_to_meta;
  // [machine_name][process_name][phase_name][trace_id] -> TraceFileMeta
  std::map<Machine,
  std::map<Process,
  std::map<Phase,
  std::map<RLSFileType,
  std::map<TraceID, TraceFileMeta>>>>> _meta;

  std::string _rlscope_directory;

  TraceFileWalker(RLSAnalyzeArgs args) :
      args(args)
      , _rlscope_directory(args.FLAGS_rlscope_directory.value())
  {
  }

  void SetTimer(std::shared_ptr<SimpleTimer> timer) {
    this->timer = timer;
  }

  MyStatus ReadMeta(const std::string& path, TraceFileMeta* meta);
  MyStatus TraceMetas(RLSFileType file_type, const Machine& machine, const Process& process, const Phase& phase, std::vector<TraceFileMeta>* metas);

  std::set<RLSFileType> FileTypes() const;
  std::list<Machine> Machines() const;
  std::list<Process> Processes(const Machine& machine) const;
  std::list<Phase> Phases(const Machine& machine, const Process& process) const;

  template <typename OStream>
  void Print(OStream& out, int indent) const {
    PrintIndent(out, indent);
    out << "TraceFileWalker: size = " << _path_to_meta.size();
    size_t i = 0;
    for (const auto& pair : _path_to_meta) {
      auto const& path = pair.first;
      auto const& meta = pair.second;
      out << "\n";
      PrintIndent(out, indent + 1);
      out << "path[" << i << "] = " << path;
      out << "\n";
      meta.Print(out, indent + 2);
      i += 1;
    }
  }

  MyStatus Init();
};

class RawTraceParser {
public:
  RLSAnalyzeArgs args;
  std::shared_ptr<SimpleTimer> timer;
  std::string _rlscope_directory;
  TraceFileWalker _walker;

  nlohmann::json _cupti_overhead_json;
  nlohmann::json _LD_PRELOAD_overhead_json;
  nlohmann::json _python_annotation_json;
  nlohmann::json _python_clib_interception_tensorflow_json;
  nlohmann::json _python_clib_interception_simulator_json;

  bool _has_calibration_files;

  RawTraceParser(RLSAnalyzeArgs args) :
      args(args),
      _rlscope_directory(args.FLAGS_rlscope_directory.value()),
      _walker(args),
      _has_calibration_files(false)
  {

  }
  // ASSUMPTION: all events i have category.events[i].start_us <= category.events[i+1].start_us
  // Q: What do we do if this assumption FAILS?
  // A: This assumption should hold with a thread...if it fails split by thread-id?
  //
  // Basic: Single eo_times:
  // - Create an array of int64_t and read usec->psec for each category.
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

  void SetTimer(std::shared_ptr<SimpleTimer> timer) {
    this->timer = timer;
    _walker.SetTimer(timer);
  }

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
  MyStatus EachEntireTrace(Func func, const EntireTraceSelector& selector) {
    if (args.FLAGS_cross_process.value()) {
      return CrossProcessEachEntireTraceWithFileType(func, RLS_FILE_TYPES, selector);
    } else {
      return EachEntireTraceWithFileType(func, RLS_FILE_TYPES, selector);
    }
  }

  MyStatus CrossProcessReadEntireTrace(
      const std::set<RLSFileType>& file_types,
      CategoryTimes *category_times,
      EntireTraceMeta* entire_meta,
      const EntireTraceSelector& selector);

  template <typename Func>
  MyStatus CrossProcessEachEntireTraceWithFileType(Func func, const std::set<RLSFileType>& file_types, const EntireTraceSelector& selector) {
    MyStatus status = MyStatus::OK();
    std::unique_ptr<CategoryTimes> category_times(new CategoryTimes());
    EntireTraceMeta meta;
    status = this->CrossProcessReadEntireTrace(
        file_types,
        category_times.get(), &meta, selector);
    IF_BAD_STATUS_RETURN(status);
    status = func(std::move(category_times), meta);
    IF_BAD_STATUS_RETURN(status);
    return MyStatus::OK();
  }

  template <typename Func>
  MyStatus EachEntireTraceWithFileType(Func func, const std::set<RLSFileType>& file_types, const EntireTraceSelector& selector) {
    MyStatus status = MyStatus::OK();
    auto const& machines = this->Machines();
    for (auto const& machine : machines) {
      auto const& processes = this->Processes(machine);
      for (auto const& process : processes) {
        auto const& phases = this->Phases(machine, process);
        for (auto const &phase : phases) {
          std::unique_ptr<CategoryTimes> category_times(new CategoryTimes());
          EntireTraceMeta meta;
          status = this->ReadEntireTrace(
              machine, process, phase,
              file_types,
              category_times.get(), &meta, selector);
          IF_BAD_STATUS_RETURN(status);
          status = func(std::move(category_times), meta);
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
      const std::set<RLSFileType>& file_types,
      CategoryTimes* category_times,
      EntireTraceMeta* entire_meta,
      const EntireTraceSelector& selector);

//  MyStatus _ReadOneFileSequential(
//      const Machine& machine,
//      const Process& process,
//      const Phase& phase,
//      CategoryTimes *category_times,
//      EntireTraceMeta* entire_meta,
//      const std::map<RLSFileType, std::vector<TraceFileMeta>>& meta_map,
//      const std::map<RLSFileType, std::unique_ptr<IEventFileParser>>& parser_map);

  using EachReadMergeSortedFunc = std::function<MyStatus(RLSFileType rls_file_type, const std::vector<TraceFileMeta>& metas)>;
  MyStatus _ReadMergeSorted(
      const std::set<RLSFileType>& file_types,
      CategoryTimes *category_times,
      EntireTraceMeta* entire_meta,
      const std::map<RLSFileType, std::vector<TraceFileMeta>>& meta_map,
      const std::map<RLSFileType, std::unique_ptr<IEventFileParser>>& parser_map,
      EachReadMergeSortedFunc func);

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
      CategoryTimes *category_times,
      const Category& category_prof,
      const std::set<Category>& c_events_categories,
      double per_python_clib_interception_us);
  MyStatus _AppendOverhead_PYTHON_ANNOTATION(
      const Machine& machine,
      const Process& process,
      const Phase& phase,
      CategoryTimes *category_times);

//  CategoryTimes RawTraceParser::ReadEntireTrace(const std::string& rlscope_directory) {
//  }
//  // E = each split should have <= E events across all categories.
//  std::list<TraceSplit> FindSplits(const std::string& rlscope_directory, size_t E);
//  // Read CategoryTimes belonging to this split.
//  CategoryTimes ReadSplit(const std::string& rlscope_directory, const TraceSplit& split);

};

// TODO: remove?
struct TraceParserMeta {

  Machine machine;
  Process process;
  Phase phase;

  TraceParserMeta();
  TraceParserMeta(
      const Machine& machine,
      const Process& process,
      const Phase& phase) :
      machine(machine),
      process(process),
      phase(phase) {
  }
};

class ITraceFileReader {
public:

  virtual ~ITraceFileReader() = default;

  // Implemented in IEventFileProtoParser<ProtoKlass>
//  virtual bool IsFile(const std::string& path) const = 0;
//  virtual MyStatus ReadFile(CategoryTimes* out_category_times) = 0;
  virtual MyStatus Init() = 0;
  virtual void Clear() = 0;
  virtual MyStatus ReadMeta(nlohmann::json* meta) = 0;
  virtual MyStatus ReadTraceFileMeta(TraceFileMeta* meta) = 0;

  virtual const Machine& get_machine() const = 0;
  virtual const Process& get_process() const = 0;
  virtual const Phase& get_phase() const = 0;

};

class IEventFileParser {
public:
  CategoryTimesCount _count;
  TraceParserMeta _meta;

  virtual ~IEventFileParser() = default;

  // IEventFileParser() = default;
  IEventFileParser(TraceParserMeta meta) :
      _meta(std::move(meta))
  {
  }

//  virtual MyStatus CountCategoryTimes(const std::string& path) = 0;
//  virtual MyStatus AppendCategoryTimes(const std::string& path, CategoryTimes* out_category_times) = 0;

//  virtual MyStatus CountCategoryTimes(const std::vector<TraceFileMeta>& metas) = 0;
//  virtual MyStatus AppendCategoryTimes(const std::vector<TraceFileMeta>& metas, CategoryTimes* out_category_times) = 0;

  virtual MyStatus AppendAllCategoryTimes(const EntireTraceMeta& entire_meta, const std::vector<TraceFileMeta>& metas, CategoryTimes* out_category_times) = 0;

  virtual const CategoryTimesCount& GetCount() const {
    return _count;
  }
  virtual CategoryTimesCount* MutableGetCount() {
    return &_count;
  }

  virtual MyStatus Init() = 0;
//  virtual std::shared_ptr<IParserMeta> GetParserMeta() = 0;

};

struct TraceFileLocation {
  std::map<Category, size_t> event_idx;
  std::map<Category, TimeUsec> start_ps;
};

struct SimpleEvent {
  std::string _name;
  TimeUsec _start_time_us;
  TimeUsec _end_time_us;;

  // STL.
  SimpleEvent() = default;

  SimpleEvent(const std::string& name, TimeUsec start_time_us, TimeUsec end_time_us) :
      _name(name),
      _start_time_us(start_time_us),
      _end_time_us(end_time_us)
  {
  }

  TimeUsec start_time_us() const {
    return _start_time_us;
  }

  TimeUsec end_time_us() const {
    return _end_time_us;
  }

  const std::string& name() const {
    return _name;
  }
};

class SimpleEventReader {
public:
  std::vector<TraceFileMeta> metas;

  SimpleEventReader(std::vector<TraceFileMeta> metas)
      : metas(std::move(metas))
  {
  }

  template <class ProtoReader, class EventKlass>
  MyStatus ReadCategoryEvents(const Category& category, std::vector<EventKlass>* events) const {
    size_t n_events = 0;
    for (const auto& meta : metas) {
      auto const& it = meta.n_events.find(category);
      if (it != meta.n_events.end()) {
        n_events += it->second;
      }
    }

    events->reserve(n_events);
    for (const auto& meta : metas) {
      if (meta.categories.find(category) == meta.categories.end()) {
        continue;
      }
      ProtoReader reader(meta.get_path());
      auto status = reader.Init();
      IF_BAD_STATUS_RETURN(status);

      reader.MutableEachCategory([n_events, &events, category] (const auto& cat, auto* cat_events) {
        DBG_BREAKPOINT("negative event duration");
        if (category != cat) {
          return;
        }
        SortEvents(cat_events);
        auto n_events_so_far = events->size();
        assert(events->size() + cat_events->size() <= n_events);
        std::copy(cat_events->begin(), cat_events->end(), std::back_inserter(*events));
        std::inplace_merge(
            events->begin(),
            events->begin() + n_events_so_far,
            events->end(),
            [] (const EventKlass& lhs, const EventKlass& rhs) {
              return ProtoReader::EventStartUsec(lhs) < ProtoReader::EventStartUsec(rhs);
            });
      });
    }
    assert(events->size() == n_events);
    return MyStatus::OK();
  }

  // NOTE: This doesn't work yet... still debugging.
  template <class ProtoReader, class EventKlass>
  MyStatus ReadCategoryEventsParallel(const Category& category, std::vector<SimpleEvent>* events) const {

    auto cmp_events = [] (const SimpleEvent& lhs, const SimpleEvent& rhs) -> bool {
      return lhs.start_time_us() < rhs.start_time_us();
    };

  // - Stage 1: Read into vector (parallel for each trace-file)
  //   - Read ALL proto files into single std::vector
  //   - Limit number of threads that can read in parallel to prevent a large memory footprint.
  //     (just use number of cores)
  // - Stage 2: Sort within each vector (parallel for each trace-file)
  //   - Q: Is there an algorithm we should use for sorting a mostly sorted list? Insertion/bubble sort… screw it.
  // - Stage 3: Merge
  //   - Tree-reduce style merge_inplace
  //   - How to implement this...?
  //   - Threadpool with index pairs:
  //     - (start, middle, end)
  //    - We know events are mostly sorted, so I would expect when we do merge(list1, list2),
  //      list1 AND list2 would be left mostly intact, with some events from list2 making there way into list1.


    size_t n_events = 0;
    for (const auto& meta : metas) {
      auto const& it = meta.n_events.find(category);
      if (it != meta.n_events.end()) {
        n_events += it->second;
      }
    }

    auto num_threads = std::thread::hardware_concurrency();

    // - Stage 1: Read into vector (parallel for each trace-file)
    //   - Read ALL proto files into single std::vector
    //   - Limit number of threads that can read in parallel to prevent a large memory footprint.
    //     (just use number of cores)
    // - Stage 2: Sort within each vector (parallel for each trace-file)
    //   - Q: Is there an algorithm we should use for sorting a mostly sorted list? Insertion/bubble sort… screw it.
    // events->reserve(n_events);

    // Otherwise events.size() returns 0 after parallel insertion via assigment...
    events->resize(n_events);

    std::vector<bool> has_category(metas.size());
    std::vector<size_t> events_start_idx(metas.size());
    std::vector<size_t> events_end_idx(metas.size());
    std::vector<size_t> cat_size(metas.size());
    size_t category_events_so_far = 0;
    for (size_t i = 0; i < metas.size(); i++) {
      const auto& meta = metas[i];
      if (meta.categories.find(category) == meta.categories.end()) {
        events_start_idx[i] = category_events_so_far;
        events_end_idx[i] = category_events_so_far;
        has_category[i] = false;
        cat_size[i] = 0;
        continue;
      }
      const auto n_cat_events = meta.n_events.at(category);
      events_start_idx[i] = category_events_so_far;
      events_end_idx[i] = category_events_so_far + n_cat_events;
      has_category[i] = true;
      cat_size[i] = n_cat_events;
      category_events_so_far += n_cat_events;
    }

    {
      ctpl::thread_pool pool(num_threads);
      std::vector<std::future<void>> results;
      for (size_t i = 0; i < metas.size(); i++) {
        const auto& meta = metas[i];
        if (meta.categories.find(category) == meta.categories.end()) {
          continue;
        }
        size_t start_idx = events_start_idx[i];
        DBG_LOG("DO: parallel.read path = {}, start={}, end={}",
                meta.get_path(),
                start_idx,
                start_idx + cat_size[i]);
        results.push_back(pool.push([start_idx, &meta, n_events, &events, category] (int) {
          // DBG_LOG("parallel.read path = {}", meta.get_path());
          ProtoReader reader(meta.get_path());
          auto status = reader.Init();
          assert(status.code() == MyStatus::OK().code());
          // IF_BAD_STATUS_RETURN(status);
          reader.EachCategory([start_idx, &meta, n_events, &events, category] (const auto& cat, const auto& cat_events) {
            if (category != cat) {
              return;
            }
//            SortEvents(cat_events);
            assert(start_idx + cat_events.size() <= n_events);
            DBG_LOG("parallel.read path = {}, start={}, end={}",
                meta.get_path(),
                start_idx,
                start_idx + cat_events.size());

            auto events_it = events->begin() + start_idx;
            for (auto it = cat_events.cbegin(); it != cat_events.cend(); it++, events_it++) {
              *events_it = SimpleEvent(
                  ProtoReader::EventName(*it),
                  ProtoReader::EventStartUsec(*it),
                  ProtoReader::EventEndUsec(*it));
            }
            std::sort(
                events->begin() + start_idx,
                events->begin() + start_idx + cat_events.size(),
                // [] (const EventKlass* lhs, const EventKlass* rhs) {
                [] (const auto& lhs, const auto& rhs) {
                  return lhs.start_time_us() < rhs.start_time_us();
                });

            // WARNING: this causes a SEGFAULT because internally protobuf appears to be doing
            // non-thread-safe shared-memory crap when events get swapped!
            // SOLUTION: don't use protobuf's for anything here or afterwards.
//            auto events_it = events->begin();
//            std::advance(events_it, start_idx);
//            std::copy(cat_events->begin(), cat_events->end(),
//                      std::inserter(
//                          *events,
//                          // events->begin() + start_idx
//                          events_it
//                          ));

          });
        }));
      }
      for (auto& result : results) {
        result.get();
      }
    }
    assert(events->size() == n_events);

//    {
//      // Simple sequential merge.
//      if (metas.size() > 0) {
//        size_t events_so_far = 0;
//        size_t middle = cat_size[0];
//        size_t end = cat_size[0];
//        for (size_t i = 1; i < metas.size() - 1; i++) {
//          const auto& meta = metas[i];
//          if (meta.categories.find(category) == meta.categories.end()) {
//            continue;
//          }
//          end += cat_size[i];
//          assert(has_category[i]);
//          std::inplace_merge(
//              events->begin(),
//              events->begin() + middle,
//              events->begin() + end,
//              cmp_events);
//          middle += cat_size[i];
//        }
//      }
//    }

    {
      // - Stage 3: Tree-reduce style merge_inplace
      //   - How to implement this...?
      //   - Threadpool with index pairs:
      //     - (start, middle, end)
      //    - We know events are mostly sorted, so I would expect when we do merge(list1, list2),
      //      list1 AND list2 would be left mostly intact, with some events from list2 making there way into list1.

      struct EventRegion {
        size_t start;
        size_t end;
        EventRegion(size_t start, size_t end) :
            start(start),
            end(end) {
        }
      };


      std::vector<EventRegion> event_regions;
      event_regions.reserve(metas.size());
      for (size_t i = 0; i < metas.size(); i++) {
        const auto &meta = metas[i];
        if (meta.categories.find(category) == meta.categories.end()) {
          continue;
        }
        event_regions.push_back(EventRegion(
            events_start_idx[i],
            events_end_idx[i]));
      }

      ctpl::thread_pool pool(num_threads);
      while (event_regions.size() > 1) {
        std::vector<std::future<EventRegion>> new_regions_fut;
        new_regions_fut.reserve(event_regions.size());
        // For each consecutive pair of event regions, merge them.
        for (size_t i = 0; i < event_regions.size() - 1; i += 2) {
          const auto& e0 = event_regions[i];
          const auto& e1 = event_regions[i+1];
          assert(e0.end == e1.start);
          size_t start = e0.start;
          size_t middle = e0.end;
          size_t end = e1.end;
          assert(end <= n_events);
          new_regions_fut.push_back(pool.push([start, middle, end, &events, cmp_events] (int) {
            std::inplace_merge(
                events->begin() + start,
                events->begin() + middle,
                events->begin() + end,
                cmp_events);
            return EventRegion(start, end);
          }));
        }
        std::vector<EventRegion> new_regions;
        new_regions.reserve(new_regions_fut.size());
        for (auto& region : new_regions_fut) {
          new_regions.push_back(region.get());
        }
        event_regions = std::move(new_regions);
      }

    }

    return MyStatus::OK();
  }

};


template <class ProtoKlass, class ProtoReader>
class IEventFileProtoParser : public IEventFileParser {
public:
  RLSFileType _file_type;
  std::string _proto_nickname;

  virtual ~IEventFileProtoParser() = default;

  IEventFileProtoParser(TraceParserMeta meta, RLSFileType file_type
      , const std::string& proto_nickname
      ) :
      IEventFileParser(meta),
      _file_type(file_type),
      _proto_nickname(proto_nickname)
  {
  }

  virtual bool IsFile(const std::string& path) const {
    return isRLSFileWithType(_file_type, path);
  }

  virtual MyStatus Init() override {
    return MyStatus::OK();
  }

  virtual MyStatus AppendAllCategory(
      const EntireTraceMeta& entire_meta,
      const Category& category,
      const std::vector<typename ProtoReader::EventKlass>& events,
      CategoryTimes* out_category_times) {
    return DefaultAppendAllCategory(
        entire_meta,
        category,
        events,
        out_category_times);
  }

  virtual MyStatus DefaultAppendAllCategory(
      const EntireTraceMeta& entire_meta,
      const Category& category,
      const std::vector<typename ProtoReader::EventKlass>& events,
      CategoryTimes* out_category_times) {
    auto category_key = CategoryKey::FromCategory(entire_meta.process, category);
    out_category_times->Preallocate(category_key, events.size());
    auto& eo_events = out_category_times->MutableEvents(category_key);
    for (size_t i = 0; i < events.size(); i++) {
      auto start_us = ProtoReader::EventStartUsec(events[i]);
      auto end_us = ProtoReader::EventEndUsec(events[i]);
      OptionalString name;
      if (eo_events.KeepNames()) {
        name = ProtoReader::EventName(events[i]);
      }
      if (end_us < start_us) {
        DBG_BREAKPOINT("negative event duration");
        DBG_LOG("BUG: skip negative duration Event(name=\"{}\", start_us={}, duration_us={} us)",
                ProtoReader::EventName(events[i]), start_us, end_us - start_us);
        // Just insert a zero-length event since we've already preallocated space for it
        // (they're effectively be ignored during overlap).
        end_us = start_us;
      }
      eo_events.AppendEvent(name, start_us, end_us);
    }
    return MyStatus::OK();
  }

  virtual MyStatus DefaultAppendAllCategoryExtra(
      const EntireTraceMeta& entire_meta,
      const Category& category,
      const std::vector<typename ProtoReader::EventKlass>& events,
      CategoryTimes* out_category_times) {
    auto category_key = CategoryKey::FromCategory(entire_meta.process, category);
    out_category_times->PreallocateExtra(category_key, events.size());
    auto& eo_events = out_category_times->MutableEventsExtra(category_key);
    for (size_t i = 0; i < events.size(); i++) {
      auto start_us = ProtoReader::EventStartUsec(events[i]);
      auto end_us = ProtoReader::EventEndUsec(events[i]);
      OptionalString name;
      if (eo_events.KeepNames()) {
        name = ProtoReader::EventName(events[i]);
      }
      eo_events.AppendEvent(name, start_us, end_us);
    }
    return MyStatus::OK();
  }

  virtual MyStatus AppendAllCategoryTimes(const EntireTraceMeta& entire_meta, const std::vector<TraceFileMeta>& metas, CategoryTimes* out_category_times) override {
    auto status = MyStatus::OK();
    SimpleEventReader simple_reader(metas);
    std::set<Category> categories;
    std::set<Process> processes;
    for (const auto& meta : metas) {
      categories.insert(meta.categories.begin(), meta.categories.end());
    }
    for (const auto& category : categories) {
      std::vector<typename ProtoReader::EventKlass> events;
      status = simple_reader.ReadCategoryEvents<ProtoReader>(category, &events);
      IF_BAD_STATUS_RETURN(status);

      // Lazy quick testing: compare serial against parallel result.
      // NOTE: this still doesn't work yet, still debugging...
//      std::vector<SimpleEvent> parallel_events;
//      status = simple_reader.ReadCategoryEventsParallel<ProtoReader, typename ProtoReader::EventKlass>(category, &parallel_events);
//      IF_BAD_STATUS_RETURN(status);
//
//      assert(events.size() == parallel_events.size());
//      for (size_t i = 0; i < events.size(); i++) {
//        assert(ProtoReader::EventStartUsec(events[i]) == parallel_events[i].start_time_us());
//        assert(ProtoReader::EventEndUsec(events[i]) == parallel_events[i].end_time_us());
//        assert(ProtoReader::EventName(events[i]) == parallel_events[i].name());
//      }

      status = this->AppendAllCategory(entire_meta, category, events, out_category_times);
      IF_BAD_STATUS_RETURN(status);

    }
    return MyStatus::OK();
  }


};


std::vector<std::string> ParseCSVRow(const std::string& line);

enum NvprofFileTypeCode {
  NVPROF_CSV_UNKNOWN = 0,
  NVPROF_CSV_API_TRACE = 1,
  NVPROF_CSV_GPU_TRACE = 2,
};
class NvprofFileType {
public:
  struct EventRow {
    std::string name;
    TimeUsec start_us;
    TimeUsec end_us;
    std::vector<std::string> event_metadata;
    EventRow() : start_us(0), end_us(0) {
    }
  };
  struct HeaderMeta {
    size_t start_idx;
    size_t duration_idx;
    size_t name_idx;
    std::map<std::string, size_t> col_idx_map;
    // --nvprof_keep_column_names
    std::vector<std::string> event_metadata_cols;
  };
  std::vector<std::string> header;
  NvprofFileTypeCode file_type;
  Category category;

  NvprofFileType(
      std::vector<std::string> header,
      NvprofFileTypeCode file_type,
      Category category) :
      header(header)
      , file_type(file_type)
      , category(category)
  {
  }
  // API trace files contain cudaLaunchKernel start/end durations.
  // - category = CATEGORY_CUDA_API_CPU
  // GPU trace files contain GPU kernel start/end durations.
  // - category = CATEGORY_GPU
  // CategoryKey:
  //   process = entire_meta.process
  //   categoy = ^^^
  // auto category_key = CategoryKey::FromCategory(entire_meta.process, category);
  // NOTE: I think the category is the same for all the events in these csv files...
  // Only additional events we need to create that don't match this category is the
  // "fake" operation event (CATEGORY_OPERATION).
  virtual std::string RowCategory() const = 0;
  virtual MyStatus ParseRowEvent(const HeaderMeta& header_meta, const std::vector<std::string>& row, NvprofFileType::EventRow* event_row) const = 0;
  bool HeaderMatches(const std::vector<std::string>& row) const;
  virtual HeaderMeta ParseHeaderMeta(const RLSAnalyzeArgs& args, const std::vector<std::string>& row) const = 0;
  std::map<std::string, size_t> ParseColIdxMap(const std::vector<std::string>& row) const;

  CategoryKey RowCategoryKey(const Process& proc) const {
    return CategoryKey::FromCategory(proc, RowCategory());
  }
};
class NvprofAPITraceFileType : public NvprofFileType {
public:
  NvprofAPITraceFileType() :
      NvprofFileType(
          {"Start", "Duration", "Name", "Correlation_ID"},
          NVPROF_CSV_API_TRACE,
          CATEGORY_CUDA_API_CPU)
  {
  }
  virtual std::string RowCategory() const override {
    return CATEGORY_CUDA_API_CPU;
  }

  virtual HeaderMeta ParseHeaderMeta(const RLSAnalyzeArgs& args, const std::vector<std::string>& row) const override;
  virtual MyStatus ParseRowEvent(const HeaderMeta& header_meta, const std::vector<std::string>& row, NvprofFileType::EventRow* event_row) const override;
};
class NvprofGPUTraceFileType : public NvprofFileType {
public:
  NvprofGPUTraceFileType() :
      NvprofFileType(
          {
              "Start",
              "Duration",
              "Grid X",
              "Grid Y",
              "Grid Z",
              "Block X",
              "Block Y",
              "Block Z",
//              "Registers Per Thread",
//              "Static SMem",
//              "Dynamic SMem",
// NOTE: This columns aren't always present...
//              "Size",
//              "Throughput",
//              "SrcMemType",
//              "DstMemType",
              "Device",
              "Context",
              "Stream",
              "Name",
              "Correlation_ID",
          },
          NVPROF_CSV_GPU_TRACE,
          CATEGORY_GPU)
  {
  }
  virtual std::string RowCategory() const override {
    return CATEGORY_CUDA_API_CPU;
  }

  virtual HeaderMeta ParseHeaderMeta(const RLSAnalyzeArgs& args, const std::vector<std::string>& row) const override;
  virtual MyStatus ParseRowEvent(const HeaderMeta& header_meta, const std::vector<std::string>& row, NvprofFileType::EventRow* event_row) const override;
};
static MyStatus GetNvprofFileType(
    const std::vector<std::string>& header,
    RLSFileType rls_file_type,
    std::unique_ptr<NvprofFileType>* ret) {
  switch (rls_file_type) {
    case NVPROF_API_TRACE_CSV_FILE:
      ret->reset(new NvprofAPITraceFileType());
      break;
    case NVPROF_GPU_TRACE_CSV_FILE:
      ret->reset(new NvprofGPUTraceFileType());
      break;
    default:
      std::stringstream ss;
      ret->reset(nullptr);
      ss << "Not sure what nvprof csv file type to use for RLS file type = " << RLSFileTypeString(rls_file_type);
      return MyStatus(error::INVALID_ARGUMENT, ss.str());
  }
//  for (const auto& nvprof_file_type : NVPROF_FILE_TYPES) {
//    if (nvprof_file_type->HeaderMatches(header)) {
//      *ret = nvprof_file_type;
//      return MyStatus::OK();
//    }
//  }
//  std::stringstream ss;
//  ss << "Not sure what nvprof csv file type this is for header that looks like:\n";
//  PrintValue(ss, header);
//  return MyStatus(error::INVALID_ARGUMENT, ss.str());
  return MyStatus::OK();
}
class NvprofEventMetadata : public IEventMetadata {
public:
  // const std::vector<std::string>& header;
  std::shared_ptr<std::vector<std::string>> header;
  std::vector<std::string> row;
  NvprofEventMetadata(
      std::shared_ptr<std::vector<std::string>> header,
  std::vector<std::string> row) : header(header), row(row) {
  }
  // Internally, we don't care how parser stores event metadata.
  // However, we do require that we can output the metadata to a standard format (csv).
  virtual const std::vector<std::string>& GetHeader() const override;
  virtual const std::vector<std::string>& GetRow() const override;
  virtual IEventMetadata* clone() const override;
};
class NvprofCSVParser : public IEventFileParser {
public:
  RLSAnalyzeArgs args;
  RLSFileType _file_type;
  std::shared_ptr<std::vector<std::string>> _event_metadata_header;

  virtual ~NvprofCSVParser() = default;

  NvprofCSVParser(RLSAnalyzeArgs args, TraceParserMeta meta, RLSFileType file_type) :
      IEventFileParser(meta)
      , args(args)
      , _file_type(file_type)
  {
    if (args.FLAGS_nvprof_keep_column_names.has_value()) {
      _event_metadata_header.reset(new std::vector<std::string>());
      *_event_metadata_header = args.FLAGS_nvprof_keep_column_names.value();
    }

  }

  virtual bool IsFile(const std::string& path) const {
    return isRLSFileWithType(_file_type, path);
  }

  virtual MyStatus Init() override {
    return MyStatus::OK();
  }

  virtual MyStatus AppendAllCategoryTimes(
      const EntireTraceMeta& entire_meta,
      const std::vector<TraceFileMeta>& metas,
      CategoryTimes* out_category_times) override;

  MyStatus _ReadEOEvents(
      const Category& category,
      const TraceFileMeta& meta,
      EOEvents* eo_events);

};


template <class ProtoKlass, class ProtoReader>
class ISimpleProtoParser : public IEventFileProtoParser<ProtoKlass, ProtoReader> {
public:

  virtual ~ISimpleProtoParser() = default;

  ISimpleProtoParser(TraceParserMeta meta, RLSFileType file_type, const std::string& proto_nickname) :
      IEventFileProtoParser<ProtoKlass, ProtoReader>(std::move(meta), file_type, proto_nickname) {
  }

//  virtual MyStatus AppendCategoryTimes(const std::vector<TraceFileMeta>& metas, CategoryTimes* out_category_times) override {
//    MyStatus status = MyStatus::OK();
//
//    for (const auto& meta : metas) {
//      ProtoReader reader(meta.get_path());
//      status = reader.Init();
//      IF_BAD_STATUS_RETURN(status);
//
//      reader.MutableEachCategory([] (const auto& category, auto* events) {
//        SortEvents(events);
//      });
//
//      status = _AppendCategoryTimes(out_category_times, reader.MutableProto());
//      IF_BAD_STATUS_RETURN(status);
//    }
//
//    return status;
//  }
//
//  virtual MyStatus CountCategoryTimes(const std::vector<TraceFileMeta>& metas) override {
//    MyStatus status = MyStatus::OK();
//
//    for (const auto& meta : metas) {
//      ProtoReader reader(meta.get_path());
//      status = reader.Init();
//      IF_BAD_STATUS_RETURN(status);
//
//      reader.MutableEachCategory([] (const auto& category, auto* events) {
//        SortEvents(events);
//      });
//
//      status = _CountCategoryTimes(&this->_count, reader.MutableProto());
//      IF_BAD_STATUS_RETURN(status);
//    }
//
//    return status;
//  }

};


template <class ProtoKlass, class ProtoReader>
class ITraceFileProtoReader : public ITraceFileReader {
public:
  ProtoKlass _proto;

  std::string _path;
  RLSFileType _file_type;
  std::string _proto_nickname;

  Machine _machine;
  Process _process;
  Phase _phase;

  bool _initialized;

  virtual ~ITraceFileProtoReader() = default;

//  std::shared_ptr<IParserMeta> _parser_meta;

  ITraceFileProtoReader(const std::string& path, RLSFileType file_type, const std::string& proto_nickname) :
      _path(path),
      _file_type(file_type),
      _proto_nickname(proto_nickname),
      _initialized(false)
  {
  }

  virtual const Machine& get_machine() const override {
    assert(_initialized);
    return _machine;
  }
  virtual const Process& get_process() const override {
    assert(_initialized);
    return _process;
  }
  virtual const Phase& get_phase() const override {
    assert(_initialized);
    return _phase;
  }

  virtual bool IsFile(const std::string& path) const {
    return isRLSFileWithType(_file_type, path);
  }

  virtual MyStatus _InitFromProto(const ProtoKlass& proto) {
    // Default: no extra initialization.
    return MyStatus::OK();
  }

  virtual MyStatus _ReadMetaFromProto(nlohmann::json* meta, const ProtoKlass& proto) {
    // Default: no extra initialization.
    return MyStatus::OK();
  }

  virtual const ProtoKlass& Proto() const {
    assert(_initialized);
    return _proto;
  }

  virtual ProtoKlass* MutableProto() {
    assert(_initialized);
    return &_proto;
  }

  virtual MyStatus Init() override {
    if (_initialized) {
      return MyStatus::OK();
    }
    // Initialization happens in _ReadProto.
    MyStatus status = MyStatus::OK();
    status = _ReadProto(_path, &_proto);

    _machine = ProtoReader::ProtoMachine(_proto);
    _process = ProtoReader::ProtoProcess(_proto);
    _phase = ProtoReader::ProtoPhase(_proto);

    IF_BAD_STATUS_RETURN(status);
    assert(_initialized);
    return MyStatus::OK();
  }

  virtual MyStatus ReadMeta(nlohmann::json* meta) override {
    MyStatus status = MyStatus::OK();
    status = Init();
    IF_BAD_STATUS_RETURN(status);
    status = _ReadMetaFromProto(meta, _proto);
    IF_BAD_STATUS_RETURN(status);
    return MyStatus::OK();
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

//  virtual MyStatus ReadFile(CategoryTimes* out_category_times) override {
//    MyStatus status = MyStatus::OK();
//
//    _proto.Clear();
//    status = this->_ReadProto(_path, &_proto);
//    IF_BAD_STATUS_RETURN(status);
//
//    return MyStatus::OK();
//  }

  virtual void Clear() override {
    _proto.Clear();
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

MyStatus GetRLSEventParser(const RLSAnalyzeArgs& args, const std::string& path, TraceParserMeta parser_meta, std::unique_ptr<IEventFileParser>* parser, const EntireTraceSelector& selector);
MyStatus GetRLSEventParserFromType(const RLSAnalyzeArgs& args, RLSFileType file_type, TraceParserMeta parser_meta, std::unique_ptr<IEventFileParser>* parser, const EntireTraceSelector& selector);
MyStatus GetTraceFileReader(const RLSAnalyzeArgs& args, const std::string& path, std::unique_ptr<ITraceFileReader>* reader);

using EachEventFunc = std::function<MyStatus(NvprofFileType::EventRow event_row)>;

template <typename EventProto>
class EventFlattener {
public:
  struct EventCompare {
    bool operator() (const EventProto* lhs, const EventProto* rhs) const {
      // NOTE: really subtle ordering constraint here needed for correctness.
      //
      //   A: [   ]
      //   B: [            ]
      //      0   1    2   3
      //      [ A ][   B   ]
      //
      //    1. EITHER
      //      - (START_EMPTY) A @ 0
      //      - (START) B @ 0
      //    2. (END) A @ 1
      //    3. (END) B @ 3
      //    To ensure (2, 3), we need to order by [event->start_time_us()] second.
      //
      //
      //   A:          [   ]
      //   B: [            ]
      //      0   1    2   3
      //      [   B   ][ A ]
      //
      //    1. (START_EMPTY) B @ 0
      //    2. (START) A @ 2
      //    3. (END) A @ 3
      //    4. (END) B @ 3
      //    To ensure (3, 4), we need to order by [-1 * event->start_time_us()] second.
      //
#define EVENT_KEY(event) \
  std::make_tuple(event->start_time_us() + event->duration_us(), -1 * event->start_time_us())
      return EVENT_KEY(lhs) < EVENT_KEY(rhs);
#undef EVENT_KEY
    }
  };

  template <typename EventProtoList, typename Func>
  static void EachOpEvent(const EventProtoList& events, Func func) {
    EventFlattener<EventProto> flattener;
    flattener.ProcessUntilFinish(events, func);
  }

  struct OpStack {
    using OpSet = std::set<const EventProto*, EventCompare>;
    OpSet op_set;
    TimeUsec start_us;
    TimeUsec end_us;

    std::map<TimeUsec, int> start_times;
    std::map<TimeUsec, int> end_times;

    OpStack() :
        start_us(std::numeric_limits<TimeUsec>::max()),
        end_us(std::numeric_limits<TimeUsec>::min())
    {
    }

    template <class Map, typename Key>
    static void _DecRef(Map* map, Key key) {
      auto it = map->find(key);
      assert(it != map->end());
      assert(it->second > 0);
      it->second -= 1;
      if (it->second == 0) {
        map->erase(it);
      }
    }

    template <class Map, typename Key>
    static void _IncRef(Map* map, Key key) {
      (*map)[key] += 1;
    }

    inline void insert(const EventProto* event) {
      _IncRef(&start_times, event->start_time_us());
      _IncRef(&end_times, GetEndTime(event));
      assert((start_times.size() == 0 && start_times.size() == 0) ||
             (start_times.size() > 0 && start_times.size() > 0));
      start_us = std::min(start_us, event->start_time_us());
      end_us = std::max(end_us, GetEndTime(event));

      op_set.insert(event);
    }

    inline const EventProto* PopNextEvent() {
      auto it = op_set.begin();
      auto min_event = *it;
      op_set.erase(it);

      _DecRef(&start_times, min_event->start_time_us());
      _DecRef(&end_times, GetEndTime(min_event));
      assert((start_times.size() == 0 && start_times.size() == 0) ||
             (start_times.size() > 0 && start_times.size() > 0));
      if (start_times.size() > 0) {
        start_us = start_times.begin()->first;
        end_us = end_times.rbegin()->first;
      } else {
        start_us = std::numeric_limits<TimeUsec>::max();
        end_us = std::numeric_limits<TimeUsec>::min();
      }

      return min_event;
    }

    inline const EventProto* PeekNextEvent() const {
      auto it = op_set.begin();
      auto min_event = *it;
      return min_event;
    }

    inline size_t size() const {
      return op_set.size();
    }

  };

  OpStack _ops;
  size_t _event_idx;
  TimeUsec _last_time;

  EventFlattener() :
      _event_idx(0),
      _last_time(std::numeric_limits<TimeUsec>::max()){
  }

  static inline bool Subsumes(const EventProto* A, const EventProto* B) {
    //     [ B ]
    // [     A     ]
    // return A->start_time_us() <= B->start_time_us() <= B->end_time_us() <= A->end_time_us();
    // ===
    // return A->start_time_us() <= B->start_time_us() <= A->end_time_us();
    return A->start_time_us() <= B->start_time_us() &&
           B->start_time_us() <= GetEndTime(A);
  }
  static inline bool Subsumes(const EventProto* A, TimeUsec time_us) {
    //     [ B ]
    // [     A     ]
    return A->start_time_us() <= time_us &&
           time_us <= GetEndTime(A);
  }
  static inline bool Subsumes(TimeUsec A_start, TimeUsec A_end, TimeUsec time_us) {
    //     [ B ]
    // [     A     ]
    return A_start <= time_us &&
           time_us <= A_end;
  }

  static inline TimeUsec GetEndTime(const EventProto* A) {
    return A->start_time_us() + A->duration_us();
  }

  template <typename EventProtoList, typename Func, typename SkipFunc>
  void ProcessUntilFinish(const EventProtoList& event_protos, Func func, SkipFunc skip_func) {
    auto max_time = std::numeric_limits<TimeUsec>::max();
    ProcessUntil(event_protos, max_time, func, skip_func);
  }
  template <typename EventProtoList, typename Func>
  void ProcessUntilFinish(const EventProtoList& event_protos, Func func) {
    ProcessUntilFinish(
        event_protos, func,
        [] (const auto& event) {
          return false;
        });
  }

  template <typename EventProtoList, typename Func, typename SkipFunc>
  void ProcessUntil(const EventProtoList& event_protos, TimeUsec before, Func func, SkipFunc skip_func) {

    size_t i = 0;
    // For some reason, protobuf uses "int" for the size() of its repeated fields.
    assert(event_protos.size() >= 0);
    const auto events_size = static_cast<size_t>(event_protos.size());

    auto can_end_events = [this, before] () -> bool {
      return _ops.size() > 0 && !Subsumes(_ops.start_us, _ops.end_us, before);
    };

    auto append_event = [this] (const EventProto* event) {
      _ops.insert(event);
    };

    while (i < events_size || can_end_events()) {
      if (i < events_size && skip_func(event_protos[i])) {
        if (SHOULD_DEBUG(FEATURE_PREPROCESS_DATA)) {
          DBG_LOG("(SKIP_FUNC) Event(name={}) @ {} us", event_protos[i].name(), event_protos[i].start_time_us());
        }
        i += 1;
        continue;
      }

      // Skip empty events:
      if (i < events_size && event_protos[i].start_time_us() == GetEndTime(&event_protos[i])) {
        if (SHOULD_DEBUG(FEATURE_PREPROCESS_DATA)) {
          DBG_LOG("(SKIP) Event(name={}) @ {} us", event_protos[i].name(), event_protos[i].start_time_us());
        }
        i += 1;
        continue;
      }

      if (_ops.size() == 0) {
        assert(i < events_size);
        append_event(&event_protos[i]);
        _last_time = event_protos[i].start_time_us();
        if (SHOULD_DEBUG(FEATURE_PREPROCESS_DATA)) {
          DBG_LOG("(EMPTY_START) last_time = Event(name={}) @ {} us", event_protos[i].name(), _last_time);
        }
        i += 1;
        continue;
      }

      if (i < events_size && event_protos[i].start_time_us() < GetEndTime(_ops.PeekNextEvent())) {
        auto op = _ops.PeekNextEvent();
        auto start_time_us = _last_time;
        auto end_time_us = event_protos[i].start_time_us();
        if (start_time_us < end_time_us) {
          // Add operation event only if duration is non-zero.
          func(op->name(), start_time_us, end_time_us);
        }
        append_event(&event_protos[i]);
        _last_time = end_time_us;
        if (SHOULD_DEBUG(FEATURE_PREPROCESS_DATA)) {
          DBG_LOG("(START) last_time = Event(name={}) @ {} us", event_protos[i].name(), _last_time);
        }
        i += 1;
      } else if (can_end_events()) {
        auto op = _ops.PopNextEvent();
        auto start_time_us = _last_time;
        auto end_time_us = GetEndTime(op);
        if (start_time_us < end_time_us) {
          // Add operation event only if duration is non-zero.
          func(op->name(), start_time_us, end_time_us);
        }
        _last_time = end_time_us;
        if (SHOULD_DEBUG(FEATURE_PREPROCESS_DATA)) {
          DBG_LOG("(END) last_time = Event(name={}) @ {} us", op->name(), _last_time);
        }
      }
    }

    if (_ops.size() > 0) {
      if (SHOULD_DEBUG(FEATURE_OVERLAP)) {
        DBG_LOG("Ran until before={} us, still have {} ops remaining", before, _ops.size());
      }
    }

  }
  template <typename EventProtoList, typename Func>
  void ProcessUntil(const EventProtoList& event_protos, TimeUsec before, Func func) {
    ProcessUntil(
        event_protos, before, func,
        [] (const auto& event) {
          return false;
        });
  }

};


template <class ProtoReader>
MyStatus GenericReadTraceFileMeta(ProtoReader* reader, TraceFileMeta* meta) {
  auto status = reader->Init();
  IF_BAD_STATUS_RETURN(status);
  reader->EachCategory([meta] (const auto& category, auto& events) {
    meta->n_events[category] += events.size();
    meta->categories.insert(category);
  });
  return MyStatus::OK();
}
#define DEFINE_READ_TRACE_FILE_META \
  virtual MyStatus ReadTraceFileMeta(TraceFileMeta* meta) override { \
    return GenericReadTraceFileMeta(this, meta); \
  }

class CategoryEventsProtoReader : public ITraceFileProtoReader<rlscope::CategoryEventsProto, CategoryEventsProtoReader> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CATEGORY_EVENTS_FILE;

  using ProtoKlass = rlscope::CategoryEventsProto;
  using EventKlass = rlscope::Event;
  using ProtoReader = CategoryEventsProtoReader;

  // Need to know:
  // - The START time of the first "Operation" event.
  // - The START time of the first "Operation" event.
  //TimeUsec _start_operation_usec;

  virtual ~CategoryEventsProtoReader() = default;

  CategoryEventsProtoReader(const std::string& path) :
      ITraceFileProtoReader<ProtoKlass, ProtoReader>(path, RLSFileType::CATEGORY_EVENTS_FILE, "category_events")
  {
  }

  static const std::string& EventName(const EventKlass& event) {
    return event.name();
  }
  static TimeUsec EventStartUsec(const EventKlass& event) {
    return event.start_time_us();
  }
  static TimeUsec EventEndUsec(const EventKlass& event) {
    return event.start_time_us() + event.duration_us();
  }

  static const std::string& ProtoMachine(const ProtoKlass& proto) {
    return proto.machine_name();
  }
  static const std::string& ProtoProcess(const ProtoKlass& proto) {
    return proto.process_name();
  }
  static const std::string& ProtoPhase(const ProtoKlass& proto) {
    return proto.phase();
  }

  DEFINE_READ_TRACE_FILE_META

//  virtual MyStatus ReadTraceFileMeta(TraceFileMeta* meta) override {
//    return GenericReadTraceFileMeta(this, meta);
//  }

  template <typename Func>
  void EachCategory(Func func) {
    auto const& proto = Proto();
    for (const auto& pair : proto.category_events()) {
      const auto& category = pair.first;
      const auto& events = pair.second.events();
      func(category, events);
    }
  }

  template <typename Func>
  void MutableEachCategory(Func func) {
    auto* proto = MutableProto();
    for (auto& pair : *(proto->mutable_category_events())) {
      const auto& category = pair.first;
      auto* events = pair.second.mutable_events();
      func(category, events);
    }
  }

  static inline bool SkipOperationEvent(const Operation& name) {
    return std::regex_match(name, PROCESS_OPERATION_REGEX);
  }

  virtual MyStatus _ReadMetaFromProto(nlohmann::json* meta, const ProtoKlass& proto) override {

//    TimeUsec start_operation_usec = std::numeric_limits<TimeUsec>::max();
//    auto it = proto.category_events().find(CATEGORY_OPERATION);
//    if (it != proto.category_events().end()) {
//      const auto& events =  it->second.events();
//      for (const auto& event : events) {
//        start_operation_usec = std::min(start_operation_usec, event.start_time_us());
//      }
//    }
//    (*meta)["start_operation_usec"] = start_operation_usec;

    for (const auto& pair : proto.category_events()) {
      auto const& category = pair.first;
      const auto& events =  pair.second.events();
      auto start_operation_usec = std::numeric_limits<TimeUsec>::max();
      const std::string* start_usec_name = nullptr;
      bool is_category_operation = (category == CATEGORY_OPERATION);
      for (const auto& event : events) {
        if (is_category_operation && SkipOperationEvent(event.name())) {
          // Skip
          continue;
        }
        if (event.start_time_us() < start_operation_usec) {
          start_operation_usec = event.start_time_us();
          start_usec_name = &event.name();
        }
        // start_operation_usec = std::min(start_operation_usec, event.start_time_us());
      }
      (*meta)["start_usec"][category] = start_operation_usec;
      if (start_usec_name) {
        (*meta)["start_usec_name"][category] = *start_usec_name;
      } else {
        (*meta)["start_usec_name"][category] = "";
      }
    }

//    TimeUsec start_operation_usec = std::numeric_limits<TimeUsec>::max();
//    auto it = proto.category_events().find(CATEGORY_OPERATION);
//    if (it != proto.category_events().end()) {
//      const auto& events =  it->second.events();
//      for (const auto& event : events) {
//        start_operation_usec = std::min(start_operation_usec, event.start_time_us());
//      }
//    }
//    (*meta)["start_operation_usec"] = start_operation_usec;

    return MyStatus::OK();
  }

};
class CategoryEventsParser : public IEventFileProtoParser<rlscope::CategoryEventsProto, CategoryEventsProtoReader> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CATEGORY_EVENTS_FILE;

  using ProtoKlass = rlscope::CategoryEventsProto;
  using ProtoReader = CategoryEventsProtoReader;

  EventFlattener<rlscope::Event> _event_flattener;

  virtual ~CategoryEventsParser() = default;

  CategoryEventsParser(TraceParserMeta meta) :
      IEventFileProtoParser<ProtoKlass, ProtoReader>(std::move(meta), RLSFileType::CATEGORY_EVENTS_FILE, "category_events")
  {
  }

//  virtual MyStatus _CountCategoryTimes(CategoryTimesCount* count, ProtoKlass* proto, boost::optional<const TraceFileMeta&> next_meta);
//  MyStatus _CountCategoryTimesOperation(CategoryTimesCount* count, ProtoKlass* proto, boost::optional<const TraceFileMeta&> next_meta);
//  virtual MyStatus _AppendCategoryTimes(CategoryTimes* out_category_times, ProtoKlass* proto, boost::optional<const TraceFileMeta&> next_meta);

  virtual MyStatus AppendAllCategory(
      const EntireTraceMeta& entire_meta,
      const Category& category,
      const std::vector<typename ProtoReader::EventKlass>& events,
      CategoryTimes* out_category_times) override;
  virtual MyStatus _AppendCategoryOperation(
      const EntireTraceMeta& entire_meta,
      const Category& category,
      const std::vector<typename ProtoReader::EventKlass>& events,
      CategoryTimes* out_category_times);

//  virtual MyStatus AppendCategoryTimes(const std::vector<TraceFileMeta>& metas, CategoryTimes* out_category_times) override {
//    MyStatus status = MyStatus::OK();
//
//    for (auto it = metas.begin(); it != metas.end(); it++) {
//      const auto& meta = *it;
//
//      boost::optional<const TraceFileMeta&> next_meta;
//      auto next_it = it;
//      next_it++;
//      if (next_it != metas.end()) {
//        next_meta = *next_it;
//      }
//
//
//      ProtoReader reader(meta.get_path());
//      status = reader.Init();
//      IF_BAD_STATUS_RETURN(status);
//
//      reader.MutableEachCategory([] (const auto& category, auto* events) {
//        SortEvents(events);
//      });
//
////      reader.EachCategory([] (const auto& category, const auto& events) {
////        CheckEventsSorted(category, events);
////      });
//
//      status = _AppendCategoryTimes(out_category_times, reader.MutableProto(), next_meta);
//      IF_BAD_STATUS_RETURN(status);
//    }
//
//    return status;
//  }

//  virtual MyStatus CountCategoryTimes(const std::vector<TraceFileMeta>& metas) override {
//    MyStatus status = MyStatus::OK();
//
//    for (auto it = metas.begin(); it != metas.end(); it++) {
//      const auto& meta = *it;
//
//      boost::optional<const TraceFileMeta&> next_meta;
//      auto next_it = it;
//      next_it++;
//      if (next_it != metas.end()) {
//        assert(next_it->get_trace_id() == it->get_trace_id() + 1);
//        next_meta = *next_it;
//      }
//
//      ProtoReader reader(meta.get_path());
//      status = reader.Init();
//      IF_BAD_STATUS_RETURN(status);
//
//      reader.MutableEachCategory([] (const auto& category, auto* events) {
//        SortEvents(events);
//      });
//
//      status = _CountCategoryTimes(&_count, reader.MutableProto(), next_meta);
//      IF_BAD_STATUS_RETURN(status);
//    }
//
//    return status;
//  }


//  DECLARE_GET_PARSER_META

  // PROBLEM:
  // - we need to determine ahead of time how many events we need to read so we can preallocate an array.
  // - to do this, we need to load each protobuf file.
  // - our "ReadCategory" function should instead become "AppendCategory", and we should return an error or Assert if its too big.
};

class CUDAAPIStatsProtoReader : public ITraceFileProtoReader<rlscope::CUDAAPIPhaseStatsProto, CUDAAPIStatsProtoReader> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CUDA_API_STATS_FILE;

  using ProtoKlass = rlscope::CUDAAPIPhaseStatsProto;
  using EventKlass = rlscope::CUDAAPIEvent;
  using ProtoReader = CUDAAPIStatsProtoReader;

  virtual ~CUDAAPIStatsProtoReader() = default;

  CUDAAPIStatsProtoReader(const std::string& path) :
      ITraceFileProtoReader<ProtoKlass, ProtoReader>(path, RLSFileType::CUDA_API_STATS_FILE, "cuda_api_stats")
  {
  }

  static const std::string& EventName(const EventKlass& event) {
    return event.api_name();
  }
  static TimeUsec EventStartUsec(const EventKlass& event) {
    return event.start_time_us();
  }
  static TimeUsec EventEndUsec(const EventKlass& event) {
    return event.start_time_us() + event.duration_us();
  }

  static const std::string& ProtoMachine(const ProtoKlass& proto) {
    return proto.machine_name();
  }
  static const std::string& ProtoProcess(const ProtoKlass& proto) {
    return proto.process_name();
  }
  static const std::string& ProtoPhase(const ProtoKlass& proto) {
    return proto.phase();
  }

  DEFINE_READ_TRACE_FILE_META

  template <typename Func>
  void EachCategory(Func func) {
    auto const& proto = Proto();
    const std::string category = CATEGORY_CUDA_API_CPU;
    const auto& events = proto.events();
    func(category, events);
  }

  template <typename Func>
  void MutableEachCategory(Func func) {
    auto* proto = MutableProto();
    const std::string category = CATEGORY_CUDA_API_CPU;
    auto* events = proto->mutable_events();
    func(category, events);
  }

};
class CUDAAPIStatsParser : public ISimpleProtoParser<rlscope::CUDAAPIPhaseStatsProto, CUDAAPIStatsProtoReader> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CUDA_API_STATS_FILE;

  using ProtoKlass = rlscope::CUDAAPIPhaseStatsProto;
  using ProtoReader = CUDAAPIStatsProtoReader;

  virtual ~CUDAAPIStatsParser() = default;

  CUDAAPIStatsParser(TraceParserMeta meta) :
      ISimpleProtoParser<ProtoKlass, ProtoReader>(std::move(meta), RLSFileType::CUDA_API_STATS_FILE, "cuda_api_stats")
  {
  }

//  DECLARE_GET_PARSER_META

};


class NvprofTraceFileReader : public ITraceFileReader {
public:
  RLSAnalyzeArgs args;
  std::string _path;
  RLSFileType _file_type;

  Machine _machine;
  Process _process;
  Phase _phase;

  bool _initialized;

  NvprofFileType::HeaderMeta _header_meta;

  std::vector<std::string> _header;
  std::vector<std::string> _units;

  size_t _num_skip_lines;
  size_t _num_other_lines;
  size_t _num_data_lines;

  std::unique_ptr<NvprofFileType> _nvprof_file_type;

  NvprofTraceFileReader(RLSAnalyzeArgs args, const std::string& path, RLSFileType file_type) :
      args(args)
      , _path(path)
      , _file_type(file_type)
      , _initialized(false)
      , _num_skip_lines(0)
      , _num_other_lines(0)
      , _num_data_lines(0)
  {
  }
  virtual ~NvprofTraceFileReader() = default;

  // Implemented in IEventFileProtoParser<ProtoKlass>
//  virtual bool IsFile(const std::string& path) const = 0;
//  virtual MyStatus ReadFile(CategoryTimes* out_category_times) = 0;
  virtual MyStatus Init() override;
  virtual void Clear() override;
  virtual MyStatus ReadMeta(nlohmann::json* meta) override;
  virtual MyStatus ReadTraceFileMeta(TraceFileMeta* meta) override;

  virtual const Machine& get_machine() const override;
  virtual const Process& get_process() const override;
  virtual const Phase& get_phase() const override;

  std::set<Category> Categories() const;
  virtual MyStatus EachEvent(const Category& category, EachEventFunc) const;

  MyStatus _ReadProcess();
  MyStatus _ReadCSVMeta();

};

class CUDADeviceEventsProtoReader : public ITraceFileProtoReader<rlscope::MachineDevsEventsProto, CUDADeviceEventsProtoReader> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CUDA_DEVICE_EVENTS_FILE;

  using ProtoKlass = rlscope::MachineDevsEventsProto;
  using EventKlass = rlscope::CUDAEventProto;
  using ProtoReader = CUDADeviceEventsProtoReader;

  virtual ~CUDADeviceEventsProtoReader() = default;

  CUDADeviceEventsProtoReader(const std::string& path) :
      ITraceFileProtoReader<ProtoKlass, ProtoReader>(path, RLSFileType::CUDA_DEVICE_EVENTS_FILE, "cuda_device_events")
  {
  }

  static const std::string& EventName(const EventKlass& event) {
    return event.name();
  }
  static TimeUsec EventStartUsec(const EventKlass& event) {
    return event.start_time_us();
  }
  static TimeUsec EventEndUsec(const EventKlass& event) {
    return event.start_time_us() + event.duration_us();
  }

  static const std::string& ProtoMachine(const ProtoKlass& proto) {
    return proto.machine_name();
  }
  static const std::string& ProtoProcess(const ProtoKlass& proto) {
    return proto.process_name();
  }
  static const std::string& ProtoPhase(const ProtoKlass& proto) {
    return proto.phase();
  }

  DEFINE_READ_TRACE_FILE_META

  template <typename Func>
  void EachCategory(Func func) {
    auto const& proto = Proto();
    const std::string category = CATEGORY_GPU;
    for (const auto& dev_events_pair : proto.dev_events()) {
      const auto& dev = dev_events_pair.first;
      const auto& events = dev_events_pair.second.events();
      func(category, events);
    }
  }

  template <typename Func>
  void MutableEachCategory(Func func) {
    auto* proto = MutableProto();
    const std::string category = CATEGORY_GPU;
    for (auto& dev_events_pair : *(proto->mutable_dev_events())) {
      const auto& dev = dev_events_pair.first;
      auto* events = dev_events_pair.second.mutable_events();
      func(category, events);
    }
  }

};
class CUDADeviceEventsParser : public ISimpleProtoParser<rlscope::MachineDevsEventsProto, CUDADeviceEventsProtoReader> {
public:
  static const RLSFileType FILE_TYPE = RLSFileType::CUDA_DEVICE_EVENTS_FILE;

  using ProtoKlass = rlscope::MachineDevsEventsProto;
  using ProtoReader = CUDADeviceEventsProtoReader;

  EntireTraceSelector _selector;

  virtual ~CUDADeviceEventsParser() = default;

  CUDADeviceEventsParser(TraceParserMeta meta, const EntireTraceSelector& selector) :
      ISimpleProtoParser<ProtoKlass, ProtoReader>(std::move(meta), RLSFileType::CUDA_DEVICE_EVENTS_FILE, "cuda_device_events"),
      _selector(selector)
  {
  }

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

  RegionMetadataReducer() :
      start_time_usec(0),
      end_time_usec(0)
  {
  }

  RegionMetadataReducer(const CategoryKey& category_key) :
      category_key(category_key),
      start_time_usec(0),
      end_time_usec(0)
      {
  }

  RegionMetadataReducer(
      const CategoryKey& category_key_,
      TimeUsec start_time_usec_,
      TimeUsec end_time_usec_
      ) :
      category_key(category_key_),
      start_time_usec(start_time_usec_),
      end_time_usec(end_time_usec_)
      {
    assert(start_time_usec >= 0);
    assert(end_time_usec >= 0);
  }

  void CheckIntegrity() const {
    assert(start_time_usec >= 0);
    assert(end_time_usec >= 0);
  }

  template <typename OStream>
  void Print(OStream& out, int indent) const {

    PrintIndent(out, indent);
    out << "RegionMetadataReducer:";

    out << "\n";
    category_key.Print(out, indent + 1);

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "start_us = " << start_time_usec << " us";

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "end_us   = " << end_time_usec << " us";

    double dur_sec = (((double)end_time_usec) - ((double)start_time_usec))/((double)USEC_IN_SEC);
    out << "\n";
    PrintIndent(out, indent + 1);
    out << "dur_sec  = " << dur_sec << " sec";

  }

  template <typename OStream>
  friend OStream& operator<<(OStream& os, const RegionMetadataReducer& obj) {
    os << "RegionMetadataReducer("
       << "start=" << obj.start_time_usec << " us"
       << ", end=" << obj.end_time_usec << " us"
       << ")";
    return os;
  }

  void MergeInplace(const RegionMetadataReducer& region2) {
    RegionMetadataReducer& region1 = *this;

    assert(region1.start_time_usec >= 0);
    assert(region1.end_time_usec >= 0);

    assert(region2.start_time_usec >= 0);
    assert(region2.end_time_usec >= 0);

//    DBG_LOG("region1 before: {}", region1);
//    DBG_LOG("region2 before: {}", region2);

    region1.start_time_usec = KeepExtreme<TimeUsec>(
        std::vector<TimeUsec>{region1.start_time_usec, region2.start_time_usec},
        0,
        /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
          return x < min_x;
        },
        /*should_skip=*/[] (TimeUsec x) {
          return x == 0;
        });

    region1.end_time_usec = KeepExtreme<TimeUsec>(
        std::vector<TimeUsec>{region1.end_time_usec, region2.end_time_usec},
        0,
        /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
          return x > min_x;
        },
        /*should_skip=*/[] (TimeUsec x) {
          return x == 0;
        });


//    DBG_LOG("region1 after: {}", region1);

    assert(this->start_time_usec >= 0);
    assert(this->end_time_usec >= 0);

  }

};
struct RegionMetadata {
  CategoryKeyBitset category_key;
  TimeUsec start_time_usec;
  TimeUsec end_time_usec;

  RegionMetadata() :
      start_time_usec(0),
      end_time_usec(0)
  {
  }

  RegionMetadata(
      const CategoryKeyBitset& category_key,
      TimeUsec start_time_usec,
      TimeUsec end_time_usec) :
      category_key(category_key),
      start_time_usec(start_time_usec),
      end_time_usec(end_time_usec)
  {
  }


  RegionMetadata(const CategoryKeyBitset& category_key) :
      category_key(category_key),
      start_time_usec(0),
      end_time_usec(0)
  {
  }

  bool operator==(const RegionMetadata& rhs) const {
    auto& lhs = *this;
    return lhs.start_time_usec == rhs.start_time_usec
           && lhs.end_time_usec == rhs.end_time_usec;
  }

  void ConvertPsecToUsec() {
    start_time_usec = start_time_usec / PSEC_IN_USEC;
    end_time_usec = end_time_usec / PSEC_IN_USEC;
  }

  inline void AddEvent(TimeUsec start_us, TimeUsec end_us) {
    if (this->start_time_usec == 0 || start_us < this->start_time_usec) {
      this->start_time_usec = start_us;
    }

    if (this->end_time_usec == 0 || end_us > this->end_time_usec) {
      this->end_time_usec = end_us;
    }

  }

  void Print(std::ostream& out, int indent) const {

    PrintIndent(out, indent);
    out << "RegionMetadata:";

    out << "\n";
    category_key.Print(out, indent + 1);

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "start_us = " << start_time_usec << " us";

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "end_us   = " << end_time_usec << " us";

    double dur_sec = (((double)end_time_usec) - ((double)start_time_usec))/((double)USEC_IN_SEC);
    out << "\n";
    PrintIndent(out, indent + 1);
    out << "dur_sec  = " << dur_sec << " sec";

  }


};
struct OverlapMetadata {
  std::map<CategoryKeyBitset, RegionMetadata> regions;

  bool operator==(const OverlapMetadata& rhs) const {
    auto& lhs = *this;
    return lhs.regions == rhs.regions;
  }

  size_t size() const {
    return regions.size();
  }

  void ConvertPsecToUsec() {
    for (auto& pair : regions) {
      // Change overlap from psec to usec.
      auto& region_meta = pair.second;
      region_meta.ConvertPsecToUsec();
    }
  }

  void Print(std::ostream& out, int indent) const {
    PrintIndent(out, indent);
    out << "OverlapMetadata: size = " << regions.size();
    for (const auto& pair : regions) {
      auto const& bitset = pair.first;
      auto const& region_meta = pair.second;

      out << "\n";
      region_meta.Print(out, indent + 1);
    }
  }

  void AddEvent(const CategoryKeyBitset& category_key, TimeUsec start_us, TimeUsec end_us) {
    if (regions.find(category_key) == regions.end()) {
      regions[category_key] = RegionMetadata(category_key);
    }
    regions.at(category_key).AddEvent(start_us, end_us);
  }
};
struct OverlapResult;
struct OverlapMetadataReducer {
  std::map<CategoryKey, RegionMetadataReducer> regions;

  void Print(std::ostream& out, int indent) const {
    PrintIndent(out, indent);
    out << "OverlapMetadataReducer: size = " << regions.size();
    for (const auto& pair : regions) {
      auto const& category_key = pair.first;
      auto const& region_meta = pair.second;

      out << "\n";
      region_meta.Print(out, indent + 1);
    }
  }

  inline OverlapMetadataReducer OnlyKeys(const std::set<CategoryKey>& category_keys) const {
    OverlapMetadataReducer r;
    for (const auto& category_key : category_keys) {
      this->regions.at(category_key).CheckIntegrity();
      r.regions[category_key] = this->regions.at(category_key);
      r.regions[category_key].CheckIntegrity();
    }
    return r;
  }

  RegionMetadataReducer GetMetaForAll() const {
    RegionMetadataReducer r;
    for (const auto& pair : regions) {
      auto const& meta = pair.second;
      r.MergeInplace(meta);
    }
    return r;
  }

  void MergeRegion(const CategoryKey& category_key, const RegionMetadataReducer& region) {
    if (regions.find(category_key) == regions.end()) {
      regions[category_key] = RegionMetadataReducer(category_key);
    }
    regions.at(category_key).MergeInplace(region);
  }

  void AddRegion(const CategoryKey& category_key, const RegionMetadata& region_metadata) {
    assert(regions.find(category_key) == regions.end());
    regions[category_key] = RegionMetadataReducer(
        category_key,
        region_metadata.start_time_usec,
        region_metadata.end_time_usec
        );
  }

};

template <typename OStream>
void PrintVector(OStream& os, const Eigen::Array<size_t, Eigen::Dynamic, 1>& value) {
  os << "[";
  size_t i = 0;
  // NOTE: Eigen 3.3.9 doesn't support iterator for Eigen::Array for some reason; oh well.
//  for (auto it = value.begin(); it != value.end(); it++) {
//    auto val = *it;
//  for (auto const& val : value) {
  for (int i = 0; i < value.size(); i++) {
    auto val = value(i);
    if (i > 0) {
      os << ", ";
    }
    PrintValue(os, val);
    i += 1;
  }
  os << "]";
}

struct OverlapInterval {
  using IdxArray = Eigen::Array<size_t, Eigen::Dynamic, 1>;
  size_t interval_id;
  // CategoryIdx => EOEvent index
  // inclusive.
  IdxArray start_events;
  // exclusive (o/w we cannot represent empty categories)
  IdxArray end_events;
  TimeUsec start_us;
  TimeUsec duration_us;

  OverlapInterval (
      size_t interval_id,
      IdxArray start_events,
      IdxArray end_events,
      TimeUsec start_us,
      TimeUsec duration_us) :
      interval_id(interval_id)
      , start_events(start_events)
      , end_events(end_events)
      , start_us(start_us)
      , duration_us(duration_us)
  {
  }
  // TODO: EachEvent(...)

  void ConvertPsecToUsec() {
    start_us = start_us / PSEC_IN_USEC;
    duration_us = duration_us / PSEC_IN_USEC;
  }

  using EachEventFunc = std::function<MyStatus (const CategoryKey& category, const EOEvent& event)>;
  MyStatus EachEvent(const OverlapResult& result, EachEventFunc func) const;

  template <typename OStream>
  void Print(OStream& out, int indent) const {
    PrintIndent(out, indent);
    auto size = (end_events - start_events).sum();
    out << "OverlapInterval[" << interval_id << "] size = " << size;

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "start_events=";
//    PrintValue(out, this->start_events);
    PrintVector(out, this->start_events);

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "end_events=";
//    PrintValue(out, this->end_events);
    PrintVector(out, this->end_events);

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "start_us=";
    PrintValue(out, this->start_us);

    out << "\n";
    PrintIndent(out, indent + 1);
    out << "duration_us=";
    PrintValue(out, this->duration_us);
  }


  template <typename OStream>
  friend OStream &operator<<(OStream &os, const OverlapInterval &obj)
  {
    obj.Print(os, 0);
    return os;
  }

};
struct IntervalMeta {
  const OverlapResult& result;
  std::vector<OverlapInterval> intervals;
  IntervalMeta(const OverlapResult& result) :
      result(result) {
  }

  void ConvertPsecToUsec() {
    for (auto& interval : intervals) {
      interval.ConvertPsecToUsec();
    }
  }

  using IndexType = int;

  using EachIntervalFunc = std::function<MyStatus (const OverlapInterval& interval)>;
  MyStatus EachInterval(const OverlapResult& result, EachIntervalFunc func) const;

};
struct OverlapResult {
  Overlap overlap;
  CategoryTransitionCounts category_trans_counts;
  OverlapMetadata meta;
  std::shared_ptr<CategoryIdxMap> idx_map;
  const CategoryTimes& ctimes;
  IntervalMeta interval_meta;

  OverlapResult(const CategoryTimes& ctimes) :
      ctimes(ctimes),
      interval_meta(*this) {
  }

  void ConvertPsecToUsec() {
    for (auto& pair : overlap) {
      // Change overlap from psec to usec.
      pair.second = pair.second / PSEC_IN_USEC;
    }
    meta.ConvertPsecToUsec();
    interval_meta.ConvertPsecToUsec();
  }

  inline const RegionMetadata& GetMeta(const CategoryKeyBitset& bitset) const {
    return meta.regions.at(bitset);
  }

  std::map<std::set<CategoryKey>, TimeUsec> AsOverlapMap() const;
  std::map<std::tuple<std::set<CategoryKey>, std::set<CategoryKey>>, size_t> AsCategoryTransCountsMap() const;

  void DumpVennJS(
      const std::string& directory,
      const Machine& machine,
      const Process& process,
      const Phase& phase) const;

  void Print(std::ostream& out, int indent) const;

  MyStatus DumpCSVFiles(const std::string& base_path) const;
  MyStatus DumpIntervalEventsCSV(const std::string& base_path) const;
  MyStatus DumpIntervalCSV(const std::string& base_path) const;

  DECLARE_PRINT_DEBUG
  DECLARE_PRINT_OPERATOR(OverlapResult)
};
template <typename T>
MyStatus DumpValueAsJson(const std::string& path, const T& value) {
  auto status = MyStatus::OK();
  auto js = ValueAsJson(value);
  status = WriteJson(path, js);
  IF_BAD_STATUS_RETURN(status);
  return MyStatus::OK();
}
class OverlapResultReducer {
public:
  std::map<CategoryKey, TimeUsec> overlap;
  OverlapMetadataReducer regions;
  bool debug{false};

  OverlapResultReducer() = default;

  void Print(std::ostream& out, int indent) const {

    PrintIndent(out, indent);
    out << "OverlapResultReducer: size = " << overlap.size();
    size_t i = 0;
    for (auto const& pair : overlap) {
      auto const& category_key = pair.first;
      auto time_usec = pair.second;
      auto time_sec = ((double)time_usec)/((double)USEC_IN_SEC);
      out << "\n";
      PrintIndent(out, indent + 1);
      out << "[" << i << "] " << category_key << " = " << time_sec << " sec";
      i += 1;
    }

    out << "\n";
    regions.Print(out, indent + 1);

  }

  inline const RegionMetadataReducer& GetMeta(const CategoryKey& category_key) const {
    return regions.regions.at(category_key);
  }

  inline RegionMetadataReducer GetMetaForAll() const {
    return regions.GetMetaForAll();
  }

  inline OverlapResultReducer OnlyKeys(const std::set<CategoryKey>& category_keys) const {
    OverlapResultReducer r;
    for (const auto& category_key : category_keys) {
      r.overlap[category_key] = this->overlap.at(category_key);
    }
    r.regions = this->regions.OnlyKeys(category_keys);
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

  static bool IsEmptyKey(const OverlapType& overlap_type, const CategoryKey& category_key) {
    if (overlap_type == "ResourceOverlap" || overlap_type == "ResourceSubplot" || overlap_type == "ProfilingOverhead") {
      return category_key.non_ops.size() == 0;
    } else if (overlap_type == "OperationOverlap" || overlap_type == "CategoryOverlap") {
      return category_key.ops.size() == 0 ||
             category_key.non_ops.size() == 0;
    }
    assert(false);
    return false;
  }

  static bool IsCPUCategory(const Category& category) {
    return CATEGORIES_CPU.count(category) > 0;
  }

  static bool IsGPUCategory(const Category& category) {
    return CATEGORIES_GPU.count(category) > 0;
  }

  static bool IsProfOverheadCategory(const Category& category) {
    return CATEGORIES_PROF.count(category) > 0;
  }

//  static bool IsProfOverheadKey(const CategoryKey& category_key) {
//    return IsProfOverheadCategory(category_key.non_ops);
//  }

  static std::set<Category> ProfCategories(const std::set<Category>& categories) {
    std::set<Category> prof_categories;
    std::set_intersection(
        categories.begin(), categories.end(),
        CATEGORIES_PROF.begin(), CATEGORIES_PROF.end(),
        std::inserter(prof_categories, prof_categories.begin()));
    return prof_categories;
  }

  static bool HasCPUOverhead(const CategoryKey& category_key) {
    for (const auto& category : category_key.non_ops) {
      if (IsProfOverheadCategory(category)) {
        return true;
      }
    }
    return false;
  }

  static bool HasGPU(const CategoryKey& category_key) {
    for (const auto& category : category_key.non_ops) {
      if (IsGPUCategory(category)) {
        return true;
      }
    }
    return false;
  }

  static bool IsCPUOnlyKey(const CategoryKey& category_key) {
    for (auto const& category : category_key.non_ops) {
      if (!IsProfOverheadCategory(category) && !IsCPUCategory(category)) {
        return false;
      }
    }
    return true;
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
      for (auto const& category : CATEGORIES_CPU) {
        new_key.non_ops.erase(category);
      }
      // Q: Should we remove the profiling category as well...? I think so yes.
      for (auto const& category : CATEGORIES_PROF) {
        new_key.non_ops.erase(category);
      }
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

  void AddOverlapWithKeyAllowOverhead(
      const OverlapType& overlap_type,
      const CategoryKey& old_key,
      const CategoryKey& new_key,
      const OverlapResultReducer& old_reducer) {
    _AddOverlapWithKey(
        overlap_type,
        old_key,
        new_key,
        old_reducer,
        true);
  }

  void AddOverlapWithKey(
      const OverlapType& overlap_type,
      const CategoryKey& old_key,
      const CategoryKey& new_key,
      const OverlapResultReducer& old_reducer) {
    _AddOverlapWithKey(
        overlap_type,
        old_key,
        new_key,
        old_reducer,
        false);
  }


  void _AddOverlapWithKey(
      const OverlapType& overlap_type,
      const CategoryKey& old_key,
      const CategoryKey& new_key,
      const OverlapResultReducer& old_reducer,
      bool allow_overhead) {

    CategoryKey insert_key;
    if (allow_overhead) {
      insert_key = new_key;
    } else {
      insert_key = OverlapResultReducer::NoOverheadKey(new_key);

      if (debug) {
        std::stringstream ss;
        ss << "AddOverlapWithKey:";

        ss << "\n";
        PrintIndent(ss, 1);
        ss << "old_key = " << old_key;

        ss << "\n";
        PrintIndent(ss, 1);
        ss << "new_key = " << new_key;

        ss << "\n";
        PrintIndent(ss, 1);
        ss << "no_overhead_key = " << insert_key;

        DBG_LOG("{}", ss.str());
      }
    }

    if (OverlapResultReducer::IsEmptyKey(overlap_type, insert_key)) {
      return;
    }
    auto time_usec = old_reducer.GetTimeUsec(old_key);
    overlap[insert_key] += time_usec;
    regions.MergeRegion(insert_key, old_reducer.GetMeta(old_key));
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
    assert(ctimes.size() == category_times.size());
  }
  OverlapResult ComputeOverlap(
      bool keep_empty_time = false,
      bool keep_intervals = false) const;
};

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
  virtual ~OverlapTypeReducerInterface() = default;

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
    js_dumper.Init();
    js_dumper.DumpJSONs();
  }

};

class OverlapMapToVennJSConverter {
public:

//  std::map<std::string, TimeUsec> _ComputeSetSizes(const OverlapMap& overlap_map) const {
//#if 0
//    V = dict()
//    for region in O.keys():
//        V[region] = 0
//        add_keys = set([k for k in O.keys() if set(region).issubset(set(k))])
//        for k in add_keys:
//            V[region] += O[k]
//    return V
//#endif
//    std::map<std::string, TimeUsec> set_to_size;
//    for (const auto& pair :overlap_map) {
//      const auto& overlap_region = pair.first;
//      auto time_usec = pair.second;
//      for (const auto& set_region : overlap_region) {
//        set_to_size[set_region] += time_usec;
//      }
//    }
//    return set_to_size;
//  }

  static std::map<std::set<std::string>, TimeUsec> OverlapAsVennDict(const OverlapMap& overlap_map) {
#if 0
    ##
    ## To calcuate V[...] from O[...]:
    ##
    # Add anything from O which is subset-eq of [0,1,2]
    V[0,1,2] = O[0,1,2]
    # Add anything from O which is subset-eq of [0,2]
    V[0,2] = O[0,2] + O[0,1,2]
    V[1,2] = O[1,2] + O[0,1,2]
    V[0,1] = O[0,1] + O[0,1,2]
    # Add anything from O which is subset-eq of [0]
    V[0] = O[0] + O[0,1] + O[0,2] + O[0,1,2]
    V[1] = O[1] + O[0,1] + O[1,2] + O[0,1,2]
    V[2] = O[2] + O[0,2] + O[1,2] + O[0,1,2]

    V = dict()
    for region in O.keys():
        V[region] = 0
        add_keys = set([k for k in O.keys() if set(region).issubset(set(k))])
        for k in add_keys:
            V[region] += O[k]

    # Add any "single regions" that are missing.
    labels = set()
    For region in O.keys():
        for label in region:
            labels.insert(label)
    single_regions = set({l} for l in labels)
    for region in single_regions:
        if region not in V:
            V[region] = 0
            for k in O.keys():
                if region.issubset(k):
                    V[region] += O[k]

    return V
#endif
    std::map<std::set<std::string>, TimeUsec> V;
    for (const auto & pair : overlap_map) {
      auto const& region = pair.first;
      V[region] = 0;
      std::set<std::set<std::string>> add_keys;
      for (const auto& O_pair : overlap_map) {
        const auto& k = O_pair.first;
        if (IsSubset(region, k)) {
          add_keys.insert(k);
        }
      }
      for (const auto& k : add_keys) {
        V[region] += overlap_map.at(k);
      }
    }

    // Add any "single regions" that are missing.
    std::set<std::string> labels;
    for (const auto & pair : overlap_map) {
      auto const &region = pair.first;
      for (const auto& label : region) {
        labels.insert(label);
      }
    }
    std::set<std::set<std::string>> single_regions;
    for (const auto& label : labels) {
      single_regions.insert({label});
    }
    for (const auto & region : single_regions) {
      if (V.find(region) != V.end()) {
        continue;
      }
      V[region] = 0;
      for (const auto & pair : overlap_map) {
        const auto& k = pair.first;
        if (IsSubset(region, k)) {
          V[region] += pair.second;
        }
      }
    }

    return V;
  }

  nlohmann::json convert(const OverlapMap& overlap_map) const {
    nlohmann::json venn_js;
//    auto const& set_to_size = _ComputeSetSizes(overlap_map);
    auto const& venn_sizes = OverlapAsVennDict(overlap_map);

    if (SHOULD_DEBUG(FEATURE_SAVE_JS)) {
      std::stringstream ss;
      ss << "OverlapMapToVennJSConverter.convert; overlap_map = \n  ";
      PrintValue(ss, overlap_map);
      DBG_LOG("{}", ss.str());
    }

    std::map<std::string, size_t> label_to_id;
    std::set<std::string> labels;
    for (const auto& pair : overlap_map) {
      auto const& overlap = pair.first;
      labels.insert(overlap.begin(), overlap.end());
    }
    size_t i = 0;
    for (const auto& label : labels) {
      label_to_id[label] = i;
      i += 1;
    }

    auto as_sets = [&label_to_id] (const std::set<std::string>& overlap) {
      std::set<size_t> set_ids;
      for (auto const& category : overlap) {
        auto ident = label_to_id[category];
        set_ids.insert(ident);
      }
      return set_ids;
    };

    for (const auto& pair : venn_sizes) {
      const auto& overlap = pair.first;
      const auto size = pair.second;
      if (overlap.size() == 1) {
        const auto& label = *overlap.cbegin();
        nlohmann::json venn_set = {
            {"sets", as_sets(overlap)},
            {"size", size},
            {"label", label},
        };
        venn_js.push_back(venn_set);
      }
    }

    for (const auto& pair : venn_sizes) {
      const auto& overlap = pair.first;
      const auto size = pair.second;
      if (overlap.size() > 1) {
        nlohmann::json venn_set = {
            {"sets", as_sets(overlap)},
            {"size", size},
        };
        venn_js.push_back(venn_set);
      }
    }


//    for (const auto& pair : set_to_size) {
//      const auto& label = pair.first;
//      const auto size = pair.second;
//      nlohmann::json venn_set = {
//          {"sets", as_sets({label})},
//          {"size", size},
//          {"label", label},
//      };
//      venn_js.push_back(venn_set);
//    }
//
//    for (const auto& pair : overlap_map) {
//      const auto& overlap = pair.first;
//      auto size = pair.second;
//      if (overlap.size() == 1) {
//        // Single "set region" doesn't include overlap with other sets.
//        // "Set region" is handled in for-loop above this one.
//        continue;
//      }
//      nlohmann::json venn_set = {
//          {"sets", as_sets(overlap)},
//          {"size", size},
//      };
//      venn_js.push_back(venn_set);
//    }

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
  const OverlapResultReducer& reducer;

  std::map<CategoryKey, OverlapResultReducer> reducer_map;

  virtual ~BaseOverlapJSDumper() = default;

  BaseOverlapJSDumper(
      const OverlapType& overlap_type_,
      const std::string& directory_,
      const Machine& machine_,
      const Process& process_,
      const Phase& phase_,
      const OverlapResultReducer& reducer_) :
      overlap_type(overlap_type_),
      directory(directory_),
      machine(machine_),
      process(process_),
      phase(phase_),
      reducer(reducer_)
  {
    if (SHOULD_DEBUG(FEATURE_SAVE_JS)) {
      std::stringstream ss;
      ss << "overlap_type = " << overlap_type;
      ss << "\n";
      reducer.Print(ss, 1);
      DBG_LOG("{}", ss.str());
    }
  }

  void Init() {
    // NOTE: We cannot do this within the constructor, since we need to call
    // the virtual function GroupCategoryKey, and C++ doesn't allow that.
    _BuildReducerMap();
  }

  void _BuildReducerMap() {
    // When we dump *Overlap, we want all these CategoryKey's:
    //
    // ResourceOverlap / ResourceSubplot:
    //   CategoryKey(non_ops)
    //     where non_ops.intersect({CPU, GPU, Total}).size() > 0
    //   We expect only one file:
    //     ops.size() = 0
    //
    // OperationOverlap:
    //   CategoryKey(ops)
    //     where ops.size() > 0
    //   We expect a file for every resource-type:
    //     non_ops.size() > 0 && non_ops.intersect({CPU, GPU, Total}) > 0
    //
    // CategoryOverlap:
    //   CategoryKey(ops, non_ops)
    //     where ops.size() > 0, non_ops.size() > 0
    //   We expect a file for every [resource-type, operation]:
    //     ops.size() > 0 && non_ops.intersect((CPU_CATEGORIES - CPU) U GPU_CATEGORIES) > 0

    // group_map =
    //   GroupKey -> { CategoryKey's that map to GroupKey }
    // reducer_map =
    //   GroupKey -> OverlapResultReducer containing CategoryKey's that map to GroupKey

    std::map<CategoryKey, std::set<CategoryKey>> group_map;
    for (auto const& pair : reducer.overlap) {
      const auto& category_key = pair.first;
      const auto& group_key = GroupCategoryKey(category_key);
      group_map[group_key].insert(category_key);
    }

    for (auto const& pair : group_map) {
      const auto& group_key = pair.first;
      const auto& category_keys = pair.second;
      reducer_map[group_key] = reducer.OnlyKeys(category_keys);
    }
  }

  OverlapMap AsOverlapMap(const OverlapResultReducer& reducer_) const {
    OverlapMap overlap_map;
    for (auto const& category_key_pair : reducer_.overlap) {
      auto const& category_key = category_key_pair.first;
      auto time_usec = category_key_pair.second;

      auto const& strings = CategoryKeyToStrings(category_key);
      overlap_map[strings] = time_usec;
    }
    return overlap_map;
  }

  void DumpJSONs() const {
    size_t i = 0;
    for (auto const& pair : reducer_map) {
      const auto& group_key = pair.first;
      const auto& reducer_ = pair.second;

      auto overlap_map = AsOverlapMap(reducer_);
      i += 1;

      nlohmann::json js;

      Metadata md;
      md["machine"] = machine;
      md["process"] = process;
      md["phase"] = phase;
      md["overlap_type"] = overlap_type;

      // Control for bugs in old data-formats.
      md["version"] = RLSCOPE_VERSION;

      auto const& meta = reducer.GetMetaForAll();
      md["start_time_usec"] = meta.start_time_usec;
      md["end_time_usec"] = meta.end_time_usec;

      AddMetadataFields(&md, group_key);
      js["metadata"] = md;

      OverlapMapToVennJSConverter converter;
      auto const& venn_js = converter.convert(overlap_map);

      js["venn"] = venn_js;
      const auto& venn_js_path = VennJSPath(md);
      if (SHOULD_DEBUG(FEATURE_SAVE_JS)) {
        std::stringstream ss;
        ss << "Write json to path = " << venn_js_path;
        DBG_LOG("{}", ss.str());
      }

      boost::filesystem::path parent = boost::filesystem::path(venn_js_path).parent_path();
      boost::filesystem::create_directories(parent.string());

      MyStatus status = MyStatus::OK();
      status = WriteJson(venn_js_path, js);
      assert(status.code() == MyStatus::OK().code());
    }
  }

  std::string VennJSPath(const Metadata& md) const {
    boost::filesystem::path direc(directory);
    boost::filesystem::path base = VennJSBasename(md);
    return (direc / base).string();
  }

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const = 0;
  virtual std::string VennJSBasename(const Metadata& md) const = 0;
  virtual CategoryKey GroupCategoryKey(const CategoryKey& category_key) const = 0;
  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const = 0;

  struct CPUAndGPUCategories {
    std::set<Category> cpus;
    std::set<Category> gpus;
  };
  static CPUAndGPUCategories SplitCpuGpuCategories(const std::set<Category>& non_ops) {
    CPUAndGPUCategories cats;
    for (auto const& category : non_ops) {
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

  static std::set<Category> AsCPUAndGPU(const std::set<Category>& non_ops) {
    auto const& cpus_gpus = SplitCpuGpuCategories(non_ops);
    std::set<Category> cpu_and_gpu;
    if (cpus_gpus.cpus.size() > 0) {
      cpu_and_gpu.insert(CATEGORY_CPU);
    }
    if (cpus_gpus.gpus.size() > 0) {
      cpu_and_gpu.insert(CATEGORY_GPU);
    }
    return cpu_and_gpu;
  }


};

class ResourceJSDumper : public BaseOverlapJSDumper {
public:
  virtual ~ResourceJSDumper() = default;

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

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const override {
    // set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
    assert(category_key.ops.size() == 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const override {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual CategoryKey GroupCategoryKey(const CategoryKey& category_key) const override {
    // ResourceOverlap / ResourceSubplot:
    // - ALL the CategoryKey's are output to the same file.
    // - In: CategoryKey(ops={}, non_ops={CPU, GPU})
    // - Out: CategoryKey(ops={}, non_ops={})
    CategoryKey group_key(
        /*procs=*/category_key.procs,
        /*ops=*/{},
        /*non_ops=*/{});
    return group_key;
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const override {
    // pass
  }

};
class ResourceOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = ResourceJSDumper;
  DEFINE_DumpOverlapJS(ResourceOverlapType)

  virtual ~ResourceOverlapType() = default;

  virtual OverlapType GetOverlapType() const {
    return "ResourceOverlap";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    OverlapResultReducer r;
    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;
      if (OverlapResultReducer::IsEmptyKey(GetOverlapType(), old_key)) {
        continue;
      }
      if (old_key.ops.size() > 1) {
        // Operations can only overlap cross-process, not within a single-process
        assert(old_key.procs.size() > 1);
      }
      CategoryKey new_key(
          /*procs=*/{},
          /*ops=*/{},
          /*non_ops=*/old_key.non_ops);
      auto cpu_gpu_key = OverlapResultReducer::ReduceCategoryKey(new_key, /*as_cpu_gpu=*/true);
      r.AddOverlapWithKey(GetOverlapType(), old_key, cpu_gpu_key, old_reducer);
    }
    return r;
  }

};

class ResourceSubplotJSDumper : public BaseOverlapJSDumper {
public:

  virtual ~ResourceSubplotJSDumper() = default;

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

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const override {
    // set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
    assert(category_key.ops.size() == 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const override {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual CategoryKey GroupCategoryKey(const CategoryKey& category_key) const override {
    // ResourceOverlap / ResourceSubplot:
    // - ALL the CategoryKey's are output to the same file.
    // - In: CategoryKey(ops={}, non_ops={CPU, GPU})
    // - Out: CategoryKey(ops={}, non_ops={})
    CategoryKey group_key(
        /*procs=*/category_key.procs,
        /*ops=*/{},
        /*non_ops=*/{});
    return group_key;
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const override {
    // pass
  }

};
class ResourceSubplotOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = ResourceSubplotJSDumper;
  DEFINE_DumpOverlapJS(ResourceSubplotOverlapType)

  virtual ~ResourceSubplotOverlapType() = default;

  virtual OverlapType GetOverlapType() const {
    return "ResourceSubplot";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    OverlapResultReducer r;
    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;
      if (OverlapResultReducer::IsEmptyKey(GetOverlapType(), old_key)) {
        continue;
      }

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

      if (!(OverlapResultReducer::IsCPUOnlyKey(old_key) && OverlapResultReducer::HasCPUOverhead(old_key))) {
        CategoryKey new_key(
            /*procs=*/{},
            /*ops=*/{},
            /*non_ops=*/{CATEGORY_TOTAL});
        r.AddOverlapWithKey(GetOverlapType(), old_key, new_key, old_reducer);
      }

      std::set<Category> prof_categories;
      std::set_intersection(
          old_key.non_ops.begin(), old_key.non_ops.end(),
          CATEGORIES_PROF.begin(), CATEGORIES_PROF.end(),
          std::inserter(prof_categories, prof_categories.begin()));
      auto cpu_gpu_key = OverlapResultReducer::ReduceCategoryKey(old_key, /*as_cpu_gpu=*/true);
      for (auto const& resource_type : cpu_gpu_key.non_ops) {
        // NOTE: This is sort of hacky;
        // we AREN'T outputting disjoint overlap regions here;
        // instead we are outputting an entire "set" including its overlaps:
        // i.e.
        // CPU   = [CPU only time] + [CPU overlapped with GPU time]
        // GPU   = [GPU only time] + [CPU overlapped with GPU time]
        // Total = [CPU only time] + [GPU only time] + [CPU overlapped with GPU time]
        CategoryKey add_key(
            /*procs=*/{},
            /*ops=*/{},
            /*non_ops=*/{});
        add_key.non_ops.insert(resource_type);
        // Add any profiling categories (if any) so CPU time gets subtracted properly by AddOverlapWithKey.
        for (const auto& category : prof_categories) {
          add_key.non_ops.insert(category);
        }
        r.AddOverlapWithKey(GetOverlapType(), old_key, add_key, old_reducer);
      }

    }
    return r;
  }

};


class OperationJSDumper : public BaseOverlapJSDumper {
public:
  virtual ~OperationJSDumper() = default;

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

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const override {
    // set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec
    assert(category_key.ops.size() > 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const override {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    AddResourcesSuffix(ss, md);
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual CategoryKey GroupCategoryKey(const CategoryKey& category_key) const override {
    // OperationOverlap:
    // - Same resource-type goes to the same file.
    // - In: CategoryKey(ops={compute_advantage_estimates}, non_ops={CPU, GPU})
    // - Out: CategoryKey(ops={}, non_ops={CPU, CPU})
    CategoryKey group_key(
        /*procs=*/category_key.procs,
        /*ops=*/{},
        /*non_ops=*/category_key.non_ops);
    return group_key;
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const override {
    (*md)["resource_overlap"] = category_key.non_ops;
  }

};
class OperationOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = OperationJSDumper;
  DEFINE_DumpOverlapJS(OperationOverlapType)

  virtual ~OperationOverlapType() = default;

  virtual OverlapType GetOverlapType() const {
    return "OperationOverlap";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    // reduce_overlap_resource_operation

    OverlapResultReducer r;

    if (r.debug) {
      DBG_LOG("{}", "OperationOverlapType.PostReduceCategoryKey");
    }

    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;
      if (OverlapResultReducer::IsEmptyKey(GetOverlapType(), old_key)) {
        continue;
      }

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
      r.AddOverlapWithKey(GetOverlapType(), old_key, new_key, old_reducer);

    }

    return r;
  }

};

class CategoryJSDumper : public BaseOverlapJSDumper {
public:

  virtual ~CategoryJSDumper() = default;

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

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const override {
    // set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec
    assert(category_key.ops.size() > 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const override {
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

  virtual CategoryKey GroupCategoryKey(const CategoryKey& category_key) const override {
    // CategoryOverlap:
    // - Same ops and resource-type goes to the same file.
    // - In: CategoryKey(ops={compute_advantage_estimates}, non_ops={CUDA API CPU, Framework API C})
    // - Out: CategoryKey(ops={compute_advantage_estimates}, non_ops={CPU})
    CategoryKey group_key(
        /*procs=*/category_key.procs,
        /*ops=*/category_key.ops,
        /*non_ops=*/AsCPUAndGPU(category_key.non_ops));
    return group_key;
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const override {

    // add md["operation"]
    assert(category_key.ops.size() == 1);
    (*md)["operation"] = category_key.ops;

    // add md["resource_overlap"]
    auto const& cpus_gpus = SplitCpuGpuCategories(category_key.non_ops);
    std::set<Category> resource_key;
    if (cpus_gpus.cpus.size() > 0) {
      resource_key.insert(CATEGORY_CPU);
    }
    if (cpus_gpus.gpus.size() > 0) {
      resource_key.insert(CATEGORY_GPU);
    }
    (*md)["resource_overlap"] = resource_key;

  }

};
class CategoryOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = CategoryJSDumper;
  DEFINE_DumpOverlapJS(CategoryOverlapType)

  virtual ~CategoryOverlapType() = default;

  virtual OverlapType GetOverlapType() const {
    return "CategoryOverlap";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {
    // reduce_overlap_resource_operation

    OverlapResultReducer r;

    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;
      if (OverlapResultReducer::IsEmptyKey(GetOverlapType(), old_key)) {
        continue;
      }

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
      r.AddOverlapWithKey(GetOverlapType(), old_key, new_key, old_reducer);

    }

    return r;
  }

};

class ProfilingOverheadJSDumper : public BaseOverlapJSDumper {
public:

  virtual ~ProfilingOverheadJSDumper() = default;

  ProfilingOverheadJSDumper(
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

  virtual std::set<std::string> CategoryKeyToStrings(const CategoryKey& category_key) const override {
    // set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
    //   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
    assert(category_key.ops.size() == 0);
    assert(category_key.non_ops.size() > 0);
    assert(category_key.procs.size() == 0 || category_key.procs.size() == 1);
    return category_key.non_ops;
  }

  virtual std::string VennJSBasename(const Metadata& md) const override {
    std::stringstream ss;
    AddOverlapTitle(ss, md);
    AddMachineSuffix(ss, md);
    AddProcessSuffix(ss, md);
    AddPhaseSuffix(ss, md);
    // PROBLEM: what if phases overlap like in minigo...?  How to handle this?  Show each phase separately?
    ss << ".venn_js.json";
    return ss.str();
  }

  virtual CategoryKey GroupCategoryKey(const CategoryKey& category_key) const override {
    // ResourceOverlap / ProfilingOverhead:
    // - ALL the CategoryKey's are output to the same file.
    // - In: CategoryKey(ops={}, non_ops={CPU, GPU})
    // - Out: CategoryKey(ops={}, non_ops={})
    CategoryKey group_key(
        /*procs=*/category_key.procs,
        /*ops=*/{},
        /*non_ops=*/{});
    return group_key;
  }

  virtual void AddMetadataFields(Metadata* md, const CategoryKey& category_key) const override {
    // pass
  }

};
class ProfilingOverheadOverlapType : public OverlapTypeReducerInterface {
public:
  using OverlapJSDumper = ProfilingOverheadJSDumper;
  DEFINE_DumpOverlapJS(ProfilingOverheadOverlapType)

  virtual ~ProfilingOverheadOverlapType() = default;

  virtual OverlapType GetOverlapType() const {
    return "ProfilingOverhead";
  }
  virtual OverlapResultReducer PostReduceCategoryKey(const OverlapResultReducer& old_reducer) const {

    // We want to bin "Corrected training time" for any:
    // - GPU only time (may have CPU overhead)
    // - CPU/GPU only time (may have CPU overhead)
    // - CPU only time that does NOT have CPU overhead in it

    // If the key contains any CATEGORIES_PROF, and it is a CPU-only key (NO GPU categories):
    // - Record CategoryKey(
    //     procs=procs,
    //     ops=ops,
    //     non_ops=non_ops.intersect(CATEGORIES_PROF)
    //   )
    //   i.e. We just want to keep CATEGORIES_PROF if it ONLY intersects CPU time (NOT GPU time).

    OverlapResultReducer r;
    for (auto const& pair : old_reducer.overlap) {
      auto const& old_key = pair.first;
      if (OverlapResultReducer::IsEmptyKey(GetOverlapType(), old_key)) {
        continue;
      }
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

      auto cpu_gpu_key = OverlapResultReducer::ReduceCategoryKey(old_key, /*as_cpu_gpu=*/true);
      if (
          (OverlapResultReducer::IsCPUOnlyKey(cpu_gpu_key) && !OverlapResultReducer::HasCPUOverhead(cpu_gpu_key))
          || (OverlapResultReducer::HasGPU(cpu_gpu_key))) {
        CategoryKey new_key(
            /*procs=*/cpu_gpu_key.procs,
            /*ops=*/{},
            /*non_ops=*/{CATEGORY_CORRECTED_TRAINING_TIME});
        r.AddOverlapWithKeyAllowOverhead(GetOverlapType(), old_key, new_key, old_reducer);
      }

      if (
          OverlapResultReducer::HasCPUOverhead(cpu_gpu_key) &&
          OverlapResultReducer::IsCPUOnlyKey(cpu_gpu_key)) {
        CategoryKey new_key(
            /*procs=*/cpu_gpu_key.procs,
            /*ops=*/{},
            /*non_ops=*/OverlapResultReducer::ProfCategories(cpu_gpu_key.non_ops));
        r.AddOverlapWithKeyAllowOverhead(GetOverlapType(), old_key, new_key, old_reducer);
      }

    }
    return r;
  }

};

template <typename T>
inline nlohmann::json ValueAsJson(const T& value) {
  nlohmann::json js;
  js = value;
  return js;
}

template <typename T>
inline nlohmann::json ValueAsJson(const std::set<T>& value) {
  nlohmann::json js;

  std::vector<nlohmann::json> xs;
  xs.reserve(value.size());
  for (const auto& x : value) {
    xs.push_back(ValueAsJson(x));
  }
  js = xs;
  return js;
}

//template <typename T>
//inline nlohmann::json ValueAsJson(const std::tuple<T>& value) {
//  nlohmann::json js;
//
//  std::vector<nlohmann::json> xs;
//  xs.reserve(value.size());
//  for (const auto& x : value) {
//    xs.push_back(ValueAsJson(x));
//  }
//  js = xs;
//  return js;
//}

template <typename Tuple, typename F, std::size_t ...Indices>
void for_each_tuple_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) {
  using swallow = int[];
  (void)swallow{1,
                (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...
  };
}
template <typename Tuple, typename F>
void for_each_tuple(Tuple&& tuple, F&& f) {
  constexpr std::size_t N = std::tuple_size<std::remove_reference_t<Tuple>>::value;
  for_each_tuple_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                      std::make_index_sequence<N>{});
}


template <typename... Args>
inline nlohmann::json ValueAsJson(const std::tuple<Args...>& value) {
  nlohmann::json js;

  std::vector<nlohmann::json> xs;
  // xs.reserve(2);
  for_each_tuple(value, [&xs] (auto const& x) {
    xs.push_back(ValueAsJson(x));
  });
  js = xs;
  return js;
}
template <typename T>
inline nlohmann::json ValueAsJson(const std::list<T>& value) {
  nlohmann::json js;

  std::vector<nlohmann::json> xs;
  xs.reserve(value.size());
  for (const auto& x : value) {
    xs.push_back(ValueAsJson(x));
  }
  js = xs;
  return js;
}
template <typename T>
inline nlohmann::json ValueAsJson(const std::vector<T>& value) {
  nlohmann::json js;

  std::vector<nlohmann::json> xs;
  xs.reserve(value.size());
  for (const auto& x : value) {
    xs.push_back(ValueAsJson(x));
  }
  js = xs;
  return js;

}

template <typename T>
inline nlohmann::json ValueAsJson(const std::initializer_list<T>& value) {
  nlohmann::json js;

  std::vector<nlohmann::json> xs;
  xs.reserve(value.size());
  for (const auto& x : value) {
    xs.push_back(ValueAsJson(x));
  }
  js = xs;
  return js;
}

template <>
inline nlohmann::json ValueAsJson(const nlohmann::json& value) {
  return value;
}

template <typename K, typename V>
inline nlohmann::json ValueAsJson(const std::map<K, V>& value) {
  nlohmann::json js;
  js["typename"] = "dict";
//  std::vector<std::tuple<const K&, const V&>> kv_pairs;
  std::vector<std::vector<nlohmann::json>> kv_pairs;
  kv_pairs.reserve(value.size());
  for (const auto& pair : value) {
    kv_pairs.push_back({ValueAsJson(pair.first), ValueAsJson(pair.second)});
  }
  js["key_value_pairs"] = kv_pairs;
  return js;
}


template <>
inline nlohmann::json ValueAsJson(const CategoryKey& value) {
  nlohmann::json js;

  js["typename"] = "CategoryKey";
  js["procs"] = ValueAsJson(value.procs);
  js["ops"] = ValueAsJson(value.ops);
  js["non_ops"] = ValueAsJson(value.non_ops);

  return js;
}
template <>
inline nlohmann::json ValueAsJson(const OverlapResult& value) {
  nlohmann::json js;

  auto overlap_map = value.AsOverlapMap();
  js["overlap"] = ValueAsJson(overlap_map);

//  std::map<std::tuple<int, int, std::string>, int> junk = { { {1, 2, "banana"}, 1 } };
//  js["junk_js"] = ValueAsJson(std::make_tuple(1, 2, "banana"));
//  js["junk_js"] = ValueAsJson(junk);
// FAIL
//  std::map<std::tuple<std::set<CategoryKey>, std::set<int>>, size_t> junk;
//  std::map<std::tuple<std::set<CategoryKey>, int>, size_t> junk;
// PASS
//  std::tuple<std::set<CategoryKey>> junk;
//  std::tuple<std::set<CategoryKey>> junk;
//  std::tuple<std::set<CategoryKey>, int> junk;
//  js["junk_js"] = ValueAsJson(junk);

  auto category_trans_counts = value.AsCategoryTransCountsMap();
  js["category_trans_counts"] = ValueAsJson(category_trans_counts);

  return js;
}

} // namespace rlscope

#endif //RLSCOPE_TRACE_FILE_PARSER_H
