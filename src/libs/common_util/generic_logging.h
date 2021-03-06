//
// Created by jagle on 11/13/2019.
//

#pragma once

#include <assert.h>
#include "defines.h"

#include <ostream>
#include <map>
#include <set>
#include <vector>
#include <list>

#include <sys/time.h>
#include <sys/resource.h>

#include <backward.hpp>

namespace rlscope {

template <class OStream>
void DumpStacktrace(OStream& os, size_t skip_frames) {
  backward::StackTrace st;
  const size_t MAX_STACKFRAMES = 32;
  st.load_here(MAX_STACKFRAMES);
  // Last 4 frames are always related to backward.hpp or logging.cc.
  // Skip those frames; make the latest frame the LOG(FAIL) or DCHECK failure.
  size_t idx;
  if (st.size() < skip_frames) {
    // Print the whole thing.
    idx = 0;
  } else {
    // Skip the last 4 frames.
    idx = skip_frames;
  }
  st.load_from(st[idx].addr, MAX_STACKFRAMES);
  backward::Printer p;
  p.print(st, os);
}


template <typename K, typename V>
class OrderedMap {
public:
  using ID = int;
  std::map<K, V> kv_map;
  std::map<ID, K> id_map;
  ID _next_id;
  OrderedMap() :
      _next_id(0) {
  }
  inline size_t size() const {
    return kv_map.size();
  }
  template <typename Func>
  bool EachPair(Func func) {
    for (ID id = 0; static_cast<size_t>(id) < id_map.size(); id++) {
      auto const& key = id_map.at(id);
      auto const& value = kv_map.at(key);
      bool should_continue = func(key, value);
      if (!should_continue) {
        // Stop early; signal to caller by returning false.
        return false;
      }
    }
    // Iterated over entire map; signal to caller by returning true.
    return true;
  }
  int &operator[] (int);

  ID LastID() const {
    return _next_id - 1;
  }

//  V& operator[](const K& index) // for non-const objects: can be used for assignment
//  {
//    return kv_map[index];
//  }

//  V& operator[](const K& index, const V& value) // for non-const objects: can be used for assignment
  V& Set(const K& index, const V& value) {
    // NOTE: we need access the the value we are setting, so we don't overload operator[]
    assert(kv_map.find(index) == kv_map.end());
    assert(kv_map.size() == id_map.size());
    auto id = _next_id;
    _next_id += 1;
    id_map[id] = index;
    kv_map[index] = value;
    assert(kv_map.size() == id_map.size());
    return kv_map[index];
  }

//  const V& operator[](const K& index) const // for const objects: can only be used for access
//  {
//    return kv_map[index];
//  }

  V& at(const K& index)
  {
    return kv_map.at(index);
  }
  const V& at(const K& index) const
  {
    return kv_map.at(index);
  }

  bool contains(const K& index) const
  {
    return kv_map.find(index) != kv_map.end();
  }

};

static inline size_t ResidentMemBytes() {
  int who = RUSAGE_SELF;
  struct rusage usage;
  int ret;
  ret = getrusage(who,&usage);
  assert(ret != -1);
  return usage.ru_maxrss * 1024;
}

// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
template <typename OStream>
void PrintBytes(OStream& out, uint64_t bytes) {
  std::vector<std::string> suffixes {
      "B",
      "KB",
      "MB",
      "GB",
      "TB",
      "PB",
      "EB",
  };
  size_t s = 0; // which suffix to use
  double count = bytes;
  while (count >= 1024 && s < suffixes.size()) {
    s++;
    count /= 1024;
  }

//  if (count - floor(count) == 0.0) {
//    // sprintf(buf, "%d %s", (int)count, suffixes[s]);
//    out << static_cast<int>(count) << " " << suffixes[s];
//  } else {
//    // sprintf(buf, "%.1f %s", count, suffixes[s]);
//    auto old_precision = out.precision(1);
//    out << count << " " << suffixes[s];
//    out.precision(old_precision);
//  }

  // sprintf(buf, "%.1f %s", count, suffixes[s]);
  auto old_precision = out.precision(1);

  out << std::fixed << count << " " << suffixes[s];
  out.precision(old_precision);
  out.unsetf(std::ios_base::floatfield);
}

class SimpleTimer {
public:
  using MetricValue = float;
  using TimeUsec = int64_t;
  std::string _name;
  std::ostream* _out;

  struct OpStats {
    TimeUsec duration_us;
    size_t mem_bytes;
    OpStats() :
        duration_us(0),
        mem_bytes(0) {
    }
    OpStats(
        TimeUsec duration_us,
        size_t mem_bytes) :
        duration_us(duration_us),
        mem_bytes(mem_bytes) {
    }
  };

  OrderedMap<std::string, OpStats> _op_stats;
//  OrderedMap<std::string, TimeUsec> _op_duration_usec;
//  OrderedMap<std::string, size_t> _op_mem_bytes;
  OrderedMap<std::string, MetricValue> _metrics;

  TimeUsec _last_time_usec;
  size_t _last_mem_bytes;
  TimeUsec _start_time_usec;
  size_t _start_mem_bytes;
  SimpleTimer(const std::string& name);

  void MakeVerbose(std::ostream* out);

  template <typename OStream>
  void _PrintLine(OStream& out,
                  int i, const std::string& operation, TimeUsec duration_us, size_t mem_bytes) {
    // e.g.
    // [23] name="ReadProto(category_events.trace_4.proto)" = 0.469798 sec, mem=1MB
    MetricValue duration_sec = static_cast<MetricValue>(duration_us) / static_cast<MetricValue>(USEC_IN_SEC);
    out << "[" << i << "] "
        << "name=\"" << operation << "\"" << " = " << duration_sec << " sec"
        << ", mem=";
    PrintBytes(out, mem_bytes);
  }


  void ResetStartTime();
  double TotalTimeSec() const;
  size_t TotalMemBytes() const;
  void EndOperation(const std::string& operation);
  void Print(std::ostream& out, int indent);
  void RecordThroughput(const std::string& metric_name, MetricValue metric);

};


//std::ostream& PrintIndent(std::ostream& out, int indent);
template <typename OStream>
OStream& PrintIndent(OStream& out, int indent) {
  for (int i = 0; i < indent; i++) {
    out << "  ";
  }
  return out;
}

//// NOTE: this result in a "multiple definition" compilation error; not sure why.
//template <>
//void PrintValue<std::string>(std::ostream& os, const std::string& value) {
//  os << "\"" << value << "\"";
//}

template <typename OStream, typename T>
struct PrintValueImpl {
  static void impl(OStream& os, const T& value) {
    os << value;
  }
};
template <typename OStream, typename T>
void PrintValue(OStream& os, const T& value) {
  PrintValueImpl<OStream, T>::impl(os, value);
}

//template <typename OStream, typename T>
//struct PrintValueImpl<OStream, ...> {
//  static void impl(OStream& os, const T& value) {
//  }
//};

template <typename OStream>
struct PrintValueImpl<OStream, std::string> {
  static void impl(OStream& os, const std::string& value) {
    os << "\"" << value << "\"";
  }
};

template <typename OStream, typename T>
struct PrintValueImpl<OStream, std::set<T>> {
  static void impl(OStream& os, const std::set<T>& value) {
    os << "{";
    size_t i = 0;
    for (auto const& val : value) {
      if (i > 0) {
        os << ", ";
      }
      PrintValue(os, val);
      i += 1;
    }
    os << "}";
  }
};

template <typename OStream, typename T>
struct PrintValueImpl<OStream, std::list<T>> {
  static void impl(OStream& os, const std::list<T>& value) {
    os << "[";
    size_t i = 0;
    for (auto const& val : value) {
      if (i > 0) {
        os << ", ";
      }
      PrintValue(os, val);
      i += 1;
    }
    os << "]";
  }
};

template <typename OStream, typename T>
struct PrintValueImpl<OStream, std::vector<T>> {
  static void impl(OStream& os, const std::vector<T>& value) {
    os << "[";
    size_t i = 0;
    for (auto const& val : value) {
      if (i > 0) {
        os << ", ";
      }
      PrintValue(os, val);
      i += 1;
    }
    os << "]";
  }
};

template <typename OStream, typename T>
struct PrintValueImpl<OStream, std::initializer_list<T>> {
  static void impl(OStream& os, const std::initializer_list<T>& value) {
    os << "[";
    size_t i = 0;
    for (auto const& val : value) {
      if (i > 0) {
        os << ", ";
      }
      PrintValue(os, val);
      i += 1;
    }
    os << "]";
  }
};

template <typename OStream, typename K, typename V>
struct PrintValueImpl<OStream, std::map<K, V>> {
  static void impl(OStream& os, const std::map<K, V>& value) {
    os << "{";
    size_t i = 0;
    for (auto const& pair : value) {
      if (i > 0) {
        os << ", ";
      }
      PrintValue(os, pair.first);
      os << "=";
      PrintValue(os, pair.second);
      i += 1;
    }
    os << "}";
  }
};

template <typename OStream>
void print_args(OStream& os, int arg_idx) {
}
template <typename OStream, typename Arg, typename ...Args>
void print_args(OStream& os, int arg_idx, const Arg& arg, const Args&... args) {
  if (arg_idx > 0) {
    os << ", ";
  }
  rlscope::PrintValue(os, arg);
  print_args(os, arg_idx + 1, args...);
}

template <typename OStream, typename ...Args>
OStream& log_func_call_impl(OStream& os, char const * fname, Args &&... args)
{
    os << fname << "(";
    print_args(os, 0, args...);
    os << ")";
    return os;
}

#define LOG_FUNC_CALL(os, ...) log_func_call_impl(os, __func__, ##__VA_ARGS__)

#define RLS_LOG_FUNC(TAG, ...)        \
  {                                   \
    std::stringstream ss;             \
    LOG_FUNC_CALL(ss, ##__VA_ARGS__); \
    RLS_LOG(TAG, "{}", ss.str());     \
  }


}

