//
// Created by jagle on 11/13/2019.
//

#ifndef IML_GENERIC_LOGGING_H
#define IML_GENERIC_LOGGING_H

#include <assert.h>
#include "cuda_api_profiler/generic_logging.h"

#include <ostream>
#include <map>

namespace tensorflow {

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

class SimpleTimer {
public:
  using MetricValue = float;
  using TimeUsec = int64_t;
  std::string _name;

  OrderedMap<std::string, TimeUsec> _op_duration_usec;
  OrderedMap<std::string, MetricValue> _metrics;

//  std::map<std::string, TimeUsec> _op_duration_usec;
//  std::map<std::string, MetricValue> _metrics;

  TimeUsec _last_time_usec;
  TimeUsec _start_time_usec;
  SimpleTimer(const std::string& name);

  void ResetStartTime();
  double TotalTimeSec() const;
  void EndOperation(const std::string& operation);
  void Print(std::ostream& out, int indent);
  void RecordThroughput(const std::string& metric_name, MetricValue metric);

};

std::ostream& PrintIndent(std::ostream& out, int indent);

}

#endif //IML_GENERIC_LOGGING_H
