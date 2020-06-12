//
// Created by jagle on 11/13/2019.
//

#include "common_util.h"

#include <assert.h>
#include <ostream>
#include <iostream>

namespace rlscope {

SimpleTimer::SimpleTimer(const std::string& name) :
    _name(name),
    _out(nullptr),
    _last_time_usec(0),
    _last_mem_bytes(0),
    _start_time_usec(0),
    _start_mem_bytes(0)
{
}

void SimpleTimer::MakeVerbose(std::ostream* out) {
  _out = out;
}

void SimpleTimer::ResetStartTime() {
//  auto now_us = rlscope::TimeNowMicros();
  auto now_us = TimeNowMicros();
  auto now_bytes = ResidentMemBytes();
  _last_time_usec = now_us;
  _last_mem_bytes = now_bytes;
  _start_time_usec = now_us;
  _start_mem_bytes = now_bytes;
}

double SimpleTimer::TotalTimeSec() const {
  auto time_us = _last_time_usec - _start_time_usec;
  return static_cast<double>(time_us) / static_cast<double>(USEC_IN_SEC);
}

size_t SimpleTimer::TotalMemBytes() const {
  auto mem_bytes = _last_mem_bytes - _start_mem_bytes;
  return mem_bytes;
}


void SimpleTimer::EndOperation(const std::string& operation) {
//  auto now_us = rlscope::TimeNowMicros();
  auto now_us = TimeNowMicros();
  auto now_bytes = ResidentMemBytes();
  assert(_last_time_usec != 0);
  assert(_last_mem_bytes != 0);
  auto duration_us = now_us - _last_time_usec;
  auto mem_bytes = now_bytes - _last_mem_bytes;
//  _op_duration_usec.Set(operation, duration_us);
//  _op_mem_bytes.Set(operation, mem_bytes);
  _op_stats.Set(operation, OpStats(duration_us, mem_bytes));
  if (_out) {
    this->_PrintLine(*_out, _op_stats.LastID(), operation, duration_us, mem_bytes);
    (*_out) << std::endl;
  }
  _last_time_usec = now_us;
  _last_mem_bytes = now_bytes;
}


void SimpleTimer::Print(std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << "SimpleTimer: name = " << _name;

  {
    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Operations: size = " << _op_stats.size();
    size_t i = 0;
    _op_stats.EachPair([this, &i, &out, indent] (const auto& operation, const auto& op_stats) {
      out << "\n";
      PrintIndent(out, indent + 2);
      this->_PrintLine(out, i, operation, op_stats.duration_us, op_stats.mem_bytes);
      i++;
      return true;
    });
  }

  {
    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Metrics: size = " << _metrics.size();
    size_t i = 0;
    _metrics.EachPair([&i, &out, indent] (const auto& metric, const auto& metric_value) {
      out << "\n";
      PrintIndent(out, indent + 2);
      out << "[" << i << "] metric=\"" << metric << "\"" << " = " << metric_value;
      i++;
      return true;
    });
  }

}

void SimpleTimer::RecordThroughput(const std::string& metric_name, MetricValue metric) {
  _metrics.Set(metric_name, metric);
}

}
