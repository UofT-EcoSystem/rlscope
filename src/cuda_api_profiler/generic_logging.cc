//
// Created by jagle on 11/13/2019.
//

#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/usec_timer.h"
#include "cuda_api_profiler/defines.h"


#include <assert.h>
#include <ostream>
#include <iostream>

namespace tensorflow {

SimpleTimer::SimpleTimer(const std::string& name) :
    _name(name),
    _out(nullptr),
    _last_time_usec(0),
    _start_time_usec(0)
{
}

void SimpleTimer::MakeVerbose(std::ostream* out) {
  _out = out;
}

void SimpleTimer::ResetStartTime() {
//  auto now_us = Env::Default()->NowMicros();
  auto now_us = TimeNowMicros();
  _last_time_usec = now_us;
  _start_time_usec = now_us;
}

double SimpleTimer::TotalTimeSec() const {
  // auto now_us = TimeNowMicros();
  // auto time_us = now_us - _start_time_usec;
  auto time_us = _last_time_usec - _start_time_usec;
  // return ((double)time_us) / ((double)USEC_IN_SEC);
  return static_cast<double>(time_us) / static_cast<double>(USEC_IN_SEC);
}

void SimpleTimer::EndOperation(const std::string& operation) {
//  auto now_us = Env::Default()->NowMicros();
  auto now_us = TimeNowMicros();
  assert(_last_time_usec != 0);
  // _op_duration_usec[operation] = now_us - _last_time_usec;
  auto duration_us = now_us - _last_time_usec;
  _op_duration_usec.Set(operation, duration_us);
  if (_out) {
    MetricValue duration_sec = static_cast<MetricValue>(duration_us) / static_cast<MetricValue>(USEC_IN_SEC);
    (*_out) << "[" << _op_duration_usec.LastID() << "] name=\"" << operation << "\"" << " = " << duration_sec << " sec" << std::endl;
  }
  _last_time_usec = now_us;
}

void SimpleTimer::Print(std::ostream& out, int indent) {
  PrintIndent(out, indent);
  out << "SimpleTimer: name = " << _name;

  {
    out << "\n";
    PrintIndent(out, indent + 1);
    out << "Operations: size = " << _op_duration_usec.size();
    size_t i = 0;
    _op_duration_usec.EachPair([&i, &out, indent] (const auto& op, const auto& duration_usec) {
      double duration_sec = duration_usec / ((double) 1e6);
      out << "\n";
      PrintIndent(out, indent + 2);
      out << "[" << i << "] name=\"" << op << "\"" << " = " << duration_sec << " sec";
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

//  for (auto const& pair : _op_duration_usec) {
//    const auto& op = pair.first;
//    auto duration_usec = pair.second;
//    double duration_sec = duration_usec / ((double) 1e6);
//
//    out << "\n";
//    PrintIndent(out, indent + 1);
//    out << "name=\"" << pair.first << "\"" << " = " << duration_sec << " sec";
//
//  }
}

void SimpleTimer::RecordThroughput(const std::string& metric_name, MetricValue metric) {
  // _metrics[metric_name] = metric;
  _metrics.Set(metric_name, metric);
}

}
