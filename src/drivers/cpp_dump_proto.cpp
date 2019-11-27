//
// Created by jagle on 11/12/2018.
//

//#include "common/debug.h"

#include "error_codes.pb.h"

#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <boost/any.hpp>

#include "cuda_api_profiler/generic_logging.h"
#include "cuda_api_profiler/debug_flags.h"

// Time breakdown:
// - metric: how many events are processed per second by compute overlap.
// - loading data from proto files
// - running overlap computation

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <backward.hpp>

#include <iostream>

#include <assert.h>

//#include "tensorflow/core/lib/core/status.h"
#include "analysis/my_status.h"

#include <list>
#include <initializer_list>

#include <gflags/gflags.h>
#include <memory>

#include "analysis/trace_file_parser.h"

DEFINE_bool(debug, false, "Debug");
DEFINE_string(proto, "", "Path to RLS trace-file protobuf file");
DEFINE_string(iml_directory, "", "Path to --iml-directory used when collecting trace-files");
DEFINE_string(mode, "", "One of: [stats, ls, proto]");

DEFINE_string(cupti_overhead_json, "", "Path to calibration file: mean per-CUDA API CUPTI overhead when GPU activities are recorded (see: CUPTIOverheadTask) ");
DEFINE_string(LD_PRELOAD_overhead_json, "", "Path to calibration file: mean overhead for intercepting CUDA API calls with LD_PRELOAD  (see: CallInterceptionOverheadTask)");
DEFINE_string(pyprof_overhead_json, "", "Path to calibration file: means for (1) Python->C++ interception overhead, (2) operation annotation overhead (see: PyprofOverheadTask)");

using namespace tensorflow;

#define IF_BAD_STATUS_EXIT(msg, status)  \
      if (status.code() != MyStatus::OK().code()) { \
        std::cout << "ERROR: " << msg << ": " << status.ToString() << std::endl; \
        exit(EXIT_FAILURE); \
      }

enum Mode {
  MODE_DUMP_PROTO = 0,
  MODE_LS_FILES = 1,
  MODE_STATS = 2,
  MODE_OVERLAP = 3,
};

void Usage() {
  std::cout << "Usage: " << std::endl;
  std::cout << "  # Dump protobuf file" << std::endl;
  std::cout << "  $ cpp_dump_proto [--mode proto] --proto path/to/trace_file.proto" << std::endl;
  std::cout << std::endl;
  std::cout << "  # ls trace-files" << std::endl;
  std::cout << "  $ cpp_dump_proto --mode ls --iml_directory path/to/iml_directory" << std::endl;
  std::cout << std::endl;
  std::cout << "  # read eo_times for entire trace and dump statistics for each category" << std::endl;
  std::cout << "  $ cpp_dump_proto --mode stats --iml_directory path/to/iml_directory" << std::endl;
}
void UsageAndExit(const std::string& msg) {
  Usage();
  std::cout << "ERROR: " << msg << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  backward::SignalHandling sh;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

//  return 0;

  // std::cout << "SPDLOG_ACTIVE_LEVEL = " << SPDLOG_ACTIVE_LEVEL << std::endl;

  // NOTE: If you only define SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG, this doesn't enable debug logging.
  // It just ensures that the SPDLOG_DEBUG statements are **compiled in**!
  // We still need to turn them on though!
  spdlog::set_level(static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
  // spdlog::set_level(spdlog::level::debug);

  // WARNING: this log statements ALWAYS get compiled in REGARDLESS of SPDLOG_ACTIVE_LEVEL.
  // (i.e. don't use them...)
  // spdlog::info("Always compiled, enabled by default at runtime");

//  SPDLOG_TRACE("Some trace message that will not be evaluated.{} ,{}", 1, 3.23);
//  SPDLOG_DEBUG("Some Debug message that will be evaluated.. {} ,{}", 1, 3.23);
//  SPDLOG_DEBUG("Some debug message to default logger that will be evaluated");
//  SPDLOG_INFO("Compile time info message");

  MyStatus status = MyStatus::OK();

  boost::filesystem::path proto_path(FLAGS_proto);
  if (FLAGS_proto != "" && !boost::filesystem::is_regular_file(proto_path)) {
    std::cout << "ERROR: --proto_path must be a path to a protobuf trace-file." << std::endl;
    exit(EXIT_FAILURE);
  }

  boost::filesystem::path iml_path(FLAGS_iml_directory);
  if (FLAGS_iml_directory != "" && !boost::filesystem::is_directory(iml_path)) {
    std::cout << "ERROR: --iml_directory must be a path to a root --iml-directory given when collecting traces" << std::endl;
    exit(EXIT_FAILURE);
  }

//  if (FLAGS_cupti_overhead_json != "") {
//    json j;
//    status = ReadJson(FLAGS_cupti_overhead_json, &j);
//    IF_BAD_STATUS_EXIT("Failed to read json from --cupti_overhead_json", status);
//    DBG_LOG("Read json from: {}", FLAGS_cupti_overhead_json);
//    std::cout << j << std::endl;
//    exit(EXIT_SUCCESS);
//  }

  Mode mode;
  if (FLAGS_mode != "") {
    if (FLAGS_mode == "ls") {
      mode = Mode::MODE_LS_FILES;
    } else if (FLAGS_mode == "stats") {
      mode = Mode::MODE_STATS;
    } else if (FLAGS_mode == "overlap") {
      mode = Mode::MODE_OVERLAP;
    } else if (FLAGS_mode == "proto") {
      mode = Mode::MODE_DUMP_PROTO;
    } else {
      UsageAndExit("--mode must be one of [stats, ls, proto]");
    }
  } else if (FLAGS_proto != "") {
    mode = Mode::MODE_DUMP_PROTO;
  } else {
    UsageAndExit("not sure what --mode to run in");
  }

  if (mode == Mode::MODE_LS_FILES) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=ls");
    }
  }

  if (mode == Mode::MODE_STATS) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=stats");
    }
  }

  if (mode == Mode::MODE_OVERLAP) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=overlap");
    }
  }

  if (mode == Mode::MODE_DUMP_PROTO) {
    if (FLAGS_proto == "") {
      UsageAndExit("--proto is required for --mode=proto");
    }
  }

  if (mode == Mode::MODE_DUMP_PROTO) {
    std::unique_ptr<IEventFileParser> parser;
    status = GetRLSEventParser(FLAGS_proto, &parser);
    IF_BAD_STATUS_EXIT("Not sure how to parse", status);
    CategoryTimes category_times;
    status = parser->ReadFile(&category_times);
    IF_BAD_STATUS_EXIT("Failed to read --proto", status);
    category_times.Print(std::cout, 0);
    std::cout << "\n";
    exit(EXIT_SUCCESS);
  }

  if (mode == Mode::MODE_LS_FILES) {
    std::list<std::string> paths;
    status = FindRLSFiles(FLAGS_iml_directory, &paths);
    IF_BAD_STATUS_EXIT("Failed to ls trace-files in --iml_directory", status);
    for (const auto& path : paths) {
      std::cout << path << std::endl;
    }
    exit(EXIT_SUCCESS);
  }

  auto mk_parser = [] {
    MyStatus status = MyStatus::OK();
    RawTraceParser parser(FLAGS_iml_directory,
                          FLAGS_cupti_overhead_json,
                          FLAGS_LD_PRELOAD_overhead_json,
                          FLAGS_pyprof_overhead_json);
    status = parser.Init();
    IF_BAD_STATUS_EXIT("Failed to collect stats for --iml_directory", status);
    return parser;
  };

  if (mode == Mode::MODE_STATS) {
    auto parser = mk_parser();
    parser.EachEntireTrace([] (const CategoryTimes& category_times, const EntireTraceMeta& meta) {
            std::cout << "Machine=" << meta.machine
                      << ", " << "Process=" << meta.process
                      << ", " << "Phase=" << meta.phase
                      << std::endl;
            category_times.PrintSummary(std::cout, 1);
            std::cout << std::endl;
            return MyStatus::OK();
    });
    exit(EXIT_SUCCESS);
  }

  if (mode == Mode::MODE_OVERLAP) {
    auto parser = mk_parser();
    std::shared_ptr<SimpleTimer> timer(new SimpleTimer("mode_overlap"));
    timer->ResetStartTime();
    timer->MakeVerbose(&std::cout);
    parser.timer = timer;
    size_t n_total_events = 0;
    parser.EachEntireTrace([timer, &n_total_events] (const CategoryTimes& category_times, const EntireTraceMeta& meta) {
      std::cout << "Machine=" << meta.machine
                << ", " << "Process=" << meta.process
                << ", " << "Phase=" << meta.phase
                << std::endl;
      category_times.PrintSummary(std::cout, 1);
      std::cout << std::endl;

      n_total_events += category_times.TotalEvents();
      OverlapComputer overlap_computer(category_times);
      auto r = overlap_computer.ComputeOverlap();
      if (timer) {
        std::stringstream ss;
        ss << "ComputeOverlap(machine=" << meta.machine << ", process=" << meta.process << ", phase=" << meta.phase << ")";
        timer->EndOperation(ss.str());
      }
      std::cout << std::endl;
      r.Print(std::cout, 1);
      std::cout << std::endl;
      return MyStatus::OK();
    });
    if (timer) {
      auto total_time_sec = timer->TotalTimeSec();
      timer->RecordThroughput("overlap events", n_total_events);
      timer->RecordThroughput("total_sec", total_time_sec);
      SimpleTimer::MetricValue events_per_sec =
          static_cast<SimpleTimer::MetricValue>(n_total_events) /
          static_cast<SimpleTimer::MetricValue>(total_time_sec);
      timer->RecordThroughput("overlap events/sec", events_per_sec);

      std::cout << std::endl;
      timer->Print(std::cout, 0);
      std::cout << std::endl;
    }
    exit(EXIT_SUCCESS);
  }

  // Shouldn't reach here.
  assert(false);

  return 0;
}

