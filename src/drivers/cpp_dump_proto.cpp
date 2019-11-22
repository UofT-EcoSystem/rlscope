//
// Created by jagle on 11/12/2018.
//

//#include "common/debug.h"

#include <spdlog/spdlog.h>

#include <boost/filesystem.hpp>
#include <boost/any.hpp>

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

  if (mode == Mode::MODE_STATS) {
    RawTraceParser parser(FLAGS_iml_directory);
    status = parser.Init();
    IF_BAD_STATUS_EXIT("Failed to collect stats for --iml_directory", status);
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
    RawTraceParser parser(FLAGS_iml_directory);
    status = parser.Init();
    IF_BAD_STATUS_EXIT("Failed to collect stats for --iml_directory", status);
    parser.EachEntireTrace([] (const CategoryTimes& category_times, const EntireTraceMeta& meta) {
      std::cout << "Machine=" << meta.machine
                << ", " << "Process=" << meta.process
                << ", " << "Phase=" << meta.phase
                << std::endl;
      category_times.PrintSummary(std::cout, 1);
      OverlapComputer overlap_computer(category_times);
      auto r = overlap_computer.ComputeOverlap();
      r.Print(std::cout, 1);
      std::cout << std::endl;
      return MyStatus::OK();
    });
    exit(EXIT_SUCCESS);
  }

  // Shouldn't reach here.
  assert(false);

  return 0;
}

