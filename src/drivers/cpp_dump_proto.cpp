//
// Created by jagle on 11/12/2018.
//

//#include "common/debug.h"

#include <boost/filesystem.hpp>
#include <boost/any.hpp>

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

using namespace tensorflow;

#define IF_BAD_STATUS_EXIT(msg, status)  \
      if (status.code() != MyStatus::OK().code()) { \
        std::cout << "ERROR: " << msg << ": " << status.ToString() << std::endl; \
        exit(EXIT_FAILURE); \
      }

enum Mode {
  MODE_DUMP_PROTO = 0,
  MODE_LS_FILES = 1,
};

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

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
  if (FLAGS_iml_directory != "") {
    mode = Mode::MODE_LS_FILES;
  } else if (FLAGS_proto != "") {
    mode = Mode::MODE_DUMP_PROTO;
  } else {
    std::cout << "Usage: " << std::endl;
    std::cout << "  # Dump protobuf file" << std::endl;
    std::cout << "  $ cpp_dump_proto --proto path/to/trace_file.proto" << std::endl;
    std::cout << std::endl;
    std::cout << "  # ls trace-files" << std::endl;
    std::cout << "  $ cpp_dump_proto --iml_directory path/to/iml_directory" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (mode == Mode::MODE_DUMP_PROTO) {
    if (CategoryEventsParser::IsFile(FLAGS_proto)) {
      CategoryEventsParser parser;
      CategoryTimes category_times;
      status = parser.ReadFile(FLAGS_proto, &category_times);
      IF_BAD_STATUS_EXIT("Failed to read --proto", status);
      PrintCategoryTimes(category_times, std::cout, 0);
      exit(EXIT_SUCCESS);
    } else {
      std::cout << "Not sure how to parse protobuf file @ --proto=" << FLAGS_proto;
      exit(EXIT_FAILURE);
    }
  } else if (mode == Mode::MODE_LS_FILES) {
    std::list<std::string> paths;
    status = FindRLSFiles(FLAGS_iml_directory, RLSFileType::CATEGORY_EVENTS_FILE, &paths);
    IF_BAD_STATUS_EXIT("Failed to ls trace-files in --iml_directory", status);
    for (const auto& path : paths) {
      std::cout << path << std::endl;
    }
  } else {
    // Shouldn't reach here.
    assert(false);
  }

  return 0;
}

