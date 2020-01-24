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
#include "analysis/sample_periods.h"

#include <list>
#include <initializer_list>

#include <gflags/gflags.h>
#include <memory>

#include "analysis/trace_file_parser.h"

DEFINE_bool(debug, false, "Debug: give additional verbose output");
DEFINE_bool(ignore_memcpy, false, "Ignore CUDA memcpy events when reading cuda_device_events*.proto");
DEFINE_string(proto, "", "Path to RLS trace-file protobuf file");
DEFINE_string(iml_directory, "", "Path to --iml-directory used when collecting trace-files");
DEFINE_string(mode, "", "One of: [stats, ls, proto, polling_util]");

DEFINE_string(cupti_overhead_json, "", "Path to calibration file: mean per-CUDA API CUPTI overhead when GPU activities are recorded (see: CUPTIOverheadTask) ");
DEFINE_string(LD_PRELOAD_overhead_json, "", "Path to calibration file: mean overhead for intercepting CUDA API calls with LD_PRELOAD  (see: CallInterceptionOverheadTask)");
DEFINE_string(python_annotation_json, "", "Path to calibration file: means for operation annotation overhead (see: PyprofOverheadTask)");
DEFINE_string(python_clib_interception_tensorflow_json, "", "Path to calibration file: means for TensorFlow Python->C++ interception overhead (see: PyprofOverheadTask)");
DEFINE_string(python_clib_interception_simulator_json, "", "Path to calibration file: means for Simulator Python->C++ interception overhead (see: PyprofOverheadTask)");


// Window size (a.k.a. sample period [NVIDIA documentation]): the number of "bins" we look at when calculating the GPU kernel time.
//   NOTE: if window size is in milliseconds, then it should be evenly divisible by the polling interval.
// Polling interval: the size of each "bin" where we check whether or not a GPU kernel ran, and assign 0/1 to it.
DEFINE_int64(polling_interval_us, 0, "nvidia-smi sampling period in microseconds (http://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf): Percent of time over the past sample period during which one or more kernels was executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product");
//DEFINE_int64(window_size_us, 0, "nvidia-smi sampling period in microseconds (http://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf): Percent of time over the past sample period during which one or more kernels was executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product");

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
  MODE_READ_FILES = 4,
  MODE_POLLING_UTIL = 5,
  MODE_GPU_KERNELS = 6,
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
    } else if (FLAGS_mode == "read") {
      mode = Mode::MODE_READ_FILES;
    } else if (FLAGS_mode == "overlap") {
      mode = Mode::MODE_OVERLAP;
    } else if (FLAGS_mode == "proto") {
      mode = Mode::MODE_DUMP_PROTO;
    } else if (FLAGS_mode == "polling_util") {
      mode = Mode::MODE_POLLING_UTIL;
    } else if (FLAGS_mode == "gpu_kernels") {
      mode = Mode::MODE_GPU_KERNELS;
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

  if (mode == Mode::MODE_READ_FILES) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=read");
    }
  }

  if (mode == Mode::MODE_OVERLAP) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=overlap");
    }
  }

  if (mode == Mode::MODE_POLLING_UTIL) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=polling_util");
    }

    if (FLAGS_polling_interval_us == 0) {
      UsageAndExit("--polling_interval_us is required for --mode=polling_util");
    }

    if (FLAGS_cupti_overhead_json != ""
        || FLAGS_LD_PRELOAD_overhead_json != ""
        || FLAGS_python_clib_interception_tensorflow_json != ""
        || FLAGS_python_clib_interception_simulator_json != "")
    {
      UsageAndExit("Calibration files (e.g., --cupti-overhead-json-path) should NOT be provided for mode=polling_util");
    }
  }

  if (mode == Mode::MODE_DUMP_PROTO) {
    if (FLAGS_proto == "") {
      UsageAndExit("--proto is required for --mode=proto");
    }
  }

  if (mode == Mode::MODE_GPU_KERNELS) {
    if (FLAGS_iml_directory == "") {
      UsageAndExit("--iml-directory is required for --mode=gpu_kernels");
    }
  }

  EntireTraceSelector selector;
  if (FLAGS_ignore_memcpy) {
    selector.ignore_memcpy = true;
  }
  {
    std::stringstream ss;
    ss << "How to parse trace-files:\n";
    selector.Print(ss, 1);
    DBG_LOG("{}", ss.str());
  }

//  if (mode == Mode::MODE_DUMP_PROTO) {
//    std::unique_ptr<IEventFileParser> parser;
//    TraceParserMeta parser_meta("", "", "");
//    status = GetRLSEventParser(FLAGS_proto, parser_meta, &parser);
//    IF_BAD_STATUS_EXIT("Not sure how to parse", status);
//    CategoryTimes category_times;
//    status = parser->ReadFile(&category_times);
//    IF_BAD_STATUS_EXIT("Failed to read --proto", status);
//    category_times.Print(std::cout, 0);
//    std::cout << "\n";
//    exit(EXIT_SUCCESS);
//  }

  if (mode == Mode::MODE_LS_FILES) {
    std::list<std::string> paths;
    status = FindRLSFiles(FLAGS_iml_directory, &paths);
    IF_BAD_STATUS_EXIT("Failed to ls trace-files in --iml_directory", status);
    for (const auto& path : paths) {
      std::cout << path << std::endl;
    }
    exit(EXIT_SUCCESS);
  }

  auto mk_timer = [] () {
    std::shared_ptr<SimpleTimer> timer(new SimpleTimer("timer"));
    timer->ResetStartTime();
    timer->MakeVerbose(&std::cout);
    return timer;
  };

  auto mk_parser = [mk_timer] () {
    MyStatus status = MyStatus::OK();
    RawTraceParser parser(FLAGS_iml_directory,
                          FLAGS_cupti_overhead_json,
                          FLAGS_LD_PRELOAD_overhead_json,
                          FLAGS_python_annotation_json,
                          FLAGS_python_clib_interception_tensorflow_json,
                          FLAGS_python_clib_interception_simulator_json);

    auto timer = mk_timer();
    parser.SetTimer(timer);

    status = parser.Init();
    IF_BAD_STATUS_EXIT("Failed to collect stats for --iml_directory", status);
    return parser;
  };

  if (mode == Mode::MODE_STATS) {
    auto parser = mk_parser();
    parser.EachEntireTrace([] (std::unique_ptr<CategoryTimes> category_times, const EntireTraceMeta& meta) {
            std::cout << "Machine=" << meta.machine
                      << ", " << "Process=" << meta.process
                      << ", " << "Phase=" << meta.phase
                      << std::endl;
            category_times->PrintSummary(std::cout, 1);
            std::cout << std::endl;
            return MyStatus::OK();
    }, selector);
    exit(EXIT_SUCCESS);
  }

  if (mode == Mode::MODE_READ_FILES) {
    auto timer = mk_timer();
    std::list<std::string> paths;
    status = FindRLSFiles(FLAGS_iml_directory, &paths);
    IF_BAD_STATUS_EXIT("Failed to ls trace-files in --iml_directory", status);
    for (const auto& path : paths) {
      std::cout << path << std::endl;
      {
        std::unique_ptr<ITraceProtoReader> reader;
        status = GetTraceProtoReader(path, &reader);
        if (IS_BAD_STATUS(status)) {
          std::stringstream ss;
          ss << "Not sure how to read files like " << path;
          IF_BAD_STATUS_EXIT(ss.str(), status);
        }
        status = reader->Init();
        if (IS_BAD_STATUS(status)) {
          std::stringstream ss;
          ss << "Failed to read trace-file = " << path;
          IF_BAD_STATUS_EXIT(ss.str(), status);
        }
      }

      if (timer) {
        std::stringstream ss;
        boost::filesystem::path bpath(path);
        ss << "ReadProto(" << bpath.filename().string() << ")";
        timer->EndOperation(ss.str());
      }

    }
    if (timer) {
      std::cout << std::endl;
      timer->Print(std::cout, 0);
      std::cout << std::endl;
    }
    exit(EXIT_SUCCESS);
  }

  if (mode == Mode::MODE_OVERLAP) {
    auto parser = mk_parser();
    auto timer = parser.timer;
    size_t n_total_events = 0;
    parser.EachEntireTrace([timer, &n_total_events] (std::unique_ptr<CategoryTimes> category_times, const EntireTraceMeta& meta) {
      if (FLAGS_debug) {
        std::cout << "Machine=" << meta.machine
                  << ", " << "Process=" << meta.process
                  << ", " << "Phase=" << meta.phase
                  << std::endl;
        category_times->PrintSummary(std::cout, 1);
        std::cout << std::endl;
      }

      if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
        std::stringstream ss;
        ss << "CategoryTimes details:";

        std::set<Operation> ops;
        for (const auto& pair : category_times->eo_times) {
          auto const &category_key = pair.first;
          ops.insert(category_key.ops.begin(), category_key.ops.end());
        }
        ss << "\n";
        PrintIndent(ss, 1);
        // op_names = {sample_action, step}
        ss << "op_names = ";
        PrintValue(ss, ops);

        for (const auto& pair : category_times->eo_times) {
          auto const& category_key = pair.first;
          auto const& eo_events = pair.second;

          if (eo_events.KeepNames()) {
            ss << "\n";
            category_key.Print(ss, 1);
            auto const& names = eo_events.UniqueNames();
            ss << "\n";
            PrintIndent(ss, 2);
            ss << "names = ";
            PrintValue(ss, names);
          }
        }

        DBG_LOG("{}", ss.str());
      }

      n_total_events += category_times->TotalEvents();
      OverlapComputer overlap_computer(*category_times);
      if (SHOULD_DEBUG(FEATURE_OVERLAP)) {
        overlap_computer.debug = true;
      }
      auto r = overlap_computer.ComputeOverlap();

      if (SHOULD_DEBUG(FEATURE_ANY) || FLAGS_debug) {
        std::cout << std::endl;
        r.Print(std::cout, 1);
        std::cout << std::endl;
      }

      if (timer) {
        std::stringstream ss;
        ss << "ComputeOverlap(machine=" << meta.machine << ", process=" << meta.process << ", phase=" << meta.phase << ")";
        timer->EndOperation(ss.str());
      }
      r.DumpVennJS(
          FLAGS_iml_directory,
          meta.machine, meta.process, meta.phase);
      if (timer) {
        std::stringstream ss;
        ss << "DumpVennJS(machine=" << meta.machine << ", process=" << meta.process << ", phase=" << meta.phase << ")";
        timer->EndOperation(ss.str());
      }

      return MyStatus::OK();
    }, selector);
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

  if (mode == Mode::MODE_POLLING_UTIL) {
    auto parser = mk_parser();
    auto timer = parser.timer;
    size_t n_total_events = 0;
    // Q: how can we instruct this thing ONLY to read GPU kernels times and to skip reading other trace-files...?
    // A: just limit the RLS file types.
    // Q: How can we merge GPU kernels time ACROSS phases?
    // A: Options
    // - merge EOEvents
    //   - PRO: simpler to split
    //   - CON: doubles memory usage in worst-case naive implementation... we can probably keep memory the same if we are smart.
    // - iterable structure
    //   - PRO: no space increase
    //   - CON: much harder to split since splits could be across several files (really complicated to do this).
    std::set<RLSFileType> file_types = {CUDA_DEVICE_EVENTS_FILE};
    std::list<std::tuple<std::unique_ptr<CategoryTimes>, EntireTraceMeta>> all_category_times;
    parser.EachEntireTraceWithFileType([timer, &n_total_events, &all_category_times] (std::unique_ptr<CategoryTimes> category_times, const EntireTraceMeta& meta) {
      if (FLAGS_debug) {
        std::cout << "Machine=" << meta.machine
                  << ", " << "Process=" << meta.process
                  << ", " << "Phase=" << meta.phase
                  << std::endl;
        category_times->PrintSummary(std::cout, 1);
        std::cout << std::endl;
      }

      // Get rid of procs since we just want to know if a GPU is running (don't care which process it belongs to)...
      //   CategoryKey(procs={selfplay_worker_0_generation_0}, ops={}, non_ops={GPU})
      //   =>
      //   CategoryKey(procs={}, ops={}, non_ops={GPU})
      category_times->RemapKeysInplace([] (const CategoryKey& old_key) {
        CategoryKey new_key = old_key;
        new_key.procs.clear();
        return new_key;
      });

      if (SHOULD_DEBUG(FEATURE_LOAD_DATA)) {
        std::stringstream ss;
        ss << "CategoryTimes details:";

        std::set<Operation> ops;
        for (const auto& pair : category_times->eo_times) {
          auto const &category_key = pair.first;
          ops.insert(category_key.ops.begin(), category_key.ops.end());
        }
        ss << "\n";
        PrintIndent(ss, 1);
        // op_names = {sample_action, step}
        ss << "op_names = ";
        PrintValue(ss, ops);

        for (const auto& pair : category_times->eo_times) {
          auto const& category_key = pair.first;
          auto const& eo_events = pair.second;

          if (eo_events.KeepNames()) {
            ss << "\n";
            category_key.Print(ss, 1);
            auto const& names = eo_events.UniqueNames();
            ss << "\n";
            PrintIndent(ss, 2);
            ss << "names = ";
            PrintValue(ss, names);
          }
        }

        DBG_LOG("{}", ss.str());
      }

      n_total_events += category_times->TotalEvents();
      all_category_times.emplace_back(std::move(category_times), meta);

      return MyStatus::OK();
    },
    file_types, selector);
    if (timer) {
      auto total_time_sec = timer->TotalTimeSec();
      timer->RecordThroughput("overlap events", n_total_events);
      timer->RecordThroughput("total_sec", total_time_sec);
//      SimpleTimer::MetricValue events_per_sec =
//          static_cast<SimpleTimer::MetricValue>(n_total_events) /
//          static_cast<SimpleTimer::MetricValue>(total_time_sec);
//      timer->RecordThroughput("overlap events/sec", events_per_sec);

    }
    std::list<const CategoryTimes*> all_ctimes;
    for (const auto& tupl : all_category_times) {
      all_ctimes.push_back(std::get<0>(tupl).get());
    }
    CategoryTimes merged = CategoryTimes::MergeAll(all_ctimes);
    if (timer) {
      timer->EndOperation("CategoryTimes::MergeAll()");
    }
    if (FLAGS_debug) {
      std::cout << "Merged CategoryTimes summary:" << std::endl;
      merged.PrintSummary(std::cout, 1);
      std::cout << std::endl;
    }

    PollingUtil polling_util(merged, FLAGS_polling_interval_us, FLAGS_iml_directory);
    auto polling_util_js = polling_util.Compute();
    auto polling_util_js_path = polling_util.JSPath();
    status = WriteJson(polling_util_js_path, polling_util_js);
    if (timer) {
      timer->EndOperation("PollingUtil.Compute()");
    }

    if (status.code() != MyStatus::OK().code()) {
      std::stringstream ss;
      ss << "Failed to write json @ path=" <<  polling_util_js_path << " for --mode=" << FLAGS_mode;
      IF_BAD_STATUS_EXIT(ss.str(), status);
    }

    if (timer) {
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

