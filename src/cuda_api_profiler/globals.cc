//
// Created by jagle on 8/19/2019.
//

#include <string>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <signal.h>
#include <regex>

#include "common/util.h"

#include <boost/process.hpp>
#include <boost/algorithm/string.hpp>
//#include <boost/algorithm/string/join.hpp>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

//#include <spdlog/spdlog.h>

#include "cuda_api_profiler/cupti_logging.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/device_tracer.h"

#include "cuda_api_profiler/cuda_ld_preload.h"
#include "cuda_api_profiler/globals.h"
#include "cuda_api_profiler/get_env_var.h"

//#include "cuda_api_profiler/generic_logging.h"
//#include "cuda_api_profiler/debug_flags.h"
#include "common/my_status.h"
#include "common/json.h"

#define MAYBE_RETURN_ERROR(status) \
  if (status.code() != Status::OK().code()) { \
    VLOG(INFO) << "iml-prof C++ API @ " << __func__ << " failed with " << status.ToString(); \
    return; \
  }

namespace bp = boost::process;

namespace tensorflow {

Globals globals;

Globals::Globals() {

  Status status = Status::OK();

#ifdef WITH_CUDA_LD_PRELOAD
  VLOG(1) << "dlopen(\"libcudart.so\")";
  cuda_library = GetCudaLibrary();
  VLOG(1) << "dlopen(\"libcudart.so\"): success!";
#endif

//  std::ifstream cmdline_stream("/proc/self/cmdline");
//  std::string cmdline((std::istreambuf_iterator<char>(cmdline_stream)),
//                      std::istreambuf_iterator<char>());
//
//  VLOG(1) << "Initialize globals\n"
//          << "  CMD = " << cmdline;

//  VLOG(1) << "SKIP creating device_tracer";
  device_tracer = tensorflow::CreateDeviceTracer();
  auto IML_TRACE_AT_START = getenv("IML_TRACE_AT_START");
  VLOG(1) << "IML_TRACE_AT_START = " << IML_TRACE_AT_START;
  if (device_tracer && env_is_on("IML_TRACE_AT_START", false, true)) {
    VLOG(1) << "TraceAtStart";
    this->TraceAtStart();
  }

}

std::string Globals::IMLConfigPath() const {
  auto iml_config_path = boost::filesystem::path(_directory) /
              "process" / _process_name /
              "phase" / _phase_name /
              "iml_config.json";
  return iml_config_path.string();
}

bool is_machine_util_file(const boost::filesystem::path& path) {
  std::regex machine_util_regex(R"(^machine_util\..*.proto)");
  std::smatch match;
  return std::regex_search(path.filename().string(), match, machine_util_regex);
}

bool is_cuda_device_events_file(const boost::filesystem::path& path) {
  std::regex machine_util_regex(R"(^cuda_device_events\..*.proto)");
  std::smatch match;
  return std::regex_search(path.filename().string(), match, machine_util_regex);
}

bool is_cuda_api_stats_file(const boost::filesystem::path& path) {
  std::regex machine_util_regex(R"(^cuda_api_stats\..*.proto)");
  std::smatch match;
  return std::regex_search(path.filename().string(), match, machine_util_regex);
}

bool is_trace_file(const boost::filesystem::path& path) {
  return is_machine_util_file(path) || is_cuda_device_events_file(path) || is_cuda_api_stats_file(path);
}

void Globals::DeleteOldTraceFiles() const {
  boost::filesystem::path root_path(_directory);
  // https://rosettacode.org/wiki/Walk_a_directory/Recursively#C.2B.2B
  if (!boost::filesystem::is_directory(root_path)) {
    return;
  }
  std::list<boost::filesystem::path> to_remove;
  for (boost::filesystem::recursive_directory_iterator iter(root_path), end;
       iter != end;
       ++iter)
  {
    auto path = iter->path();
    if (is_trace_file(path)) {
      to_remove.push_back(path);
//      boost::filesystem::remove(path);
    }
  }
  for (const auto& path : to_remove) {
    LOG(INFO) << "RM trace-file @ " << path;
    boost::filesystem::remove(path);
  }
}
void Globals::DumpIMLConfig() const {
  MyStatus status = MyStatus::OK();
  auto cur_env = boost::this_process::environment();
  json js;
  std::map<std::string, std::string> env_dict;
  for (const auto& pair : cur_env) {
    env_dict[pair.get_name()] = pair.to_string();
//    env_dict[pair.]
  }
  js["env"] = env_dict;
  js["process_name"] = _process_name;
  js["machine_name"] = _machine_name;
  js["phase"] = _phase_name;
  js["directory"] = _directory;
  auto path = IMLConfigPath();
  boost::filesystem::path bpath(path);
  boost::filesystem::create_directories(bpath.parent_path());
  status = WriteJson(path, js);
  assert(status.code() == MyStatus::OK().code());
}

boost::process::environment Globals::PatchedEnviron() {
  using namespace boost::process;
  environment env = boost::this_process::environment();
  if (env.find("LD_PRELOAD") == env.end()) {
    return env;
  }
  std::string LD_PRELOAD = env["LD_PRELOAD"].to_string();
//  auto LD_PRELOAD_cstr = getenv("LD_PRELOAD");
//  if (LD_PRELOAD_cstr == nullptr) {
//    return;
//  }
//  std::string LD_PRELOAD = LD_PRELOAD_cstr;
  std::vector<std::string> paths;
  boost::split(paths, LD_PRELOAD, [] (char c) {
    return c == ':';
  });
  auto is_sample_lib = [] (const std::string& path) -> bool {
    boost::filesystem::path bpath = path;
    auto basename = bpath.filename().string();
    return (basename == "librlscope.so");
  };
  std::vector<std::string> keep_paths;
  keep_paths.reserve(paths.size());
  for (const auto& path : paths) {
    if (!is_sample_lib(path)) {
      keep_paths.push_back(path);
    }
  }
  auto new_LD_PRELOAD = boost::algorithm::join(keep_paths, ":");
//  setenv("LD_PRELOAD", new_LD_PRELOAD.c_str(), 1);
  env["LD_PRELOAD"] = new_LD_PRELOAD;
  return env;
}

bool Globals::env_is_yes(const std::string& var) const {
  return !env_is_no(var);
}

bool Globals::env_is_no(const std::string& var) const {
  auto cur_env = boost::this_process::environment();
  if (cur_env.find(var) != cur_env.end()) {
    auto value = cur_env[var].to_string();
    return (value == "") || (value == "no") || (value == "0") || (value == "false");
  }
  return true;
}

void Globals::StartUtilSampler() {
  auto cur_env = boost::this_process::environment();
  if (cur_env.find("IML_UTIL_SAMPLER_PID") != cur_env.end()) {
    // Utilization sampler is already running; don't start it again.
    return;
  }
  bp::environment env = PatchedEnviron();
  bp::ipstream pipe_stream;
  auto util_sampler_exe = bp::search_path("iml-util-sampler");
  if (util_sampler_exe == "") {
    LOG(INFO) << "Couldn't find path to iml-util-sampler on $PATH; have you installed the iml python package and activated your environment?";
    exit(EXIT_FAILURE);
  }
  auto pid = getpid();
  auto pid_str = std::to_string(pid);
//    child c("gcc --version", env, std_out > pipe_stream);
  std::vector<std::string> cmd_list{
      util_sampler_exe.string(),
      "--iml-root-pid", pid_str,
      "--iml-directory", _directory,
  };
  if (env_is_yes("IML_DEBUG")) {
    cmd_list.push_back("--iml-debug");
  }
  auto cmd_str = boost::algorithm::join(cmd_list, " ");

  _util_sampler = bp::child(
      cmd_list,
//      util_sampler_exe,
//      "--iml-root-pid", pid_str,
//      "--iml-directory", _directory,
      env);
  LOG(INFO) << "Start GPU utilization sampler " << util_sampler_exe << " @ pid=" << pid;
  LOG(INFO) << "  $ " << cmd_str;

  auto util_sampler_pid = _util_sampler.id();
  int ret = setenv("IML_UTIL_SAMPLER_PID", std::to_string(util_sampler_pid).c_str(), 1);
  assert(ret == 0);

//  std::string line;
//
//  size_t i = 0;
//  while (pipe_stream && std::getline(pipe_stream, line) && !line.empty()) {
//    std::cerr << "LINE[" << i << "]: " << line << std::endl;
//    i += 1;
//  }
//
//  c.wait();
}

void Globals::TraceAtStart() {
  Status status = Status::OK();

  // Should we trace sub-processes?
  // The problem here is that we want to run a sub-process during library loading.
  // But that means the sub-process will also load the library, and launch a sub-process... etc
//  PatchEnviron();

  auto IML_DIRECTORY = getenv("IML_DIRECTORY");
  if (IML_DIRECTORY == nullptr || strcmp("", IML_DIRECTORY) == 0) {
    LOG(INFO) << "ERROR: env variable IML_DIRECTORY must be set to a directory for storing trace-files.";
    exit(EXIT_FAILURE);
  }
  VLOG(1) << "Check: IML_DIRECTORY";

  auto IML_PROCESS_NAME = getenv("IML_PROCESS_NAME");
  if (IML_PROCESS_NAME == nullptr || strcmp("", IML_PROCESS_NAME) == 0) {
    LOG(INFO) << "ERROR: env variable IML_PROCESS_NAME must be set to the process name.";
    exit(EXIT_FAILURE);
  }
  LOG(INFO) << "Check: IML_PROCESS_NAME";

  auto IML_PHASE_NAME = getenv("IML_PHASE_NAME");
  if (IML_PHASE_NAME == nullptr || strcmp("", IML_PHASE_NAME) == 0) {
    LOG(INFO) << "ERROR: env variable IML_PHASE_NAME must be set to the process name.";
    exit(EXIT_FAILURE);
  }
  LOG(INFO) << "Check: IML_PHASE_NAME";

  char hostname[256];
  int ret = gethostname(hostname, 256);
  assert(ret == 0);

  _directory = IML_DIRECTORY;
  _process_name = IML_PROCESS_NAME;
  _machine_name = hostname;
  _phase_name = IML_PHASE_NAME;

  CheckAvailGpus();

  DeleteOldTraceFiles();

//  LOG(INFO) << "BLAH: hostname = " << hostname;

  status = device_tracer->SetMetadata(
      /*directory*/IML_DIRECTORY,
      /*process_name*/IML_PROCESS_NAME,
      /*machine_name*/hostname,
      /*phase_name*/IML_PHASE_NAME);
//  MAYBE_RETURN_ERROR(status);
  MAYBE_LOG_ERROR(LOG(INFO), __func__, status);
  MAYBE_EXIT(status);

  mkdir_p(_directory);

  DumpIMLConfig();

  StartUtilSampler();

  LOG(INFO) << "Starting tracing at program start (export IML_TRACE_AT_START=yes)";
  status = device_tracer->Start();
//  MAYBE_RETURN_ERROR(status);
  MAYBE_LOG_ERROR(LOG(FATAL), __func__, status);
  MAYBE_EXIT(status);
}

void Globals::CheckAvailGpus() const {
  auto report_error_and_exit = [] () {
    std::stringstream ss;
    ss << "IML ERROR: you must set CUDA_VISIBLE_DEVICES to one of the available GPU's in the system; currently doesn't support multi-GPU use-cases; for example:\n";
    ss << "  $ export CUDA_VISIBLE_DEVICES=\"0\"";
    LOG(INFO) << ss.str();
    exit(EXIT_FAILURE);
  };
  auto cur_env = boost::this_process::environment();
  if (cur_env.find("CUDA_VISIBLE_DEVICES") == cur_env.end()) {
    report_error_and_exit();
  }
  auto CUDA_VISIBLE_DEVICES_str = cur_env["CUDA_VISIBLE_DEVICES"].to_string();
  std::vector<std::string> device_id_strs;
  boost::split(device_id_strs, CUDA_VISIBLE_DEVICES_str, [] (char c) {
    return c == ',';
  });
  std::set<int> device_ids;
  std::transform(device_id_strs.begin(), device_id_strs.end(), std::inserter(device_ids, device_ids.end()), [] (const std::string& device_id_str) {
    return std::stoi(device_id_str);
  });
  if (device_ids.size() != 1) {
    report_error_and_exit();
  }
}

Globals::~Globals() {
  // NOTE: some programs will close stdout/stderr BEFORE this gets called.
  // This will cause log message to be LOST.
  // HOWEVER, the destructor will still execute.
  // You can confirm this behaviour by creating a file.
  //
  // https://stackoverflow.com/questions/23850624/ld-preload-does-not-work-as-expected
  //
//  std::ofstream myfile;
//  myfile.open("globals.destructor.txt");
//  myfile << "Writing this to a file.\n";
//  myfile.close();

  if (device_tracer && device_tracer->IsEnabled()) {
    VLOG(FATAL) << "Looks like DeviceTracer was still running... "
                << "please call sample_cuda_api.disable_tracing() in python BEFORE exiting to avoid stranger behavior in C++ destructors during library unload.";
  }

//#define MAYBE_RETURN_ERROR(status)
//  if (status.code() != Status::OK().code()) {
//    DBG_LOG("iml-prof C++ API @ {} failed with {}", __func__, status.ToString());
//    return;
//  }

  if (_util_sampler.valid() && _util_sampler.running()) {
    LOG(INFO) << "Terminate GPU utilization sampler @ pid = " << _util_sampler.id();
    // NOTE: this send SIGKILL to process which doesn't allow it to cleanup and dump trace-files...
    // use SIGTERM instead.
    // _util_sampler.terminate();
    int ret = kill(_util_sampler.id(), SIGTERM);
    if (ret != 0) {
      int err = errno;
      if (err == ESRCH) {
        LOG(INFO) << "Failed to terminate GPU utilization sampler since its pid didn't exist: " << strerror(err);
      } else {
        LOG(INFO) << "Failed to terminate GPU utilization sampler (not sure why): " << strerror(err);
        exit(EXIT_FAILURE);
      }
    }
    _util_sampler.wait();
  }

  auto IML_TRACE_AT_START = getenv("IML_TRACE_AT_START");
//  VLOG(1) << "IML_TRACE_AT_START = " << IML_TRACE_AT_START;
  if (device_tracer && env_is_on("IML_TRACE_AT_START", false, true)) {
    Status status;
    status = device_tracer->Print();
    MAYBE_RETURN_ERROR(status);
    status = device_tracer->AsyncDump();
    MAYBE_RETURN_ERROR(status);
    status = device_tracer->AwaitDump();
    MAYBE_RETURN_ERROR(status);
  }

  // Dump CUDA API call counts and total CUDA API time to a protobuf file.
//    device_tracer->Collect();
}


}
