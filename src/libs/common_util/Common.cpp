//
// Created by jgleeson on 2020-05-14.
//

#include "Common.h"
#include "my_status.h"
#include "env_var.h"

#include <boost/filesystem.hpp>

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <nvperf_host.h>

#include <vector>
#include <string>
#include <regex>
#include <chrono>

namespace rlscope {

const bool TRACE_CUDA_DEFAULT = false;
bool get_TRACE_CUDA(boost::optional<bool> user_value) {
  return ParseEnvOrDefault("bool", "TRACE_CUDA", user_value, TRACE_CUDA_DEFAULT);
}
const bool TRACE_CUDA = get_TRACE_CUDA(boost::none);

std::vector<std::string> StringSplit(const std::string& s, std::string rgx_str) {
    std::vector<std::string> elems;

    std::regex rgx (rgx_str);
    std::sregex_token_iterator iter(s.begin(), s.end(), rgx, -1);
    std::sregex_token_iterator end;
    while (iter != end)  {
        elems.push_back(*iter);
        ++iter;
    }
    return elems;
}

// https://stackoverflow.com/questions/32188956/get-current-timestamp-in-microseconds-since-epoch
timestamp_us get_timestamp_us() {
    timestamp_us ts = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
    return ts;
}
uint64_t timestamp_as_us(timestamp_us timestamp) {
    return timestamp.time_since_epoch().count();
}

char path_separator()
{
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

std::string os_dirname(const std::string &path) {
    boost::filesystem::path bpath(path);
    auto parent = bpath.parent_path();
    return parent.string();

//  std::string path_copy(path);
//  std::unique_ptr<char> c_path (new char [path.length()+1]);
//  std::strcpy(c_path.get(), path.c_str());
//  auto dir = ::dirname(c_path.get());
//  return std::string(dir);
}

std::string os_basename(const std::string &path) {
    std::string path_copy(path);
    std::unique_ptr<char> c_path(new char[path.length() + 1]);
    std::strcpy(c_path.get(), path.c_str());
    auto dir = ::basename(c_path.get());
    return std::string(dir);
}

void mkdir_p(const std::string &dir, bool exist_ok) {
//  VLOG(1) << "mkdir_p @ dir = " << dir;
    boost::filesystem::path path(dir);
    bool was_created = boost::filesystem::create_directories(path);
    if (!exist_ok) {
        assert(was_created);
    }
}

std::string nvperfGetErrorString(NVPA_Status status) {
    // nvperf_target.h
    switch (status) {
        case NVPA_STATUS_SUCCESS:
            return "NVPA_STATUS_SUCCESS: Success";
        case NVPA_STATUS_ERROR:
            return "NVPA_STATUS_ERROR: Generic error.";
        case NVPA_STATUS_INTERNAL_ERROR:
            return "NVPA_STATUS_INTERNAL_ERROR: Internal error.  Please file a bug!";
        case NVPA_STATUS_NOT_INITIALIZED:
            return "NVPA_STATUS_NOT_INITIALIZED: NVPA_Init() has not been called yet.";
        case NVPA_STATUS_NOT_LOADED:
            return "NVPA_STATUS_NOT_LOADED: The NvPerfAPI DLL/DSO could not be loaded during init.";
        case NVPA_STATUS_FUNCTION_NOT_FOUND:
            return "NVPA_STATUS_FUNCTION_NOT_FOUND: The function was not found in this version of the NvPerfAPI DLL/DSO.";
        case NVPA_STATUS_NOT_SUPPORTED:
            return "NVPA_STATUS_NOT_SUPPORTED: The request is intentionally not supported by NvPerfAPI.";
        case NVPA_STATUS_NOT_IMPLEMENTED:
            return "NVPA_STATUS_NOT_IMPLEMENTED: The request is not implemented by this version of NvPerfAPI.";
        case NVPA_STATUS_INVALID_ARGUMENT:
            return "NVPA_STATUS_INVALID_ARGUMENT: Invalid argument.";
        case NVPA_STATUS_INVALID_METRIC_ID:
            return "NVPA_STATUS_INVALID_METRIC_ID: A MetricId argument does not belong to the specified NVPA_Activity or NVPA_Config.";
        case NVPA_STATUS_DRIVER_NOT_LOADED:
            return "NVPA_STATUS_DRIVER_NOT_LOADED: No driver has been loaded via NVPA_*_LoadDriver().";
        case NVPA_STATUS_OUT_OF_MEMORY:
            return "NVPA_STATUS_OUT_OF_MEMORY: Failed memory allocation.";
        case NVPA_STATUS_INVALID_THREAD_STATE:
            return "NVPA_STATUS_INVALID_THREAD_STATE: The request could not be fulfilled due to the state of the current thread.";
        case NVPA_STATUS_FAILED_CONTEXT_ALLOC:
            return "NVPA_STATUS_FAILED_CONTEXT_ALLOC: Allocation of context object failed.";
        case NVPA_STATUS_UNSUPPORTED_GPU:
            return "NVPA_STATUS_UNSUPPORTED_GPU: The specified GPU is not supported.";
        case NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION:
            return "NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION: The installed NVIDIA driver is too old.";
        case NVPA_STATUS_OBJECT_NOT_REGISTERED:
            return "NVPA_STATUS_OBJECT_NOT_REGISTERED: Graphics object has not been registered via NVPA_Register*().";
        case NVPA_STATUS_INSUFFICIENT_PRIVILEGE:
            return "NVPA_STATUS_INSUFFICIENT_PRIVILEGE: The operation failed due to a security check.";
        case NVPA_STATUS_INVALID_CONTEXT_STATE:
            return "NVPA_STATUS_INVALID_CONTEXT_STATE: The request could not be fulfilled due to the state of the context.";
        case NVPA_STATUS_INVALID_OBJECT_STATE:
            return "NVPA_STATUS_INVALID_OBJECT_STATE: The request could not be fulfilled due to the state of the object.";
        case NVPA_STATUS_RESOURCE_UNAVAILABLE:
            return "NVPA_STATUS_RESOURCE_UNAVAILABLE: The request could not be fulfilled because a system resource is already in use.";
        case NVPA_STATUS_DRIVER_LOADED_TOO_LATE:
            return "NVPA_STATUS_DRIVER_LOADED_TOO_LATE: The NVPA_*_LoadDriver() is called after the context, command queue or device is created.";
        case NVPA_STATUS_INSUFFICIENT_SPACE:
            return "NVPA_STATUS_INSUFFICIENT_SPACE: The provided buffer is not large enough.";
        case NVPA_STATUS_OBJECT_MISMATCH:
            return "NVPA_STATUS_OBJECT_MISMATCH: The API object passed to NVPA_[API]_BeginPass/NVPA_[API]_EndPass and NVPA_[API]_PushRange/NVPA_[API]_PopRange does not match with the NVPA_[API]_BeginSession.";
        default:
            assert(false);
            return "";
//        NVPA_STATUS__COUNT
    }
}

// I think the CUPTI function for getting error strings is not up-to-date with the
// new CUDA profiling API.
std::string cuptiGetDetailedErrorString(CUptiResult status) {
    // cupti_result.h
    switch (status) {
        case CUPTI_SUCCESS:
            return "CUPTI_SUCCESS: No error.";
        case CUPTI_ERROR_INVALID_PARAMETER:
            return "CUPTI_ERROR_INVALID_PARAMETER: One or more of the parameters is invalid.";
        case CUPTI_ERROR_INVALID_DEVICE:
            return "CUPTI_ERROR_INVALID_DEVICE: The device does not correspond to a valid CUDA device.";
        case CUPTI_ERROR_INVALID_CONTEXT:
            return "CUPTI_ERROR_INVALID_CONTEXT: The context is NULL or not valid.";
        case CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID:
            return "CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID: The event domain id is invalid.";
        case CUPTI_ERROR_INVALID_EVENT_ID:
            return "CUPTI_ERROR_INVALID_EVENT_ID: The event id is invalid.";
        case CUPTI_ERROR_INVALID_EVENT_NAME:
            return "CUPTI_ERROR_INVALID_EVENT_NAME: The event name is invalid.";
        case CUPTI_ERROR_INVALID_OPERATION:
            return "CUPTI_ERROR_INVALID_OPERATION: The current operation cannot be performed due to dependency on other factors.";
        case CUPTI_ERROR_OUT_OF_MEMORY:
            return "CUPTI_ERROR_OUT_OF_MEMORY: Unable to allocate enough memory to perform the requested operation.";
        case CUPTI_ERROR_HARDWARE:
            return "CUPTI_ERROR_HARDWARE: An error occurred on the performance monitoring hardware.";
        case CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT:
            return "CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT: The output buffer size is not sufficient to return all requested data.";
        case CUPTI_ERROR_API_NOT_IMPLEMENTED:
            return "CUPTI_ERROR_API_NOT_IMPLEMENTED: API is not implemented.";
        case CUPTI_ERROR_MAX_LIMIT_REACHED:
            return "CUPTI_ERROR_MAX_LIMIT_REACHED: The maximum limit is reached.";
        case CUPTI_ERROR_NOT_READY:
            return "CUPTI_ERROR_NOT_READY: The object is not yet ready to perform the requested operation.";
        case CUPTI_ERROR_NOT_COMPATIBLE:
            return "CUPTI_ERROR_NOT_COMPATIBLE: The current operation is not compatible with the current state of the object";
        case CUPTI_ERROR_NOT_INITIALIZED:
            return "CUPTI_ERROR_NOT_INITIALIZED: CUPTI is unable to initialize its connection to the CUDA driver.";
        case CUPTI_ERROR_INVALID_METRIC_ID:
            return "CUPTI_ERROR_INVALID_METRIC_ID: The metric id is invalid.";
        case CUPTI_ERROR_INVALID_METRIC_NAME:
            return "CUPTI_ERROR_INVALID_METRIC_NAME: The metric name is invalid.";
        case CUPTI_ERROR_QUEUE_EMPTY:
            return "CUPTI_ERROR_QUEUE_EMPTY: The queue is empty.";
        case CUPTI_ERROR_INVALID_HANDLE:
            return "CUPTI_ERROR_INVALID_HANDLE: Invalid handle (internal?).";
        case CUPTI_ERROR_INVALID_STREAM:
            return "CUPTI_ERROR_INVALID_STREAM: Invalid stream.";
        case CUPTI_ERROR_INVALID_KIND:
            return "CUPTI_ERROR_INVALID_KIND: Invalid kind.";
        case CUPTI_ERROR_INVALID_EVENT_VALUE:
            return "CUPTI_ERROR_INVALID_EVENT_VALUE: Invalid event value.";
        case CUPTI_ERROR_DISABLED:
            return "CUPTI_ERROR_DISABLED: CUPTI is disabled due to conflicts with other enabled profilers";
        case CUPTI_ERROR_INVALID_MODULE:
            return "CUPTI_ERROR_INVALID_MODULE: Invalid module.";
        case CUPTI_ERROR_INVALID_METRIC_VALUE:
            return "CUPTI_ERROR_INVALID_METRIC_VALUE: Invalid metric value.";
        case CUPTI_ERROR_HARDWARE_BUSY:
            return "CUPTI_ERROR_HARDWARE_BUSY: The performance monitoring hardware is in use by other client.";
        case CUPTI_ERROR_NOT_SUPPORTED:
            return "CUPTI_ERROR_NOT_SUPPORTED: The attempted operation is not supported on the current system or device.";
        case CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED:
            return "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED: Unified memory profiling is not supported on the system. Potential reason could be unsupported OS or architecture.";
        case CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE:
            return "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE: Unified memory profiling is not supported on the device";
        case CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES:
            return "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES: Unified memory profiling is not supported on a multi-GPU configuration without P2P support between any pair of devices";
        case CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS:
            return "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS: Unified memory profiling is not supported under the Multi-Process Service (MPS) environment. CUDA 7.5 removes this restriction.";
        case CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED:
            return "CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED: In CUDA 9.0, devices with compute capability 7.0 don't support CDP tracing";
        case CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED:
            return "CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED: Profiling on virtualized GPU is not supported.";
        case CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE:
            return "CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE: Profiling results might be incorrect for CUDA applications compiled with nvcc version older than 9.0 for devices with compute capability 6.0 and 6.1. Profiling session will continue and CUPTI will notify it using this error code. User is advised to recompile the application code with nvcc version 9.0 or later. Ignore this warning if code is already compiled with the recommended nvcc version.";
        case CUPTI_ERROR_INSUFFICIENT_PRIVILEGES:
            return "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES: User doesn't have sufficient privileges which are required to start the profiling session.  One possible reason for this may be that the NVIDIA driver or your system administrator may have restricted access to the NVIDIA GPU performance counters.  To learn how to resolve this issue and find more information, please visit https://developer.nvidia.com/CUPTI_ERROR_INSUFFICIENT_PRIVILEGES";
        case CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED:
            return "CUPTI_ERROR_OLD_PROFILER_API_INITIALIZED: Old profiling api's are not supported with new profiling api's";
        case CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE:
            return "CUPTI_ERROR_OPENACC_UNDEFINED_ROUTINE: Missing definition of the OpenACC API routine in the linked OpenACC library.  One possible reason is that OpenACC library is linked statically in the user application, which might not have the definition of all the OpenACC API routines needed for the OpenACC profiling, as compiler might ignore definitions for the functions not used in the application. This issue can be mitigated by linking the OpenACC library dynamically.";
        case CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED:
            return "CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED: Legacy CUPTI Profiling is not supported on devices with Compute Capability 7.5 or higher (Turing+). Using this error to specify this case and differentiate it from other errors.";
        case CUPTI_ERROR_UNKNOWN:
            return "CUPTI_ERROR_UNKNOWN: An unknown internal error has occurred.";
        case CUPTI_ERROR_FORCE_INT:
            return "CUPTI_ERROR_FORCE_INT: An unknown internal error has occurred.";

        default:
            assert(false);
            return "";
            //        NVPA_STATUS__COUNT
    }
}

using KeepFunc = std::function<bool(boost::filesystem::path)>;
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

// FROM TENSORFLOW
void* Malloc(size_t size) { return malloc(size); }

void* Realloc(void* ptr, size_t size) { return realloc(ptr, size); }

void Free(void* ptr) { free(ptr); }

void* AlignedMalloc(size_t size, int minimum_alignment) {
#if defined(__ANDROID__)
  return memalign(minimum_alignment, size);
#else  // !defined(__ANDROID__)
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return Malloc(size);
  int err = posix_memalign(&ptr, minimum_alignment, size);
  if (err != 0) {
    return nullptr;
  } else {
    return ptr;
  }
#endif
}

void AlignedFree(void* aligned_memory) { Free(aligned_memory); }


} // namespace rlscope
