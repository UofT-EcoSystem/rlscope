//
// Created by jgleeson on 2020-05-14.
//
#ifndef CUPTI_SAMPLES_COMMON_H
#define CUPTI_SAMPLES_COMMON_H

#include <vector>
#include <string>
#include <regex>
#include <chrono>

#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <nvperf_host.h>

#include "my_status.h"
#include "debug_flags.h"

#include <boost/filesystem.hpp>

#define FLOAT_MICROSECONDS_IN_SEC ((float)1e6)

#define PRINT_AND_DBG_BREAKPOINT(name, status) \
    std::cerr << "ERROR: " << status << std::endl; \
    DBG_BREAKPOINT(name);

#ifdef SHOULD_PRINT_CUDA_API_CALLS

//#define DEBUG_PRINT_API_CALL(apiFuncCall)
//    do {
//        std::cerr << "[DBG] " << __FILE__ << ":" << __LINE__ << ": " << #apiFuncCall << std::endl;
//    } while(0)

#define DEBUG_PRINT_API_CALL(apiFuncCall) \
    do { \
        DBG_LOG("[TRACE] {}", #apiFuncCall); \
    } while(0)

#else

#define DEBUG_PRINT_API_CALL(apiFuncCall)

#endif // SHOULD_PRINT_CUDA_API_CALLS

#define NVPW_API_CALL_MAYBE_STATUS(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        NVPA_Status _status = apiFuncCall; \
        if (_status != NVPA_STATUS_SUCCESS) { \
            std::stringstream _err_ss; \
            auto _err_str = rlscope::nvperfGetErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(rlscope::error::INVALID_ARGUMENT, _err_ss.str()); \
            PRINT_AND_DBG_BREAKPOINT("NVPW_API_CALL", _my_status); \
            return _my_status; \
        } \
    } while (0)

//const char* _err_str = nullptr;
//auto _err_status = cuptiGetResultString(_status, &_err_str);
#define CUPTI_API_CALL_MAYBE_STATUS(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        CUptiResult _status = apiFuncCall; \
        if (_status != CUPTI_SUCCESS) { \
            std::stringstream _err_ss; \
            auto _err_str = rlscope::cuptiGetDetailedErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(rlscope::error::INVALID_ARGUMENT, _err_ss.str()); \
            PRINT_AND_DBG_BREAKPOINT("CUPTI_API_CALL", _my_status); \
            return _my_status; \
        } \
    } while (0)

#define DRIVER_API_CALL_MAYBE_STATUS(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        CUresult _status = apiFuncCall; \
        if (_status != CUDA_SUCCESS) { \
            std::stringstream _err_ss; \
            const char* _err_str = nullptr; \
            const char* _err_name = nullptr; \
            auto _err_status = cuGetErrorString(_status, &_err_str); \
            assert(_err_status == CUDA_SUCCESS); \
            _err_status = cuGetErrorName(_status, &_err_name); \
            assert(_err_status == CUDA_SUCCESS); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with " << _err_name << " (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(rlscope::error::INVALID_ARGUMENT, _err_ss.str()); \
            PRINT_AND_DBG_BREAKPOINT("DRIVER_API_CALL", _my_status); \
            return _my_status; \
        } \
    } while (0)

#define NVPW_API_CALL_MAYBE_EXIT(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        NVPA_Status _status = apiFuncCall; \
        if (_status != NVPA_STATUS_SUCCESS) { \
            std::stringstream _err_ss; \
            auto _err_str = rlscope::nvperfGetErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            std::cerr << "ERROR: " << _err_ss.str() << std::endl; \
            exit(-1); \
        } \
    } while (0)

#define CUPTI_API_CALL_MAYBE_EXIT(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        CUptiResult _status = apiFuncCall; \
        if (_status != CUPTI_SUCCESS) { \
            std::stringstream _err_ss; \
            auto _err_str = rlscope::cuptiGetDetailedErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            std::cerr << "ERROR: " << _err_ss.str() << std::endl; \
            DBG_BREAKPOINT("CUPTI_API_CALL"); \
            exit(-1); \
        } \
    } while (0)

#define DRIVER_API_CALL_MAYBE_EXIT(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        CUresult _status = apiFuncCall; \
        if (_status != CUDA_SUCCESS) { \
            std::stringstream _err_ss; \
            const char* _err_str = nullptr; \
            const char* _err_name = nullptr; \
            auto _err_status = cuGetErrorString(_status, &_err_str); \
            assert(_err_status == CUDA_SUCCESS); \
            _err_status = cuGetErrorName(_status, &_err_name); \
            assert(_err_status == CUDA_SUCCESS); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with " << _err_name << " (" << _status << ") " << _err_str; \
            std::cerr << "ERROR: " << _err_ss.str() << std::endl; \
            DBG_BREAKPOINT("DRIVER_API_CALL"); \
            exit(-1); \
        } \
    } while (0)

#define RUNTIME_API_CALL_MAYBE_EXIT(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        cudaError_t _status = apiFuncCall; \
        if (_status != cudaSuccess) { \
            std::stringstream _err_ss; \
            auto _err_str = cudaGetErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            std::cerr << "ERROR: " << _err_ss.str() << std::endl; \
            DBG_BREAKPOINT("RUNTIME_API_CALL"); \
            exit(-1); \
        } \
    } while (0)

#define RUNTIME_API_CALL_MAYBE_STATUS(apiFuncCall) \
    do { \
        DEBUG_PRINT_API_CALL(apiFuncCall); \
        cudaError_t _status = apiFuncCall; \
        if (_status != cudaSuccess) { \
            std::stringstream _err_ss; \
            auto _err_str = cudaGetErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(rlscope::error::INVALID_ARGUMENT, _err_ss.str()); \
            PRINT_AND_DBG_BREAKPOINT("RUNTIME_API_CALL", _my_status); \
            return _my_status; \
        } \
    } while (0)

namespace rlscope {

std::vector<std::string> StringSplit(const std::string& s, std::string rgx_str);

// using timestamp_us = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>;
typedef std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> timestamp_us ;
timestamp_us get_timestamp_us();
uint64_t timestamp_as_us(timestamp_us timestamp);

std::string os_dirname(const std::string &path);
std::string os_basename(const std::string &path);
void mkdir_p(const std::string &dir, bool exist_ok = true);

std::string nvperfGetErrorString(NVPA_Status status);
std::string cuptiGetDetailedErrorString(CUptiResult status);

using KeepFileFunc = std::function<bool(boost::filesystem::path)>;
MyStatus RecursiveFindFiles(std::list<std::string>* paths, const std::string& root, KeepFileFunc func);

char path_separator();

// FROM TENSORFLOW
// Aligned allocation/deallocation. `minimum_alignment` must be a power of 2
// and a multiple of sizeof(void*).
void* AlignedMalloc(size_t size, int minimum_alignment);
void AlignedFree(void* aligned_memory);


template <typename T1>
std::string StrCat(const T1& t1) {
  std::stringstream ss;
  ss << t1;
  return ss.str();
}
template <typename T1, typename T2>
std::string StrCat(const T1& t1, const T2& t2) {
  std::stringstream ss;
  ss << t1;
  ss << t2;
  return ss.str();
}
template <typename T1, typename T2, typename T3>
std::string StrCat(const T1& t1, const T2& t2, const T3& t3) {
  std::stringstream ss;
  ss << t1;
  ss << t2;
  ss << t3;
  return ss.str();
}
template <typename T1, typename T2, typename T3, typename T4>
std::string StrCat(const T1& t1, const T2& t2, const T3& t3, const T4& t4) {
  std::stringstream ss;
  ss << t1;
  ss << t2;
  ss << t3;
  ss << t4;
  return ss.str();
}
template <typename T1, typename T2, typename T3, typename T4, typename T5>
std::string StrCat(const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5) {
  std::stringstream ss;
  ss << t1;
  ss << t2;
  ss << t3;
  ss << t4;
  ss << t5;
  return ss.str();
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
std::string StrCat(const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6) {
  std::stringstream ss;
  ss << t1;
  ss << t2;
  ss << t3;
  ss << t4;
  ss << t5;
  ss << t6;
  return ss.str();
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
std::string StrCat(const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7) {
  std::stringstream ss;
  ss << t1;
  ss << t2;
  ss << t3;
  ss << t4;
  ss << t5;
  ss << t6;
  ss << t7;
  return ss.str();
}

} // namespace rlscope

#endif //CUPTI_SAMPLES_COMMON_H
