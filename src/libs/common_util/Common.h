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
            auto _err_str = CuptiSamples::nvperfGetErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(CuptiSamples::error::INVALID_ARGUMENT, _err_ss.str()); \
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
            auto _err_str = CuptiSamples::cuptiGetDetailedErrorString(_status); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(CuptiSamples::error::INVALID_ARGUMENT, _err_ss.str()); \
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
            auto _err_status = cuGetErrorString(_status, &_err_str); \
            assert(_err_status == CUDA_SUCCESS); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
            auto _my_status = MyStatus(CuptiSamples::error::INVALID_ARGUMENT, _err_ss.str()); \
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
            auto _err_str = CuptiSamples::nvperfGetErrorString(_status); \
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
            auto _err_str = CuptiSamples::cuptiGetDetailedErrorString(_status); \
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
            auto _err_status = cuGetErrorString(_status, &_err_str); \
            assert(_err_status == CUDA_SUCCESS); \
            _err_ss << __FILE__ << ":" << __LINE__ << ": error: function " << #apiFuncCall << " failed with error (" << _status << ") " << _err_str; \
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
            auto _my_status = MyStatus(CuptiSamples::error::INVALID_ARGUMENT, _err_ss.str()); \
            PRINT_AND_DBG_BREAKPOINT("RUNTIME_API_CALL", _my_status); \
            return _my_status; \
        } \
    } while (0)

namespace CuptiSamples {

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

} // namespace CuptiSamples

#endif //CUPTI_SAMPLES_COMMON_H
