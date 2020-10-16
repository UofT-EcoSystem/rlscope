//
// Created by jagle on 8/16/2019.
//

#ifndef IML_CUPTI_LOGGING_H
#define IML_CUPTI_LOGGING_H


#include <cupti_target.h>
#include <cupti.h>
#include <cuda.h>

#include <ostream>
#include <map>

#define CHECK_CU_ERROR(out, err, cufunc) \
  if (err != CUDA_SUCCESS) { \
    out << "error " << err << " for CUDA Driver API function " << cufunc; \
  }

#define CHECK_CUPTI_ERROR(out, err, cuptifunc) \
  if (err != CUPTI_SUCCESS) { \
    const char* errstr = nullptr; \
    auto errstrRet = cuptiGetResultString(err, &errstr); \
    DCHECK(errstrRet == CUPTI_SUCCESS) << "Failed to obtain error std::string for CUPTI error code = " << err; \
    out << "error " << err << " for CUPTI API function " << cuptifunc <<  ": " << errstr; \
  }

#define MAYBE_RETURN(status) \
  if (status.code() != MyStatus::OK().code()) { \
    return status.code(); \
  }

#define MAYBE_EXIT(status) \
  if (status.code() != MyStatus::OK().code()) { \
    exit(status.code()); \
  }

#define MAYBE_LOG_ERROR(out, func, status) \
  if (status.code() != MyStatus::OK().code()) { \
    out << "rls-prof C++ API '" << func << "' failed with: " << status; \
  }

namespace rlscope {

const char* runtime_cbid_to_string(CUpti_CallbackId cbid);
const char* driver_cbid_to_string(CUpti_CallbackId cbid);

void printActivity(const CUpti_Activity *record);

//inline static void MAYBE_LOG_ERROR(std::ostream&& out, const char* func, const MyStatus& status) {
//  if (status.code() != MyStatus::OK().code()) {
//    out << "rls-prof C++ API '" << func << "' failed with: " << status;
//  }
//}


}

#endif //IML_CUPTI_LOGGING_H
