//
// Created by jagle on 8/16/2019.
//

#ifndef IML_CUPTI_LOGGING_H
#define IML_CUPTI_LOGGING_H

#include "tensorflow/core/lib/core/status.h"

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
    DCHECK(errstrRet == CUPTI_SUCCESS) << "Failed to obtain error string for CUPTI error code = " << err; \
    out << "error " << err << " for CUPTI API function " << cuptifunc <<  ": " << errstr; \
  }

#define MAYBE_RETURN(status) \
  if (status.code() != Status::OK().code()) { \
    return status.code(); \
  }

#define MAYBE_EXIT(status) \
  if (status.code() != Status::OK().code()) { \
    exit(status.code()); \
  }

#define MAYBE_LOG_ERROR(out, func, status) \
  if (status.code() != Status::OK().code()) { \
    out << "iml-prof C++ API '" << func << "' failed with: " << status; \
  }

namespace tensorflow {

std::ostream& PrintIndent(std::ostream& out, int indent);

const char* runtime_cbid_to_string(CUpti_CallbackId cbid);
const char* driver_cbid_to_string(CUpti_CallbackId cbid);

void printActivity(const CUpti_Activity *record);

//inline static void MAYBE_LOG_ERROR(std::ostream&& out, const char* func, const Status& status) {
//  if (status.code() != Status::OK().code()) {
//    out << "iml-prof C++ API '" << func << "' failed with: " << status;
//  }
//}

class SimpleTimer {
public:
  using TimeUsec = int64;
  std::string _name;
  std::map<std::string, TimeUsec> _op_duration_usec;
  TimeUsec _last_time_usec;
  SimpleTimer(const std::string& name);

  void ResetStartTime();
  void EndOperation(const std::string& operation);
  void Print(std::ostream& out, int indent);

};

}

#endif //IML_CUPTI_LOGGING_H
