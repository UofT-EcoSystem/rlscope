//
// Created by jagle on 8/16/2019.
//

#ifndef IML_CUPTI_LOGGING_H
#define IML_CUPTI_LOGGING_H

#include <cupti.h>
#include <cuda.h>

#include <ostream>

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

namespace tensorflow {

std::ostream& PrintIndent(std::ostream& out, int indent);

const char* runtime_cbid_to_string(CUpti_CallbackId cbid);
const char* driver_cbid_to_string(CUpti_CallbackId cbid);

}

#endif //IML_CUPTI_LOGGING_H
