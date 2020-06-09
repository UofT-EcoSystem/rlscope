/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//#include "tensorflow/core/lib/core/status.h"
#include "my_status.h"
#include <stdio.h>
#include <assert.h>

namespace CuptiSamples {

MyStatus::MyStatus(CuptiSamples::error::Code code, const std::string& msg) {
  assert(code != CuptiSamples::error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = msg;
}

void MyStatus::Update(const MyStatus& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void MyStatus::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string& MyStatus::empty_string() {
//  static std::string* empty = new string;
//  return *empty;
  static std::string empty = std::string("");
  return empty;
}

std::string MyStatus::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    char tmp[30];
    const char* type;
    switch (code()) {
      case CuptiSamples::error::CANCELLED:
        type = "Cancelled";
        break;
      case CuptiSamples::error::UNKNOWN:
        type = "Unknown";
        break;
      case CuptiSamples::error::INVALID_ARGUMENT:
        type = "Invalid argument";
        break;
      case CuptiSamples::error::DEADLINE_EXCEEDED:
        type = "Deadline exceeded";
        break;
      case CuptiSamples::error::NOT_FOUND:
        type = "Not found";
        break;
      case CuptiSamples::error::ALREADY_EXISTS:
        type = "Already exists";
        break;
      case CuptiSamples::error::PERMISSION_DENIED:
        type = "Permission denied";
        break;
      case CuptiSamples::error::UNAUTHENTICATED:
        type = "Unauthenticated";
        break;
      case CuptiSamples::error::RESOURCE_EXHAUSTED:
        type = "Resource exhausted";
        break;
      case CuptiSamples::error::FAILED_PRECONDITION:
        type = "Failed precondition";
        break;
      case CuptiSamples::error::ABORTED:
        type = "Aborted";
        break;
      case CuptiSamples::error::OUT_OF_RANGE:
        type = "Out of range";
        break;
      case CuptiSamples::error::UNIMPLEMENTED:
        type = "Unimplemented";
        break;
      case CuptiSamples::error::INTERNAL:
        type = "Internal";
        break;
      case CuptiSamples::error::UNAVAILABLE:
        type = "Unavailable";
        break;
      case CuptiSamples::error::DATA_LOSS:
        type = "Data loss";
        break;
      default:
        snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                 static_cast<int>(code()));
        type = tmp;
        break;
    }
    std::string result(type);
    result += ": ";
    result += state_->msg;
    return result;
  }
}

void MyStatus::IgnoreError() const {
  // no-op
}

std::ostream& operator<<(std::ostream& os, const MyStatus& x) {
  os << x.ToString();
  return os;
}

//std::string* TfCheckOpHelperOutOfLine(const ::CuptiSamples::MyStatus& v,
//                                 const char* msg) {
//  std::string r("Non-OK-status: ");
//  r += msg;
//  r += " status: ";
//  r += v.ToString();
//  // Leaks string but this is only to be used in a fatal error message
//  return new std::string(r);
//}

}  // namespace CuptiSamples
