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

#include "tensorflow/core/lib/core/status.h"
#include <stdio.h>

namespace rlscope {

Status::Status(rlscope::error::Code code, StringPiece msg) {
  assert(code != rlscope::error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = string(msg);
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const string& Status::empty_string() {
  static string* empty = new string;
  return *empty;
}

string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    char tmp[30];
    const char* type;
    switch (code()) {
      case rlscope::error::CANCELLED:
        type = "Cancelled";
        break;
      case rlscope::error::UNKNOWN:
        type = "Unknown";
        break;
      case rlscope::error::INVALID_ARGUMENT:
        type = "Invalid argument";
        break;
      case rlscope::error::DEADLINE_EXCEEDED:
        type = "Deadline exceeded";
        break;
      case rlscope::error::NOT_FOUND:
        type = "Not found";
        break;
      case rlscope::error::ALREADY_EXISTS:
        type = "Already exists";
        break;
      case rlscope::error::PERMISSION_DENIED:
        type = "Permission denied";
        break;
      case rlscope::error::UNAUTHENTICATED:
        type = "Unauthenticated";
        break;
      case rlscope::error::RESOURCE_EXHAUSTED:
        type = "Resource exhausted";
        break;
      case rlscope::error::FAILED_PRECONDITION:
        type = "Failed precondition";
        break;
      case rlscope::error::ABORTED:
        type = "Aborted";
        break;
      case rlscope::error::OUT_OF_RANGE:
        type = "Out of range";
        break;
      case rlscope::error::UNIMPLEMENTED:
        type = "Unimplemented";
        break;
      case rlscope::error::INTERNAL:
        type = "Internal";
        break;
      case rlscope::error::UNAVAILABLE:
        type = "Unavailable";
        break;
      case rlscope::error::DATA_LOSS:
        type = "Data loss";
        break;
      default:
        snprintf(tmp, sizeof(tmp), "Unknown code(%d)",
                 static_cast<int>(code()));
        type = tmp;
        break;
    }
    string result(type);
    result += ": ";
    result += state_->msg;
    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

string* TfCheckOpHelperOutOfLine(const ::rlscope::Status& v,
                                 const char* msg) {
  string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new string(r);
}

}  // namespace rlscope
