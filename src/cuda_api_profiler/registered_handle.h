//
// Created by jagle on 8/21/2019.
//

#ifndef IML_REGISTERED_HANDLE_H
#define IML_REGISTERED_HANDLE_H

#include "tensorflow/core/platform/logging.h"

#include <functional>

namespace rlscope {

struct RegisteredHandleInterface {
  virtual ~RegisteredHandleInterface() = default;
};

template <typename HandleType>
struct RegisteredHandle : public RegisteredHandleInterface {
  using UnregisterCallback = std::function<void (HandleType handle)>;

  HandleType handle;
  UnregisterCallback unregister_cb;
  bool registered;

  RegisteredHandle() :
      unregister_cb(nullptr),
      registered(false)
  {
  }
  RegisteredHandle(HandleType handle, UnregisterCallback unregister_cb) :
      handle(handle),
      unregister_cb(unregister_cb),
      registered(true)
  {
  }
  // To prevent double calls to UnregisterCallback(), only allow for move constructor.
  RegisteredHandle(const RegisteredHandle&) = delete;
  RegisteredHandle& operator=(const RegisteredHandle&) = delete;
  RegisteredHandle& operator=(RegisteredHandle&& other) {
    if (this != &other) {
      // Move assignment operator: this is initialized, need to free existing resources first.
      this->_UnregisterFunc();
    }
    this->handle = other.handle;
    this->unregister_cb = other.unregister_cb;
    this->registered = other.registered;
    // Prevent double-calls to unregister_cb.
    other.registered = false;
    return *this;
  }
  RegisteredHandle( RegisteredHandle&& other ) {
    // Move constructor: this is uninitialized, no need to free existing resources.
    this->handle = other.handle;
    this->unregister_cb = other.unregister_cb;
    this->registered = other.registered;
    // Prevent double-calls to unregister_cb.
    other.registered = false;
  }
  void _UnregisterFunc() {
    if (registered) {
      VLOG(1) << "Unregister handle = " << handle;
      unregister_cb(handle);
      registered = false;
    }
  }
  ~RegisteredHandle() {
    _UnregisterFunc();
  }
};

}

#endif //IML_REGISTERED_HANDLE_H
