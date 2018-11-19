//
// Created by jagle on 11/12/2018.
//

#ifndef DNN_TENSORFLOW_CPP_WRAPPERS_H
#define DNN_TENSORFLOW_CPP_WRAPPERS_H

#include "tensorflow/c/c_api.h"

#include "common/debug.h"

#include <cassert>
#include <memory>
#include <vector>

// C++ wrappers around the tensorflow C-API.
//
// These wrappers are minimal and are strictly meant to make
// resource management less painful (RAII instead of C malloc/free).

template <typename TFType, typename TFContainer>
std::vector<TFType*> as_pointer_vector(const TFContainer& elems) {
  int i;

  std::vector<TFType*> ptr_vector(elems.size());
  i = 0;
  for (const auto& elem : elems) {
    TFType* ptr = elem.get();
    ptr_vector[i] = ptr;
    i++;
  }

  return ptr_vector;
}

static void TensorDeleter(void *ptr) {
  TF_Tensor* tensor = static_cast<TF_Tensor*>(ptr);
  if (tensor) {
    TF_DeleteTensor(tensor);
  }
}

class TFTensor {
public:
  std::shared_ptr<TF_Tensor> _tensor;

  TFTensor(TF_Tensor* tensor) :
      _tensor(tensor, TensorDeleter) {
    // Q: How do we handle this gracefully?
    assert(tensor != nullptr);
//    assert(_tensor != nullptr);
  }

  TFTensor() :
      _tensor(nullptr, TensorDeleter) {
  }

  TF_Tensor* get() const {
    return _tensor.get();
  }

  void reset(TF_Tensor* ptr) {
    _tensor.reset(ptr, TensorDeleter);
  }

};


static void StatusDeleter(void *ptr) {
  TF_Status* status = static_cast<TF_Status*>(ptr);
  if (status) {
    TF_DeleteStatus(status);
  }
}

class TFStatus {
public:
  std::shared_ptr<TF_Status> _status;

  TFStatus() :
      _status(TF_NewStatus(), StatusDeleter) {
    // Q: How do we handle this gracefully?
    assert(_status != nullptr);
  }

  TF_Status* get() const {
    return _status.get();
  }

};

static void GraphDeleter(void *ptr) {
  TF_Graph* graph = static_cast<TF_Graph*>(ptr);
  if (graph) {
    TF_DeleteGraph(graph);
  }
}

class TFGraph {
public:
  std::shared_ptr<TF_Graph> _graph;
//  TFStatus _status;

  TFGraph() :
  _graph(TF_NewGraph(), GraphDeleter) {
//    _graph = std::make_shared(TF_NewGraph(), GraphDeleter);
    assert(_graph != nullptr);
  }

//  ~TFGraph() {
//    if (_graph) {
//      TF_DeleteGraph(_graph);
//    }
//  }

  TF_Graph* get() const {
    return _graph.get();
  }

};

struct _TF_Session {
  TF_Session* _session;
//  short dummy;
  _TF_Session(TF_Session* session) : _session(session) {
  }
};

static void SessionDeleter(void *ptr) {
//  TF_Session* session = static_cast<TF_Session*>(ptr);
  _TF_Session* session = static_cast<_TF_Session*>(ptr);
  if (session) {
    TFStatus status;
    TF_CloseSession(session->_session, status.get());
    MY_ASSERT_EQ(TF_OK, TF_GetCode(status.get()), status.get());
    TF_DeleteSession(session->_session, status.get());
    MY_ASSERT_EQ(TF_OK, TF_GetCode(status.get()), status.get());
  }
  delete session;
}

//https://stackoverflow.com/questions/6012157/is-stdunique-ptrt-required-to-know-the-full-definition-of-t
class TFSession {
public:
  std::shared_ptr<_TF_Session> _session;
  TFStatus _status;
  TFGraph _graph;

  // For Run().
  std::vector<TF_Output> inputs_;
  std::vector<TFTensor> input_values_;
  std::vector<TF_Output> outputs_;
  std::vector<TFTensor> output_values_;
  std::vector<TF_Operation*> targets_;

  TFSession() :
      _session(nullptr, SessionDeleter) {
  }

  TF_Session* get() const {
    return _session.get()->_session;
  }

  TFTensor output_tensor(int i) { return output_values_[i]; }

  static TFSession LoadSessionFromSavedModel(const std::string& path, const char** tags);

  void SetInputs(
      std::vector<std::pair<TF_Operation*, TFTensor>> inputs) {
    DeleteInputValues();
    inputs_.clear();
    for (const auto& p : inputs) {
      inputs_.emplace_back(TF_Output{p.first, 0});
      input_values_.emplace_back(p.second);
    }
  }

  void DeleteInputValues() {
    input_values_.clear();
  }

  void ResetOutputValues() {
    output_values_.clear();
  }

  void SetOutputs(std::initializer_list<TF_Operation*> outputs) {
    ResetOutputValues();
    outputs_.clear();
    for (TF_Operation* o : outputs) {
      outputs_.emplace_back(TF_Output{o, 0});
    }
    output_values_.resize(outputs_.size());
  }

  void SetTargets(std::initializer_list<TF_Operation*> targets) {
    targets_.clear();
    for (TF_Operation* t : targets) {
      targets_.emplace_back(t);
    }
  }

  void Run() {
    if (inputs_.size() != input_values_.size()) {
//    ADD_FAILURE() << "Call SetInputs() before Run()";
      LOG(INFO) << "Call SetInputs() before Run()";
//    std::cout << "Call SetInputs() before Run()" << std::endl;
      assert(false);
      return;
    }
    ResetOutputValues();
    output_values_.resize(outputs_.size());

    const TF_Output* inputs_ptr = inputs_.empty() ? nullptr : inputs_.data();
    std::vector<TF_Tensor*> input_values = as_pointer_vector<TF_Tensor>(input_values_);

    const TF_Output* outputs_ptr = outputs_.empty() ? nullptr : outputs_.data();
    std::vector<TF_Tensor*> output_values(outputs_.size(), nullptr);

    TF_Operation* const* targets_ptr = targets_.empty() ? nullptr : targets_.data();

    TF_SessionRun(_session.get()->_session, nullptr,
                  inputs_ptr, input_values.data(), inputs_.size(),
                  outputs_ptr, output_values.data(), outputs_.size(),
                  targets_ptr, targets_.size(),
                  nullptr, _status.get());
    MY_ASSERT_EQ(TF_OK, TF_GetCode(_status.get()), _status.get());

    int i = 0;
    for (TF_Tensor* tf_ptr : output_values) {
      output_values_[i].reset(tf_ptr);
      i++;
    }

    DeleteInputValues();
  }

};


static void DeviceListDeleter(void *ptr) {
  TF_DeviceList* device_list = static_cast<TF_DeviceList*>(ptr);
  if (device_list) {
    TF_DeleteDeviceList(device_list);
  }
}

class TFDeviceList {
public:
  std::shared_ptr<TF_DeviceList> _device_list;
  TFSession _session;
  TFStatus _status;

  TFDeviceList(TFSession session) :
      _session(session) {
    _init();
  }

  void _init() {
    _device_list.reset(
        TF_SessionListDevices(_session.get(), _status.get()),
        DeviceListDeleter);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(_status.get()), _status.get());
  }

  int count() const {
    assert(get() != nullptr);
    return TF_DeviceListCount(get());
  }

  TFDeviceList() :
      _device_list(nullptr, DeviceListDeleter) {
  }

  const char* DeviceName(int index) const {
    assert(index >= 0 && index < count());
    assert(get() != nullptr);
    auto ret = TF_DeviceListName(get(), index, _status.get());
    return ret;
  }

  TF_DeviceList* get() const {
    return _device_list.get();
  }

};

#endif //DNN_TENSORFLOW_CPP_WRAPPERS_H
