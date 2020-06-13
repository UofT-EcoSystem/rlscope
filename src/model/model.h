//
// Created by jagle on 11/12/2018.
//

#ifndef DNN_TENSORFLOW_CPP_MODEL_H
#define DNN_TENSORFLOW_CPP_MODEL_H


#include "tensorflow/c/c_test_util.h"
//#include "tensorflow/core/platform/logging.h"
#include "common_util.h"

#include "tensorflow/c/c_api.h"
//#include "tensorflow/c/c_api_internal.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <initializer_list>
#include <cassert>
#include <utility>
#include <iostream>

#include "dqn/Hyperparameters.h"
#include "model/data_set.h"

#include "tf/wrappers.h"

#define REPO_PATH "clone/dnn_tensorflow_cpp"
#define CHECKPOINTS_PATH "checkpoints/model"
#define CSV_BASENAME "normalized_car_features.csv"

//extern const std::string PATH_SEP;
#define PATH_SEP std::string("/")

std::string get_cartpole_hyp_path();
std::string model_path();
std::string get_model_path(const std::string& model_path_subdir);
std::string csv_dir_path();
int DirExists(const char *path);
void print_devices(const TFDeviceList& devices);

class Model {
public:

  TFSession _session;
  std::string _model_path;
  TFDeviceList _devices;
  std::vector<std::string> _variables;
  bool _debug;
  DQNHyperparameters _hyp;

  Model(DQNHyperparameters& hyp, const std::string model_path = std::string(""), bool debug = false);

  TF_Operation* lookup_op(const std::string& pretty_name, const std::string& name);
  void lookup_and_set_op(const std::string& name, TF_Operation** op_member);

  template <typename T>
  void set_tensor_data(TF_Tensor* tensor, std::vector<T> vec) {
    set_tensor_data(tensor, vec.data(), vec.size());
  }

  template <typename T>
  void set_tensor_data(TF_Tensor* tensor, const T* src, size_t length) {
    auto data_size = TF_DataTypeSize(TF_TensorType(tensor));
    assert(data_size == sizeof(T));

    auto src_nbytes = sizeof(T)*length;
    auto dst_nbytes = TF_TensorByteSize(tensor);
    assert(src_nbytes == dst_nbytes);

    T* dst = reinterpret_cast<T*>(TF_TensorData(tensor));
    memcpy(dst, src, dst_nbytes);
  }

  template <typename T>
  std::string _tensor_to_string(TF_Tensor* out) {
    auto dtype = TF_TensorType(out);
    assert(sizeof(T) == TF_DataTypeSize(dtype));
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < tensor_num_elements(out); i++) {
      if (i != 0) {
        ss << ", ";
      }
      auto output_value = reinterpret_cast<T*>(TF_TensorData(out))[i];
      ss << output_value;
    }
    ss << "]";
    return ss.str();
  }


  void print_variables();
  void print_devices();

  size_t tensor_num_elements(TF_Tensor* tensor);

  std::string tf_tensor_to_string(TF_Tensor* out);

  template <class Container>
  std::string container_to_string(Container& elems) {
    std::stringstream ss;
    ss << "[";
    int i = 0;
    for (auto& it : elems) {
      if (i != 0) {
        ss << ", ";
      }
      ss << it;
      i += 1;
    }
    ss << "]";
    return ss.str();
  }

//      input.flat<string>()(i) = example.SerializeAsString();

  void print_variable(std::string read_op_name);

  bool model_exists();

  void LoadModel();

  void SaveModel();

  virtual void InitOps() = 0;

};

#endif //DNN_TENSORFLOW_CPP_MODEL_H
