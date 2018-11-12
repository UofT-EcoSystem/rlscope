//
// Created by jagle on 11/12/2018.
//

#ifndef DNN_TENSORFLOW_CPP_MODEL_H
#define DNN_TENSORFLOW_CPP_MODEL_H


#ifdef DEFINE_MODEL
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/c/c_test_util.h"
using namespace tensorflow;
using namespace tensorflow::ops;
#else // DEFINE_MODEL

#include "tensorflow/c/c_test_util.h"
#include "tensorflow/core/platform/logging.h"

#endif // DEFINE_MODEL

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
#include <cassert>
#include <utility>
#include <iostream>

#include "data_set.h"

#include "tf/wrappers.h"

#define REPO_PATH "clone/dnn_tensorflow_cpp"
#define CHECKPOINTS_PATH "checkpoints/model"
#define CSV_BASENAME "normalized_car_features.csv"

// If define, then hook into the tensorflow C++ API to allow us to define the model,
// its layers, and the backprop computation.
//#define DEFINE_MODEL

std::string model_path();
int DirExists(const char *path);

class Model {
public:

  TF_Operation* _loss_op;
  TF_Operation* _step_op;
  TF_Operation* _predictions_op;
  TF_Operation* _features_op;
  TF_Operation* _labels_op;

  TFSession _session;
  std::string _model_path;

//  Placeholder _features;
//  Placeholder _labels;
  DataSet _data_set;

#ifdef DEFINE_MODEL
  Scope _scope;
  ClientSession _session;
  std::vector<Output> _init_ops;
  std::vector<Output> _update_ops;
  Output _loss;
  Tensor _x_data;
  Tensor _y_data;
  Output _features;
  Output _labels;
  Output _layer_1;
  Output _layer_2;
  Output _layer_3;
#endif

  Model(const std::string model_path = std::string(""));

#if DEFINE_MODEL
  void LoadModelCPPAPI();
#endif // DEFINE_MODEL

  void print_graph(TF_Graph* graph);

  TF_Operation* lookup_op(const std::string& pretty_name, const std::string& name, TF_Graph* graph);

  template <typename T>
  void set_tensor_data(TF_Tensor* tensor, std::vector<T> vec) {
    auto data_size = TF_DataTypeSize(TF_TensorType(tensor));
    assert(data_size == sizeof(T));

    auto src_nbytes = sizeof(T)*vec.size();
    auto dst_nbytes = TF_TensorByteSize(tensor);
    assert(src_nbytes == dst_nbytes);

    T* dst = reinterpret_cast<T*>(TF_TensorData(tensor));
    const T* src = vec.data();
    memcpy(dst, src, dst_nbytes);
  }

  void print_variables();

  void LoadModelCAPI();

  size_t tensor_num_elements(TF_Tensor* tensor);

  std::string tf_tensor_to_string(TF_Tensor* out);

#ifdef DEFINE_MODEL
  std::string tensor_to_string(Tensor out);
#endif

  template <class Container>
  std::string container_to_string(Container& elems) {
    stringstream ss;
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

#ifdef DEFINE_MODEL
  void ReadData();
  void DefineModel();
  void TrainModel();
  void Inference();
#endif

};

#endif //DNN_TENSORFLOW_CPP_MODEL_H
