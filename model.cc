// If define, then hook into the tensorflow C++ API to allow us to define the model,
// its layers, and the backprop computation.
//#define DEFINE_MODEL

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

#include <gflags/gflags.h>

using namespace std;

#define STR(s) #s

#ifndef ASSERT_EQ
#define MY_ASSERT_EQ(a, b, s) ({ \
  if ((a) != (b)) { \
    LOG(INFO) << "ERROR: " << STR(a) << " " << "==" << " " << STR(b) << ": " << TF_Message(s); \
    exit(EXIT_FAILURE); \
  } \
})
#else
#define MY_ASSERT_EQ(a, b, s) ({ \
  ASSERT_EQ(a, b) << TF_Message(s); \
})
#endif

#define MY_ASSERT(t) ({ \
  if (!(t)) { \
    LOG(INFO) << "ERROR: " << STR(t) << " failed. "; \
    exit(EXIT_FAILURE); \
  } \
})

const std::string PATH_SEP = "/";
//#ifdef _WIN32
//                            "\\";
//#else
//                            "/";
//#endif

#define REPO_PATH "clone/dnn_tensorflow_cpp"
#define CHECKPOINTS_PATH "checkpoints/model"
#define CSV_BASENAME "normalized_car_features.csv"
static std::string model_path() {
  auto home = std::getenv("HOME");
  std::string path = home + PATH_SEP + REPO_PATH + PATH_SEP + CHECKPOINTS_PATH + PATH_SEP + "model_checkpoint";
  return path;
}

static std::string csv_dir_path() {
  auto home = std::getenv("HOME");
//  std::string path = home + PATH_SEP + REPO_PATH + PATH_SEP + CSV_BASENAME;
  std::string path = home + PATH_SEP + REPO_PATH;
  return path;
}

inline bool PathExists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

int DirExists(const char *path)
{
    struct stat info;

    if(stat( path, &info ) != 0)
        return 0;
    else if(info.st_mode & S_IFDIR)
        return 1;
    else
        return 0;
}

DEFINE_bool(debug, false, "Debug");
DEFINE_bool(dummy, false, "dummy");
DEFINE_bool(load_model, false, "Load model that was defined and trained in the python version of this script.");
//DEFINE_bool(rebuild_model, false, "Rebuild model");
//DEFINE_int32(measurement_period_sec, 5, "Number of seconds to measure cycles for");
//DEFINE_int32(repetitions, 10, "Repetitions");
//DEFINE_int32(cpu, -1, "CPU core to bind to when measuring CPU frequency; default = num_cpus/2");
//DEFINE_bool(no_bind, false, "DON'T bind to a CPU when measure CPU frequency.");
//DEFINE_string(hz_unit, "GHz", "Unit to report cpu frequency measurements in");

class Model {
public:


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

  Model() :
#ifdef DEFINE_MODEL
      _scope(Scope::NewRootScope()),
      _session(_scope),
#endif
      _data_set(
          csv_dir_path() + "/",
//          "tensorflow/cc/models/",
          CSV_BASENAME)
  {
  }

#if DEFINE_MODEL
  void LoadModel() {
    auto path = model_path();
    if (!DirExists(path.c_str())) {
     std::cerr << "Couldn't find checkpointed model @ " << path << "; run model.py --save-checkpoint first!";
     exit(EXIT_FAILURE);
    }

    SavedModelBundle bundle;
    SessionOptions session_options;
    RunOptions run_options;
    TF_CHECK_OK(LoadSavedModel(session_options, run_options, path, {kSavedModelTagTrain},
                               &bundle));
    std::cout << "> Loaded the model in C++." << std::endl;
    exit(EXIT_SUCCESS);

    // TODO: Set all the Output nodes by querying the loaded computational graph.

  }
#endif // DEFINE_MODEL

  void print_graph(TF_Graph* graph) {
#if DEFINE_MODEL
    mutex_lock l(graph->mu);
    LOG(INFO) << "> Print graph; num_nodes = " << graph->name_map.size();
    int i = 0;
    for (auto& it : graph->name_map) {
      LOG(INFO) << "  name_map[" << i << "] = " << it.first;
      i++;
    }
#endif
  }

  TF_Operation* lookup_op(const std::string& pretty_name, const std::string& name, TF_Graph* graph) {
    TF_Operation* op = TF_GraphOperationByName(graph, name.c_str());
    if (op == nullptr) {
      print_graph(graph);
      LOG(WARNING) << "Couldn't find op = " << name;
      MY_ASSERT(op != nullptr);
    }
    MY_ASSERT(op != nullptr);
//    auto debug_str = op->node.DebugString();
//    std::cout << "> " << pretty_name << " = " << debug_str << std::endl;
    return op;
  }

  template <typename T>
  void set_tensor_data(TF_Tensor* tensor, std::vector<T> vec) {
    T* dst = reinterpret_cast<T*>(TF_TensorData(tensor));
    const T* src = vec.data();
    auto src_nbytes = sizeof(T)*vec.size();
    auto dst_nbytes = TF_TensorByteSize(tensor);
    assert(src_nbytes == dst_nbytes);
    memcpy(dst, src, sizeof(T)*dst_nbytes);
  }

  void print_variables(TF_Session* session, TF_Graph* graph) {
//    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32_ref>,
//    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>,
//    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>,
//    <tf.Variable 'predictions/kernel:0' shape=(2, 1) dtype=float32_ref>,
//    <tf.Variable 'predictions/bias:0' shape=(1,) dtype=float32_ref>]
    std::vector<std::string> variables{
        "dense/kernel/read",
        "dense/bias/read",
        "dense_1/kernel/read",
        "dense_1/bias/read",
        "predictions/kernel/read",
        "predictions/bias/read",
    };
    for (auto& var : variables) {
      print_variable(session, graph, var);
    }
  }

  void LoadModelCAPI() {
    auto path = model_path();
    if (!DirExists(path.c_str())) {
      std::cerr << "Couldn't find checkpointed model @ " << path << "; run model.py --save-checkpoint first!";
      exit(EXIT_FAILURE);
    }
    // Load the saved model.
    LOG(INFO) << "> Loading model from path = " << path;
    TF_SessionOptions* opt = TF_NewSessionOptions();
    TF_Buffer* run_options = TF_NewBufferFromString("", 0);
    TF_Buffer* metagraph = TF_NewBuffer();
    TF_Status* s = TF_NewStatus();
    // NOTE: The "tag" used when loading the model must match the tag used when the model was saved.
    // (e.g. using tag kSavedModelTagServe will result in an error).
//    const char* tags[] = {tensorflow::kSavedModelTagTrain};
    const char* tags[] = {"train"};
    TF_Graph* graph = TF_NewGraph();
    TF_Session* session = TF_LoadSessionFromSavedModel(
        opt, run_options, path.c_str(), tags, 1, graph, metagraph, s);
    if (TF_OK != TF_GetCode(s)) {
      LOG(INFO) << "Failed to load model: " << TF_Message(s);
      exit(EXIT_FAILURE);
    }
//    assert(session->graph == graph);
    TF_DeleteBuffer(run_options);
    TF_DeleteSessionOptions(opt);

//    tensorflow::MetaGraphDef metagraph_def;
//    auto result = metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
//    assert(result);
//    TF_DeleteBuffer(metagraph);

    print_graph(graph);

    // Doesn't print anything.
//    const auto signature_def_map = metagraph_def.signature_def();
//    LOG(INFO) << "signature_def_map = ";
//    for (auto& it : signature_def_map) {
//      LOG(INFO) << "  key=" << it.first;
//      LOG(INFO) << "    " << it.second.DebugString();
//    }

    // JAMES NOTE: TF_Tensor's here are basically numpy arrays...
    // Looks like the feed-dict here is more like:
    //   { TF_Operation : TF_Tensor }
    //   unlike in python where its
    //   { get_tensor(op) : nump.array(...) }

    auto loss_op = lookup_op("loss", "loss/loss", graph);
    auto step_op = lookup_op("step", "step/step", graph);

    auto predictions_op = lookup_op("predictions", "outputs/predictions/Tanh", graph);
    auto features_op = lookup_op("features", "inputs/features", graph);
    auto labels_op = lookup_op("labels", "inputs/labels", graph);

    print_variables(session, graph);

    // JAMES TODO: How can we create a TF_Tensor with a set of float-values?
//    exit(EXIT_SUCCESS);

    CSession csession(session, /*close_session=*/false);

    vector<float> feature_vector_02 = _data_set.input_vector(110000.f, Fuel::DIESEL, 7.f);
    LOG(INFO) << "> Features vector 02: " << container_to_string(feature_vector_02);

    vector<int64_t> dims{1, static_cast<int64_t>(feature_vector_02.size())};
    TF_Tensor* tf_features_tensor = TF_AllocateTensor(TF_FLOAT,
        dims.data(), static_cast<int>(dims.size()),
        sizeof(float)*feature_vector_02.size());
    assert(tf_features_tensor != nullptr);

    set_tensor_data(tf_features_tensor, feature_vector_02);
    LOG(INFO) << "> Features tf_tensor: " << tf_tensor_to_string(tf_features_tensor);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs{{features_op, tf_features_tensor}};
    csession.SetInputs(inputs);
    csession.SetOutputs({predictions_op});

    csession.Run(s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);

    auto print_prediction = [this, &csession] () {
      MY_ASSERT(csession.output_values_.size() == 1);
      TF_Tensor* out = csession.output_tensor(0);
      MY_ASSERT(out != nullptr);
      MY_ASSERT(TF_FLOAT == TF_TensorType(out));

      MY_ASSERT(tensor_num_elements(out) == 1);
      auto output_value = reinterpret_cast<float*>(TF_TensorData(out))[0];
      LOG(INFO) << "> Output value from neural-network: " << output_value;
      auto pred_price = _data_set.output(output_value);
      LOG(INFO) << "> Predicted price: " << pred_price;
    };
    print_prediction();

    csession.CloseAndDelete(s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
    TF_DeleteGraph(graph);

    TF_CloseSession(session, s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
    TF_DeleteSession(session, s);

    TF_DeleteStatus(s);

    //
    // NOTE: CSession was written to manage the lifetime of any SetInputs(Tensor) and output tensors.
    // Weird, but OK whatever.
    //

    exit(EXIT_SUCCESS);

  }

  size_t tensor_num_elements(TF_Tensor* tensor) {
    auto nbytes = TF_TensorByteSize(tensor);
    auto dt_nbytes = TF_DataTypeSize(TF_TensorType(tensor));
    auto n_elems = nbytes / dt_nbytes;
    return n_elems;
  }

  std::string tf_tensor_to_string(TF_Tensor* out) {
    stringstream ss;
    ss << "[";
    for (size_t i = 0; i < tensor_num_elements(out); i++) {
      if (i != 0) {
        ss << ", ";
      }
      auto output_value = reinterpret_cast<float*>(TF_TensorData(out))[i];
      ss << output_value;
    }
    ss << "]";
    return ss.str();
  }

#ifdef DEFINE_MODEL
  std::string tensor_to_string(Tensor out) {
    stringstream ss;
    ss << "[";
    for (size_t i = 0; i < out.shape().num_elements(); i++) {
      if (i != 0) {
        ss << ", ";
      }
      auto output_value = out.flat<float>()(i);
      ss << output_value;
    }
    ss << "]";
    return ss.str();
  }
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

  void print_variable(TF_Session* session, TF_Graph* graph, std::string read_op_name) {
//    [ <tf.Variable 'dense/kernel:0' shape=(3, 3) dtype=float32_ref>,
//    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32_ref>,
//    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>,
//    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>,
//    <tf.Variable 'predictions/kernel:0' shape=(2, 1) dtype=float32_ref>,
//    <tf.Variable 'predictions/bias:0' shape=(1,) dtype=float32_ref>]
//    operation is: predictions/bias/read
    TF_Status* s = TF_NewStatus();
    CSession csession(session, /*close_session=*/false);

    auto read_op = lookup_op(read_op_name, read_op_name, graph);
    csession.SetOutputs({read_op});
    csession.Run(s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);

    TF_Tensor* out = csession.output_tensor(0);
    MY_ASSERT(out != nullptr);
    MY_ASSERT(TF_FLOAT == TF_TensorType(out));

//    MY_ASSERT(out->shape.num_elements() == 1);
    LOG(INFO) << "> Variable " << read_op_name << " = " << tf_tensor_to_string(out);
    csession.CloseAndDelete(s);
    TF_DeleteStatus(s);
  }


  bool model_exists() {
    auto path_01 = model_path() + PATH_SEP + "saved_model.pbtxt";
    auto path_02 = model_path() + PATH_SEP + "saved_model.pb";
    return PathExists(path_01) || PathExists(path_02);
  }

  void Run() {
    if (FLAGS_dummy) {
      std::cout << "./model" << std::endl;
      exit(EXIT_SUCCESS);
    }

#ifdef DEFINE_MODEL
    ReadData();
#endif
    if (!FLAGS_load_model or !model_exists()) {
      MY_ASSERT(!model_exists());
#ifdef DEFINE_MODEL
      DefineModel();
      TrainModel();
#else
      LOG(INFO) << "ERROR: Cannot define model without DEFINE_MODEL set";
      assert(false);
#endif
      SaveModel();
    } else {
//      LoadModel();
      LoadModelCAPI();
    }
#ifdef DEFINE_MODEL
    Inference();
#endif
  }

  void SaveModel() {
    // Not implemented.
    MY_ASSERT(false);
  }

#ifdef DEFINE_MODEL
  void ReadData() {
//    DataSet data_set("tensorflow/cc/models/", "normalized_car_features.csv");
    _x_data = Tensor(DataTypeToEnum<float>::v(),
                     TensorShape{static_cast<int>(_data_set.x().size())/3, 3});
    copy_n(_data_set.x().begin(), _data_set.x().size(),
           _x_data.flat<float>().data());

    _y_data = Tensor(DataTypeToEnum<float>::v(),
                     TensorShape{static_cast<int>(_data_set.y().size()), 1});
    copy_n(_data_set.y().begin(), _data_set.y().size(),
           _y_data.flat<float>().data());


  }
#endif

#ifdef DEFINE_MODEL
  void DefineModel() {

    _features = tensorflow::ops::Placeholder(_scope, DT_FLOAT);
    _labels = tensorflow::ops::Placeholder(_scope, DT_FLOAT);

    // weights init
    auto w1 = Variable(_scope, {3, 3}, DT_FLOAT);
    auto assign_w1 = Assign(_scope, w1, RandomNormal(_scope, {3, 3}, DT_FLOAT));
    _init_ops.push_back(assign_w1);

    auto w2 = Variable(_scope, {3, 2}, DT_FLOAT);
    auto assign_w2 = Assign(_scope, w2, RandomNormal(_scope, {3, 2}, DT_FLOAT));
    _init_ops.push_back(assign_w2);

    auto w3 = Variable(_scope, {2, 1}, DT_FLOAT);
    auto assign_w3 = Assign(_scope, w3, RandomNormal(_scope, {2, 1}, DT_FLOAT));
    _init_ops.push_back(assign_w3);

    // bias init
    auto b1 = Variable(_scope, {1, 3}, DT_FLOAT);
    auto assign_b1 = Assign(_scope, b1, RandomNormal(_scope, {1, 3}, DT_FLOAT));
    _init_ops.push_back(assign_b1);

    auto b2 = Variable(_scope, {1, 2}, DT_FLOAT);
    auto assign_b2 = Assign(_scope, b2, RandomNormal(_scope, {1, 2}, DT_FLOAT));
    _init_ops.push_back(assign_b2);

    auto b3 = Variable(_scope, {1, 1}, DT_FLOAT);
    auto assign_b3 = Assign(_scope, b3, RandomNormal(_scope, {1, 1}, DT_FLOAT));
    _init_ops.push_back(assign_b3);

    // layers
    auto _layer_1 = Tanh(_scope, Tanh(_scope, tensorflow::ops::Add(_scope, MatMul(_scope, _features, w1), b1)));
    auto _layer_2 = Tanh(_scope, tensorflow::ops::Add(_scope, MatMul(_scope, _layer_1, w2), b2));
    auto _layer_3 = Tanh(_scope, tensorflow::ops::Add(_scope, MatMul(_scope, _layer_2, w3), b3));

    // regularization
    auto regularization = AddN(_scope,
                               initializer_list<Input>{L2Loss(_scope, w1),
                                                       L2Loss(_scope, w2),
                                                       L2Loss(_scope, w3)});

    // loss calculation
    _loss = tensorflow::ops::Add(_scope,
                    ReduceMean(_scope, Square(_scope, Sub(_scope, _layer_3, _labels)), {0, 1}),
                    tensorflow::ops::Mul(_scope, Cast(_scope, 0.01,  DT_FLOAT), regularization));

    // add the gradients operations to the graph
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(_scope, {_loss}, {w1, w2, w3, b1, b2, b3}, &grad_outputs));

    // update the weights and bias using gradient descent
    auto apply_w1 = ApplyGradientDescent(_scope, w1, Cast(_scope, 0.01,  DT_FLOAT), {grad_outputs[0]});
    auto apply_w2 = ApplyGradientDescent(_scope, w2, Cast(_scope, 0.01,  DT_FLOAT), {grad_outputs[1]});
    auto apply_w3 = ApplyGradientDescent(_scope, w3, Cast(_scope, 0.01,  DT_FLOAT), {grad_outputs[2]});
    auto apply_b1 = ApplyGradientDescent(_scope, b1, Cast(_scope, 0.01,  DT_FLOAT), {grad_outputs[3]});
    auto apply_b2 = ApplyGradientDescent(_scope, b2, Cast(_scope, 0.01,  DT_FLOAT), {grad_outputs[4]});
    auto apply_b3 = ApplyGradientDescent(_scope, b3, Cast(_scope, 0.01,  DT_FLOAT), {grad_outputs[5]});
    _update_ops.push_back(apply_w1);
    _update_ops.push_back(apply_w2);
    _update_ops.push_back(apply_w3);
    _update_ops.push_back(apply_b1);
    _update_ops.push_back(apply_b2);
    _update_ops.push_back(apply_b3);

  }
#endif // DEFINE_MODEL

#ifdef DEFINE_MODEL
  void TrainModel() {
    ClientSession _session(_scope);
    std::vector<Tensor> outputs;

    // init the weights and biases by running the assigns nodes once
//    TF_CHECK_OK(_session.Run({assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3}, nullptr));
    TF_CHECK_OK(_session.Run(_init_ops, nullptr));

    // training steps
    for (int i = 0; i < 5000; ++i) {
      if (i % 100 == 0) {
        TF_CHECK_OK(_session.Run({{_features, _x_data}, {_labels, _y_data}}, {_loss}, &outputs));
        std::cout << "Loss after " << i << " steps " << outputs[0].scalar<float>() << std::endl;
      }
      // nullptr because the output from the run is useless
      TF_CHECK_OK(_session.Run({{_features, _x_data}, {_labels, _y_data}}, _update_ops, nullptr));
//      TF_CHECK_OK(_session.Run({{_features, _x_data}, {_labels, _y_data}}, {apply_w1, apply_w2, apply_w3, apply_b1, apply_b2, apply_b3}, nullptr));
    }

  }
#endif

#ifdef DEFINE_MODEL
  void Inference() {
    std::vector<Tensor> outputs;
    // prediction using the trained neural net
    TF_CHECK_OK(_session.Run(
        // FeedType: (like the feed_dict in python)
        //   typedef std::unordered_map<Output,     Input::Initializer, OutputHash> FeedType;
        //                              _features   np.array(...)
        {{_features, {_data_set.input(110000.f, Fuel::DIESEL, 7.f)}}},
        {_layer_3},
        &outputs));
    cout << "DNN output: " << *outputs[0].scalar<float>().data() << endl;
    std::cout << "Price predicted " << _data_set.output(*outputs[0].scalar<float>().data()) << " euros" << std::endl;

    // saving the model
    //GraphDef graph_def;
    //TF_ASSERT_OK(_scope.ToGraphDef(&graph_def));
  }
#endif

};

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "> FLAGS_load_model = " << FLAGS_load_model << std::endl;

  Model model;
  model.Run();

  return 0;
}
