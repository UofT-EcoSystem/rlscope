#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/c/c_test_util.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <utility>

#include "data_set.h"

#include <gflags/gflags.h>

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;

#define STR(s) #s

#ifndef ASSERT_EQ
#define MY_ASSERT_EQ(a, b, s) ({ \
  if ((a) != (b)) { \
    LOG(INFO) << "ERROR: " << STR(a) << " " << "==" << " " << STR(b) << ": " << TF_Message(s); \
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


  Scope _scope;
  ClientSession _session;
  std::vector<Output> _init_ops;
  std::vector<Output> _update_ops;
  Output _loss;
  Tensor _x_data;
  Tensor _y_data;
//  Placeholder _features;
//  Placeholder _labels;
  Output _features;
  Output _labels;
  DataSet _data_set;

  Output _layer_1;
  Output _layer_2;
  Output _layer_3;

  Model() :
      _scope(Scope::NewRootScope())
      , _session(_scope)
      , _data_set(
          csv_dir_path() + "/",
//          "tensorflow/cc/models/",
          CSV_BASENAME)
  {
  }

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

  void LoadModelCAPI() {
    auto path = model_path();
    if (!DirExists(path.c_str())) {
      std::cerr << "Couldn't find checkpointed model @ " << path << "; run model.py --save-checkpoint first!";
      exit(EXIT_FAILURE);
    }
    // Load the saved model.
//    const char kSavedModel[] = "cc/saved_model/testdata/half_plus_two/00000123";
//    const string saved_model_dir = tensorflow::io::JoinPath(
//        tensorflow::testing::TensorFlowSrcRoot(), kSavedModel);
    TF_SessionOptions* opt = TF_NewSessionOptions();
    TF_Buffer* run_options = TF_NewBufferFromString("", 0);
    TF_Buffer* metagraph = TF_NewBuffer();
    TF_Status* s = TF_NewStatus();
    const char* tags[] = {tensorflow::kSavedModelTagServe};
    TF_Graph* graph = TF_NewGraph();
    TF_Session* session = TF_LoadSessionFromSavedModel(
        opt, run_options, path.c_str(), tags, 1, graph, metagraph, s);
    TF_DeleteBuffer(run_options);
    TF_DeleteSessionOptions(opt);
    tensorflow::MetaGraphDef metagraph_def;
    metagraph_def.ParseFromArray(metagraph->data, metagraph->length);
    TF_DeleteBuffer(metagraph);

    // Operations:
    const string loss_name = "loss/loss";
    const string step_name = "step/step";
    // Tensors:
    const string predictions_name = "'outputs/predictions/Tanh:0";
    const string features_name = "inputs/features:0";
    const string labels_name = "inputs/labels:0";

    auto lookup_op = [graph] (const char* pretty_name, const std::string& name) -> TF_Operation* {
      TF_Operation* op = TF_GraphOperationByName(graph, name.c_str());
      MY_ASSERT(op != nullptr);
      auto debug_str = op->node.DebugString();
      std::cout << "> " << pretty_name << " = " << debug_str << std::endl;
      return op;
    };

//    auto lookup_tensor = [graph] (const char* pretty_name, const std::string& name) -> TF_Tensor* {
////      TF_Operation* op = TF_GraphOperationByName(graph, loss_name.c_str());
////      MY_ASSERT(op != nullptr);
////      auto debug_str = op->node.DebugString();
////      std::cout << "> " << pretty_name << " = " << debug_str << std::endl;
////      return op;
//
//      const tensorflow::string op_name =
//          std::string(tensorflow::ParseTensorName(name).first);
//      TF_Operation* input_op =
//          TF_GraphOperationByName(graph, op_name.c_str());
//      MY_ASSERT(input_op != nullptr);
////      csession.SetInputs({{input_op, TF_TensorFromTensor(input, s)}});
//      TF_Tensor* tensor = TF_TensorFromTensor(input, s);
//      return tensor;
//
//    };

    // JAMES NOTE: TF_Tensor's here are basically numpy arrays...looks like the feed-dict here is more like:
    //   { TF_Operation : TF_Tensor }
    //   unlike in python where its
    //   { get_tensor(op) : nump.array(...) }

    auto loss_op = lookup_op("loss", loss_name);
    auto step_op = lookup_op("step", step_name);

    auto predictions_op = lookup_op("predictions", predictions_name);
    auto features_op = lookup_op("features", features_name);
    auto labels_op = lookup_op("labels", labels_name);

    // JAMES TODO: How can we create a TF_Tensor with a set of float-values?
//    exit(EXIT_SUCCESS);

    CSession csession(session);

//    csession.SetInputs({{input_op, TF_TensorFromTensor(input, s)}});

    tensorflow::Tensor features_tensor = Input::Initializer({_data_set.input(110000.f, Fuel::DIESEL, 7.f)}).tensor;
    TF_Tensor* tf_features_tensor = TF_TensorFromTensor(features_tensor, s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
//    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs = {{features_op, &features_tensor}};
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs{{features_op, tf_features_tensor}};
//    auto pair = std::pair<TF_Operation*, TF_Tensor*>({features_op, &features_tensor});
//    std::pair<TF_Operation*, TF_Tensor*> pair;
//    pair = std::make_pair();
//    std::pair<TF_Operation*, TF_Tensor*> pair{features_op, &features_tensor};
//    inputs.push_back({features_op, &features_tensor});
//    inputs.push_back(pair);
    csession.SetInputs(inputs);

//    ASSERT_TRUE(output_op != nullptr);
    csession.SetOutputs({predictions_op});
    csession.Run(s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);

    TF_Tensor* out = csession.output_tensor(0);
    MY_ASSERT(out != nullptr);
    MY_ASSERT(TF_FLOAT == TF_TensorType(out));

    MY_ASSERT(out->shape.MaxDimensions() == 1);
    MY_ASSERT(out->shape.num_elements() == 1);
    auto output_value = reinterpret_cast<float*>(out->buffer->data())[0];
    LOG(INFO) << "> Output value from neural-network: " << output_value;
    auto pred_price = _data_set.output(output_value);
    LOG(INFO) << "> Predicted price: " << pred_price;

    csession.CloseAndDelete(s);
    MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(s);

    exit(EXIT_SUCCESS);

////    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
//    TF_CHECK_OK(TF_GetCode(s));
//    CSession csession(session);
//
//    // Retrieve the regression signature from meta graph def.
//    const auto signature_def_map = metagraph_def.signature_def();
//    const auto signature_def = signature_def_map.at("regress_x_to_y");
//
//    const string input_name = "loss/loss";
//    const string output_name = "loss/loss";
////    const string input_name =
////        signature_def.inputs().at(tensorflow::kRegressInputs).name();
////    const string output_name =
////        signature_def.outputs().at(tensorflow::kRegressOutputs).name();
//
//    // Write {0, 1, 2, 3} as tensorflow::Example inputs.
//    Tensor input(tensorflow::DT_STRING, TensorShape({4}));
//    for (tensorflow::int64 i = 0; i < input.NumElements(); ++i) {
//      tensorflow::Example example;
//      auto* feature_map = example.mutable_features()->mutable_feature();
//      (*feature_map)["x"].mutable_float_list()->add_value(i);
//      input.flat<string>()(i) = example.SerializeAsString();
//    }
//
//    const tensorflow::string input_op_name =
//        std::string(tensorflow::ParseTensorName(input_name).first);
//    TF_Operation* input_op =
//        TF_GraphOperationByName(graph, input_op_name.c_str());
//    ASSERT_TRUE(input_op != nullptr);
//    csession.SetInputs({{input_op, TF_TensorFromTensor(input, s)}});
//    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
//
//    const tensorflow::string output_op_name =
//        std::string(tensorflow::ParseTensorName(output_name).first);
//    TF_Operation* output_op =
//        TF_GraphOperationByName(graph, output_op_name.c_str());
//    ASSERT_TRUE(output_op != nullptr);
//    csession.SetOutputs({output_op});
//    csession.Run(s);
//    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
//
//    TF_Tensor* out = csession.output_tensor(0);
//    ASSERT_TRUE(out != nullptr);
//    EXPECT_EQ(TF_FLOAT, TF_TensorType(out));
////    EXPECT_EQ(2, TF_NumDims(out));
////    EXPECT_EQ(4, TF_Dim(out, 0));
////    EXPECT_EQ(1, TF_Dim(out, 1));
////    float* values = static_cast<float*>(TF_TensorData(out));
////    // These values are defined to be (input / 2) + 2.
////    EXPECT_EQ(2, values[0]);
////    EXPECT_EQ(2.5, values[1]);
////    EXPECT_EQ(3, values[2]);
////    EXPECT_EQ(3.5, values[3]);
//
//    csession.CloseAndDelete(s);
//    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
//    TF_DeleteGraph(graph);
//    TF_DeleteStatus(s);

  }


  bool model_exists() {
    auto path_01 = model_path() + PATH_SEP + "saved_model.pbtxt";
    auto path_02 = model_path() + PATH_SEP + "saved_model.pb";
    return PathExists(path_01) || PathExists(path_02);
//    return _e(_j(self.args.model_path, 'saved_model.pbtxt')) or
//           _e(_j(self.args.model_path, 'saved_model.pb'))
  }

  void Run() {
    if (FLAGS_dummy) {
      std::cout << "./model" << std::endl;
      exit(EXIT_SUCCESS);
    }

    ReadData();
    if (!FLAGS_load_model or !model_exists()) {
      MY_ASSERT(!model_exists());
      DefineModel();
      TrainModel();
      SaveModel();
    } else {
//      LoadModel();
      LoadModelCAPI();
    }
    Inference();
  }

  void SaveModel() {
    // Not implemented.
    MY_ASSERT(false);
  }

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

};

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::cout << "> FLAGS_load_model = " << FLAGS_load_model << std::endl;

  Model model;
  model.Run();

  return 0;
}
