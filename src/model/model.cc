#include "common/debug.h"
#include "model/model.h"
#include "tf/wrappers.h"

using namespace std;

const std::string PATH_SEP = "/";

std::string model_path() {
  auto home = std::getenv("HOME");
  std::string path = home + PATH_SEP + REPO_PATH + PATH_SEP + CHECKPOINTS_PATH + PATH_SEP + "model_checkpoint";
  return path;
}

static std::string csv_dir_path() {
  auto home = std::getenv("HOME");
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

Model::Model(const std::string path) :
#ifdef DEFINE_MODEL
_scope(Scope::NewRootScope()),
      _session(_scope),
#endif
    _data_set(
        csv_dir_path() + "/",
        CSV_BASENAME)
{
  if (path == "") {
    _model_path = model_path();
  } else {
    _model_path = path;
  }
}

#if DEFINE_MODEL
void Model::LoadModelCPPAPI() {
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

void Model::print_graph(TF_Graph* graph) {
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

TF_Operation* Model::lookup_op(const std::string& pretty_name, const std::string& name, TF_Graph* graph) {
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

void Model::print_variables() {
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
    print_variable(var);
  }
}

void Model::LoadModelCAPI() {
  if (!DirExists(_model_path.c_str())) {
    std::cerr << "Couldn't find checkpointed model @ " << _model_path << "; run model.py --save-checkpoint first!";
    exit(EXIT_FAILURE);
  }
  // Load the saved model.
  LOG(INFO) << "> Loading model from path = " << _model_path;

  // NOTE: The "tag" used when loading the model must match the tag used when the model was saved.
  // (e.g. using tag kSavedModelTagServe will result in an error).
  const char* tags[] = {"train"};

  _session = TFSession::LoadSessionFromSavedModel(_model_path, tags);

  print_graph(_session._graph.get());

  _loss_op = lookup_op("loss", "loss/loss", _session._graph.get());
  _step_op = lookup_op("step", "step/step", _session._graph.get());
  _predictions_op = lookup_op("predictions", "outputs/predictions/Tanh", _session._graph.get());
  _features_op = lookup_op("features", "inputs/features", _session._graph.get());
  _labels_op = lookup_op("labels", "inputs/labels", _session._graph.get());

  print_variables();

}

size_t Model::tensor_num_elements(TF_Tensor* tensor) {
  auto nbytes = TF_TensorByteSize(tensor);
  auto dt_nbytes = TF_DataTypeSize(TF_TensorType(tensor));
  auto n_elems = nbytes / dt_nbytes;
  return n_elems;
}

std::string Model::tf_tensor_to_string(TF_Tensor* out) {
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
std::string Model::tensor_to_string(Tensor out) {
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

void Model::print_variable(std::string read_op_name) {
//    [ <tf.Variable 'dense/kernel:0' shape=(3, 3) dtype=float32_ref>,
//    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32_ref>,
//    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>,
//    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>,
//    <tf.Variable 'predictions/kernel:0' shape=(2, 1) dtype=float32_ref>,
//    <tf.Variable 'predictions/bias:0' shape=(1,) dtype=float32_ref>]
//    operation is: predictions/bias/read

  auto read_op = lookup_op(read_op_name, read_op_name, _session._graph.get());
  _session.SetOutputs({read_op});
  _session.Run();

  TFTensor out = _session.output_tensor(0);
  MY_ASSERT(out.get() != nullptr);
  MY_ASSERT(TF_FLOAT == TF_TensorType(out.get()));

  LOG(INFO) << "> Variable " << read_op_name << " = " << tf_tensor_to_string(out.get());
}


bool Model::model_exists() {
  auto path_01 = model_path() + PATH_SEP + "saved_model.pbtxt";
  auto path_02 = model_path() + PATH_SEP + "saved_model.pb";
  return PathExists(path_01) || PathExists(path_02);
}

void Model::LoadModel() {
  LoadModelCAPI();
}

void Model::SaveModel() {
  // Not implemented.
  MY_ASSERT(false);
}

#ifdef DEFINE_MODEL
void Model::ReadData() {
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
void Model::DefineModel() {

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
void Model::TrainModel() {
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
void Model::Inference() {
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
