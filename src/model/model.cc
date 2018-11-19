#include "common/debug.h"
#include "model/model.h"
#include "dqn/ReplayBuffer.h"
#include "dqn/Algorithm.h"
#include "simulator/CartPoleEnv.h"
#include "tf/wrappers.h"

using namespace std;

std::string model_path() {
  auto home = std::getenv("HOME");
  std::string path = home + PATH_SEP + REPO_PATH + PATH_SEP + CHECKPOINTS_PATH + PATH_SEP + "model_checkpoint";
  return path;
}

std::string get_cartpole_hyp_path() {
  auto home = std::getenv("HOME");
  std::string path = home + PATH_SEP + REPO_PATH + PATH_SEP + "config" + PATH_SEP + "cartpole_dqn.json";
  return path;
}

std::string get_model_path(const std::string& model_path_subdir) {
  auto home = std::getenv("HOME");
  std::string path = home
                     + PATH_SEP + REPO_PATH
                     + PATH_SEP + CHECKPOINTS_PATH
                     + PATH_SEP + model_path_subdir
                     + PATH_SEP + "model_checkpoint";
  return path;
}


std::string csv_dir_path() {
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

Model::Model(DQNHyperparameters& hyp, const std::string path, bool debug) :
    _debug(debug)
    , _hyp(hyp)
{
  if (path == "") {
    _model_path = model_path();
  } else {
    _model_path = path;
  }
}


TF_Operation* Model::lookup_op(const std::string& pretty_name, const std::string& name) {
  TF_Graph* graph = _session._graph.get();
  TF_Operation* op = TF_GraphOperationByName(graph, name.c_str());
  if (op == nullptr) {
    LOG(WARNING) << "Couldn't find op = " << name;
    MY_ASSERT(op != nullptr);
  }
  MY_ASSERT(op != nullptr);
//    auto debug_str = op->node.DebugString();
//    std::cout << "> " << pretty_name << " = " << debug_str << std::endl;
  return op;
}

void Model::lookup_and_set_op(const std::string& name, TF_Operation** op_member) {
  *op_member = lookup_op(name, name);
}

void Model::print_variables() {
//    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32_ref>,
//    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>,
//    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>,
//    <tf.Variable 'predictions/kernel:0' shape=(2, 1) dtype=float32_ref>,
//    <tf.Variable 'predictions/bias:0' shape=(1,) dtype=float32_ref>]
  for (auto& var : _variables) {
    print_variable(var);
  }
}

void print_devices(const TFDeviceList& devices) {
  auto ndevices = devices.count();
  for (int i = 0; i < ndevices; i++) {
    auto device_name = devices.DeviceName(i);
    assert(device_name != nullptr);
    LOG(INFO) << "devices[" << i << "] = " << device_name;
  }
}


void Model::print_devices() {
  ::print_devices(_devices);
}

void Model::LoadModel() {
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

  _devices = TFDeviceList(_session);

  print_variables();
  print_devices();

  InitOps();
}

size_t Model::tensor_num_elements(TF_Tensor* tensor) {
  auto nbytes = TF_TensorByteSize(tensor);
  auto dt_nbytes = TF_DataTypeSize(TF_TensorType(tensor));
  auto n_elems = nbytes / dt_nbytes;
  return n_elems;
}

std::string Model::tf_tensor_to_string(TF_Tensor* out) {
  TF_DataType dtype = TF_TensorType(out);
  switch (dtype) {
    case TF_FLOAT:
      return _tensor_to_string<float>(out);
    case TF_INT32:
      return _tensor_to_string<int32_t>(out);
    case TF_INT64:
      return _tensor_to_string<int64_t>(out);
    case TF_BOOL:
      return _tensor_to_string<char>(out);
    default:
      assert(0);
      break;
  }
  return "";
}

void Model::print_variable(std::string read_op_name) {
//    [ <tf.Variable 'dense/kernel:0' shape=(3, 3) dtype=float32_ref>,
//    <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32_ref>,
//    <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32_ref>,
//    <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>,
//    <tf.Variable 'predictions/kernel:0' shape=(2, 1) dtype=float32_ref>,
//    <tf.Variable 'predictions/bias:0' shape=(1,) dtype=float32_ref>]
//    operation is: predictions/bias/read

  auto read_op = lookup_op(read_op_name, read_op_name);
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

void Model::SaveModel() {
  // Not implemented.
  MY_ASSERT(false);
}

