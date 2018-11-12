//
// Created by jagle on 11/12/2018.
//

#include <gtest/gtest.h>

#include "model/model.h"


// NOTE: These tests assume that a model checkpoint has been created already
// using the python script; i.e. we assume you've run this successfully:
//
// python3 python/model.py

TEST(Model, TestPythonModelCheckpointExists) {
  auto path = model_path();
  assert(DirExists(path.c_str()));
}

TEST(Model, TestLoadModel) {
  auto path = model_path();
  Model model(path);
  model.LoadModel();
}

TEST(Model, TestInference) {
  auto path = model_path();
  Model model(path);
  model.LoadModel();

  // std::cerr << "Couldn't find checkpointed model @ " << path << "; run model.py --save-checkpoint first!";
  ASSERT_TRUE(DirExists(path.c_str()));

  // Load the saved model.
  LOG(INFO) << "> Loading model from path = " << path;

//  TF_SessionOptions* opt = TF_NewSessionOptions();
//  TF_Buffer* run_options = TF_NewBufferFromString("", 0);
//  TF_Buffer* metagraph = TF_NewBuffer();
//  TF_Status* s = TF_NewStatus();

  // NOTE: The "tag" used when loading the model must match the tag used when the model was saved.
  // (e.g. using tag kSavedModelTagServe will result in an error).
//    const char* tags[] = {tensorflow::kSavedModelTagTrain};
//  const char* tags[] = {"train"};

//  TFSession session = TFSession::LoadSessionFromSavedModel(path, tags);

//  auto predictions_op = model.lookup_op("predictions", "outputs/predictions/Tanh", session._graph.get());
//  auto features_op = lookup_op("features", "inputs/features", session._graph.get());

//  TFSession();
//  CSession csession(session, /*close_session=*/false);

  vector<float> feature_vector_02 = model._data_set.input_vector(110000.f, Fuel::DIESEL, 7.f);
  LOG(INFO) << "> Features vector 02: " << model.container_to_string(feature_vector_02);

  vector<int64_t> dims{1, static_cast<int64_t>(feature_vector_02.size())};
  TFTensor tf_features_tensor = TFTensor(TF_AllocateTensor(TF_FLOAT,
                                                           dims.data(), static_cast<int>(dims.size()),
                                                           sizeof(float)*feature_vector_02.size()));

  TFSession session = model._session;

  model.set_tensor_data(tf_features_tensor.get(), feature_vector_02);
  LOG(INFO) << "> Features tf_tensor: " << model.tf_tensor_to_string(tf_features_tensor.get());
  std::vector<std::pair<TF_Operation*, TFTensor>> inputs{{model._features_op, tf_features_tensor}};
  session.SetInputs(inputs);
  session.SetOutputs({model._predictions_op});

  session.Run();

  auto print_prediction = [this, &model, &session] () {
    MY_ASSERT(session.output_values_.size() == 1);
    TFTensor out = session.output_tensor(0);
    MY_ASSERT(out.get() != nullptr);
    MY_ASSERT(TF_FLOAT == TF_TensorType(out.get()));

    MY_ASSERT(model.tensor_num_elements(out.get()) == 1);
    auto output_value = reinterpret_cast<float*>(TF_TensorData(out.get()))[0];
    LOG(INFO) << "> Output value from neural-network: " << output_value;
    auto pred_price = model._data_set.output(output_value);
    LOG(INFO) << "> Predicted price: " << pred_price;
  };
  print_prediction();

//  csession.CloseAndDelete(s);
//  MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
//  TF_DeleteGraph(graph);

//  TF_CloseSession(session, s);
//  MY_ASSERT_EQ(TF_OK, TF_GetCode(s), s);
//  TF_DeleteSession(session, s);

//  TF_DeleteStatus(s);

  //
  // NOTE: CSession was written to manage the lifetime of any SetInputs(Tensor) and output tensors.
  // Weird, but OK whatever.
  //

//  exit(EXIT_SUCCESS);

}
