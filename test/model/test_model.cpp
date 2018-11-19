//
// Created by jagle on 11/12/2018.
//

#include <vector>

#include <gtest/gtest.h>

#include "model/CarModel.h"


// NOTE: These tests assume that a model checkpoint has been created already
// using the python script; i.e. we assume you've run this successfully:
//
// python3 python/model.py

CarModel make_car_model(const std::string& path) {
  // WARNING: hyp isn't initialized.
  DQNHyperparameters hyp;
  CarModel model(hyp, path);
  return model;
}

TEST(Model, TestPythonModelCheckpointExists) {
  auto path = model_path();
  assert(DirExists(path.c_str()));
}

TEST(Model, TestLoadModel) {
  auto path = model_path();
  CarModel model = make_car_model(path);
  model.LoadModel();
}

void InferenceTest(const std::string& path) {
  CarModel model = make_car_model(path);
  model.LoadModel();

  ASSERT_TRUE(DirExists(path.c_str()));

  // Load the saved model.
  LOG(INFO) << "> Loading model from path = " << path;

  std::vector<float> feature_vector_02 = model._data_set.input_vector(110000.f, Fuel::DIESEL, 7.f);
  LOG(INFO) << "> Features vector 02: " << model.container_to_string(feature_vector_02);

  std::vector<int64_t> dims{1, static_cast<int64_t>(feature_vector_02.size())};
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

  MY_ASSERT(session.output_values_.size() == 1);
  TFTensor out = session.output_tensor(0);
  MY_ASSERT(out.get() != nullptr);
  MY_ASSERT(TF_FLOAT == TF_TensorType(out.get()));

  MY_ASSERT(model.tensor_num_elements(out.get()) == 1);
  auto output_value = reinterpret_cast<float*>(TF_TensorData(out.get()))[0];
  LOG(INFO) << "> Output value from neural-network: " << output_value;
  auto pred_price = model._data_set.output(output_value);
  LOG(INFO) << "> Predicted price: " << pred_price;

}

TEST(Model, TestInference) {
  auto path = model_path();
  InferenceTest(path);
}

TEST(Model, TestGPUInference) {
  auto path = get_model_path("all_gpus");
  InferenceTest(path);
}
