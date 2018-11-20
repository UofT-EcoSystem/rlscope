//
// Created by jagle on 11/13/2018.
//

#ifndef DNN_TENSORFLOW_CPP_CARTPOLEMODEL_H
#define DNN_TENSORFLOW_CPP_CARTPOLEMODEL_H

#include "model/model.h"

#include "tensorflow/c/c_api.h"

#include <vector>
#include <tuple>

#include "model/data_set.h"
#include "simulator/CartPoleEnv.h"
#include "common/util.h"
#include "dqn/ReplayBuffer.h"
#include "dqn/Algorithm.h"

#include "tf/wrappers.h"

#include <type_traits>

class CartPoleModel : public Model {
public:
  using ObsType = CartPoleObservation;
  using ActionType = CartPoleAction;

  template <typename T, TF_DataType TFDataType>
  TFTensor ScalarTFTensor(T scalar) {
    assert(sizeof(T) == TF_DataTypeSize(TFDataType));
    const size_t num_scalar_dims = 0;
    const int64_t scalar_dims[0] = {};
    TFTensor tf_scalar = TFTensor(TF_AllocateTensor(
        TFDataType,
        scalar_dims, num_scalar_dims,
        sizeof(T)*1));
    this->set_tensor_data(tf_scalar.get(), &scalar, 1);
    return tf_scalar;
  }

  template <typename T, TF_DataType TFDataType>
  TFTensor VectorTFTensor(std::vector<T>& vec, const std::vector<int64_t>& vector_dims) {
    assert(sizeof(T) == TF_DataTypeSize(TFDataType));

    int64_t n_elems;
    if (vector_dims.size() == 0) {
      n_elems = 0;
    } else {
      n_elems = 1;
      for (auto dim : vector_dims) {
        n_elems *= dim;
      }
    }
    assert(vec.size() == static_cast<size_t>(n_elems));

    TFTensor tf_vector = TFTensor(TF_AllocateTensor(
        TFDataType,
        vector_dims.data(), vector_dims.size(),
        sizeof(T)*vec.size()));
    this->set_tensor_data(tf_vector.get(), vec.data(), vec.size());
    return tf_vector;
  }

  template <typename T, TF_DataType TFDataType>
  TFTensor VectorTFTensor(std::vector<T>& vec) {
    std::vector<int64_t> vector_dims{static_cast<int64_t>(vec.size())};
    return VectorTFTensor<T, TFDataType>(vec, vector_dims);
  }

  ActionType Act(ObsType obs, bool stochastic, double epsilon) {
    assert(sizeof(obs.state[0]) == TF_DataTypeSize(TF_FLOAT));
    assert(sizeof(float) == TF_DataTypeSize(TF_FLOAT));

    const size_t state_length = CartPoleObservation::State::LENGTH;
    // obs shape = (1, 4, 1) for CartPole when using baselines Q-network (for some reason...).
    std::vector<int64_t> dims{1, static_cast<int64_t>(state_length), 1};
    TFTensor tf_obs_tensor = TFTensor(TF_AllocateTensor(
        TF_FLOAT,
        dims.data(), static_cast<int>(dims.size()),
        sizeof(obs.state[0])*state_length));
    this->set_tensor_data(tf_obs_tensor.get(), obs.state, state_length);

    TFTensor tf_stochastic = ScalarTFTensor<bool, TF_BOOL>(stochastic);
    TFTensor tf_epsilon = ScalarTFTensor<float, TF_FLOAT>(epsilon);

    TFSession session = this->_session;

    std::vector<std::pair<TF_Operation*, TFTensor>> inputs{
        {_act_observation_op, tf_obs_tensor},
        {_act_stochasic_op, tf_stochastic},
        {_act_update_eps_op, tf_epsilon},
    };
    session.SetInputs(inputs);
    session.SetOutputs({
                           _act_output_merge_op,
                       });
    session.SetTargets({
                           _act_output_group_deps_op,
                       });
    session.Run();

    MY_ASSERT(session.output_values_.size() == 1);
    TFTensor out = session.output_tensor(0);
    MY_ASSERT(out.get() != nullptr);
    auto actual_type = TF_TensorType(out.get());
    MY_ASSERT(TF_INT64 == actual_type);

    MY_ASSERT(this->tensor_num_elements(out.get()) == 1);
    auto output_value = reinterpret_cast<int64_t*>(TF_TensorData(out.get()))[0];
//    if (_debug) {
//      LOG(INFO) << "> Observation: " << tf_tensor_to_string(tf_obs_tensor.get()) << ", Action = " << output_value;
//    }

    ActionType action = output_value;
    return action;
  }
  template <typename T, TF_DataType TFDataType>
  void _AddTensorInput(
      std::vector<std::pair<TF_Operation*, TFTensor>>& inputs,
      const ReplayBufferMinibatch& batch, const std::string& name,
      TF_Operation* op,
      const std::vector<int64_t>& tensor_dims) {
    auto vec = _LookupBatchVector<T>(batch, name);
    auto tf_tensor = VectorTFTensor<T, TFDataType>(*vec.get(), tensor_dims);
    inputs.push_back({op, tf_tensor});
  }

  template <typename T>
//  inline std::vector<T>& _LookupBatchVector(const ReplayBufferMinibatch& batch, const std::string& name) const {
  inline std::shared_ptr<std::vector<T>> _LookupBatchVector(const ReplayBufferMinibatch& batch, const std::string& name) const {
    auto it = batch.find(name);
    assert(it != batch.end());

    // WARNING: any_cast behaves weird if you pass it the wrong arguments...
    // Before, I passed it (*it) which is a key/value pair, instead of (it->second),
    // and it reported that as a bad-any-cast error; I would expect that to
    // be a compile-time error!  Perhaps it auto-boxed the pair in boost:any?
    auto vec = boost::any_cast<std::shared_ptr<std::vector<T>>>( it->second );
    return vec;
  }

  size_t _GetBatchSize(const ReplayBufferMinibatch& batch) const {
    auto vec = _LookupBatchVector<RewardType>(batch, "rewards");
    return vec->size();
  }
  void TrainingUpdate(ReplayBufferMinibatch& batch) {
    TFSession session = this->_session;
    std::vector<std::pair<TF_Operation*, TFTensor>> inputs;

    std::vector<int64_t> obs_dims{_hyp.batch_size, static_cast<int64_t>(ObsType::NumElemsPerEntry())};
    std::vector<int64_t> batch_dims{_hyp.batch_size};
    _AddTensorInput<StateType, TF_FLOAT>(inputs, batch, "obs_t", _train_input_obs_t_op, obs_dims);
//    _AddTensorInput<ActionType, TF_INT64>(inputs, batch, "action", _train_input_action_op);
    _AddTensorInput<ActionType, TF_INT32>(inputs, batch, "action", _train_input_action_op, batch_dims);
    _AddTensorInput<RewardType, TF_FLOAT>(inputs, batch, "reward", _train_input_reward_op, batch_dims);
    // THIS is wrong since the function is for storing a 1-D vector of scalars.
    _AddTensorInput<StateType, TF_FLOAT>(inputs, batch, "obs_tp1", _train_input_obs_tp1_op, obs_dims);
//    _AddTensorInput<DoneVectorType, TF_BOOL>(inputs, batch, "done", _train_input_done_op);
//    done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
    _AddTensorInput<float, TF_FLOAT>(inputs, batch, "done", _train_input_done_op, batch_dims);

    std::vector<float> ones(_hyp.batch_size, 1.0);
    auto tf_ones = VectorTFTensor<float, TF_FLOAT>(ones, batch_dims);
    inputs.push_back({_train_input_weight_op, tf_ones});

    session.SetInputs(inputs);
    session.SetOutputs({
                           _train_op_tensor,
                       });
    session.SetTargets({
                           _train_op_target,
                       });

    session.Run();

    TFTensor out = session.output_tensor(0);
    MY_ASSERT(out.get() != nullptr);
    auto actual_type = TF_TensorType(out.get());
    MY_ASSERT(TF_FLOAT == actual_type);

//    if (_debug) {
// NOTE: output is the TD-error vector (for each sample)
//    LOG(INFO) << "> TD-error output: " << tf_tensor_to_string(out.get());
//    }

    MY_ASSERT(this->tensor_num_elements(out.get()) == static_cast<size_t>(_hyp.batch_size));
  }
  void SyncWeights() {
    TFSession session = this->_session;
    std::vector<std::pair<TF_Operation*, TFTensor>> inputs;
    session.SetInputs(inputs);
//    session.SetOutputs({
//                       });
    session.SetTargets({
                           _sync_op,
                       });
    session.Run();
  }

//  [<baselines.deepq.utils.ObservationInput object at 0x7f885c09b6d8>,
//  <tf.Tensor 'deepq/stochastic:0' shape=() dtype=bool>,
//  <tf.Tensor 'deepq/update_eps:0' shape=() dtype=float32>]
// First thing turns into:
//
//  {<tf.Tensor 'deepq/observation:0' shape=(?, 4, 1) dtype=float32>: array([[[-0.02621594],
//    [ 0.04859311],
//    [-0.01906417],
//    [-0.02885343]]])}
// observation receives a (1 x 4 x 1) tensor.
// update_eps receives a numpy.float64 of 1.0
// stochastic receives a bool of True
//
// Outputs:
// [<tf.Tensor 'deepq/cond/Merge:0' shape=(?,) dtype=int64>,
//  <tf.Operation 'deepq/group_deps' type=NoOp>]
// They discard the last result though since it is None.
// _Function.__init__ @ baselines/common/tf_util.py:186
// It creates this "group_deps" operation by calling:
//  self.update_group = tf.group(*updates)
// I'm guessing updates are just some variable update operations (IF the operation even contains any)
//
// The act function was created using this call:
//   eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
//   update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
//   _act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
//                     outputs=output_actions,
//                     givens={update_eps_ph: -1.0, stochastic_ph: True},
//                     updates=[update_eps_expr])
// So, looks like this is a hacky way of keeping the last value of epsilon inside the graph.
// 'givens' are defaults to use if values weren't provided (will never happen for us).
// Not really sure why they do this... could just provide epsilon every time. Oh well.

  // Act(obs): Input
  // "deepq/stochastic"
  TF_Operation* _act_stochasic_op;
  // "deepq/update_eps"
  TF_Operation* _act_update_eps_op;
  // "deepq/observation"
  TF_Operation* _act_observation_op;
  // Act(obs): Output
  // "deepq/cond/Merge"
  TF_Operation* _act_output_merge_op;
  // "deepq/group_deps"
  TF_Operation* _act_output_group_deps_op;

  // SyncOp() [ a target ]
//  self = <tensorflow.python.client.session.InteractiveSession object at 0x7fd2b8e595c0>
//  fetches = [<tf.Operation 'deepq_1/group_deps_2' type=NoOp>]
//  feed_dict = {}
//  options = None
//  run_metadata = None

// "deepq_1/group_deps_2"
  TF_Operation* _sync_op;

  // TrainOp
//  [<tf.Tensor 'deepq_1/sub_1:0' shape=(?,) dtype=float32>,
//  <tf.Operation 'deepq_1/group_deps_1' type=NoOp>]
  // "deepq_1/sub_1:0"
  TF_Operation* _train_op_tensor;
  // "deepq_1/group_deps_1"
  TF_Operation* _train_op_target;

  TF_Operation* _train_input_obs_t_op;
  TF_Operation* _train_input_action_op;
  TF_Operation* _train_input_reward_op;
  TF_Operation* _train_input_obs_tp1_op;
  TF_Operation* _train_input_done_op;
  TF_Operation* _train_input_weight_op;

  CartPoleModel(DQNHyperparameters& hyp, const std::string model_path, bool debug) :
      Model(hyp, model_path, debug) {
  }

  virtual void InitOps() {

    // Act(obs): Input
    lookup_and_set_op("deepq/stochastic", &_act_stochasic_op);
    lookup_and_set_op("deepq/update_eps", &_act_update_eps_op);
    lookup_and_set_op("deepq/observation", &_act_observation_op);

    // Act(obs): Output
    lookup_and_set_op("deepq/cond/Merge", &_act_output_merge_op);
    lookup_and_set_op("deepq/group_deps", &_act_output_group_deps_op);

    // SyncOp()
    lookup_and_set_op("deepq_1/group_deps_2", &_sync_op);

    // TrainOp()
    lookup_and_set_op("deepq_1/sub_1", &_train_op_tensor);
    lookup_and_set_op("deepq_1/group_deps_1", &_train_op_target);
    // Baselines Q-network definition also defines these placeholders;
    // I don't understand why it does this though.
    // train_op inputs (Q: are these inputs for train_op tensor or target...?
//    <tf.Operation 'deepq_1/obs_t' type=Placeholder>,
//    <tf.Operation 'deepq_1/action' type=Placeholder>,
//    <tf.Operation 'deepq_1/reward' type=Placeholder>,
//    <tf.Operation 'deepq_1/obs_tp1' type=Placeholder>,
//    <tf.Operation 'deepq_1/done' type=Placeholder>,
//    <tf.Operation 'deepq_1/weight' type=Placeholder>,
    lookup_and_set_op("deepq_1/obs_t", &_train_input_obs_t_op);
    lookup_and_set_op("deepq_1/action", &_train_input_action_op);
    lookup_and_set_op("deepq_1/reward", &_train_input_reward_op);
    lookup_and_set_op("deepq_1/obs_tp1", &_train_input_obs_tp1_op);
    lookup_and_set_op("deepq_1/done", &_train_input_done_op);
    lookup_and_set_op("deepq_1/weight", &_train_input_weight_op);

  }

};

void cartpole_add_tuple(
    const DQNHyperparameters& hyp,
    int i,
    ReplayBufferMinibatch& batch,
    const StorageEntry<CartPoleObservation, CartPoleAction, CartPoleStepTuple>& entry,
    DQNCtx& ctx);


#endif //DNN_TENSORFLOW_CPP_CARTPOLEMODEL_H
