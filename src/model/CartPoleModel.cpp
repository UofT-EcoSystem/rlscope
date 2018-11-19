//
// Created by jagle on 11/13/2018.
//

#include "common/debug.h"
#include "model/model.h"
#include "tf/wrappers.h"

#include "model/CarModel.h"
#include "dqn/ReplayBuffer.h"
#include "dqn/Algorithm.h"
#include "model/CartPoleModel.h"
#include "simulator/CartPoleEnv.h"

using ObsType = CartPoleObservation;
using ActionType = CartPoleAction;
using TupleType = CartPoleStepTuple;
using CtxType = DQNCtx;
using StorageEntryType = StorageEntry<ObsType, ActionType, TupleType>;

template <typename FieldType, typename GetField, class StorageEntryType, class CtxType>
void _AddFieldToBatch(int i, ReplayBufferMinibatch& batch, const DQNHyperparameters& hyp, CtxType& ctx, StorageEntryType tupl,
                      const std::string& field, GetField get_field) {
  auto it = batch.find(field);
  std::shared_ptr<std::vector<FieldType>> vec;
  if (it == batch.end()) {
    vec = std::make_shared<std::vector<FieldType>>(hyp.batch_size);
    boost::any any_vec = vec;
    batch.insert(std::make_pair(field, any_vec));
  } else {
    vec = boost::any_cast<std::shared_ptr<std::vector<FieldType>>>( it->second );
  }
  (*vec)[i] = get_field(tupl);
}


template <typename FieldType, typename AddField, class StorageEntryType, class CtxType>
void _AddMultiDimFieldToBatch(int i, ReplayBufferMinibatch& batch, const DQNHyperparameters& hyp, CtxType& ctx, StorageEntryType tupl,
                              const std::string& field, size_t num_elems_per_entry, AddField add_field) {
  auto it = batch.find(field);
  std::shared_ptr<std::vector<FieldType>> vec;
  if (it == batch.end()) {
    vec = std::make_shared<std::vector<FieldType>>(hyp.batch_size*num_elems_per_entry);
    boost::any any_vec = vec;
    batch.insert(std::make_pair(field, any_vec));
  } else {
    vec = boost::any_cast<std::shared_ptr<std::vector<FieldType>>>( it->second );
  }
  add_field(i, tupl, *vec);
}

void cartpole_add_tuple(
    const DQNHyperparameters& hyp,
    int i,
    ReplayBufferMinibatch& batch,
    const StorageEntry<ObsType, ActionType, TupleType>& entry,
    DQNCtx& ctx) {
  assert(i < hyp.batch_size);
  _AddMultiDimFieldToBatch<StateType>(i, batch, hyp, ctx, entry, "obs_t", ObsType::NumElemsPerEntry(),
                                      [&ctx] (int i, const StorageEntryType& entry, std::vector<StateType>& vec) {
                                        ObsType::CopyToNDArray(i, entry.tupl.obs, vec);
                                      });
  _AddFieldToBatch<RewardType>(i, batch, hyp, ctx, entry, "reward", [] (const StorageEntryType& entry) { return entry.tupl.reward; });
//    _AddFieldToBatch<DoneVectorType>(i, batch, hyp, ctx, entry, "done", [] (const StorageEntryType& entry) { return entry.tupl.done; });
//    done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
  _AddFieldToBatch<float>(i, batch, hyp, ctx, entry, "done", [] (const StorageEntryType& entry) { return entry.tupl.done; });
  _AddFieldToBatch<ActionType>(i, batch, hyp, ctx, entry, "action", [] (const StorageEntryType& entry) { return entry.action; });
  _AddMultiDimFieldToBatch<StateType>(i, batch, hyp, ctx, entry, "obs_tp1", ObsType::NumElemsPerEntry(),
                                      [&ctx] (int i, const StorageEntryType& entry, std::vector<StateType>& vec) {
                                        ObsType::CopyToNDArray(i, entry.new_obs, vec);
                                      });
  // TODO: add action, and FIRST state ("obs" is the state after the action is taken.)

//  ctx.i += 1;
}
