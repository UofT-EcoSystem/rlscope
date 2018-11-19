//
// Created by jagle on 11/12/2018.
//

#include <vector>

#include <gtest/gtest.h>

#include "dqn/ReplayBuffer.h"
#include "model/CartPoleModel.h"

using ObsType = CartPoleObservation;
using ActionType = CartPoleAction;
using TupleType = CartPoleStepTuple;
using CtxType = DQNCtx;
using StorageEntryType = StorageEntry<ObsType, ActionType, TupleType>;
using ReplayBufferType = ReplayBuffer<ObsType, ActionType, TupleType, CtxType>;

#define SEED 0

DQNHyperparameters get_hyp() {
  auto hyp_path = get_cartpole_hyp_path();
  DQNHyperparameters hyp = DQNHyperparameters::FromJson(hyp_path);
  return hyp;
}

ReplayBufferType make_replay_buffer() {
  auto hyp = get_hyp();
  FuncAddTupleType<ObsType, ActionType, TupleType, CtxType> func_add_tuple = [hyp]
      (int i, ReplayBufferMinibatch& batch, const StorageEntry<ObsType, ActionType, TupleType>& entry, CtxType& ctx) {
    cartpole_add_tuple(hyp, i, batch, entry, ctx);
  };
  ReplayBufferType buffer(SEED, hyp.replay_capacity, func_add_tuple);
  return buffer;
}

TEST(ReplayBuffer, TestCreateReplayBuffer) {
  auto buffer = make_replay_buffer();
}

void add_to_replay_buffer(ReplayBufferType& buffer, DQNHyperparameters& hyp) {
  CartPoleObservation obs = CartPoleObservation::Empty();
  CartPoleAction action = CartPoleAction::Empty();
  CartPoleStepTuple tupl = CartPoleStepTuple::Empty();
  for (int i = 0; i < hyp.replay_capacity; i++) {
    buffer.Add(/*old_obs=*/obs, /*action=*/action, /*tupl=*/tupl);
  }
}

TEST(ReplayBuffer, TestAddToReplayBuffer) {
  auto buffer = make_replay_buffer();
  auto hyp = get_hyp();
  add_to_replay_buffer(buffer, hyp);
}

TEST(ReplayBuffer, TestSampleReplayBuffer) {
  auto buffer = make_replay_buffer();
  auto hyp = get_hyp();
  add_to_replay_buffer(buffer, hyp);

  DQNCtx ctx;
  ReplayBufferMinibatch batch = buffer.Sample(hyp.batch_size, ctx);
}
