//
// Created by jagle on 11/12/2018.
//

#ifndef DNN_TENSORFLOW_CPP_ALGORITHM_H
#define DNN_TENSORFLOW_CPP_ALGORITHM_H


#include <memory>

#include <boost/filesystem.hpp>

#include "model/model.h"
#include "simulator/Environment.h"
#include "dqn/ReplayBuffer.h"
#include "dqn/Hyperparameters.h"
#include "common/util.h"

struct DQNAlgoStats {
  int num_training_steps;

  int training_iteration;

  int training_step;
  // Number of episodes observed for the current "iteration"
  // (iteration in the Dopamine sense of the word: e.g. 200 episodes)
  int training_episode;
  RewardType reward_train_sum;

  CircularBuffer<RewardType> episode_rewards;

  // Total number of episodes every observed.
  int num_episodes;

  DQNAlgoStats(int num_training_steps, int replay_capacity) :
      num_training_steps(num_training_steps)

      , training_iteration(0)

      , training_step(0)
      , training_episode(0)
      , reward_train_sum(0)

      , episode_rewards(replay_capacity)
      , num_episodes(0)
  {
  }

  void UpdateStats(RewardType rew, bool done) {
    training_step += 1;
    reward_train_sum += rew;
    if (done) {
      training_episode += 1;
      num_episodes += 1;
    }
    episode_rewards.Add(rew);
  }

  void RecordStats(int t, bool done) {
    if (training_step >= num_training_steps and done) {
      RewardType average_reward_train = reward_train_sum / ((RewardType)training_episode);
      LOG(INFO) << "> Save summary: "
                   << "i=" << training_iteration
                   << ", " << "average_reward_train=" << average_reward_train
                   << ", " << "num_training_steps=" << num_training_steps;

      // TODO: Save tensorboard summaries...not sure how to do that in C++.

      training_iteration += 1;

      training_step = 0;
      training_episode = 0;
      reward_train_sum = 0;
    }
  }
};

template <class ObsType>
struct DQNAlgoState {
  DQNHyperparameters hyp;
  ObsType obs;
  bool done;
  int step_count;
  int episode_step;

  RewardType saved_mean_reward;

  struct DQNAlgoStats stats;
  DQNAlgoState(DQNHyperparameters& hyp) :
      hyp(hyp)
      , done(false)
      , step_count(0)
      , episode_step(0)
      , saved_mean_reward(-1)
      , stats(hyp.num_training_steps, hyp.replay_capacity)
  {
  }
  bool is_done() {
    return step_count >= hyp.num_training_steps;
  }
};

template <typename T>
T clip(T x, T lower, T upper) {
  if (x < lower)
    return lower;

  if (x > upper)
    return upper;

  return x;
}

float LinearlyDecayingEpsilon(int decay_period, int step, int warmup_steps, float epsilon);

template <class Model, class EnvironmentType, class ObsType, class ActionType, class TupleType, class CtxType>
class DQNAlgorithm {
public:

  DQNAlgorithm(int seed, DQNHyperparameters& hyp, std::shared_ptr<Model> model, std::shared_ptr<EnvironmentType> env, FuncAddTupleType<ObsType, ActionType, TupleType, CtxType> func_add_tuple) :
  _hyp(hyp)
  , _model(model)
  , _env(env)
  , _seed(seed)
  , _buffer(seed, _hyp.replay_capacity, func_add_tuple) {
  }
  void Run(DQNAlgoState<ObsType>& state, CtxType& ctx) {
    Setup(state);
    while (!state.is_done()) {
      NextIter(state, ctx);
    }
  }
  void Setup(DQNAlgoState<ObsType>& state) {
    state = DQNAlgoState<ObsType>(_hyp);
    state.obs = _env->Reset();
  }
  void NextIter(DQNAlgoState<ObsType>& state, CtxType& ctx) {
    auto eps = GetEpsilon(state);
    ActionType action = SelectAction(state);
    TupleType tupl = _env->Step(action);
    state.episode_step += 1;
    if (state.episode_step >= _hyp.max_steps_per_episode) {
      tupl.done = true;
    }
    _buffer.Add(/*old_obs=*/state.obs, /*action=*/action, /*(new_obs, rew, done)=*/tupl);
    state.obs = tupl.obs;

    state.stats.UpdateStats(tupl.reward, tupl.done);
    if (tupl.done) {
      state.episode_step = 0;
    }

    if (state.step_count > _hyp.min_replay_history && state.step_count % _hyp.update_period == 0) {
      ReplayBufferMinibatch batch = _buffer.Sample(_hyp.batch_size, ctx);
    }

    if (state.step_count > _hyp.min_replay_history && state.step_count % _hyp.target_update_period == 0) {
      _model->SyncWeights();
    }

    auto mean_100ep_reward = Average(state.stats.episode_rewards);
    if (tupl.done && state.stats.num_episodes % _hyp.print_freq_per_episode == 0) {
      std::stringstream ss;
      ss << "% time spent exploring = " << (100 * eps) << std::endl;
      ss << "episodes = " << state.stats.num_episodes << std::endl;
      ss << "mean 100 episode reward = " << mean_100ep_reward << std::endl;
      ss << "steps = " << state.step_count << std::endl;
      ss << std::endl;
      LOG(INFO) << ss.str();
    }

    state.stats.RecordStats(state.step_count, tupl.done);

    if (_hyp.checkpoint_freq_per_step != -1 && state.step_count > _hyp.min_replay_history &&
        state.stats.num_episodes && state.step_count % _hyp.checkpoint_freq_per_step == 0) {
      if (state.saved_mean_reward == -1 || mean_100ep_reward > state.saved_mean_reward) {
        LOG(INFO) << "> Saving model due to mean reward increase: "
            << state.saved_mean_reward << " -> " << mean_100ep_reward;
        state.saved_mean_reward = mean_100ep_reward;
      }
    }

    state.step_count += 1;
  }

  ActionType SelectAction(const DQNAlgoState<ObsType>& state) {
    auto eps = GetEpsilon(state);
    ActionType action = _model->Act(/*obs=*/state.obs, /*stochastic=*/true, /*epsilon=*/eps);
    return action;
  }

  float GetEpsilon(const DQNAlgoState<ObsType>& state) {
    auto eps = LinearlyDecayingEpsilon(
        /*decay_period=*/_hyp.epsilon_decay_period,
        /*step=*/state.step_count,
        /*warmup_steps=*/_hyp.min_replay_history,
        /*epsilon=*/_hyp.epsilon_train);
    return eps;
  }

  DQNHyperparameters _hyp;
  std::shared_ptr<Model> _model;
  std::shared_ptr<EnvironmentType> _env;
  int _seed;
  ReplayBuffer<ObsType, ActionType, TupleType, CtxType> _buffer;
};

struct DQNCtx {
  DQNCtx() {
  }
};

#endif //DNN_TENSORFLOW_CPP_ALGORITHM_H
