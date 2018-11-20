//
// Created by jagle on 11/12/2018.
//

#include "common/debug.h"
#include "model/CartPoleModel.h"
#include "dqn/Algorithm.h"
#include "tf/wrappers.h"

#include "common/util.h"

#include <boost/filesystem.hpp>
#include <boost/any.hpp>

#include <initializer_list>

#include <gflags/gflags.h>
#include <memory>

DEFINE_bool(debug, false, "Debug");
DEFINE_bool(train, false, "Train");
DEFINE_int64(num_episodes, 100, "Run the loaded model for this many episodes");
DEFINE_int32(mean_length, 100, "Report the average total episode reward over the past 100 (default) episodes");
DEFINE_int32(print_freq_per_episode, 10, "Print the average total episode reward every 10 (default) episodes");
DEFINE_int32(seed, 1000, "Random seed");
DEFINE_string(model_path, "", "Path to directory containing a checkpoint taken by the Dopamine framework when training cartpole.");
DEFINE_string(hyp_path, get_cartpole_hyp_path(), "Path to json file containing DQN training hyperparameters");

struct EpisodeResults {
  RewardType episode_reward;
  size_t timesteps;
};
EpisodeResults RunEpisode(std::shared_ptr<CartPoleEnv> env, std::shared_ptr<CartPoleModel> model, CartPoleObservation* obs, double epsilon_eval) {
  RewardType episode_reward = 0;
  *obs = env->Reset();
  bool done = false;
  size_t timesteps = 0;
  while (!done) {
    CartPoleAction action = model->Act(*obs, /*stochastic=*/true, epsilon_eval);
    CartPoleStepTuple tupl = env->Step(action);
    *obs = tupl.obs;
    episode_reward += tupl.reward;
    timesteps += 1;
    done = tupl.done;
  }
  return EpisodeResults{.episode_reward=episode_reward, .timesteps=timesteps};
}

void RunCartpole(std::shared_ptr<CartPoleModel> model, std::shared_ptr<CartPoleEnv> env) {
  double epsilon_eval = 0.001;
//  CartPoleObservation obs = env.Reset();
  CartPoleObservation obs;
  CircularBuffer<RewardType> episode_rewards(FLAGS_mean_length);
  int64_t episode_num = 1;
  for (int64_t t = 0; episode_num < FLAGS_num_episodes; episode_num += 1) {
    auto episode_results = RunEpisode(env, model, &obs, epsilon_eval);
    if (FLAGS_debug) {
      LOG(INFO) << "episode[" << episode_num << "] reward = " << episode_results.episode_reward << ", "
      << "timesteps = " << episode_results.timesteps;
    }
    episode_rewards.Add(episode_results.episode_reward);
    if (episode_num % FLAGS_print_freq_per_episode == 0) {
      auto mean_episode_reward = Average(episode_rewards);
      LOG(INFO) << "mean " << FLAGS_mean_length << " episode reward = " << mean_episode_reward;
    }
    t += episode_results.timesteps;
  }
}

void TrainCartpole(
    DQNHyperparameters& hyp,
    std::shared_ptr<CartPoleModel> model,
    std::shared_ptr<CartPoleEnv> env) {

  using ObsType = CartPoleObservation;
  using ActionType = CartPoleAction;
  using TupleType = CartPoleStepTuple;
  using CtxType = DQNCtx;
  using StorageEntryType = StorageEntry<ObsType, ActionType, TupleType>;

  FuncAddTupleType<ObsType, ActionType, TupleType, CtxType> func_add_tuple = [hyp]
      (int i, ReplayBufferMinibatch& batch, const StorageEntry<ObsType, ActionType, TupleType>& entry, CtxType& ctx) {
    cartpole_add_tuple(hyp, i, batch, entry, ctx);
  };

  DQNAlgorithm<CartPoleModel, CartPoleEnv, CartPoleObservation, CartPoleAction, CartPoleStepTuple, CtxType> algorithm(
      FLAGS_seed, hyp, model, env, func_add_tuple, FLAGS_debug);

  DQNAlgoState<CartPoleObservation> state(hyp, FLAGS_debug);
  DQNCtx ctx;
  algorithm.Run(state, ctx);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  boost::filesystem::path model_dir(FLAGS_model_path);
  if (FLAGS_model_path == "" || !boost::filesystem::is_directory(model_dir)) {
    LOG(INFO) << "ERROR: --model_path must be a directory containing a checkpoint taken by the Dopamine framework when training cartpole.";
    exit(EXIT_FAILURE);
  }

  DQNHyperparameters hyp = DQNHyperparameters::FromJson(FLAGS_hyp_path);
  auto model = std::make_shared<CartPoleModel>(hyp, FLAGS_model_path, FLAGS_debug);
  model->LoadModel();

  auto env = std::make_shared<CartPoleEnv>(FLAGS_seed);

  if (!FLAGS_train) {
    RunCartpole(model, env);
    return 0;
  }

  TrainCartpole(hyp, model, env);

  return 0;
}

