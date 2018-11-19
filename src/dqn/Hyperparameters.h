#ifndef DNN_TENSORFLOW_CPP_HYPERPARAMETERS_H
#define DNN_TENSORFLOW_CPP_HYPERPARAMETERS_H

#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include <fstream>

//{'cartpole_config':
//    {'batch_size': 32,
//     'env': 'CartPole-v0',
//     'epsilon_decay_period': 10000.0,
//     'epsilon_train': 0.02,
//     'evaluation_steps': 1000,
//     'gamma': 1.0,
//     'gradient_norm_clip': 10.0,
//     'learning_rate': 0.001,
//     'min_replay_history': 1000,
//     'num_iterations': 20,
//     'optimizer': <class 'tensorflow.python.training.adam.AdamOptimizer'>,
//     'replay_capacity': 5000,
//     'target_update_period': 500,
//     'training_steps': 5000,
//     'update_period': 1}}
struct DQNHyperparameters {
  int num_iterations;
  int num_training_steps;
  int min_replay_history;
  int max_steps_per_episode;
  int update_period;
  int target_update_period;
  int checkpoint_freq_per_step;
  int print_freq_per_episode;
  int replay_capacity;
  int batch_size;
  double epsilon_decay_period;
  double epsilon_train;

  static DQNHyperparameters FromJson(std::string path) {
    boost::filesystem::path file(path);

    if(!boost::filesystem::exists(file)) {
      LOG(FATAL) << "Couldn't find DQN hyperparameter JSON file @ path: " << path;
      exit(EXIT_FAILURE);
    }
    // read a JSON file
    std::ifstream inp(path);
    json j;
    inp >> j;

    DQNHyperparameters hyp;
    hyp.num_iterations = j["num_iterations"];
    hyp.num_training_steps = j["num_training_steps"];
    hyp.min_replay_history = j["min_replay_history"];
    hyp.max_steps_per_episode = j["max_steps_per_episode"];
    hyp.update_period = j["update_period"];
    hyp.target_update_period = j["target_update_period"];
    hyp.checkpoint_freq_per_step = j["checkpoint_freq_per_step"];
    hyp.print_freq_per_episode = j["print_freq_per_episode"];
    hyp.replay_capacity = j["replay_capacity"];
    hyp.batch_size = j["batch_size"];
    hyp.epsilon_decay_period = j["epsilon_decay_period"];
    hyp.epsilon_train = j["epsilon_train"];

    return hyp;
  }
};

#endif // DNN_TENSORFLOW_CPP_HYPERPARAMETERS_H
