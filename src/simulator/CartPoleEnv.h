//
// Created by jagle on 11/9/2018.
//

#ifndef DNN_TENSORFLOW_CPP_CARTPOLEENV_H
#define DNN_TENSORFLOW_CPP_CARTPOLEENV_H

#include "Environment.h"

#include <random>
#include <cassert>
#include <cstring>

struct CartPoleAction {
  enum Action {
    FIRST = 0,

    LEFT = 0,
    RIGHT = 1,

    LAST = 1,
  };
//  Action action;
//  int64_t action;
  // The action placeholder variable for baselines DQN uses int32_t.
  int32_t action;

  // STL
  CartPoleAction() {}

  CartPoleAction(int action_) :
      action(static_cast<Action>(action_)) {
    assert(action_ >= static_cast<int>(Action::FIRST) && action_ <= static_cast<int>(Action::LAST));
  }

  static CartPoleAction Empty() {
    CartPoleAction action;
    action.action = LEFT;
    return action;
  }
};

struct CartPoleObservation {
  enum State {
    x = 0,
    x_dot = 1,
    theta = 2,
    theta_dot = 3,
    LENGTH = 4,
  };
  StateType state[4];
  void Update(
      StateType x,
      StateType x_dot,
      StateType theta,
      StateType theta_dot) {
    state[State::x] = x;
    state[State::x_dot] = x_dot;
    state[State::theta] = theta;
    state[State::theta_dot] = theta_dot;
  }
  /* Number of elements for storing this state in a contiguous minibatch NDArray. */
  static size_t NumElemsPerEntry() {
    return LENGTH;
  }
  static void CopyToNDArray(int i, const CartPoleObservation& from_obs, std::vector<StateType>& ndarray) {
    assert(static_cast<size_t>(i) <= ndarray.size() - NumElemsPerEntry());
    StateType* dst = ndarray.data() + (i * NumElemsPerEntry());
    memcpy(dst, from_obs.state, sizeof(StateType)*NumElemsPerEntry());
  }
  static CartPoleObservation Empty() {
    CartPoleObservation obs;
    memset(obs.state, 0, sizeof(StateType)*LENGTH);
    return obs;
  }
};

struct CartPoleStepTuple {
  CartPoleObservation obs;
  RewardType reward;
  bool done;
  // info?

  static CartPoleStepTuple Empty() {
    CartPoleStepTuple tupl;
    tupl.obs = CartPoleObservation::Empty();
    tupl.reward = 0;
    tupl.done = false;
    return tupl;
  }
};

class CartPoleEnv {
public:
  std::default_random_engine _generator;

  StateType _gravity;
  StateType _masscart;
  StateType _masspole;
  StateType _total_mass;
  StateType _length;
  StateType _polemass_length;
  StateType _force_mag;
  StateType _tau;

// Angle at which to fail the episode
  StateType _theta_threshold_radians;
  StateType _x_threshold;

  int _steps_beyond_done;

// Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
//  high = np.array([
//      self.x_threshold * 2,
//          np.finfo(np.float32).max,
//          self.theta_threshold_radians * 2,
//          np.finfo(np.float32).max])

//  self.action_space = spaces.Discrete(2)
//  self.observation_space = spaces.Box(-high, high)

//  self.seed()
//  self.viewer = None
//  self.state = None

//  self.steps_beyond_done = None



  CartPoleObservation _state;
  int _seed;

  CartPoleEnv(int seed);
  CartPoleStepTuple Step(CartPoleAction action);
  CartPoleObservation Reset();
  void Render();
  void Seed(int seed);
};


#endif //DNN_TENSORFLOW_CPP_CARTPOLEENV_H
