//
// Created by jagle on 11/9/2018.
//

#include "CartPoleEnv.h"

//#include "tensorflow/core/platform/logging.h"
#include "common_util.h"

#include <iostream>
#include <random>
#include <cassert>
//#define _USE_MATH_DEFINES
//#include <math.h>
#include <cmath>
#include <sstream>
//#include <math.h>

/* Just a straight C++ port of CartPoleEnv from OpenAI gym; see original python file:
 *
 *   https://github.com/openai/gym
 *   gym/envs/classic_control/cartpole.py
 */

CartPoleEnv::CartPoleEnv(int seed) :
    _gravity(9.8)
    , _masscart(1.0)
    , _masspole(0.1)
    , _total_mass(_masspole + _masscart)
    , _length(0.5)
    , _polemass_length(_masspole * _length)
    , _force_mag(10.0)
    , _tau(0.02)
    , _theta_threshold_radians(12 * 2 * M_PI / 360)
    , _x_threshold(2.4)
    , _steps_beyond_done(-1)
{
  // bernoulli random number generator example:
//  const int nrolls=10000;
//
//  std::default_random_engine generator;
//  std::bernoulli_distribution distribution(0.5);
//
//  int count=0;  // count number of trues
//
//  for (int i=0; i<nrolls; ++i) if (distribution(generator)) ++count;
//
//  std::cout << "bernoulli_distribution (0.5) x 10000:" << std::endl;
//  std::cout << "true:  " << count << std::endl;
//  std::cout << "false: " << nrolls-count << std::endl;
  Seed(seed);
}
CartPoleStepTuple CartPoleEnv::Step(CartPoleAction action) {
  assert(CartPoleAction::Action::FIRST <= action.action && action.action <= CartPoleAction::Action::LAST);
  auto x = _state.state[CartPoleObservation::State::x];
  auto x_dot = _state.state[CartPoleObservation::State::x_dot];
  auto theta = _state.state[CartPoleObservation::State::theta];
  auto theta_dot = _state.state[CartPoleObservation::State::theta_dot];

  StateType force;
  if (action.action == CartPoleAction::Action::RIGHT) {
    force = _force_mag;
  } else {
    force = -_force_mag;
  }
  auto costheta = cos(theta);
  auto sintheta = sin(theta);
  auto temp = (force + _polemass_length * theta_dot * theta_dot * sintheta) / _total_mass;
  auto thetaacc = (_gravity * sintheta - costheta* temp) / (_length * (4.0/3.0 - _masspole * costheta * costheta / _total_mass));
  auto xacc = temp - _polemass_length * thetaacc * costheta / _total_mass;
  x = x + _tau * x_dot;
  x_dot = x_dot + _tau * xacc;
  theta = theta + _tau * theta_dot;
  theta_dot = theta_dot + _tau * thetaacc;
  // _state = (x,x_dot,theta,theta_dot)
  _state.Update(x,x_dot,theta,theta_dot);
  auto done =  x < -_x_threshold \
                || x > _x_threshold \
                || theta < -_theta_threshold_radians \
                || theta > _theta_threshold_radians;

  RewardType reward;
  if (!done) {
    reward = 1.0;
  } else if (_steps_beyond_done == -1) {
    // Pole just fell!
    _steps_beyond_done = 0;
    reward = 1.0;
  } else {
    if (_steps_beyond_done == 0) {
      LOG(INFO) << "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.";
    }
    _steps_beyond_done += 1;
    reward = 0.0;
  }

  // return np.array(_state), reward, done, {}
  return CartPoleStepTuple{.obs=_state, .reward=reward, .done=done};
}
CartPoleObservation CartPoleEnv::Reset() {
  //  self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
  //  self.steps_beyond_done = None
  //  return np.array(self.state)
  //  std::default_random_engine generator;
  std::uniform_real_distribution<StateType> distribution(/*a=*/-0.05, /*b=*/0.05);
  _state.state[CartPoleObservation::State::x] = distribution(_generator);
  _state.state[CartPoleObservation::State::x_dot] = distribution(_generator);
  _state.state[CartPoleObservation::State::theta] = distribution(_generator);
  _state.state[CartPoleObservation::State::theta_dot] = distribution(_generator);
  return _state;
}
void CartPoleEnv::Render() {
  // JAMES TODO: figure out how to render visually like in OpenAI gym.
  std::stringstream ss;
  ss << "CartPole:\n";
  ss << "  _state = (";
  ss << "x=" << _state.state[CartPoleObservation::State::x];
  ss << ", " << "x_dot=" << _state.state[CartPoleObservation::State::x_dot];
  ss << ", " << "theta=" << _state.state[CartPoleObservation::State::theta];
  ss << ", " << "theta_dot=" << _state.state[CartPoleObservation::State::theta_dot];
  ss << ")";
  LOG(INFO) << ss.str();
}
void CartPoleEnv::Seed(int seed) {
  _generator.seed(seed);
  _seed = seed;
}
