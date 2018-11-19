//
// Created by jagle on 11/12/2018.
//

#include "Algorithm.h"

float LinearlyDecayingEpsilon(int decay_period, int step, int warmup_steps, float epsilon) {
  //  Returns the current epsilon for the agent's epsilon-greedy policy.
  //
  //  This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
  //  al., 2015). The schedule is as follows:
  //  Begin at 1. until warmup_steps steps have been taken; then
  //  Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
  //  Use epsilon from there on.
  //
  //      Args:
  //  decay_period: float, the period over which epsilon is decayed.
  //      step: int, the number of training steps completed so far.
  //      warmup_steps: int, the number of steps taken before epsilon is decayed.
  //      epsilon: float, the final value to which to decay the epsilon parameter.
  //
  //      Returns:
  //  A float, the current epsilon value computed according to the schedule.
  auto steps_left = decay_period + warmup_steps - step;
  auto bonus = (1.0 - epsilon) * steps_left / decay_period;
  bonus = clip(bonus, 0., 1. - epsilon);
  return epsilon + bonus;
}
