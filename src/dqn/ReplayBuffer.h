//
// Created by jagle on 11/9/2018.
//

#ifndef DNN_TENSORFLOW_CPP_REPLAYBUFFER_H
#define DNN_TENSORFLOW_CPP_REPLAYBUFFER_H

#include <cassert>
#include <functional>
#include <vector>
#include <unordered_map>
#include <random>

// Only in C++17;
// Use boost instead.
// https://theboostcpplibraries.com/boost.any
//#include <any>
#include <boost/any.hpp>

// Straight port of ReplayBuffer from baselines repository.

// TupleType:
//   obs_t, action, reward, obs_tp1, done

#define REPLAY_BUFFER_SEED(seed) (seed+12345)

template <class ObsType, class ActionType, class TupleType>
class StorageEntry {
public:
  /* new_obs = self.env.step(action)
   */
  ObsType new_obs;
  ActionType action;
  /* (new_obs, rew, done) = tupl
   *   tupl.reward
   *   tupl.done
   *   tupl.obs
   *
   *   (possibly other fields also, but at least the above.)
   */
  TupleType tupl;

  // STL.
  StorageEntry() {
  }

  StorageEntry(ObsType new_obs_, ActionType action_, TupleType tupl_) :
      new_obs(new_obs_)
      , action(action_)
      , tupl(tupl_) {
  }
};

using ReplayBufferMinibatch = std::unordered_map<std::string, boost::any>;

template <class ObsType, class ActionType, class TupleType, class CtxType>
using FuncAddTupleType = std::function<
    void(
        int i,
        ReplayBufferMinibatch& batch,
        const StorageEntry<ObsType, ActionType, TupleType>& tupl,
        CtxType& ctx)>;

template <class ObsType, class ActionType, class TupleType, class CtxType>
class ReplayBuffer {
public:

  std::vector<StorageEntry<ObsType, ActionType, TupleType>> _storage;
  size_t _next_idx;
  int _seed;
  size_t _seen;
  size_t _size;
  FuncAddTupleType<ObsType, ActionType, TupleType, CtxType> _func_add_tuple;
  std::default_random_engine _generator;


  ReplayBuffer(
      int seed, size_t size, FuncAddTupleType<ObsType, ActionType, TupleType, CtxType> func_add_tuple) :
      _next_idx(0),
      _seed(REPLAY_BUFFER_SEED(seed)),
      _seen(0),
      _size(size),
      _func_add_tuple(func_add_tuple)
  {
    _storage.resize(size);
    _generator.seed(_seed);
  }

  void Add(
      ObsType old_obs, ActionType action,
      /*(new_obs, rew, done)=*/TupleType tupl)
  {
    _storage[_next_idx] = StorageEntry<ObsType, ActionType, TupleType>(old_obs, action, tupl);
    _next_idx = (_next_idx + 1) % _size;
    _seen = std::min(_seen + 1, _size);
  }

  ReplayBufferMinibatch Sample(size_t batch_size, CtxType& ctx) {
    assert(_seen > 0);
    assert(_storage.size() > 0);
    ReplayBufferMinibatch batch;
    std::uniform_int_distribution<> dis(0, _seen);
    for (size_t i = 0; i < batch_size; i++) {
      int idx = dis(_generator);
      auto& tupl = _storage[idx];
      // _func_add_tuple will append to each np.array(...) entry using single element from the tuple.
      // _func_add_tuple is also responsible for adding entries if they do not already exist.
      _func_add_tuple(i, batch, tupl, ctx);
    }
    return batch;
  }



};


#endif //DNN_TENSORFLOW_CPP_REPLAYBUFFER_H
