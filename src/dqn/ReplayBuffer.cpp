//
// Created by jagle on 11/9/2018.
//

#include "ReplayBuffer.h"
#include "src/simulator/Environment.h"

//#include <any>
#include <boost/any.hpp>
#include <random>
#include <algorithm>
#include <unordered_map>

template <class TupleType, class FuncAddTupleCtxType>
ReplayBuffer<TupleType, FuncAddTupleCtxType>::ReplayBuffer(int seed, size_t size, FuncAddTupleType func_add_tuple) :
_next_idx(0),
_seed(REPLAY_BUFFER_SEED(seed)),
_seen(0),
_size(size),
_func_add_tuple(func_add_tuple)
{
  _storage.reserve(size);
  _generator.seed(_seed);
}

template <class TupleType, class FuncAddTupleCtxType>
void ReplayBuffer<TupleType, FuncAddTupleCtxType>::Add(TupleType tupl) {
//  data = (obs_t, action, reward, obs_tp1, done)
//
//  if self._next_idx >= len(self._storage):
//  self._storage.append(data)
//  else:
//  self._storage[self._next_idx] = data
//  self._next_idx = (self._next_idx + 1) % self._maxsize
  _storage[_next_idx] = tupl;
  _next_idx = (_next_idx + 1) % _size;
  _seen = std::min(_seen + 1, _size);
}

template <class TupleType, class FuncAddTupleCtxType>
std::unordered_map<std::string, boost::any> ReplayBuffer<TupleType, FuncAddTupleCtxType>::Sample(
    size_t batch_size, FuncAddTupleCtxType ctx) {
  assert(_storage.size() > 0);
  std::unordered_map<std::string, boost::any> batch;
  std::uniform_int_distribution<> dis(0, _seen);
  for (size_t i = 0; i < batch_size; i++) {
    int idx = dis(_generator);
    auto& tupl = _storage[idx];
    // _func_add_tuple will append to each np.array(...) entry using single element from the tuple.
    // _func_add_tuple is also responsible for adding entries if they do not already exist.
    //
    _func_add_tuple(batch, tupl, ctx);
  }
  return batch;
}
