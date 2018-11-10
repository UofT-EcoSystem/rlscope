//
// Created by jagle on 11/9/2018.
//

#ifndef DNN_TENSORFLOW_CPP_REPLAYBUFFER_H
#define DNN_TENSORFLOW_CPP_REPLAYBUFFER_H

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

template <class TupleType, class FuncAddTupleCtxType>
class ReplayBuffer {
public:
  using FuncAddTupleType = std::function<
      void(std::unordered_map<std::string, boost::any>& batch,
          const TupleType& tupl,
          FuncAddTupleCtxType& ctx)>;

  ReplayBuffer(int seed, size_t size, FuncAddTupleType func_add_tuple);

  void Add(TupleType tupl);
  std::unordered_map<std::string, boost::any> Sample(size_t batch_size, FuncAddTupleCtxType ctx);

  std::vector<TupleType> _storage;
  size_t _next_idx;
  FuncAddTupleType _func_add_tuple;
  int _seed;
  size_t _seen;
  size_t _size;
  std::default_random_engine _generator;
};


#endif //DNN_TENSORFLOW_CPP_REPLAYBUFFER_H
