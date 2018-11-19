//
// Created by jagle on 11/9/2018.
//

#ifndef DNN_TENSORFLOW_CPP_ENVIRONMENT_H
#define DNN_TENSORFLOW_CPP_ENVIRONMENT_H

#include "tensorflow/c/c_api.h"
#include <cassert>

using StateType = float;
using RewardType = float;
// NOTE: vector<bool> doesn't generalize well like other container types...
// In particular, vector<bool>.data() doesn't work...
// I'm not sure why... is it because it's stored as a bit vector?
using DoneVectorType = char;
using DoneType = bool;

template <typename CType>
inline TF_DataType get_tf_type() {
  /* Not sure what TensorFlow type CType belongs to. */
  assert(false);
  return 0;
}

template <>
inline TF_DataType get_tf_type<float>() {
  return TF_FLOAT;
}

template <>
inline TF_DataType get_tf_type<bool>() {
  return TF_BOOL;
}

template <>
inline TF_DataType get_tf_type<int32_t>() {
  return TF_INT32;
}

template <>
inline TF_DataType get_tf_type<int64_t>() {
  return TF_INT64;
}

#endif //DNN_TENSORFLOW_CPP_ENVIRONMENT_H
