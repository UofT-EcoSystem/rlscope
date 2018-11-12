//
// Created by jagle on 11/12/2018.
//

#ifndef DNN_TENSORFLOW_CPP_ALGORITHM_H
#define DNN_TENSORFLOW_CPP_ALGORITHM_H


#include <memory>

#include "model/model.h"

class DQNAlgorithm {
public:
  DQNAlgorithm(std::shared_ptr<Model> model);
  void Run();
  void Setup();
  void NextIter();

  std::shared_ptr<Model> _model;
};


#endif //DNN_TENSORFLOW_CPP_ALGORITHM_H
