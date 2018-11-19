//
// Created by jagle on 11/13/2018.
//

#ifndef DNN_TENSORFLOW_CPP_CARMODEL_H
#define DNN_TENSORFLOW_CPP_CARMODEL_H

#include "model/model.h"

//#include "tensorflow/c/c_test_util.h"
//#include "tensorflow/core/platform/logging.h"

#include "tensorflow/c/c_api.h"
//#include "tensorflow/c/c_api_internal.h"

//#include <sys/types.h>
//#include <sys/stat.h>
//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//
//#include <cstdlib>
//#include <string>
//#include <vector>
//#include <cassert>
//#include <utility>
//#include <iostream>

#include "model/data_set.h"
#include "model/model.h"

#include "tf/wrappers.h"


class CarModel : public Model {
public:

  TF_Operation* _loss_op;
  TF_Operation* _step_op;
  TF_Operation* _predictions_op;
  TF_Operation* _features_op;
  TF_Operation* _labels_op;

  DataSet _data_set;

  CarModel(DQNHyperparameters& hyp, const std::string model_path = std::string(""), bool debug = false) :
      Model(hyp, model_path, debug),
      _data_set(
          csv_dir_path() + "/",
          CSV_BASENAME)
  {
    _variables = {
        "dense/kernel/read",
        "dense/bias/read",
        "dense_1/kernel/read",
        "dense_1/bias/read",
        "predictions/kernel/read",
        "predictions/bias/read",
    };
  }

  virtual void InitOps();

};


#endif //DNN_TENSORFLOW_CPP_CARMODEL_H
