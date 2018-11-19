//
// Created by jagle on 11/13/2018.
//

#include "common/debug.h"
#include "model/model.h"
#include "tf/wrappers.h"

#include "model/CarModel.h"

void CarModel::InitOps() {
  _loss_op = lookup_op("loss", "loss/loss");
  _step_op = lookup_op("step", "step/step");
  _predictions_op = lookup_op("predictions", "outputs/predictions/Tanh");
  _features_op = lookup_op("features", "inputs/features");
  _labels_op = lookup_op("labels", "inputs/labels");
}
