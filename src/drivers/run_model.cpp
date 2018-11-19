//
// Created by jagle on 11/12/2018.
//

#include "common/debug.h"
#include "model/CarModel.h"
#include "tf/wrappers.h"

#include <gflags/gflags.h>

DEFINE_bool(debug, false, "Debug");
DEFINE_bool(dummy, false, "dummy");
DEFINE_bool(load_model, false, "Load model that was defined and trained in the python version of this script.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "> FLAGS_load_model = " << FLAGS_load_model << std::endl;

  if (FLAGS_dummy) {
    std::cout << "./model" << std::endl;
    exit(EXIT_SUCCESS);
  }

  // WARNING: hyp isn't initialized.
  DQNHyperparameters hyp;
  CarModel model(hyp);
  model.LoadModel();

  return 0;
}
