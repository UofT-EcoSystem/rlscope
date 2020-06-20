//
// Created by jagle on 8/12/2019.
//

#ifndef IML_GET_ENV_VAR_H
#define IML_GET_ENV_VAR_H

#include <boost/optional.hpp>

#include <cstdlib>
#include "common_util.h"

namespace rlscope {


// TF_CUDA_API_PRINT_EVERY_SEC
static const float TF_CUDA_API_PRINT_EVERY_SEC_DEFAULT = 5.0;
float get_TF_CUDA_API_PRINT_EVERY_SEC(boost::optional<float> user_value);

// IML_SAMPLE_EVERY_SEC
static const float IML_SAMPLE_EVERY_SEC_DEFAULT = 1.0;
float get_IML_SAMPLE_EVERY_SEC(boost::optional<float> user_value);

// IML_GPU_HW_CONFIG_PASSES
static const int IML_GPU_HW_CONFIG_PASSES_DEFAULT = 1;
int get_IML_GPU_HW_CONFIG_PASSES(boost::optional<int> user_value);

// IML_GPU_HW_METRICS
std::vector<std::string> get_IML_GPU_HW_METRICS(boost::optional<std::string> user_value);

}

#endif //IML_GET_ENV_VAR_H
