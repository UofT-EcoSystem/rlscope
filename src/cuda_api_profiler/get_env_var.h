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
static const std::string IML_GPU_HW_METRICS_DEFAULT = "sm__warps_active.avg.pct_of_peak_sustained_active+";
std::vector<std::string> get_IML_GPU_HW_METRICS(boost::optional<std::string> user_value);

bool env_is_on(const char* var, bool dflt, bool debug);

bool is_yes(const char* env_var, bool default_value);
bool is_no(const char* env_var, bool default_value);

}

#endif //IML_GET_ENV_VAR_H
