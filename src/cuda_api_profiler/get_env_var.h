//
// Created by jagle on 8/12/2019.
//

#ifndef IML_GET_ENV_VAR_H
#define IML_GET_ENV_VAR_H

#include <cstdlib>
#include "common_util.h"

namespace rlscope {


int ParseFloat(const char* str, size_t size);
float ParseEnvFloatOrDefault(const char* env_name, float user_value, float dflt);

// TF_CUDA_API_PRINT_EVERY_SEC
static const float TF_CUDA_API_PRINT_EVERY_SEC_DEFAULT = 5.0;
float get_TF_CUDA_API_PRINT_EVERY_SEC(float user_value);

// IML_SAMPLE_EVERY_SEC
static const float IML_SAMPLE_EVERY_SEC_DEFAULT = 1.0;
float get_IML_SAMPLE_EVERY_SEC(float user_value);

bool env_is_on(const char* var, bool dflt, bool debug);

bool is_yes(const char* env_var, bool default_value);
bool is_no(const char* env_var, bool default_value);

}

#endif //IML_GET_ENV_VAR_H
