//
// Created by jgleeson on 2020-05-14.
//

#ifndef CUPTI_SAMPLES_COMMON_UNTIL_H
#define CUPTI_SAMPLES_COMMON_UNTIL_H

// Include lots of files.
#include "common_util/Common.h"
#include "common_util/CommonCuda.cuh"
#include "common_util/my_status.h"
#include "common_util/debug_flags.h"
#include "common_util/defines.h"
#include "common_util/error_codes.h"
#include "common_util/generic_logging.h"
#include "common_util/usec_timer.h"
#include "common_util/env_var.h"
#include "common_util/concurrency.h"

// nvcc bugs: cannot import json.hpp without errors:
// https://github.com/nlohmann/json/issues/1347
#ifndef RLS_IGNORE_JSON
#include "common_util/json.h"
#endif // RLS_IGNORE_JSON

#include "common_util/notify.h"
#include "common_util/logging.h"

#endif //CUPTI_SAMPLES_COMMON_UNTIL_H
