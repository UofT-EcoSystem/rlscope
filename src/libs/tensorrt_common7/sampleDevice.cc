//
// Created by jgleeson on 2020-07-03.
//

#include <mutex>

#include "sampleDevice.h"

namespace sample {

std::mutex TrtCudaGraph::capture_lock;

}
