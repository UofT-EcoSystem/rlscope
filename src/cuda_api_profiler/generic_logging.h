//
// Created by jagle on 11/13/2019.
//

#ifndef IML_GENERIC_LOGGING_H
#define IML_GENERIC_LOGGING_H

#include "cuda_api_profiler/generic_logging.h"

#include <ostream>

namespace tensorflow {

std::ostream& PrintIndent(std::ostream& out, int indent);

}

#endif //IML_GENERIC_LOGGING_H
