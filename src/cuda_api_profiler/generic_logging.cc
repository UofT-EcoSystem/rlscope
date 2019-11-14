//
// Created by jagle on 11/13/2019.
//

#include "generic_logging.h"

#include <ostream>

namespace tensorflow {

std::ostream &PrintIndent(std::ostream &out, int indent) {
  for (int i = 0; i < indent; i++) {
    out << "  ";
  }
  return out;
}

}
