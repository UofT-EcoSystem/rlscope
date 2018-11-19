//
// Created by jagle on 11/12/2018.
//

#ifndef DNN_TENSORFLOW_CPP_DEBUG_H
#define DNN_TENSORFLOW_CPP_DEBUG_H

#include "tensorflow/core/platform/logging.h"

#define STR(s) #s

#ifndef ASSERT_EQ
#define MY_ASSERT_EQ(a, b, s) ({ \
  if ((a) != (b)) { \
    LOG(INFO) << "ERROR: " << STR(a) << " " << "==" << " " << STR(b) << ": " << TF_Message(s); \
    assert((a) == (b)); \
  } \
})
#else
#define MY_ASSERT_EQ(a, b, s) ({ \
  ASSERT_EQ(a, b) << TF_Message(s); \
})
#endif

#define MY_ASSERT(t) ({ \
  if (!(t)) { \
    LOG(INFO) << "ERROR: " << STR(t) << " failed. "; \
    assert(t); \
  } \
})

//exit(EXIT_FAILURE);

#endif //DNN_TENSORFLOW_CPP_DEBUG_H
