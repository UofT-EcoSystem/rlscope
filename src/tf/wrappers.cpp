//
// Created by jagle on 11/12/2018.
//

#include "tensorflow/c/c_api.h"

#include "common/debug.h"
#include "tf/wrappers.h"

#include <cassert>
#include <memory>
#include <vector>

TFSession TFSession::LoadSessionFromSavedModel(const std::string& path, const char** tags) {
  TFSession session;

  TF_SessionOptions* opt = TF_NewSessionOptions();
  TF_Buffer* run_options = TF_NewBufferFromString("", 0);
  TF_Buffer* metagraph = TF_NewBuffer();

  TF_Session* sess = TF_LoadSessionFromSavedModel(
      opt, run_options, path.c_str(), tags, 1, session._graph.get(), metagraph, session._status.get());
  session._session.reset(new _TF_Session(sess), SessionDeleter);
  MY_ASSERT_EQ(TF_OK, TF_GetCode(session._status.get()), session._status.get());

  TF_DeleteBuffer(run_options);
  TF_DeleteSessionOptions(opt);
  TF_DeleteBuffer(metagraph);

  return session;
}
