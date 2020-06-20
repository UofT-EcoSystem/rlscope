//
// Created by jgleeson on 2020-06-19.
//

#include "debug_flags.h"

#include "env_var.h"

namespace rlscope {

static inline void DumpStacktrace(size_t skip_frames, bool snippet) {
  backward::StackTrace st;
  const size_t MAX_STACKFRAMES = 32;
  // Skip stackframes added by this callframe.
  skip_frames += 3;
  st.load_here(MAX_STACKFRAMES);
// Last 4 frames are always related to backward.hpp or logging.cc.
// Skip those frames; make the latest frame the LOG(FAIL) or DCHECK failure.
  size_t idx;
  if (st.size() < skip_frames) {
// Print the whole thing.
    idx = 0;
  } else {
// Skip the last 4 frames.
    idx = skip_frames;
  }
  st.load_from(st[idx].addr, MAX_STACKFRAMES);
  backward::Printer p;
  p.snippet = snippet;
  p.print(st);
}

void dbg_breakpoint(const std::string& name, const char* file, int lineno) {
  std::cerr << "";
}
void _dbg_breakpoint(const std::string& name, const char* file, int lineno) {
  if (!is_yes("DBG_BREAKPOINT", false)) {
    return;
  }
  if (FEATURE_BREAKPOINT_DUMP_STACK) {
    DumpStacktrace(1, true);
  }
  std::cerr << "[ HIT BREAKPOINT \"" << name << "\" @ " << file << ":" << lineno << " ]" << std::endl;
  dbg_breakpoint(name, file, lineno);
}
void _dbg_breakpoint_with_stacktrace(const std::string& name, const char* file, int lineno) {
  if (!is_yes("DBG_BREAKPOINT", false)) {
    return;
  }
  DumpStacktrace(1, true);
  std::cerr << "[ HIT BREAKPOINT \"" << name << "\" @ " << file << ":" << lineno << " ]" << std::endl;
  dbg_breakpoint(name, file, lineno);
}

} // namespace rlscope
