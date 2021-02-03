"""
RL-Scope user API.

Users of RL-Scope should only interact with symbols in this module,
and should avoid using symbols elsewhere since they are subject to change.

.. todo::
  Provide documentation of typical usage.

Attributes
----------
prof
  Singleton RL-Scope profiler object that gets created when :py:func:`rlscope.api.handle_rlscope_args` is called.
  Users should interact with this object instead of instantiating their own profiler object manually.
"""
__all__ = []

# NOTE: only include Profiler so it appears in sphinx documentation for this module.
from rlscope.profiler.profilers import \
  Profiler, \
  fix_gflags_rlscope_args, \
  add_rlscope_arguments, \
  rlscope_argv_and_env, \
  click_add_arguments
__all__.extend([
  'Profiler',
  'fix_gflags_rlscope_args',
  'add_rlscope_arguments',
  'rlscope_argv_and_env',
  'click_add_arguments',
])

# from rlscope.profiler.util import \
#   args_to_cmdline
# __all__.extend([
#   'args_to_cmdline',
# ])

# Managed by rlscope.profiler.glbl
prof = None
from rlscope.profiler.glbl import \
  handle_rlscope_args, \
  handle_gflags_rlscope_args, \
  handle_click_rlscope_args, \
  init_session
__all__.extend([
  'prof',
  'handle_rlscope_args',
  'handle_gflags_rlscope_args',
  'handle_click_rlscope_args',
  # 'init_session',
])

from rlscope.profiler.rlscope_logging import \
  logger
__all__.extend([
  'logger',
])

from rlscope.profiler.clib_wrap import \
  wrap_module, unwrap_module, \
  register_wrap_module, \
  wrap_entire_module, unwrap_entire_module
__all__.extend([
  'wrap_module',
  'unwrap_module',
  'register_wrap_module',
  'wrap_entire_module',
  'unwrap_entire_module',
])

from rlscope import \
  __version__, \
  version
__all__.extend([
  '__version__',
  'version',
])

# from rlscope.scripts.utilization_sampler import \
#   util_sampler
# __all__.extend([
#   'util_sampler',
# ])
