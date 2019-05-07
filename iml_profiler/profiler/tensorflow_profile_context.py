# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A Context that captures profile and performs profiling/dumping.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

import re
import contextlib
import os
import random
import sys
import threading
import time
import textwrap
import pprint

import tensorflow as tf

from tensorflow.core.profiler import tfprof_log_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile

# from tensorflow.python.profiler import model_analyzer
import iml_profiler.profiler.tensorflow_model_analyzer as model_analyzer
from iml_profiler.profiler.tensorflow_model_analyzer import ENABLE_PRINT_MDL, ERR_NO_PRINT_MDL

from tensorflow.python.util import compat
from tensorflow.python.framework import c_api_util

from iml_profiler import py_config
from iml_profiler.parser.common import *


from iml_profiler.profiler import glbl

# Only allow a single traced session.run(...) call to run at a time.
# By default, tfprof code enforces this.
# Q: Will multiple tracers work now...?
# SINGLE_TRACE_AT_A_TIME = True
SINGLE_TRACE_AT_A_TIME = False

# Global mutex to serialize session.run(...) calls when tracing is enabled.
if SINGLE_TRACE_AT_A_TIME:
  GLOBAL_SESSION_RUN_LOCK = threading.Lock()
else:
  GLOBAL_SESSION_RUN_LOCK = None

THREAD_IDS = set()
THREAD_IDS_LOCK = threading.Lock()
def _check_single_threaded():
  with THREAD_IDS_LOCK:
      tid = threading.get_ident()
      THREAD_IDS.add(tid)
      if len(THREAD_IDS) > 1:
        pprint.pprint({'THREAD_IDS':THREAD_IDS})
        raise RuntimeError("IML: Detected more than 1 ({n}) python-thread; currently we don't support multiple threads.".format(
          n=len(THREAD_IDS),
        ))

WARMUP_STEPS = 10
MAX_TRACED_STEPS = 100

DEBUG = False
# DEBUG = True
DEBUG_THREADS = False
# DEBUG_THREADS = True
# DEBUG_DISABLE_TRACING = True
DEBUG_DISABLE_TRACING = False

RUN_OPTIONS_NO_TRACE = config_pb2.RunOptions(
  trace_level=config_pb2.RunOptions.NO_TRACE)

RUN_OPTIONS_FULL_TRACE = config_pb2.RunOptions(
  trace_level=config_pb2.RunOptions.FULL_TRACE)

SESSION_RUN_CALLS_TRACED = 0
def reset_session_run_calls_traced():
  global SESSION_RUN_CALLS_TRACED
  SESSION_RUN_CALLS_TRACED = 0
def get_session_run_calls_traced():
  global SESSION_RUN_CALLS_TRACED
  return SESSION_RUN_CALLS_TRACED

# def _profiled_init(self, target='', graph=None, config=None):
#   """Overwrites the session.__init__."""
#   self._profiler_init_internal(target, graph, config)  # pylint: disable=protected-access


def _profiled_run(self,
                  fetches,
                  feed_dict=None,
                  options=None,
                  run_metadata=None):
  """Overwrites the session.run()."""
  # pylint: disable=protected-access
  # Count the session steps.
  global SESSION_RUN_CALLS_TRACED

  if DEBUG_THREADS:
      _check_single_threaded()

  profile_context = getattr(self, 'profile_context', None)
  if profile_context is not None:
    profile_context_state = profile_context._new_step()
    step, locked = profile_context_state.__enter__()

    # Inherit the "phase" from the Profiler object when we first call session.run(...).
    if profile_context.phase is None:
      assert glbl.prof.phase is not None
      profile_context.phase = glbl.prof.phase
    elif profile_context.phase != glbl.prof.phase:
      raise RuntimeError("IML: Internal error; detected ProfileContext being used across multiple phases.")

    assert locked
  else:
    step = None
    locked = False

  if DEBUG:
    print(textwrap.dedent("""
    > _profile_run:
      - profile_context = {pctx}
      - step = {step}
      - locked = {locked}
    """.format(
      pctx=profile_context,
      step=step,
      locked=locked,
    )))
    if profile_context is not None:
      print(textwrap.dedent("""
      - not self.profile_context._is_fast_path(step) = {fast_path_bool}
        - self.profile_context._disable = {disable}
        - self.profile_context._trace_all = {trace_all}
      """.format(
          fast_path_bool=not self.profile_context._is_fast_path(step),
          disable=self.profile_context._disable,
          trace_all=self.profile_context._trace_all,
        )))
    # else:
    #   # Q: Why isn't this Session object being traced with a ProfileContext object?
    #   import ipdb; ipdb.set_trace()

  # print("> tfprof debug: {vars}".format(vars={
  #   'locked': locked,
  #   'step': step,
  #   'DEBUG': DEBUG,
  #   'is_fast_path': self.profile_context._is_fast_path(step),
  #   'should_trace': self.profile_context._should_trace(step, self.graph, fetches),
  # }))

  # Fast path if no need for profiling.
  if locked and not self.profile_context._is_fast_path(step):

    if DEBUG and not self.profile_context._should_trace(step, self.graph, fetches):
        print("tfprof> SKIP step={step}".format(step=step))

    # Maybe trace this step.
    if self.profile_context._should_trace(step, self.graph, fetches):
      if DEBUG:
        print("tfprof> with step={step}".format(step=step))
      if self.profile_context._debug:
        sys.stderr.write('debug: tracing step: %d\n' % step)
      # Enable tracing, perform auto profiling or auto dump.
      copy_run_metadata = True
      if not run_metadata:
        copy_run_metadata = False
        run_metadata = config_pb2.RunMetadata()

      if not options:
        options = RUN_OPTIONS_FULL_TRACE
        old_trace_level = options.trace_level
      else:
        old_trace_level = options.trace_level
        options.trace_level = config_pb2.RunOptions.FULL_TRACE

      # tfprof_step = self.profile_context._step
      SESSION_RUN_CALLS_TRACED += 1
      assert profile_context is self.profile_context
      assert step is not None
      sess = self
      # preallocate_tracer(tfprof_step, sess)
      preallocate_tracer(step, sess)

      if DEBUG:
        sess = self
        tracer_is_set = c_api_util.get_is_tracer_set(sess)
        assert tracer_is_set

      start_run_internal_t = time.time()
      ret = self._profiler_run_internal(
          fetches, feed_dict, options, run_metadata)
      end_run_internal_t = time.time()
      start_add_step_t = end_run_internal_t
      if self.profile_context._debug:
        self.profile_context._dump_file(run_metadata, 'run_meta_%d' % step)

      # JAMES NOTE: We don't care about the "graph" anymore when dumping profiling info.
      #
      # if self.profile_context._dump_on_finished:
      #   self.profile_context.graph = self.graph
      # else:
      #   self.profile_context.profiler.graph = self.graph
      #   self.profile_context.profiler.add_step(step, run_metadata)

      end_add_step_t = time.time()
      options.trace_level = old_trace_level

      # if DEBUG:
      #     # For q_backward, if profiling overhead from add_step is the issue, I expect:
      #     #   _profiler_run_internal to take ~ 0.10199415534734727 sec
      #     #   add_step to take ~ 1.196728 - 0.10199415534734727 = 1.0947338 sec
      #     print(textwrap.dedent("""
      #     tfprof> cache stats needed for add_step(step={step})
      #             _profiler_run_internal = {prof_sec} seconds
      #             add_step = {add_step_sec} seconds
      #     """.format(step=step,
      #                prof_sec=end_run_internal_t - start_run_internal_t,
      #                add_step_sec=end_add_step_t - start_add_step_t)))

    else:
      # if DEBUG:
      #   print("tfprof> (1) SKIP step={step}".format(step=step))
      ret = self._profiler_run_internal(fetches, feed_dict, options)

    # Maybe dump profile.
    # self.profile_context._maybe_dump(step)

    # Maybe profile:
    to_profiles = self.profile_context._profile_candidates()
    for to_prof in to_profiles:
      cmd, opts, _ = to_prof
      if self.profile_context._debug:
        sys.stderr.write('debug: profiling %s step: %d\n' % (cmd, step))
      if cmd == 'graph':
        self.profile_context.profiler.profile_graph(opts)
      elif cmd == 'scope':
        self.profile_context.profiler.profile_name_scope(opts)
      elif cmd == 'op':
        self.profile_context.profiler.profile_operations(opts)
      elif cmd == 'code':
        self.profile_context.profiler.profile_python(opts)
      else:
        raise ValueError('Unknown cmd: %s\n' % cmd)

    if profile_context is not None:
      profile_context_state.__exit__(None, None, None)

    return ret

  if profile_context is not None:
    profile_context_state.__exit__(None, None, None)
  # Fast no lock path.
  # if DEBUG:
  #     print("tfprof> (2) SKIP step={step}".format(step=step))
  return self._profiler_run_internal(
      fetches, feed_dict, options, run_metadata)
  # pylint: enable=protected-access


class ProfileContext(object):
  """A Context that captures RunMetadata and performs profiling.

  ```python
    # Trace steps 100~200, profile at [150, 200] and dump profile at 200.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=range(100, 200, 3),
                                          dump_steps=[200]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling('op', opts, [150, 200])
      train_loop().

    # Tracing only.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
      # Run train/eval loop for at least few hundred steps. Profiles will be
      # dumped to train_dir. Use web UI or command line to do profiling.
      train_loop().

    # When session object is available, do explicit trace, profile and dump.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=[],
                                          dump_steps=[]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.trace_next_step()
      _ = session.run(train_op)
      pctx.profiler.profile_operations(options=opts)
  ```

  Args:
    profile_dir: Directory to store profiles.
    trace_steps: A list of session run steps to trace. If None, use
        pre-defined steps.
    dump_steps: A list of steps to dump the profile to `profile_dir`. If None,
        use pre-defined steps.
    enabled: If false, everything is disabled with minimal overhead. It allows
        user to only enable profiling when needed.
    debug: If true, also dumps the raw trace RunMetadata text file to
        profile_dir. And print debugging message. Useful for bug report.
    dump_on_finished: If true, ignore dump_steps, and avoid calling profiler.add_step while
        profiling to avoid adding profiling overhead.
        Instead, save all the data so we can dump it all at the end of profililng.
  """

  def __init__(self,
               profile_dir=None,
               trace_steps=None,
               dump_steps=None,
               enabled=True,
               debug=False,
               dump_on_finished=False,
               # process_name=None,
               # phase=None,
               trace_all=False,
               session=None,
               phase=None):
    # self.process_name = process_name
    # self.phase = phase
    self._sess = session
    self.phase = phase
    assert self._sess is not None
    self._trace_all = trace_all
    self._disable = False

    self._enabled = enabled
    if not self._enabled:
      return

    self._dump_on_finished = dump_on_finished

    self._step_data = []

    self._debug = debug
    # if not profile_dir:
    #   raise ValueError('Must have a directory for profile.\n')
    self._profiler_dir = profile_dir

    if trace_steps is None:
      self._trace_steps = set()
      self._auto_tracing = True
    else:
      if len(trace_steps) > MAX_TRACED_STEPS:
        raise ValueError('Only support tracing up to 100 steps.\n')
      self._trace_steps = set(trace_steps[:])
      self._auto_tracing = False

    if dump_steps is None:
      self._dump_steps = set([MAX_TRACED_STEPS])
    else:
      self._dump_steps = set(dump_steps[:])

    self._rng = random.Random(111)
    self._fetched = set()
    self._slow_path_steps = self._dump_steps | self._trace_steps
    self._trace_next_step = False
    self._dump_next_step = False
    self._step = 0
    self._traced_steps = 0
    self._auto_profiles = []
    self._profiler = None
    if SINGLE_TRACE_AT_A_TIME:
      self._lock = GLOBAL_SESSION_RUN_LOCK
    else:
      self._lock = None

    print("Ported profiler")

  def add_auto_profiling(self, cmd, options, profile_steps):
    """Traces and profiles at some session run steps.

    Args:
      cmd: The profiling commands. (i.e. scope, op, python, graph)
      options: The profiling options.
      profile_steps: A list/set of integers. The profiling command and options
          will be run automatically at these integer steps. Each step is
          a session.run.
    """
    if not self._enabled:
      return
    self._auto_profiles.append((cmd, options, profile_steps[:]))
    self._slow_path_steps |= set(profile_steps)
    self._trace_steps |= set(profile_steps)

  def disable_tracing(self):
    self._disable = True

  def enable_tracing(self):
    self._disable = False

  @property
  def profiler(self):
    """Returns the current profiler object."""
    if not self._enabled:
      return None
    if not self._profiler:
      self._profiler = model_analyzer.Profiler(ops.get_default_graph())
    return self._profiler

  def trace_next_step(self):
    """Enables tracing and adds traces to profiler at next step."""
    if not self._enabled:
      return
    self._trace_next_step = True
    self._slow_path_steps.add(self._step)

  def dump_next_step(self):
    """Enable tracing and dump profiles at next step."""
    if not self._enabled:
      return
    self._dump_next_step = True
    self._slow_path_steps.add(self._step)

  def _is_fast_path(self, step):
    if self._disable:
      return True

    if self._trace_all:
      return False

    if step in self._slow_path_steps:
      return False
    # When user doesn't set the tracing steps explicitly, auto decide it.
    if (self._auto_tracing and step > WARMUP_STEPS and
        self._traced_steps <= MAX_TRACED_STEPS):
      return False
    return True

  def set_trace_all(self, trace_all):
    self._trace_all = trace_all

  def _should_trace(self, step, graph, fetches):
    """Whether should do tracing at current step."""
    if DEBUG_DISABLE_TRACING:
      return False

    if self._disable:
      return False

    if self._trace_all:
      return True

    if self._traced_steps > MAX_TRACED_STEPS:
      return False
    # Check user-set tracing steps.
    if step in self._trace_steps or self._trace_next_step:
      self._traced_steps += 1
      return True

    # If no user-set tracing steps set and passes warm up steps, auto trace.
    if self._auto_tracing and step > WARMUP_STEPS:
      # If the fetches have not been seen before, trace it.
      with graph.as_default():
        fetch_names = [f.name for f in
                       session._FetchMapper.for_fetch(fetches).unique_fetches()]  # pylint: disable=protected-access
      fetch_name = '-'.join(sorted(fetch_names))
      if self._debug:
        sys.stderr.write('debug: trace fetches: %s\n' % fetch_name)
      if fetch_name not in self._fetched:
        self._fetched.add(fetch_name)
        self._traced_steps += 1
        return True
      # If the trace coverage is low, does some random tracing.
      if (self.profiler._coverage < 0.5 and step < MAX_TRACED_STEPS and  # pylint: disable=protected-access
          self._rng.randint(0, 10) < 2):
        self._traced_steps += 1
        return True
    return False

  def _maybe_dump(self, step):
    """Maybe dump the profile file."""
    if self._dump_on_finished or not (step in self._dump_steps or self._dump_next_step):
      return
    self._dump(step)

  def _dump(self, step):
    if self._debug:
      sys.stderr.write('debug: dumping file at step: %d\n' % step)
    assert self._profiler_dir is not None
    if not gfile.Exists(self._profiler_dir):
      gfile.MakeDirs(self._profiler_dir)

    if DEBUG:
      print("tfprof> Dump @ step={step}".format(step=step))
    filename = os.path.join(compat.as_bytes(self._profiler_dir),
                            compat.as_bytes('profile_%d' % step))
    self.profiler._write_profile(filename)  # pylint: disable=protected-access

  def _dump_file(self, pb, basename):
    assert self._profiler_dir is not None
    if not gfile.Exists(self._profiler_dir):
      gfile.MakeDirs(self._profiler_dir)
    with gfile.Open(os.path.join(self._profiler_dir, basename), 'w') as f:
      f.write('%s' % pb)

  @contextlib.contextmanager
  def _new_step(self):
    if self._lock is not None:
      acquired = self._lock.acquire(False)
    else:
      # We haven't acquired a lock.
      # However, the caller uses this flag to decide whether to profile.
      # So just set it to True.
      acquired = True
    # if DEBUG:
    #     print("> tfprof: step={step}, locked={locked}".format(
    #       step=self._step,
    #       locked=acquired))
    if DEBUG:
      print("> tfprof: step={step}".format(
        step=self._step,
      ))
    yield (self._step, acquired)
    # if DEBUG:
    #   print("> tfprof: AFTER step={step}, locked={locked}".format(
    #     step=self._step,
    #     locked=acquired))
    self._step += 1
    self._trace_next_step = False
    self._dump_next_step = False
    if self._lock is not None and acquired:
      self._lock.release()

  def _profile_candidates(self):
    to_profile = []
    for auto_prof in self._auto_profiles:
      _, _, prof_steps = auto_prof
      if self._step in prof_steps:
        to_profile.append(auto_prof)
    return to_profile

  def add_step(self, step, graph, run_metadata, copy_run_metadata):
    if copy_run_metadata:
      # WARNING: copying this protobuf could be expensive...
      raise NotImplementedError("Need to make a copy of protobuf in case it gets reused...")
    self._step_data.append((step, graph, run_metadata))

  def __enter__(self):
    if self._enabled:
      old_pctx = getattr(self._sess, 'profile_context', None)
      assert old_pctx is None
      self._sess.profile_context = self
    return self

  def clear(self):
    assert self.phase is not None
    sess = self._cur_session(allow_none=True)

    # if process_name is None:
    #   process_name = self.process_name

    # if phase is None:
    #   phase = self.phase

    if sess is None:
      print(("> WARNING: tfprof didn't trace sessions for cmd:\n"
             "  cmd: {cmd}").format(cmd=" ".join(sys.argv)))
      return

    if DEBUG and self._disable:
      print("> SKIP pctx.clear(); --iml-disable")
      return

    c_api_util.clear_trace_data(sess)

  def dump(self, dump_path, process_name):
    assert self.phase is not None
    sess = self._cur_session(allow_none=True)

    # if process_name is None:
    #   process_name = self.process_name

    # if phase is None:
    #   phase = self.phase

    if sess is None:
        print(("> WARNING: tfprof didn't trace sessions for cmd:\n"
               "  cmd: {cmd}").format(cmd=" ".join(sys.argv)))
        return

    if DEBUG and self._disable:
        print("> SKIP pctx.dump(dump_path={path}, process={proc}); --iml-disable".format(
          path=dump_path,
          proc=process_name))
        return

    self.trace_data = c_api_util.get_trace_data(sess)
    byte_size = self.trace_data.ByteSize()
    print("> trace_data size = {b} bytes".format(b=byte_size))
    print("> tfprof steps: ")
    steps = sorted(list(self.trace_data.traced_steps.keys()))
    pprint.pprint({'len(steps)':len(steps)})

    profile_proto_builder = ProfileProtoBuilder(process_name, self.phase)
    for step, traces in self.trace_data.traced_steps.items():
      for run_meta in traces.traces:
          profile_proto_builder.add_run_meta(step, run_meta)

    if len(self.trace_data.traced_steps) == 0:
      # No sess.run() calls were traced.
      print(
        "> WARNING: tfprof didn't capture any session.run(...) calls!\n",
        "Maybe try setting --iml-start-measuring-call lower (e.g. 0)?")
    #   dump_step = 0
    # else:
    #   dump_step = max(self.trace_data.traced_steps.keys())

    # assert self._profiler_dir is not None
    # profile_path = _j(self._profiler_dir, 'profile_{d}'.format(d=dump_step))
    # profile_path = self.get_dump_path()
    size_bytes = profile_proto_builder.size_bytes()
    print("> Dump tfprof ({b} bytes) to: {path}".format(
      b=size_bytes, path=dump_path))
    profile_proto_builder.dump(dump_path)

  def get_dump_path(self):
    assert self.dump_basename is not None
    assert self._profiler_dir is not None
    profile_path = _j(self._profiler_dir, self.dump_basename)
    return profile_path

  def set_dump_basename(self, dump_basename):
    assert self._profiler_dir is not None
    self.dump_basename = dump_basename

  def _cur_session(self, allow_none=False):
    if self._sess is not None:
      return self._sess

    # sess = tf.get_default_session()
    # if sess is None and not allow_none:
    raise RuntimeError(
      "Couldn't find current session; you either need to call Profiler.set_session(sess), "
      "or do \"with sess.as_default():\"")

    # return sess

  # def set_session(self, sess):
  #   self._sess = sess

  def _old_dump_custom_tf(self):
    """
    Used to use this for dumping tfprof data, however stopped using it since it results in memory corruption
    (not sure why!).
    Besides memory corruption, it's inefficient to re-serialize tracing data we just read from C++ anyways
    via get_trace_data().

    Basically we are adding each run_meta_data to C++, and C++ is responsible for writing the proto file.
    """
    sess = self._cur_session()
    self.trace_data = c_api_util.get_trace_data(sess)
    byte_size = self.trace_data.ByteSize()
    print("> trace_data size = {b} bytes".format(b=byte_size))
    print("> tfprof steps: ")
    steps = sorted(list(self.trace_data.traced_steps))
    pprint.pprint({'steps':steps})
    graph = self.graph
    dump_step = 0
    for step, run_metadata in self.trace_data.traced_steps.items():
      dump_step = max(dump_step, step)
      self.profiler._graph = graph
      self.profiler.add_step(step, run_metadata)
    self._dump(dump_step)

  def __exit__(self, exec_type, exec_value, exec_tb):
    if not self._enabled:
      return

    # Q: How much of ProfilerContext is single-session specific?
    # (i.e. do we need a ProfilerContext for each active-session?)
    # - print_mdl.DeleteProfiler below deletes a singleton "TF_Stat" object from C++,
    #   if it exists.
    # - However, C++ code is OK to call DeleteProfiler multiple times
    #   (only first delete counts)
    #
    # - tf.Profiler object creates singleton object.
    #   Q: What if we call create multiple times?
    #   Sadly, that will cause a memory leak.
    #   So, we need to wrap tf.Profiler to remove print_mdl.NewProfiler calls.
    if self._dump_on_finished:
        self.dump()

    if ENABLE_PRINT_MDL:
      print_mdl.DeleteProfiler()
    # setattr(session.BaseSession, 'run', self.old_run)
    # setattr(session.BaseSession, '__init__', self.old_init)
    # setattr(session.BaseSession, '_profiler_run_internal', None)
    # setattr(session.BaseSession, '_profiler_init_internal', None)
    # setattr(session.BaseSession, 'profile_context', None)
    assert self._sess.profile_context is self
    self._sess.profile_context = None


class ProfileProtoBuilder:
  """
  Given a bunch of RunMetadata from traced session.run(...) calls, condense them all
  into the final tfprof output protobuf:
  i.e. ProfileProto

  NOTE: This code used to be done in C++, but there is no need
  since it's more annoying to modify.
  """
  def __init__(self, process_name, phase):
    self.process_name = process_name
    self.phase = phase

    self.profile_proto = tfprof_log_pb2.ProfileProto()
    self.profile_proto.process_name = self.process_name
    self.profile_proto.phase = self.phase

    self.next_node_id = 0
    self.name_to_id = dict()

  def add_run_meta(self, step, run_meta):
    if step not in self.profile_proto.steps:
      self.profile_proto.steps.extend([step])

    for dev_stat in run_meta.step_stats.dev_stats:
      dev = dev_stat.device.lower()
      for node_stat in dev_stat.node_stats:
        name = node_stat.node_name
        m = re.search(r'(?P<name>.*):', name)
        if m:
          name = m.group('name')

        if name in self.name_to_id:
          node_id = self.name_to_id[name]
        else:
          node_id = self.next_node_id
          self.next_node_id += 1
          self.name_to_id[name] = node_id

        has_node = node_id in self.profile_proto.nodes
        profile_node = self.profile_proto.nodes[node_id]
        if not has_node:
          profile_node.name = name

        # TFGraphNode::AddStepStat
        # Skip the "IsCanonicalDevice" crap...

        # ExecStep::AddTimeStats

        if node_stat.all_start_micros > 0:
          op_end_rel_micros = max(1, node_stat.op_end_rel_micros)

          start_us = node_stat.all_start_micros
          end_us = op_end_rel_micros

          exec_profile = profile_node.execs[step]
          # exec_time = exec_profile[dev]

          if IsGPUTime(dev):
            exec_time = exec_profile.accelerator_execs[dev]
          elif IsCPUTime(dev):
            exec_time = exec_profile.cpu_execs[dev]

          tupl = exec_time.times.add()
          tupl.int64_values.extend([start_us, end_us])
          # exec_time = tfprof_log_pb2.ExecTime()
          # exec_time.times.int64_values.extend([start_us, end_us])
          # exec_time.times.extend([]).int64_values.extend([start_us, end_us])
          # if IsGPUTime(dev):
          #   exec_profile.accelerator_execs[dev].extend([exec_time])
          # elif IsCPUTime(dev):
          #   exec_profile.cpu_execs[dev].extend([exec_time])

  def size_bytes(self):
    return self.profile_proto.ByteSize()

  def dump(self, path):
    with open(path, 'wb') as f:
      f.write(self.profile_proto.SerializeToString())

SETUP_DONE = False
def setup(allow_skip=False):
  global SETUP_DONE
  if allow_skip and SETUP_DONE:
    return
  assert not SETUP_DONE

  setup_wrap_Session()

  SETUP_DONE = True

def setup_wrap_Session():
  old_run = getattr(session.BaseSession, 'run', None)
  if not old_run:
    raise errors.InternalError(None, None, "BaseSession doesn't have a run method.")
  # old_init = getattr(session.BaseSession, '__init__', None)

  # Overwrite tf.Session.run(...) and tf.Session.__init__(...)
  setattr(session.BaseSession, 'run', _profiled_run)
  # setattr(session.BaseSession, '__init__', _profiled_init)

  # Keep old tf.Session.run(...) and tf.Session.__init__(...)
  setattr(session.BaseSession, '_profiler_run_internal', old_run)
  # setattr(session.BaseSession, '_profiler_init_internal', old_init)

  # elif not old_init:
  #   raise errors.InternalError(None, None,
  #                              'BaseSession misses __init__ method.')
  # elif getattr(session.BaseSession, '_profiler_run_internal', None):
  #   raise errors.InternalError(None, None,
  #                              'Already in context or context not cleaned.')
  # elif getattr(session.BaseSession, '_profiler_init_internal', None):
  #   raise errors.InternalError(None, None,
  #                              'Already in context or context not cleaned.')

def now_in_usec():
  time_us = print_mdl.NowInUsec()
  return time_us

def preallocate_tracer(step, sess):
  if DEBUG:
    print("> PREALLOCATE_TRACER for step={step}".format(step=step))
  assert sess is not None
  # sess = tf.get_default_session()
  # print("> preallocate_tracer: _session = {session}".format(session=sess._session))
  print_mdl.TF_PreallocateTracer(sess._session, step)

def IsGPUTime(device):
  return re.search('stream:all', device)

def IsCPUTime(device):
  return re.search(".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+", device)
