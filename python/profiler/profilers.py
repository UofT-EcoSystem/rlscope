import cProfile, pstats, io
import codecs
import json
import os

from profiler import cudaprofile

class CUDAProfiler:
  def __init__(self):
    # NOTE: CUDA profiling output has already been specified when this script was launched.
    # self.profile_basename = profile_basename
    self.already_enabled = False

  def start(self):
    # NOTE: we assume the CUDA
    self.already_enabled = cudaprofile.is_profiler_enabled()
    if not self.already_enabled:
      cudaprofile.start()

  def stop(self):
    if not self.already_enabled:
      cudaprofile.stop()

  def __enter__(self):
    self.start()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()

class PythonProfiler:
  """
  Run profiler on python code block.

  with profiler(python_profile_basename="some/profile"):
      # profile this code
  """
  def __init__(self, python_profile_basename, record_call_times=True, clock='monotonic_clock'):
    self.profile = cProfile.Profile()
    self.python_profile_basename = python_profile_basename
    self.clock = clock
    assert self.clock in ['monotonic_clock']
    self.use_cycle_counter = (self.clock == 'cycle_counter')
    self.use_monotonic_clock = (self.clock == 'monotonic_clock')
    self.record_call_times = record_call_times

    assert not ( self.use_cycle_counter and self.use_monotonic_clock )

    if self.use_cycle_counter:
      self.profile.make_use_cycle_counter()

    if self.use_monotonic_clock:
      self.profile.make_use_monotonic_clock()

    if self.record_call_times:
      self.profile.make_record_call_times()

  def __enter__(self):
    self.start()

  def start(self):
    self.profile.enable()

  def stop(self):
    self.profile.disable()
    # self.dump()

  def dump(self):
    # sortby = ('calls', 'filename', 'name', 'line')
    sortby = ('tottime', 'filename', 'line', 'name')

    os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)
    with open(self._stats_path, mode='w') as f:
      ps = pstats.Stats(self.profile, stream=f).sort_stats(*sortby)
      if self.record_call_times:
        call_times = ps.call_times
      ps.print_stats()

    if self.record_call_times:

      # Tuple keys are not OK; convert to strings.
      new_call_times = dict()
      for func_tuple, times in call_times.items():
        func = func_std_string(func_tuple)
        new_call_times[func] = times
      json.dump(new_call_times,
                codecs.open(self._call_times_path, mode='w', encoding='utf-8'),
                sort_keys=True, indent=4)

    os.makedirs(os.path.dirname(self._prof_path), exist_ok=True)
    ps.dump_stats(self._prof_path)

  @property
  def _prof_path(self):
    return "{base}.prof".format(base=self.python_profile_basename)

  @property
  def _call_times_path(self):
    return "{base}.call_times.json".format(base=self.python_profile_basename)

  @property
  def _stats_path(self):
    return "{base}.txt".format(base=self.python_profile_basename)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()

# Taken from cProfile's Lib/pstats.py;
# func_name is 3-tuple of (file-path, line#, function_name)
# e.g.
#   ('Lib/test/my_test_profile.py', 259, '__getattr__'):
def func_std_string(func_tuple): # match what old profile produced
  if func_tuple[:2] == ('~', 0):
    # special case for built-in functions
    name = func_tuple[2]
    if name.startswith('<') and name.endswith('>'):
      return '{%s}' % name[1:-1]
    else:
      return name
  else:
    path, lineno, func_name = func_tuple
    new_path = path
    # Full path is useful for vim when visiting lines; keep it.
    # new_path = re.sub(r'.*/site-packages/', '', new_path)
    # new_path = re.sub(r'.*/clone/', '', new_path)
    # new_path = re.sub(r'.*/lib/python[^/]*/', '', new_path)
    return "{path}:{lineno}({func_name})".format(
      path=new_path,
      lineno=lineno,
      func_name=func_name,
    )
