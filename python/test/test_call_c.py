import argparse
import textwrap
from glob import glob
import codecs
import json
import time
import os
import subprocess
import re
import sys
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e

from baselines.deepq.simple_refactor import CUDAProfiler, profiler as python_profiler

from test import py_interface

import py_config


class TestCallC:
  def __init__(self, args, parser):
    self.args = args
    self.parser = parser
    self.lib = py_interface.PythonInterface()

    self.python_profiler = python_profiler(python_profile_basename=self.python_profile_path)
    self.cuda_prof = CUDAProfiler()
    self.profile_time_sec = 0.

    self.time_slept_gpu_sec = 0.
    self.time_slept_cpp_sec = 0.
    self.time_run_python_sec = 0.

  @property
  def python_profile_path(self):
    return _j(self.directory, "python_profile")

  @property
  def directory(self):
    return self.args.directory

  @property
  def debug(self):
    return self.args.debug

  def init(self):
    args = self.args
    self.lib.call_c()
    if self.debug and _e(self.args.gpu_clock_freq_json):
      self.load_json()
      self.lib.set_gpu_freq_mhz(self.gpu_mhz)
    else:
      self._gpu_mhz = self.lib.guess_gpu_freq_mhz()
      self.dump_json()
    print("GPU mhz = {mhz}".format(mhz=self.gpu_mhz))

  def dump_json(self):
    print("> Dump GPU clock frequency data to: {path}".format(path=self.args.gpu_clock_freq_json))
    self.gpu_clock_freq_data = {
      'gpu_mhz':self._gpu_mhz,
    }
    do_dump_json(self.gpu_clock_freq_data, self.args.gpu_clock_freq_json)

  def load_json(self):
    print("> Load GPU clock frequency from: {path}".format(path=self.args.gpu_clock_freq_json))
    self.gpu_clock_freq_data = load_json(self.args.gpu_clock_freq_json)

  @property
  def gpu_mhz(self):
    return self.gpu_clock_freq_data['gpu_mhz']

  def run_gpu(self):
    args = self.args
    if self.debug:
      print("> Running on GPU for {sec} seconds".format(sec=args.gpu_time_sec))
    self.time_slept_gpu_sec += self.lib.gpu_sleep(args.gpu_time_sec)

  def run_cpp(self):
    args = self.args
    if self.debug:
      print("> Running in CPP for {sec} seconds".format(sec=args.gpu_time_sec))
    self.time_slept_cpp_sec += self.lib.run_cpp(args.cpp_time_sec)

  def run_python(self):
    args = self.args
    if self.debug:
      print("> Running inside python for {sec} seconds".format(sec=args.python_time_sec))
    start_t = time.time()
    time.sleep(args.python_time_sec)

    # NOTE: This creates huge profiler output because of a lot of function calls...
    # instead just sleep.
    #
    # while True:
    #   end_t = time.time()
    #   total_sec = end_t - start_t
    #   if total_sec >= args.python_time_sec:
    #     break
    end_t = time.time()
    total_sec = end_t - start_t
    self.time_run_python_sec += total_sec

  def run(self):
    args = self.args

    self.init()

    print(textwrap.dedent("""
    > Running {r} repetitions, {i} iterations, each iteration is:
      Run in python for {python_sec} seconds
      Run in C++ for {cpp_sec} seconds
      Run in GPU for {gpu_sec} seconds
    """.format(
      r=args.repetitions,
      i=args.iterations,
      python_sec=args.python_time_sec,
      cpp_sec=args.cpp_time_sec,
      gpu_sec=args.gpu_time_sec,
    )))

    # TODO: try repetitions as well.
    for r in range(args.repetitions):
      print("> Repetition {r}".format(r=r))
      self.enable_profiling()
      for i in range(args.iterations):
        self.run_python()
        self.run_cpp()
        self.run_gpu()
      self.disable_profiling()

    results_json = _j(self.directory, "test_call_c.json")
    print("> Dump test_call_c.py results @ {path}".format(path=results_json))
    results = {
      'time_gpu_sec':self.time_slept_gpu_sec,
      'time_python_sec':self.time_run_python_sec,
      'time_cpp_sec':self.time_slept_cpp_sec,
      'time_profile_sec':self.profile_time_sec,
    }
    do_dump_json(results, results_json)

    print("> Dump python profiler output...")
    start_python_profiler_dump = time.time()
    self.python_profiler.dump()
    end_python_profiler_dump = time.time()
    print("  Took {sec} seconds".format(sec=end_python_profiler_dump - start_python_profiler_dump))

  def enable_profiling(self):
    # if self.profile_cuda:
    # nvprof reports the average time over all the repetitions.
    # All the ncalls for the functions called should be divisible by 10 (=repetitions).
    if self.debug:
      print("  > Start CUDA profiler")
    self.cuda_prof.start()

    # if self.profile_python:
    if self.debug:
      print("> Start python profiler")
    self.python_profiler.start()

    self.profile_start_sec = time.time()

  def disable_profiling(self):
    self.profile_end_sec = time.time()
    self.profile_time_sec += self.profile_end_sec - self.profile_start_sec

    # if self.profile_python:
    self.python_profiler.stop()
    if self.debug:
      print("> Stop python profiler")

    # if self.profile_cuda:
    self.cuda_prof.stop()
    if self.debug:
      print("> Stop CUDA profiler")

  def reinvoke_with_nvprof(self):
    run_with_nvprof(self.parser, self.args)

# Default time period for Python/C++/GPU.
DEFAULT_TIME_SEC = 5
# DEFAULT_TIME_SEC_DEBUG = 1
DEFAULT_TIME_SEC_DEBUG = 5
def main():
  parser = argparse.ArgumentParser(textwrap.dedent(
    """
    Test profiling scripts to make sure we correctly measure time spent in Python/C++/GPU.
    """))
  parser.add_argument("--debug", action='store_true',
                      help=textwrap.dedent("""
                      Run quickly.
                      """))
  parser.add_argument("--gpu-time-sec",
                      help=textwrap.dedent("""
                      Time to spend in GPU.
                      5 seconds (default)
                      """))
  parser.add_argument("--cpp-time-sec",
                      help=textwrap.dedent("""
                      Time to spend inside C++.
                      5 seconds (default)
                      """))
  parser.add_argument("--python-time-sec",
                      help=textwrap.dedent("""
                      Time to spend inside Python. 
                      5 seconds (default)
                      """))
  parser.add_argument("--directory",
                      help=textwrap.dedent("""
                      Where to store results.
                      """))
  parser.add_argument("--nvprof-enabled",
                      action='store_true',
                      help=textwrap.dedent("""
                        Internal use only; 
                        used to determine whether this python script has been invoked using nvprof.
                        If it hasn't, the script will re-invoke itself with nvprof.
                        """))
  parser.add_argument("--nvprof-logfile",
                      help=textwrap.dedent("""
                        Internal use only; 
                        output file for nvprof.
                        """))
  parser.add_argument("--gpu-clock-freq-json",
                      help=textwrap.dedent("""
                        Internal use only.
                        """))
  parser.add_argument("--iterations",
                      type=int,
                      help=textwrap.dedent("""
                        --num-calls = --iterations * --repetitions
                        """),
                      default=1)
  parser.add_argument("--repetitions",
                      type=int,
                      help=textwrap.dedent("""
                        --num-calls = --iterations * --repetitions
                        """),
                      default=1)
  # parser.add_argument("--num-calls",
  #                     help=textwrap.dedent("""
  #                       --num-calls = --iterations * --repetitions
  #                       """),
  #                     default=1)
  args = parser.parse_args()
  num_calls = args.iterations * args.repetitions

  if args.directory is None:
    args.directory = _j(py_config.ROOT, "checkpoints", "test_call_c")

  if args.gpu_clock_freq_json is None:
    args.gpu_clock_freq_json = _j(args.directory, "gpu_clock_freq.json")

  for attr in ['gpu_time_sec', 'cpp_time_sec', 'python_time_sec']:
    if args.debug:
      default_time_sec = DEFAULT_TIME_SEC_DEBUG
    else:
      default_time_sec = DEFAULT_TIME_SEC
    setattr(args, attr, default_time_sec)

  config_path = _j(args.directory, "config.json")
  dump_config(config_path,
              num_calls=num_calls,
              iterations=args.iterations,
              repetitions=args.repetitions)

  test_call_c = TestCallC(args, parser)

  if not args.nvprof_enabled:
    test_call_c.reinvoke_with_nvprof()
    return

  args.num_calls = num_calls
  test_call_c.run()

def run_with_nvprof(parser, args):
  print("> Reinvoking script with nvprof.")

  nvprof_logfile = _j(args.directory, "nvidia.nvprof_logfile.txt")
  if _e(nvprof_logfile):
    os.remove(nvprof_logfile)

  nvprof_sqlite_file = _j(args.directory, "nvidia.nvprof")
  if _e(nvprof_sqlite_file):
    # Nvprof fails if the output file already exists.
    os.remove(nvprof_sqlite_file)
  os.makedirs(_d(nvprof_logfile), exist_ok=True)
  nvprof_args = ["nvprof",
                 "-o", nvprof_sqlite_file,
                 "--log-file", nvprof_logfile,
                 "--profile-from-start", "off"]
  cmdline = args_to_cmdline(parser, args)
  argv_exec = nvprof_args + cmdline + [
    "--nvprof-enabled",
    "--nvprof-logfile", nvprof_logfile
  ]

  print_cmd(argv_exec)
  subprocess.run(argv_exec, stdout=sys.stdout, stderr=sys.stderr, check=True)

def args_to_cmdline(parser, args):
  """
  # To convert args namespace into cmdline:

  if args.option == True and option in parser:
      cmdline.append(--option)
  elif args.option == False and option in parser:
      pass
  elif type(args.option) in [int, str, float]:
      cmdline.append(--option value)
  elif type(args.open) in [list]:
      cmdline.append(--option elem[0] ... elem[n])
  else:
      raise NotImplementedError
  """

  def option_in_parser(parser, option):
    return parser.get_default(option) is not None

  def optname(option):
    return "--{s}".format(s=re.sub(r'_', '-', option))

  py_script_idx = 0
  while not re.search(r'\.py$', sys.argv[py_script_idx]):
    py_script_idx += 1
  extra_opts = []
  if hasattr(args, 'debug') and args.debug:
    extra_opts.extend(["-m", "ipdb"])
  cmdline = [sys.executable] + extra_opts + sys.argv[0:py_script_idx+1]
  for option, value in args.__dict__.items():
    opt = optname(option)
    if value is None:
      continue

    if type(value) == bool:
      if value and option_in_parser(parser, option):
        cmdline.extend([opt])
      else:
        pass
    elif type(value) in [int, str, float]:
      cmdline.extend([opt, value])
    elif type(value) in [list] and len(value) > 0:
      cmdline.extend([opt])
      cmdline.extend(value)
    else:
      raise NotImplemented

  return [str(x) for x in cmdline]

def dump_config(path, **kwargs):
  config = dict()
  defaults = {
    'clock': "monotonic_clock",
    'device_name': None,
    'impl_name': None,
    # For tensorflow: r'(?:built-in.*pywrap_tensorflow)'
    'c_lib_func_pyprof_pattern':"CLIB__.*",
    'discard_first_sample':False,
    'num_calls':kwargs['num_calls'],
    'iterations':kwargs['iterations'],
    'repetitions':kwargs['repetitions'],
    # 'bench_name_labels': {
    #   ...
    # },
  }
  config = dict(defaults)
  config.update(kwargs)
  do_dump_json(config, path)

def print_cmd(cmd):
  print(textwrap.dedent("""
                RUN:
                  cwd = {cwd}
                  cmd = {cmd}
                """.format(
    cwd=os.getcwd(),
    cmd=" ".join([str(x) for x in cmd]),
  )))

def load_json(path):
  with codecs.open(path, mode='r', encoding='utf-8') as f:
    data = json.load(f)
    # data = fixup_json(data)
    return data

def do_dump_json(data, path):
  os.makedirs(_d(path), exist_ok=True)
  json.dump(data,
            codecs.open(path, mode='w', encoding='utf-8'),
            sort_keys=True, indent=4,
            skipkeys=False)

if __name__ == '__main__':
    main()
