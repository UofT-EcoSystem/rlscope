#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from glob import glob
import time
import pprint
import textwrap
import sys
import re
import os
import fnmatch
import subprocess
from os.path import dirname as _d, basename as _b
from os.path import join as _j, abspath as _P
from os.path import splitext
from io import StringIO

from baselines.deepq.experiments import benchmark_dqn

import py_config

# NOTE:
# - do NOT use the absolute paths from expr_config
# - if you try to, scons won't realize absolute paths == relative paths and refuse to 
#   build stuff silently 
#   (look at paths in "scons --debug=stree" to debug if you suspect this is happening).
_ROOT = "."

# RULES = ['PythonProfileParser']
# Not initialized yet.
RULES = []

AddOption(
  '--src-direc',
  help=textwrap.dedent("""
    Path to look for source files for running --target.
    """))
# https://stackoverflow.com/questions/4109436/processing-multiple-values-for-one-single-option-using-getopt-optparse

# ListVariable('--src-direcs',
#            'Path to look for source files for running --target.',
#            'all',
#            names = "src_direcs",
# # list_of_libs
#            ),

AddOption(
  '--src-direcs',
  help=textwrap.dedent("""
    Path to look for source files for running --target.
    (Comma-separated list)
    """),
  # action='append', type='string',
)

AddOption(
  '--rule',
  help=textwrap.dedent("""
    Rule to run.
  """),
)
AddOption(
  '--list-rules',
  action='store_true',
  help=textwrap.dedent("""
    List rules that can be run.
  """))
AddOption(
  '--rebuild',
  action='store_true',
  help="force summary.csv to be rebuilt, even if they're up to date")
AddOption(
  '--dbg',
  action='store_true',
  help="debug")
AddOption(
  '--print',
  action='store_true',
  help="debug")

# Use this --expr-type if the input/output files are found in the given <directory>.
# NOTE: <directory> is typically a figure name
DIRECTORY_TO_EXPR_TYPE = {
  'execution-times':'expr_execution_times',
}
# Add a "repetition" column for each *.txt file (1.txt -> repetition=1)
DIRECTORY_WITH_REPETITION_COLUMN = set([
  'execution-times',
])

def _get_directories(direcs):
  # direcs = self.opt('src_direcs')
  if direcs is not None:
    if re.search(r',', direcs):
      return ",".split(direcs)
    return direcs
  return direcs

def _get_directory(self):
  direcs = self.directories
  if direcs is not None:
    return direcs
  return self.opt('src_direc')

def timestamp():
  return time.time()

class ProfileParserRunner:
  def __init__(self, ParserKlass):
    self.main = None
    self.py = py_config.BENCH_DQN_PY
    self.ParserKlass = ParserKlass
    self.rule = self.ParserKlass.__name__

  @property
  def direc(self):
    return _get_directory(self)
  @property
  def directories(self):
    return _get_directories(self.opt('src_direcs'))

  def python_cmd_str(self, debug=None):
    python_cmd = " ".join(self.main.python_cmd(debug))
    return python_cmd

  @property
  def debug(self):
      return self.main.debug
  @property
  def should_print(self):
    return self.main.should_print
  @property
  def bench_name(self):
    return self.opt('bench_name')

  def shell_cmd(self, srcs, debug=None):
    direc = self.direc
    rule = self.rule
    py = self.py

    python = self.python_cmd_str(debug=debug)

    # if self.ParserKlass.uses_all_benches():
    #   src_files = self.ParserKlass.as_source_files_group(srcs)
    #   bench_names = src_files.bench_names
    # else:
    # Need to first group srcs into matching directories,
    # then call as_source_files(...) on each group of files.
    src_files = self.ParserKlass.as_source_files(srcs)
    bench_names = src_files.bench_names
    if not self.ParserKlass.uses_all_benches():
      assert len(bench_names) == 1
      bench_name = list(bench_names)[0]
      if bench_name is not None:
        bench_str = "--bench-name {bench_name}".format(bench_name=self.bench_name)
      else:
        bench_str = ""
    else:
      bench_str = ""

    debug_str = ""
    if self.should_print:
      debug_str = "--debug"

    directory_str = None
    if src_files.is_group:
      directory_str = "--directories {directories}".format(
        directories=" ".join(src_files.directories))
    else:
      directory_str = "--directory {direc}".format(
        direc=direc)

    cmdline = '{python} {py} {directory_str} --rule {rule} {bench_str} {debug_str}'.format(**locals())
    return cmdline

  def get_source_files(self):
    src_files = self.ParserKlass.get_source_files(self.direc, debug=self.debug)
    return src_files

  def get_targets(self, src_files, bench_name=None):
    targets = self.ParserKlass.get_targets(src_files, bench_name)
    return targets

  def opt(self, name):
    return self.main.opt(name)

  def __call__(self, target, source, env):

    # Target
    # ['/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/TotalTimeSec.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/TotalTimeSec.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/CppAndGPUTimeSec.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/CppAndGPUTimeSec.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/CppTimeSec.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/CppTimeSec.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/GPUTimeSec.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/GPUTimeSec.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/TheoreticalSpeedup.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/TheoreticalSpeedup.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/PercentTimeInPython.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/PercentTimeInPython.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/PythonTimeSec.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/PythonTimeSec.summary.png',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/PythonOverheadPercent.summary.plot_data.txt',
    #  '/mnt/data/james/clone/dnn_tensorflow_cpp/checkpoints/test_call_c/PythonOverheadPercent.summary.png']
    # Source: []
    srcs = [s.abspath for s in source]
    py = py_config.BENCH_DQN_PY
    rule = self.opt('rule')
    cmd = self.shell_cmd(srcs)
    cmdline = '{cmd} '.format(**locals())
    check_call(cmdline, shell=True, print_cmd=True)

class _Main:
  def __init__(self):
    self.start_build_time = timestamp()
    self._init_env()
    self.check_args()

  def python_cmd(self, debug=None):
    if debug is None:
      debug = self.debug
    if debug:
      return ["python", "-m", "ipdb"]
    else:
      return ["python", "-u"]

  @property
  def debug(self):
    return self.opt('dbg')
  @property
  def should_print(self):
    return self.debug or self.opt('print')

  @property
  def list_rules(self):
    return self.opt('list_rules')
  @property
  def rule(self):
    return self.opt('rule')
  @property
  def bench_name(self):
    return self.opt('bench_name')
  @property
  def direc(self):
    return _get_directory(self)
  @property
  def directories(self):
    return _get_directories(self.opt('src_direcs'))

  def build_all_benches_rule(self, src_files, builder, env_rule):
    sources = src_files.all_sources(all_bench_names=True)
    # targets = builder.get_targets(src_files, all_bench_names=True)
    try:
      pass  # Cmd
      targets = builder.get_targets(src_files)
    except Exception as e:
      import ipdb;
      ipdb.set_trace()
      raise e


    if self.should_print:
      print(textwrap.dedent("""
            > Run rule: 
              rule       = {rule}
              source     = {source}
              target     = {target}
            """.format(
        rule=env_rule,
        source=sources,
        target=targets)))

    self.check_src_files(src_files, builder)
    env_rule(source=sources, target=targets)
    if self.opt('rebuild'):
      self.env.AlwaysBuild(targets)

  def check_src_files(self, src_files, builder):
    if not src_files.has_all_required_paths:
      print(
        textwrap.dedent("""
ERROR: Didn't find all required source files in directory={dir} for parser={parser}
  src_files =
{src_files}
  required_files = 
{required_files}
            """.format(
          dir=src_files.directory,
          parser=builder.ParserKlass.__name__,
          # src_files=str(src_files),
          src_files=textwrap.indent(str(src_files), prefix="  "*2),
          required_files=benchmark_dqn.as_str(builder.ParserKlass.required_source_basename_regexes(), indent=2),
        )))
      Exit(1)

  def build_bench_rule(self, bench_name, src_files, builder, env_rule):
    if self.should_print:
      print("  > bench_name = {bench_name}".format(bench_name=bench_name))

    sources = src_files.all_sources(bench_name)
    if not src_files.has_all_required_paths:
      print(textwrap.dedent("""
            > Skip --rule; couldn't find all require source paths: 
              rule          = {rule}
              bench_name    = {bench_name}
              sources found = {source}
            """.format(
        rule=env_rule,
        bench_name=bench_name,
        source=sources)))
      return

    targets = builder.get_targets(src_files, bench_name)

    if self.should_print:
      print(textwrap.dedent("""
            > Run rule: 
              rule       = {rule}
              bench_name = {bench_name}
              source     = {source}
              target     = {target}
            """.format(
        rule=env_rule,
        bench_name=bench_name,
        source=sources,
        target=targets)))

    # Cannot pass non-path arguments to build-rule:
    # Options:
    # - Force to build all bench_names when we call the script.
    # - Hide additional information in a target path that is never built
    # - Extract bench_name from source paths (pass --srcs)
    #   - Already have code to do this; just need to match it match
    #     over pre-existing files instead of a --directory
    # - Extract bench_name from target paths (pass --targets)
    # env_sources = {'srcs':sources, 'bench_name':bench_name}

    self.check_src_files(src_files, builder)
    env_rule(source=sources, target=targets)
    if self.opt('rebuild'):
      self.env.AlwaysBuild(targets)

  def run(self):
    rules = [self.opt('rule')]

    for rule in rules:
      if self.should_print:
        print("> Rule = {rule}".format(rule=rule))
      env_rule = getattr(self.env, rule)
      builder = getattr(self, rule)
      if self.should_print:
        print("  > get_source_files".format(rule=rule))
      src_files = builder.get_source_files()

      # TODO: src_files might take all bench_names at once; in that case, should invoke the rule ONCE.
      bench_names = src_files.bench_names


      if builder.ParserKlass.uses_all_benches():
        self.build_all_benches_rule(src_files, builder, env_rule)
      else:
        for bench_name in bench_names:
          self.build_bench_rule(bench_name, src_files, builder, env_rule)

  def check_args(self):
    if self.list_rules:
      print("Valid --rule choices:")
      pprint.pprint(RULES, indent=2)
      Exit(0)

    if self.rule is None:
      print("ERROR: --rule must be provided; choices = {rules}".format(rules=RULES))
      Exit(1)

    if self.rule not in RULES:
      print("ERROR: unknown --rule={rule}; choices = {rules}".format(
        rule=self.rule,
        rules=RULES))
      Exit(1)

    if self.rule is not None and self.opt('src_direc') is None and self.opt('src_direcs') is None:
      print("ERROR: --src-direc/--src-direcs must be provided when --rule is provided")
      Exit(1)

  def opt(self, name):
    return self.env.GetOption(name)

  def _add_parser(self, ParserKlassName):
    # self.PythonProfileParser = ProfileParserRunner(benchmark_dqn.PythonProfileParser)
    # self.builders.append(self.PythonProfileParser)
    # BUILDERS['PythonProfileParser'] = Builder(action=self.PythonProfileParser)
    ParserKlass = getattr(benchmark_dqn, ParserKlassName)
    runner = ProfileParserRunner(ParserKlass)
    setattr(self, ParserKlassName, runner)
    self.builders.append(runner)
    self.env_builders[ParserKlassName] = Builder(action=runner)
    RULES.append(ParserKlassName)

  def _init_env(self):
    self.builders = []
    self.env_builders = dict()

    self._add_parser("PythonProfileParser")
    self._add_parser("CUDASQLiteParser")
    self._add_parser("CombinedProfileParser")
    self._add_parser("PlotSummary")

    self.env = Environment(BUILDERS=self.env_builders)

    for builder in self.builders:
      builder.main = self

    for rule in RULES:
      assert hasattr(self.env, rule)

def PhonyTargets(env = None, **kw):
  if not env:
    env = DefaultEnvironment()
  for target, action in kw.items():
    env.AlwaysBuild(env.Alias(target, [], action))

def _print_cmd(cmd):
  if type(cmd) == list:
    cmd_str = " ".join([str(x) for x in cmd])
  else:
    cmd_str = cmd
  print(textwrap.dedent("""
                RUN:
                  cwd = {cwd}
                  cmd = {cmd}
                """.format(
    cwd=os.getcwd(),
    cmd=cmd_str,
  )))

def check_call(cmd, cat_output=True, print_cmd=False, **kwargs):
  """
  :param cmd:
  :param cat_output:
  If True, cat output to
  :param kwargs:
  :return:
  """
  if print_cmd:
    _print_cmd(cmd)
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
  # for c in iter(lambda: process.stdout.read(1), ''):  # replace '' with b'' for Python 3
  ss = StringIO()
  for line in iter(process.stdout.readline, b''):  # replace '' with b'' for Python 3
    if line == b'':
      break
    line = line.decode('utf-8')
    if cat_output:
      sys.stdout.write(line)
    ss.write(line)
  process.wait()
  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, cmd)
  proc = subprocess.CompletedProcess(cmd, process.returncode, stdout=ss.getvalue())
  return proc
  # return subprocess.run(cmd, stdout=sys.stdout, stder=sys.stderr, check=True, **kwargs)

def as_list(proc):
  xs = [x for x in re.split(r'\s*\n', proc.stdout) \
        if x != '']
  return xs

def paths_as_str(paths):
  return ' '.join([str(path) for path in paths])



def main():
  Main = _Main()
  Main.run()

main()

