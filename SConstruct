#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from glob import glob
import time
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

RULES = ['PythonProfileParser']

AddOption('--srcs-direc',
    help=textwrap.dedent("""
    Path to look for source files for running --target.
    """))
AddOption('--rule',
          help=textwrap.dedent("""
    Rule to run.
    """),
          choices=RULES)
AddOption('--rebuild',
      action='store_true',
      help="force summary.csv to be rebuilt, even if they're up to date")
AddOption('--dbg',
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
    return self.opt('srcs_direc')

  def python_cmd_str(self, debug=None):
    python_cmd = " ".join(self.main.python_cmd(debug))
    return python_cmd

  @property
  def debug(self):
    return self.main.debug

  def shell_cmd(self, debug=None):
    direc = self.direc
    rule = self.rule
    py = self.py
    python = self.python_cmd_str(debug=debug)
    cmdline = '{python} {py} --directory {direc} --rule {rule}'.format(**locals())
    return cmdline

  def get_source_files(self):
    src_files = self.ParserKlass.get_source_files(self.direc)
    return src_files

  def get_targets(self, src_files):
    targets = self.ParserKlass.get_targets(src_files)
    return targets

  def opt(self, name):
    return self.main.opt(name)

  def __call__(self, target, source, env):
    # srcs = ' '.join([s.abspath for s in source])
    py = py_config.BENCH_DQN_PY
    rule = self.opt('rule')
    cmd = self.shell_cmd()
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

  def run(self):
    rules = [self.opt('rule')]
    for rule in rules:
      # import ipdb; ipdb.set_trace()
      env_rule = getattr(self.env, rule)
      builder = getattr(self, rule)
      # import ipdb; ipdb.set_trace()
      src_files = builder.get_source_files()
      if src_files.has_all_required_paths:
        targets = builder.get_targets(src_files)

        # builder.run()
        if self.debug:
          print(textwrap.dedent("""
            > Run rule: 
              rule   = {rule}
              source = {source}
              target = {target}
            """.format(rule=env_rule, source=src_files.all_sources, target=targets)))
        env_rule(source=src_files.all_sources, target=targets)
        if self.opt('rebuild'):
          self.env.AlwaysBuild(targets)

  def check_args(self):
    if self.opt('rule') is None:
      print("ERROR: --rule must be provided; choices = {rules}".format(rules=RULES))
      Exit(1)

    if self.opt('rule') is not None and self.opt('srcs_direc') is None:
      print("ERROR: --srcs-direc must be provided when --rule is provided")
      Exit(1)

  def opt(self, name):
    return self.env.GetOption(name)

  def _init_env(self):
    self.builders = []

    BUILDERS = dict()

    self.PythonProfileParser = ProfileParserRunner(benchmark_dqn.PythonProfileParser)
    self.builders.append(self.PythonProfileParser)
    BUILDERS['PythonProfileParser'] = Builder(action=self.PythonProfileParser)

    self.CUDASQLiteParser = ProfileParserRunner(benchmark_dqn.CUDASQLiteParser)
    self.builders.append(self.CUDASQLiteParser)
    BUILDERS['CUDASQLiteParser'] = Builder(action=self.CUDASQLiteParser)

    self.env = Environment(BUILDERS=BUILDERS)

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

