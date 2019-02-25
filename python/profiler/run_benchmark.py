# DQN specific, disable for now
# import gym
gym = None

from io import StringIO

from os import environ as ENV
# import baselines
# from baselines import deepq
# from baselines.deepq.simple_refactor import DQN, NAME_TO_MICRO, CUDA_MICROBENCH_NAMES, CUDA_MICROBENCH_NAME_TO_PRETTY, BENCH_NAME_REGEX, BENCH_AND_TOTAL_NAME_REGEX, BENCH_NAME_ORDER, BENCH_TYPES, MICRO_DEFAULTS, ARGPARSE_DEFAULTS, ARGPARSE_DEFAULTS_DEBUG, BENCH_NAME_TO_PRETTY, \
#     get_nvprof_name, get_c_only_name, pyprof_func_std_string, get_microbench_path, get_microbench_basename, is_microbench_path
import re

# pip install py-cpuinfo
import cpuinfo


from glob import glob
import subprocess
import pprint
import argparse
import numpy as np
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b
# import tensorflow as tf
# from tensorflow.python.client import device_lib as tf_device_lib
import sys
import textwrap
import os
import copy

import py_config

from profiler import profilers

from parser.nvprof import CUDASQLiteParser
from parser.tfprof import TotalTimeParser, TraceEventsParser
from parser.pyprof import PythonProfileParser, PythonFlameGraphParser, PythonProfileTotalParser
from parser.plot import TimeBreakdownPlot, PlotSummary, CombinedProfileParser, CategoryOverlapPlot, UtilizationPlot
from parser.db import SQLParser

from parser.common import *

# Deprecated:
# CUDAProfileParser, PythonProfileTotalParser, CUDASQLiteParser, CombinedProfileParser,
PARSER_KLASSES = [PythonProfileParser, PythonFlameGraphParser, PlotSummary, TimeBreakdownPlot, CategoryOverlapPlot, UtilizationPlot, TotalTimeParser, TraceEventsParser, SQLParser]
PARSER_NAME_TO_KLASS = dict((ParserKlass.__name__, ParserKlass) \
                            for ParserKlass in PARSER_KLASSES)

ARGPARSE_DEFAULTS = {
    # 'repetitions':3,
    'repetitions':10,
    'iterations':1000,
    'warmup_iters':10,
}
ARGPARSE_DEFAULTS_DEBUG = copy.deepcopy(ARGPARSE_DEFAULTS)
ARGPARSE_DEFAULTS_DEBUG.update({
    'iterations': 100,
})

def test_glob_targets(parser, args):
    PythonProfileParser.test_targets()

def _device_proto_as_dict(device_proto):
    # For GPU's
    # ipdb> device_proto.physical_device_desc
    # 'device: 0, name: Quadro P4000, pci bus id: 0000:04:00.0, compute capability: 6.1'

    # For CPU's
    # ipdb> device_proto
    # name: "/device:CPU:0"
    # device_type: "CPU"
    # memory_limit: 268435456
    # locality {
    # }
    # incarnation: 11976653475402273625

    m = re.search(r'device: (?P<device>\d+), name: (?P<name>.*), pci bus id: (?P<pci_bus_id>[^,]+), compute capability: (?P<compute_capability>.*)',
                  device_proto.physical_device_desc)
    device = int(m.group('device'))
    name = m.group('name')
    return {"device_number":device, "device_name":name}

def get_available_cpus():
    local_device_protos = tf_device_lib.list_local_devices()
    device_protos = [x for x in local_device_protos if x.device_type != 'GPU']
    assert len(device_protos) == 1
    cpu = cpuinfo.get_cpu_info()
    # 'brand': 'Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz',
    device_dict = {
        'device_name':cpu['brand'],
        'device_number':0,
    }
    return [device_dict]
    # device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    # return device_dicts

def get_available_gpus():
    local_device_protos = tf_device_lib.list_local_devices()
    device_protos = [x for x in local_device_protos if x.device_type == 'GPU']
    device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    return device_dicts

def test_get_gpus(parser, args):
    gpus = get_available_gpus()
    cpus = get_available_cpus()
    pprint.pprint({'gpus':gpus, 'cpus':cpus})

def parse_profile(rule, parser, args):

    ParserKlass = PARSER_NAME_TO_KLASS[rule]

    if issubclass(ParserKlass, ProfilerParserCommonMixin):
        return old_parse_profile(rule, parser, args)

    if args.directories is not None:
        directories = list(args.directories)
    else:
        directories = [args.directory]

    # if args.rules is None:
    #     args.rules = []

    if len(directories) > 1:
        print("ERROR: expected a single --directory")
        sys.exit(1)

    directory = directories[0]

    parser_klass_kwargs = dict(args.__dict__)
    parser_klass_kwargs.update({
        'directory': directory,
        # 'debug': args.debug,
    })
    parser = ParserKlass(**parser_klass_kwargs)
    try:
        parser.run()
    except ParserException as e:
        print(str(e))
        sys.exit(1)

def old_parse_profile(rule, parser, args):

    ParserKlass = PARSER_NAME_TO_KLASS[rule]

    if args.directories is not None:
        directories = list(args.directories)
    else:
        directories = [args.directory]

    src_files = ParserKlass.get_source_files(directories)
    if not src_files.has_all_required_paths:
        print(
            textwrap.dedent("""
ERROR: Didn't find all required source files in directory={dir} for parser={parser}
  src_files =
{src_files}
  required_files = 
{required_files}
            """.format(
                dir=args.directory,
                parser=ParserKlass.__name__,
                # src_files=str(src_files),
                src_files=textwrap.indent(str(src_files), prefix="  "*2),
                required_files=as_str(ParserKlass.required_source_basename_regexes(), indent=2),
            )))

        sys.exit(1)

    if ParserKlass.uses_all_benches():
        expr = ParserKlass(parser, args, src_files)
        expr.run()
    else:
        bench_names = src_files.bench_names
        if len(bench_names) == 0:
            bench_names = [NO_BENCH_NAME]

        if args.bench_name is not None:
            if args.bench_name not in bench_names:
                print(textwrap.dedent("""
                ERROR: Didn't find --bench-name={b} in --directory={d}
                """.format(
                    b=args.bench_name,
                    d=args.directory,
                )))
                sys.exit(1)
            expr = ParserKlass(parser, args, src_files, bench_name=args.bench_name)
            expr.run(args.bench_name)
        else:
            for bench_name in bench_names:
                expr = ParserKlass(parser, args, src_files, bench_name=bench_name)
                expr.run(bench_name)

def main():

    # from test import test_plot
    # test_plot.test_plot_position()
    # return

    # from test import test_plot
    # test_plot.test_subplot()
    # return

    parser = argparse.ArgumentParser("benchmark DQN")

    parser.add_argument("--benchmarks",
                        choices=list(BENCH_NAME_ORDER),
                        action='append', default=[],
                        )
    parser.add_argument("--clock",
                        choices=['monotonic_clock'],
                        default='monotonic_clock',
                        help=textwrap.dedent("""
                        Clock type to use inside python profiler.
                        """))
    parser.add_argument("--benchmark-type",
                        choices=list(BENCH_TYPES),
                        default='dqn',
                        )
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help=textwrap.dedent("""
                        For CUDA microbenchmarks, which GPU to run benchmarks on.
                        """))
    parser.add_argument("--repetition-time-limit-sec",
                        type=float,
                        default=10.,
                        # default=100.,
                        help=textwrap.dedent("""
                        Minimum amount of time for it to take to run a single repetition.
                        """))
    parser.add_argument("--rules",
                        nargs='+', default=[],
                        # action='append', default=[],
                        help='rule to run')
    parser.add_argument("--bench-name",
                        # default=NO_BENCH_NAME,
                        help='bench_name to use for rule to run')
    parser.add_argument("--test-plot",
                        action='store_true',
                        help='test some plotting functions')
    parser.add_argument("--test-glob-targets",
                        action='store_true',
                        help='test some globbing functions')
    parser.add_argument("--test-get-gpus",
                        action='store_true',
                        help='test getting available TensorFlow gpus')
    parser.add_argument("--debug",
                        action='store_true',
                        help='debug')
    parser.add_argument("--op-name",
                        help='Only handle <op_name> (instead of all op_names)')
    parser.add_argument("--debug-ops",
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug where you've appropriately decorated you ML codebase 
                        with set_operation/end_operation annotations. 
                        
                        In particular, this will cause UtilizationPlot to show time captured by 
                        prof.start()/prof.start() but NOT captured by a set_operation/end_operation. 
                        
                        This time may or may not be worth capturing (you must decide).
                        (i.e. you may or may not decide not to capture some one-time initialization 
                        that happens just after prof.start())
                        """))
    parser.add_argument("--debug-memoize",
                        action='store_true',
                        help=textwrap.dedent("""
                        Pickle/unpickle intermediate results to allow quick re-runs of Plot's.
                        """))
    parser.add_argument("--debug-single-thread",
                        action='store_true',
                        help=textwrap.dedent("""
                        Run any multiprocessing stuff using a single thread for debugging.
                        """))
    parser.add_argument("--overlaps-event-id", type=int,
                        help=textwrap.dedent("""
                        TraceEventsParser: dump events that overlap with this one.
                        """))
    parser.add_argument("--no-profile-cuda",
                        action='store_true',
                        help=textwrap.dedent("""
                        DON'T run nvprof.
                        If enabled, only collect nvprof profiling data during benchmark repetitions.
                        """))
    parser.add_argument("--no-discard-first-sample",
                        action='store_true',
                        help=textwrap.dedent("""
                        (see --profile-cuda)
                        The first sample of a call to a CUDA API function has been known to take longer than all remaining calls.
                        I suspect this is profiler related overhead (given we've already done warmup runs before 
                        recording, and the only difference on the fist sample being that the profiler is on).
                        To correct for this
                        
                        e.g. of anomaly I'm talking about:
                        
                            Type      | Time(%) | Avg                 | Std                  | Std/Avg(%) | Call# | Name          | Sample# | Time
                            API calls | 18.16%  | 744.1750360000001us | 22.734390363072784ms | 3054.98%   | 0     | cudaLaunch    | 0       | 719.309048ms
                            API calls | 18.16%  | 744.1750360000001us | 22.734390363072784ms | 3054.98%   | 0     | cudaLaunch    | 1       | 16.817us
                            API calls | 18.16%  | 744.1750360000001us | 22.734390363072784ms | 3054.98%   | 0     | cudaLaunch    | 2       | 16.058us
                            ...
                            API calls | 18.34%  | 751.518926us        | 22.734840524645147ms | 3025.19%   | 1     | cudaLaunch    | 0       | 719.330608ms
                            ...
                            API calls | 18.44%  | 755.5107290000001us | 22.735680628346746ms | 3009.31%   | 2     | cudaLaunch    | 0       | 719.3610540000001ms
                            ...
                            API calls | 18.02%  | 738.5409140000002us | 22.736257068592472ms | 3078.54%   | 7     | cuEventRecord | 0       | 719.3622639999999ms
                        
                        """))
    parser.add_argument("--nvprof-enabled",
                        action='store_true',
                        help=textwrap.dedent("""
                        Internal use only; 
                        used to determine whether this python script has been invoked using nvprof.
                        If it hasn't, the script will re-invoke itself with nvprof.
                        """))
    # parser.add_argument("--c-only",
    #                     action='store_true',
    #                     help=textwrap.dedent("""
    #                     Benchmark just the C++ portion of the tensorflow call,
    #                     so we can see how much time python-c_only code takes up.
    #                     """))
    parser.add_argument("--checkpointing",
                        action='store_true',
                        help=textwrap.dedent("""
                        Plot checkpointing-related benchmarks.
                        """))
    parser.add_argument("--training-iteration",
                        action='store_true',
                        help=textwrap.dedent("""
                        Show "Training iteration" column in plot.
                        Useful for a sanity check:
                        i.e.
                        sum(individual training-loop operations) = time(Training iteration)
                        """))
    parser.add_argument("--env-type",
                        default="PongNoFrameskip-v4")
    parser.add_argument("--directory")
    parser.add_argument("--direc")
    parser.add_argument("--subdir")
    parser.add_argument("--replace", action='store_true', help="delete old checkpoint before training")
    parser.add_argument("--replace-parse", action='store_true', help="redo any parsing/plotting")
    parser.add_argument("--repetitions",
                        type=int,
                        # default=3,
                        )
    parser.add_argument("--iterations",
                        type=int,
                        # default=1000,
                        )
    parser.add_argument("--no-dynamic-iterations",
                        action='store_true',
                        )
    parser.add_argument("--compress-num-checkpoints",
                        type=int,
                        # default=1000,
                        )
    parser.add_argument("--warmup-iters",
                        type=int,
                        # default=10,
                        )
    parser.add_argument("--python-profile",
                        help=textwrap.dedent("""
                        Python profile output; convert from cycle to seconds (if its in cycles!)
                        """))
    parser.add_argument('--plot', help="Just plot results, don't run any benchmarks.",
                        action='store_true')
    parser.add_argument('--directories',
                        help=textwrap.dedent("""
                        Directories containing a "microbenchmark" folder (or the microbenchmark folder itself, 
                        which must contain microbenchmark.json).
                        
                        These are the bars that will be included in our plot.
                        """),
                        nargs='+')
    parser.add_argument('--plot-labels',
                        help=textwrap.dedent("""
                        Labels for use for --directories
                        (see --plot-summary)
                        Labels default to $(basename --directories) otherwise.
                        """),
                        nargs='+')
    parser.add_argument('--prettify-python-profile', help="Just pretty python profile, don't run any benchmarks.",
                        action='store_true')
    parser.add_argument('--prettify-cuda-profile', help="Just pretty cuda profile, don't run any benchmarks.",
                        action='store_true')
    parser.add_argument('--show',
                        action='store_true')
    args = parse_args(parser)

    if len(args.rules) > 0:
        for rule in args.rules:
            print("> Rule = {rule}".format(
                rule=rule))
            parse_profile(rule, parser, args)
        return

    global tf
    global tf_device_lib
    import tensorflow as tf
    from tensorflow.python.client import device_lib as tf_device_lib

    if args.test_plot:
        import test.test_plot
        test.test_plot.test_grouped_stacked_bar_plot(parser, args)
        return

    if args.test_glob_targets:
        test_glob_targets(parser, args)
        return

    if args.test_get_gpus:
        test_get_gpus(parser, args)
        return

    if 'dqn' == args.benchmark_type:
        expr = BenchmarkDQN(parser, args)
        expr.run()
        return

    # if 'cuda' == args.benchmark_type:
    #     expr = BenchmarkCUDA(parser, args)
    #     expr.run()
    #     return

    raise NotImplementedError("Not sure how to run --benchmark-type={typ}".format(
        typ=args.benchmark_type))

def get_bench_name(basename, allow_none=False):
    # m = re.search(r'\b(?P<bench>{regex})\b'.format(regex=BENCH_AND_TOTAL_NAME_REGEX), basename)
    m = re.search(r'\b(?P<bench>{regex})\b'.format(regex=BENCH_NAME_REGEX), basename)
    if not m:
        if allow_none:
            return NO_BENCH_NAME
            # Should we return None here?
            # return None
        assert m
    if m.group('bench') is None:
        return NO_BENCH_NAME
    return m.group('bench')

def get_bench_names(paths):
    return [get_bench_name(_b(path)) for path in paths]

def glob_json_files(search_dirs, json_basename_glob, ignore_missing=False, add_dirs=False):
    json_dirs = []

    def _maybe_add(glob_str):
        json_files = glob(glob_str)
        if len(json_files) != 0:
            if add_dirs:
                json_dirs.extend([_d(path) for path in json_files])
            else:
                json_dirs.extend(json_files)
            return True
        return False
    for direc in search_dirs:

        glob_str = _j(direc, json_basename_glob)
        if _maybe_add(glob_str):
            continue

        glob_str = _j(direc, "microbenchmark", json_basename_glob)
        if _maybe_add(glob_str):
            continue

        msg = ("{direc} does not point to a benchmark result "
               "directory (couldn't find {base})").format(direc=direc, base=json_basename_glob)
        if not ignore_missing:
            raise RuntimeError(msg)
        else:
            print(msg)

    json_dirs = unique(json_dirs)

    return json_dirs

class BenchJsonReader:
    """
    json_glob e.g.
    "*.breakdown.json"

    json_format_str e.g.
    "{bench}.breakdown.json".format(bench=bench)
    """
    def __init__(self, json_glob, json_format_str,
                 # ignore_bench_names=['total']
                 ):
        self.json_glob = json_glob
        self.json_format_str = json_format_str
        # self.ignore_bench_names = set(ignore_bench_names)

    def _bench_data_path(self, direc, bench):
        assert os.path.isdir(direc)
        base = self.json_format_str.format(bench=bench)
        path = _j(direc, base)
        return path

    def read_bench_data(self, direc, bench):
        assert os.path.isdir(direc)
        path = self._bench_data_path(direc, bench)
        bench_data = load_json(path)
        return bench_data

    def _profile_paths(self, direc):
        profile_paths = [path for path \
                         in glob("{direc}/{glob}".format(direc=direc, glob=self.json_glob))]
        profile_paths = [path for path in profile_paths \
                         if get_bench_name(_b(path), allow_none=True) is not None]
        return profile_paths

    def all_profile_paths(self, search_dirs):
        directories = glob_json_files(search_dirs, self.json_glob,
                                      ignore_missing=True,
                                      add_dirs=True)

        profile_paths = []
        for direc in directories:
            profile_paths.extend(self._profile_paths(direc))
        return profile_paths

    def _get_bench_names(self, direc):
        assert os.path.isdir(direc)
        profile_paths = self._profile_paths(direc)
        bench_names = []
        pretty_bench_names = []
        for profile_path in profile_paths:
            bench_name = get_bench_name(_b(profile_path))
            # m = re.search(r'^python_profile\.(?P<bench_name>{bench})\.python_overhead\.pyprof\.json$'.format(bench=BENCH_NAME_REGEX), _b(profile_path))
            # bench_name = m.group('bench_name')
            bench_names.append(bench_name)
            pretty_bench_name = BENCH_NAME_TO_PRETTY.get(bench_name, bench_name)
            pretty_bench_names.append(pretty_bench_name)

        return bench_names, pretty_bench_names

    def each_direc_bench_data(self, search_dirs, ignore_missing=False):
        directories = glob_json_files(search_dirs, self.json_glob,
                                      ignore_missing=ignore_missing,
                                      add_dirs=True)
        for direc in directories:
            bench_names, pretty_bench_names = self._get_bench_names(direc)
            # device = self.direc_to_label[direc]
            for bench, pretty_bench in zip(bench_names, pretty_bench_names):
                data = self.read_bench_data(direc, bench)
                yield data, direc, bench, pretty_bench

def glob_files(directory, basename_glob, basename_negate_re=None):
    """
    profile_paths = <args.directory>/python_profile*.txt exists and not matches *call_times*:

    :param directory:
    :param basename_glob:
    :param basename_negate_re:
    :return:
    """
    paths = [path for path \
             in glob("{dir}/{glob}".format(dir=directory, glob=basename_glob)) \
             if not re.search(r'\.pretty\.', _b(path)) and (
                     basename_negate_re is None or not re.search(basename_negate_re, _b(path)))]
    return paths

"""
Q: What is this code ACTUALLY doing?
If profile_path = <args.directory>/python_profile*.txt exists and not matches *call_times*:
    bench_name = re.match(r"*.q_forward.*", profile_path)
    PythonProfileParser(profile_path, bench_name)
    
NOTE: This code already assumes the files belong to a dqn benchmark.
We need to modify this code to pass ProfilerParserKlass any microbenchmark.json files.
Would be better if ProfilerParserKlass had a static method that searched for src_files given a directory.
Would be nice if ProfileParserKlass could:

- Given a directory, find all "source files" needed to run ProfilerParserKlass.
  SEE: ProfilerParserKlass.get_source_files(direc)
  
- Given a root dir, recursively find subdirectories that have inputs.
  SEE: ProfilerParserKlass.find_source_directories(root_dir)

PSEUDOCODE:

  def ProfilerParserKlass.find_source_directories(root_dir):
      for direc in $(find $root_dir -type d):
          ProfilerParserKlass.get_source_files(direc)
          
  def ProfilerParserKlass.get_required_basename_regexes():
      return [...]
      
  def ProfilerParserKlass.get_optional_basename_regexes():
      return [...]
      
  def ProfilerParserKlass.get_source_files(direc):
      req_regexes = ProfilerParserKlass.get_required_basename_regexes()
      if all(re.search(regex) for regex in req_regexes):
      opt_paths = [path for path in os.listdir(direc) \
                   if any(re.search(regex, path) for regex in req_regexes)]
      
      opt_regexes = ProfilerParserKlass.get_optional_basename_regexes()
      opt_paths = [path for path in os.listdir(direc) \
                   if any(re.search(regex, path)]
      
      src_paths = req_paths + opt_paths
      
   def ProfilerParserKlass.get_target_files(src_files):
       return [...]
    
"""

class BenchmarkDQN:
    def __init__(self, parser, args):
        self.args = args
        self.parser = parser

        self.CUDA_PROFILE_BASENAME_GLOB = "nvidia.*.nvprof.txt"
        self.CUDA_PROFILE_SQLITE_GLOB = "nvidia.*.nvprof"
        self.PYTHON_PROFILE_GLOB = "python_profile*.txt"
        self.PYTHON_PROFILE_NEGATE_RE = r'call_times'

        self.env = gym.make(args.env_type)

    def postprocessing(self):
        # self.combine_profiles()
        # self.plot_benchmarks()
        pass

    def run(self):
        args = self.args
        parser = self.parser

        should_run_benchmarks = not args.plot and not args.prettify_python_profile and not args.prettify_cuda_profile

        if should_run_benchmarks and ( args.profile_cuda and not args.nvprof_enabled ):
            for bench_name in args.benchmarks:
                profilers.run_with_nvprof(args.directory, parser, args,
                                          bench_name=bench_name)
            self.postprocessing()
            return

        if should_run_benchmarks:
            # If we don't JUST want to plot our results.
            for bench_name in args.benchmarks:
                self.run_benchmark(bench_name)

        if not self.args.nvprof_enabled:
            # Let the parent script that re-launched itself do postprocessing.
            # Important since the cuda profile file is still being written to if
            # self.nvprof_enabled is True.
            self.postprocessing()

        return

    def plot_summary(self):
        plot_summary = PlotSummary(self.parser, self.args)

        # Glob all the time_breakdown json files.
        # Read them into json.
        # Pass them to time_breakdown.add_json_data
        readers = [
            BenchJsonReader("*.breakdown.json", "{bench}.breakdown.json"),
            BenchJsonReader("*.python_overhead.pyprof.json", "python_profile.{bench}.python_overhead.pyprof.json"),
        ]

        def add_json(plot_obj, json_data, config,
                     direc, bench_name, pretty_bench):
            plot_obj.add_json_data(json_data, bench_name, pretty_bench,
                                   device=config['device_name'])
        self.do_json_plot(plot_summary, readers, add_json)

    def plot_time_breakdown(self):
        time_breakdown = TimeBreakdownPlot(self._time_breakdown_png, show=self.args.show)

        # Glob all the time_breakdown json files.
        # Read them into json.
        # Pass them to time_breakdown.add_json_data
        readers = [
            BenchJsonReader("*.breakdown.json", "{bench}.breakdown.json"),
            BenchJsonReader("*.python_overhead.pyprof.json", "python_profile.{bench}.python_overhead.pyprof.json"),
        ]


        def add_json(plot_obj, json_data, config,
                     direc, bench_name, pretty_bench):
            plot_obj.add_json_data(json_data,
                                   bench_name=bench_name,
                                   device=config['device_name'],
                                   impl_name=config['impl_name'])
        self.do_json_plot(time_breakdown, readers, add_json)

    @property
    def _time_breakdown_png(self):
        return _j(self.args.directory, "time_breakdown.png")

    def do_json_plot(self, plot_obj, readers, add_json):
        if len(self.args.directories) == 0:
            return

        def _do_plot(bench_json_reader):
            for json_data, direc, bench_name, pretty_bench in bench_json_reader.each_direc_bench_data(self.args.directories):
                config_path = self._config_path_at(direc)
                config = load_json(config_path)
                add_json(plot_obj, json_data, config,
                         direc, bench_name, pretty_bench)
            plot_obj.plot()

        # profile_paths = bench_json_reader.all_profile_paths(self.args.directories)
        # if len(profile_paths) == 0:

        for reader_i, bench_json_reader in enumerate(readers):

            profile_paths = bench_json_reader.all_profile_paths(self.args.directories)
            if len(profile_paths) == 0:
                if reader_i == len(readers) - 1:
                    raise RuntimeError(textwrap.dedent("""
                    Couldn't find any profile paths in provided directories.
                      globs={globs}
                      dirs={dirs}
                    """.format(
                        dirs=self.args.directories,
                        globs=[reader.json_glob for reader in readers],
                    )))
                continue

            _do_plot(bench_json_reader)
            break


    @property
    def _time_breakdown_png(self):
        return _j(self.args.directory, "time_breakdown.png")

    def run_benchmark(self, bench_name):
        g = tf.Graph()
        sess = tf.Session(graph=g)
        with sess.as_default(), g.as_default():
            args_dict = get_default_args_dict(self.args, bench_name)
            MicroKlass = NAME_TO_MICRO[bench_name]
            bench = MicroKlass(name=bench_name, env=self.env, **args_dict)
            # if self.args.c_only and not bench.can_c_only:
            #     print("> Skip benchmark={name}; cannot measure C++ code for it separately (can_c_only = False)".format(
            #         name=bench_name))
            #     return
            print("> Benchmark: {name}".format(name=bench.name))
            self.dump_config(bench_name)
            bench.run()

    def dump_config(self, bench_name):
        avail_gpus = get_available_gpus()
        avail_cpus = get_available_cpus()
        # We want to be CERTAIN about which device TensorFlow is using.
        # If no GPUs are available, TF will use the CPU.
        # If a GPU is available, make sure only 1 is available so we are certain it's using that one.
        if not( (len(avail_gpus) == 1) or
                (len(avail_gpus) == 0 and len(avail_cpus) == 1) ):
            CUDA_VISIBLE_DEVICES = ENV.get('CUDA_VISIBLE_DEVICES', None)
            pprint.pprint({
                'avail_gpus':avail_gpus,
                'avail_cpus':avail_cpus,
                'CUDA_VISIBLE_DEVICES':CUDA_VISIBLE_DEVICES,
            })
            assert(
                (len(avail_gpus) == 1) or
                (len(avail_gpus) == 0 and len(avail_cpus) == 1)
            )
        if len(avail_gpus) == 1:
            device_dict = avail_gpus[0]
        else:
            device_dict = avail_cpus[0]
        data = {
            'clock':self.args.clock,
            'impl_name':"DQN Python",
            'bench_name_labels': {
                'q_update_target_network':'Update target network',
                'q_forward':'Q-forward',
                'q_backward':'Q-backward',
                'step':'Step',
                'total':'Total',
            },
        }
        data.update(device_dict)
        do_dump_json(data, self._config_path(bench_name))

    def _config_path(self, bench_name):
        return _j(self.args.directory, "config.json")

    def _config_path_at(self, direc):
        return _j(direc, "config.json")

    def plot_benchmarks(self):
        bench_name = list(BENCH_NAME_ORDER)[0]
        args_dict = get_default_args_dict(self.args, bench_name)
        MicroKlass = NAME_TO_MICRO[bench_name]
        bench = MicroKlass(name=bench_name, env=self.env, **args_dict)
        # If the user didn't EXPLICITLY ask to plot stuff,
        # allow plotting to fail since maybe not all the data is ready.
        allow_failure = not self.args.plot

        def _do_plot():
            self.plot_summary()
            self.plot_time_breakdown()
            # bench.plot_microbenchmarks(allow_failure)

        if allow_failure and not self.args.debug:
            try:
                _do_plot()
            except Exception as e:
                print("> Got exception while plotting after running benchmark; skipping plot:")
                print(textwrap.indent(str(e), prefix="  "))
        else:
            _do_plot()

    def microbench_data(self):
        bench_name = list(BENCH_NAME_ORDER)[0]
        args_dict = get_default_args_dict(self.args, bench_name)
        MicroKlass = NAME_TO_MICRO[bench_name]
        bench = MicroKlass(name=bench_name, env=self.env, **args_dict)
        bench.load()
        return bench.data

    def glob_files(self, basename_glob, basename_negate_re=None):
        paths = [path for path \
                         in glob("{dir}/{glob}".format(dir=self.args.directory, glob=basename_glob)) \
                         if not re.search(r'\.pretty\.', _b(path)) and (
                                 basename_negate_re is None or not re.search(basename_negate_re, _b(path)))]
        return paths

    def _prettify(self, name, profile_basename_glob, ProfileParserKlass, profile_basename_negate_re=None):
        self.load(reload=True)
        profile_paths = self.glob_files(profile_basename_glob, basename_negate_re=profile_basename_negate_re)
        for profile_path in profile_paths:
            bench_name = get_bench_name(_b(profile_path))
            prof = ProfileParserKlass(self.parser, self.args, profile_path, bench_name=bench_name, data=self.data)
            if _e(prof.dump_path):
                if self.args.replace_parse:
                    print("> Replacing {f}".format(f=prof.dump_path))
                elif os.path.getsize(prof.dump_path) == 0:
                    os.remove(prof.dump_path)
                else:
                    # print("> SKIP EXISTS: {f}".format(f=prof.dump_path))
                    continue
            # print("> DOES NOT YET EXIST: {f}".format(f=prof.dump_path))
            print("> Prettify {name} profile: {path}".format(name=name, path=prof.dump_path))
            prof.run(bench_name)

    def python_profile_json_path(self, bench_name):
        if bench_name is not None:
            return _j(self.args.directory, "python_profile.{bench}.python_overhead.pyprof.json".format(bench=bench_name))
        return _j(self.args.directory, "python_profile.python_overhead.pyprof.json")

    def nvprof_profile_json_path(self, bench_name):
        if bench_name is not None:
            return _j(self.args.directory, "nvidia.{bench}.gpu_overhead.nvprof.json".format(bench=bench_name))
        return _j(self.args.directory, "nvidia.gpu_overhead.nvprof.json")

    def load(self, reload=False):
        if reload or self.data is None:
            self.data = self.microbench_data()


def exists_nonempty(path):
    if _e(path):
        if os.path.getsize(path) == 0:
            os.remove(path)
            return False
        return True
    return False

def get_default_args_dict(args, bench_name):
    defaults = dict(args.__dict__)
    choices = list(defaults.items())
    for aname, val in choices:
        if val is not None:
            continue

        if bench_name in MICRO_DEFAULTS and aname in MICRO_DEFAULTS[bench_name]:
            defaults[aname] = MICRO_DEFAULTS[bench_name][aname]
            continue

        if args.debug:
            argparse_defaults = ARGPARSE_DEFAULTS_DEBUG
        else:
            argparse_defaults = ARGPARSE_DEFAULTS

        if aname in argparse_defaults:
            defaults[aname] = argparse_defaults[aname]
            continue

    return defaults

def parse_args(parser):
    # TODO: Make sure the things we are measuring capture all the expensive portions of the DQN training loop.
    args = parser.parse_args()
    if args.direc is not None:
        args.directory = args.direc
    if args.directory is None:
        args.directory = _j(*[x for x in
                              [
                                  py_config.ROOT,
                                  "checkpoints",
                                  args.env_type,
                                  args.subdir,
                                  "microbenchmark",
                              ]
                              if x is not None] )

    if len(args.benchmarks) == 0:
        args.benchmarks = list(BENCH_NAME_ORDER)

    args.dynamic_iterations = not args.no_dynamic_iterations
    args.discard_first_sample = not args.no_discard_first_sample

    args.profile_cuda = not args.no_profile_cuda

    return args

if __name__ == '__main__':
    main()
