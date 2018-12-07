# JAMES LEFT OFF:
# - refactoring CUDASQLitePaser
# - need to refactor benchmark_dqn.py to just call scons instead manually managing crap.
#   TODO: limit rebuild to a rooted directory? or delete old crap that fails
# - do it for plots maybe...
# -

# DQN specific, disable for now
# import gym
gym = None

from io import StringIO

from os import environ as ENV
import pandas as pd
import time
# import baselines
# from baselines import deepq
# from baselines.deepq.simple_refactor import DQN, NAME_TO_MICRO, CUDA_MICROBENCH_NAMES, CUDA_MICROBENCH_NAME_TO_PRETTY, BENCH_NAME_REGEX, BENCH_AND_TOTAL_NAME_REGEX, BENCH_NAME_ORDER, BENCH_TYPES, MICRO_DEFAULTS, ARGPARSE_DEFAULTS, ARGPARSE_DEFAULTS_DEBUG, BENCH_NAME_TO_PRETTY, \
#     get_nvprof_name, get_c_only_name, func_std_string, get_microbench_path, get_microbench_basename, is_microbench_path
import re

# pip install py-cpuinfo
import cpuinfo

import csv

# pip install progressbar2
import progressbar
import itertools
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
import json
import codecs
import pickle
import seaborn as sns
import copy

# from baselines.deepq.experiments.collect_atari_checkpoints import fixup_json

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import py_config

CUDA_MICROBENCH_NAMES = [
    'cuda_launch',
    'd2h',
    'h2d',
]
CUDA_MICROBENCH_NAME_TO_PRETTY = {
    'cuda_launch':'Launch Kernel',
    'd2h':'Device-to-Host',
    'h2d':'Host-to-Device',
}
BENCH_NAME_REGEX = r"(?:[^\.]+)"
BENCH_NAME_ORDER = [
    'step',
    'q_forward',
    'q_backward',
    'q_update_target_network',
    # 'training_iteration',
    # 'checkpoint',
    # 'restore',
    # 'compress',
    # 'decompress',
    # 'total',
]
BENCH_TYPES = [
    # Microbenchmarks for the CUDA API
    # (e.g. cudaLaunch latency).
    'cuda',
    # Microbenchmarks for individual operations from the DQN training loop.
    'dqn',
    'cpufreq',
    'nvprof',
    # 'python_profile',
]
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
BENCH_NAME_TO_PRETTY = {
    "checkpoint":"Checkpoint",
    "q_backward":"Q-backward",
    "q_forward":"Q-forward",
    "q_update_target_network":"Q update \ntarget-network",
    "restore":"Restore",
    "training_iteration":"Training iteration",
    "step":"Step",
    "total":"Total",
    # "compress":_compress_name("Compress"),
    # "decompress":_compress_name("Decompress"),
}



# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

cxxfilt = None
try:
    import cxxfilt
except ImportError:
    pass

sqlite3 = None
try:
    import sqlite3
except ImportError:
    pass

DQN_BENCH_NAMES = ['q_forward', 'q_backward', 'q_update_target_network']

MICROSECONDS_IN_SECOND = float(1e6)
MILLISECONDS_IN_SECOND = float(1e3)
NANOSECONDS_IN_SECOND = float(1e9)

# figsize (W x H) in inches
aspect_ratio = 16./9.
fig_width = 2*7
fig_height = float(fig_width) / aspect_ratio
FIG_SIZE = (fig_width, fig_height)

LINE_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
LINE_STYLES = ['-','--',':','-.']
LINE_THEMES = []
for linestyle in LINE_STYLES:
    for color in LINE_COLORS:
        LINE_THEMES.append({'color':color, 'linestyle':linestyle})

def load_json(path):
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        data = fixup_json(data)
        return data

def do_dump_json(data, path):
    os.makedirs(_d(path), exist_ok=True)
    json.dump(data,
              codecs.open(path, mode='w', encoding='utf-8'),
              sort_keys=True, indent=4,
              skipkeys=False)

def test_grouped_stacked_bar_plot(parser, args):
    # Stacked barplot:
    # https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html
    def stacked_barplot():
        N = 5
        menMeans = (20, 35, 30, 35, 27)
        womenMeans = (25, 32, 34, 20, 25)
        menStd = (2, 3, 4, 1, 2)
        womenStd = (3, 5, 2, 3, 3)
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, menMeans, width, yerr=menStd)
        p2 = plt.bar(ind, womenMeans, width,
                     bottom=menMeans, yerr=womenStd)

        plt.ylabel('Scores')
        plt.title('Scores by group and gender')
        plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
        plt.yticks(np.arange(0, 81, 10))
        plt.legend((p1[0], p2[0]), ('Men', 'Women'))

        plt.show()

    # Grouped barplot
    # https://python-graph-gallery.com/11-grouped-barplot/
    def grouped_barplot():
        # libraries
        # set width of bar
        bar_width = 0.25

        # set height of bar
        bars1 = [12, 30, 1, 8, 22]
        bars2 = [28, 6, 16, 5, 10]
        bars3 = [29, 3, 24, 25, 17]

        # Set position of bar on X axis
        r1 = np.arange(len(bars1))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]

        # Make the plot
        plt.bar(r1, bars1, color='#7f6d5f', width=bar_width, edgecolor='white', label='var1')
        plt.bar(r2, bars2, color='#557f2d', width=bar_width, edgecolor='white', label='var2')
        plt.bar(r3, bars3, color='#2d7f5e', width=bar_width, edgecolor='white', label='var3')

        # Add xticks on the middle of the group bars
        plt.xlabel('group', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

        # Create legend & Show graphic
        plt.legend()
        plt.show()

    # How to put patterns (hatches) on bars of a barplot.
    # https://matplotlib.org/gallery/shapes_and_collections/hatch_demo.html#sphx-glr-gallery-shapes-and-collections-hatch-demo-py
    # http://kitchingroup.cheme.cmu.edu/blog/2013/10/26/Hatched-symbols-in-matplotlib/
    # patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.', '/')
    def hatch_barplot():
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Polygon

        fig, (ax1, ax2, ax3) = plt.subplots(3)

        ax1.bar(range(1, 5), range(1, 5), color='red', edgecolor='black', hatch="/")
        ax1.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
                color='blue', edgecolor='black', hatch='//')
        ax1.set_xticks([1.5, 2.5, 3.5, 4.5])

        bars = ax2.bar(range(1, 5), range(1, 5), color='yellow', ecolor='black') + \
            ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5),
                    color='green', ecolor='black')
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

        ax3.fill([1, 3, 3, 1], [1, 1, 2, 2], fill=False, hatch='\\')
        ax3.add_patch(Ellipse((4, 1.5), 4, 0.5, fill=False, hatch='*'))
        ax3.add_patch(Polygon([[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]], closed=True,
                              fill=False, hatch='/'))
        ax3.set_xlim((0, 6))
        ax3.set_ylim((0, 2.5))

        plt.show()

    # Multiple legends in a single plot.
    # https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph
    def multi_legend_plot():
        # I have a plot where different colors are used for different parameters, and
        # where different line styles are used for different algorithms.

        colors = ['b', 'r', 'g', 'c']
        cc = itertools.cycle(colors)
        plot_lines = []
        parameters = np.arange(3)
        n_points = 3

        def algo(algo_num, p):
            slope = p*algo_num
            return np.zeros(n_points) + slope*np.arange(n_points)

        for p in parameters:

            d1 = algo(1, p)
            d2 = algo(2, p)
            d3 = algo(3, p)

            # plt.hold(True)
            c = next(cc)
            # algo1 uses -, algo2 uses --, ...
            l1, = plt.plot(d1, '-', color=c)
            l2, = plt.plot(d2, '--', color=c)
            l3, = plt.plot(d3, '.-', color=c)

            plot_lines.append([l1, l2, l3])

        legend1 = plt.legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
        plt.legend([l[0] for l in plot_lines], parameters, loc=4)
        plt.gca().add_artist(legend1)

        plt.show()

    def grouped_stacked_barplot():
        png_path = "grouped_stacked_barplot.png"

        def _timings(base_sec, iterations):
            def _t(i, offset):
                return (base_sec + offset) + 0.1*i
            def _ts(offset):
                return [_t(i, offset) for i in range(iterations)]
            return {
                'GPUTimeSec': _ts(offset=0),
                'CppTimeSec': _ts(offset=1),
                'PythonTimeSec': _ts(offset=2),
            }

        def _generate_timings(bench_name_order, base_sec, iterations):
            timings = dict()
            for offset, bench_name in enumerate(bench_name_order):
                timings[bench_name] = _timings(base_sec + offset, iterations)
            return timings

        def _generate_json_datas(time_breakdown):
            iterations = 4

            json_datas = []

            data_quadro_k4000 = {
                'attrs': {
                    'name':'Quadro K4000',
                    'impl_name':'DQN Python',
                },
            }
            data_quadro_k4000.update(_generate_timings(time_breakdown.bench_name_order, 1, iterations))
            json_datas.append(data_quadro_k4000)

            data_quadro_p4000 = {
                'attrs': {
                    'name':'Quadro P4000',
                    'impl_name':'DQN Python',
                },
            }
            data_quadro_p4000.update(_generate_timings(time_breakdown.bench_name_order, 2, iterations))
            json_datas.append(data_quadro_p4000)

            data_gtx_1080 = {
                'attrs': {
                    'name':'GTX 1080',
                    'impl_name':'DQN Python',
                },
            }
            data_gtx_1080.update(_generate_timings(time_breakdown.bench_name_order, 3, iterations))
            json_datas.append(data_gtx_1080)

            return json_datas

        time_breakdown = TimeBreakdownPlot(png_path, show=args.show)

        json_datas = _generate_json_datas(time_breakdown)
        for json_data in json_datas:
            bench_names = [k for k in json_data.keys() if k not in set(['attrs'])]
            for bench_name in bench_names:
                time_breakdown.add_json_data(json_data[bench_name],
                                             bench_name=bench_name,
                                             device=json_data['attrs']['name'],
                                             impl_name=json_data['attrs']['impl_name'])
        time_breakdown.plot()

    # stacked_barplot()
    # grouped_barplot()
    # hatch_barplot()
    # multi_legend_plot()
    grouped_stacked_barplot()

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

def parse_profile(parser, args):
    assert args.rule is not None

    ParserKlass = PARSER_NAME_TO_KLASS[args.rule]

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

        for bench_name in bench_names:
            expr = ParserKlass(parser, args, src_files, bench_name=bench_name)
            expr.run(bench_name)

def main():
    parser = argparse.ArgumentParser("benchmark DQN")

    parser.add_argument("--benchmarks",
                        choices=list(BENCH_NAME_ORDER),
                        action='append', default=[])
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
    parser.add_argument("--rule",
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
    parser.add_argument("--nvprof-logfile",
                        help=textwrap.dedent("""
                        Internal use only; 
                        output file for nvprof.
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

    if args.rule is not None:
        parse_profile(parser, args)
        return

    global tf
    global tf_device_lib
    import tensorflow as tf
    from tensorflow.python.client import device_lib as tf_device_lib

    if args.test_plot:
        test_grouped_stacked_bar_plot(parser, args)
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

    if 'cuda' == args.benchmark_type:
        expr = BenchmarkCUDA(parser, args)
        expr.run()
        return

    raise NotImplementedError("Not sure how to run --benchmark-type={typ}".format(
        typ=args.benchmark_type))

def get_pretty_bench(bench_name):
    return BENCH_NAME_TO_PRETTY.get(bench_name, bench_name)

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

def as_order_map(xs):
    order_map = dict()
    for i, x in enumerate(xs):
        order_map[x] = i
    return order_map

def each_order(order_map, rev_order_map):
    for x_order in range(max(order_map.values()) + 1):
        x = rev_order_map[x_order]
        yield x, x_order

def reverse_dict(dic):
    r_dic = dict()
    for k, v in dic.items():
        r_dic[v] = k
    return r_dic


class DataFrame:
    """
    Useful functions for manipulating pandas dataframes.
    """

    @staticmethod
    def get_mean_std(df, value_field):
        groupby_cols = DataFrame.get_groupby_cols(df, value_field)
        mean = df.groupby(groupby_cols).mean().rename(columns={value_field: 'mean'}).reset_index()
        std = df.groupby(groupby_cols).std().rename(columns={value_field: 'std'}).reset_index()
        mean['std'] = std['std']
        return mean

    @staticmethod
    def print_df(df, **kwargs):
        pd.options.display.max_rows = None
        pd.options.display.max_columns = None
        pd.options.display.width = 9999999
        print(df, **kwargs)
        pd.reset_option('all')


    @staticmethod
    def get_groupby_cols(df, value_field):
        groupby_cols = [field for field in list(df.keys()) if field != value_field]
        return groupby_cols

class ProfilerParserCommonMixin:

    def run(self, bench_name):
        self.parse(bench_name)
        self.dump(bench_name)

    @classmethod
    def _match_regexes(Klass, regexes, paths, allow_multiple,
                       # prefix_path=None,
                       # If the bench_name matches this pattern, consider the pattern a failed match.
                       ignore_bench_re=r"call_times",
                       debug=False):
        """
        Return all paths matching some regex in <regexes>.
        If a single regex matches multiple paths,
        only one of those paths will be added for that regex.

        :param regexes:
          {'<regex_name>':python_regex,
           .... }
        :param paths:
        :param allow_multiple:
          Allow multiple files to match a single regex.
          If that's the case, we store results as
          m = re.search(regex, path)
          matching_paths[regex][m.group('bench_name')].append(path)
        :return:
            # PSEUDOCODE:
            # if uses bench_name:
            #     src_files.get('profile_path', bench_name)[bench_name][0...N]
            # else:
            #     src_files.get('profile_path', bench_name)[None][0]
          matching_paths['<regex_name>'][bench_name/None][0...N] =

            {
              '<regex_name>': {
                <bench_name[0]/None>: [
                  matching_path[0],
                  matching_path[1],
                  ...,
                  ]
                  ...,
              }
            }
        """
        regexes_left = list(regexes.items())
        paths_left = list(paths)
        # matching_paths = []
        matching_paths = dict()
        j = 0
        if debug:
            print("> allow_multiple={v}".format(v=allow_multiple))
        while j < len(regexes):
            regex_name, regex = regexes_left[j]

            i = 0
            while i < len(paths_left):
                path = paths_left[i]
                m = re.search(regex, _b(path))

                # if m and ignore_bench_re is not None and re.search(ignore_bench_re, _b(path)) and 'bench_name' in m.groupdict() and m.group('bench_name'):
                if m and ignore_bench_re is not None \
                        and m.groupdict().get('bench_name', None) is not None \
                        and re.search(ignore_bench_re, m.group('bench_name')):
                    if debug:
                        print("> regex={regex}, ignore_bench_re={ignore_regex} matches _b(path)={path}".format(
                            regex=regex,
                            ignore_regex=ignore_bench_re,
                            path=_b(path)))
                    del paths_left[i]
                elif m:
                    if debug:
                        print("> regex={regex} matches _b(path)={path}".format(regex=regex, path=_b(path)))
                    # PSEUDOCODE:
                    # if uses bench_name:
                    #     src_files.get('profile_path', bench_name)[bench_name]
                    # else:
                    #     src_files.get('profile_path', bench_name)
                    mdict = m.groupdict()
                    # if prefix_path is not None:
                    #     full_path = _j(prefix_path, path)
                    # else:
                    full_path = path
                    if 'bench_name' in mdict:
                        _mk(matching_paths, regex_name, dict())
                        bench_name = mdict['bench_name']
                        if bench_name is None:
                            bench_name = NO_BENCH_NAME
                        assert allow_multiple or bench_name not in matching_paths[regex_name]
                        matching_paths[regex_name][bench_name] = full_path
                    else:
                        _mk(matching_paths, regex_name, dict())
                        assert allow_multiple or NO_BENCH_NAME not in matching_paths[regex_name]
                        matching_paths[regex_name][NO_BENCH_NAME] = full_path

                    del paths_left[i]
                    if not allow_multiple:
                        break
                else:
                    if debug:
                        print("> regex={regex} DOES NOT MATCH _b(path)={path}".format(regex=regex, path=_b(path)))
                    i += 1

            if len(paths_left) == 0:
                break

            # if not allow_multiple:
            j += 1
        return matching_paths

    @classmethod
    def list_files(Klass, direc,
                   # keep_direc=True
                   ):
        def _list_files(direc):
            def _path(path):
                return _j(direc, path)
            try:
                pass  # Cmd
                return [_path(path) for path in os.listdir(direc)]
            except Exception as e:
                import ipdb;
                ipdb.set_trace()
                raise e


        if type(direc) == list:
            all_files = []
            for d in direc:
                all_files.extend(_list_files(d))
            return all_files

        return _list_files(direc)

    @classmethod
    def all_directories_rooted_at(Klass, root_dir):
        direcs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            direcs.append(_j(root_dir, dirpath))
        return direcs

    @classmethod
    def find_source_directories(Klass, root_dir):
        src_dirs = dict()
        all_dirs = Klass.all_directories_rooted_at(root_dir)
        for direc in all_dirs:
            src_files = Klass.get_source_files(direc)
            if src_files.has_all_required_paths:
                assert direc not in src_dirs
                src_dirs[direc] = src_files
        return src_dirs

    @classmethod
    def as_source_files(Klass, srcs, debug=False):
        assert type(srcs) == list
        return Klass._get_source_files(srcs, debug=debug)

    @classmethod
    def get_source_files(Klass, direc, debug=False):
        basenames = Klass.list_files(direc,
                                     # keep_direc=False
                                     )
        return Klass._get_source_files(basenames, debug=debug)


    @classmethod
    def _get_source_files(Klass, paths, debug=False):
        return Klass._as_src_files(paths,
                                   req_regexes=Klass.required_source_basename_regexes(),
                                   opt_regexes=Klass.optional_source_basename_regexes(),
                                   allow_multiple_src_matches=Klass.allow_multiple_src_matches(),
                                   debug=debug)

    @classmethod
    def glob_target_files(Klass, paths, debug=False):
        return Klass._as_src_files(paths,
                                   req_regexes=Klass.target_basename_regexes(),
                                   # opt_regexes=Klass.optional_source_basename_regexes(),
                                   # allow_multiple_src_matches=Klass.allow_multiple_src_matches(),
                                   debug=debug)

    @classmethod
    def _as_src_files(Klass, paths, req_regexes, opt_regexes=None, allow_multiple_src_matches=True, debug=False):
        # basenames = Klass.list_files(direc, keep_direc=False)

        def _mk_src_files(src_paths):

            # req_regexes = Klass.required_source_basename_regexes()
            req_paths = Klass._match_regexes(req_regexes, src_paths, allow_multiple_src_matches,
                                             debug=debug)

            # opt_regexes = Klass.optional_source_basename_regexes()
            if opt_regexes is not None:
                opt_paths = Klass._match_regexes(opt_regexes, src_paths, allow_multiple_src_matches)
            else:
                opt_paths = None

            directory = common_dir(src_paths)
            src_files = SrcFiles(directory, req_paths, opt_paths,
                                 has_all_required_paths=len(req_paths) == len(req_regexes),
                                 allow_multiple_src_matches=allow_multiple_src_matches)
            return src_files

        # directory = common_dir(paths)

        direc_to_srcs = dict()
        for path in paths:
            direc = _d(path)
            _mk(direc_to_srcs, direc, [])
            direc_to_srcs[direc].append(path)

        if len(direc_to_srcs) == 1:
            direc = list(direc_to_srcs.keys())[0]
            return _mk_src_files(direc_to_srcs[direc])

        src_files_list = []
        for direc, src_paths in direc_to_srcs.items():
            src_files_list.append(_mk_src_files(src_paths))
        src_files = SrcFilesGroup(src_files_list)
        return src_files

    @classmethod
    def config_get(Klass, src_files, attr, default):
        config_path = src_files.get('config_json', or_none=True)
        if config_path is None:
            value = None
        else:
            config_json = load_json(config_path)
            if attr in config_json and config_json[attr] is None:
                config_json[attr] = default
            value = config_json[attr]
        return value


# Refactor this to read in json files.
# Q: where to record device/impl_name?  Can we add it to json_data?
# TimeBreakdownPlot.add_json_data(json_data, impl_name=..., device=...)
# TimeBreakdownPlot.add_json_data(...)
# ...
# TimeBreakdownPlot.plot()
class TimeBreakdownPlot(ProfilerParserCommonMixin):
    """
    Create a stacked bar plot.
    For the list of times to show, see self.time_name_order.
    """

    @staticmethod
    def target_basename_regexes():
        return {
            'plot_data_path': r"^time_breakdown.plot_data.txt$",
            'png': r"^time_breakdown.png$",
        }

    @staticmethod
    def required_source_basename_regexes():
        # Same requirement; breakdown_json.
        return PlotSummary.required_source_basename_regexes()

    @staticmethod
    def optional_source_basename_regexes():
        # Same requirement; config_json.
        return PlotSummary.optional_source_basename_regexes()

    @staticmethod
    def allow_multiple_src_matches():
        return PlotSummary.allow_multiple_src_matches()

    @staticmethod
    def uses_all_benches():
        return PlotSummary.uses_all_benches()

    @staticmethod
    def uses_multiple_dirs():
        return True

    @classmethod
    def get_targets(Klass, src_files, bench_name=NO_BENCH_NAME):
        # TODO: get rid of bench_name (if uses_all_benches is set?)
        # TODO: fields depends on contents of src_files technically... ok to not output a file we say we will output?
        targets = [
            Klass.get_plot_data_path(src_files),
            Klass.get_time_breakdown_png(src_files),
        ]
        return targets

    @staticmethod
    def get_time_breakdown_png(src_files):
        return _j(src_files.directory, "time_breakdown.png")

    @property
    def _time_breakdown_png(self):
        return TimeBreakdownPlot.get_time_breakdown_png(self.src_files)

    @staticmethod
    def get_plot_data_path(src_files):
        return _j(src_files.directory, "time_breakdown.plot_data.txt")

    @property
    def _plot_data_path(self):
        return TimeBreakdownPlot.get_plot_data_path(self.src_files)

    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, bar_width=0.25, show=False):
        # def __init__(self, png_path, bar_width=0.25, show=False):
        # set width of bar
        self.parser = parser
        self.args = args
        self.src_files = src_files
        self.bar_width = bar_width

        self.value_field = 'time_sec'

        self.show = show
        self.df_data = {
            "bench_name":[],
            "bench_name_order":[],
            "impl_name":[],
            "impl_name_order":[],
            "device":[],
            "device_order":[],
            # Name of time: PythonTimeSec, CppTimeSec, etc.
            "time_name":[],
            "time_name_order":[],
            # Value for <time_name>.
            "time_sec":[],
        }
        self.df = None

        # hatch_size = 3
        # self.time_name_hatch_map = {
        #     'GPUTimeSec':".",
        #     'CudaCppTimeSec':"\\",
        #     'FrameworkCppTimeSec':"/",
        #     'PythonTimeSec':"x",
        # }
        self.time_name_hatch_map = {
            'GPUTimeSec':"\\",
            'CppTimeSec':"|",
            'PythonTimeSec':"/",
        }

        # self.bench_name_order = ['q_update_target_network', 'q_forward', 'q_backward', 'step', 'total']

        # # Delay making this until we know all the bench_name's from add_json_data
        # self.bench_name_order = ['q_update_target_network', 'q_forward', 'q_backward', 'step']
        # self.bench_name_order_map = as_order_map(self.bench_name_order)
        # self.rev_bench_name_order_map = reverse_dict(self.bench_name_order_map)
        # self.bench_name_color_map = self.as_color_map(self.bench_name_order)

        # self.config_path = src_files.get('config_json', bench_name, or_none=True)
        # if self.config_path is not None:
        #     self.config = load_json(self.config_path)
        #     print("> Found optional config_json @ {f}".format(f=self.config_path))
        # else:
        #     self.config = {
        #         'clock':'monotonic_clock',
        #     }

        # self.bench_name_labels = {
        #     'q_update_target_network':'Update target network',
        #     'q_forward':'Q-forward',
        #     'q_backward':'Q-backward',
        #     'step':'Step',
        #     # 'total':'Total',
        # }
        # NOTE: these labels should come from an input file...config_json
        # self._check_has_keys(self.bench_name_order, self.bench_name_labels)

        self.time_name_order = ['GPUTimeSec', 'CppTimeSec', 'PythonTimeSec']
        # This works for end-to-end test, but not for DQN benchmark.
        # self.time_name_order = ['GPUTimeSec', 'CudaCppTimeSec', 'FrameworkCppTimeSec', 'PythonTimeSec']
        self.time_name_order_map = as_order_map(self.time_name_order)
        self.rev_time_name_order_map = reverse_dict(self.time_name_order_map)
        self.time_name_labels = {
            'GPUTimeSec':'GPU time',
            'CppTimeSec':'C++ time',
            'CudaCppTimeSec':'CUDA C time (API calls, driver)',
            'FrameworkCppTimeSec':'Framework C time',
            'PythonTimeSec':'Python time',
        }
        self._check_has_keys(self.time_name_order, self.time_name_labels)
        self._check_has_keys(self.time_name_order, self.time_name_hatch_map)

        self.impl_name_order = ["DQN Python", NO_IMPL_NAME]
        self.impl_name_order_map = as_order_map(self.impl_name_order)
        self.rev_impl_name_order_map = reverse_dict(self.impl_name_order_map)

        self.device_order = ['NoDeviceName', 'Quadro K4000', 'Quadro P4000', 'GTX 1080']
        self.device_order_map = as_order_map(self.device_order)
        self.rev_device_order_map = reverse_dict(self.device_order_map)

    def _build_bench_name_order(self):
        # # Delay making this until we know all the bench_name's from add_json_data
        # self.bench_name_order = ['q_update_target_network', 'q_forward', 'q_backward', 'step']
        self.bench_name_order = sorted(unique(self.df_data['bench_name']))
        self.bench_name_order_map = as_order_map(self.bench_name_order)
        self.rev_bench_name_order_map = reverse_dict(self.bench_name_order_map)
        self.bench_name_color_map = self.as_color_map(self.bench_name_order)
        self.bench_name_labels = {
            'q_update_target_network':'Update target network',
            'q_forward':'Q-forward',
            'q_backward':'Q-backward',
            'step':'Step',
            # 'total':'Total',
        }

    def add_json_data(self, json_data, bench_name, device, impl_name):
        # bench_names = self._get_bench_names(json_data)
        # # device = json_data['attrs']['name']
        # # impl_name = json_data['attrs']['impl_name']
        # for bench_name in bench_names:
        time_names = self._get_time_names(json_data)
        for time_name in time_names:
            for time_sec in json_data[time_name]:
                self.df_data['bench_name'].append(bench_name)
                # self.df_data['bench_name_order'].append(self.bench_name_order_map[bench_name])
                self.df_data['impl_name'].append(impl_name)
                self.df_data['impl_name_order'].append(self.impl_name_order_map[impl_name])
                self.df_data['device'].append(device)
                self.df_data['device_order'].append(self.device_order_map[device])
                self.df_data['time_name'].append(time_name)
                self.df_data['time_name_order'].append(self.time_name_order_map[time_name])
                self.df_data['time_sec'].append(time_sec)

    def run(self, bench_name=NO_BENCH_NAME):
        # for
        #     self.add_json_data()
        # We need to add all the json files, regardless of the field type?
        # for in self.src_files.all_sources(all_bench_names)
        for directory in self.src_files.directories:
            src_files = self.src_files.get_src_files(directory)

            device_name = self.config_get(src_files, 'device_name', NO_DEVICE_NAME)
            impl_name = self.config_get(src_files, 'impl_name', NO_IMPL_NAME)

            bench_names = src_files.bench_names
            for bench in bench_names:
                json_path = src_files.get('breakdown_json', bench)
                json_data = load_json(json_path)
                # TODO: still dqn specific.
                pretty_bench = get_pretty_bench(bench)
                # assert len(bench_names) == 0
                # bench = bench_names[0]
                self.add_json_data(json_data, bench,
                                   device_name, impl_name)
                                   # pretty_bench, device_name)
        self._build_bench_name_order()
        self.df_data['bench_name_order'] = [self.bench_name_order_map[bench_name] for bench_name in self.df_data['bench_name']]
        # self.df_data['bench_name_order'].append(self.bench_name_order_map[bench_name])
        self.plot()

    def plot(self):
        if self.df is None:
            self._as_dataframe()

        with open(self._plot_data_path, 'w') as f:
            DataFrame.print_df(self.mean_df, file=f)
        print("> DataFrame:")
        print(self.mean_df)

        fig = plt.figure()

        self._add_lines()
        self._add_legend()
        self._add_axis_labels()
        self._show()

    def _as_dataframe(self):

        # devices = list(data.keys())
        self.orig_df = pd.DataFrame(self.df_data)

        self.df = DataFrame.get_mean_std(self.orig_df, self.value_field)
        self.df = self.df.sort_values(by=['impl_name_order', 'device_order', 'bench_name_order', 'time_name_order'])
        # groupby_cols = DataFrame.get_groupby_cols(self.orig_df, value_field)

        self.mean_df = self.df

    def _add_legend(self):
        self.legend_makers = []

        # We need two groups of lines:
        #
        # 1) Hatch-type:
        #    - Should have the same color
        #    - # of hash-types = len(time_name_order = ['GPUTimeSec', 'CppTimeSec', 'PythonTimeSec'])
        #                      = 3
        #
        # 2) Color-type:
        #    - Should have the same hatch.
        #    - # of color-types = len(bench_name_order = ['q_forward', 'q_backward', 'step'])
        #                       = 3

        hatch_legend = LegendMaker(attr_name='hatch',
                                   field_to_attr_map=self.time_name_hatch_map,
                                   field_order=self.time_name_order,
                                   labels=self.time_name_labels,
                                   legend_kwargs={
                                       'loc':'upper right',
                                   })
        self.legend_makers.append(hatch_legend)

        color_legend = LegendMaker(attr_name='facecolor',
                                   field_to_attr_map=self.bench_name_color_map,
                                   field_order=self.bench_name_order,
                                   labels=self.bench_name_labels,
                                   edgecolor='white',
                                   legend_kwargs={
                                       'loc':'upper left',
                                   })
        self.legend_makers.append(color_legend)

        LegendMaker.add_legends(self.legend_makers)

    def _add_lines(self):
        for impl_name in self.impl_name_order:
            bottom = None
            for bench_name in self.bench_name_order:
                for time_name in self.time_name_order:
                    rows = self.df[
                        (self.df['bench_name'] == bench_name)
                        & (self.df['time_name'] == time_name)
                        ]
                    if len(rows) == 0:
                        continue
                    xvalues = self._get_xvalues(rows['impl_name'], rows['device'])
                    yvalues = rows['mean'].values
                    yerr = rows['std'].values
                    hatch = self.time_name_hatch_map[time_name]
                    color = self.bench_name_color_map[bench_name]

                    if bottom is None:
                        bottom = np.zeros_like(yvalues)

                    # PROBLEM: if data is missing for step
                    assert bottom.shape == yvalues.shape

                    plot = plt.bar(xvalues, yvalues, color=color, width=self.bar_width, edgecolor='white', label=bench_name,
                                   bottom=bottom,
                                   hatch=hatch,
                                   yerr=yerr)

                    bottom += yvalues

    def _show(self):
        if self.show:
            plt.show()
        else:
            print("> Save figure to {path}".format(path=self._time_breakdown_png))
            print("> Save plot data to {path}".format(path=self._plot_data_path))
            plt.savefig(self._time_breakdown_png)
            plt.close()

    def _check_has_keys(self, xs, xs_map):
        for x in xs:
            assert x in xs_map

    def get_color_map(self):
        cmap = plt.get_cmap('Pastel1')
        return cmap

    def as_color_map(self, xs):
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # https://matplotlib.org/tutorials/colors/colormaps.html
        color_map = dict()
        cmap = self.get_color_map()
        # np.arange(0,1,(1 - 0)/5)
        for i, x in enumerate(xs):
            color = cmap(i % cmap.N)
            color_map[x] = color
        return color_map

    def _add_axis_labels(self):
        plt.xlabel('DQN implementation', fontweight='bold')

        n_bars = len(self.device_order)
        xtick_xvalues = self._xtick_xvalues(self.impl_name_order, self.impl_name_order_map, n_bars)
        plt.xticks(xtick_xvalues, self.impl_name_order)

    def _get_bench_names(self, json_data):
        return json_data.keys()

    def _get_time_names(self, json_data):
        return list(k for k in json_data.keys() if k in self.time_name_order)

    def _get_xvalue(self, impl_name, device):
        bench_order = self.impl_name_order_map[impl_name]
        graphics_order = self.device_order_map[device]
        return bench_order + graphics_order*self.bar_width

    def _get_xvalues(self, impl_names, devices):
        return np.array([self._get_xvalue(impl_name, device) \
                         for impl_name, device in zip(impl_names, devices)])

    # Add xticks on the middle of the group bars
    def _xtick_xvalues(self, xvalues, order_map, n_bars):
        idxes = [order_map[xvalue] for xvalue in xvalues]
        all_bars_width = n_bars * self.bar_width
        # This may be wrong.
        center_width = ((n_bars - 1)*self.bar_width)/2
        return [i*all_bars_width + center_width \
                for i in idxes]

class LegendMaker:
    """
    Create "Patches" to create a legend.
    https://matplotlib.org/users/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists
    """
    def __init__(self, attr_name, field_to_attr_map, field_order, labels,
                 edgecolor="black",
                 facecolor="white",
                 legend_kwargs=None,
                 **kwargs):
        self.attr_name = attr_name
        self.field_order = field_order
        self.legend_kwargs = legend_kwargs
        self.patch_attrs = {
            'edgecolor':edgecolor,
            'facecolor':facecolor,
        }
        self.attrs_with_default = list(self.patch_attrs.keys())
        if self.attr_name in self.attrs_with_default:
            del self.patch_attrs[self.attr_name]
        self.labels = labels
        self.patch_attrs.update(kwargs)
        self.patches = []
        self.field_to_attr_map = field_to_attr_map
        self.legend = None

        self._init_patches()

    def _init_patches(self):
        attr_types = [self.field_to_attr_map[field] for field in self.field_order]
        for attr in attr_types:
            patch_kwargs = dict(self.patch_attrs)
            patch_kwargs.update({self.attr_name:attr})
            patch = mpatches.Patch(**patch_kwargs)
            self.patches.append(patch)

    def get_legend(self, **kwargs):
        if self.legend is not None:
            return self.legend
        legend_labels = [self.labels.get(field, field) for field in self.field_order]
        if self.legend_kwargs is not None:
            legend_kwargs = dict(self.legend_kwargs)
        else:
            legend_kwargs = dict()
        legend_kwargs.update(kwargs)
        self.legend = plt.legend(handles=self.patches, labels=legend_labels, **legend_kwargs)
        return self.legend

    @staticmethod
    def add_legends(legend_makers):
        legends = []
        for legend_maker in legend_makers:
            legend = legend_maker.get_legend()
            legends.append(legend)
        for legend in legends:
            plt.gca().add_artist(legend)
        return legends

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

    # def profile_paths(self, search_dir):
    #     profile_paths = [path for path \
    #                      in glob("{direc}/{glob}".format(direc=direc, glob=self.json_glob))]
    #     return profile_paths

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


class PlotSummary(ProfilerParserCommonMixin):
    # TODO: Generalize "bench_names"; currently this has hardcoded DQN specific bench_names
    # Really, this plots each-and-every bench_name in a single plot.
    # The order of the bench_names is based on increasing time spent in the bench_name.
    # We can automate that; or just use alphabetical for now.
    def __init__(self, parser, args, src_files):
        # Each device we measure has its own directory of files.
        # So, we expect a src_files instance, one for each device-directory.
        self.args = args
        self.parser = parser
        self.src_files = src_files
        self.df = None

        self.value_field = 'value'

        # self.bench_json_reader = BenchJsonReader("*.breakdown.json", "{bench}.breakdown.json")

        self.df_data = {
            'field':[],
            # 'mean':[],
            # 'std':[],
            'value':[],
            'device':[],
            'device_order':[],
            'bench_order':[],
            'bench':[],
            'pretty_bench':[],
        }

        # NOTE: We want to be able to provide multiple directories when creating this plot,
        # in order to support multiple devices.
        # How should we do that, given currently we only allow provided a single directory?
        # Basically, we should have multiple src_files (src_file_group?), one for each directory.
        # self.directories = glob_json_files(args.directories, get_microbench_basename(),
        #                                    ignore_missing=False,
        #                                    add_dirs=True)

        self.pretty_label_map = {
            'gtx1080':'GTX 1080',
            'quadro_k4000':'Quadro K4000',
            'quadro_p4000':'Quadro P4000',
        }

        self.pretty_label_order_map = {
            'Quadro K4000':0,
            'Quadro P4000':1,
            'GTX 1080':2,
        }
        self.device_tie_order = max(self.pretty_label_order_map.values()) + 1


        self.bench_names = sorted(src_files.bench_names)
        self.bench_order_map = as_order_map(self.bench_names)
        bench_total_idx = max(self.bench_order_map.values()) + 2
        self.bench_tie_order = bench_total_idx - 1
        self.bench_order_map['total'] = bench_total_idx

        # self.bench_order_map = {
        #     'q_update_target_network':0,
        #     'q_forward':1,
        #     'q_backward':2,
        #     'total':4,
        # }
        # bench_total_idx = max(self.bench_order_map.values()) + 2
        # self.bench_tie_order = bench_total_idx - 1
        # self.bench_order_map['total'] = bench_total_idx

        # self.direc_to_label = dict((direc, self.get_label(i, direc)) for i, direc in enumerate(self.directories))

        # self.df = self.read_plot_data()

        # self.dirs, self.devices = self.ordered_directory_device()

    @staticmethod
    def required_source_basename_regexes():
        return {'breakdown_json':r"^((?P<bench_name>{bench})\.)?breakdown.json$".format(bench=BENCH_NAME_REGEX)}

    @staticmethod
    def target_basename_regexes():
        return {
            "plot_data_path":"summary.plot_data.txt$",
            "png":"summary.png$",
        }

    @staticmethod
    def optional_source_basename_regexes():
        return {'config_json':r"^config.json$"}

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return True

    # @staticmethod
    # def uses_multiple_dirs():
    #     return False

    @staticmethod
    def uses_multiple_dirs():
        return True

    @classmethod
    def get_targets(Klass, src_files, bench_name=NO_BENCH_NAME):
        # TODO: get rid of bench_name (if uses_all_benches is set?)
        # TODO: fields depends on contents of src_files technically... ok to not output a file we say we will output?
        targets = []
        for field in PLOT_SUMMMARY_FIELDS:
            targets.extend([
                Klass.get_plot_data_path(field, src_files),
                Klass.get_png_path(field, src_files),
            ])
        return targets

    def plot(self):
        if self.df is None:
            self._as_dataframe()

        # args = self.args
        # if args.directories is None:
        #     return

        # https://stackoverflow.com/questions/36018681/stop-seaborn-plotting-multiple-figures-on-top-of-one-another
        fig = plt.figure()

        fields = self.get_fields()

        # for field in fields:
        for field in ['PercentTimeInGPU']:
            self.plot_field(field)

    def _order_key(self, order, label):
        return "{order}, {label}".format(order=order, label=label)

    def device_order_key(self, pretty_label):
        if pretty_label not in self.pretty_label_order_map:
            return self._order_key(self.device_tie_order, pretty_label)
        order = self.pretty_label_order_map[pretty_label]
        return self._order_key(order, pretty_label)
    def bench_order_key(self, bench):
        if bench not in self.bench_order_map:
            return self._order_key(self.bench_tie_order, bench)
        order = self.bench_order_map[bench]
        return self._order_key(order, bench)

    def get_label(self, i, direc):
        args = self.args
        if args.plot_labels is not None and i < len(args.plot_labels):
            return args.plot_labels[i]

        label = _b(_d(direc))
        if label in self.pretty_label_map:
            pretty_label = self.pretty_label_map[label]
        else:
            pretty_label = label

        return pretty_label

    def get_json_fields(self, json_data):
        return list(k for k in json_data.keys() if k in PLOT_SUMMMARY_FIELDS)

    def get_fields(self):
        return unique(self.df['field'])

    def run(self, bench_name=NO_BENCH_NAME):
        # for
        #     self.add_json_data()
        # We need to add all the json files, regardless of the field type?
        # for in self.src_files.all_sources(all_bench_names)
        for directory in self.src_files.directories:
            src_files = self.src_files.get_src_files(directory)

            device_name = self.config_get(src_files, 'device_name', NO_DEVICE_NAME)

            bench_names = src_files.bench_names
            for bench in bench_names:
                json_path = src_files.get('breakdown_json', bench)
                json_data = load_json(json_path)
                # TODO: still dqn specific.
                pretty_bench = get_pretty_bench(bench)
                # assert len(bench_names) == 0
                # bench = bench_names[0]
                self.add_json_data(json_data, bench, pretty_bench, device_name)
        self.plot()

    def add_json_data(self, json_data, bench, pretty_bench, device):
        assert device is not None
        for field in self.get_json_fields(json_data):
            for value in json_data[field]:
                self.df_data['field'].append(field)
                self.df_data['value'].append(value)
                # mean = json_data[field]
                # self.df_data['mean'].append(mean)
                # std = json_data[std_field_name(field)]
                # self.df_data['std'].append(std)
                self.df_data['device'].append(device)

                device_order = self.device_order_key(device)
                self.df_data['device_order'].append(device_order)

                bench_order = self.bench_order_key(bench)
                self.df_data['bench_order'].append(bench_order)

                self.df_data['bench'].append(bench)
                self.df_data['pretty_bench'].append(pretty_bench)

    def _sort_df(self, df):
        df = df.sort_values(by=['device_order', 'bench_order'])
        return df

    def _as_dataframe(self):
        self.df = pd.DataFrame(data=self.df_data)
        self.df = self._sort_df(self.df)
        self.df = self.df[self.df['value'] != float('inf')]
        self.df = self.df[self.df['bench'] != 'step']

        self.mean_df = DataFrame.get_mean_std(self.df, self.value_field)

    def ordered_directory_device(self):
        labels = [self.direc_to_label[direc] for direc in self.directories]
        label_order = [self.device_order_key(pretty_label) for pretty_label in labels]
        ordered_labels = sort_xs_by_ys(labels, label_order)
        ordered_directories = sort_xs_by_ys(self.directories, label_order)
        return ordered_directories, ordered_labels

    @staticmethod
    def get_png_path(field, src_files):
        png_path = _j(src_files.directory, "{field}.summary.png".format(field=field))
        return png_path

    def _png_path(self, field):
        return self.get_png_path(field, self.src_files)

    @staticmethod
    def get_plot_data_path(field, src_files):
        png_path = _j(src_files.directory, "{field}.summary.plot_data.txt".format(field=field))
        return png_path

    def _plot_data_path(self, field):
        return self.get_plot_data_path(field, self.src_files)

    def get_ylabel(self, field):
        if field in PLOT_SUMMARY_FIELDS_TIME_SEC + ['Time (seconds)']:
            ylabel = 'Time (seconds)'
        elif field == "TheoreticalSpeedup":
            ylabel = r'($\times$) Speedup = $\frac{time(python + C)}{time(C)}$'
        elif field == "PercentTimeInPython":
            ylabel = r'(%) Percent = $\frac{time(python)}{time(python + C)}$'
        elif field == "PercentTimeInGPU":
            ylabel = r'(%) Percent'
        elif field == "PythonOverheadPercent":
            ylabel = r"(%) Percent overhead = $\frac{time(python)}{time(C)}$"
        else:
            raise NotImplementedError

        return ylabel

    def get_subtitle(self, field):
        if field == "TotalTimeSec":
            subtitle = r"Time spent in python and C $= time(python + C)$"
        elif field == "CppAndGPUTimeSec":
            subtitle = r"Time spent in C + GPU $= time(C + GPU)$"
        elif field == "CppTimeSec":
            subtitle = r"Time spent in C (CUDA C API call, CUDA driver, or Framework C) $= time(C)$"
        elif field == "FrameworkCppTimeSec":
            subtitle = r"Time spent in Framework C $= time(Framework C)$"
        elif field == "CudaCppTimeSec":
            subtitle = r"Time spent in CUDA C API call (NOTE: includes CUDA driver calls) $= time(CUDA C)$"
        elif field == "GPUAndCudaCppTimeSec":
            subtitle = r"Time spent in GPU / CUDA C API call (NOTE: includes CUDA driver calls) $= time(GPU + CUDA C)$"
        elif field == "GPUTimeSec":
            subtitle = r"Time spent in GPU $= time(GPU)$"
        elif field == "TheoreticalSpeedup":
            subtitle = r"Theoretical speedup if $time(python) = 0$"
        elif field == "PercentTimeInPython":
            subtitle = r"Percent time in python $= \frac{time(python)}{time(python + C)}$"
        elif field == "PercentTimeInGPU":
            subtitle = r"Percent time in GPU $= \frac{time(GPU)}{time(python + C)}$"
        elif field == "PythonTimeSec":
            subtitle = r"Time spent in python $= time(python)$"
        elif field == "PythonOverheadPercent":
            subtitle = r"% overhead from python-glue code $= \frac{time(python)}{time(C)}$"
        else:
            raise NotImplementedError

        return subtitle

    def is_time_sec_field(self, field):
        return re.search('TimeSec$', field)


    def _get_min_length(self, fields, device):
        """
        min_length =
          Minimum length across PythonTimeSec (for all benches),
                                CppAndGpuTimeSec (for all benches)

        :param field:
        :param device:
        :return:
        """
        df = self.df
        df = df[df['field'].isin(fields) & (df['device'] == device)]

        groupby = df.groupby(['field', 'bench'])
        agg = groupby.agg(['count'])
        key = agg.keys()[0]
        min_length = agg[key].min()
        return min_length

    def _get_bench_to_df(self, field, device, min_length):
        # Take the first min_length sample of each bench
        df = self.df[
            (self.df['field'] == field) &
            (self.df['device'] == device)]
        bench_names = unique(df['bench'])
        def first_min_length(bench):
            return df[df['bench'] == bench].head(min_length)
        bench_to_df = dict((bench, first_min_length(bench)) for bench in bench_names)

        return bench_to_df

    def _get_time_value(self, field, device, min_length):
        # Doesn't make sense to call this for (for e.g.) percent fields.
        # Only makes sense for TimeSec fields (or other absolute values).
        assert self.is_time_sec_field(field)

        df = self.df[
            (self.df['field'] == field) &
            (self.df['device'] == device)]
        bench_names = unique(df['bench'])

        bench_to_df = self._get_bench_to_df(field, device, min_length)

        # CppAndGPUTimeSec and PythonTimeSec
        value = np.zeros(min_length)
        for b in bench_names:
            # assert pd.isna(bench_to_df[b]['value']).sum() == 0
            # I don't know why, but if we removes ".values" from any of these,
            # we end up with a bunch of NaN's in plot_df['value']
            value = value + bench_to_df[b]['value'].values
            # assert pd.isna(plot_df['value']).sum() == 0
        return value

    def _get_value(self, field, device):
        if self.is_time_sec_field(field):
            min_length = self._get_min_length([field], device)
            value = self._get_time_value(field, device, min_length)
        elif field in ['PercentTimeInGPU', 'PercentTimeInPython', 'TheoreticalSpeedup', 'PythonOverheadPercent']:
            min_length = self._get_min_length(['CppAndGPUTimeSec', 'PythonTimeSec'], device)
            cpp_and_gpu_times = self._get_time_value('CppAndGPUTimeSec', device, min_length)
            python_times = self._get_time_value('PythonTimeSec', device, min_length)
            total_times = compute_total_times(cpp_and_gpu_times, python_times)
            if field == 'TheoreticalSpeedup':
                theoretical_speedup = compute_theoretical_speedup(cpp_and_gpu_times, python_times)
                value = theoretical_speedup
            elif field == 'PercentTimeInPython':
                percent_time_in_python = compute_percent_time_in_python(cpp_and_gpu_times, python_times)
                value = percent_time_in_python
            elif field == 'PercentTimeInGPU':
                gpu_times = self._get_time_value('GPUTimeSec', device, min_length)
                percent_time_in_gpu = compute_percent_time_in_gpu(gpu_times, total_times)
                value = percent_time_in_gpu
            elif field == 'PythonOverheadPercent':
                python_overhead_percent = compute_python_overhead_percent(cpp_and_gpu_times, python_times)
                value = python_overhead_percent
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("Not sure how to get total column for field={field}".format(field=field))

        return value


    def _add_total_rows(self, df, field):
        """
        Add total column to the plot.
        Basically, we need to add the samples across different iterations.
        http://pandas.pydata.org/pandas-docs/stable/groupby.html#applying-multiple-functions-at-once
        To compute TheoreticalSpeedup = time(python + C)/time(C):
        Need to get the first min_length samples from time(python), time(C)
        Then, need to compute:
          plot_df['value'] = (time(python) + time(C)) / time(C)
          plot_df['bench'] = 'total'
        """

        devices = unique(df['device'])

        dfs = []
        for device in devices:

            value = self._get_value(field, device)

            plot_df = pd.DataFrame()
            plot_df['value'] = value
            assert pd.isna(plot_df['value']).sum() == 0
            bench_name = 'total'
            plot_df['field'] = field
            plot_df['bench'] = bench_name
            plot_df['bench_order'] = self.bench_order_key(bench_name)
            plot_df['pretty_bench'] = get_pretty_bench(bench_name)
            plot_df['device'] = device
            plot_df['device_order'] = self.device_order_key(device)
            dfs.append(plot_df)
        total_rows = pd.concat(dfs)
        print("> Added Total column for field={field}:".format(field=field))
        print(total_rows)
        df = pd.concat([total_rows, df])
        df = self._sort_df(df)

        return df

    def plot_field(self, field):

        df = self.df
        df = df[df['field'] == field]


        df = self._add_total_rows(df, field)

        fig = plt.figure()
        # ax = fig.add_subplot(111)
        errorbar_capsize = 0.15
        errorbar_elinewidth = 1

        # colors = LINE_COLORS[:len(self.devices)]

        png_path = self._png_path(field)

        sns.set(style="whitegrid")

        print("> DataFrame:")

        mean_df = DataFrame.get_mean_std(df, self.value_field)
        plot_data_path = self._plot_data_path(field)
        with open(plot_data_path, 'w') as f:
            DataFrame.print_df(mean_df, file=f)
        print(mean_df)
        # df.sort(columns=[''])
        g = sns.catplot(x="pretty_bench", y="value", hue="device", data=df,
                        # col="field",
                        # yerr=df["std"].values,
                        capsize=errorbar_capsize,
                        # elinewidth=errorbar_elinewidth,
                        errwidth=errorbar_elinewidth,
                        height=6, kind="bar", palette="muted")

        g.despine(left=True)
        # https://seaborn.pydata.org/examples/grouped_barplot.html
        ylabel = self.get_ylabel(field)
        g.set_ylabels(ylabel)
        g.set_xlabels("Operation")
        subtitle = self.get_subtitle(field)
        g.set_titles(subtitle)
        plt.title(subtitle)
        g._legend.set_title('Device')
        print("> Save plot to {path}".format(path=png_path))
        print("> Save plot data to {path}".format(path=plot_data_path))
        g.savefig(png_path)

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
# def _prettify(name, profile_basename_glob, ProfileParserKlass, profile_basename_negate_re=None):
#     self.load(reload=True)
#     profile_paths = self.glob_files(profile_basename_glob, basename_negate_re=profile_basename_negate_re)
#     for profile_path in profile_paths:
#         bench_name = get_bench_name(_b(profile_path))
#         prof = ProfileParserKlass(self.parser, self.args, profile_path, bench_name=bench_name, data=self.data)
#         if _e(prof.dump_path):
#             if self.args.replace_parse:
#                 print("> Replacing {f}".format(f=prof.dump_path))


#             elif os.path.getsize(prof.dump_path) == 0:
#                 os.remove(prof.dump_path)
#             else:
#                 # print("> SKIP EXISTS: {f}".format(f=prof.dump_path))
#                 continue
#         # print("> DOES NOT YET EXIST: {f}".format(f=prof.dump_path))
#         print("> Prettify {name} profile: {path}".format(name=name, path=prof.dump_path))
#         prof.parse()
#         prof.dump()

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
        # self.prettify_python_profile()
        # # self.prettify_cuda_profile()
        # self.prettify_cuda_profile_sqlite()
        # self.combine_profiles()
        # self.plot_benchmarks()
        pass

    def run(self):
        args = self.args
        parser = self.parser

        should_run_benchmarks = not args.plot and not args.prettify_python_profile and not args.prettify_cuda_profile

        if should_run_benchmarks and ( args.profile_cuda and not args.nvprof_enabled ):
            run_with_nvprof(parser, args)
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

    def prettify_python_profile(self):
        # Q: What is this code ACTUALLY doing?
        # If profile_path = <args.directory>/python_profile*.txt exists and not matches *call_times*:
        #     bench_name = re.match(r"*.q_forward.*", profile_path)
        #     PythonProfileParser(profile_path, bench_name)
        #
        # TODO: This code ought to run in SConstruct for finding files to run on.
        return self._prettify('python', self.PYTHON_PROFILE_GLOB, PythonProfileParser,
                              profile_basename_negate_re=self.PYTHON_PROFILE_NEGATE_RE)

    def prettify_cuda_profile(self):
        return self._prettify('cuda', self.CUDA_PROFILE_BASENAME_GLOB, CUDAProfileParser)

    def prettify_cuda_profile_sqlite(self):
        return self._prettify('sqlite', self.CUDA_PROFILE_SQLITE_GLOB, CUDASQLiteParser)

def line_iter(f, lstrip=False):
    for line in f:
        line = line.rstrip()
        if lstrip:
            line = line.lstrip()
        yield line

float_re = r'(?:[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'

def put_key(d, key, value):
    if key not in d:
        d[key] = value


def as_value(x):
    if type(x) in [int, float, list, set, dict]:
        return x

    assert type(x) == str

    try:
        val = int(x)
        return val
    except ValueError:
        pass

    try:
        val = float(x)
        return val
    except ValueError:
        pass

    return x

def store_group(dic, m, types=dict(), store_as=None, replace=False):
    groupdict = m.groupdict()
    for k, v in groupdict.items():
        if k in types:
            value = types[k](v)
        else:
            value = as_value(v)

        if store_as is None:
            assert replace or k not in dic
            dic[k] = value
        elif store_as == 'list':
            if k not in dic:
                dic[k] = []
            dic[k].append(value)

def store_as(dic, k, v, types=dict(), store_type=None, replace=False):
    if k in types:
        value = types[k](v)
    else:
        value = as_value(v)

    if store_type is None:
        assert replace or k not in dic
        dic[k] = value
    elif store_type == 'list':
        if k not in dic:
            dic[k] = []
        dic[k].append(value)
    else:
        raise NotImplementedError

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class KernelTime:
    def __init__(self, time_usec, start_usec=None, end_usec=None, name=None):
        self.time_usec = time_usec
        self.start_usec = start_usec
        self.end_usec = end_usec
        self.name = name

    def overlaps(self, ktime_b):
        ktime_a = self
        assert ktime_a.start_usec <= ktime_b.start_usec
        return ktime_a.end_usec > ktime_b.start_usec

    def overlap(self, ktime_b):
        assert self.overlaps(ktime_b)
        ktime_a = self
        return ktime_b.start_usec - ktime_a.end_usec

VARIABLE_HEADER = ['Type', 'Time(%)',
                   'Avg', 'Std', 'Std/Avg(%)',
                   # The n-th call to this function during a repetition.
                   'Call#', 'Name',
                   'Sample#',
                   'Time',
                   # 'Time', 'Calls',
                   # 'Min', 'Max',
                   ]

SEPARATE_CALLS_HEADER = ['Type', 'Time(%)',
                         'Avg', 'Std', 'Std/Avg(%)',
                         # The n-th call to this function during a repetition.
                         'Call#', 'Name',
                         'Time', 'Calls',
                         'Min', 'Max',
                         ]

class Stat:
    """
    Compute either the sum/min/avg/stdev of calls to a function.
    """
    def __init__(self, name, discard_first_sample, debug=False):
        self.kernel_times = []
        # self.num_calls = num_calls
        # self._call_num = 0
        self.num_calls = None
        self.debug = debug
        self.name = name
        self.discard_first_sample = discard_first_sample

    def add(self, time_usec, start_usec=None, end_usec=None):
        assert self.num_calls is None
        self.kernel_times.append(KernelTime(time_usec, start_usec, end_usec, name=self.name))

    def add_times_sec(self, times_sec):
        for time_sec in times_sec:
            time_usec = sec_to_us(time_sec)
            self.kernel_times.append(KernelTime(time_usec, name=self.name))

    def iteration_times_sec(self, num_calls):
        """
        :param num_calls:
            Total number of iterations (iters*reps)
        :return:
        """

        total_iterations = self.get_total_iterations(num_calls)
        # n_calls = len(self.kernel_times)
        # if n_calls % num_calls != 0:
        # if self.num_diff_calls == 1:
        if self.num_calls == 1:
            # Evenly divide the total time spent calling this function across all num_calls iterations.
            ret = np.empty(total_iterations)
            fill = (self.sum()/len(ret))/MICROSECONDS_IN_SECOND
            ret.fill(fill)
            return ret

        # Say we have num_calls=1000 calls to Forward, but we have 22000 calls to this function.
        # Then each Forward has num_diff_calls=22 calls to this function.
        # We want to sum up those 22 calls (for each call to Forward) to get an array of 1000.
        ret = np.zeros(total_iterations)
        for call_idx in range(self.num_diff_calls):
            times_sec = self.times_sec(call_idx)
            if len(times_sec) != len(ret):
                import ipdb; ipdb.set_trace()
            ret += times_sec

        return ret

    def all_calls_during(self, call_num):
        """
        Return all the (e.g. 22) times this API call was made during the <call_num>-th call to Forward.

        :param call_num:
            call_num = One of the times Forward was called.
            call_num = 1...1000 if iterations*repetitions = 1000
        """
        assert 0 <= call_num < self.num_calls
        times = []
        for call_idx in range(self.num_diff_calls):
            times.append(self.kernel_times[call_idx][call_num])
        return times

    def split(self, num_calls):
        """
        Q: Am I dividing up calls correctly?
        If we call cudaLaunch 5 times for each iteration, and the number of iterations/calls is 1000, calls will look like:Initially, kernel_times contains every time the function was ever called.
        kernel_times[0..4999] = [
            # First iteration…
            cudaLaunch[i=0]
            cudaLaunch[i=1]
            cudaLaunch[i=2]
            cudaLaunch[i=3]
            cudaLaunch[i=4]
            # Next iteration…
            cudaLaunch[i=5]
            cudaLaunch[i=6]
            cudaLaunch[i=7]
            cudaLaunch[i=8]
            cudaLaunch[i=9]
            ...
            # 5 * 1000 = 5000 calls in total
        ]

        Once we find out the number of iterations (num_calls) is 1000, we discover num_diff_calls=5000/1000 = 5. We use this to split up the times into calls with the same arguments
        kernel_times[0..4] = [
            [
                # Take the 1st call from each iteration-group
                cudaLaunch[i=0],
                cudaLaunch[i=5],
                ...
                # 1000 calls to cudaLaunch with the same arguments
            ]
            [
                # Take the 2nd call from each iteration-group
                cudaLaunch[i=1],
                cudaLaunch[i=6],
                ...
            ]
            ...
            [
                # Take the 5th call from each iteration-group
                cudaLaunch[i=4],
                cudaLaunch[i=9],
                ...
            ]
            # 5 "different" calls to cudaLaunch during a single call to Forward.
        ]

        :param num_calls:
        The number of times (e.g.) Forward was called.
        num_calls = iterations * repetitions
        :return:
        """

        # n_calls = the number of times this function was called.
        n_calls = len(self.kernel_times)
        # if self.discard_first_sample:
        #     n_calls = n_calls - 1

        if n_calls % num_calls != 0:
            # Number of calls to this function isn't divisible by the number
            # of times we expect it to have been called (num_calls = iterations*repetitions);
            # instead, just make num_calls = 1.
            if self.debug:
                print("[n_calls={n_calls}, num_calls={num_calls}] Use num_calls=1 for function={name}".format(
                    n_calls=n_calls,
                    num_calls=num_calls,
                    name=self.name))
            kernel_times = [self.kernel_times]
            self.kernel_times = kernel_times
            self.num_calls = 1
            self.num_diff_calls = 1
            self.not_divisible = True
            return

        # num_diff_calls = # of times this function is called (likely with different arguments)
        #                  during a single call to Forward
        self.num_diff_calls = int(n_calls / num_calls)

        if self.debug:
            print("[n_calls={n_calls}, num_calls={num_calls}] Use num_calls={num_calls} for function={name}".format(
                n_calls=n_calls,
                num_calls=num_calls,
                name=self.name))
        self.num_calls = num_calls
        self.not_divisible = False
        kernel_times = [[] for i in range(self.num_diff_calls)]
        for i, kt in enumerate(self.kernel_times):
            # Q: Initially, kernel_times[0...num_diff_calls] are different calls to the same function,
            # for one call to Forward.
            call_idx = i % self.num_diff_calls
            kernel_times[call_idx].append(kt)
        self.kernel_times = kernel_times

    def _check_call_idx(self, call_idx):
        # if call_idx is None:
        #     assert self.num_calls is None
        # else:
        if call_idx is not None:
            assert call_idx < self.num_diff_calls

    def _maybe_discard(self, times):
        if self.discard_first_sample and len(times) > 1:
            return times[1:]
        return times

    def times_sec(self, call_idx=None):
        # JAMES TODO: This looks wrong with clock_monotonic...when did this make any sense?
        time_usecs = self._times_usec(call_idx)
        return [usec/MICROSECONDS_IN_SECOND for usec in time_usecs]

    def _times_usec(self, call_idx=None):
        self._check_call_idx(call_idx)

        """
        self.kernel_times:
        
        num_calls = the number of times Forward was called (iterations*repetitions)
        
        Before calling self.split(num_calls):
            self.num_calls = None
            kernel_times = [
                all the times this API call was ever made, across all 1000 iterations*repetitions calls to Forward
            ].
        
        After call self.split(num_calls):
            self.num_calls = num_calls
            kernel_times = [
                [the 1st time the API call was made, across all 1000 iterations*repetitions calls to Forward],
                [the 2nd time the API call was made, across all 1000 iterations*repetitions calls to Forward],
                ...
                [the <num_diff_calls>th time the API call was made ....],
            ]
        NOTE: 
        """
        if self.num_calls is None:
            assert call_idx is None
            times = self._maybe_discard([kt.time_usec for kt in self.kernel_times])
        elif call_idx is not None:
            times = self._maybe_discard([kt.time_usec for kt in self.kernel_times[call_idx]])
        else:
            # Return min/max/avg/etc over ALL calls.
            # Useful when we want to do kt.sum().
            times = []
            for kts in self.kernel_times:
                for kt in self._maybe_discard(kts):
                    times.append(kt.time_usec)

        return times

    @property
    def total_iterations(self):
        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    def get_total_iterations(self, num_calls):
        if num_calls == 1:
            return num_calls
        if self.discard_first_sample:
            return num_calls - 1
        return num_calls

    def n_calls(self, call_idx=None):
        """
        Total number of times this function was called.
        :return:
        """
        self._check_call_idx(call_idx)

        n_calls = 0
        if self.num_calls is None:
            assert call_idx is None
            n_calls = len(self._maybe_discard(self.kernel_times))
        elif call_idx is not None:
            n_calls = len(self._maybe_discard(self.kernel_times[call_idx]))
        else:
            # Return min/max/avg/etc over ALL calls.
            # Useful when we want to do kt.sum().
            for kts in self.kernel_times:
                n_calls += len(self._maybe_discard(kts))

        return n_calls

    def avg(self, call_idx=None):
        self._check_call_idx(call_idx)
        mean_usec = np.mean(self._times_usec(call_idx))
        return mean_usec

    def std(self, call_idx=None):
        self._check_call_idx(call_idx)
        std_usec = np.std(self._times_usec(call_idx))
        return std_usec

    def sum(self, call_idx=None):
        self._check_call_idx(call_idx)
        sum_usec = sum(self._times_usec(call_idx))
        return sum_usec

    def min(self, call_idx=None):
        self._check_call_idx(call_idx)
        min_usec = min(self._times_usec(call_idx))
        return min_usec

    def max(self, call_idx=None):
        self._check_call_idx(call_idx)
        max_usec = max(self._times_usec(call_idx))
        return max_usec

    def calls(self, call_idx=None):
        self._check_call_idx(call_idx)
        return len(self._times_usec(call_idx))

    def dump(self, writer, summary_type, header, total_time, profile_data_type):
        if summary_type == 'nvprof':
            self.dump_nvprof(writer, header, total_time, profile_data_type)
        elif summary_type == 'separate_calls':
            self.dump_separate_calls(writer, header, total_time, profile_data_type)
        else:
            raise NotImplementedError

    def _ptime(self, usec):
        return pretty_time(time_sec=us_to_sec(usec), use_space=False)

    def _percent(self, percent):
        return "{percent:.2f}%".format(percent=100.*percent)

    def dump_variable(self, variable_writer, total_time, profile_data_type):
        """
        If Time(%) >= 1% and Std/Avg > 50%:
          Then report individual timings to nvprof.pretty.variable.txt:
          Same columns as before, but add Sample# column that goes from 1..1000
        """
        for call_idx in range(self.num_diff_calls):
            avg = self.avg(call_idx)
            std = self.std(call_idx)
            sm = self.sum(call_idx)
            time_percent = 100.*sm/total_time
            std_avg_percent = 100.*std/avg
            if time_percent >= 1. and std_avg_percent >= 50.:
                for sample_idx, time_usec in enumerate(self._times_usec(call_idx)):
                    row = {
                        'Type':profile_data_type,
                        'Time(%)':self._percent(sm/total_time),
                        'Std/Avg(%)':self._percent(std/avg),
                        'Time/Avg(%)':self._percent(time_usec/avg),
                        # 'Time':self._ptime(sm),
                        'Call#':call_idx,
                        'Name':self.name,
                        'Sample#':sample_idx,
                        'Time':self._ptime(time_usec),
                        'Avg':self._ptime(avg),
                        'Std':self._ptime(std),
                        # 'Calls':self.calls(call_idx),
                        # 'Avg':self._ptime(self.avg(call_idx)),
                        # 'Std':self._ptime(self.std(call_idx)),
                        # 'Std/Avg(%)':self._percent(self.std(call_idx)/self.avg(call_idx)),
                        # 'Min':self._ptime(self.min(call_idx)),
                        # 'Max':self._ptime(self.max(call_idx)),
                    }
                    variable_writer.writerow([row[k] for k in VARIABLE_HEADER])

    def dump_separate_calls(self, writer, header, total_time, profile_data_type):
        # Q: How to handle diff calls?
        # A:
        # # (a) create a Stat object for each Call#, and output a row for it.
        # # (b) split self.kernel_times for each Call#;
        #       adjust avg/sum/etc. to take a Call# index that determines over which calls to compute the statistic.
        #       In this case, the regular nvprof output is a special-case where the Call# is always 1.
        assert self.num_diff_calls is not None
        for call_idx in range(self.num_diff_calls):
            row = {
                'Type':profile_data_type,
                'Time(%)':self._percent(self.sum(call_idx)/total_time),
                'Time':self._ptime(self.sum(call_idx)),
                'Calls':self.calls(call_idx),
                'Avg':self._ptime(self.avg(call_idx)),
                'Std':self._ptime(self.std(call_idx)),
                'Std/Avg(%)':self._percent(self.std(call_idx)/self.avg(call_idx)),
                'Min':self._ptime(self.min(call_idx)),
                'Max':self._ptime(self.max(call_idx)),
                'Call#':call_idx,
                'Name':self.name,
            }
            writer.writerow([row[k] for k in header])

    def dump_nvprof(self, writer, header, total_time, profile_data_type):
        row = {
            'Type':profile_data_type,
            'Time(%)':"{percent:.2f}%".format(percent=100.*(self.sum()/total_time)),
            'Time':self._ptime(self.sum()),
            'Calls':self.calls(),
            'Avg':self._ptime(self.avg()),
            'Min':self._ptime(self.min()),
            'Max':self._ptime(self.max()),
            'Name':self.name,
        }
        writer.writerow([row[k] for k in header])

def start_end_nsec_to_usec(start_nsec, end_nsec):
    return nsec_to_usec(end_nsec - start_nsec)

def nsec_to_usec(nsec):
    return nsec/1e3

class KernelStat(Stat):
    def __init__(self, name, discard_first_sample, debug=False):
        super().__init__(name, discard_first_sample, debug=debug)

class Stats:
    def __init__(self, discard_first_sample, debug=False, name=None, has_overlap=True):
        self.discard_first_sample = discard_first_sample
        self.name_to_stat = dict()
        self.num_calls = None
        self.debug = debug
        self.name = name
        self.has_overlap = has_overlap

    def sum_calls_sec(self):
        """
        Returns total_times[call_num]

        for call_num = 1...1000 if Forward is called iterations*repetitions=1000 times
        """
        assert self.num_calls is not None
        total_times = np.zeros(self.num_calls)

        equally_divided_times = np.zeros(self.num_calls)
        api_calls_not_divisible = 0
        api_calls_divisible = 0
        for stat in self.stats:
            # If this is a call that isn't divisible by num_calls, then instead,
            # we'd like to evenly divide its time across each iteration.
            #
            # TODO: we should really make sure this isn't a large contributor...
            if stat.not_divisible:
                api_calls_not_divisible += 1
                total_time_sec = np.sum(stat.times_sec())
                equally_divided_times += total_time_sec/self.num_calls

        for call_num in range(self.num_calls):
            ktimes = []
            for stat in self.stats:
                if stat.not_divisible:
                    continue
                ktimes.extend(stat.all_calls_during(call_num))
            api_calls_divisible = len(ktimes)
            ktimes.sort(key=lambda k: k.start_usec)
            total_time = 0
            for ktime_a, ktime_b in zip(ktimes, ktimes[1:]):
                if ktime_a.overlaps(ktime_b):
                    # This warning goes off a LOT for CUDA API stuff.
                    #
                    # if not self.has_overlap:
                    #     print(textwrap.dedent("""
                    #     WARNING: ktime_a={a} overlaps ktime_b={b}
                    #     > {a}
                    #       start = {a_start}
                    #       end = {a_end}
                    #     > {b}
                    #       start = {b_start}
                    #       end = {b_end}
                    #     """.format(
                    #         a=ktime_a.name,
                    #         a_start=ktime_a.start_usec,
                    #         a_end=ktime_a.end_usec,
                    #         b=ktime_b.name,
                    #         b_start=ktime_b.start_usec,
                    #         b_end=ktime_b.end_usec,
                    #     )))
                    overlap = ktime_a.overlap(ktime_b)
                    total_time += ktime_a.time_usec - overlap
                else:
                    total_time += ktime_a.time_usec
            if len(ktimes) > 0:
                total_time += ktimes[-1].time_usec

            total_times[call_num] = total_time

        if self.debug:
            print(textwrap.dedent("""
            > {name} stats:
              num_calls = {num_calls}
              api_calls.not_divisible = {not_divisible}
              api_calls.divisible = {divisible}
            """.format(
                name=self.name,
                num_calls=self.num_calls,
                not_divisible=api_calls_not_divisible,
                divisible=api_calls_divisible,
            )))


        return total_times/MICROSECONDS_IN_SECOND

    def sum_calls_sec_no_overlap(self, check_overlap=False):
        """
        Returns total_times[call_num]

        for call_num = 1...1000 if Forward is called iterations*repetitions=1000 times
        """

        if check_overlap:
            self.check_overlap()

        total_times = np.zeros(self.num_calls)
        for stat in self.stats:
            times_sec = stat.iteration_times_sec(self.num_calls)
            total_times += times_sec
        return total_times

    def check_overlap(self):
        """
        Make sure the times that get summed together for a given call_num do NOT overlap with each other.
        This could happen if, for example, a CUDA kernel and cuda Memcpy run at the same time!
        :return:
        """

        # Looks like overlap CAN happen, so we must account for it:
        #
        # ipdb> pp ktime_a.__dict__
        # {'end_usec': 1542751119289568.2,
        #  'name': 'void cudnn::detail::implicit_convolve_sgemm<float, float, 128, 5, 5, '
        #          '3, 3, 3, 1, true, false, true>(int, int, int, float const*, int, '
        #          'float*, float*, kernel_conv_params, int, float, float, int, float*, '
        #          'float*, int, int)',
        #  'start_usec': 1542751119289384.0,
        #  'time_usec': 184.126}
        # ipdb> pp ktime_b.__dict__
        # {'end_usec': 1542751119289412.8,
        #  'name': '[CUDA memcpy HtoD]',
        #  'start_usec': 1542751119289411.8,
        #  'time_usec': 0.992}

        def overlaps(ktime_a, ktime_b):
            # ktime_a.end_usec and ktime_b.start_usec CAN be equal:
            #
            # e.g.
            # ipdb> pp ktime_a.__dict__
            # {'end_usec': 1542751117249387.2,
            #  'name': 'void '
            #          'Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long '
            #          'long, 1, 1, int>, 16, Eigen::MakePointer>, '
            #          'Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, '
            #          '1, int>, 16, Eigen::MakePointer> const, '
            #          'Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, '
            #          'Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long '
            #          'const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, '
            #          'Eigen::GpuDevice>, '
            #          'int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long '
            #          'long, 1, 1, int>, 16, Eigen::MakePointer>, '
            #          'Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, '
            #          '1, int>, 16, Eigen::MakePointer> const, '
            #          'Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, '
            #          'Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long '
            #          'const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, '
            #          'Eigen::GpuDevice>, int)',
            #  'start_usec': 1542751117249385.0,
            #  'time_usec': 2.176}
            # ipdb> pp ktime_b.__dict__
            # {'end_usec': 1542751117249388.5,
            #  'name': '[CUDA memcpy DtoH]',
            #  'start_usec': 1542751117249387.2,
            #  'time_usec': 1.312}
            assert ktime_a.start_usec <= ktime_b.start_usec
            return ktime_a.overlaps(ktime_b)

        def _check_overlap(kernel_times):
            for k in kernel_times:
                assert k.start_usec is not None and k.end_usec is not None

            sorted_ktimes = sorted(kernel_times, key=lambda k: k.start_usec)
            for ktime_a, ktime_b in zip(sorted_ktimes, sorted_ktimes[1:]):
                assert not overlaps(ktime_a, ktime_b)

        assert self.num_calls is not None
        for call_num in range(self.num_calls):
            all_times = []
            for stat in self.stats:
                all_times.extend(stat.all_calls_during(call_num))
            _check_overlap(all_times)

    @property
    def total_iterations(self):
        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    def split(self, num_calls):
        self.num_calls = num_calls
        with progressbar.ProgressBar(max_value=len(self.name_to_stat.values())) as bar:
            for i, kt in enumerate(self.name_to_stat.values()):
                kt.split(num_calls)
                bar.update(i)

    def add(self, name, time_usec,
            start_usec=None, end_usec=None):
        assert (start_usec is None and end_usec is None) or \
               (start_usec is not None and end_usec is not None)
        kt = self._get_stat(name)
        kt.add(time_usec, start_usec=start_usec, end_usec=end_usec)

    def _get_stat(self, name):
        if name not in self.name_to_stat:
            self.name_to_stat[name] = KernelStat(name, self.discard_first_sample, debug=self.debug)
        return self.name_to_stat[name]

    def add_times_sec(self, name, times_sec):
        stat = self._get_stat(name)
        stat.add_times_sec(times_sec)

    def dump(self, f, profile_data_type, skip_header=False, summary_type='nvprof'):
        if summary_type == 'nvprof':
            self.dump_nvprof(f, profile_data_type, skip_header)
        elif summary_type == 'separate_calls':
            self.dump_separate_calls(f, profile_data_type, skip_header)
        else:
            raise NotImplementedError

    def dump_nvprof(self, f, profile_data_type, skip_header=False):
        """
        Dump the same "summary" that nvprof outputs.
        """
        writer = csv.writer(f, delimiter='|')
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        header = ['Type', 'Time(%)', 'Time', 'Calls', 'Avg', 'Min', 'Max', 'Name']
        if not skip_header:
            writer.writerow(header)
        total_time = self.total_time()
        for kt in stats:
            kt.dump(writer, 'nvprof', header, total_time, profile_data_type)

    def dump_variable(self, f, profile_data_type, skip_header=False):
        variable_writer = csv.writer(f, delimiter='|')
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        # Q: What do we really want to know?
        # A: What's the mean/stdev time spent in each function called during a single Forward call?
        # To answer this, we need to separate the profile output into the n-th calls to the function.
        # If 'Calls' isn't divisible by num_calls, either:
        # (a) remove it entirely from the profile output, or [ PROBLEM: it might take up a LOT of time ]
        # (b) treat it as a function that gets called once
        #     Call# = 0 or N/A?
        #     Stdev = stdev of all n-calls (stdev will be big)
        #     ^^^ I would like to show this, since if the Stdev is tiny, we don't care; if it's big, we care.
        # (c) let Call# = N/A
        #         Stdev = N/A
        #     ^^^ This is more accuracte.
        #
        # PSEUDOCODE:
        # num_calls = the total number of times Forward was called.
        if not skip_header:
            variable_writer.writerow(VARIABLE_HEADER)
        total_time = self.total_time()
        for kt in stats:
            kt.dump_variable(variable_writer, total_time, profile_data_type)

    @property
    def stats(self):
        return self.name_to_stat.values()

    @property
    def ordered_stats(self):
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        return stats

    def total_time(self):
        total_time = 0
        for kt in self.name_to_stat.values():
            total_time += kt.sum()
        return total_time

    def dump_separate_calls(self, f, profile_data_type, skip_header=False):
        """
        Dump the same "summary" that nvprof outputs.
        """
        writer = csv.writer(f, delimiter='|')
        stats = sorted(self.name_to_stat.values(), key=lambda kt: -1*kt.sum())
        # Q: What do we really want to know?
        # A: What's the mean/stdev time spent in each function called during a single Forward call?
        # To answer this, we need to separate the profile output into the n-th calls to the function.
        # If 'Calls' isn't divisible by num_calls, either:
        # (a) remove it entirely from the profile output, or [ PROBLEM: it might take up a LOT of time ]
        # (b) treat it as a function that gets called once
        #     Call# = 0 or N/A?
        #     Stdev = stdev of all n-calls (stdev will be big)
        #     ^^^ I would like to show this, since if the Stdev is tiny, we don't care; if it's big, we care.
        # (c) let Call# = N/A
        #         Stdev = N/A
        #     ^^^ This is more accuracte.
        #
        # PSEUDOCODE:
        # num_calls = the total number of times Forward was called.
        # if self.debug:
        #     import ipdb; ipdb.set_trace()
        if not skip_header:
            writer.writerow(SEPARATE_CALLS_HEADER)
        total_time = 0
        for kt in stats:
            total_time += kt.sum()
        for kt in stats:
            kt.dump(writer, 'separate_calls', SEPARATE_CALLS_HEADER, total_time, profile_data_type)

# Here's the "API calls" summary output from "nvprof -i file.nvprof" for Forward.
#
# 76.69%  2.60481s     31000  84.026us  5.7450us  711.15ms  cudaLaunch
# 9.50%  322.74ms      1000  322.74us  3.9820us  425.32us  cuCtxSynchronize
# 4.92%  167.25ms     18000  9.2910us     368ns  350.38us  cuEventRecord
# 3.77%  128.08ms      6000  21.346us  7.2610us  378.31us  cuMemcpyDtoHAsync
# 1.74%  59.123ms      3000  19.707us  6.4430us  74.321us  cuMemcpyHtoDAsync
# 0.92%  31.369ms     23532  1.3330us     485ns  344.94us  cuEventQuery
# 0.75%  25.609ms      1000  25.608us  16.958us  355.11us  cudaMemcpyAsync
# 0.49%  16.504ms     92000     179ns     113ns  333.72us  cudaSetupArgument
# 0.47%  15.835ms      3000  5.2780us  1.3280us  62.487us  cudaEventRecord
# 0.38%  13.013ms     31000     419ns     143ns  337.91us  cudaConfigureCall
# 0.29%  9.7108ms      9000  1.0780us     577ns  333.85us  cuStreamWaitEvent
# 0.08%  2.6487ms     12000     220ns     108ns  326.60us  cudaGetLastError

# Here's my manually generated "API calls" summary output for Forward, by querying the SQLite3 database file.
#
# Time(%)|Time|Calls|Avg|Min|Max|Name
# 76.69|2.6048135660000136sec|31000|8.40262440645161e-05|5.745e-06|0.71114529|13
# 9.50|322.7421669999999ms|1000|0.0003227421669999999|3.982e-06|0.000425317|17
# 4.92|167.2546419999997ms|18000|9.291924555555556e-06|3.68e-07|0.000350382|119
# 3.77|128.0785700000002ms|6000|2.1346428333333336e-05|7.2610000000000004e-06|0.000378309|279
# 1.74|59.123038000000015ms|3000|1.9707679333333333e-05|6.443e-06|7.4321e-05|277
# 0.92|31.369078999999882ms|23532|1.3330392231854497e-06|4.85e-07|0.000344937|120
# 0.75|25.608579000000034ms|1000|2.5608579e-05|1.6958e-05|0.00035511|41
# 0.49|16.50449599999818ms|92000|1.7939669565217395e-07|1.13e-07|0.000333715|9
# 0.47|15.835319999999976ms|3000|5.2784399999999996e-06|1.328e-06|6.2487e-05|135
# 0.38|13.013477000000066ms|31000|4.197895806451613e-07|1.43e-07|0.000337914|8
# 0.29|9.71083400000003ms|9000|1.0789815555555554e-06|5.769999999999999e-07|0.00033384699999999996|295
# 0.08|2.6487239999999996ms|12000|2.2072700000000002e-07|1.08e-07|0.0003266|10

# From the above outputs, we can disambiguate which cbid maps to which CUDA function call:
CBID_TO_CUDA_FUNC = {
    13:'cudaLaunch',
    17:'cuCtxSynchronize',
    119:'cuEventRecord',
    279:'cuMemcpyDtoHAsync',
    277:'cuMemcpyHtoDAsync',
    120:'cuEventQuery',
    41:'cudaMemcpyAsync',
    9:'cudaSetupArgument',
    135:'cudaEventRecord',
    8:'cudaConfigureCall',
    295:'cuStreamWaitEvent',
    10:'cudaGetLastError',
}

# Here's the "GPU activities" part of the output from nvprof:
#
# Type  Time(%)      Time     Calls       Avg       Min       Max  Name
# 68.88%  446.48ms      2000  223.24us  2.0160us  537.38us  void gemv2N_kernel_val<float, float, float, int=128, int=32, int=4, int=4, int=1>(float, float, cublasGemv2Params_v2<float, float, float>)
# 8.82%  57.172ms      1000  57.171us  56.128us  58.432us  maxwell_scudnn_128x64_relu_small_nn
# 3.99%  25.871ms      1000  25.871us  24.672us  26.656us  maxwell_scudnn_128x32_relu_medium_nn
# 3.17%  20.540ms      1000  20.540us  18.657us  21.952us  maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148n_nt
# 1.85%  11.977ms      3000  3.9920us     832ns  11.456us  [CUDA memcpy HtoD]
# 1.46%  9.4930ms      3000  3.1640us  2.0160us  4.0640us  void tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>(int, float const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>*)
# 1.39%  8.9873ms      1000  8.9870us  8.7360us  10.208us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>(unsigned char const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>*)
# 1.24%  8.0238ms      3000  2.6740us  2.2080us  3.7440us  void tensorflow::BiasNCHWKernel<float>(int, float const *, float const , tensorflow::BiasNCHWKernel<float>*, int, int)
# 1.20%  7.8063ms      1000  7.8060us  7.5200us  8.0640us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>)
# 0.98%  6.3426ms      1000  6.3420us  6.2080us  6.4960us  void tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>(int, float const *, tensorflow::functor::Dimension<int=4>, tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>*, tensorflow::functor::Dimension, float const *)
# 0.95%  6.1683ms      6000  1.0280us     416ns  2.0160us  [CUDA memcpy DtoH]
# 0.92%  5.9620ms      4000  1.4900us  1.1200us  2.0160us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const , float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const , Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(float, int=1)
# 0.76%  4.8959ms      1000  4.8950us  3.6480us  5.5680us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>(unsigned int const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>*)
# 0.74%  4.7984ms      1000  4.7980us  4.0960us  6.9760us  void cudnn::winograd::generateWinogradTilesKernel<int=0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)
# 0.67%  4.3490ms      1000  4.3490us  4.0960us  4.7680us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<__int64, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float>>, Eigen::array<long, unsigned long=1> const , Eigen::TensorMap<Eigen::Tensor<float const , int=2, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(__int64, int=1)
# 0.64%  4.1744ms      1000  4.1740us  3.7440us  4.9280us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>)
# 0.61%  3.9572ms      2000  1.9780us  1.7600us  3.4880us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(float, int=1)
# 0.48%  3.1422ms      2000  1.5710us  1.3440us  1.9840us  cudnn::maxwell::gemm::computeOffsetsKernel(cudnn::maxwell::gemm::ComputeOffsetsParams)
# 0.40%  2.6120ms      2000  1.3050us     992ns  1.6640us  void tensorflow::BiasNHWCKernel<float>(int, float const *, float const , tensorflow::BiasNHWCKernel<float>*, int)
# 0.25%  1.5927ms      1000  1.5920us  1.3760us  1.7920us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(__int64, int=1)
# 0.21%  1.3723ms      1000  1.3720us  1.3120us  1.5040us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
# 0.21%  1.3680ms      1000  1.3670us  1.2160us  1.5040us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
# 0.18%  1.1568ms      1000  1.1560us     928ns  1.4400us  [CUDA memcpy DtoD]

# Here's my manaully generated "GPU activities":
# Time(%)|Time|Calls|Avg|Min|Max|Name
# 68.88|446.4804500000032ms|2000|0.00022324022499999998|2.016e-06|0.000537379|void gemv2N_kernel_val<float, float, float, 128, 32, 4, 4, 1>(float, float, cublasGemv2Params_v2<float, float, float>)
# 8.82|57.171979ms|1000|5.717197900000001e-05|5.6128e-05|5.8432e-05|maxwell_scudnn_128x64_relu_small_nn
# 3.99|25.871324000000005ms|1000|2.5871324e-05|2.4672e-05|2.6656e-05|maxwell_scudnn_128x32_relu_medium_nn
# 3.17|20.54018699999997ms|1000|2.0540186999999998e-05|1.8657e-05|2.1952000000000003e-05|maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148n_nt
# 1.85|11.976681999999892ms|3000|3.992227333333333e-06|8.319999999999999e-07|1.1456e-05|1
# 1.46|9.492982000000076ms|3000|3.1643273333333336e-06|2.016e-06|4.064e-06|void tensorflow::functor::SwapDimension0And2InTensor3Simple<float, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)
# 1.39|8.987316999999981ms|1000|8.987317000000001e-06|8.736e-06|1.0208e-05|void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 1024, 1024, 2, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)
# 1.24|8.02377599999999ms|3000|2.674591999999999e-06|2.2080000000000003e-06|3.744e-06|void tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)
# 1.20|7.806344000000012ms|1000|7.806344e-06|7.519999999999999e-06|8.064e-06|void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, long long> >(tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, long long>::ResultElementType*, long long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, long long>)
# 0.98|6.342567000000009ms|1000|6.342567000000001e-06|6.2080000000000005e-06|6.496e-06|void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>)
# 0.95|6.16826100000015ms|6000|1.0280435000000004e-06|4.1599999999999997e-07|2.016e-06|2
# 0.92|5.961957999999928ms|4000|1.4904895000000004e-06|1.12e-06|2.016e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)
# 0.76|4.895866000000007ms|1000|4.8958659999999995e-06|3.648e-06|5.568e-06|void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)
# 0.74|4.798367999999998ms|1000|4.798367999999999e-06|4.096e-06|6.976e-06|void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)
# 0.67|4.349043000000004ms|1000|4.349043e-06|4.096e-06|4.768e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)
# 0.64|4.1744210000000015ms|1000|4.174421000000001e-06|3.744e-06|4.928e-06|void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)
# 0.61|3.957243000000013ms|2000|1.9786215e-06|1.76e-06|3.488e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)
# 0.48|3.14222300000002ms|2000|1.5711115000000003e-06|1.3440000000000002e-06|1.984e-06|cudnn::maxwell::gemm::computeOffsetsKernel(cudnn::maxwell::gemm::ComputeOffsetsParams)
# 0.40|2.6119829999999906ms|2000|1.3059915000000002e-06|9.92e-07|1.6639999999999999e-06|void tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)
# 0.25|1.592743999999982ms|1000|1.5927440000000002e-06|1.3759999999999998e-06|1.792e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)
# 0.21|1.372327000000001ms|1000|1.372327e-06|1.312e-06|1.504e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)
# 0.21|1.3679750000000017ms|1000|1.3679750000000002e-06|1.2159999999999999e-06|1.504e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)
# 0.18|1.1568079999999958ms|1000|1.156808e-06|9.28e-07|1.44e-06|8

# enum cudaMemcpyKind:
#   cudaMemcpyHostToHost       Host -> Host
#   cudaMemcpyHostToDevice     Host -> Device
#   cudaMemcpyDeviceToHost     Device -> Host
#   cudaMemcpyDeviceToDevice   Device -> Device
#   cudaMemcpyDefault          Default based unified virtual address space
COPYKIND_TO_MEMCPY = {
    1:'[CUDA memcpy HtoD]',
    2:'[CUDA memcpy DtoH]',
    8:'[CUDA memcpy DtoD]',
}

def compute_num_calls(bench_data):
    # This is the number of times (for e.g.) Forward was called.
    # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
    # num_calls = iterations*repetitions
    #
    # NOTE: +1 for the extra initial call we do to account for profiler weirdness during warmup (i.e. nvprof).

    num_calls = bench_data['iterations']*bench_data['repetitions'] + 1
    # num_calls = bench_data['iterations']*bench_data['repetitions']

    return num_calls

BENCH_SUFFIX_RE = r"(:?\.(?P<bench_name>{bench}))?".format(bench=BENCH_NAME_REGEX)
BENCH_PREFIX_RE = r"(:?(?P<bench_name>{bench})\.)?".format(bench=BENCH_NAME_REGEX)

class CUDASQLiteParser(ProfilerParserCommonMixin):
    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths
        self.src_files = src_files

        assert cxxfilt is not None
        self.parser = parser
        self.args = args
        self.bench_name = bench_name
        self.data = data
        self.skip = False
        self.conn = None

        self.config_path = src_files.get('config_json', bench_name, or_none=True)
        if self.config_path is not None:
            self.config = load_json(self.config_path)
            print("> Found optional config_json @ {f}".format(f=self.config_path))
        else:
            self.config = {
            }

        self.num_calls = self._parse_num_calls(bench_name)

        self.discard_first_sample = self.config.get('discard_first_sample', self.args.discard_first_sample)

        self.kernel_stats = Stats(self.discard_first_sample, debug=self.args.debug, name="CUDA_Kernel_stats",
                                  has_overlap=True)
        self.api_stats = Stats(self.discard_first_sample, debug=self.args.debug, name="CUDA_API_stats",
                               # I don't expect CUDA API calls to overlap (at least, for the single-machine,
                               # single-GPU workloads I'm looking at).
                               has_overlap=False)

    @staticmethod
    def required_source_basename_regexes():
        return {'profile_path': r"^nvidia{bench}\.nvprof$".format(bench=BENCH_SUFFIX_RE)}

    @staticmethod
    def target_basename_regexes():
        return {
            'gpu_overhead_json': r"^nvidia{bench}\.gpu_overhead\.nvprof\.json$".format(bench=BENCH_SUFFIX_RE),
            'pretty_profile_path': r"^nvidia{bench}\.nvprof\.pretty\.txt$".format(bench=BENCH_SUFFIX_RE),
            'variable_path': r"^nvidia{bench}\.variable\.pretty\.txt$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def optional_source_basename_regexes():
        return {
            'microbenchmark_json':r"^microbenchmark.json$",
            'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return False

    @staticmethod
    def uses_multiple_dirs():
        return False

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [
            Klass.get_gpu_overhead_path(src_files, bench_name),
            Klass.get_pretty_profile_path(src_files, bench_name),
            Klass.get_variable_path(src_files, bench_name),
        ]

    def profile_path(self, bench_name):
        return self.src_files.get('profile_path', bench_name)

    def get_micro_name(self):
        return self.bench_name

    def _parse_num_calls(self, bench_name):
        if self.is_dqn:
            data = self.load_microbench(bench_name)
            # bench_data = data[get_nvprof_name(self.bench_name)]
            bench_data = data[self.bench_name]
            num_calls = compute_num_calls(bench_data)
        elif 'num_calls' in self.config:
            num_calls = self.config['num_calls']
        else:
            num_calls = 1

        print("> num_calls = {num_calls}".format(
            num_calls=num_calls))
        return num_calls

    def _pr_rows(self, sql_rows):
        if len(sql_rows) == 0 or type(sql_rows[0]) not in set([sqlite3.Row]):
            pprint.pprint(sql_rows)
            return

        xs = []
        for row in sql_rows:
            d = dict()
            for k in row.keys():
                d[k] = sql_rows[k]
        pprint.pprint(xs)

    def gpu_count(self):
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) as c FROM CUPTI_ACTIVITY_KIND_DEVICE")
        return c.fetchone()["c"]

    def api_times(self):
        """
        Select all CUDA API call times: (start_nsec, end_nsec)
        :return:
        """
        c = self.conn.cursor()
        # The times are in nanoseconds; lets convert them to microseconds as our base unit
        def fetch_cbid(table):
            c.execute(textwrap.dedent("""
                SELECT 
                    A.start AS start_nsec, 
                    A.end AS end_nsec,
                    A.cbid as name 
                FROM 
                    {table} AS A 
                ORDER BY A.start
            """.format(table=table)))
            results = c.fetchall()
            return results
        results = fetch_cbid('CUPTI_ACTIVITY_KIND_DRIVER') + fetch_cbid('CUPTI_ACTIVITY_KIND_RUNTIME')
        results.sort(key=lambda r: r['start_nsec'])

        for r in results:
            r['name'] = CBID_TO_CUDA_FUNC.get(r['name'], r['name'])

        for r in results:
            self.api_stats.add(r['name'],
                               start_end_nsec_to_usec(r['start_nsec'], r['end_nsec']),
                               nsec_to_usec(r['start_nsec']),
                               nsec_to_usec(r['end_nsec']))

    def kernel_times(self):
        """
        Select all CUDA kernel call times: (start_nsec, end_nsec)
        :return:
        """
        c = self.conn.cursor()
        # The times are in nanoseconds; lets convert them to microseconds as our base unit
        c.execute(textwrap.dedent("""
            SELECT 
                A.start AS start_nsec, 
                A.end AS end_nsec,
                S.value as name 
            FROM 
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS A 
                join StringTable AS S 
                    ON A.name == S._id_ 
            ORDER BY A.start
        """))
        kernel_results = c.fetchall()

        for r in kernel_results:
            r['name'] = cxxfilt.demangle(r['name'])
        for r in kernel_results:
            self.kernel_stats.add(r['name'],
                                  start_end_nsec_to_usec(r['start_nsec'], r['end_nsec']),
                                  nsec_to_usec(r['start_nsec']),
                                  nsec_to_usec(r['end_nsec']))

        def fetch_memcpy(table):
            c.execute(textwrap.dedent("""
                SELECT 
                    A.start AS start_nsec, 
                    A.end AS end_nsec,
                    copyKind as name 
                FROM 
                    {table} AS A 
                ORDER BY A.start
            """.format(table=table)))
            results = c.fetchall()
            return results

        memcpy_results = fetch_memcpy('CUPTI_ACTIVITY_KIND_MEMCPY') + fetch_memcpy('CUPTI_ACTIVITY_KIND_MEMCPY2')
        for r in memcpy_results:
            r['name'] = COPYKIND_TO_MEMCPY.get(r['name'], r['name'])
            self.kernel_stats.add(r['name'],
                                  start_end_nsec_to_usec(r['start_nsec'], r['end_nsec']),
                                  nsec_to_usec(r['start_nsec']),
                                  nsec_to_usec(r['end_nsec']))

    def parse(self, bench_name):
        # with open(self.profile_path(bench_name)) as f:
        self.conn = sqlite3.connect(self.profile_path(bench_name))
        # https://stackoverflow.com/questions/3300464/how-can-i-get-dict-from-sqlite-query
        # self.conn.row_factory = sqlite3.Row
        self.conn.row_factory = dict_factory
        with self.conn:
            # PSEUDOCODE:
            # select func_name,
            #   ordered by order of occurrence when executing 1000 repetitions.
            # num_gpus = self.gpu_count()
            # print("> num_gpus = {num_gpus}".format(num_gpus=num_gpus))

            self.kernel_times()
            self.api_times()

    def _microbench_path(self, bench_name):
        return get_microbench_path(_d(self._pretty_profile_path(bench_name)))

    def load_microbench(self, bench_name):
        microbench_path = self._microbench_path(bench_name)
        assert _e(microbench_path)

        with codecs.open(microbench_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            data = fixup_json(data)
            return data

    def dump(self, bench_name):
        if self.skip:
            return

        summary_type = 'separate_calls'

        self.kernel_stats.split(self.num_calls)
        self.api_stats.split(self.num_calls)

        self.parse_gpu_overhead(bench_name)

        with open(self._pretty_profile_path(bench_name), 'w') as f:
            with open(self._variable_path(bench_name), 'w') as f_variable:

                def dump_data(stats, profile_data_type, skip_header=False):
                    print("> {t}".format(t=profile_data_type))
                    stats.dump(f, profile_data_type, skip_header=skip_header, summary_type=summary_type)
                    stats.dump_variable(f_variable, profile_data_type, skip_header=skip_header)
                    print()

                dump_data(self.kernel_stats, profile_data_type='GPU activities')
                dump_data(self.api_stats, profile_data_type='API calls', skip_header=True)

    def total_iterations(self, bench_name):
        if self.num_calls is None:
            self.num_calls = self._parse_num_calls(bench_name)
            assert self.num_calls is not None

        if self.args.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    def parse_gpu_overhead(self, bench_name, dump_json=True):
        # stats = self.api_stats

        if self.num_calls is None:
            self.num_calls = self._parse_num_calls(bench_name)
        # gpu_times = np.zeros(self.total_iterations(bench_name))
        # for stat in stats.stats:
        #     times_sec = stat.iteration_times_sec(self.num_calls)
        #     gpu_times += times_sec

        # {
        #     "CudaCppTimeSec": [
        #         0.20482133099999977
        #     ],
        #     "FrameworkCppTimeSec": [
        #         0.0
        #     ],
        #     "GPUAndCudaCppTimeSec": [
        #         5.3178501549999995
        #     ],
        #     "GPUTimeSec": [
        #         5.113028824
        #     ]
        # }

        gpu_times = self.kernel_stats.sum_calls_sec()
        gpu_and_cuda_cpp_times = self.api_stats.sum_calls_sec()
        assert gpu_times.shape == gpu_and_cuda_cpp_times.shape
        cuda_cpp_times = gpu_and_cuda_cpp_times - gpu_times
        # WRONG:
        # framework_cpp_times = gpu_and_cuda_cpp_times - gpu_times - cuda_cpp_times
        # WANT:
        # framework_cpp_times = pyprof(gpu_and_cpp_times) - gpu_times - cuda_cpp_times

        # 'CUDACppAndGpuTimeSec' = sum("API calls" from nvprof)
        # 'CUDACppTimeSec' = 'CUDACppAndGpuTimeSec' - 'GPUTimeSec'
        # 'CppTimeSec' = 'CppAndGpuTimeSec' - 'GPUTimeSec'

        raw_data = {
            "GPUTimeSec":list(gpu_times),
            "GPUAndCudaCppTimeSec":list(gpu_and_cuda_cpp_times),
            "CudaCppTimeSec":list(cuda_cpp_times),
            # "FrameworkCppTimeSec":list(framework_cpp_times),
        }
        if dump_json:
            json_data = make_json_serializable(raw_data)
            do_dump_json(json_data, self._gpu_overhead_path(bench_name))
        return raw_data

    def _gpu_overhead_path(self, bench_name):
        return self.get_gpu_overhead_path(self.src_files, bench_name)

    @classmethod
    def get_gpu_overhead_path(ParseKlass, src_files, bench_name):
        ret = re.sub(r'.nvprof$', '.gpu_overhead.nvprof.json', src_files.get('profile_path', bench_name))
        assert ret != src_files.get('profile_path', bench_name)
        return ret

    def dump_path(self, bench_name):
        return self._pretty_profile_path(bench_name)

    @classmethod
    def get_pretty_profile_path(ParserKlass, src_files, bench_name):
        pretty_base = "{base}.pretty.txt".format(base=_b(src_files.get('profile_path', bench_name)))
        return _j(_d(src_files.get('profile_path', bench_name)), pretty_base)

    def _pretty_profile_path(self, bench_name):
        return CUDASQLiteParser.get_pretty_profile_path(self.src_files, bench_name)

    @classmethod
    def get_variable_path(ParserKlass, src_files, bench_name):
        pretty_base = "{base}.variable.pretty.txt".format(base=_b(src_files.get('profile_path', bench_name)))
        return _j(_d(src_files.get('profile_path', bench_name)), pretty_base)

    def _variable_path(self, bench_name):
        return self.get_variable_path(self.src_files, bench_name)

class SrcFilesMixin:
    def get_bench_name(self, ParserKlass, allow_none=False):
        if not ParserKlass.uses_all_benches():
            bench_names = self.bench_names
            if len(bench_names) == 0 and allow_none:
                return None
            try:
                pass  # Cmd
                assert len(bench_names) == 1
            except Exception as e:
                import pdb;
                pdb.set_trace()
                raise e

            bench_name = list(bench_names)[0]
            return bench_name
        return NO_BENCH_NAME

    def all_targets(self, ParserKlass):
        targets = []
        if ParserKlass.uses_all_benches():
            bench_name = NO_BENCH_NAME
            targets.extend(ParserKlass.get_targets(self, bench_name))
        else:
            for bench_name in self.bench_names:
                targets.extend(ParserKlass.get_targets(self, bench_name))
        return targets

    def check_has_all_required_paths(self, ParserKlass):
        if not self.has_all_required_paths:
            print(
                textwrap.dedent("""
ERROR: Didn't find all required source files in directory={dir} for parser={parser}
  src_files =
{src_files}
  required_files = 
{required_files}
                """.format(
                    dir=self.directory,
                    parser=ParserKlass.__name__,
                    # src_files=str(src_files),
                    src_files=textwrap.indent(str(self), prefix="  "*2),
                    required_files=as_str(ParserKlass.required_source_basename_regexes(), indent=2),
                )))

class SrcFiles(SrcFilesMixin):
    """
    A bunch of files belongining to the same directory matching certain patterns.

    Files have been grouped together based on regex's that they match.
    Each regex has a name associated with it.
    Also, each regex will have a 'bench_name' capturing group.
    """
    def __init__(self, directory, req_paths, opt_paths, has_all_required_paths, allow_multiple_src_matches):
        # PSEUDOCODE:
        # if uses bench_name:
        #     req_paths['profile_path'][bench_name] = ...
        # else:
        #     req_paths['profile_path'][None] = ...
        self.req_paths = req_paths
        self.opt_paths = opt_paths
        self.has_all_required_paths = has_all_required_paths
        self.allow_multiple_src_matches = allow_multiple_src_matches
        self.directory = directory

    @property
    def bench_names(self):
        # bench_names = set()
        def _bench_names(matches):
            bench_names = set()
            for match_name, bench_matches in matches.items():
                for k in bench_matches.keys():
                    bench_names.add(k)
            return bench_names
        req_bench_names = _bench_names(self.req_paths)
        # opt_bench_names = _bench_names(self.opt_paths)
        # assert opt_bench_names.issubset(req_bench_names)
        # For each bench_name, there should be a full set of required/optional files.
        # assert req_bench_names == opt_bench_names

        return list(req_bench_names)

    def __str__(self):
        return as_str(self)

    def _all_sources_all_bench_names(self):
        bench_names = self.bench_names
        srcs = []
        for bench_name in bench_names:
            srcs.extend(self.all_sources(bench_name))
        return srcs

    def all_sources(self, bench_name=NO_BENCH_NAME, all_bench_names=False):
        if all_bench_names:
            return self._all_sources_all_bench_names()

        def srcs_for(paths):
            if paths is None:
                return []
            sources = []
            for regex_name in paths.keys():
                if bench_name in paths[regex_name]:
                    sources.append(paths[regex_name][bench_name])
            return sources

        return srcs_for(self.req_paths) + \
               srcs_for(self.opt_paths)

    def get(self, src_file_name, bench_name=NO_BENCH_NAME, or_none=False):
        def _get(paths):
            if paths is None:
                return None
            if src_file_name in paths:
                if bench_name in paths[src_file_name]:
                    return paths[src_file_name][bench_name]
            return None

        path = _get(self.req_paths)
        if path is not None:
            return path

        path = _get(self.opt_paths)
        if path is not None:
            return path

        if or_none:
            return None

        raise KeyError((bench_name, src_file_name))


    @property
    def is_group(self):
        return False

    @property
    def directories(self):
        return [self.directory]

    def get_src_files(self, directory):
        assert directory == self.directory
        return self

class SrcFilesGroup(SrcFilesMixin):
    """
    A list of directories, with each directory having a bunch of files matching certain patterns.
    A list of SrcFiles objects.

    Implements a lot of the same interface as SrcFiles (needed by SConstruct).
    """
    def __init__(self, src_files_list):
                 # req_paths, opt_paths, has_all_required_paths, allow_multiple_src_matches):
        self.src_files_list = src_files_list
        self._direc_to_src_files = dict(
            (src_files.directory, src_files) for src_files in self.src_files_list)

    @property
    def directory(self):
        srcs = self.all_sources(all_bench_names=True)
        return common_dir(srcs)

    @property
    def bench_names(self):
        benches = set()
        for src_files in self.src_files_list:
            for bench in src_files.bench_names:
                benches.add(bench)
        return list(benches)

    @property
    def is_group(self):
        return True

    @property
    def directories(self):
        return [src_files.directory for src_files in self.src_files_list]

    def get_src_files(self, directory):
        return self._direc_to_src_files[directory]

    @property
    def has_all_required_paths(self):
        return all(src_files.has_all_required_paths \
                   for src_files in self.src_files_list)

    def all_sources(self, bench_name=NO_BENCH_NAME, all_bench_names=False):
        """
        If all_bench_names=True, get source files spanning all bench_names.

        :return:
        """
        srcs = []
        for src_files in self.src_files_list:
            srcs.extend(src_files.all_sources(bench_name=bench_name, all_bench_names=all_bench_names))
        return srcs

class ProfilerParser(ProfilerParserCommonMixin):
    """
         13 function calls in 0.000 seconds

       Ordered by: call count, file name, function name, line number

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/atari-py/atari_py/ale_python_interface.py:305(decodeState)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/atari-py/atari_py/ale_python_interface.py:291(deleteState)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/atari-py/atari_py/ale_python_interface.py:287(restoreSystemState)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/baselines/baselines/deepq/simple_refactor.py:2753(__exit__)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/baselines/baselines/deepq/simple_refactor.py:853(iter)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/core.py:138(unwrapped)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/core.py:291(unwrapped)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/envs/atari/atari_env.py:177(restore)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/envs/atari/atari_env.py:167(restore_full_state)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/envs/benchmark_tf/lib/python3.5/site-packages/numpy/ctypeslib.py:438(as_ctypes)
            1    0.000    0.000    0.000    0.000 {built-in method builtins.len}
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
            1    0.000    0.000    0.000    0.000 {method 'from_address' of '_ctypes.PyCArrayType' objects}

    {
    'ncalls', [1, 1, 1, ...],
    'tottime', [0.0, ...],
    ...
    'filename:lineno(function)', [...],
    }
    """
    def __init__(self, parser, args, src_files,
                 # profile_path,
                 # bench_name,
                 data=None,
                 bench_name=NO_BENCH_NAME):
        self.parser = parser
        self.args = args
        self.bench_name = bench_name
        self.src_files = src_files
        self.data = data
        # assert data is not None
        self.time_fields = ['tottime', 'cumtime', 'tottime_percall', 'cumtime_percall']
        self.skip = False

    def parse_columns(self, line, it):
        raise NotImplemented

    def profile_path(self, bench_name):
        return self.src_files.get('profile_path', bench_name)

    def dump_path(self, bench_name):
        return self._pretty_profile_path(bench_name)

    @classmethod
    def get_pretty_profile_path(Klass, src_files, bench_name):
        assert re.search(r'\.txt$', src_files.get('profile_path', bench_name))

        pretty_base = re.sub(r'\.txt$', '.pretty.csv', _b(src_files.get('profile_path', bench_name)))
        return _j(_d(src_files.get('profile_path', bench_name)), pretty_base)

    def _pretty_profile_path(self, bench_name):
        return self.get_pretty_profile_path(self.src_files, bench_name)

    def store(self, *args, **kwargs):
        store_group(self.results, *args, **kwargs)

    def pre_parse(self, bench_name):
        pass

    def parse(self, bench_name):
        assert self.time_fields is not None
        assert self.total_time_fields is not None
        assert self.sortby is not None

        self.results = dict()

        self.pre_parse(bench_name)

        self.header = None
        with open(self.profile_path(bench_name)) as f:
            it = line_iter(f, lstrip=True)
            for line in it:

                if self.args.debug:
                    print("> {klass}, line :: {line}".format(
                        klass=self.__class__.__name__,
                        line=line))

                if self.parse_other(line, it):
                    continue

                if self.header is None and self.parse_header(line, it):
                    continue

                if re.search(r'^\s*$', line):
                    continue

                if self.header is not None:
                    if self.parse_columns(line, it):
                        continue

        self.post_parse(bench_name)

        # if not self.skip:
        #     repetitions = self.data[self.bench_name]['repetitions']
        #     iterations = self.data[self.bench_name]['iterations']
        #     # NOTE: We since we measure like this:
        #     # profiler.start()
        #     # for r in repetitions:
        #     #   for i in iterations:
        #     #     do_iteration()
        #     # profiler.end()
        #     #
        #     # We the number of python calls to EVERY function to be a multiple of (repetitions * iterations)
        #
        #     # TODO: generalize ncalls for CUDA profile
        #     # assert (np.array(self.results[self.bench_name]['ncalls']) % (repetitions * iterations) == 0).all()
        #
        #     for key in self.total_time_fields:
        #         time_sec = self.results[key]
        #         new_key = self.per_iter_field(key)
        #         time_sec_all_calls = self.results[key]
        #         self.results[new_key] = list(np.array(time_sec_all_calls)/float(repetitions * iterations))

    def per_iter_field(self, name):
        assert name in self.total_time_fields
        return "{name}_per_iter".format(name=name)

    def seconds_field(self, name):
        assert name in self.time_fields
        return "{name}_seconds".format(name=name)

    def dump_header(self):
        header = []
        for k in self.header:
            # NOTE: must match order of append inside each_line.
            header.append(k)
            if k in self.time_fields:
                header.append(self.seconds_field(k))
            if k in self.total_time_fields:
                header.append(self.per_iter_field(k))
        return header

    def time_per_call(self, time_sec_all_calls):
        repetitions = self.data[self.bench_name]['repetitions']
        iterations = self.data[self.bench_name]['iterations']
        return time_sec_all_calls/float(repetitions * iterations)

        # for key in self.total_time_fields:
        #     time_sec = self.results[key]
        #     new_key = self.per_iter_field(key)
        #     time_sec_all_calls = self.results[key]
        #     self.results[new_key] = list(np.array(time_sec_all_calls)/float(repetitions * iterations))

    def dump(self, bench_name):
        if self.skip:
            return

        dump_header = self.dump_header()
        rows = []

        sortby_indices = [dump_header.index(h) for h in self.sortby]
        def key_func(row):
            return [row[i] for i in sortby_indices]

        with open(self._pretty_profile_path(bench_name), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(dump_header)
            for line in self.each_line():
                rows.append(line)
            rows.sort(key=key_func, reverse=True)
            for row in rows:
                writer.writerow(row)

    def each_line(self):
        raise NotImplemented

class CUDAProfileParser(ProfilerParser):
    """
    NOTE: This is for q_forward with:
    - iterations = 1000
    - repetitions = 10
    - Q: Why is "GPU activities" Calls not a multiple of 10000?
      Looks more like it's only a multiple of iterations=1000 ...
      but cuEventQuery isn't quite a multiple of 1000 (WEIRD)

      - NOTE: this is just b/c of an outdated file...
      GPU activities 'Time' total:
      65.78635
      API calls 'Time' total:
      149.37072

      Q: Why's there a discrepancy?

    ==18229== NVPROF is profiling process 18229, command: /home/jgleeson/envs/cycle_counter_prod/bin/python3 /home/jgleeson/clone/baselines/baselines/deepq/experiments/benchmark_dqn.py --profile-cuda --directory /home/jgleeson/clone/baselines/checkpoints/PongNoFrameskip-v4/glue/gpu/cycle_counter/02/microbenchmar
    ==18229== Profiling application: /home/jgleeson/envs/cycle_counter_prod/bin/python3 /home/jgleeson/clone/baselines/baselines/deepq/experiments/benchmark_dqn.py --profile-cuda --directory /home/jgleeson/clone/baselines/checkpoints/PongNoFrameskip-v4/glue/gpu/cycle_counter/02/microbenchmar
    ==18229== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
     GPU activities:   68.45%  45.0280s    200000  225.14us  2.0160us  525.09us  void gemv2N_kernel_val<float, float, float, int=128, int=32, int=4, int=4, int=1>(float, float, cublasGemv2Params_v2<float, float, float>)
                        8.85%  5.82178s    100000  58.217us  56.641us  58.913us  maxwell_scudnn_128x64_relu_small_nn
                        3.81%  2.50656s    100000  25.065us  23.072us  28.545us  maxwell_scudnn_128x32_relu_medium_nn
                        3.12%  2.05037s    100000  20.503us  18.048us  22.400us  maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148n_nt
                        2.11%  1.38482s    600000  2.3080us     320ns  27.457us  [CUDA memcpy DtoH]
                        1.74%  1.14494s    300000  3.8160us     416ns  27.457us  [CUDA memcpy HtoD]
                        1.43%  939.29ms    300000  3.1300us  1.9200us  4.4160us  void tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>(int, float const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>*)
                        1.37%  902.15ms    100000  9.0210us  8.5120us  10.976us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>(unsigned char const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>*)
                        1.20%  790.26ms    300000  2.6340us  2.1120us  4.6090us  void tensorflow::BiasNCHWKernel<float>(int, float const *, float const , tensorflow::BiasNCHWKernel<float>*, int, int)
                        1.18%  774.10ms    100000  7.7410us  6.8480us  9.0570us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>)
                        0.96%  632.73ms    100000  6.3270us  5.8880us  6.6240us  void tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>(int, float const *, tensorflow::functor::Dimension<int=4>, tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>*, tensorflow::functor::Dimension, float const *)
                        0.88%  580.17ms    400000  1.4500us  1.0240us  3.4240us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const , float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const , Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(float, int=1)
                        0.81%  534.80ms    100000  5.3480us  3.6480us  7.0720us  void cudnn::winograd::generateWinogradTilesKernel<int=0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)
                        0.66%  431.47ms    100000  4.3140us  4.0000us  4.8640us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<__int64, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float>>, Eigen::array<long, unsigned long=1> const , Eigen::TensorMap<Eigen::Tensor<float const , int=2, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(__int64, int=1)
                        0.61%  401.79ms    100000  4.0170us  3.2000us  5.1840us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>)
                        0.60%  393.82ms    200000  1.9690us  1.6640us  4.1280us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(float, int=1)
                        0.56%  366.66ms    100000  3.6660us  3.0400us  5.0560us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>(unsigned int const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>*)
                        0.42%  276.90ms    200000  1.3840us  1.0240us  3.8080us  void tensorflow::BiasNHWCKernel<float>(int, float const *, float const , tensorflow::BiasNHWCKernel<float>*, int)
                        0.42%  274.60ms    200000  1.3730us  1.1840us  3.0080us  cudnn::maxwell::gemm::computeOffsetsKernel(cudnn::maxwell::gemm::ComputeOffsetsParams)
                        0.24%  156.99ms    100000  1.5690us  1.2800us  4.3530us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(__int64, int=1)
                        0.21%  137.27ms    100000  1.3720us  1.0240us  2.5280us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
                        0.21%  137.20ms    100000  1.3720us  1.0560us  3.0400us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
                        0.18%  119.68ms    100000  1.1960us     928ns  2.4640us  [CUDA memcpy DtoD]
          API calls:   41.62%  62.1707s   3100000  20.055us  6.4550us  2.47057s  cudaLaunch
                       21.09%  31.5067s    100000  315.07us  3.9630us  2.4058ms  cuCtxSynchronize
                       14.75%  22.0380s   1800000  12.243us     328ns  2.47052s  cuEventRecord
                       10.86%  16.2258s    600000  27.042us  6.0910us  2.47050s  cuMemcpyDtoHAsync
                        4.12%  6.15277s    300000  20.509us  6.2390us  1.2364ms  cuMemcpyHtoDAsync
                        1.99%  2.97150s   2416996  1.2290us     476ns  389.70us  cuEventQuery
                        1.73%  2.59132s    100000  25.913us  16.561us  1.2549ms  cudaMemcpyAsync
                        1.09%  1.62282s   9200000     176ns     111ns  387.56us  cudaSetupArgument
                        1.03%  1.53828s    300000  5.1270us  1.3230us  87.835ms  cudaEventRecord
                        0.86%  1.27935s   3100000     412ns     152ns  416.18us  cudaConfigureCall
                        0.71%  1.06470s    900000  1.1830us     549ns  392.19us  cuStreamWaitEvent
                        0.14%  208.78ms   1000000     208ns     108ns  386.07us  cudaGetLastError

    {
        'gpu_activities': {
            'Time(%)': [68.45, ...],
            'Time': [45.0280, ...],
            'Calls': [20000, ...],
            # Q: If I divide a floating point number by a really big number, will I lose precision at all...?
            # Or is it only once I start combining 2 small floating points?
            'Avg': [225.14, ...],
            'Min': [2.0160, ...],
            'Max': [525.09, ...],
            'Name': [...],
        }
        'api_calls': {
            ...
        }
    }
    """
    def __init__(self, parser, args, profile_path, bench_name, data=None):
        super().__init__(parser, args, profile_path, bench_name, data)
        self.time_fields = ['Time', 'Avg', 'Min', 'Max']
        self.total_time_fields = ['Time_seconds']
        self.sortby = ('Type', 'Time_seconds')

        self.parsing = None
        self.no_cuda_calls_expected = False
        self.has_error = False

    CUDA_TIME_UNIT_REGEX = r'(?:s|ms|us|ns)'
    def as_seconds(self, time_as_unit, unit):
        if unit == 's':
            return time_as_unit
        elif unit == 'ms':
            return time_as_unit/float(MILLISECONDS_IN_SECOND)
        elif unit == 'us':
            return time_as_unit/float(MICROSECONDS_IN_SECOND)
        elif unit == 'ns':
            return time_as_unit/float(NANOSECONDS_IN_SECOND)
        else:
            raise NotImplemented

    def as_value(self, x):
        if re.search(r'%$', x):
            # Convert to percent in [0..1]
            x = re.sub(r'%', '', x)
            x = as_value(x) / 100.
            return x

        m = re.search(r'(?P<float>{float})(?P<unit>{unit})$'.format(
            float=float_re,
            unit=CUDAProfileParser.CUDA_TIME_UNIT_REGEX),
            x)
        if m:
            time_as_unit = float(m.group('float'))
            unit = m.group('unit')
            time_sec = self.as_seconds(time_as_unit, unit)
            return time_sec

        return as_value(x)

    def parse_other(self, line, it):
        self.no_cuda_calls_expected |= bool(re.search("|".join([r'^==.*== Generated result file',
                                                                r'No kernels were profiled']),
                                                      line))
        self.has_error |= bool(re.search(r'^=+\s*Error:.*Application', line))
        return False

    def parse_header(self, line, it):
        assert self.header is None
        if re.search(r'^\s*Type\s*Time', line):
            self.header = re.split(r'\s+', line.strip())
            # for i in range(len(self.header)):
            #     if self.header[i] == 'percall':
            #         self.header[i] = "{last_col}_percall".format(last_col=self.header[i-1])
            return True
        return False

    def parse_columns(self, line, it):
        fields = re.split(r'\s+', line.strip())
        first_two_fields = " ".join(fields[0:2])
        remaining_fields = fields[2:]
        if re.search(r'GPU activities', first_two_fields):
            self.parsing = 'gpu_activities'
            fields = remaining_fields
        elif re.search(r'API calls', first_two_fields):
            self.parsing = 'api_calls'
            fields = remaining_fields

        assert self.parsing is not None

        last_field_i = len(self.header) - 1
        field = " ".join(fields[last_field_i:])
        new_fields = [self.parsing] + fields[0:last_field_i] + [field]
        fields = new_fields
        fields = [self.as_value(x) for x in fields]

        put_key(self.results, self.parsing, dict())

        for i, name in enumerate(self.header):
            field = fields[i]
            store_as(self.results[self.parsing], name, field, store_type='list')

        return True

    def post_parse(self, bench_name):
        assert ( self.has_error or self.no_cuda_calls_expected ) or self.header is not None
        if self.no_cuda_calls_expected:
            print("> Skip pretty cuda profile; didn't see any CUDA calls in {path}".format(path=self.profile_path(bench_name)))
            self.skip = True

        if self.has_error:
            print("> Skip pretty cuda profile; WARNING: saw an ERROR in {path}".format(path=self.profile_path(bench_name)))
            self.skip = True

    def each_line(self):
        assert 'api_calls' in self.results
        assert 'gpu_activities' in self.results
        assert sorted(self.results['api_calls']) == sorted(self.results['gpu_activities'])
        num_lines = len(self.results['api_calls'][self.header[0]])
        for parsing in ['gpu_activities', 'api_calls']:
            for i in range(num_lines):
                row = []
                for k in self.header:
                    value = self.results[parsing][k][i]
                    if k in self.time_fields:
                        pty_time = pretty_time(value)
                    row.append(value)
                    if k in self.total_time_fields:
                        time_per_iter = self.time_per_call(value)
                        row.append(pretty_time(time_per_iter))
                yield row

def exists_nonempty(path):
    if _e(path):
        if os.path.getsize(path) == 0:
            os.remove(path)
            return False
        return True
    return False

PLOT_SUMMMARY_FIELDS = [
    "TotalTimeSec",
    "CppAndGPUTimeSec",
    "CppTimeSec",
    "FrameworkCppTimeSec",
    "CudaCppTimeSec",
    "PercentTimeInGPU",
    "GPUTimeSec",
    "GPUAndCudaCppTimeSec",
    "TheoreticalSpeedup",
    "PercentTimeInPython",
    "PythonTimeSec",
    "PythonOverheadPercent",
]
PLOT_SUMMARY_FIELDS_TIME_SEC = [field for field in PLOT_SUMMMARY_FIELDS if re.search(r'TimeSec$', field)]

def make_json_serializable(data):
    new_data = dict()
    for k in data.keys():
        new_data[k] = list(data[k])
    return new_data

def make_json_ndarray(data):
    new_data = dict()
    for k in data.keys():
        if type(data[k]) == list:
            new_data[k] = np.array(data[k])
        else:
            new_data[k] = data[k]
    return new_data

def compute_total_times(cpp_and_gpu_times, python_times):
    total_times = cpp_and_gpu_times + python_times
    return total_times

def compute_theoretical_speedup(cpp_and_gpu_times, python_times):
    total_times = compute_total_times(cpp_and_gpu_times, python_times)
    theoretical_speedup = total_times / cpp_and_gpu_times
    return theoretical_speedup

def compute_percent_time_in_python(cpp_and_gpu_times, python_times):
    total_times = compute_total_times(cpp_and_gpu_times, python_times)
    percent_time_in_python = 100. * python_times / total_times
    return percent_time_in_python

def compute_percent_time_in_gpu(gpu_times, total_times):
    percent_time_in_gpu = 100. * gpu_times / total_times
    return percent_time_in_gpu

def compute_python_overhead_percent(cpp_and_gpu_times, python_times):
    python_overhead_percent = 100. * python_times / cpp_and_gpu_times
    return python_overhead_percent

def compute_plot_data(cpp_and_gpu_times, python_times):
    total_times = compute_total_times(cpp_and_gpu_times, python_times)
    theoretical_speedup = compute_theoretical_speedup(cpp_and_gpu_times, python_times)
    percent_time_in_python = compute_percent_time_in_python(cpp_and_gpu_times, python_times)
    python_overhead_percent = compute_python_overhead_percent(cpp_and_gpu_times, python_times)

    def _no_nans(xs):
        assert not np.isnan(xs).any()

    # assert not (cpp_times == 0.).any()
    _no_nans(cpp_and_gpu_times)
    _no_nans(python_times)
    _no_nans(percent_time_in_python)

    data = {
        "TotalTimeSec":total_times,
        "CppAndGPUTimeSec":cpp_and_gpu_times,
        "TheoreticalSpeedup":theoretical_speedup,
        "PercentTimeInPython":percent_time_in_python,
        "PythonTimeSec":python_times,
        "PythonOverheadPercent":python_overhead_percent,
    }
    data = make_json_serializable(data)

    for field in data:
        assert field in PLOT_SUMMMARY_FIELDS

    return data

def _pyprof_total_base():
    return "python_profile.total.python_overhead.pyprof.json"

class PythonProfileTotalParser(ProfilerParser):
    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME):

        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths

        super().__init__(parser, args, src_files,
                         bench_name=bench_name)

    @staticmethod
    def required_source_basename_regexes():
        return {
            "python_overhead_path":"^python_profile_python_overhead_pyprof{bench}.json$".format(bench=BENCH_SUFFIX_RE),
        }

    @property
    def _python_pyprof_total_path(self):
        return self.get_python_pyprof_total_path(self.src_files)
    @classmethod
    def get_python_pyprof_total_path(Klass, src_files):
        return PythonProfileParser.get_python_overhead_path(src_files, bench_name='total')

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [
            Klass.get_python_pyprof_total_path(src_files),
        ]

    @staticmethod
    def optional_source_basename_regexes():
        return {'microbenchmark_json':r"^microbenchmark.json$",
                'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE)}

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return True

    @staticmethod
    def uses_multiple_dirs():
        return False

    def run(self, bench_name=NO_BENCH_NAME):
        """
        NOTE: This ought to be its own parser...
        :return:
        """
        parser = self.parser
        args = self.args
        directory = self.src_files.directory
        bench_names = [bench_name for bench_name in self.src_files.bench_names \
                       if bench_name != 'total']
        # bench_names = DQN_BENCH_NAMES
        # profile_paths = [_j(directory, 'python_profile.{b}.txt'.format(b=bench_name))
        #                  for bench_name in bench_names]
        # if not all(_e(path) for path in profile_paths):
        #     return

        path = self._python_pyprof_total_path
        # self.get_python_overhead_path()
        # self.get_python_overhead_path(self.src_files, bench_name='total')
        # path = _j(directory, _pyprof_total_base())

        # if _e(path):
        #     if os.path.getsize(path) == 0:
        #         os.remove(path)
        #     else:
        #         return

        cpp_and_gpu_times = None
        python_times = None
        smallest_length = None
        raw_datas = []
        profile_paths = []
        for bench_name in bench_names:
            # prof = PythonProfileParser(parser, args, profile_path, bench_name)
            # prof = PythonProfileParser(parser, args, self.src_files,
            #                            bench_name=bench_name)
            # raw_data = prof.parse_python_overhead(bench_name, dump_json=False, get_raw_data=True)
            python_overhead_path = PythonProfileParser.get_python_overhead_path(self.src_files, bench_name)
            raw_data = load_json(python_overhead_path)
            if smallest_length is None or smallest_length > len(raw_data['CppAndGPUTimeSec']):
                smallest_length = len(raw_data['CppAndGPUTimeSec'])
            raw_datas.append(raw_data)
            profile_paths.append(python_overhead_path)

        print("> Using the first {n} samples from each DQN stage to compute Total...".format(
            n=smallest_length))

        cpp_and_gpu_times = np.zeros(smallest_length)
        python_times = np.zeros(smallest_length)
        for bench_name, raw_data in zip(bench_names, raw_datas):
            cpp_and_gpu_times += raw_data['CppAndGPUTimeSec'][0:smallest_length]
            python_times += raw_data['PythonTimeSec'][0:smallest_length]

        data = compute_plot_data(cpp_and_gpu_times, python_times)
        do_dump_json(data, path)
        print("> Output pyprof Total: {f}".format(f=path))

        # parse_pyprof_total(self.directory(bench_name), src_files.bench_names, self.parser, self.args)
        # self.parse_pyprof_total()

class PythonProfileParser(ProfilerParser):
    def __init__(self, parser, args, src_files,
                 convert_to_seconds=True, data=None, bench_name=NO_BENCH_NAME):

        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths

        super().__init__(parser, args, src_files,
                         data=data,
                         bench_name=bench_name)

        self.config_path = src_files.get('config_json', bench_name, or_none=True)
        if self.config_path is not None:
            self.config = load_json(self.config_path)
            print("> Found optional config_json @ {f}".format(f=self.config_path))
        else:
            self.config = {
                'clock':'monotonic_clock',
            }

        if not self.is_dqn:
            assert 'num_calls' in self.config

        if 'num_calls' in self.config:
            self.num_calls = self.config['num_calls']
        else:
            self.num_calls = None

        if 'discard_first_sample' in self.config:
            self.discard_first_sample = self.config['discard_first_sample']
        else:
            self.discard_first_sample = self.args.discard_first_sample

        self.time_fields = ['tottime', 'cumtime', 'tottime_percall', 'cumtime_percall']
        self.total_time_fields = ['tottime_seconds']
        self.sortby = ('tottime_seconds', 'filename:lineno(function)')

        self.pyfunc_stats = Stats(self.discard_first_sample, debug=self.args.debug)
        self.convert_to_seconds = convert_to_seconds

        # config_path = self._config_path(bench_name)
        # if _e(config_path):
        # self.config = load_json(self._config_path(bench_name))

        # if self.is_dqn:
        #     self.config = load_json(self._config_path(bench_name))
        # else:
        #     self.config = {
        #         'clock':'monotonic_clock',
        #     }

    @property
    def debug(self):
        return self.args.debug

    @staticmethod
    def required_source_basename_regexes():
        return {
            'profile_path': r"^python_profile{bench}\.txt$".format(bench=BENCH_SUFFIX_RE),
            'python_call_times':"^python_profile{bench}\.call_times.json$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def optional_source_basename_regexes():
        return {'microbenchmark_json':r"^microbenchmark.json$",
                'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE)}

    @classmethod
    def test_targets(ParserKlass):
        paths = [
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.q_backward.pretty.csv',
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.q_forward.pretty.csv',
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.q_update_target_network.pretty.csv',
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.step.pretty.csv',
        ]
        src_files = ParserKlass.glob_target_files(paths, debug=True)
        matching_srcs = src_files.all_sources(all_bench_names=True)
        assert set(paths).issubset(set(matching_srcs))

    @staticmethod
    def target_basename_regexes():
        return {
            'pretty_profile_path': r"^python_profile{bench}\.pretty\.csv$".format(bench=BENCH_SUFFIX_RE),
            'python_overhead_path': r"^python_profile{bench}\.python_overhead\.pyprof\.json".format(bench=BENCH_SUFFIX_RE),
            # Klass.get_call_times_path(src_files, bench_name),
            # 'total_path':r"^python_profile.total.python_overhead.pyprof.json$",
        }

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return False

    @staticmethod
    def uses_multiple_dirs():
        return False

    def pre_parse(self, bench_name):
        pass
        # We only use monotonic_clock, no longer use cycle counter since we
        # had issues getting it to work.
        # if self.convert_to_seconds:
        #     self.tsc_freq = BenchmarkTSCFreq(self.parser, self.args)
        #     self.tsc_freq.run()

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [Klass.get_pretty_profile_path(src_files, bench_name),
                Klass.get_python_overhead_path(src_files, bench_name),
                # Klass.get_python_pyprof_total_path(src_files),

                # Klass.get_call_times_path(src_files, bench_name),
                Klass.get_call_times_summary_path(src_files, bench_name),
                Klass.get_call_times_pyprof_path(src_files, bench_name),
                ]

    def _config_path(self, bench_name):
        return _j(self.directory(bench_name), "config.json")


    @classmethod
    def get_call_times_path(self, src_files, bench_name):
        # ret = _j(src_files.directory, "python_profile_call_times{bench}.json".format(bench=bench_suffix(bench_name)))
        ret = _j(src_files.directory, "python_profile{bench}.call_times.json".format(bench=bench_suffix(bench_name)))
        # call_times_path = re.sub(r'.txt$', '.call_times.json', self.profile_path(bench_name))
        # return call_times_path
        return ret
    def _call_times_path(self, bench_name):
        return self.get_call_times_path(self.src_files, bench_name)
        # call_times_path = re.sub(r'.txt$', '.call_times.json', self.profile_path(bench_name))
        # return call_times_path

    @classmethod
    def get_call_times_summary_path(ParserKlass, src_files, bench_name):
        # profile_path = src_files.get('profile_path', bench_name)
        ret = _j(src_files.directory, "python_profile_call_times{bench}.txt".format(bench=bench_suffix(bench_name)))
        # ret = re.sub(r'.txt$', '.call_times.txt', profile_path)
        # assert ret != profile_path
        return ret
    def _call_times_summary_path(self, bench_name):
        return self.get_call_times_summary_path(self.src_files, bench_name)

    @classmethod
    def get_call_times_pyprof_path(ParserKlass, src_files, bench_name):
        # profile_path = src_files.get('profile_path', bench_name)
        # ret = re.sub(r'.txt$', '.call_times.pyprof.txt', profile_path)
        ret = _j(src_files.directory, "python_profile_call_times_pyprof{bench}.txt".format(bench=bench_suffix(bench_name)))
        # assert ret != profile_path
        return ret
    def _call_times_pyprof_path(self, bench_name):
        return self.get_call_times_pyprof_path(self.src_files, bench_name)

    def _python_overhead_path(self, bench_name):
        return self.get_python_overhead_path(self.src_files, bench_name)

    @classmethod
    def get_python_overhead_path(Klass, src_files, bench_name):
        ret = _j(src_files.directory, "python_profile_python_overhead_pyprof{bench}.json".format(bench=bench_suffix(bench_name)))
        # ret = re.sub(r'.txt$', '.python_overhead.pyprof.json', src_files.get('profile_path', bench_name))
        # assert ret != src_files.get('profile_path', bench_name)
        return ret

    def _microbench_path(self, bench_name):
        return get_microbench_path(_d(self.profile_path(bench_name)))

    def load_microbench(self, bench_name):
        microbench_path = self._microbench_path(bench_name)
        assert _e(microbench_path)

        with codecs.open(microbench_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            data = fixup_json(data)
            return data

    def get_micro_name(self):
        # if self.args.c_only:
        #     return get_c_only_name(self.bench_name)
        return self.bench_name

    def _parse_num_calls(self, bench_name):
        call_times_path = self._call_times_path(bench_name)
        # if not _e(call_times_path):
        #     return
        print("> Parsing call times: {f}".format(f=call_times_path))
        self.call_times = self.load_call_times(bench_name)
        if self.is_dqn:
            micro_data = self.load_microbench(bench_name)
            micro_name = self.get_micro_name()
            bench_data = micro_data[micro_name]
            # This is the number of times (for e.g.) Forward was called.
            # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
            # num_calls = iterations*repetitions
            self.num_calls = compute_num_calls(bench_data)
        else:
            assert 'num_calls' in self.config
            assert self.num_calls is not None
            # self.num_calls = 1
        print("> num_calls = {num_calls}".format(
            num_calls=self.num_calls))

    def parse_call_times(self, bench_name):

        # call_times_path = self._call_times_path(bench_name)
        # # if not _e(call_times_path):
        # #     return
        # print("> Parsing call times: {f}".format(f=call_times_path))
        # call_times = self.load_call_times(bench_name)
        # micro_data = self.load_microbench(bench_name)
        # micro_name = self.get_micro_name()
        # bench_data = micro_data[micro_name]
        # # This is the number of times (for e.g.) Forward was called.
        # # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
        # # num_calls = iterations*repetitions
        # self.num_calls = compute_num_calls(bench_data)
        # print("> num_calls = {num_calls}".format(
        #     num_calls=self.num_calls))
        start_parse_call_times = time.time()
        self._parse_num_calls(bench_name)
        end_parse_call_times = time.time()
        print("> Parsing call times took: {sec} seconds".format(sec=end_parse_call_times - start_parse_call_times)) # 45 seconds, ~ 6-7 GB

        print("> Adding call times to pyfunc_stats:") # 249 seconds / 4min 9sec; this takes up a LOT of memory: 34.5 GIG.
        start_add_call_times = time.time()
        with progressbar.ProgressBar(max_value=len(self.call_times.items())) as bar:
            for i, (func_name, call_times) in enumerate(self.call_times.items()):
                if self.config['clock'] == 'monotonic_clock':
                    time_secs = call_times
                else:
                    raise NotImplementedError

                self.pyfunc_stats.add_times_sec(func_name, time_secs)

                bar.update(i)
        end_add_call_times = time.time()
        print("> Adding call times to pyfunc_stats took: {sec} seconds".format(sec=end_add_call_times - start_add_call_times))

        # TODO: convert cycles to usec!

        print("> Split pyfunc stats:") # 20 seconds
        start_split_call_times = time.time()
        self.pyfunc_stats.split(self.num_calls)
        end_split_call_times = time.time()
        print("> Split pyfunc stats took: {sec} seconds".format(sec=end_split_call_times - start_split_call_times))

        # This takes a while and we don't use it; skip.
        print("> Dumping pyfunc stats:") # 467 seconds
        start_dump_call_times = time.time()
        with open(self._call_times_summary_path(bench_name), 'w') as f:
            # with open(self._call_times_pyprof_path(bench_name), 'w') as f_pyprof:
            self.pyfunc_stats.dump_separate_calls(f, 'Python')
            # Dump profile similar to cProfile/pstats output so
            # we can make sure we're doing things correctly.
            # self.pyfunc_stats.dump_nvprof(f_pyprof, 'Python')
        end_dump_call_times = time.time()
        print("> Dumping pyfunc stats took: {sec} seconds".format(sec=end_dump_call_times - start_dump_call_times))
        print("> Parsed call times into: {f}".format(f=self._call_times_summary_path(bench_name)))

    def total_iterations(self, bench_name):
        if self.num_calls is None:
            self._parse_num_calls(bench_name)
            assert self.num_calls is not None

        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    # TODO: make this unit-testable (in particular, when we gather python_times/PythonTimeSec
    def parse_python_overhead(self, bench_name, dump_json=True, get_raw_data=False):
        self.parse_call_times(bench_name)

        print("> Parsing python overhead: {f}".format(f=self._python_overhead_path(bench_name)))

        # TODO: This is tensorflow specific, we need to try wrapping end-to-end test
        # and making this work for all C++ libraries
        if self.is_dqn:
            default_cpp_func_re = r'(?:built-in.*pywrap_tensorflow)'
        else:
            default_cpp_func_re = r"CLIB__.*"
        cpp_func_re = self.config.get('c_lib_func_pyprof_pattern', default_cpp_func_re)

        def is_cpp_func(func_name):
            m = re.search(cpp_func_re, func_name)
            return bool(m)
        cpp_and_gpu_times = np.zeros(self.total_iterations(bench_name))
        python_times = np.zeros(self.total_iterations(bench_name))
        # 1000 iterations
        # 22000 calls
        # 22 different calls to this function in each iteration

        print("> Compute python overhead")
        start_compute_python_overhead = time.time()
        if self.debug:
            print("> cpp_func_re = {regex}".format(regex=cpp_func_re))
        for stat in self.pyfunc_stats.stats:
            # Returns an array of size self.num_calls, one entry for each time
            # an "iteration" of the expierment was run.
            times_sec = stat.iteration_times_sec(self.num_calls)
            if self.debug:
                print("> stat.name = {name}".format(name=stat.name))
                if stat.name == "/mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:78(CLIB__run_cpp)":
                    import ipdb; ipdb.set_trace()
            if is_cpp_func(stat.name):
                cpp_and_gpu_times += times_sec
            else:
                python_times += times_sec
        end_compute_python_overhead = time.time()
        print("> Compute python overhead took: {sec} seconds".format(sec=end_compute_python_overhead - start_compute_python_overhead))

        total_times = cpp_and_gpu_times + python_times

        # NOTE: The idea behind this calculation is that in an ideal case,
        # we only need to execute the C code, and the python code doesn't need to be executed.
        # So, overhead = time(python)/time(C)
        # NOT:
        #   overhead = time(python)/time(python + C)
        #   This is what the profiler tells us.
        python_overhead_percent = 100.*python_times/cpp_and_gpu_times

        raw_data = {
            "TotalTimeSec":total_times,
            "CppAndGPUTimeSec":cpp_and_gpu_times,
            "PythonTimeSec":python_times,
            "PythonOverheadPercent":python_overhead_percent,
        }
        if get_raw_data:
            return raw_data

        data = compute_plot_data(cpp_and_gpu_times, python_times)
        if dump_json:
            do_dump_json(data, self._python_overhead_path(bench_name))


        return data

    def directory(self, bench_name):
        return _d(self.profile_path(bench_name))

    def load_call_times(self, bench_name):
        # JAMES TODO: put this into the parser class for the python profiling data; produce output similar to what we did for CUDA stuff.
        call_times_path = self._call_times_path(bench_name)
        assert _e(call_times_path)

        with codecs.open(call_times_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            # data = fixup_json(data)
            return data

    def sec_as_usec(self, sec):
        return sec * 1e6

    def parse_header(self, line, it):
        assert self.header is None
        if re.search(r'^\s*ncalls\s*tottime', line):
            self.header = re.split(r'\s+', line.strip())
            for i in range(len(self.header)):
                if self.header[i] == 'percall':
                    self.header[i] = "{last_col}_percall".format(last_col=self.header[i-1])
            return True
        return False

    def parse_other(self, line, it):
        m = re.search(r'\s*(?P<total_function_calls>{float}) function calls (?:\(\d+ primitive calls\) )?in (?P<total_time>{float}) (?P<profiling_unit>\w+)'.format(float=float_re), line)
        if m:
            store_group(self.results, m)
            return True
        return False

    def parse_columns(self, line, it):
        fields = re.split(r'\s+', line.strip())
        last_field_i = len(self.header) - 1
        for i, name in enumerate(self.header[0:len(self.header) - 1]):
            field = fields[i]
            store_as(self.results, name, field, store_type='list')
        field = " ".join(fields[last_field_i:])
        store_as(self.results, self.header[-1], field, store_type='list')

    def post_parse(self, bench_name):
        if self.convert_to_seconds and self.results['profiling_unit'] == 'cycles':
            for key in self.time_fields:
                time_sec = self.results[key]
                self.results[key] = self.cycles_to_seconds(time_sec)

        self.parse_python_overhead(bench_name)

    def each_line(self):
        num_lines = len(self.results[self.header[0]])
        for i in range(num_lines):
            row = []
            for k in self.header:
                # NOTE: must match order of append inside dump_header.
                value = self.results[k][i]
                if k in self.time_fields:
                    row.append(pretty_time(value))
                row.append(value)
                if k in self.total_time_fields:
                    # per_iter_time_sec = self.results[self.per_iter_field(k)][i]
                    time_per_iter = self.time_per_call(value)
                    row.append(pretty_time(time_per_iter))
            yield row

    def cycles_to_seconds(self, cycles):
        # cpu_freq_ghz = self.cpu_freq.results['cpu_freq_ghz_mean']
        cpu_freq_ghz = self.tsc_freq.results['tsc_freq_ghz_mean']
        HZ_IN_GHZ = 1e9
        cpu_freq_hz = cpu_freq_ghz*HZ_IN_GHZ
        return [x/cpu_freq_hz for x in cycles]

class CombinedProfileParser(ProfilerParserCommonMixin):
    """
    Merge the results of PythonProfileParser and CUDASQLiteParser
    into a single json file that can be easily plotted.

    PROBLEM:
    The source regex might match several different, one for each bench_name.
    Options:
    1. Only grab an nvprof file, then map it to the corresponding pyprof file.
    2. Get ALL files matching a regex, and match pairs of files with matching bench_name's.
       For each such pair, output a target file.
       Will need to make it so _match_regexes allows handling this way...
    """
    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
        # self.is_dqn = 'microbenchmark_json' in src_files.opt_paths
        self.parser = parser
        self.args = args
        self.src_files = src_files
        self.bench_name = bench_name
        self.data = data

    @staticmethod
    def required_source_basename_regexes():
        nvprof_regexes = CUDASQLiteParser.required_source_basename_regexes()
        def new_key_func(k):
            return "nvprof_{k}".format(k=k)
        nvprof_regexes = remap_keys(nvprof_regexes, new_key_func)

        pyprof_regexes = PythonProfileParser.required_source_basename_regexes()
        def new_key_func(k):
            return "pyprof_{k}".format(k=k)
        pyprof_regexes = remap_keys(pyprof_regexes, new_key_func)

        regexes = dict()
        regexes.update(nvprof_regexes)
        regexes.update(pyprof_regexes)

        # {'nvprof_profile_path': ..., 'pyprof_profile_path': ...}
        return regexes

    @staticmethod
    def target_basename_regexes():
        return {
            'breakdown_json': r"^{bench}breakdown.json$".format(bench=BENCH_PREFIX_RE),
        }

    @staticmethod
    def optional_source_basename_regexes():
        return dict()

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return False

    @staticmethod
    def uses_multiple_dirs():
        return False

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [
            Klass.get_combined_path(src_files, bench_name),
        ]

    # def combine_profiles(self):
    def parse(self, bench_name):

        def check_all_same_length(dic):
            length = None
            for k, v in dic.items():
                if length is None:
                    length = len(v)
                else:
                    assert length == len(v)

        def get_length(dic):
            key = list(dic.keys())[0]
            return len(dic[key])

        def combine(dic1, dic2, length):
            dic = dict()
            for k, v in dic1.items():
                dic[k] = v[:length]
            for k, v in dic2.items():
                dic[k] = v[:length]
            return dic

        json_path = self._combined_path(bench_name)

        pyprof_json_path = self.python_profile_json_path(bench_name)
        pyprof_json = load_json(pyprof_json_path)
        check_all_same_length(pyprof_json)
        pyprof_len = get_length(pyprof_json)

        nvprof_json_path = self.nvprof_profile_json_path(bench_name)
        nvprof_json = load_json(nvprof_json_path)
        check_all_same_length(nvprof_json)
        nvprof_len = get_length(nvprof_json)

        keep_len = min(pyprof_len, nvprof_len)

        combined = combine(pyprof_json, nvprof_json, keep_len)
        combined_nd = make_json_ndarray(combined)
        # PercentTimeInGpu
        combined_nd['CppTimeSec'] = combined_nd['CppAndGPUTimeSec'] - combined_nd['GPUTimeSec']
        combined_nd['SanityTotalTimeSec'] = combined_nd['CppTimeSec'] + combined_nd['GPUTimeSec'] + combined_nd['PythonTimeSec']
        # framework_cpp_times = pyprof(gpu_and_cpp_times) - nvprof(gpu_times) - nvprof(cuda_cpp_times)
        combined_nd['FrameworkCppTimeSec'] = combined_nd['CppAndGPUTimeSec'] - combined_nd['GPUTimeSec'] - combined_nd['CudaCppTimeSec']
        combined_nd['PercentTimeInGPU'] = compute_percent_time_in_gpu(combined_nd['GPUTimeSec'], combined_nd['TotalTimeSec'])

        combined = make_json_serializable(combined_nd)
        # combined['CppAndGPUTimeSec']
        do_dump_json(combined, json_path)
        print("> Created combined profile breakdown @ {path}".format(path=json_path))

    def dump(self, bench_name):
        pass

    @classmethod
    def get_directory(Klass, src_files, bench_name):
        return _d(src_files.get('pyprof_profile_path', bench_name))

    def python_profile_json_path(self, bench_name):
        return self.get_python_profile_json_path(self.src_files, bench_name)

    @classmethod
    def get_python_profile_json_path(Klass, src_files, bench_name):
        directory = Klass.get_directory(src_files, bench_name)
        assert bench_name is not None
        return _j(directory, "python_profile_python_overhead_pyprof{bench}.json".format(bench=bench_suffix(bench_name)))

    def nvprof_profile_json_path(self, bench_name):
        return self.get_nvprof_profile_json_path(self.src_files, bench_name)

    @classmethod
    def get_nvprof_profile_json_path(Klass, src_files, bench_name):
        directory = Klass.get_directory(src_files, bench_name)
        assert bench_name is not None
        if bench_name != NO_BENCH_NAME:
            return _j(directory, "nvidia.{bench}.gpu_overhead.nvprof.json".format(bench=bench_name))
        return _j(directory, "nvidia.gpu_overhead.nvprof.json")

    @classmethod
    def get_combined_path(Klass, src_files, bench_name):
        directory = _d(src_files.get('nvprof_profile_path', bench_name))
        # bench_name = get_bench_name(src_files.get('nvprof_profile_path', bench_name),
        #                             allow_none=True)
        assert bench_name is not None
        if bench_name != NO_BENCH_NAME:
            return _j(directory, "{bench}.breakdown.json".format(bench=bench_name))
        return _j(directory, "breakdown.json")

    def _combined_path(self, bench_name):
        return self.get_combined_path(self.src_files, bench_name)

class _FreqParser:
    """
    > Running on CPU: 14
    > cycles[0] = 12021055605 cycles
    > seconds[0] = 5 sec
    > cpu_freq[0] = 2.40421 GHz
    > cycles[1] = 12021049029 cycles
    > seconds[1] = 5 sec
    > cpu_freq[1] = 2.40421 GHz
    > cycles[2] = 12021048798 cycles
    > seconds[2] = 5 sec
    > cpu_freq[2] = 2.40421 GHz
    > cycles[3] = 12021049617 cycles
    > seconds[3] = 5 sec
    > cpu_freq[3] = 2.40421 GHz
    > cycles[4] = 12021049059 cycles
    > seconds[4] = 5 sec
    > cpu_freq[4] = 2.40421 GHz
    > cycles[5] = 12021051003 cycles
    > seconds[5] = 5 sec
    > cpu_freq[5] = 2.40421 GHz
    > cycles[6] = 12021049767 cycles
    > seconds[6] = 5 sec
    > cpu_freq[6] = 2.40421 GHz
    > cycles[7] = 12021049392 cycles
    > seconds[7] = 5 sec
    > cpu_freq[7] = 2.40421 GHz
    > cycles[8] = 12021050112 cycles
    > seconds[8] = 5 sec
    > cpu_freq[8] = 2.40421 GHz
    > cycles[9] = 12021050535 cycles
    > seconds[9] = 5 sec
    > cpu_freq[9] = 2.40421 GHz
    > Mean CPU frequency: 2.40421 GHz
    > Std CPU frequency: 3.77961e-07 GHz
    > Num measurements: 10

    {
    'repetitions':10,
    'cpu_freq_ghz':[ ... ]
    'cpu_freq_ghz_mean':...,
    'cpu_freq_ghz_std':...,
    'cpu_id':...,
    }
    """
    def __init__(self, parser, args,
                 rep_array_name_regex=r"cpu_freq",
                 stat_name_regex=r"CPU frequency",
                 field_name='cpu_freq_ghz'):
        self.parser = parser
        self.args = args
        self.rep_array_name_regex = rep_array_name_regex
        self.stat_name_regex = stat_name_regex
        self.field_name = field_name
        self.mean_name = '{f}_mean'.format(f=self.field_name)
        self.std_name = '{f}_std'.format(f=self.field_name)

    def parse(self, it, all_results):

        # self.results = dict()
        def store(*args, **kwargs):
            store_group(all_results, *args, **kwargs)

        for line in line_iter(it):
            m = re.search(r'> Running on CPU: (?P<cpu_id>\d+)', line)
            if m:
                store(m)
                continue

            m = re.search(r'> Num measurements: (?P<repetitions>\d+)', line)
            if m:
                store(m)
                continue

            regex = r'> {array_name}\[\d+\]\s*[=:]\s*(?P<freq_ghz>{float}) GHz'.format(
                float=float_re,
                array_name=self.rep_array_name_regex)
            m = re.search(regex, line)
            if m:
                store_as(all_results,
                         self.field_name,
                         float(m.group('freq_ghz')),
                         store_type='list')
                continue

            m = re.search(r'> Mean {stat_name}\s*[=:]\s*(?P<freq_ghz_mean>{float}) GHz'.format(
                float=float_re,
                stat_name=self.stat_name_regex), line)
            if m:
                store_as(all_results,
                         self.mean_name,
                         float(m.group('freq_ghz_mean')))
                assert type(all_results[self.mean_name]) == float
                continue

            m = re.search(r'> Std {stat_name}\s*[=:]\s*(?P<freq_ghz_std>{float}) GHz'.format(
                float=float_re,
                stat_name=self.stat_name_regex), line)
            if m:
                store_as(all_results,
                         self.std_name,
                         float(m.group('freq_ghz_std')))
                continue

        expected_keys = set([
            'repetitions',
            self.field_name,
            self.mean_name,
            self.std_name,
            'cpu_id',
        ])
        missing_keys = expected_keys.difference(set(all_results.keys()))
        assert len(missing_keys) == 0

        assert type(all_results[self.mean_name]) == float
        return all_results

class CPUFreqParser(_FreqParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         rep_array_name_regex=r"cpu_freq",
                         stat_name_regex=r"CPU frequency",
                         field_name='cpu_freq_ghz')

class TSCFreqParser(_FreqParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         rep_array_name_regex=r"TSC frequency",
                         stat_name_regex=r"TSC frequency",
                         field_name='tsc_freq_ghz')

class CUDAMicroParser:
    def __init__(self, parser, args, microbench_name, start_header_regex, result_format):
        self.parser = parser
        self.args = args
        self.microbench_name = microbench_name
        self.start_header_regex = start_header_regex
        assert result_format in ['vector']
        self.result_format = result_format

    def parse(self, it, all_results):
        if self.result_format == 'vector':
            self._parse_vector(it, all_results)
        else:
            raise NotImplementedError

    def parse_specific(self, results, line, it):
        return False

    def _parse_vector(self, it, all_results):
        """
        CudaLaunch latency:
        CUDA kernel launch with 1 blocks of 256 threads
        > CudaLaunch latencies:
        Time for running operations on the stream, measuring GPU-side.
                      GPU
                        0  3.90 +/-   0.04
                        1  3.54 +/-   0.00
                        2  3.51 +/-   0.01
                        3  4.44 +/-   0.01

        Time for scheduling operations to the stream (not running them), measuring CPU-side.
                      CPU
                        0  3.18 +/-   0.04
                        1  3.17 +/-   0.04
                        2  3.22 +/-   0.11
                        3  3.17 +/-   0.07

        Time for scheduling + running operations on the stream, measuring CPU-side.
                      CPU
                        0  7.16 +/-   0.06
                        1  6.79 +/-   0.04
                        2  6.82 +/-   0.12
                        3  7.70 +/-   0.07

        {
            'thread_blocks': 1,
            'threads_per_block': 256,
            'gpu_time_usec': {
                'device': {
                    0: (3.90, 0.04),
                    1: (3.54, 0.00),
                    ...
                },
            },
            'cpu_sched_time_usec': {
                'device': {
                    0: (3.18, 0.04),
                    ...
                },
            },
            'cpu_time_usec': {
                'device': {
                    0: (7.16, 0.06),
                    ...
                },
            },
        },
        """
        gpu_time_header_regex = r'(?:Time for running operations on the stream, measuring GPU-side)'
        cpu_sched_time_header_regex = r'(?:Time for scheduling operations to the stream \(not running them\), measuring CPU-side)'
        cpu_time_header_regex = r'(?:Time for scheduling \+ running operations on the stream, measuring CPU-side)'

        def parse_vector(vector, it):
            for line in it:
                m = re.search(r'\s*(?P<device>\d+)\s+(?P<mean>{float})\s+\+/-\s+(?P<std>{float})'.format(float=float_re), line)
                if m:
                    device = int(m.group('device'))
                    mean = float(m.group('mean'))
                    std = float(m.group('std'))
                    put_key(vector, 'device', dict())
                    put_key(vector['device'], device, (mean, std))
                    continue

                m = re.search('^\s*$', line)
                if m:
                    break
            return vector

        def parse_time_vector(results, it, header_regex, time_name):
            m = re.search(header_regex, line)
            if m:
                put_key(results, time_name, dict())
                parse_vector(results[time_name], it)
                return True
            return False

        results = dict()
        # with open(filename) as f:
        #     it = line_iter(f)
        for line in it:
            m = re.search(self.start_header_regex, line)
            if m:
                # Start of CudaLaunch experiment
                break

        for line in it:

            if self.args.debug:
                print("> {klass}, line :: {line}".format(
                      klass=self.__class__.__name__,
                      line=line))

            if self.parse_specific(results, line, it):
                continue

            if parse_time_vector(results, it, gpu_time_header_regex, 'gpu_time_usec'):
                continue
            if parse_time_vector(results, it, cpu_sched_time_header_regex, 'cpu_sched_time_usec'):
                continue
            if parse_time_vector(results, it, cpu_time_header_regex, 'cpu_time_usec'):
                # End of CudaLaunch results
                break

        put_key(all_results, self.microbench_name, results)

        return results

class CUDALaunchParser(CUDAMicroParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         microbench_name='cuda_launch',
                         start_header_regex=r'^CudaLaunch latency:',
                         result_format='vector')

    def parse_specific(self, results, line, it):
        m = re.search(r'CUDA kernel launch with (?P<thread_blocks>\d+) blocks of (?P<threads_per_block>\d+) threads', line)
        if m:
            put_key(results, 'thread_blocks', int(m.group('thread_blocks')))
            put_key(results, 'threads_per_block', int(m.group('threads_per_block')))
            return True
        return False

class CUDAD2HParser(CUDAMicroParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         microbench_name='d2h',
                         start_header_regex=r'^> Device-to-Host latencies',
                         result_format='vector')

class CUDAH2DParser(CUDAMicroParser):
    def __init__(self, parser, args):
        super().__init__(parser, args,
                         microbench_name='h2d',
                         start_header_regex=r'^> Host-to-Device latencies',
                         result_format='vector')

class _BenchmarkFreq:
    def __init__(self, parser, args, ParserType, exec_path):
        self.args = args
        self.parser = parser
        self.ParserType = ParserType
        self.exec_path = exec_path

    @property
    def dump_path(self):
        raise NotImplementedError

    def error(self, msg):
        print("ERROR: {msg}".format(msg=msg))
        sys.exit(1)

    def run_freq(self):
        print("> Running {name} microbenchmarks; output to {path}".format(
            name=type(self).__name__,
            path=self.dump_path))
        if not _e(self.exec_path):
            self.error("Couldn't find CPUFreq microbenchmark executable {exec}; please run 'cmake' from {dir}".format(
                exec=self.exec_path,
                dir=_j(py_config.ROOT, 'build')))
        os.makedirs(_d(self.dump_path), exist_ok=True)
        with open(self.dump_path, 'w') as f:
            cmdline = self.cmdline_array()
            subprocess.check_call(cmdline, stderr=subprocess.PIPE, stdout=f)

    def cmdline_array(self):
        return [self.exec_path]

    def parse_freq(self):
        parser = self.ParserType(self.parser, self.args)
        with open(self.dump_path) as f:
            it = line_iter(f)
            all_results = dict()
            parser.parse(it, all_results)
        return all_results

    def run(self):
        args = self.args
        parser = self.parser

        # PSEUDOCODE:
        # - run p2pBandwidthLatencyTest > cuda_microbench.txt
        # - nice to have, but not a priority:
        #   for num_floating_point_operations in powers_of_two(1, 2, 4, ...):
        #     run cuda-launch microbenchmark
        #     record y=usec/num_floating_point_operations
        #            x=num_floating_point_operations
        #     # NOTE: Just record it for GPU 0

        # if _e(self.dump_path) and not self.args.replace:
        # Ignore replace.
        if _e(self.dump_path):
            print("> Skip {name}; {path} already exists".format(
                name=type(self).__name__,
                path=self.dump_path))
        else:
            self.run_freq()

        self.results = self.parse_freq()

# class BenchmarkCPUFreq(_BenchmarkFreq):
#     def __init__(self, parser, args):
#         exec_path = _j(py_config.ROOT, 'build', 'cpufreq')
#         super().__init__(parser, args, CPUFreqParser, exec_path)
#
#     @property
#     def dump_path(self):
#         return _j(self.args.directory, 'cpufreq.txt')

# class BenchmarkTSCFreq(_BenchmarkFreq):
#     def __init__(self, parser, args):
#         exec_path = _j(py_config.ROOT, 'build', 'clocks')
#         super().__init__(parser, args, TSCFreqParser, exec_path)
#
#     def cmdline_array(self):
#         return [self.exec_path, '--measure_tsc_freq']
#
#     @property
#     def dump_path(self):
#         return _j(self.args.directory, 'tsc_freq.txt')

class BenchmarkCUDA:
    def __init__(self, parser, args):
        self.args = args
        self.parser = parser

    @property
    def _cuda_microbench_path(self):
        return _j(self.args.directory, 'cuda_microbench.txt')

    def error(self, msg):
        print("ERROR: {msg}".format(msg=msg))
        sys.exit(1)

    def cuda_microbench_all(self, bench_name):
        print("> Running CUDA microbenchmarks; output to {path}".format(
            path=self._cuda_microbench_path(bench_name)))
        CUDA_MICROBENCH_EXEC = _j(py_config.ROOT, 'cpp', 'p2pBandwidthLatencyTest', 'p2pBandwidthLatencyTest')
        if not _e(CUDA_MICROBENCH_EXEC):
            self.error("Couldn't find CUDA microbenchmark executable {exec}; please run 'make' from {dir}".format(
                exec=CUDA_MICROBENCH_EXEC,
                dir=_d(CUDA_MICROBENCH_EXEC)))
        with open(self._cuda_microbench_path(bench_name), 'w') as f:
            subprocess.check_call([CUDA_MICROBENCH_EXEC], stderr=subprocess.PIPE, stdout=f)

    def parse_cuda_microbench(self, bench_name):
        parsers = [
            CUDAH2DParser(self.parser, self.args),
            CUDAD2HParser(self.parser, self.args),
            CUDALaunchParser(self.parser, self.args),
        ]
        with open(self._cuda_microbench_path(bench_name)) as f:
            it = line_iter(f)

            all_results = dict()
            for parser in parsers:
                parser.parse(it, all_results)

        return all_results

    def run(self, bench_name):
        args = self.args
        parser = self.parser

        # PSEUDOCODE:
        # - run p2pBandwidthLatencyTest > cuda_microbench.txt
        # - nice to have, but not a priority:
        #   for num_floating_point_operations in powers_of_two(1, 2, 4, ...):
        #     run cuda-launch microbenchmark
        #     record y=usec/num_floating_point_operations
        #            x=num_floating_point_operations
        #     # NOTE: Just record it for GPU 0

        if not args.plot:
            # If we don't JUST want to plot our results.
            if _e(self._cuda_microbench_path(bench_name)) and not self.args.replace:
                print("> Skip CUDA microbenchmarks; {path} already exists".format(path=self._cuda_microbench_path(bench_name)))
            else:
                self.cuda_microbench_all(bench_name)

        self.plot_benchmarks()
        return

    def plot_benchmarks(self, bench_name):
        self.results = self.parse_cuda_microbench(bench_name)

        xs = []
        ys = []
        yerr = []
        # Plot the latency seen by the CUDA API user
        # (don't care how long it ran on the GPU itself, or how long operations get scheduled for).
        time_measurement = 'cpu_time_usec'
        for microbench_name in CUDA_MICROBENCH_NAMES:
            mean, std = self.results[microbench_name][time_measurement]['device'][self.args.gpu]
            # ys.append(mean/MICROSECONDS_IN_SECOND)
            # yerr.append(std/MICROSECONDS_IN_SECOND)
            ys.append(mean)
            yerr.append(std)
            xs.append(microbench_name)

        ys_microseconds = ys
        yerr_microseconds = yerr

        def as_seconds(micros):
            return [x/MICROSECONDS_IN_SECOND for x in micros]

        ys_seconds = as_seconds(ys)
        yerr_seconds = as_seconds(yerr)

        def _plot(ys, yerr, png_basename, ylabel):
            plot_xs_vs_ys(xs, ys, yerr, CUDA_MICROBENCH_NAME_TO_PRETTY,
                          png_basename=png_basename,
                          xlabel='CUDA operation',
                          ylabel=ylabel,
                          title="Latency of GPU operations",
                          directory=self.args.directory)

        _plot(ys_microseconds, yerr_microseconds,
              'cuda.microseconds.png', 'Time (microseconds)')
        _plot(ys_seconds, yerr_seconds,
              'cuda.seconds.png', 'Time (seconds)')


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
    cmdline = [sys.executable] + sys.argv[0:py_script_idx+1]
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

def run_with_nvprof(parser, args):
    print("> Reinvoking script with nvprof.")
    benchmarks = args.benchmarks
    args.benchmarks = None
    assert args.benchmark_type == 'dqn'
    # TODO: for each benchmark, invoke a separate call to this script ( to ensure we only collect data for a specific part of DQN )
    # When collecting nvprof data, should we execute calls repeatedly w/ python or with C?
    # Well, basically it shouldn't matter...but it's probably going to give lower standard
    # deviation if we call the C function directly repeatedl
    # TODO: try both, but use the python call for now.
    for benchmark in benchmarks:
        args.benchmarks = [benchmark]

        # nvprof_files = glob("{dir}/nvidia.{benchmark}.pid_*.nvprof.txt".format(
        #     dir=args.directory,
        #     benchmark=benchmark))
        nvprof_files = glob("{dir}/nvidia.{benchmark}.nvprof.txt".format(
            dir=args.directory,
            benchmark=benchmark))
        if len(nvprof_files) > 0 and not args.replace:
            print("> Skip nvprof; {path} already exists".format(path=nvprof_files[0]))
            continue

        # nvprof_logfile = _j(args.directory, "nvidia.{benchmark}.pid_%p.nvprof.txt".format(
        #     benchmark=benchmark))
        nvprof_logfile = _j(args.directory, "nvidia.{benchmark}.nvprof.txt".format(
            benchmark=benchmark))
        nvprof_sqlite_file = _j(args.directory, "nvidia.{benchmark}.nvprof".format(
            benchmark=benchmark))
        if args.replace and _e(nvprof_sqlite_file):
            # Nvprof fails if the output file already exists.
            os.remove(nvprof_sqlite_file)
        os.makedirs(_d(nvprof_logfile), exist_ok=True)
        nvprof_args = ["nvprof", "-o", nvprof_sqlite_file, "--log-file", nvprof_logfile,
                       "--profile-from-start", "off"]
        cmdline = args_to_cmdline(parser, args)
        argv_exec = nvprof_args + cmdline + [
            "--nvprof-enabled",
            "--nvprof-logfile", nvprof_logfile
        ]
        my_call(argv_exec)

def my_call(argv, env=None):
    if env is None:
        env = os.environ
    # p = subprocess.Popen(argv, env=env)
    # try:
    #     retcode = p.wait()
    # except KeyboardInterrupt:
    #     try:
    #         print("> TERMINATE CHILD: {pid}".format(pid=p.pid))
    #         p.terminate()
    #     except OSError:
    #         pass
    #     retcode = p.wait()
    # if retcode:
    #     # cmd = kwargs.get("args")
    #     # if cmd is None:
    #     # cmd = popenargs[0]
    #     # TODO: print stderr when stuff fails
    #     print("> CMD FAILED: {pid}\n  cmd: {cmd}".format(pid=p.pid, cmd=" ".join(argv)))
    #     raise subprocess.CalledProcessError(retcode, argv)

    p = subprocess.run(argv, env=env, stdout=sys.stdout, stderr=sys.stderr)
    # p = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # For SOME reason this fails from python, but works hen we run the command manually.
    # p = subprocess.run(["ls", "-z"], env=env, stdout=sys.stdout, stderr=sys.stderr)
    # import pprint; pprint.pprint({'p':dir(p)})
    # sys.exit(0)
    # p = subprocess.run(argv, env=env, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        # print("> CMD FAILED:\n  cmd: {cmd}\n> STDERR:\n{stderr}\n> STDOUT:\n{stdout}".format(
        #     cmd=" ".join(argv),
        #     stderr=p.stderr,
        #     stdout=p.stdout))
        print("> CMD FAILED:\n  cmd: {cmd}".format(
            cmd=" ".join(argv),
        ))
        p.check_returncode()
        # Shouldn't reach here.
        assert False
        # raise subprocess.CalledProcessError(p.returncode, argv)
    return 0

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

def plot_xs_vs_ys(
        xs,
        ys,
        yerr,
        name_to_pretty,
        png_basename,
        directory=None,
        log_scale=True,
        std_label=True,
        xlabel=None,
        ylabel=None,
        title=None,
        show=False):
    # c_only=False, python_overhead=False):
    """
                     |
                     |
    Time (seconds)   |
                     |
                     |---------------------------------------------
                       micros[0].name     .....

    PROBLEM: not sure what scale each operation will be on.
    TODO: we want each operation to reflect the proportion of operations that run in DQN.
    """
    assert len(xs) == len(ys) == len(yerr)

    png_path = _j(*[x for x in [directory, png_basename] if x is not None])

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # https://matplotlib.org/examples/api/barchart_demo.html
    width = 0.35
    errorbar_capsize = 10

    xs = [name_to_pretty[name] for name in xs]
    xs = sort_xs_by_ys(xs, ys)
    yerr = sort_xs_by_ys(yerr, ys)
    ys = sorted(ys)

    ind = np.arange(len(xs))/2.

    lists = {'xs':xs,
             'ys':ys,
             'yerr':yerr,
             'ind':ind}
    positive, negative = separate_plus_minus_by(ys, lists)

    def label_bars(rects, xs, ys, yerr, positive):
        """
        Attach a text label above each bar displaying its height
        """
        assert len(rects) == len(xs) == len(ys)
        for rect, x, y, err in zip(rects, xs, ys, yerr):
            # Are we talking about the same bar?
            assert rect.get_height() == y
            # assert rect.get_x() == x

            if std_label:
                bar_label = "{y:f} +/- {std:f}".format(y=y, std=err)
            else:
                bar_label = "{y:f}".format(y=y)

            if positive:
                # Place it above the top of the bar.
                y_pos = 1.05*y
            else:
                # Bar faces downward, place it above the "bottom" of the bar.
                y_pos = 0.05*max(ys)

            ax.text(rect.get_x() + rect.get_width()/2.,
                    y_pos,
                    bar_label,
                    ha='center', va='bottom')

    def add_to_plot(plot_data, color):

        if len(plot_data['ys']) == 0:
            return

        any_is_positive = any(y > 0 for y in plot_data['ys'])
        all_is_positive = all(y > 0 for y in plot_data['ys'])
        assert ( any_is_positive and all_is_positive ) or \
               ( not any_is_positive and not all_is_positive )

        rects1 = ax.bar(plot_data['ind'], plot_data['ys'], width, color=color, yerr=plot_data['yerr'],
                        # bottom=smallest_y,
                        error_kw={
                            'capsize':errorbar_capsize,
                        })

        label_bars(rects1, plot_data['xs'], plot_data['ys'], plot_data['yerr'], positive=any_is_positive)

    add_to_plot(positive, color='r')
    add_to_plot(negative, color='r')

    # import ipdb; ipdb.set_trace()
    ax.set_xticks(ind)
    ax.set_xticklabels(xs)

    if log_scale and not any(y < 0 for y in ys):
        ax.set_yscale("log")
    else:
        print("> WARNING: Saw negative value for {png}; using regular scale instead of log-scale".format(png=png_path))

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           title=title)

    ax.legend()
    ax.grid()
    print("> Save plot to {path}".format(path=png_path))
    fig.savefig(png_path)
    if show:
        plt.show()

def sort_xs_by_ys(xs, ys):
    """
    Sort xs by the elements in ys.

    >>> xs = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    >>> ys = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
    >>> sort_xs_by_ys(xs, ys)
    ["a", "d", "h", "b", "c", "e", "i", "f", "g"]
    """
    assert len(xs) == len(ys)
    return [x for x, y in sorted(zip(xs, ys), key=lambda x_y: x_y[1])]

def separate_plus_minus_by(ys, lists):

    def append_to_list(new_lists, i):
        for key in lists.keys():
            new_lists[key].append(lists[key][i])

    def mk_lists():
        return dict((key, []) for key in lists.keys())
    positive = mk_lists()
    negative = mk_lists()

    for i, y in enumerate(ys):
        if y > 0:
            append_to_list(positive, i)
        else:
            append_to_list(negative, i)

    return positive, negative

def pretty_time(time_sec, use_space=True):
    MS_IN_SEC = 1e3
    US_IN_SEC = 1e6
    NS_IN_SEC = 1e9

    unit_names = ['ms', 'us', 'ns']
    unit_in_sec = [MS_IN_SEC, US_IN_SEC, NS_IN_SEC]

    def format_str(time_as_unit, unit):
        if use_space:
            return "{time} {unit}".format(time=time_as_unit, unit=unit)
        return "{time}{unit}".format(time=time_as_unit, unit=unit)

    if time_sec == 0 or time_sec > 1:
        return format_str(time_sec, 'sec')
    for i, (time_unit, sec_as_unit) in enumerate(zip(unit_names, unit_in_sec)):
        time_as_unit = time_sec*sec_as_unit
        if time_as_unit > 1 or i == len(unit_names) - 1:
            return format_str(time_as_unit, time_unit)
    assert False

def us_to_sec(usec):
    return usec/1e6

def sec_to_us(usec):
    return usec*1e6

def unique(xs):
    return list(set(xs))

def remap_keys(dic, new_key_func):
    new_dic = dict()
    for k, v in dic.items():
        new_key = new_key_func(k)
        new_dic[new_key] = v
    return new_dic

def _mk(dic, key, default):
    if key not in dic:
        dic[key] = default
    return dic[key]

# CUDAProfileParser,
PARSER_KLASSES = [PythonProfileParser, PythonProfileTotalParser, CUDASQLiteParser, CombinedProfileParser, PlotSummary, TimeBreakdownPlot]
PARSER_NAME_TO_KLASS = dict((ParserKlass.__name__, ParserKlass) \
                            for ParserKlass in PARSER_KLASSES)

def common_dir(paths):
    assert len(paths) > 0
    directory = os.path.commonprefix(paths)
    if not os.path.isdir(directory):
        directory = _d(directory)
    try:
        pass  # Cmd
        assert os.path.isdir(directory)
    except Exception as e:
        import pdb;
        pdb.set_trace()
        raise e

    return directory

def as_str(obj, indent=None):
    ss = StringIO()
    if type(obj) == dict:
        pprint.pprint(obj, stream=ss)
    else:
        pprint.pprint(obj.__dict__, stream=ss)
    string = ss.getvalue()
    if indent is not None:
        string = textwrap.indent(string, "  "*indent)
    return string

def bench_suffix(bench):
    if bench != NO_BENCH_NAME:
        return ".{bench}".format(bench=bench)
    return ""

def get_microbench_basename():
  return 'microbenchmark.json'

def get_microbench_path(direc):
  return _j(direc, 'microbenchmark.json')

def is_microbench_path(path):
  return re.search(r"^microbenchmark\.json$", _b(path))

def fixup_json(obj):
  def fixup_scalar(scalar):
    if type(scalar) != str:
      ret = scalar
      return ret

    try:
      ret = int(scalar)
      return ret
    except ValueError:
      pass

    try:
      ret = float(scalar)
      return ret
    except ValueError:
      pass

    ret = scalar
    return ret

  def fixup_list(xs):
    return [fixup_json(x) for x in xs]

  def fixup_dic(dic):
    items = list(dic.items())
    keys = [k for k, v in items]
    values = [v for k, v in items]
    keys = fixup_json(keys)
    values = fixup_json(values)
    new_dic = dict()
    for k, v in zip(keys, values):
      new_dic[k] = v
    return new_dic

  if type(obj) == dict:
    return fixup_dic(obj)
  elif type(obj) == list:
    return fixup_list(obj)
  return fixup_scalar(obj)

if __name__ == '__main__':
    main()
