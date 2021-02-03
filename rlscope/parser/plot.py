"""
matplotlib plotting code for various RL-Scope plots (e.g., time breakdown plot).
"""
from rlscope.profiler.rlscope_logging import logger
import argparse
import copy
import re
import sys
import itertools
import os
import csv
import textwrap
import pprint
import math
from io import StringIO
import json
import codecs
import pandas as pd

from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib
# matplotlib.use('agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from rlscope.profiler.util import pprint_msg
from rlscope import py_config
from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.parser.nvprof import CUDASQLiteParser
from rlscope.parser.pyprof import PythonProfileParser
from rlscope.parser.tfprof import OverlapComputer, overlap_type_to_instance, OverlapJSONToVennConverter
from rlscope.parser.heatscale import HeatScale, exponential_moving_average
from rlscope.parser.db import SQLCategoryTimesReader, sql_get_source_files, sql_input_path
from rlscope.parser.dataframe import UtilDataframeReader, OverlapDataframeReader, VennData, read_rlscope_config_metadata, CUDADeviceEventsReader

Y_LABEL_TRAINING_TIME_SEC = 'Training time (sec)'

# figsize (W x H) in inches
aspect_ratio = 16./9.
fig_width = 2*7
fig_height = float(fig_width) / aspect_ratio
FIG_SIZE = (fig_width, fig_height)

DEVICE_ORDER = ['NoDeviceName', 'Quadro K4000', 'Quadro P4000', 'GTX 1080']
# IMPL_NAME_ORDER = ["DQN Python", NO_IMPL_NAME]
IMPL_NAME_ORDER = [NO_IMPL_NAME]

DQN_BENCH_NAME_LABELS = {
    'q_update_target_network':'Update target network',
    'q_forward':'Q-forward',
    'q_backward':'Q-backward',
    'step':'Step',
    # 'total':'Total',
}

# "dots-per-inch" (pixels-per-inch).
# Matplotlib functions set figure size in inches.
# If we want to set them in pixels, we need to use this to do the conversion.
# NOTE: This takes a WHILE to run...
DPI = plt.figure().get_dpi()
def pixels_as_inches(px):
    px = float(px) / float(DPI)
    return px

# Documentation from
# matplotlib/collections.py @ Collection.set_hatch:
r"""
*hatch* can be one of::

  /   - diagonal hatching
  \   - back diagonal
  |   - vertical
  -   - horizontal
  +   - crossed
  x   - crossed diagonal
  o   - small circle
  O   - large circle
  .   - dots
  *   - stars

Letters can be combined, in which case all the specified
hatchings are done.  If same letter repeats, it increases the
density of hatching of that pattern.

Hatching is supported in the PostScript, PDF, SVG and Agg
backends only.

Unlike other properties such as linewidth and colors, hatching
can only be specified for the collection as a whole, not separately
for each member.

Parameters
----------
hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
"""
HATCH_STYLE_EMPTY = ""
HATCH_STYLES = [
    # 7
    '/', '\\', '|', 'x', 'o', '.', '*',

    # 6
    '/.', '\\.', '|.', 'x.', 'o.', '*.',

    # 5
    '/o', '\\o', '|o', 'xo', '*o',

    # '/', '\\', '|', 'x', 'o', '.', '*',
    # '/', '\\', '|', 'x', 'o', '.', '*',
    # '/', '\\', '|', 'x', 'o', '.', '*',
]
# '+',
# '-',
# 'O',

LINE_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
LINE_STYLES = ['-','--',':','-.']
LINE_THEMES = []
for linestyle in LINE_STYLES:
    for color in LINE_COLORS:
        LINE_THEMES.append({'color':color, 'linestyle':linestyle})

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

        for field in fields:
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
        return re.search(r'TimeSec$', field)


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
        logger.info("> Added Total column for field={field}:".format(field=field))
        logger.info(total_rows)
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

        logger.info("> DataFrame:")

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

# Generalize:
# Instead of just "Python", "C++", "GPU", we want to break down the labels arbitrairily.
# Also, we want to control the order of the legend labels.

class CategoryOverlapPlot:
    """
    Create a stacked bar plot.
    For the list of times to show, see self.category_order.

    CategoryOverlapPlot

      PSEUDOCODE:
      For each op/bench_name:
          op_overlaps = []
          For each process:
              For each "step"/call to op:
                  overlaps.add { Compute category-overlap for this call to op }
          Report average category-overlap across all calls to op (across ALL processes)
            i.e. avg(overlaps)

    """

    # Make accessible from analyze.py
    OVERLAP_TYPES = OverlapComputer.OVERLAP_TYPES

    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 debug=False,
                 debug_single_thread=False,
                 group_by_phase=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.host = host
        self.user = user
        self.password = password

        self.bar_width = 0.25
        self.show = False

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    def _category_overlap_png(self, bench_name):
        return CategoryOverlapPlot.get_category_overlap_png(self.directory, bench_name)

    @staticmethod
    def get_category_overlap_png(directory, bench_name):
        return _j(directory, "category_overlap{bench}.png".format(
            bench=bench_suffix(bench_name)))

    @staticmethod
    def get_plot_data_path(directory, bench_name):
        return _j(directory, "category_overlap.plot_data{bench}.txt".format(
            bench=bench_suffix(bench_name)))

    def _plot_data_path(self, bench_name):
        return CategoryOverlapPlot.get_plot_data_path(self.directory, bench_name)

    @staticmethod
    def get_stats(directory, bench_name):
        return _j(directory, "category_overlap.stats{bench}.json".format(
            bench=bench_suffix(bench_name)))

    def _stats(self, bench_name):
        return CategoryOverlapPlot.get_stats(self.directory, bench_name)

    def run(self):

        """
        PSEUDOCODE:
        if group_by_phase:
            for phase in phases

        """

        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)
        self.bench_names = self.sql_reader.bench_names()
        assert len(self.bench_names) == len(unique(self.bench_names))
        self.categories = self.sql_reader.categories

        overlap_computer = OverlapComputer(self.db_path, host=self.host, user=self.user, password=self.password,
                                           debug=self.debug,
                                           debug_single_thread=self.debug_single_thread)

        all_categories = set()
        json_datas = []
        for bench_name in self.bench_names:
            start_t = time.time()
            json_data = overlap_computer.compute_per_operation_overlap(bench_name)
            json_datas.append(json_data)

            for combo_and_times in json_data['category_combo_times']:
                category = _category_str(combo_and_times['category_combo'])
                all_categories.add(category)
            end_t = time.time()
            # > compute_per_operation_overlap(bench_name=tree_search) took 400.79692363739014 seconds
            # > compute_per_operation_overlap(bench_name=eval_game) took 778.2323503494263 seconds
            # This thing takes a LONG time.
            logger.info("> compute_per_operation_overlap(bench_name={op}) took {sec} seconds".format(
                op=bench_name,
                sec=end_t - start_t,
            ))

        pprint.pprint({'all_categories': all_categories})

        self.category_order = sorted(all_categories)
        self.bench_name_labels = DQN_BENCH_NAME_LABELS
        self.category_color_map = None
        self.category_labels = None
        self.impl_name_order = IMPL_NAME_ORDER
        self.device_order = DEVICE_ORDER
        self.plotter = StackedBarPlotter(
            self._category_overlap_png, self._plot_data_path,
            self.category_order,
            self.impl_name_order,
            self.device_order,
            bench_name_labels=self.bench_name_labels,
            category_color_map=self.category_color_map,
            category_labels=self.category_labels,
            bar_width=self.bar_width, show=self.show,
            json_reader_klass=TFProfReader,
            title='DQN iteration time breakdown',
            xlabel='DQN',
            ylabel='Time (seconds)',
        )
        for bench_name, json_data in zip(self.bench_names, json_datas):
            device_name = NO_DEVICE_NAME
            impl_name = NO_IMPL_NAME
            self.plotter.add_json_data(json_data, bench_name,
                                       device_name, impl_name, debug=True)

        # for bench_name in [NO_BENCH_NAME]:
        for bench_name in self.bench_names + [NO_BENCH_NAME]:
            self.plotter.plot(bench_name)
            self._dump_cpu_gpu_stats(bench_name)

    def _dump_cpu_gpu_stats(self, bench_name):
        js_stats = dict()
        _add_cpu_gpu_stats(js_stats, self.plotter, bench_name)
        print("> Save plot stats to {path}".format(path=self._stats(bench_name)))
        do_dump_json(js_stats, self._stats(bench_name))

    @property
    def db_path(self):
        return sql_input_path(self.directory)

class StackedBarPlotter:
    def __init__(self,
                 get_png,
                 # get_png_legend,
                 get_plot_data_path,
                 category_order,
                 impl_name_order,
                 device_order,
                 bench_name_labels=None,
                 category_color_map=None,
                 category_labels=None,
                 bar_width=0.25, show=False,
                 json_reader_klass=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 reversed_labels=False,
                 yvalue_per_pixel=None,
                 width_px=500,
                 dynamic_size=False,
                 # reversed_labels=True,
                 ):

        if callable(get_png):
            self.get_png = get_png
            self.png = None
        else:
            assert type(get_png) == str
            self.get_png = None
            self.png = get_png

        # if callable(get_png_legend):
        #     self.get_png_legend = get_png_legend
        #     self.png_legend = None
        # else:
        #     assert type(get_png_legend) == str
        #     self.get_png_legend = None
        #     self.png_legend = get_png_legend

        if callable(get_plot_data_path):
            self.get_plot_data_path = get_plot_data_path
            self.plot_data_path = None
        else:
            assert type(get_plot_data_path) == str
            self.get_plot_data_path = None
            self.plot_data_path = get_plot_data_path

        self.yvalue_per_pixel = yvalue_per_pixel
        self.width_px = width_px
        self.dynamic_size = dynamic_size
        if self.dynamic_size:
            assert self.yvalue_per_pixel is not None and self.width_px is not None

        self.bar_width = bar_width
        self.show = show
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.reversed_labels = reversed_labels
        assert json_reader_klass is not None
        self.json_reader_klass = json_reader_klass
        self.value_field = 'time_sec'
        self.df_data = {
            "bench_name":[],
            "bench_name_order":[],
            "impl_name":[],
            "impl_name_order":[],
            "device":[],
            "device_order":[],
            # Name of time: PythonTimeSec, CppTimeSec, etc.
            "category":[],
            "category_order":[],
            # Value for <category>.
            "time_sec":[],
        }
        self.df = None

        self.category_order = category_order
        if category_labels is None:
            category_labels = dict((k, k) for k in self.category_order)
        self.category_labels = category_labels
        self.impl_name_order = impl_name_order
        self.device_order = device_order

        self.bench_name_labels = bench_name_labels

        if category_color_map is None:
            category_color_map = self.as_color_map(self.category_order)
        self.category_color_map = category_color_map
        self._check_has_keys(self.category_order, self.category_color_map)

        self.category_order_map = as_order_map(self.category_order)
        self.rev_category_order_map = reverse_dict(self.category_order_map)
        self._check_has_keys(self.category_order, self.category_labels)

        self.impl_name_order_map = as_order_map(self.impl_name_order)
        self.rev_impl_name_order_map = reverse_dict(self.impl_name_order_map)

        self.device_order_map = as_order_map(self.device_order)
        self.rev_device_order_map = reverse_dict(self.device_order_map)

    def as_hatch_map(self, xs):
        # Need enough distinct hash-styles to fit categories.
        # assert len(xs) <= len(HATCH_STYLES)
        if len(xs) > len(HATCH_STYLES):
            logger.info("> WARNING: We only have {h} HATCH_STYLES, but there are {x} category-overlap labels".format(
                h=len(HATCH_STYLES),
                x=len(xs),
            ))
        hatch_map = dict()
        for x, hatch_style in zip(xs, itertools.cycle(HATCH_STYLES)):
            hatch_map[x] = hatch_style
        return hatch_map

    def _build_bench_name_order(self):
        # # Delay making this until we know all the bench_name's from add_json_data
        # self.bench_name_order = ['q_update_target_network', 'q_forward', 'q_backward', 'step']
        self.bench_name_order = sorted(unique(self.df_data['bench_name']))
        # Doesn't work, hatches used change but label order still backwards.
        # self.bench_name_order = list(reversed(sorted(unique(self.df_data['bench_name']))))
        pprint.pprint({'bench_name_order':self.bench_name_order})
        self.bench_name_order_map = as_order_map(self.bench_name_order)
        self.rev_bench_name_order_map = reverse_dict(self.bench_name_order_map)

        # if self.reversed_labels:
        #     # Show "simplest" hatch styles at the top of the plot always
        #     self.bench_name_hatch_map = self.as_hatch_map(list(reversed(self.bench_name_order)))
        # else:
        #     self.bench_name_hatch_map = self.as_hatch_map(self.bench_name_order)

        # we WANT this if reversed_labels = True
        # AND
        # we WANT this if reversed_labels = False
        self.bench_name_hatch_map = self.as_hatch_map(list(reversed(self.bench_name_order)))

        if self.bench_name_labels is None:
            self.bench_name_labels = dict((k, k) for k in self.bench_name_order)
        self.df_data['bench_name_order'] = [self.bench_name_order_map[bench_name] for bench_name in self.df_data['bench_name']]

    def plot_data(self, bench_name=NO_BENCH_NAME):
        if self.df is None:
            self._as_dataframe()

        if bench_name == NO_BENCH_NAME:
            bench_df = self.mean_df
        else:
            bench_df = self.mean_df[self.mean_df['bench_name'] == bench_name]

        return bench_df

    def plot_legend(self):
        figlegend = plt.figure(figsize=(3,2))
        # ax = fig.add_subplot(111)
        # lines = ax.plot(range(10), range(10), range(10), range(10))
        figlegend.legend(plot, ('one', 'two'), 'center')
        figlegend.savefig(
            'test_just_legend.png',
            bbox_inches="tight")


    def plot(self, bench_name=NO_BENCH_NAME):
        if self.df is None:
            self._as_dataframe()

        # Keep this...
        # fig = plt.figure()

        MATPLOTLIB_PIXEL_FACTOR = 1e-2
        self.width_px = 500

        if bench_name is None:
            all_benches = [NO_BENCH_NAME]
        elif bench_name == NO_BENCH_NAME:
            all_benches = self.get_plot_bench_names()
        else:
            all_benches = [bench_name]

        for bench_name in all_benches:
            bench_df = self.plot_data(bench_name)

            if self.dynamic_size:
                # TODO: Use test_pixel_bar to make a plot exactly pixels_as_inches(height_px) in height
                # TODO: Use test_just_legend to make a plot just the legend
                # TODO: Use test_just_legend to make a plot just the legend
                # TODO: Use test_stacked_bar to test.
                total_yvalue = bench_df['mean'].max()
                height_px = math.ceil(total_yvalue / self.yvalue_per_pixel)
                assert height_px > 1
                fig = plt.figure(figsize=(pixels_as_inches(self.width_px), pixels_as_inches(height_px)))

                # left = 0.0
                # bottom = 0.0
                # width = 1.0
                # height = 1.0
                # ax = fig.add_axes([left, bottom, width, height])

                #
                # Test test_pixel_bar
                #

                # How much wiggle-room to reserve along the outside of the plot-area for the plot-area-outline?
                pixels_for_outline = 2

                width_percent_per_pixel = 1.0 / self.width_px
                width_percent_line_spacer = pixels_for_outline * width_percent_per_pixel

                height_percent_per_pixel = 1.0 / height_px
                height_percent_line_spacer = pixels_for_outline * height_percent_per_pixel

                left = 0.0 + width_percent_line_spacer
                bottom = 0.0 + height_percent_line_spacer
                width = 1.0 - 2*width_percent_line_spacer
                height = 1.0 - 2*height_percent_line_spacer

                # This guarantees the plot-area fills the entire 500x500 figure size.
                ax = fig.add_axes([left, bottom, width, height])


                # Q: Pass ax around? What do use for ax below?
                assert len(fig.get_axes()) == 1
            else:
                fig = plt.figure()
                # Q: What's this affect?
                # ax = plt.add_subplot(111)
                ax = fig.add_subplot(111)

            with open(self.get_plot_data_pt(bench_name), 'w') as f:
                DataFrame.print_df(bench_df, file=f)
            logger.info("> DataFrame:")
            print(bench_df)

            self._add_lines(bench_name)

            if self.dynamic_size:
                #
                # Test test_just_legend
                #

                fig_legend = plt.figure(figsize=(3,2))
                # ax = fig.add_subplot(111)
                # lines = ax.plot(range(10), range(10), range(10), range(10))
                # fig_legend.legend(lines, ('one', 'two'), 'center')
                ax_legend = fig_legend.add_subplot(111)
                legend_png = self.get_png_legend_path(bench_name)
                self._add_legend(bench_name, axis=ax_legend)
                fig_legend.savefig(
                    legend_png,
                    bbox_inches="tight")
                assert ax.get_legend() is None
                # ax.get_legend().remove()
            else:
                self._add_legend(bench_name)
            self._add_axis_labels(bench_name)

            if self.dynamic_size:
                #
                # Test test_pixel_bar
                #

                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False, # labels along the bottom edge are off
                )

                # Put y-label inside the plot
                ax.tick_params(axis="y", direction="in", pad=-22)

                # Adjust y-min we can see the "0" y-label still.
                ymin, ymax = ax.get_ylim()
                assert ymin == 0
                ax.set_ylim(-( 0.05*ymax ), ymax)

            self._show(fig, bench_name)

    def get_plot_bench_names(self):
        if self.get_png is not None:
            all_benches = [NO_BENCH_NAME] + unique(self.mean_df['bench_name'])
            return all_benches

        return [NO_BENCH_NAME]

    def get_png_path(self, bench_name):
        if self.get_png is not None:
            return self.get_png(bench_name)

        return self.png

    def get_png_legend_path(self, bench_name):
        png_path = self.get_png_path(bench_name)
        legend_png_path = re.sub(r'\.png$', '.legend.png', png_path)
        return legend_png_path

        # if self.get_png_legend is not None:
        #     return self.get_png_legend(bench_name)
        #
        # return self.png_legend

    def get_plot_data_pt(self, bench_name):
        if self.get_plot_data_path is not None:
            return self.get_plot_data_path(bench_name)

        return self.plot_data_path

    def _show(self, fig, bench_name=None):
        if self.show:
            plt.show()
        else:
            print("> Save figure to {path}".format(path=self.get_png_path(bench_name)))
            print("> Save plot data to {path}".format(path=self.get_plot_data_pt(bench_name)))
            if not self.dynamic_size:
                fig.savefig(self.get_png_path(bench_name), bbox_inches="tight")
            else:
                fig.savefig(self.get_png_path(bench_name))
            plt.close()

    def add_json_data(self, json_data, bench_name, device, impl_name, debug=False,
                      expect_times=False):
        # bench_names = self._get_bench_names(json_data)
        # # device = json_data['attrs']['name']
        # # impl_name = json_data['attrs']['impl_name']
        # for bench_name in bench_names:
        json = self.json_reader_klass(json_data)
        categories = [k for k in json.get_categories() if k in self.category_order]
        # categories = self._get_categories(json_data)
        if debug:
            logger.info("> bench_name={op}, json_data = ".format(op=bench_name))
            logger.info(textwrap.indent(pprint.pformat(json_data), prefix="  "))
            logger.info("  > categories = {c}".format(c=categories))
        for category in categories:
            if debug:
                logger.info("> add category={c}: {times}".format(
                    c=category,
                    times=json.get_times_sec(category)))
                    # times=json_data[category]))
            times_seconds = json.get_times_sec(category)
            if expect_times:
                assert len(times_seconds) > 0
            for time_sec in times_seconds:
                self.df_data['bench_name'].append(bench_name)
                # self.df_data['bench_name_order'].append(self.bench_name_order_map[bench_name])
                self.df_data['impl_name'].append(impl_name)
                self.df_data['impl_name_order'].append(self.impl_name_order_map[impl_name])
                self.df_data['device'].append(device)
                self.df_data['device_order'].append(self.device_order_map[device])
                self.df_data['category'].append(category)
                self.df_data['category_order'].append(self.category_order_map[category])
                self.df_data[self.value_field].append(time_sec)

    def _add_lines(self, bench_name=None):

        self._bottom = None
        def _add_line(impl_name, bench_name):
            for category in self.category_order:
                rows = self.df[
                    (self.df['bench_name'] == bench_name)
                    & (self.df['category'] == category)
                    ]
                if len(rows) == 0:
                    continue
                xvalues = self._get_xvalues(rows['impl_name'], rows['device'])
                yvalues = rows['mean'].values
                yerr = rows['std'].values

                color = self.category_color_map[category]
                hatch = self.bench_name_hatch_map[bench_name]

                if self._bottom is None:
                    self._bottom = np.zeros_like(yvalues)

                # PROBLEM: if data is missing for step
                assert self._bottom.shape == yvalues.shape

                plot = plt.bar(xvalues, yvalues, color=color, width=self.bar_width, edgecolor='white', label=bench_name,
                               bottom=self._bottom,
                               hatch=hatch,
                               yerr=yerr)

                self._bottom += yvalues

        for impl_name in self.impl_name_order:
            if bench_name == NO_BENCH_NAME:
                for bench_name in self.bench_name_order:
                    _add_line(impl_name, bench_name)
            else:
                _add_line(impl_name, bench_name)

    def _add_legend(self, bench_name=None, axis=None):
        self.legend_makers = []

        # We need two groups of lines:
        #
        # 1) Hatch-type:
        #    - Should have the same color
        #    - # of hash-types = len(category_order = ['GPUTimeSec', 'CppTimeSec', 'PythonTimeSec'])
        #                      = 3
        #
        # 2) Color-type:
        #    - Should have the same hatch.
        #    - # of color-types = len(bench_name_order = ['q_forward', 'q_backward', 'step'])
        #                       = 3

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        legend_kwargs = []

        hatch_legend = LegendMaker(attr_name='hatch',
                                   field_to_attr_map=self.bench_name_hatch_map,
                                   field_order=self.bench_name_order,
                                   labels=self.bench_name_labels,
                                   legend_kwargs={
                                       'labelspacing': 1.2,
                                       'handlelength': 3,
                                       'handleheight': 2,
                                   },
                                   reversed=self.reversed_labels)
        legend_kwargs.append({
            # NOTE:
            # - Internally LegendMaker uses the figure coordinate system.
            # - So, (1, 1) is the (right, top) of the whole figure,
            #   so 1.04 makes it just a bit to the right of the whole figure
            'bbox_to_anchor': (1.04, 1)})
        self.legend_makers.append(hatch_legend)

        color_legend = LegendMaker(attr_name='facecolor',
                                   field_to_attr_map=self.category_color_map,
                                   field_order=self.category_order,
                                   labels=self.category_labels,
                                   edgecolor='white',
                                   legend_kwargs={
                                       'handlelength': 3,
                                       'handleheight': 2,
                                   },
                                   reversed=self.reversed_labels)
        # legend_kwargs.append({
        #
        #     # NOTE:
        #     # - Internally LegendMaker uses the figure coordinate system.
        #     # - So, (1, 0) is the (right, bottom) of the whole figure,
        #     #   so 1.04 makes it just a bit to the right of the whole figure
        #     'bbox_to_anchor': (1.04, 0)})
        legend_kwargs.append({
            # 'loc':'lower left',
            # 'loc':'left',
            # 'bbox_to_anchor': (0, -1),

            # Place legend beneath plot so it has room to grow when there are lots of labels,
            # without overlapping the other legend.
            # Sadly, I still don't understand how this thing works.
            # (e.g. I'm not sure how to left-align the legend beneath the plot... OH WELL).
            #
            # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
            'loc':'upper center',
            'bbox_to_anchor':(0.5, -0.05),
            'fancybox':True,
            'shadow':True,
            # 'ncol':5,

        })
        self.legend_makers.append(color_legend)

        LegendMaker.add_legends_multiple(
            self.legend_makers,
            axis=axis,
            legend_kwargs=legend_kwargs)

        # LegendMaker.add_legends_vertically(self.legend_makers,
        #                         legend_kwargs=legend_kwargs)


    # def _get_categories(self, json_data):
    #     return list(k for k in json_data.keys() if k in self.category_order)

    @property
    def dataframe(self):
        if self.df is None:
            self._as_dataframe()
        return self.df

    def _as_dataframe(self):
        self._build_bench_name_order()

        # devices = list(data.keys())
        self.orig_df = pd.DataFrame(self.df_data)

        self.df = DataFrame.get_mean_std(self.orig_df, self.value_field)
        logger.info("> DATAFRAME BEFORE SORT:")
        logger.info(self.df)
        self.df = self.df.sort_values(by=['impl_name_order', 'device_order', 'bench_name_order', 'category_order'])
        # self.df = self.df.sort_values(by=['impl_name_order', 'device_order', 'bench_name_order', 'category_order'], ascending=False)
        # self.df = self.df.sort_values(by=['bench_name_order'])
        logger.info("> DATAFRAME AFTER SORT:")
        logger.info(self.df)
        # groupby_cols = DataFrame.get_groupby_cols(self.orig_df, value_field)
        self.df['std_div_mean_percent'] = 100 * self.df['std']/self.df['mean']

        self.mean_df = self.df


    def _add_axis_labels(self, bench_name=None):
        if self.title is not None and not self.dynamic_size:
            plt.title(self.title)

        if self.xlabel is not None and not self.dynamic_size:
            # , fontweight='bold'
            if bench_name == NO_BENCH_NAME:
                plt.xlabel(self.xlabel)
            else:
                plt.xlabel(get_pretty_bench(bench_name))

        if self.ylabel is not None and not self.dynamic_size:
            plt.ylabel(self.ylabel)

        if not self.dynamic_size:
            n_bars = len(self.device_order)
            xtick_xvalues = self._xtick_xvalues(self.impl_name_order, self.impl_name_order_map, n_bars)
            plt.xticks(xtick_xvalues, self.impl_name_order)


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

    def _check_has_keys(self, xs, xs_map):
        for x in xs:
            assert x in xs_map

class PlotSummaryReader:
    def __init__(self, json_data):
        self.json_data = json_data

    def get_categories(self):
        return list(k for k in self.json_data.keys() if k in self.category_order)
        return list(k for k in self.json_data.keys())

    def get_times_sec(self, category):
        return self.json_data[category][1:]

def _category_str(category_combo):
    """
    This determines "execution category" plot labels in the graphs.
    e.g. "CUDA API CPU + GPU"

    :param category_combo:
    :return:
    """
    assert type(category_combo) in [list, tuple, frozenset, set]

    # HACK to make CategoryOverlapPlot more readable...
    # technically "Framework API C" overlaps with all the "GPU" time and other stuff, but it makes things annoying to read.
    # So, only keep "Framework API C" if it is the only category in the combo, otherwise remove it.

    new_category_combo = list(category_combo)
    def _maybe_remove(category):
        if len(new_category_combo) > 1 and category in new_category_combo:
            new_category_combo.remove(category)

    # NOTE: order of category removal matters here.
    # If it comes down to a single category, only the LAST element of the list will be kept.
    for category in [constants.CATEGORY_OPERATION, constants.CATEGORY_TF_API]:
        _maybe_remove(category)

    return " + ".join(sorted(new_category_combo))

class TFProfReader:
    """
    > json_data =
    { 'categories': ['CUDA API CPU', 'Framework API C', 'GPU', 'Python'],
      'category_combinations': [ ['CUDA API CPU'],
                                 ['CUDA API CPU', 'GPU'],
                                 ['Framework API C'],
                                 ['Framework API C', 'Python'],
                                 ['GPU'],
                                 ['Python']],
      'category_combo_times': [ { 'category_combo': ['CUDA API CPU', 'GPU'],
                                  'times_usec': [1697.0, 1706.0, 1909.0, 1567.0]},
                                { 'category_combo': ['Python'],
                                  'times_usec': [884.0, 848.0, 921.0, 955.0]},
                                { 'category_combo': ['CUDA API CPU'],
                                  'times_usec': [5153.0, 5762.0, 4238.0, 5038.0]},
                                { 'category_combo': ['Framework API C'],
                                  'times_usec': [6355.0, 6291.0, 7505.0, 6915.0]},
                                { 'category_combo': ['Framework API C', 'Python'],
                                  'times_usec': [0.0, 0.0, 0.0, 0.0]},
                                { 'category_combo': ['GPU'],
                                  'times_usec': [1391.0, 1390.0, 1172.0, 1531.0]}]}
    """
    def __init__(self, json_data):
        self.json_data = json_data
        self.categories = [_category_str(category_combo)
                           for category_combo in self.json_data['category_combinations']]

    def get_categories(self):
        return self.categories

    def get_times_sec(self, category):
        for cat_data in self.json_data['category_combo_times']:
            category_combo = cat_data['category_combo']
            cat_str = _category_str(category_combo)
            # times_sec = np.array(cat_data['times_usec'])/constants.MICROSECONDS_IN_SECOND
            if len(cat_data['times_usec']) > 0:
                float_type = type(cat_data['times_usec'][0])
            else:
                float_type = type(constants.MICROSECONDS_IN_SECOND)
            times_sec = np.array(cat_data['times_usec'])/as_type(constants.MICROSECONDS_IN_SECOND, float_type)

            if category == cat_str:
                # Ignore the first time since it includes libcupti.so load time.
                return times_sec[1:]
        logger.info("> json_data = ")
        pprint.pprint(self.json_data, indent=2)
        raise RuntimeError("Couldn't find category=\"{cat}\"".format(cat=category))

class ProcessTimelineReader:
    """
    > json_data = operation_overlap[operation category]
    # operation_overlap: dict
    #   set(operation categories) -> set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
    #    <q_forward, q_backward>       <CPU>, <GPU>, <CPU, GPU>             0.001 sec
    #
    #                                -----------------------------------------------------------

    { 'categories': ['CUDA API CPU', 'Framework API C', 'GPU', 'Python'],
      'category_combinations': [ ['CUDA API CPU'],
                                 ['CUDA API CPU', 'GPU'],
                                 ['Framework API C'],
                                 ['Framework API C', 'Python'],
                                 ['GPU'],
                                 ['Python']],
      'category_combo_times': [ { 'category_combo': ['CUDA API CPU', 'GPU'],
                                  'times_usec': [1697.0, 1706.0, 1909.0, 1567.0]},
                                { 'category_combo': ['Python'],
                                  'times_usec': [884.0, 848.0, 921.0, 955.0]},
                                { 'category_combo': ['CUDA API CPU'],
                                  'times_usec': [5153.0, 5762.0, 4238.0, 5038.0]},
                                { 'category_combo': ['Framework API C'],
                                  'times_usec': [6355.0, 6291.0, 7505.0, 6915.0]},
                                { 'category_combo': ['Framework API C', 'Python'],
                                  'times_usec': [0.0, 0.0, 0.0, 0.0]},
                                { 'category_combo': ['GPU'],
                                  'times_usec': [1391.0, 1390.0, 1172.0, 1531.0]}]}
    """
    def __init__(self, json_data):
        self.json_data = json_data
        self.categories = [_category_str(category_combo)
                           for category_combo in self.json_data.keys()]

    def get_categories(self):
        return self.categories

    def get_times_sec(self, category):
        for category_combo, time_us in self.json_data.items():
            cat_str = _category_str(category_combo)
            if category == cat_str:
                return [time_us/as_type(constants.MICROSECONDS_IN_SECOND, type(time_us))]
        # for cat_data in self.json_data['category_combo_times']:
        #     category_combo = cat_data['category_combo']
        #     cat_str = _category_str(category_combo)
        #     times_sec = np.array(cat_data['times_usec'])/constants.MICROSECONDS_IN_SECOND
        #     if category == cat_str:
        #         # Ignore the first time since it includes libcupti.so load time.
        #         return times_sec[1:]
        logger.info("> json_data = ")
        pprint.pprint(self.json_data, indent=2)
        raise RuntimeError("Couldn't find category=\"{cat}\"".format(cat=category))

# Refactor this to read in json files.
# Q: where to record device/impl_name?  Can we add it to json_data?
# TimeBreakdownPlot.add_json_data(json_data, impl_name=..., device=...)
# TimeBreakdownPlot.add_json_data(...)
# ...
# TimeBreakdownPlot.plot()
class TimeBreakdownPlot(ProfilerParserCommonMixin):
    """
    Create a stacked bar plot.
    For the list of times to show, see self.category_order.
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
        self.parser = parser
        self.args = args
        self.src_files = src_files

        self.category_hatch_map = {
            'GPUTimeSec':"\\",
            'CppTimeSec':"|",
            'PythonTimeSec':"/",
        }
        self.category_order = ['GPUTimeSec', 'CppTimeSec', 'PythonTimeSec']
        self.category_labels = {
            'GPUTimeSec':'GPU time',
            'CppTimeSec':'C++ time',
            'CudaCppTimeSec':'CUDA C time (API calls, driver)',
            'FrameworkCppTimeSec':'Framework C time',
            'PythonTimeSec':'Python time',
        }
        self.bench_name_labels = {
            'q_update_target_network':'Update target network',
            'q_forward':'Q-forward',
            'q_backward':'Q-backward',
            'step':'Step',
            # 'total':'Total',
        }
        self.impl_name_order = IMPL_NAME_ORDER
        self.device_order = DEVICE_ORDER
        self.plotter = StackedBarPlotter(
            self._time_breakdown_png, self._plot_data_path,
            self.category_order,
            self.impl_name_order,
            self.device_order,
            bench_name_labels=self.bench_name_labels,
            category_hatch_map=self.category_hatch_map,
            category_labels=self.category_labels,
            bar_width=bar_width, show=show,
            json_reader_klass=PlotSummaryReader,
        )

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
                self.plotter.add_json_data(json_data, bench,
                                           device_name, impl_name)
                # pretty_bench, device_name)
        self.plotter.plot()

    def _get_bench_names(self, json_data):
        return json_data.keys()

class LegendMaker:
    """
    Create "Patches" to create a legend.
    https://matplotlib.org/users/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists

    :param reversed
        If true, label-order is reversed from order of bar-stacks in plot.
    """
    def __init__(self, attr_name, field_to_attr_map, field_order, labels=None,
                 edgecolor="black",
                 facecolor="white",
                 legend_kwargs=None,
                 reversed=False,
                 **kwargs):
        self.attr_name = attr_name
        self.reversed = reversed
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

    def get_legend_handles_labels_kwargs(self, **kwargs):
        if self.labels is not None:
            legend_labels = [self.labels.get(field, field) for field in self.field_order]
        else:
            legend_labels = [field for field in self.field_order]

        if self.legend_kwargs is not None:
            legend_kwargs = dict(self.legend_kwargs)
        else:
            legend_kwargs = dict()
        legend_kwargs.update(kwargs)

        if not self.reversed:
            handles = list(reversed(self.patches))
            labels = list(reversed(legend_labels))
            return handles, labels, legend_kwargs

        # Default stacked-bar behaviour shows labels in reverse order from the stacked bars.
        # That's just silly.
        #
        # NOTE: "reversed=True" implies this case (the backwards default behaviour)
        handles = list(self.patches)
        labels = legend_labels
        return handles, labels, legend_kwargs

    def get_legend(self, axis=None, **kwargs):
        handles, labels, legend_kwargs = self.get_legend_handles_labels_kwargs(**kwargs)
        self.legend = axis.legend(handles=handles, labels=labels, **legend_kwargs)
        return self.legend

    @staticmethod
    def add_legends_single(legend_makers, axis=None, legend_kwargs=dict()):
        legends = []
        all_handles = []
        all_labels = []
        for i, legend_maker in enumerate(legend_makers):
            handles, labels, _ = legend_maker.get_legend_handles_labels_kwargs(**legend_kwargs)
            all_handles.extend(handles)
            all_labels.extend(labels)

        if axis is None:
            axis = plt.gca()

        legend = axis.legend(handles=all_handles, labels=all_labels, **legend_kwargs)
        axis.add_artist(legend)

        return legends

    @staticmethod
    def add_legends_multiple(legend_makers, axis=None, legend_kwargs=[]):
        legends = []
        for i, legend_maker in enumerate(legend_makers):
            if i < len(legend_kwargs):
                legend_kwarg = legend_kwargs[i]
            else:
                legend_kwarg = dict()
            legend = legend_maker.get_legend(axis=axis, **legend_kwarg)
            legends.append(legend)

        if axis is None:
            axis = plt.gca()

        for legend in legends:
            axis.add_artist(legend)

        return legends

    # @staticmethod
    # def add_legends_vertically(legend_makers, legend_kwargs=[]):
    #
    #     # left, bottom, width, height
    #     # (right, top + spacing)
    #     ax = plt.axes([1.04, 1, 0, 0])
    #
    #     legends = []
    #     for i, legend_maker in enumerate(legend_makers):
    #         if i < len(legend_kwargs):
    #             legend_kwarg = legend_kwargs[i]
    #         else:
    #             legend_kwarg = dict()
    #         legend = legend_maker.get_legend(axis=ax, **legend_kwarg)
    #         legends.append(legend)
    #     transform = None
    #     for legend in legends:
    #         if transform is None:
    #             # transform = legend.get_transform()
    #             # transform = legend.transAxes
    #             transform = ax.transAxes
    #         else:
    #             legend.set_transform(transform)
    #             # Place the legend beneath the previous legend.
    #             # Use the previous legend's axes coordinate system to do this.
    #             # (0, 1) is the (left, bottom) of the previous legend,
    #             # so (0, 1.04) adds vertical spacing of .04
    #             legend.set_bbox_to_anchor((0, 1.04))
    #         art = plt.gca().add_artist(legend)
    #         # art = ax.add_artist(legend)
    #         # logger.info('HI')
    #     return legends


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
        logger.info("> Created combined profile breakdown @ {path}".format(path=json_path))

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

class ResourceOverlapPlotData:
    """
    CPU/GPU utilization over training.
    """
    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 step_sec=1.,
                 pixels_per_square=10,
                 dynamic_size=False,
                 debug=False,
                 debug_single_thread=False,
                 debug_ops=False,
                 debug_memoize=False,
                 entire_trace=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directory:

        :param step_sec:
        :param pixels_per_square:
            Condensed:
            e.g.
            sec_per_pixel = 0.1 sec / pixel
            sec_per_pixel = step_sec / pixels_per_square
            PROBLEM: A pixel is the smallest visual unit, so if we attempt to show
            a unit of time SMALLER than a pixel, it will be impossible to plot.
            => for all plots:
                 assert plot.total_sec >= sec_per_pixel
            If this fails, then the use must provide a larger "sec_per_pixel" value.

        :param debug:
        :param debug_ops:
        :param debug_memoize:
        :param entire_trace:
        :param kwargs:
        """
        self.directory = directory
        self.host = host
        self.user = user
        self.password = password
        self.step_sec = step_sec
        self.pixels_per_square = pixels_per_square
        self.dynamic_size = dynamic_size
        self.sec_per_pixel = self.step_sec / float(self.pixels_per_square)
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.debug_ops = debug_ops
        self.debug_memoize = debug_memoize
        self.entire_trace = entire_trace

        self.bar_width = 0.25
        self.show = False

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    def get_process_timeline_png(self, process_name, phase_name, bench_name, debug_ops):
        return _j(self.directory, "process_timeline{proc}{phase}{bench}{debug}.png".format(
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            bench=bench_suffix(bench_name),
            debug=debug_suffix(debug_ops),
        ))

    @staticmethod
    def get_plot_data_path(directory, bench_name, debug_ops):
        return _j(directory, "process_timeline.plot_data{bench}{debug}.txt".format(
            bench=bench_suffix(bench_name),
            debug=debug_suffix(debug_ops),
        ))

    def _plot_data_path(self, bench_name):
        return UtilizationPlot.get_plot_data_path(self.directory, bench_name, self.debug_ops)

    def run(self):
        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password, debug_ops=self.debug_ops)

        if self.entire_trace:
            # Plot a single plot for the entire trace across ALL processes.
            self.plot_process_phase()
            return

        process_names = self.sql_reader.process_names()

        for process_name in process_names:
            # phases = self.sql_reader.process_phase_start_end_times(process_name, debug=self.debug)
            phases = self.sql_reader.phases(process_name=process_name, debug=self.debug)
            for phase in phases:
                pprint.pprint({'process_name':process_name, 'phase': phase})
                # self.plot_process_phase(process_name, phase['phase_name'])
                self.plot_process_phase(process_name, phase)

    def plot_process_phase(self, process_name=None, phase_name=None):
        assert ( process_name is None and phase_name is None ) or \
               ( process_name is not None and phase_name is not None )

        # self.bench_names = self.sql_reader.bench_names(self.debug_ops) + [NO_BENCH_NAME]
        # assert len(self.bench_names) == len(unique(self.bench_names))
        # self.categories = self.sql_reader.categories

        overlap_computer = OverlapComputer(self.db_path,
                                           host=self.host, user=self.user, password=self.password,
                                           debug=self.debug,
                                           debug_single_thread=self.debug_single_thread,
                                           debug_ops=self.debug_ops)

        raise NotImplementedError("Haven't bother to maintain this (e.g. visible_overhead argument).")
        operation_overlap, metadata = overlap_computer.compute_process_timeline_overlap(
            process_name=process_name,
            phase_name=phase_name,
            debug_memoize=self.debug_memoize)
        assert len(operation_overlap) > 0

        # all_categories = set()
        # json_datas = []
        # for bench_name in self.bench_names:
        #     # json_data = overlap_computer.compute_per_operation_overlap(bench_name)
        #     json_datas.append(json_data)
        #
        #     for combo_and_times in json_data['category_combo_times']:
        #         category = _category_str(combo_and_times['category_combo'])
        #         all_categories.add(category)

        all_categories = set()
        for operation_key, combo_to_time in operation_overlap.items():
            for category_key, time_us in combo_to_time.items():
                category = _category_str(category_key)
                all_categories.add(category)

        pprint.pprint({'all_categories': all_categories})

        # def get_png(bench_name):
        #     return self.get_process_timeline_png(process_name, phase_name, bench_name, self.debug_ops)

        # self.category_order = sorted(all_categories)
        # self.bench_name_labels = DQN_BENCH_NAME_LABELS
        # TODO: create bench name labels
        # self.bench_name_labels = None
        # self.category_color_map = None
        # self.category_labels = None
        # self.impl_name_order = IMPL_NAME_ORDER
        # self.device_order = DEVICE_ORDER
        # self.plotter = StackedBarPlotter(
        #     get_png, self._plot_data_path,
        #     self.category_order,
        #     self.impl_name_order,
        #     self.device_order,
        #     bench_name_labels=self.bench_name_labels,
        #     category_color_map=self.category_color_map,
        #     category_labels=self.category_labels,
        #     bar_width=self.bar_width, show=self.show,
        #     json_reader_klass=ProcessTimelineReader,
        #     title='CPU/GPU utilization over training',
        #     # TODO: use "minigo"
        #     xlabel='',
        #     ylabel='Total training time (seconds)',
        #     yvalue_per_pixel=self.sec_per_pixel,
        #     dynamic_size=self.dynamic_size,
        # )

        # for bench_name, json_data in zip(self.bench_names, json_datas):
        #     device_name = NO_DEVICE_NAME
        #     impl_name = NO_IMPL_NAME
        #     self.plotter.add_json_data(json_data, bench_name,
        #                                device_name, impl_name, debug=True)

        for operation_key, combo_to_time in operation_overlap.items():
            json_data = combo_to_time
            # for category_key, time in operation_overlap.items():
            category = _category_str(category_key)
            all_categories.add(category)
            operations_name = _category_str(operation_key)

            device_name = NO_DEVICE_NAME
            impl_name = NO_IMPL_NAME
            self.plotter.add_json_data(json_data, operations_name,
                                       device_name, impl_name, debug=True,
                                       expect_times=True)

        # df = self.plotter.dataframe
        # assert len(df) != 0
        # self.plotter.plot(bench_name=None)

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    @staticmethod
    def get_stats(directory):
        return _j(directory, "process_timeline.stats.json")

    def _stats(self):
        return UtilizationPlot.get_stats(self.directory)

class PollingUtilParser:
    def __init__(self,
                 polling_util_json=None,
                 bins=None, start_time_us=None, end_time_us=None, polling_interval_us=None):

        def is_from_json():
            return polling_util_json is not None

        def is_from_bins():
            return bins is not None and \
                   start_time_us is not None and \
                   end_time_us is not None and \
                   polling_interval_us is not None

        configs = [is_from_json(), is_from_bins()]
        assert any(configs) and not all(configs)

        if is_from_bins():
            self._from_bins(bins, start_time_us, end_time_us, polling_interval_us)
        elif is_from_json():
            self._from_json(polling_util_json)
        else:
            raise NotImplementedError()

    def _from_json(self, polling_util_json):
        self.polling_util_json_path = polling_util_json
        self.polling_util_json = load_json(self.polling_util_json_path)
        self.bins = np.array(self.polling_util_json['bins'], dtype=np.bool)
        self.start_time_us = self.polling_util_json['metadata']['start_time_us']
        self.end_time_us = self.polling_util_json['metadata']['end_time_us']
        self.polling_interval_us = self.polling_util_json['metadata']['polling_interval_us']
        # assert len(self.bins) == math.ceil((self.end_time_us - self.start_time_us)/self.polling_interval_us)
        assert len(self.bins) == self._get_num_bins(self.start_time_us, self.end_time_us)
        return self

    def _from_bins(self, bins, start_time_us, end_time_us, polling_interval_us):
        self.bins = np.array(bins, dtype=np.bool)
        self.start_time_us = start_time_us
        self.end_time_us = end_time_us
        self.polling_interval_us = polling_interval_us
        # assert len(self.bins) == math.ceil((self.end_time_us - self.start_time_us)/self.polling_interval_us)
        assert len(self.bins) == self._get_num_bins(self.start_time_us, self.end_time_us)
        # math.floor(self.end_time_us/self.polling_interval_us) - math.floor(self.start_time_us/self.polling_interval_us) + 1
        return self

    def _get_num_bins(self, start_us, end_us):
        # wrong = math.ceil((end_us - start_us)/self.polling_interval_us)
        right = math.floor(end_us/self.polling_interval_us) - math.floor(start_us/self.polling_interval_us) + 1
        # assert right == wrong
        return right

    def extend(self, start_time_us, end_time_us):
        # NOTE: need to extend "bins" with "false" to cover GPU polling intervals that we don't have readings for...
        # this should be a SMALL number of intervals since sampler should run for entirety of the run.
        #
        # [ train.start ][ sample.start ][ sample.end ][ train.end ]
        # A             B                C            D            E
        #
        # train.start.i = math.floor(train.start/poll)*poll
        # sample.start.i = math.floor(sample.start/poll)*poll
        # sample.end.i = math.ceil(sample.end/poll)*poll
        # train.end.i = math.ceil(train.end/poll)*poll
        #
        # assert len(self.bins) == (sample.end.i - sample.start.i)
        # add_bins_before = sample.start.i - train.start.i
        # add_bins_after = train.end.i - start.end.i

        poll = self.polling_interval_us
        train_start = start_time_us
        train_end = end_time_us
        sample_start = self.start_time_us
        sample_end = self.end_time_us

        assert train_start <= sample_start <= sample_end <= train_end

        train_start_i = math.floor(train_start/poll)
        sample_start_i = math.floor(sample_start/poll)
        # sample_end_i = math.ceil(sample_end/poll)
        # train_end_i = math.ceil(train_end/poll)
        sample_end_i = math.floor(sample_end/poll)
        train_end_i = math.floor(train_end/poll)

        # assert (sample_start_i - train_start_i) == get_num_bins(train_start, sample_start)
        # assert (train_end_i - sample_end_i) == get_num_bins(sample_end, train_end)
        # assert (sample_end_i - sample_start_i) == get_num_bins(sample_start, sample_end)

        # [                                    ]   [                          ]   [                                ]
        # train_start_i ... (sample_start_i - 1)   sample_start_i  sample_end_i   (sample_end_i + 1) ... train_end_i
        # add_bins_before = (sample_start_i - 1) - train_start_i + 1
        # add_bins_after = train_end_i - (sample_end_i + 1) + 1
        # in_between = sample_start_i - sample_end_i + 1

        add_bins_before = (sample_start_i - 1) - train_start_i + 1
        add_bins_after = train_end_i - (sample_end_i + 1) + 1
        assert (sample_end_i - sample_start_i + 1) == len(self.bins)
        assert add_bins_before + add_bins_after + len(self.bins) == self._get_num_bins(train_start, train_end)

        assert add_bins_before >= 0
        assert add_bins_after >= 0

        bins = [False]*add_bins_before + list(self.bins) + [False]*add_bins_after

        extend = PollingUtilParser(
            bins=bins,
            start_time_us=min(self.start_time_us, start_time_us),
            end_time_us=max(end_time_us, self.end_time_us),
            polling_interval_us=self.polling_interval_us)

        return extend

    def get_util(self, time_us, window_size_us):
        i = (time_us - self.start_time_us) // self.polling_interval_us
        assert i >= 0
        assert i < len(self.bins)
        assert window_size_us % self.polling_interval_us == 0
        n_bins = int(window_size_us // self.polling_interval_us)
        start_i = max(0, i - n_bins)
        end_i = i
        assert start_i <= end_i
        # Q: Should we include the CURRENT bin...or just previous bins...?
        # Well, we haven't seen the current bin finish yet... so probably PREVIOUS bins.
        bin_samples = self.bins[start_i:end_i]
        if len(bin_samples) == 0:
            util = 0.
        else:
            util = np.mean(bin_samples)
        return util

class SlidingWindowUtilizationPlot:
    """
    PSEUDOCODE:

    class PollingUtilParser:
        def extend(self, polling_util, training.start_time_us, training.end_time_us):
            # NOTE: need to extend "bins" with "false" to cover GPU polling intervals that we don't have readings for...
            # this should be a SMALL number of intervals since sampler should run for entirety of the run.
            # [ train.start ][ sample.start ][ sample.end ][ train.end ]
            add_bins_before = max(
                0,
                (polling_util.start_time_us - training.start_time_us) // polling_interval_us
            )
            add_bins_after = max(
                0,
                (training.end_time_us - polling_util.end_time_us) // polling_interval_us,
            )
            bins = [False]*add_bins_before + polling_util.bins + [False]*add_bins_after

            extend = PollingUtilParser.from_bins(
                bins=bins,
                start_time_us=min(polling_util.start_time_us, training.start_time_us),
                end_time_us=max(training.end_time_us, polling_util.end_time_us))

            return extend

        def get_util(self, time_us):
            i = (time_us - self.start_time_usec) // polling_interval_us
            start_i = max(0, i)
            end_i = min(i, len(bins))
            util = np.mean(bins[start_i:end_i])
            return util

    def read_dataframe(self):
        # NOTE: optionally skip this step is a csv file is provided via cmdline (makes debugging plot faster)

        gpu_util_df = read machine_util*.proto files, keeping only CUDA_VISIBLE_DEVICES utilization readings
        orig_polling_util = read json file into PollingUtilParser
        training.start_time_us = min(gpu_util_df['time_us'])
        training.end_time_us = max(gpu_util_df['time_us'])
        polling_util = orig_polling_util.extend(training.start_time_us, training.end_time_us)
        # Match the same number of samples as collected using nvidia-smi
        sliding_window_util = np.array(len(gpu_util_df))
        assert sliding_window_us % polling_interval_us == 0
        n_bins = sliding_window_us / polling_interval_us

        sliding_window_utils = np.array(polling_util.get_util(time_us) \
          for time_us, smi_util in gpu_util_df)

        # convert sliding_window_utils into gpu_util_df dataframe format.
        #   algo, env, line_label, device_name, util, time_us, time_sec
        # line_label is one of:
        #   - nvidia-smi
        #   - Sliding window (polling interval = 1ms, window size = 1000ms)
        # NOTE: make time_sec 0-based from the beginning of training;
        # start of training is min(time_us) before training starts.
        df = ...

        Output csv @ "SlidingWindowUtilization.csv"
        return df

    def plot(self, df):
        # Create line-plot for each gpu utilization
        ...
    """
    def __init__(self, directory, polling_util_json, window_size_us, debug=False, **kwargs):
        self.directory = directory
        self.polling_util_json_path = polling_util_json
        self.window_size_us = window_size_us
        self.debug = debug

    # def maybe_add_algo_env(self, machine_util_path):
    #     assert is_machine_util_file(machine_util_path)
    #
    #     rlscope_directory = _d(machine_util_path)
    #
    #     if self.algo_env_from_dir:
    #         return self.add_algo_env_from_dir(machine_util_path)
    #     if not _e(experiment.experiment_config_path(rlscope_directory)):
    #         return self.add_experiment_config(machine_util_path)
    #
    #     # Not sure what (algo, env) is; don't add those columns.
    #     return None

    def read_nvidia_smi(self):
        gpu_util_df_reader = UtilDataframeReader(
            self.directory,
            # Q: use --algo and --env?  Or just don't add them at all since its a per rlscope-directory plot anyways.
            # add_fields=self.maybe_add_algo_env,
            debug=self.debug)
        gpu_util_df = gpu_util_df_reader.read()

        gpu_util_df = gpu_util_df[
            (gpu_util_df['used_by_tensorflow']) &
            (gpu_util_df['device_type'] == 'GPU')]

        return gpu_util_df

    def read_dataframe(self):
        gpu_util_df = self.read_nvidia_smi()
        assert len(gpu_util_df['device_id'].unique()) == 1
        assert len(gpu_util_df['device_name'].unique()) == 1
        device_name = list(gpu_util_df['device_name'].unique())[0]
        device_id = list(gpu_util_df['device_id'].unique())[0]
        start_time_us = min(gpu_util_df['start_time_us'])
        end_time_us = max(gpu_util_df['start_time_us'])

        orig_polling_util = PollingUtilParser(self.polling_util_json_path)
        polling_util = orig_polling_util.extend(start_time_us, end_time_us)
        n_smi_samples = len(gpu_util_df['start_time_us'])
        # assert len(polling_util.bins) == n_smi_samples
        sld_start_time_us = gpu_util_df['start_time_us']
        sld_util = np.zeros(n_smi_samples, dtype=np.float64)
        for i, time_us in enumerate(gpu_util_df['start_time_us']):
            sld_util[i] = polling_util.get_util(time_us, self.window_size_us)

        sld_df = pd.DataFrame(data={
            'util': sld_util,
            'start_time_us': sld_start_time_us,
        })

        # convert sliding_window_utils into gpu_util_df dataframe format.
        #   line_label, device_name, util, time_us, time_sec
        # line_label is one of:
        #   - nvidia-smi
        #   - Sliding window (polling interval = 1ms, window size = 1000ms)
        # NOTE: make time_sec 0-based from the beginning of training;
        # start of training is min(time_us) before training starts.
        sld_df['line_label'] = "Sliding window (poll interval = {poll_ms} ms, window size = {window_ms} ms)".format(
            poll_ms=polling_util.polling_interval_us / constants.MICROSECONDS_IN_MS,
            window_ms=self.window_size_us / constants.MICROSECONDS_IN_MS,
        )
        sld_df['device_name'] = device_name
        sld_df['device_id'] = device_id

        smi_df = copy.copy(gpu_util_df[['util', 'start_time_us', 'device_id', 'device_name']])
        smi_df['line_label'] = 'nvidia-smi'

        df = pd.concat([smi_df, sld_df], sort=True)
        # Nice to have utilization readings next to each other when manually inspecting csv file.
        df = df.sort_values(by=['start_time_us', 'line_label'])

        df['time_sec'] = ( df['start_time_us'] - start_time_us ) / constants.MICROSECONDS_IN_SECOND

        return df, polling_util.polling_interval_us

    def plot(self, df):
        df = copy.copy(df)
        # figsize = None
        # fig = plt.figure(figsize=figsize)
        fig, ax = plt.subplots()

        df['util_percent'] = df['util'] * 100.
        sns.lineplot(
            x="time_sec", y="util_percent", hue="line_label",
            data=df,
            ax=ax)

        fix_seaborn_legend(ax, legend_kwargs=dict(
            # Position legend in center, above plotting area.
            # bbox position is relative to the bottom-centre of the legend.
            loc='lower center',
            bbox_to_anchor=(
                0.5,
                1,
            ),
        ))

        ax.set_ylabel(r"GPU utilization (%)")
        ax.set_xlabel(Y_LABEL_TRAINING_TIME_SEC)
        fig.savefig(self._plot_path, bbox_inches="tight", pad_inches=0)
        logger.info('Save figure to {path}'.format(path=self._plot_path))
        plt.close(fig)

    def _path_with_ext(self, ext):
        return _j(self.directory, "SlidingWindowUtilizationPlot.polling_interval_us_{poll}_us.window_size_us_{wind}_us.{ext}".format(
            poll=self.polling_interval_us,
            wind=self.window_size_us,
            ext=ext,
        ))

    @property
    def _csv_path(self):
        return self._path_with_ext('csv')

    @property
    def _plot_path(self):
        return self._path_with_ext('pdf')

    def run(self):
        self.df, self.polling_interval_us = self.read_dataframe()
        self.df.to_csv(self._csv_path, index=False)
        logger.info("Output SlidingWindowUtilizationPlot data @ {path}".format(path=self._csv_path))
        self.plot(self.df)


class UtilizationPlot:
    """
    CPU/GPU utilization over training.
    """
    DEBUG_MINIGO = False
    # DEBUG_MINIGO = True
    RESOURCE_TYPES = [constants.CATEGORY_CPU, constants.CATEGORY_GPU]
    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 visible_overhead=False,
                 event_overlap_path=None,
                 n_workers=1,
                 events_per_split=10000,
                 step_sec=1.,
                 pixels_per_square=10,
                 dynamic_size=False,
                 debug=False,
                 debug_single_thread=False,
                 debug_perf=False,
                 debug_ops=False,
                 debug_memoize=False,
                 entire_trace=False,
                 overlap_type=None,
                 # If overlap_type = OperationOverlap, then dump this resource-type.
                 operation_overlap_resource=None,
                 # CategoryOverlap
                 machine_name=None,
                 process_name=None,
                 phase_name=None,
                 resource_type=None,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directory:

        :param step_sec:
        :param pixels_per_square:
            Condensed:
            e.g.
            sec_per_pixel = 0.1 sec / pixel
            sec_per_pixel = step_sec / pixels_per_square
            PROBLEM: A pixel is the smallest visual unit, so if we attempt to show
            a unit of time SMALLER than a pixel, it will be impossible to plot.
            => for all plots:
                 assert plot.total_sec >= sec_per_pixel
            If this fails, then the use must provide a larger "sec_per_pixel" value.

        :param debug:
        :param debug_ops:
        :param debug_memoize:
        :param entire_trace:
        :param kwargs:
        """
        if operation_overlap_resource is None:
            operation_overlap_resource = [constants.CATEGORY_CPU]
        self.overlap_type = overlap_type
        self.machine_name = machine_name
        self.process_name = process_name
        self.event_overlap_path = event_overlap_path
        self.phase_name = phase_name
        self.resource_type = resource_type
        self.operation_overlap_resource = frozenset(operation_overlap_resource)
        self.directory = directory
        self.host = host
        self.user = user
        self.password = password
        self.step_sec = step_sec
        self.pixels_per_square = pixels_per_square
        self.dynamic_size = dynamic_size
        self.sec_per_pixel = self.step_sec / float(self.pixels_per_square)
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.debug_perf = debug_perf
        self.visible_overhead = visible_overhead
        if self.debug_single_thread:
            self.n_workers = 1
        else:
            self.n_workers = n_workers
        self.events_per_split = events_per_split
        self.debug_ops = debug_ops
        self.debug_memoize = debug_memoize
        if self.debug_memoize:
            logger.info("debug_memoize = {b}".format(b=self.debug_memoize))
        self.entire_trace = entire_trace

        self.bar_width = 0.25
        self.show = False

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    def get_process_timeline_png(self, machine_name, process_name, phase_name, bench_name, debug_ops):
        return _j(self.directory, "process_timeline{mach}{proc}{phase}{bench}{debug}.png".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            bench=bench_suffix(bench_name),
            debug=debug_suffix(debug_ops),
        ))

    def _resource_overlap_json(self, machine_name, process_name, phase_name):
        return _j(self.directory, "ResourceOverlap{mach}{proc}{phase}.json".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def _resource_overlap_venn_js_json(self, machine_name, process_name, phase_name):
        return _j(self.directory, "ResourceOverlap{mach}{proc}{phase}.venn_js.json".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def _resource_overlap_subplot_json(self, machine_name, process_name, phase_name):
        return _j(self.directory, "ResourceOverlapSubplot{mach}{proc}{phase}.json".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def _resource_overlap_subplot_venn_js_json(self, machine_name, process_name, phase_name):
        return _j(self.directory, "ResourceOverlapSubplot{mach}{proc}{phase}.venn_js.json".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def _operation_overlap_json(self, machine_name, process_name, phase_name, resources):
        return _j(self.directory, "OperationOverlap{mach}{proc}{phase}{resources}.json".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            resources=resources_suffix(resources),
        ))

    def _operation_overlap_venn_js_json(self, machine_name, process_name, phase_name, resources):
        return _j(self.directory, "OperationOverlap{mach}{proc}{phase}.venn_js.json".format(
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            resources=resources_suffix(resources),
        ))

    @staticmethod
    def get_plot_data_path(directory, bench_name, debug_ops):
        return _j(directory, "process_timeline.plot_data{bench}{debug}.txt".format(
            bench=bench_suffix(bench_name),
            debug=debug_suffix(debug_ops),
        ))

    def _plot_data_path(self, bench_name):
        return UtilizationPlot.get_plot_data_path(self.directory, bench_name, self.debug_ops)

    def run(self):

        self.timer = SimpleTimer("EventOverlap")
        self.timer.reset_start_time()

        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password, debug_ops=self.debug_ops)

        if self.entire_trace:
            # Plot a single plot for the entire trace across ALL processes.
            self.plot_process_phase()
            return


        # HACK: quickly debug OverlapType for minigo

        # debug_process_names = ['loop_train_eval']
        # debug_phases = ['sgd_updates']

        # Debug OperationOverlap having overlap of operations within the same process:
        # debug_process_names = ['loop_selfplay']
        # debug_phases = ['default_phase']

        # debug_process_names = ['loop_train_eval']
        # debug_phases = ['sgd_updates']

        # debug_process_names = ['selfplay_worker_1']
        # debug_phases = ['selfplay_worker_1']
        # debug_process_names = ['selfplay_worker_0_generation_0']

        # if not UtilizationPlot.DEBUG_MINIGO:
        #     process_names = self.sql_reader.process_names()
        # else:
        #     process_names = debug_process_names

        machine_names = self.sql_reader.machine_names
        if self.debug:
            logger.info(pprint_msg({
                'machine_names':machine_names}))

        for machine_name in machine_names:
            process_names = self.sql_reader.process_names(machine_name, debug=self.debug)
            if self.debug:
                logger.info(pprint_msg({
                    'machine_name':machine_name,
                    'process_names':process_names}))

            # for process_name in debug_process_names:
            for process_name in process_names:

                # phase_names = self.sql_reader.phases(machine_name, process_name, debug=self.debug)
                # phases = [self.sql_reader.process_phase(process_name, phase_name, debug=self.debug)
                #           for phase_name in phase_names]
                phases = self.sql_reader.phases(machine_name, process_name, debug=self.debug)
                if self.debug:
                    logger.info(pprint_msg({
                        'machine_name':machine_name,
                        'process_name':process_name,
                        'phases':phases}))
                # if UtilizationPlot.DEBUG_MINIGO:
                #     phases = [phase for phase in phases if phase.phase_name in debug_phases]

                for phase in phases:
                    # self.plot_process_phase(process_name, phase['phase_name'])
                    self.plot_process_phase(machine_name, process_name, phase=phase)

        self.sql_reader.close()

    def plot_process_phase(self, machine_name=None, process_name=None, phase_name=None, phase=None):
        if phase is not None and phase_name is None:
            phase_name = phase.phase_name

        assert ( process_name is None and phase_name is None ) or \
               ( process_name is not None and phase_name is not None )

        overlap_obj = overlap_type_to_instance(
            self.overlap_type,
            debug=self.debug,
        )

        overlap_computer = OverlapComputer(self.db_path,
                                           host=self.host, user=self.user, password=self.password,
                                           timer=self.timer,
                                           debug=self.debug,
                                           debug_single_thread=self.debug_single_thread,
                                           debug_perf=self.debug_perf,
                                           debug_ops=self.debug_ops)

        def compute_overlap():
            memoize = True
            assert machine_name is not None
            assert process_name is not None
            assert phase_name is not None
            memoize_path = _j(
                self.directory,
                "EventOverlap",
                "machine", machine_name,
                "process", process_name,
                "phase", phase_name,
                "EventOverlap.{numba_suffix}.compute_overlap.pickle".format(
                # klass=self.__class__.__name__,
                numba_suffix='.numba' if py_config.RLSCOPE_USE_NUMBA else '',
                # overlap_type=self.overlap_type,
            ))

            if should_load_memo(memoize, memoize_path):
                ret = load_memo(memoize, memoize_path)
                return ret

            overlap, overlap_metadata = overlap_computer.compute_process_timeline_overlap(
                # overlap_obj.pre_reduce,
                visible_overhead=self.visible_overhead,
                machine_name=machine_name,
                process_name=process_name,
                phase_name=phase_name,
                n_workers=self.n_workers,
                events_per_split=self.events_per_split,
                debug_memoize=memoize,
                # overlap_type=self.overlap_type,
            )

            ret = (overlap, overlap_metadata)
            maybe_memoize(memoize, ret, memoize_path)

            return ret

        overlap, overlap_metadata = compute_overlap()
        logger.info("Event overlap results: {msg}".format(msg=pprint_msg({
            'overlap_type': self.overlap_type,
            'RLSCOPE_USE_NUMBA': py_config.RLSCOPE_USE_NUMBA,
            'overlap': overlap,
            'overlap_metadata': overlap_metadata,
        })))

        new_overlap = overlap
        # assert len(new_overlap) > 0

        new_overlap, new_overlap_metadata = overlap_obj.post_reduce(new_overlap, overlap_metadata, self.visible_overhead)
        # assert len(new_overlap) > 0

        overlap_obj.dump_json_files(
            self.sql_reader,
            new_overlap,
            new_overlap_metadata,
            self.directory,
            machine_name=machine_name,
            process_name=process_name,
            phase_name=phase_name)

        if self.overlap_type == 'default':
            operation_overlap = overlap_obj.as_js_dict(new_overlap)
            # assert len(operation_overlap) > 0
            self._do_plot_process_phase(operation_overlap, machine_name, process_name, phase_name)

    def _do_plot_process_phase(self, operation_overlap, machine_name=None, process_name=None, phase_name=None):
        assert ( process_name is None and phase_name is None ) or \
               ( process_name is not None and phase_name is not None )

        assert self.overlap_type == 'default'

        all_categories = set()
        for operation_key, combo_to_time in operation_overlap.items():
            for category_key, time_us in combo_to_time.items():
                category = _category_str(category_key)
                all_categories.add(category)

        pprint.pprint({'all_categories': all_categories})

        def get_png(bench_name):
            return self.get_process_timeline_png(machine_name, process_name, phase_name, bench_name, self.debug_ops)

        self.category_order = sorted(all_categories)
        # self.bench_name_labels = DQN_BENCH_NAME_LABELS
        # TODO: create bench name labels
        self.bench_name_labels = None
        self.category_color_map = None
        self.category_labels = None
        self.impl_name_order = IMPL_NAME_ORDER
        self.device_order = DEVICE_ORDER
        self.plotter = StackedBarPlotter(
            get_png, self._plot_data_path,
            self.category_order,
            self.impl_name_order,
            self.device_order,
            bench_name_labels=self.bench_name_labels,
            category_color_map=self.category_color_map,
            category_labels=self.category_labels,
            bar_width=self.bar_width, show=self.show,
            json_reader_klass=ProcessTimelineReader,
            title='CPU/GPU utilization over training',
            # TODO: use "minigo"
            xlabel='',
            ylabel='Total training time (seconds)',
            yvalue_per_pixel=self.sec_per_pixel,
            dynamic_size=self.dynamic_size,
        )

        # for bench_name, json_data in zip(self.bench_names, json_datas):
        #     device_name = NO_DEVICE_NAME
        #     impl_name = NO_IMPL_NAME
        #     self.plotter.add_json_data(json_data, bench_name,
        #                                device_name, impl_name, debug=True)

        for operation_key, combo_to_time in operation_overlap.items():
            json_data = combo_to_time
            # for category_key, time in operation_overlap.items():
            category = _category_str(category_key)
            all_categories.add(category)
            operations_name = _category_str(operation_key)

            device_name = NO_DEVICE_NAME
            impl_name = NO_IMPL_NAME
            self.plotter.add_json_data(json_data, operations_name,
                                       device_name, impl_name, debug=True,
                                       expect_times=True)

        # for bench_name in [NO_BENCH_NAME]:
        # for bench_name in self.bench_names:
        #     self.plotter.plot(bench_name)
        df = self.plotter.dataframe
        # assert len(df) != 0
        self.plotter.plot(bench_name=None)

        # Plot the "CPU/GPU Utilization" plot.
        # Other overlap_type's will JUST output the overlap data (to be consumed by rlscope-drill).

    def _dump_process_timeline_json(self, operation_overlap):
        path = self._process_timeline_json_path()
        logger.info("> DEBUG: dump process timeline compute overlap @ {path}".format(path=path))

        # PROBLEM: overlap JSON file is usually for a single operation.
        # However, now we have multiple operations for a given overlap calculation.
        # NOTE: the only reason we have a JSON-specific format us because
        # JSON doesn't allow a "set" as a dictionary key.
        #
        # Conversion to JSON:
        # A dict whose keys are frozenset's should be converted to a list of key/value pairs:
        # [
        #   (key[0], value[0]),
        #   ...,
        # ]
        js = js_friendly(operation_overlap)
        do_dump_json(js, path, cls=DecimalEncoder)

    def _process_timeline_json_path(self):
        path = _j(self.directory, 'process_timeline.json')
        return path

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    @staticmethod
    def get_stats(directory):
        return _j(directory, "process_timeline.stats.json")

    def _stats(self):
        return UtilizationPlot.get_stats(self.directory)

def _add_cpu_gpu_stats(js_stats, plotter, bench_name=NO_BENCH_NAME):
    bench_df = plotter.plot_data(bench_name)
    def is_gpu_row(row):
        return re.search(r'\bGPU\b', row['category'])
    gpu_rows = pd.DataFrame([row for index, row in bench_df.iterrows() if is_gpu_row(row)])
    cpu_rows = pd.DataFrame([row for index, row in bench_df.iterrows() if not is_gpu_row(row)])
    def sum_time(rows):
        if len(rows) > 0:
            return rows['mean'].sum()
        return 0.
    cpu_time_sec = sum_time(cpu_rows)
    gpu_time_sec = sum_time(gpu_rows)
    total_time_sec = cpu_time_sec + gpu_time_sec
    def safe_div(numer, denom):
        if denom == 0:
            return None
        return numer/denom
    def safe_mul(a, b):
        if a is None or b is None:
            return None
        return a*b
    stats = {
        'cpu_time_sec':cpu_time_sec,
        'gpu_time_sec':gpu_time_sec,
        'total_time_sec':total_time_sec,
        'gpu_time_percent': safe_mul(100, safe_div(gpu_time_sec, total_time_sec)),
        'cpu_time_percent': safe_mul(100, safe_div(cpu_time_sec, total_time_sec)),
    }
    update_dict(js_stats, stats)


class IDSet:
    def __init__(self):
        self._next_id = 0
        self.item_to_id = dict()
        self.id_to_item = dict()

    def add(self, item):
        if item in self.item_to_id:
            pass
        else:
            ident = self._next_id
            self._next_id += 1
            self.item_to_id[item] = ident
            self.id_to_item[ident] = item

    def remove(self, item):
        ident = self.item_to_id[item]
        del self.item_to_id[item]
        del self.id_to_item[ident]

    def get_id(self, item):
        return self.item_to_id[item]

    def __len__(self):
        return len(self.item_to_id)

# def device_name_to_

class ConvertResourceOverlapToResourceSubplot:
    def __init__(self, directory,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.debug = debug

    def _plot_data_path(self, device_id, device_name):
        return _j(self.directory, "util_scale.plot_data{dev}.txt".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def _json_path(self, device_id, device_name):
        return _j(self.directory, "util_scale{dev}.js_path.json".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def convert_ResourceOverlap_to_ResourceSubplot(self, venn_path):
        """
        E.g. Input - ResourceOverlap:
        {
            "metadata": {
                "end_time_usec": 1572305950361810,
                "machine": "eco-15",
                "overlap_type": "ResourceOverlap",
                "phase": "selfplay_worker_9_generation_1",
                "process": "selfplay_worker_9_generation_1",
                "start_time_usec": 1572300331843396
            },
            "venn": [
                {
                    "label": "CPU",
                    "sets": [
                        0
                    ],
                    "size": 4951987256
                },
                {
                    "label": "GPU",
                    "sets": [
                        1
                    ],
                    "size": 22915662
                },
                {
                    "sets": [
                        0,
                        1
                    ],
                    "size": 19313474
                }
            ]
        }
        E.g. Output - ResourceSubplot:
        {
            "metadata": {
                "end_time_usec": 1572305950361810,
                "machine": "eco-15",
                "overlap_type": "ResourceSubplot",
                "phase": "selfplay_worker_9_generation_1",
                "process": "selfplay_worker_9_generation_1",
                "start_time_usec": 1572300331843396
            },
            "venn": [
                {
                    "label": "CPU",
                    "sets": [
                        0
                    ],
                    "size": 4951987256
                },
                {
                    "label": "GPU",
                    "sets": [
                        1
                    ],
                    "size": 22915662
                },
                {
                    "label": "Total",
                    "sets": [
                        2
                    ],
                    # "size": [CPU] + [GPU] - [CPU, GPU]
                    "size": 4951987256 + 22915662 - 19313474
                },
            ]
        }

        :return:
        """
        total_region = ('Total',)
        resource_overlap_venn_js = VennData(venn_path)
        resource_overlap = resource_overlap_venn_js.as_dict()
        resource_subplot = dict()
        resource_subplot[total_region] = 0
        for region, region_size in resource_overlap.items():
            assert type(region) in {tuple}
            if len(region) == 1:
                resource_subplot[region] = region_size
                resource_subplot[total_region] += region_size
            else:
                # Subtract overlap to remove double counting
                # from adding up the individual regions.
                resource_subplot[total_region] -= region_size
        md = resource_overlap_venn_js.metadata()
        md['overlap_type'] = 'ResourceSubplot'
        converter = OverlapJSONToVennConverter(
            overlap=resource_subplot,
            metadata=md)
        direc = _d(venn_path)
        base = _b(venn_path)
        new_base = re.sub(r'ResourceOverlap', 'ResourceSubplot', base)
        resource_subplot_path = _j(direc, new_base)
        assert resource_subplot_path != venn_path
        logger.info(textwrap.dedent("""\
        Convert:
          From: {src}
          To:   {dst}
        """).format(
            src=venn_path,
            dst=resource_subplot_path,
        ))
        converter.dump(resource_subplot_path)

    def run(self):
        venn_paths = []
        resource_overlap_paths = []
        for path in each_file_recursive(self.directory):
            if not is_venn_js_file(path):
                continue
            venn_paths.append(path)
            venn_js = VennData(path)
            if venn_js.md['overlap_type'] == 'ResourceOverlap':
                resource_overlap_paths.append(path)

        for venn_path in resource_overlap_paths:
            self.convert_ResourceOverlap_to_ResourceSubplot(venn_path)

class VennJsPlotter:
    def __init__(self,
                 # directory,
                 venn_js,
                 algo=None,
                 env=None,
                 width=None,
                 height=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        # TODO: make it so we can plot multiple venn_js paths.
        # self.directory = directory
        self.venn_js = venn_js
        self.algo = algo
        self.env = env
        self.width = width
        self.height = height
        self.debug = debug

    @staticmethod
    def _plot_path(venn_js_path, file_ext):
        plot_path = venn_js_path
        plot_path = re.sub(r'\.venn_js\.json$', file_ext, plot_path)
        assert plot_path != venn_js_path
        return plot_path

    @staticmethod
    def plot_paths(venn_js_path):
        file_exts = ['.pdf', '.svg']
        return [VennJsPlotter._plot_path(venn_js_path, file_ext)
                for file_ext in file_exts]

    def plot_venn_js(self, venn_path):
        """
        E.g. input
        {
            "metadata": {
                "end_time_usec": 1572305950361810,
                "machine": "eco-15",
                "overlap_type": "ResourceOverlap",
                "phase": "selfplay_worker_9_generation_1",
                "process": "selfplay_worker_9_generation_1",
                "start_time_usec": 1572300331843396
            },
            "venn": [
                {
                    "label": "CPU",
                    "sets": [
                        0
                    ],
                    "size": 4951987256
                },
                {
                    "label": "GPU",
                    "sets": [
                        1
                    ],
                    "size": 22915662
                },
                {
                    "sets": [
                        0,
                        1
                    ],
                    "size": 19313474
                }
            ]
        }
        """
        rlscope_dir = _d(venn_path)
        rlscope_metadata = read_rlscope_config_metadata(rlscope_dir)
        venn_data = VennData(venn_path)
        overlap = venn_data.as_overlap_dict()
        def region_tuple_as_label(region_tuple):
            return ' + '.join(sorted(region_tuple))
        col_data = {
           'time_us': [],
           'region_tuple': [],
            'label': [],
            'x_group': [],
            'algo': [],
            'env': [],
            'x_field': [],
        }
        X_GROUP_ALL = 0

        def get_algo_env(opt):
            if getattr(self, opt) is not None:
                value = getattr(self, opt)
            elif opt in rlscope_metadata['metadata']:
                value = rlscope_metadata['metadata'][opt]
            else:
                value = ''
            return value

        for region_tuple, time_us in overlap.items():
            col_data['time_us'].append(time_us)
            col_data['region_tuple'].append(region_tuple)
            col_data['label'].append(region_tuple_as_label(region_tuple))
            col_data['x_group'].append(X_GROUP_ALL)
            algo = get_algo_env('algo')
            col_data['algo'].append(algo)

            env = get_algo_env('env')
            col_data['env'].append(env)

            x_field = "({algo}, {env})".format(
                algo=algo,
                env=env)
            col_data['x_field'].append(x_field)

        df = pd.DataFrame(col_data)
        df['time_sec'] = df['time_us'] / constants.MICROSECONDS_IN_SECOND

        # Total up the time of each region
        total_time_sec = df['time_sec'].sum()
        df['percent'] = df['time_sec']/total_time_sec

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None

        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        fig = plt.figure(figsize=figsize)
        # plot_data['field'] = "Per-API-call interception overhead"
        ax = fig.add_subplot(111)
        # sns.barplot(x='x_field', y='training_duration_sec', hue='pretty_config', data=training_duration_plot_data, ax=ax)
        # ['algo',
        #  'env',
        #  'x_field',
        #  'duration_name',
        #  'duration_pretty_name',
        #  'duration_name_order',
        #  'config',
        #  'config_pretty_name',
        #  'duration_sec',
        #  'x_group',
        #  'bar_label',
        #  'percent_wrong']
        xgroup_barplot = add_grouped_stacked_bars(
            x='x_field',
            x_group='x_group',
            y='time_sec',
            hue='label',
            label='label',
            label_order='label',
            # bar_label='bar_label',
            # bar_label_x_offset=3,
            # bar_label_kwargs=dict(
            #     fontsize=9
            # ),
            data=df,
            ax=ax,
            # rotation=10,
            # rotation=None,
            debug=self.debug,
            **plt_kwargs)
        # for xgroup, barplot in xgroup_barplot.items():
        #     if xgroup == XGROUP_UNINS:
        #         add_ax_bar_labels(ax, barplot)
        # ax.legend().set_title(None)
        ax.set_ylabel('Total training time (sec)')
        ax.set_xlabel('(RL algorithm, Simulator)')
        # ax.set_title("Breakdown of profiling overhead")
        fig.tight_layout()

        for plot_path in self.plot_paths(venn_path):
            # plot_path = self.plot_path(venn_path)
            logger.info("Plot venn_js @ {path}".format(
                path=plot_path,
            ))
            fig.savefig(plot_path)
        plt.close(fig)

    def run(self):
        self.plot_venn_js(self.venn_js)

class HeatScalePlot:
    """
    HeatScale/colormap of overall device (CPU/GPU) utilization.
    """
    def __init__(self, directory,
                 # host=None,
                 # user=None,
                 # password=None,
                 debug=False,
                 step_sec=1.,
                 pixels_per_square=10,
                 decay=0.99,
                 # Swallow any excess arguments
                 **kwargs):
        """

        :param directory:
        :param debug:

        :param step_sec:
        :param pixels_per_square:
            `pixels_per_square` determines how many pixels are taken up by "step" units.
            `step_sec` determines how many seconds each utilization square represents.
            - step_sec should be bigger than the sampling frequency (500 ms)
            Q: Any reason we want to specify step and pixels_per_square separately?
            A: Yes I believe so:
            - Only increases pixels_per_square if we'd like the same "exponential average window", but just a larger generated image.
            - Only decrease/increase step_sec if we'd like to see a more fine/coarse grained "exponential average window"

        :param decay:
        :param kwargs:
        """
        self.directory = directory
        # self.host = host
        # self.user = user
        # self.password = password
        self.debug = debug

        self.step_sec = step_sec
        self.pixels_per_square = pixels_per_square
        self.decay = decay

        # self.bar_width = 0.25
        self.show = False

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    def get_util_scale_png(self, device_id, device_name):
        return _j(self.directory, "util_scale{dev}.png".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def _plot_data_path(self, device_id, device_name):
        return _j(self.directory, "util_scale.plot_data{dev}.txt".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def _json_path(self, device_id, device_name):
        return _j(self.directory, "util_scale{dev}.js_path.json".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def run(self):

        util_df_reader = UtilDataframeReader(
            self.directory,
            # add_fields=self.maybe_add_algo_env,
            debug=self.debug)
        util_df = util_df_reader.read()
        util_df = util_df.sort_values(['machine_name', 'device_name', 'start_time_us'])
        util_df['start_time_sec'] = util_df['start_time_us']/constants.MICROSECONDS_IN_SECOND

        overlap_df_reader = OverlapDataframeReader(
            self.directory,
            # add_fields=self.maybe_add_algo_env,
            debug=self.debug)
        overlap_df = overlap_df_reader.read()

        # self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)
        # with self.sql_reader:

        # for device_name, device_id in sql_reader.devices:
        #   samples = SELECT all utilization samples for <device_name>
        #   plotter.plot(samples) @ png=util_scale.<device_id>.png

        # The start time of all traced Events / utilization samples.
        # Use this as the "starting point" of the heat-scale.

        # TODO: How can we make things "match up" with the SummaryView?
        # Options:
        #
        # 1. Show utilization from [start_time_usec .. start_time_usec + duration_sec].
        #
        #    We currently use start_time_usec from events to decide on initial plot locations;
        #    so ideally we would sample the utilization samples running from:
        #    [start_time_usec .. start_time_usec + duration_sec] for each phase.
        #    PROBLEM:
        #    - actual time shown in subplots spans ~ 531 seconds;
        #    - total time according to utilization data is ~ 1244 seconds.
        #
        #    PRO: utilization will "appear" to match up with ResourceSubplot.
        #    CON: utilization will not actually match up with ResourceSubplot.
        #    PRO: easiest to implement.
        #
        # 2. "Condense" utilization from [subplot.start_time_usec .. subplot.end_time_usec] to fit within subplot height;
        #    don't show time-scale.
        #
        #    "Operations" may not capture all CPU/GPU activity.
        #    However, we can still show a condensed view of overall hardware utilization during that time.
        #
        #    CON: if "operations" are wrong, maybe we didn't capture high activity and so our ResourceSubplot subplot
        #         will look wrong in comparison.
        #    PRO: true hardware utilization is shown for "some" interval of time
        #
        # 3. Only keeps utilization samples that "match up" with the spans of time that make up our ResourceSubplot plots.
        #    PRO: ResourceOverlap and HeatScale will "match up".
        #    CON: We may be missing actual hardware utilization (programmer annotations are wrong)
        #    CON: time-consuming to implement.

        # start_time_sec = self.sql_reader.trace_start_time_sec
        start_time_sec = overlap_df['start_time_usec'].min() / constants.MICROSECONDS_IN_SECOND

        device_ids = IDSet()
        machine_ids = IDSet()

        class MyDevice:
            def __init__(self, device_name, machine_name):
                self._device_ids = device_ids
                self.device_name = device_name
                self._device_ids.add(self.device_name)
                self.device_id = self._device_ids.get_id(self.device_name)

                self._machine_ids = machine_ids
                self.machine_name = machine_name
                self._machine_ids.add(self.machine_name)
                self.machine_id = self._machine_ids.get_id(self.machine_name)

        # for device in self.sql_reader.util_devices():
        # samples = self.sql_reader.util_samples(device)
        util_df_groupby = util_df.groupby(['machine_name', 'device_name'])
        for group, samples in util_df_groupby:
            # Otherwise we cannot iterate over samples[0...len(samples)] due to group-by.
            samples = samples.reset_index()
            machine_name, device_name = group
            device = MyDevice(device_name, machine_name)

            # png = self.get_util_scale_png(device.device_id, device.device_name)
            # plotter = HeatScale(
            #     color_value='util', y_axis='start_time_sec',
            #     png=png,
            #     pixels_per_square=self.pixels_per_square,
            #     # Anchor colormap colors using min/max utilization values.
            #     vmin=0.0, vmax=1.0,
            #     # 1 second
            #     step=1.)

            if self.debug:
                # Print the unadjusted raw utilization + timestamp data, centered @ start_time_sec.
                raw_centered_time_secs = (np.array(samples['start_time_sec']) - start_time_sec).tolist()
                raw_df = pd.DataFrame({
                    'util':samples['util'],
                    'start_time_sec':raw_centered_time_secs,
                }).astype(float)
                logger.info("> DEBUG: Unadjusted raw utilization measurements for device={dev}".format(dev=device_name))
                logger.info(raw_df)
            norm_time_secs, norm_utils = exponential_moving_average(
                samples['start_time_sec'], samples['util'],
                start_time_sec, self.step_sec, self.decay)
            centered_time_secs = (np.array(norm_time_secs) - start_time_sec).tolist()
            unnorm_samples = {
                'util': list(samples['util']),
                'start_time_sec': list(samples['start_time_sec']),
            }
            norm_samples = {
                'util':norm_utils,
                'start_time_sec':centered_time_secs,
            }

            util_stats = pd.DataFrame(norm_utils).agg([pd.np.min, pd.np.max, pd.np.mean])
            logger.info("Utilization stats for device={dev}\n{msg}".format(
                dev=device_name,
                msg=pprint_msg(util_stats)))

            plot_df = pd.DataFrame(norm_samples).astype(float)
            self.dump_plot_data(plot_df, device)
            self.dump_js_data(norm_samples, unnorm_samples, device, start_time_sec)
            # plotter.add_data(norm_samples)
            # print("> HeatScalePlot @ {path}".format(path=png))
            # plotter.plot()

        # self.sql_reader.close()

    def dump_js_data(self, norm_samples, unnorm_samples, device, start_time_sec):

        def add_device_metadata(md, device):
            md['device_name'] = device.device_name
            md['device_id'] = device.device_id
            md['machine_name'] = device.machine_name
            md['machine_id'] = device.machine_id

        def add_HeatScale_metadata(md):
            md['plot_type'] = 'HeatScale'
            md['start_time_usec'] = float(start_time_sec)*constants.MICROSECONDS_IN_SECOND
            md['step_usec'] = self.step_sec*constants.MICROSECONDS_IN_SECOND
            md['decay'] = self.decay

        md = dict()

        add_device_metadata(md, device)
        add_HeatScale_metadata(md)

        js = {
            'metadata': md,
            'data': {
                'util': norm_samples['util'],
                'start_time_sec': norm_samples['start_time_sec'],
            },
            'unnorm_data': {
                'util': unnorm_samples['util'],
                'start_time_sec': unnorm_samples['start_time_sec'],
            },
        }
        def add_stats(js_key, data_key):
            if 'stats' not in js['metadata']:
                js['metadata']['stats'] = dict()
            if js_key not in js['metadata']['stats']:
                js['metadata']['stats'][js_key] = dict()
            if data_key not in js['metadata']['stats'][js_key]:
                js['metadata']['stats'][js_key][data_key] = dict()
            agg_df = pd.DataFrame(js[js_key][data_key]).agg(['min', 'max', 'mean'])
            agg_keys = list(iter(agg_df.index))
            for agg_key in agg_keys:
                value = agg_df.loc[agg_key].values[0]
                js['metadata']['stats'][js_key][data_key][agg_key] = value

        add_stats('data', 'util')
        add_stats('unnorm_data', 'util')
        # add_stats('start_time_sec')

        path = self._json_path(device.device_id, device.device_name)
        print("> HeatScalePlot @ plot data @ {path}".format(path=path))
        do_dump_json(js, path)

    def dump_plot_data(self, plot_df, device):
        path = self._plot_data_path(device.device_id, device.device_name)
        print("> HeatScalePlot @ plot data @ {path}".format(path=path))
        with open(path, 'w') as f:
            DataFrame.print_df(plot_df, file=f)
        logger.info(plot_df)

    @property
    def db_path(self):
        return sql_input_path(self.directory)

def disable_test_pixel_bar():
    """
    Can we make the total height-pixels of a bar-plot have a 1-to-1 correspondence for pixel-per-sec?

    :return:
    """
    width_px = 500
    height_px = 500

    # This guarantees a 500x500 plot.
    fig = plt.figure(figsize=(
        pixels_as_inches(width_px),
        pixels_as_inches(height_px)))

    # How much space to reserve for ylabels
    # (only relevant if ylabels are outside the plot).
    # ylabel_spacer_percent = 0.1
    ylabel_spacer_percent = 0.0

    # We'd use these values to make the plot-area fill the ENTIRE figure size.
    # (including the black-plot-outline even!)
    # left = 0.0
    # bottom = 0.0
    # width = 1.0
    # height = 1.0

    # How much wiggle-room to reserve along the outside of the plot-area for the plot-area-outline?
    pixels_for_outline = 2

    width_percent_per_pixel = 1.0 / width_px
    width_percent_line_spacer = pixels_for_outline * width_percent_per_pixel

    height_percent_per_pixel = 1.0 / height_px
    height_percent_line_spacer = pixels_for_outline * height_percent_per_pixel

    left = 0.0 + ylabel_spacer_percent + width_percent_line_spacer
    bottom = 0.0 + height_percent_line_spacer
    width = 1.0 - ylabel_spacer_percent - 2*width_percent_line_spacer
    height = 1.0 - 2*height_percent_line_spacer

    # This guarantees the plot-area fills the entire 500x500 figure size.
    ax = fig.add_axes([left, bottom, width, height])
    yvalues = np.array(range(1, 5+1, 1))
    xvalues = ["one", "two", "three", "four", "five"]
    bar_width = 0.25
    plot = ax.bar(xvalues, yvalues,
                  color=color, width=bar_width, edgecolor='black',
                  # label=bench_name,
                  # bottom=self._bottom,
                  # hatch=hatch,
                  # yerr=yerr,
                  )
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
    )

    # Put y-label inside the plot
    ax.tick_params(axis="y", direction="in", pad=-22)

    # Adjust y-min we can see the "0" y-label still.
    ymin, ymax = ax.get_ylim()
    assert ymin == 0
    ax.set_ylim(-0.1, ymax)

    # Overlap with next y-tick
    fig.savefig('./test_pixel_bar.png',
                # Remove excess whitespace around the plot.
                # NOTE: This does NOT guarantee a 500x500 plot; in fact it ADDS whitespace.
                # bbox_inches='tight',
                )

def disable_test_stacked_bar():
    """
    PROBLEM: StackedBarPlotter code is coupled with how data is formatted...
    Really just want it to care about what data it has to plot.

    TEST CASES:
    1. Plot existing data, legends and all
    2. Plot simulated data <- do this and refactor plotting code.
       2 operations: forward and backward
       categories: python, c++, gpu

       forward: {
         python:3,
         c++:2,
         gpu:1,
       }
       backward: {
         python:1,
         c++:2,
         gpu:3,
       }

    3. Add sec-per-pixel:
       a. plot the same simulated data, and show 10 pixels per sec
       b. plot the same simulated data, and show 5 pixels per sec
          EXPECT: the total plot height should be halved, but should have the same proportions.

    :return:
    """
    data = {
        'operations':['Forward', 'Backward'],
        'categories':['Python', 'C++', 'GPU'],
        'time_sec': {
            'Forward': {
                'Python':[3],
                'C++':[2],
                'GPU':[1],
            },
            'Backward': {
                'Python':[1],
                'C++':[2],
                'GPU':[3],
            },
        },
    }
    # bench_name here is "Forward" or "Backward"
    def get_png(bench_name):
        return "test_stacked_bar{bench}.png".format(
            bench=bench_suffix(bench_name),
        )


    directory = "."
    def get_plot_data_path(bench_name):
        return _j(directory, "test_stacked_bar.plot_data{bench}.txt".format(
            bench=bench_suffix(bench_name),
        ))

    class DebugReader:
        def __init__(self, op_data):
            self.op_data = op_data

        def get_categories(self):
            return list(self.op_data.keys())

        def get_times_sec(self, category):
            return self.op_data[category]

    all_categories = sorted(data['categories'])
    category_order = sorted(all_categories)
    bench_name_labels = DQN_BENCH_NAME_LABELS
    category_color_map = None
    category_labels = None
    impl_name_order = IMPL_NAME_ORDER
    device_order = DEVICE_ORDER

    # TODO:
    # - generate a legend separately

    bar_width = 0.25
    show = False
    dynamic_size = True
    # seconds / pixel
    sec_per_pixel = 1. / 10.
    plotter = StackedBarPlotter(
        get_png, get_plot_data_path,
        category_order,
        impl_name_order,
        device_order,
        bench_name_labels=bench_name_labels,
        category_color_map=category_color_map,
        category_labels=category_labels,
        bar_width=bar_width, show=show,
        json_reader_klass=DebugReader,
        title='CPU/GPU utilization over training',
        # TODO: use "minigo"
        xlabel='',
        ylabel='Total training time (seconds)',
        yvalue_per_pixel=sec_per_pixel,
        width_px=500,
        dynamic_size=dynamic_size,
    )

    bench_names = data['operations']
    for bench_name in bench_names:
        op_data = data['time_sec'][bench_name]
        device_name = NO_DEVICE_NAME
        impl_name = NO_IMPL_NAME
        plotter.add_json_data(op_data, bench_name, device_name, impl_name, debug=True)

    plotter.plot()

def disable_test_just_legend():
    """
    Only show the legend, not the plot.
    """
    fig = plt.figure()
    figlegend = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(111)
    lines = ax.plot(range(10), range(10), range(10), range(10))
    figlegend.legend(lines, ('one', 'two'), 'center')
    figlegend.savefig(
        'test_just_legend.png',
        bbox_inches="tight")

def get_sns_kwargs():
    sns_kwargs = dict()
    sns_kwargs['capsize'] = 0.04
    sns_kwargs['errwidth'] = 1.25
    return sns_kwargs

def get_plt_kwargs():
    plt_kwargs = dict()
    plt_kwargs['capsize'] = 5
    return plt_kwargs

def add_grouped_stacked_bars(
    x, x_group, y, hue,
    bar_label=None,
    bar_label_x_offset=0,
    bar_label_kwargs=dict(),
    label=None,
    label_order=None,
    data=None,
    ax=None,
    xfield_label_func=None,
    rotation=None,
    debug=False,
    **kwargs):
    # sns.barplot(x=.., y=.., hue=..)

    # add_stacked_bars(
    #     x='x_field',
    #     y='total_overhead_sec',
    #     hue='overhead_type_order',
    #     label='pretty_overhead_type',
    #     data=total_plot_data, ax=ax,
    #     debug=self.debug,
    #     **plt_kwargs)

    def _calc_bar_xloc(n, w):
        """
        n = total number of bars
        w = width per bar

        :param n:
        :return:
        """
        i = np.arange(n) - (n // 2)
        if n % 2 == 0:
            # even:
            # n = 4
            # i = [0, 1, 2, 3]
            ws = i*w + w/2
        else:
            # odd:
            # n = 5
            # i = [0, 1, 2, 3, 4]
            ws = i*w
        return ws

    x_groups = data[x_group].unique()
    total_bar_width = 2/3
    # width per bar:
    # bar_width = 0.33
    if len(x_groups) > 0:
        bar_width = total_bar_width/len(x_groups)
    else:
        # Prevent division by zero for empty dataframes.
        bar_width = 0.33
    bar_xloc = _calc_bar_xloc(len(x_groups), bar_width)

    def _add_stacked_bars(xgroup_idx, data=None):
        # sns.barplot(x=.., y=.., hue=..)

        # Q: Does order of "data" affect groups returned by groupby?
        if label is not None:
            groupby_cols = [hue, label]
        else:
            groupby_cols = [hue]
        data_groupby = data.groupby(groupby_cols)
        groups = [pair[0] for pair in list(data_groupby)]
        logger.info("groups: " + pprint_msg(groups))
        means = dict()
        stds = dict()
        for group, group_df in data_groupby:
            means[group] = group_df.groupby([x]).mean().reset_index()
            stds[group] = group_df.groupby([x]).std().reset_index()
        bottom = None

        def _xfield_label(xg):
            if xfield_label_func is not None:
                return xfield_label_func(xg)
            return xg

        # xtick_labels = means[groups[0]][x]
        all_xs  = means[groups[0]][x]
        xtick_labels = [_xfield_label(xfield) for xfield in means[groups[0]][x]]
        xticks = np.arange(len(xtick_labels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=rotation)

        x_to_xtick = dict()
        for xfield, xtick in zip(all_xs, xticks):
            x_to_xtick[xfield] = xtick

        for group in groups:
            xs = means[group][x]
            assert (xs == all_xs).all()
            ys = means[group][y]
            std = stds[group][y]
            # plt.bar(x=xs, height=ys, yerr=std, bottom=bottom, ax=ax)

            # Order in which we call this determines order in which stacks appear.
            if label is not None:
                # group = [hue, label]
                label_str = group[1]
            else:
                # group = hue
                label_str = group
            if debug:
                logger.info("add_stacked_bars:\n{msg}".format(
                    msg=pprint_msg({
                        'xs':xs,
                        'xticks':xticks,
                        'ys':ys,
                    })))
            # barplot = ax.bar(x=xs, height=ys, yerr=std, label=label_str, bottom=bottom, **kwargs)
            # [width/n]
            # bar_x = []
            # -1 1
            # -2 -1 1 2
            ## for
            barplot = ax.bar(x=xticks + bar_xloc[xgroup_idx], height=ys, width=bar_width, yerr=std, label=label_str, bottom=bottom, **kwargs)
            xtick_to_barplot = dict()
            for xtick, b in zip(xticks, barplot):
                xtick_to_barplot[xtick] = b
            # # TODO: add rect labels... how?
            if bar_label is not None:
                bs = []
                for xfield in data[x]:
                    xtick = x_to_xtick[xfield]
                    bs.append(xtick_to_barplot[xtick])

                bar_labels = data['bar_label']
                # add_ax_bar_labels(ax, barplot, bar_labels, x_offset=bar_label_x_offset, **bar_label_kwargs)
                add_ax_bar_labels(ax, bs, bar_labels, x_offset=bar_label_x_offset, **bar_label_kwargs)

            if bottom is None:
                bottom = ys
            else:
                bottom += ys
        return barplot

    xgroup_groupby = data.groupby([x_group])
    xgroup_barplot = dict()
    for xgroup_idx, (xgroup, xgroup_df) in enumerate(xgroup_groupby):
        logger.info("xgroup = {xgroup}".format(xgroup=xgroup))
        # Q: Does order of "data" affect groups returned by groupby?
        barplot = _add_stacked_bars(xgroup_idx, data=xgroup_df)
        xgroup_barplot[xgroup] = barplot

    handles, labels = ax.get_legend_handles_labels()
    if label_order and label:
        lb_to_order = dict()
        for order, lb in [g for (g, d) in data.groupby([label_order, label])]:
            lb_to_order[lb] = order
        new_labels = reversed(sorted(labels, key=lambda lb: lb_to_order[lb]))
        label_handle_pairs = reversed(sorted(((lb_to_order[lb], h) for lb, h in zip(labels, handles))))
        new_handles = [pair[1] for pair in label_handle_pairs]
        # ax.legend(new_handles, new_labels)
    else:
        # Reverse legend label order (when making stacked bar its in reverse order)
        # ax.legend(handles[::-1], labels[::-1], title=None, loc='upper left')
        new_handles = handles[::-1]
        new_labels = labels[::-1]
        # ax.legend(new_handles, new_labels)
    ax.legend(new_handles, new_labels,
              fancybox=True, framealpha=0.5)



    return xgroup_barplot

def add_bar_labels(y, hue, ax=None, get_bar_labels=None, y_add_factor=0.025):
    """
    :param y:
    :param hue:
    :param ax:
    :param get_bar_labels:
    :param y_add_factor:
        Multiply y-axis by y_add_factor and add it to label position;
        useful for preventing overlap with error bars.
    :return:
    """
    # Iterate through the list of axes' patches
    # Q: In what order do we iterate the patches?

    if get_bar_labels is not None:
        # Q: I have NO idea what this would do if there were multiple x_fields... oh well.
        _, labels = ax.get_legend_handles_labels()
        ys = [p.get_height() for p in ax.patches]
        df = pd.DataFrame({
            hue: labels,
            y: ys,
            # Not sure how to get x_field...only xpos.
            # x: ,
        })
        # df['bar_label'] = get_bar_labels(df)
        bar_labels = get_bar_labels(df)

    for i, (p, label) in enumerate(zip(ax.patches, labels)):
        xpos = p.get_x() + p.get_width()/2.
        y_bottom, y_top = ax.get_ylim()
        add_ypos = y_add_factor*(y_top - y_bottom)
        ypos = p.get_height() + add_ypos
        if get_bar_labels is None:
            bar_label = '%d' % int(p.get_height())
        else:
            bar_label = bar_labels[i]

        ax.text(xpos, ypos, bar_label,
                ha='center', va='bottom')

def add_ax_bar_labels(ax, rects, labels, x_offset=0, y_offset=0, **kwargs):
    """Attach a text label above each bar in *rects*, displaying its height."""
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    # 3 points vertical offset
    xy_offset = (x_offset, y_offset + 3)
    for rect, label in zip(rects, labels):
        if label is not None and label != "":
            height = rect.get_height()
            # label = '{}'.format(height)
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=xy_offset,
                        textcoords="offset points",
                        ha='center', va='bottom',
                        **kwargs)


class CUDAEventCSVReader:
    def __init__(self, directory, debug=False, **kwargs):
        self.directory = directory
        self.debug = debug

    def read_df(self, verbose=False):
        if verbose:
            logger.info("Read cuda device events dataframe @ {path}".format(path=self.directory))
        reader = CUDADeviceEventsReader(self.directory)
        df = reader.read()
        if verbose:
            logger.info("Output csv @ {path}".format(path=self._csv_path))
        # colnames = [
        #     'machine_name',
        #     'process_name',
        #     'phase_name',
        #     'device_name',
        #     'event_name',
        #     'cuda_event_type',
        #     'start_time_us',
        #     'duration_us',
        #     'CUDA_VISIBLE_DEVICES',
        # ]

        df.sort_values(by=['start_time_us'], inplace=True)
        return df

    def run(self):
        df = self.read_df(verbose=True)
        df.to_csv(self._csv_path, index=False)

    @property
    def _csv_path(self):
        return _j(self.directory, "cuda_device_events.csv")


def fix_seaborn_legend(ax, legend_kwargs=dict()):
    leg = ax.get_legend()
    handles = leg.legendHandles
    labels = [txt.get_text() for txt in leg.texts]
    # First handle/label is the legend "title"... ignore it.
    handles = handles[1:]
    labels = labels[1:]
    # Remove seaborn created legend that has a legend title.
    ax.get_legend().remove()
    # Create our own legend, without a title.
    ax.legend(
        handles=handles,
        labels=labels,
        **legend_kwargs,
    )

from rlscope.profiler.rlscope_logging import logger
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test-stacked-bar', action='store_true')
    p.add_argument('--test-pixel-bar', action='store_true')
    p.add_argument('--test-just-legend', action='store_true')
    args = p.parse_args()

    if args.test_stacked_bar:
        test_stacked_bar()
        return

    if args.test_pixel_bar:
        test_pixel_bar()
        return

    if args.test_just_legend:
        test_just_legend()
        return

if __name__ == '__main__':
    main()
