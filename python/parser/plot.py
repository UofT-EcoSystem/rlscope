import re
import sys
import os
import csv
import textwrap
import pprint
from io import StringIO
import json
import codecs
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from parser.common import *
from parser.nvprof import CUDASQLiteParser
from parser.pyprof import PythonProfileParser
from parser.tfprof import OverlapComputer
from parser.db import SQLiteCategoryTimesReader, traces_db_path

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
HATCH_STYLES = ['/', '\\', '|', 'x', 'o', '.', '*']
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

# Generalize:
# Instead of just "Python", "C++", "GPU", we want to break down the labels arbitrairily.
# Also, we want to control the order of the legend labels.

class CategoryOverlapPlot:
    """
    Create a stacked bar plot.
    For the list of times to show, see self.category_order.
    """
    def __init__(self, directory,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.debug = debug

        self.bar_width = 0.25
        self.show = False

    def get_source_files(self):
        """
        We want traces.db
        """
        src_files = []
        traces_db = traces_db_path(self.directory)
        if not _e(traces_db):
            raise MissingInputFiles(textwrap.dedent("""
                {klass}: Couldn't find any traces.db at {path}.
                """.format(
                klass=self.__class__.__name__,
                path=traces_db,
            )))
        return src_files

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

        self.sql_reader = SQLiteCategoryTimesReader(self.db_path)
        self.bench_names = self.sql_reader.bench_names + [NO_BENCH_NAME]
        assert len(self.bench_names) == len(unique(self.bench_names))
        self.categories = self.sql_reader.categories

        overlap_computer = OverlapComputer(self.db_path, debug=self.debug)

        all_categories = set()
        json_datas = []
        for bench_name in self.bench_names:
            json_data = overlap_computer.compute_per_operation_overlap(bench_name)
            json_datas.append(json_data)

            for combo_and_times in json_data['category_combo_times']:
                category = _category_str(combo_and_times['category_combo'])
                all_categories.add(category)

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
        for bench_name in self.bench_names:
            self.plotter.plot(bench_name)
            self._dump_cpu_gpu_stats(bench_name)

    def _dump_cpu_gpu_stats(self, bench_name):
        js_stats = dict()
        _add_cpu_gpu_stats(js_stats, self.plotter, bench_name)
        print("> Save plot stats to {path}".format(path=self._stats(bench_name)))
        do_dump_json(js_stats, self._stats(bench_name))

    @property
    def db_path(self):
        return traces_db_path(self.directory)

class StackedBarPlotter:
    def __init__(self, get_png, get_plot_data_path,
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
                 title=None):
        if callable(get_png):
            self.get_png = get_png
            self.png = None
        else:
            assert type(get_png) == str
            self.get_png = None
            self.png = get_png

        if callable(get_plot_data_path):
            self.get_plot_data_path = get_plot_data_path
            self.plot_data_path = None
        else:
            assert type(get_plot_data_path) == str
            self.get_plot_data_path = None
            self.plot_data_path = get_plot_data_path

        self.bar_width = bar_width
        self.show = show
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
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
        assert len(xs) <= len(HATCH_STYLES)
        hatch_map = dict()
        for x, hatch_style in zip(xs, HATCH_STYLES):
            hatch_map[x] = hatch_style
        return hatch_map

    def _build_bench_name_order(self):
        # # Delay making this until we know all the bench_name's from add_json_data
        # self.bench_name_order = ['q_update_target_network', 'q_forward', 'q_backward', 'step']
        self.bench_name_order = sorted(unique(self.df_data['bench_name']))
        self.bench_name_order_map = as_order_map(self.bench_name_order)
        self.rev_bench_name_order_map = reverse_dict(self.bench_name_order_map)
        self.bench_name_hatch_map = self.as_hatch_map(self.bench_name_order)

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

    def plot(self, bench_name=NO_BENCH_NAME):
        if self.df is None:
            self._as_dataframe()

        # Keep this...
        fig = plt.figure()

        if bench_name is None:
            all_benches = [NO_BENCH_NAME]
        elif bench_name == NO_BENCH_NAME:
            all_benches = self.get_plot_bench_names()
        else:
            all_benches = [bench_name]

        for bench_name in all_benches:

            bench_df = self.plot_data(bench_name)

            with open(self.get_plot_data_pt(bench_name), 'w') as f:
                DataFrame.print_df(bench_df, file=f)
            print("> DataFrame:")
            print(bench_df)

            self._add_lines(bench_name)
            self._add_legend(bench_name)
            self._add_axis_labels(bench_name)
            self._show(bench_name)

    def get_plot_bench_names(self):
        if self.get_png is not None:
            all_benches = [NO_BENCH_NAME] + unique(self.mean_df['bench_name'])
            return all_benches

        return [NO_BENCH_NAME]

    def get_png_path(self, bench_name):
        if self.get_png is not None:
            return self.get_png(bench_name)

        return self.png

    def get_plot_data_pt(self, bench_name):
        if self.get_plot_data_path is not None:
            return self.get_plot_data_path(bench_name)

        return self.plot_data_path

    def _show(self, bench_name=None):
        if self.show:
            plt.show()
        else:
            print("> Save figure to {path}".format(path=self.get_png_path(bench_name)))
            print("> Save plot data to {path}".format(path=self.get_plot_data_pt(bench_name)))
            plt.savefig(self.get_png_path(bench_name), bbox_inches="tight")
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
            print("> bench_name={op}, json_data = ".format(op=bench_name))
            print(textwrap.indent(pprint.pformat(json_data), prefix="  "))
            print("  > categories = {c}".format(c=categories))
        for category in categories:
            if debug:
                print("> add category={c}: {times}".format(
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

    def _add_legend(self, bench_name=None):
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
                                       'loc':'upper right',
                                       'labelspacing': 1.2,
                                       'handlelength': 3,
                                       'handleheight': 2,
                                   })
        legend_kwargs.append({'loc': 'upper left',
                              'bbox_to_anchor': (1.04, 1)})
        self.legend_makers.append(hatch_legend)

        color_legend = LegendMaker(attr_name='facecolor',
                                   field_to_attr_map=self.category_color_map,
                                   field_order=self.category_order,
                                   labels=self.category_labels,
                                   edgecolor='white',
                                   legend_kwargs={
                                       'loc':'upper left',
                                       'handlelength': 3,
                                       'handleheight': 2,
                                   })
        legend_kwargs.append({'loc': 'lower left',
                              'bbox_to_anchor': (1.04, 0)})
        self.legend_makers.append(color_legend)

        LegendMaker.add_legends(self.legend_makers,
                                legend_kwargs=legend_kwargs)


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
        self.df = self.df.sort_values(by=['impl_name_order', 'device_order', 'bench_name_order', 'category_order'])
        # groupby_cols = DataFrame.get_groupby_cols(self.orig_df, value_field)
        self.df['std_div_mean_percent'] = 100 * self.df['std']/self.df['mean']

        self.mean_df = self.df


    def _add_axis_labels(self, bench_name=None):
        if self.title is not None:
            plt.title(self.title)

        if self.xlabel is not None:
            # , fontweight='bold'
            if bench_name == NO_BENCH_NAME:
                plt.xlabel(self.xlabel)
            else:
                plt.xlabel(get_pretty_bench(bench_name))

        if self.ylabel is not None:
            plt.ylabel(self.ylabel)

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
    assert type(category_combo) in [list, tuple, frozenset, set]

    # HACK to make CategoryOverlapPlot more readable...
    # technically "Framework API C" overlaps with all the "GPU" time and other stuff, but it makes things annoying to read.
    # So, only keep "Framework API C" if it is the only category in the combo, otherwise remove it.
    if len(category_combo) > 1 and CATEGORY_TF_API in category_combo:
        new_category_combo = list(category_combo)
        new_category_combo.remove(CATEGORY_TF_API)
        category_combo = new_category_combo

    # for category in category_combo:
    #     # Otherwise, we cannot do re.split(r' \+ ', ...) to recover the category_combo.
    #     assert '+' not in category
    return " + ".join(sorted(category_combo))

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
            times_sec = np.array(cat_data['times_usec'])/MICROSECONDS_IN_SECOND
            if category == cat_str:
                # Ignore the first time since it includes libcupti.so load time.
                return times_sec[1:]
        print("> json_data = ")
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
        for category_combo, time in self.json_data.items():
            cat_str = _category_str(category_combo)
            if category == cat_str:
                return [time/MICROSECONDS_IN_SECOND]
        # for cat_data in self.json_data['category_combo_times']:
        #     category_combo = cat_data['category_combo']
        #     cat_str = _category_str(category_combo)
        #     times_sec = np.array(cat_data['times_usec'])/MICROSECONDS_IN_SECOND
        #     if category == cat_str:
        #         # Ignore the first time since it includes libcupti.so load time.
        #         return times_sec[1:]
        print("> json_data = ")
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
    def add_legends(legend_makers, legend_kwargs=[]):
        legends = []
        for i, legend_maker in enumerate(legend_makers):
            if i < len(legend_kwargs):
                legend_kwarg = legend_kwargs[i]
            else:
                legend_kwarg = dict()
            legend = legend_maker.get_legend(**legend_kwarg)
            legends.append(legend)
        for legend in legends:
            plt.gca().add_artist(legend)
        return legends


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

class UtilizationPlot:
    """
    CPU/GPU utilization over training.
    """
    def __init__(self, directory,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.debug = debug

        self.bar_width = 0.25
        self.show = False

    def get_source_files(self):
        """
        We want traces.db
        """
        src_files = []
        traces_db = traces_db_path(self.directory)
        if not _e(traces_db):
            raise MissingInputFiles(textwrap.dedent("""
                {klass}: Couldn't find any traces.db at {path}.
                """.format(
                klass=self.__class__.__name__,
                path=traces_db,
            )))
        return src_files

    def _process_timeline_png(self, bench_name):
        return UtilizationPlot.get_process_timeline_png(self.directory, bench_name)

    @staticmethod
    def get_process_timeline_png(directory, bench_name):
        return _j(directory, "process_timeline{bench}.png".format(
            bench=bench_suffix(bench_name)))

    @staticmethod
    def get_plot_data_path(directory, bench_name):
        return _j(directory, "process_timeline.plot_data{bench}.txt".format(
            bench=bench_suffix(bench_name)))

    def _plot_data_path(self, bench_name):
        return UtilizationPlot.get_plot_data_path(self.directory, bench_name)

    def run(self):

        self.sql_reader = SQLiteCategoryTimesReader(self.db_path)
        # self.bench_names = self.sql_reader.bench_names + [NO_BENCH_NAME]
        # assert len(self.bench_names) == len(unique(self.bench_names))
        # self.categories = self.sql_reader.categories

        overlap_computer = OverlapComputer(self.db_path, debug=self.debug)

        operation_overlap = overlap_computer.compute_process_timeline_overlap()
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
            for category_key, time in combo_to_time.items():
                category = _category_str(category_key)
                all_categories.add(category)

        pprint.pprint({'all_categories': all_categories})

        self.category_order = sorted(all_categories)
        # self.bench_name_labels = DQN_BENCH_NAME_LABELS
        # TODO: create bench name labels
        self.bench_name_labels = None
        self.category_color_map = None
        self.category_labels = None
        self.impl_name_order = IMPL_NAME_ORDER
        self.device_order = DEVICE_ORDER
        self.plotter = StackedBarPlotter(
            self._process_timeline_png, self._plot_data_path,
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
        assert len(df) != 0
        self._dump_stats()
        self.plotter.plot(bench_name=None)

    @property
    def db_path(self):
        return traces_db_path(self.directory)

    @staticmethod
    def get_stats(directory):
        return _j(directory, "process_timeline.stats.json")

    def _stats(self):
        return UtilizationPlot.get_stats(self.directory)

    def _dump_stats(self):
        """
        Dump some stats useful for testing the correctness of our plot.

        - Total time spent tracing:
          We expect total time spent tracing to match that total size of our bar-graph.
          NOTE: would be nice to measure this with time.time() separately, but oh well!

        -
        :param bench_name:
        :return:
        """
        total_trace_time_sec = self.sql_reader.total_trace_time_sec(debug=self.debug)
        # EXPECT:
        # - total_trace_time_sec    ~ total_time_sec
        #   --------------------      --------------
        #   Anything that's traced    Stuff covered by operations
        # IF FAILS:
        # - then we aren't profiling part of the code.
        js_stats = {
            # Select min(start_time_us) as, max(end_time_us) from Event
            # (i.e. across all processes)
            'total_trace_time_sec':total_trace_time_sec,
        }
        _add_cpu_gpu_stats(js_stats, self.plotter)
        print("> Save plot stats to {path}".format(path=self._stats()))
        do_dump_json(js_stats, self._stats())
        return js_stats

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
    js_stats.update({
        'cpu_time_sec':cpu_time_sec,
        'gpu_time_sec':gpu_time_sec,
        'total_time_sec':total_time_sec,
        'gpu_time_percent': 100*gpu_time_sec/total_time_sec,
        'cpu_time_percent': 100*cpu_time_sec/total_time_sec,
    })
