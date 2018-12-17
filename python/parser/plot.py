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

