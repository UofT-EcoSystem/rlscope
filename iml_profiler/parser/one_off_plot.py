import logging
import argparse
import traceback
import bdb
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

import matplotlib
import matplotlib.ticker
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler.parser.stacked_bar_plots import get_x_env, get_x_algo
from iml_profiler.parser.dataframe import UtilDataframeReader

from iml_profiler import py_config
from iml_profiler.parser.common import *
from typing import *

def maybe_number(x):
    if type(x) != str:
        return x

    try:
        num = int(x)
        return num
    except ValueError:
        pass

    try:
        num = float(x)
        return num
    except ValueError:
        pass

    return x

def parse_filename_attrs(
    path : str,
    file_prefix : str,
    file_suffix : str,
    attrs : Iterable[str],
    dflt_attrs : Optional[Dict[str, Any]] = None):
    attr_name_regex = r'(?:{regex})'.format(
        regex='|'.join(sorted(attrs, key=lambda attr: (-1*len(attr), attr)))
    )
    attr_string_regex = r'(?P<attr_name>{attr_name})_(?P<attr_value>[^\.]*)'.format(
        attr_name=attr_name_regex
    )
    # e.g.
    # path = 'GPUHwCounterSampler.thread_blocks_68.thread_block_size_1024.csv'

    # e.g.
    # ['GPUHwCounterSampler', 'thread_blocks_68', 'thread_block_size_1024', 'csv']
    components = re.split(r'\.', _b(path))
    assert components[0] == file_prefix
    assert components[-1] == file_suffix
    attr_strings = components[1:len(components)-1]
    attr_vals = dict()
    if dflt_attrs is not None:
        attr_vals.update(dflt_attrs)
    for attr_string in attr_strings:
        m = re.fullmatch(attr_string_regex, attr_string)
        if not m:
            raise RuntimeError(f"""
            Not sure how to parse attribute name/value from \"{attr_string}\" found in {_b(path)}.
              Attributes we recognize = {attrs}
            """)
        attr_vals[m.group('attr_name')] = m.group('attr_value')
    return attr_vals

def yes_as_bool(yes_or_no):
    if yes_or_no.lower() in {'yes', 'y', 'on', '1'}:
        return True
    return False

def parse_path_attrs(
    path : str,
    attrs : Iterable[str],
    dflt_attrs : Optional[Dict[str, Any]] = None,
    attr_types : Optional[Dict[str, Any]] = None,
    ):

    attr_name_regex = r'(?:{regex})'.format(
        regex='|'.join(sorted(attrs, key=lambda attr: (-1*len(attr), attr)))
    )

    attr_string_regex = r'(?P<attr_name>{attr_name})_(?P<attr_value>[^\.]*)\b'.format(
        attr_name=attr_name_regex
    )
    # e.g.
    # path = 'GPUHwCounterSampler.thread_blocks_68.thread_block_size_1024.csv'

    attr_vals = dict()
    if dflt_attrs is not None:
        attr_vals.update(dflt_attrs)

    path_components = os.path.split(path)
    for path_component in path_components:

        # e.g.
        # ['GPUHwCounterSampler', 'thread_blocks_68', 'thread_block_size_1024', 'csv']
        attr_strings = re.split(r'\.', path_component)
        for attr_string in attr_strings:
            m = re.search(attr_string_regex, attr_string)
            if m:
                value = m.group('attr_value')
                attr_name = m.group('attr_name')
                if attr_types is not None and attr_name in attr_types:
                    value = attr_types[attr_name](value)
                attr_vals[attr_name] = value
            # if not m:
            #     raise RuntimeError(f"""
            #     Not sure how to parse attribute name/value from \"{attr_string}\" found in {path}.
            #       Attributes we recognize = {attrs}
            #     """)

    missing_attrs = set(attrs).difference(attr_vals.keys())
    if len(missing_attrs) > 0:
        raise RuntimeError(f"""
            Couldn't find all requires attributes in {path}.
              Attributes we are missing = {missing_attrs}
            """)

    return attr_vals

METRIC_NAME_CUPTI_TO_PROF = {
    # Deprecated CUPTI metric API -- achieved_occupancy:
    #    Id        = 1205
    #    Shortdesc = Achieved Occupancy
    #    Longdesc  = Ratio of the average active warps per active cycle to the maximum number of warps supported on a multiprocessor
    'achieved_occupancy': "sm__warps_active.avg.pct_of_peak_sustained_active",

    # Deprecated CUPTI metric API -- sm_efficiency:
    #    Id        = 1203
    #    Shortdesc = Multiprocessor Activity
    #    Longdesc  = The percentage of time at least one warp is active on a multiprocessor averaged over all multiprocessors on the GPU
    # See CUPTI documentation for mapping to new "Profiling API" metric name:
    #    https://docs.nvidia.com/cupti/Cupti/r_main.html#metrics_map_table_70
    'sm_efficiency': "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",

    # Deprecated CUPTI metric API -- inst_executed:
    #    Metric# 90
    #    Id        = 1290
    #    Name      = inst_executed
    #    Shortdesc = Instructions Executed
    #    Longdesc  = The number of instructions executed
    'inst_executed': "smsp__inst_executed.sum",

    # Deprecated CUPTI metric API -- active_cycles:
    #    Event# 25
    #    Id        = 2629
    #    Name      = active_cycles
    #    Shortdesc = Active cycles
    #    Longdesc  = Number of cycles a multiprocessor has at least one active warp.
    #    Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION
    'active_cycles': "sm__cycles_active.sum",

    # Deprecated CUPTI metric API -- active_warps:
    #    Event# 26
    #    Id        = 2630
    #    Name      = active_warps
    #    Shortdesc = Active warps
    #    Longdesc  = Accumulated number of active warps per cycle. For every cycle it increments by the number of active warps in the cycle which can be in the range 0 to 64.
    #    Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION
    'active_warps': "sm__warps_active.sum",

    # Deprecated CUPTI metric API -- elapsed_cycles_sm:
    #    Event# 33
    #    Id        = 2193
    #    Name      = elapsed_cycles_sm
    #    Shortdesc = Elapsed clocks
    #    Longdesc  = Elapsed clocks
    #    Category  = CUPTI_EVENT_CATEGORY_INSTRUCTION
    'elapsed_cycles_sm': "sm__cycles_elapsed.sum",
}

# HACK: number of total SMs on the RTX 2080 GPU on the "eco" cluster machines
NUM_SMS = 68

SM_OCCUPANCY_TITLE = "SM occupancy: average percent of warps\nthat are in use within an SM"
SM_EFFICIENCY_TITLE = "SM efficiency: percent of SMs\nthat are in use across the entire GPU"

SM_EFFICIENCY_Y_LABEL = f"SM efficiency (%)\n# SMs = {NUM_SMS}"
SM_OCCUPANCY_Y_LABEL = "SM occupancy (%)\nmax threads per block = 1024"
CUPTI_METRIC_Y_LABEL = {
    'sm_efficiency': SM_EFFICIENCY_Y_LABEL,
    'achieved_occupancy': SM_OCCUPANCY_Y_LABEL,
}

SAMPLE_THROUGHPUT_Y_LABEL = "Throughput (samples/second)"
BATCH_SIZE_X_LABEL = "Batch size"
STREAMS_X_LABEL = "# of CUDA streams"

RLSCOPE_X_LABEL = "(RL algorithm, Simulator)"

SM_ID_X_LABEL = f"SM ID\n# SMs = {NUM_SMS}"

GPU_UTIL_EXPERIMENT_ATTRS = {
    'thread_blocks',
    'thread_block_size',
    'n_launches',
    'iterations',
    'num_threads',
    'processes',
    'hw_counters',
}

MULTI_TASK_ATTRS = set(GPU_UTIL_EXPERIMENT_ATTRS)
MULTI_TASK_ATTRS.update({
    ## From directory attrs
    # 'thread_blocks',
    # 'thread_block_size',
    # 'n_launches',
    # 'iterations',
    # 'num_threads',
    'iterations_per_sched_sample',
    # 'processes',
    # 'hw_counters',

    ## GPUComputeSchedInfoKernel.thread_id_9.stream_id_9.trace_id_0.json
    'thread_id',
    'stream_id',
    'trace_id',
})
MULTI_TASK_JSON_ATTRS = {
    ## From contents of: GPUComputeSchedInfoKernel.thread_id_9.stream_id_9.trace_id_0.json
    "globaltimer_ns",
    "kernel_id",
    "lane_id",
    "sm_id",
    "stream_id",
    "warp_id",
}

FLOAT_RE = r'(?:[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
UNIT_RE = r'(?:\b(?:ms|s|qps)\b)'

class TrtexecExperiment:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.read_df()
        self.plot_df()

    def read_df(self):
        self._read_trtexec_df()

    def plot_df(self):
        """
        Plot trtexec7 experiments.
        :return:
        """

        """
        batch_size = 1, 8, 16, 32, 64
        streams = 1
        plot:
          throughput
          sm_efficiency
          sm_occupancy
        """
        def _plot_batch_size_vs(streams, suffix=None):
            self._plot_batch_size_vs_throughput(
                title="Throughput with increasing batch size",
                streams=streams,
                suffix=suffix)
            self._plot_batch_size_vs_metric(
                title=SM_EFFICIENCY_TITLE,
                cupti_metric='sm_efficiency',
                streams=streams,
                suffix=suffix)
            self._plot_batch_size_vs_metric(
                title=SM_OCCUPANCY_TITLE,
                cupti_metric='achieved_occupancy',
                streams=streams,
                suffix=suffix)
        _plot_batch_size_vs(streams=1)

        def _plot_streams_vs(batch_size, suffix=None):
            self._plot_streams_vs_throughput(
                title="Throughput with increasing streams\n(batch size = {batch_size})".format(batch_size=batch_size),
                batch_size=batch_size,
                suffix=suffix)
            self._plot_streams_vs_metric(
                # title="Throughput with increasing streams\n(batch size = {batch_size})".format(batch_size=batch_size),
                title=SM_EFFICIENCY_TITLE,
                cupti_metric='sm_efficiency',
                batch_size=batch_size,
                suffix=suffix)
            self._plot_streams_vs_metric(
                # title="Throughput with increasing streams\n(batch size = {batch_size})".format(batch_size=batch_size),
                title=SM_OCCUPANCY_TITLE,
                cupti_metric='achieved_occupancy',
                batch_size=batch_size,
                suffix=suffix)
        """
        batch_size = 1
        streams = 1, 2, 3, ..., 8
        plot:
          throughput
          sm_efficiency
          sm_occupancy
        """
        _plot_streams_vs(batch_size=1)
        """
        batch_size = (best batch size for streams == 1)
        streams = 1, 2, 3, ..., 8
        plot:
          throughput
          sm_efficiency
          sm_occupancy
        """
        best_batch_size = self._compute_best_batch_size()
        _plot_streams_vs(batch_size=best_batch_size, suffix='best_batch_size')

    def _compute_best_batch_size(self):
        df = self.trtexec_df[self.trtexec_df['streams'] == 1]
        max_throughput = df['host_latency_throughput_qps'].max()
        batch_sizes = df[df['host_latency_throughput_qps'] == max_throughput]['batch_size'].unique()
        assert len(batch_sizes) == 1
        best_batch_size = batch_sizes[0]
        return best_batch_size

    def _plot_streams_vs_metric(self, title, cupti_metric, batch_size, ylabel=None, suffix=None):
        if self.trtexec_gpu_hw_df is None:
            return
        df = copy.copy(self.trtexec_gpu_hw_df)
        """
        WANT:
        x_field: batch_size
        y_field: metric_value
        group_field: num_threads
        """

        df = df[df['batch_size'] == batch_size]

        df = keep_cupti_metric(df, cupti_metric)

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="streams", y="metric_value",
                        data=df,
                        # hue="num_threads", data=df,
                        # hue=col_titles["num_threads"], data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        if ylabel is None:
            ylabel = CUPTI_METRIC_Y_LABEL[cupti_metric]
        g.set_ylabels(ylabel)
        g.set_xlabels(STREAMS_X_LABEL)
        # title = "SM efficiency: percent of SMs\nthat are in use across the entire GPU"
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        if suffix is None:
            suffix = ""
        else:
            suffix = f".{suffix}"

        save_plot(df, _j(self.args['trtexec_dir'], f'streams_vs_{cupti_metric}.batch_size_{batch_size}{suffix}.svg'))

    def _plot_batch_size_vs_metric(self, title, cupti_metric, streams, ylabel=None, suffix=None):
        if self.trtexec_gpu_hw_df is None:
            return
        df = copy.copy(self.trtexec_gpu_hw_df)
        """
        WANT:
        x_field: batch_size
        y_field: metric_value
        group_field: num_threads
        """

        df = df[df['streams'] == streams]

        df = keep_cupti_metric(df, cupti_metric)

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="batch_size", y="metric_value",
                        data=df,
                        # hue="num_threads", data=df,
                        # hue=col_titles["num_threads"], data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        if ylabel is None:
            ylabel = CUPTI_METRIC_Y_LABEL[cupti_metric]
        g.set_ylabels(ylabel)
        g.set_xlabels(BATCH_SIZE_X_LABEL)
        # title = "SM efficiency: percent of SMs\nthat are in use across the entire GPU"
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        if suffix is None:
            suffix = ""
        else:
            suffix = f".{suffix}"

        save_plot(df, _j(self.args['trtexec_dir'], f'batch_size_vs_{cupti_metric}.streams_{streams}{suffix}.svg'))

    def _plot_streams_vs_throughput(self, title, batch_size, suffix=None):
        if self.trtexec_df is None:
            return
        df = copy.copy(self.trtexec_df)
        """
        WANT:
        x_field: batch_size
        y_field: metric_value
        group_field: num_threads
        """

        df = df[df['batch_size'] == batch_size]

        # df = keep_cupti_metric(df, cupti_metric)

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        g = sns.catplot(x="streams", y="host_latency_throughput_qps",
                        # data=df,
                        hue="cuda_graph", data=df,
                        # hue="num_threads", data=df,
                        # hue=col_titles["num_threads"], data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(SAMPLE_THROUGHPUT_Y_LABEL)
        g.set_xlabels(STREAMS_X_LABEL)
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        if suffix is None:
            suffix = ""
        else:
            suffix = f".{suffix}"

        save_plot(df, _j(self.args['trtexec_dir'], f'streams_vs_throughput.batch_size_{batch_size}{suffix}.svg'))

    def _plot_batch_size_vs_throughput(self, title, streams, suffix=None):
        if self.trtexec_df is None:
            return
        df = copy.copy(self.trtexec_df)
        """
        WANT:
        x_field: batch_size
        y_field: metric_value
        group_field: num_threads
        """

        df = df[df['streams'] == streams]

        # df = keep_cupti_metric(df, cupti_metric)

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="batch_size", y="host_latency_throughput_qps",
                        # data=df,
                        hue="cuda_graph", data=df,
                        # hue=col_titles["num_threads"], data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(SAMPLE_THROUGHPUT_Y_LABEL)
        g.set_xlabels(BATCH_SIZE_X_LABEL)
        # title = "SM efficiency: percent of SMs\nthat are in use across the entire GPU"
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        if suffix is None:
            suffix = ""
        else:
            suffix = f".{suffix}"

        save_plot(df, _j(self.args['trtexec_dir'], f'batch_size_vs_throughput.streams_{streams}{suffix}.svg'))

    def parse_trtexec_logs_as_df(self, logs):

        def each_field_value(log):
            for section in log:
                for attr, value in log[section].items():
                    field = f"{section}_{attr}"
                    yield field, value

        all_fields = set()
        if len(logs) > 0:
            all_fields = set([field for field, value in each_field_value(logs[0])])

        data = dict()
        for log in logs:
            for field, value in each_field_value(log):
                if field not in all_fields:
                    raise RuntimeError(f"Saw unexpected field={field}; expected one of {all_fields}")
                if field not in data:
                    data[field] = []
                data[field].append(value)

        df = pd.DataFrame(data)
        return df

    def parse_trtexec_log(self, trtexec_log_path):
        """
        {
          'host_latency': {
            'min_ms': 0.123,
            'mean_ms': 0.123,
            ...
          }
        }
        :param trtexec_log_path:
        :return:
        """
        with open(trtexec_log_path) as f:
            section = None

            data = dict()

            def strip_log_prefix(line):
                line = re.sub(r'^\[[^\]]+\]\s+\[I\]\s+', '', line)
                return line

            def as_attr(section):
                attr = section
                attr = re.sub(' ', '_', attr)
                attr = attr.lower()
                return attr

            def parse_section(line):
                m = re.search(r'(?P<section>Host Latency|GPU Compute|Enqueue Time)$', line, flags=re.IGNORECASE)
                if m:
                    section = as_attr(m.group('section'))
                    return section
                return None

            def parse_e2e_metric(line):
                # NOTE: end-to-end is the time = endOutput - startInput
                #       non-end-to-end         = (endInput + startInput) + (endCompute + startCompute) + (endOutput + startOutput)
                # So, "end-to-end" will include some time spent host-side, whereas non-end-to-end just includes time spent GPU side
                # (the transfers, the kernel running).
                m = re.search(r'(?P<name>min|max|mean|median): (?P<value>{float}) {unit} \(end to end (?P<e2e_value>{float}) (?P<unit>{unit})\)'.format(
                    float=FLOAT_RE,
                    unit=UNIT_RE), line)
                if m:
                    # Just ignore this value...
                    value = float(m.group('value'))
                    e2e_value = float(m.group('e2e_value'))
                    name = "{name}_{unit}".format(name=m.group('name'), unit=m.group('unit'))
                    name = as_attr(name)
                    return {
                        'name': name,
                        'value': e2e_value,
                    }
                return None

            def parse_metric_with_unit(line):
                m = re.search(r'(?P<name>[a-zA-Z][a-zA-Z ]+): (?P<value>{float}) (?P<unit>{unit})'.format(
                    float=FLOAT_RE,
                    unit=UNIT_RE), line)
                if m:
                    value = float(m.group('value'))
                    name = "{name}_{unit}".format(name=m.group('name'), unit=m.group('unit'))
                    name = as_attr(name)
                    return {
                        'name': name,
                        'value': value,
                    }
                return None

            def parse_percentile(line):
                m = re.search(r'(?P<name>percentile): (?P<value>{float}) (?P<unit>{unit}) at (?P<percent>\d+)%'.format(
                    float=FLOAT_RE,
                    unit=UNIT_RE), line)
                if m:
                    value = float(m.group('value'))
                    name = "{name}_{percent}_{unit}".format(
                        name=m.group('name'),
                        percent=m.group('percent'),
                        unit=m.group('unit'))
                    name = as_attr(name)
                    return {
                        'name': name,
                        'value': value,
                    }
                return None

            def parse_e2e_percentile(line):
                m = re.search(r'(?P<name>percentile): [^(]+\(end to end (?P<value>{float}) (?P<unit>{unit}) at (?P<percent>\d+)%\)'.format(
                    float=FLOAT_RE,
                    unit=UNIT_RE), line)
                if m:
                    value = float(m.group('value'))
                    name = "{name}_{percent}_{unit}".format(
                        name=m.group('name'),
                        percent=m.group('percent'),
                        unit=m.group('unit'))
                    name = as_attr(name)
                    return {
                        'name': name,
                        'value': value,
                    }
                return None

            def _add_parsed_value(dic):
                if section not in data:
                    data[section] = dict()
                data[section][dic['name']] = dic['value']

            for lineno, line in enumerate(f, start=1):
                line = line.rstrip()

                ret = parse_section(line)
                if ret:
                    section = ret
                    continue

                if section is None:
                    continue

                line = strip_log_prefix(line)

                ret = parse_e2e_metric(line)
                if ret:
                    _add_parsed_value(ret)
                    continue

                ret = parse_e2e_percentile(line)
                if ret:
                    _add_parsed_value(ret)
                    continue

                ret = parse_percentile(line)
                if ret:
                    _add_parsed_value(ret)
                    continue

                ret = parse_metric_with_unit(line)
                if ret:
                    _add_parsed_value(ret)
                    continue

                if self.debug:
                    logging.info("Skip {path}:{lineno}: {line}".format(
                        path=trtexec_log_path,
                        lineno=lineno,
                        line=line,
                    ))

            return data

    @property
    def debug(self):
        return self.args['debug']

    def _read_trtexec_df(self):
        self.trtexec_df = None
        self.trtexec_gpu_hw_df = None
        if self.args['trtexec_dir'] is None:
            return
        """
        /home/jgleeson/clone/iml/output/trtexec7/batch_size_1.streams_1.threads_no.cuda_graph_no.hw_counters_yes
        """
        trtexec_dflt_attrs = {
        }
        trtexec_attrs = {
            'batch_size',
            'streams',
            'threads',
            'cuda_graph',
            'hw_counters',
        }
        trtexec_attr_types = {
            'batch_size': maybe_number,
            'streams': maybe_number,
            'threads': yes_as_bool,
            'cuda_graph': yes_as_bool,
            'hw_counters': yes_as_bool,
        }

        dfs = []
        for path in each_file_recursive(self.args['trtexec_dir']):
            if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                continue
            sm_attrs = parse_path_attrs(
                path,
                trtexec_attrs,
                trtexec_dflt_attrs,
                trtexec_attr_types)
            df = pd.read_csv(path, comment='#')
            for attr_name, attr_value in sm_attrs.items():
                df[attr_name] = attr_value
            dfs.append(df)
        self.trtexec_gpu_hw_df = pd.concat(dfs)

        dfs = []
        for path in each_file_recursive(self.args['trtexec_dir']):
            if not re.search(r'^trtexec\.log\.txt$', _b(path)):
                continue
            sm_attrs = parse_path_attrs(
                path,
                trtexec_attrs,
                trtexec_dflt_attrs,
                trtexec_attr_types,
            )
            trt_data = self.parse_trtexec_log(path)
            df = self.parse_trtexec_logs_as_df([trt_data])
            # logging.info("TRT DATA @ {path}\n{msg}".format(
            #     path=path,
            #     msg=textwrap.indent(pprint.pformat(trt_data), prefix='  ')))
            for attr_name, attr_value in sm_attrs.items():
                df[attr_name] = attr_value
            dfs.append(df)
        self.trtexec_df = pd.concat(dfs)

        logging.info("trtexec_gpu_hw dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.trtexec_gpu_hw_df), indent=1),
        ))
        logging.info("trtexec dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.trtexec_df), indent=1),
        ))

class GpuUtilExperiment:
    def __init__(self, args):
        self.args = args

    def read_df(self):
        self._read_sm_efficiency_df()
        self._read_achieved_occupancy_df()
        self._read_rlscope_df()
        self._read_util_data()
        self._read_multithread_df()
        self._read_multiprocess_df()
        self._read_multitask_df()

    def _read_rlscope_df(self):
        self.rlscope_df = None
        if self.args['rlscope_dir'] is None:
            return
        rlscope_dflt_attrs = {
        }
        rlscope_attrs = {
            'algo',
            'env',
        }
        dfs = []
        for path in each_file_recursive(self.args['rlscope_dir']):
            if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                continue
            sm_attrs = parse_path_attrs(
                path,
                rlscope_attrs,
                rlscope_dflt_attrs)
            df = pd.read_csv(path, comment='#')
            for attr_name, attr_value in sm_attrs.items():
                assert attr_name not in df
                df[attr_name] = maybe_number(attr_value)
            dfs.append(df)
        self.rlscope_df = pd.concat(dfs)
        logging.info("rlscope dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.rlscope_df), indent=1),
        ))

    def gpu_hw_csv_paths(self, root_dir):
        paths = []
        for path in each_file_recursive(root_dir):
            if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                continue
            paths.append(path)
        return paths

    def read_csv(self, path, attrs, dflt_attrs):
        attr_dict = parse_path_attrs(
            path,
            attrs,
            dflt_attrs)
        df = pd.read_csv(path, comment='#')
        for attr_name, attr_value in attr_dict.items():
            assert attr_name not in df
            df[attr_name] = maybe_number(attr_value)
        return df

    def _plot_util_data(self):
        if self.util_data is None:
            return
        util_data = self.util_data.copy()
        util_df = util_data['util_df']
        util_df['util'] = 100*util_df['util']

        device_id = 0
        util_df = util_df[
            (util_df['device_id'] == device_id)
            & (util_df['device_type'] == 'GPU')
            ]

        def _x_label(row):
            return '\n'.join([
                f"Thread blocks = {row['thread_blocks']}",
                f"Thread block size = {row['thread_block_size']}",
            ])
        util_df = util_df.copy()
        util_df['x_label'] = util_df.apply(_x_label, axis=1)

        sns.set(style="whitegrid")
        ax = sns.boxplot(x="x_label", y="util", data=util_df,
                         # The first samples tend to be zero since they capture time before kernel starts running.
                         showfliers=False)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max)
        ax.set_ylabel('Utilization (%)')
        ax.set_xlabel('Kernel configuration')
        ax.set_title(r'$\mathtt{nvidia-smi}$ utilization')

        save_plot(util_df, _j(self.args['util_dir'], 'util.svg'))

    def _read_multi_df(self, direc):

        def read_csv(path, dflt_attrs=None):
            attr_dict = parse_path_attrs(
                path,
                MULTI_TASK_ATTRS,
                dflt_attrs)
            with open(path) as f:
                all_js = json.load(f)
            js = { attr: all_js[attr]
                    for attr in set(MULTI_TASK_JSON_ATTRS).intersection(all_js.keys()) }
            df = pd.DataFrame(js)
            for attr_name, attr_value in attr_dict.items():
                if attr_name not in df:
                    df[attr_name] = maybe_number(attr_value)
            return df

        assert direc is not None

        dfs = []
        for path in each_file_recursive(direc):
            if not re.search(r'^GPUComputeSchedInfoKernel\..*\.json$', _b(path)):
                continue
            df = read_csv(path)
            dfs.append(df)
        multi_df = pd.concat(dfs)

        return multi_df

    def _read_multitask_df(self):
        self.multitask_df = None
        if self.multithread_df is None or self.multiprocess_df is None:
            return

        multithread_df = copy.copy(self.multithread_df)
        multiprocess_df = copy.copy(self.multiprocess_df)

        def _title(task_type, task_type_plural, n_tasks, sep=' '):
            return sep.join([
                "Multi-{task}",
                "($N_{{{tasks}}}$ = {n_tasks})",
            ]).format(
                task=task_type,
                tasks=task_type_plural,
                n_tasks=n_tasks,
            )

        def _add_title(df, task_type, task_type_plural, n_tasks):
            df['expr'] = _title(task_type, task_type_plural, n_tasks, sep=' ')
            df['expr_x'] = _title(task_type, task_type_plural, n_tasks, sep='\n')

        n_threads = len(multithread_df['thread_id'].unique())
        _add_title(multithread_df, 'thread', 'threads', n_threads)

        n_procs = len(multiprocess_df['thread_id'].unique())
        _add_title(multiprocess_df, 'process', 'processes', n_procs)

        multitask_df = pd.concat([multithread_df, multiprocess_df])

        self.multitask_df = multitask_df


    def _read_multithread_df(self):
        self.multithread_df = None
        if self.args['multithread_dir'] is None:
            return
        self.multithread_df = self._read_multi_df(self.args['multithread_dir'])

    def _read_multiprocess_df(self):
        self.multiprocess_df = None
        if self.args['multiprocess_dir'] is None:
            return
        self.multiprocess_df = self._read_multi_df(self.args['multiprocess_dir'])

    def _read_util_data(self):
        self.util_data = None
        if self.args['util_dir'] is None:
            return

        util_dflt_attrs = None
        util_attrs = GPU_UTIL_EXPERIMENT_ATTRS

        gpu_hw_csv_paths = self.gpu_hw_csv_paths(self.args['util_dir'])
        assert len(gpu_hw_csv_paths) == 1
        gpu_hw_csv_path = gpu_hw_csv_paths[0]
        gpu_hw_df = self.read_csv(
            gpu_hw_csv_path,
            util_attrs,
            util_dflt_attrs)

        thread_blocks = gpu_hw_df['thread_blocks'].unique()
        assert len(thread_blocks) == 1
        thread_blocks = thread_blocks[0]

        thread_block_size = gpu_hw_df['thread_block_size'].unique()
        assert len(thread_block_size) == 1
        thread_block_size = thread_block_size[0]

        util_df_reader = UtilDataframeReader(
            self.args['util_dir'],
            debug=self.debug)
        util_df = util_df_reader.read()
        util_df = util_df.sort_values(['machine_name', 'device_name', 'start_time_us'])
        util_df['start_time_sec'] = util_df['start_time_us']/MICROSECONDS_IN_SECOND
        util_df['thread_block_size'] = thread_block_size
        util_df['thread_blocks'] = thread_blocks

        self.util_data = dict(
            util_df=util_df,
            gpu_hw_df=gpu_hw_df,
        )

    @property
    def debug(self):
        return self.args['debug']


    def _read_sm_efficiency_df(self):
        self.sm_df = None
        if self.args['sm_efficiency_dir'] is None:
            return
        sm_efficiency_dflt_attrs = {
            'num_threads' : 1,
        }
        sm_efficiency_attrs = {
            'num_threads',
            'thread_blocks',
            'thread_block_size',
        }
        dfs = []
        for path in each_file_recursive(self.args['sm_efficiency_dir']):
            if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                continue
            # sm_attrs = parse_filename_attrs(path, 'GPUHwCounterSampler', 'csv', sm_efficiency_attrs, sm_efficiency_dflt_attrs)
            sm_attrs = parse_path_attrs(
                path,
                sm_efficiency_attrs,
                sm_efficiency_dflt_attrs)
            df = pd.read_csv(path, comment='#')
            for attr_name, attr_value in sm_attrs.items():
                assert attr_name not in df
                df[attr_name] = maybe_number(attr_value)
            dfs.append(df)
        self.sm_df = pd.concat(dfs)
        logging.info("sm_efficiency dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.sm_df), indent=1),
        ))

    def _read_achieved_occupancy_df(self):
        self.occupancy_df = None
        if self.args['achieved_occupancy_dir'] is None:
            return
        achieved_occupancy_dflt_attrs = {
            'num_threads' : 1,
        }
        achieved_occupancy_attrs = {
            'num_threads',
            'thread_blocks',
            'thread_block_size',
        }
        dfs = []
        for path in each_file_recursive(self.args['achieved_occupancy_dir']):
            if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                continue
            # sm_attrs = parse_filename_attrs(path, 'GPUHwCounterSampler', 'csv', achieved_occupancy_attrs, achieved_occupancy_dflt_attrs)
            sm_attrs = parse_path_attrs(
                path,
                achieved_occupancy_attrs,
                achieved_occupancy_dflt_attrs)
            df = pd.read_csv(path, comment='#')
            for attr_name, attr_value in sm_attrs.items():
                assert attr_name not in df
                df[attr_name] = maybe_number(attr_value)
            dfs.append(df)
        self.occupancy_df = pd.concat(dfs)
        logging.info("achieved_occupancy dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.occupancy_df), indent=1),
        ))

    def plot_df(self):
        self._plot_sm_efficiency()
        self._plot_achieved_occupancy()
        self._plot_rlscope_sm_efficiency()
        self._plot_rlscope_achieved_occupancy()
        self._plot_util_data()
        self._plot_multitask_sched_info()
        self._plot_multitask_sm_efficiency()

    def _pretty_algo(self, algo):
        return algo.upper()

    def _pretty_env(self, env):
        return env.upper()

    def _plot_rlscope_sm_efficiency(self):
        if self.rlscope_df is None:
            return
        df = copy.copy(self.rlscope_df)
        df = keep_cupti_metric(df, 'sm_efficiency')

        df = self._add_algo_env(df)

        titled_df = copy.copy(df)
        col_titles = {
            'range_name': 'Operation',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        g = sns.catplot(x="algo_env", y="metric_value", hue=col_titles["range_name"], data=titled_df,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(SM_EFFICIENCY_Y_LABEL)
        g.set_xlabels(RLSCOPE_X_LABEL)
        title = SM_EFFICIENCY_TITLE
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=15,
        )

        save_plot(df, _j(self.args['rlscope_dir'], 'rlscope_sm_efficiency.svg'))

    def _add_algo_env(self, df):
        def _algo_env(row):
            return "({algo}, {env})".format(
                algo=get_x_algo(row['algo']),
                env=get_x_env(row['env']),
            )
        df['algo_env'] = df.apply(_algo_env, axis=1)
        return df

    def _plot_rlscope_achieved_occupancy(self):
        if self.rlscope_df is None:
            return
        df = copy.copy(self.rlscope_df)
        df = keep_cupti_metric(df, 'achieved_occupancy')

        df = self._add_algo_env(df)

        titled_df = copy.copy(df)
        col_titles = {
            'range_name': 'Operation',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        g = sns.catplot(x="algo_env", y="metric_value", hue=col_titles["range_name"], data=titled_df,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(SM_OCCUPANCY_Y_LABEL)
        g.set_xlabels(RLSCOPE_X_LABEL)
        title = SM_OCCUPANCY_TITLE
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=15,
        )

        save_plot(df, _j(self.args['rlscope_dir'], 'rlscope_achieved_occupancy.svg'))

    def _plot_achieved_occupancy(self):
        if self.occupancy_df is None:
            return
        df = copy.copy(self.occupancy_df)

        df = keep_cupti_metric(df, 'achieved_occupancy')

        titled_df = copy.copy(df)
        col_titles = {
            'num_threads': 'Number of threads',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="thread_block_size", y="metric_value",
                        # hue=col_titles["num_threads"],
                        data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(SM_OCCUPANCY_Y_LABEL)
        g.set_xlabels("Number of threads per block in kernel launch")
        title = SM_OCCUPANCY_TITLE
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=45,
        )

        save_plot(df, _j(self.args['achieved_occupancy_dir'], 'achieved_occupancy.svg'))

    def _plot_sm_efficiency(self):
        if self.sm_df is None:
            return
        df = copy.copy(self.sm_df)
        """
        WANT:
        x_field: thread_blocks
        y_field: metric_value
        group_field: num_threads
        """

        df = keep_cupti_metric(df, 'sm_efficiency')

        titled_df = copy.copy(df)
        col_titles = {
            'num_threads': 'Number of threads',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="thread_blocks", y="metric_value", hue=col_titles["num_threads"], data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(SM_EFFICIENCY_Y_LABEL)
        g.set_xlabels("Number of thread blocks in kernel launch")
        title = "SM efficiency: percent of SMs\nthat are in use across the entire GPU"
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        save_plot(df, _j(self.args['sm_efficiency_dir'], 'sm_efficiency.svg'))

    def _plot_multitask_sm_efficiency(self):
        if self.multitask_df is None:
            return

        sched_df = self.multitask_df.copy()
        """
        x = sm_id
        hue = expr
        y:
          for sm_id in all available sm_ids:
            y = len(df[df['sm_id'] == sm_id]['thread_id'].unique())
        """
        plot_df = sched_df[['sm_id', 'expr_x']].drop_duplicates().reset_index()
        del plot_df['index']

        # Max num of SM ids this configuration could possibly use.
        # == num_threads
        max_num_sm_choices = sched_df['num_threads'].unique()
        assert len(max_num_sm_choices) == 1
        max_num_sm = max_num_sm_choices[0]
        assert max_num_sm <= NUM_SMS

        def _sm_efficiency(row):
            sm_ids = sched_df[(sched_df['expr_x'] == row['expr_x'])] \
                ['sm_id'].unique()
            ret = 100 * len(sm_ids) / float(max_num_sm)
            return ret
        plot_df['sm_efficiency'] = plot_df.apply(_sm_efficiency, axis=1)

        sns.set(style="whitegrid")
        g = sns.catplot(x="expr_x", y="sm_efficiency",
                        kind='bar',
                        data=plot_df)
        # x_min, x_max = ax.get_xlim()
        # -1 so we can see markers at x=0 without them being cut off.
        # ax.set_xlim(-1, NUM_SMS)
        g.despine(left=True)
        g.set_xlabels('Configuration')
        g.set_ylabels(f"SM efficiency (%)\n(normalized by max # SMs = {max_num_sm})")
        g.fig.suptitle(SM_EFFICIENCY_TITLE)
        g.fig.subplots_adjust(top=0.90)

        save_plot(plot_df, _j(_d(self.args['multithread_dir']), 'multitask_sm_efficiency.svg'))

    def _plot_multitask_sched_info(self):
        if self.multitask_df is None:
            return

        sched_df = self.multitask_df.copy()
        """
        x = sm_id
        hue = expr
        y:
          for sm_id in all available sm_ids:
            y = len(df[df['sm_id'] == sm_id]['thread_id'].unique())
        """
        plot_df = sched_df[['sm_id', 'expr']].drop_duplicates().reset_index()
        del plot_df['index']

        def _num_threads_using_sm(row):
            thread_ids = sched_df[ \
                (sched_df['sm_id'] == row['sm_id']) \
                & (sched_df['expr'] == row['expr'])] \
                ['thread_id'].unique()
            ret = len(thread_ids)
            return ret
        plot_df['num_threads_using_sm'] = plot_df.apply(_num_threads_using_sm, axis=1)

        titled_df = copy.copy(plot_df)
        col_titles = {
            'expr': 'Configuration',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # ax = sns.scatterplot(x="sm_id", y="num_threads_using_sm", hue=col_titles["expr"],
        ax = sns.lineplot(x="sm_id", y="num_threads_using_sm", hue=col_titles["expr"],
                             markers=True,
                             style=col_titles["expr"],
                             data=titled_df,
                             markersize=8,
                             # s=4,
                             )
        x_min, x_max = ax.get_xlim()
        # -1 so we can see markers at x=0 without them being cut off.
        ax.set_xlim(-2, NUM_SMS)
        ax.set_ylabel('Number of CPU workers using SM')
        ax.set_xlabel(SM_ID_X_LABEL)
        ax.set_title('Fine-grained SM scheduling of\nmulti-process and multi-thread configurations')
        plt.gcf().subplots_adjust(top=0.90)

        # import ipdb; ipdb.set_trace()
        # Q: How to select the "multi-process" line?
        # IDEA: same list index as legend title?
        # ax.lines[3].set_markersize(16)

        # HACK: the "Multi-process" line only has 1 point (x=sm_id=0, y=num_threads_using_sm=60) making it easy to miss...
        # Make it stand out by increasing the markersize of the single point.
        larger_markersize = 14
        should_set_markersize = set([
           # 0, # multi-thread line
           1, # multi-process line
        ])
        for i, line in enumerate(ax.get_lines()):
            if i in should_set_markersize:
                logging.info(f"line[{i}].set_markersize({larger_markersize})")
                line.set_markersize(larger_markersize)

        save_plot(plot_df, _j(_d(self.args['multithread_dir']), 'multitask_sched_info.svg'))

    ##
    ## PROBLEM: Plot is unplottable...way too many "hue"s to plot (60 threads for EACH of the 68 SMs...)
    ##
    # def _plot_multitask_per_sm_efficiency(self, direc, df, task_type):
    #     assert df is not None
    #     # if self.sm_df is None:
    #     #     return
    #     df = copy.copy(df)
    #     # df = keep_cupti_metric(df, 'sm_efficiency')
    #
    #     """
    #     x   = "SM", sm_id, [0..67]
    #     y   = "SM efficiency (%)", sm_efficiency, [0..1.0]
    #     hue = "CPU thread", thread_id, [0..num_threads=60]
    #
    #       groupby_samples = Groupby samples with the same thread_id:
    #
    #     samples = all the (thread_id, sm_id) samples taken over the full program execution
    #
    #     To obtain y:
    #         for thread_id, sm_id in samples.unique([thread_id, sm_id]):
    #             sm_efficiency[thread_id][sm_id] = 100 * (len(samples[sm_id == sm_id][thread_id == thread_id]) /
    #                                                      len(samples[thread_id == thread_id]))
    #     """
    #     # samples_df = df[['thread_id', 'sm_id']]
    #     # NOTE: drop_duplicates has the index set to 0 for all rows...not sure why.
    #     plot_df = df[['thread_id', 'sm_id']].drop_duplicates().copy().reset_index()
    #     del plot_df['index']
    #
    #     def _sm_efficiency(row):
    #         return 100 * ( len(df[(df['thread_id'] == row['thread_id']) &
    #                               (df['sm_id'] == row['sm_id'])]) /
    #                        len(df[(df['thread_id'] == row['thread_id'])])
    #                        )
    #     plot_df['sm_efficiency'] = plot_df.apply(_sm_efficiency, axis=1)
    #     plot_df.sort_values(['thread_id', 'sm_id'], inplace=True)
    #
    #     sns.set(style="whitegrid")
    #     import ipdb; ipdb.set_trace()
    #
    #     g = sns.catplot(x="sm_id", y="sm_efficiency", hue="thread_id", data=plot_df,
    #                     # height=6,
    #                     kind="bar",
    #                     palette="muted"
    #                     )
    #
    #     g.despine(left=True)
    #     g.set_ylabels("SM efficiency (%)")
    #     g.set_xlabels(SM_ID_X_LABEL)
    #     title = \
    #         '\n'.join([
    #             "Per SM efficiency: percent of SMs in use",
    #             r"by each parallel $\textbf{{{task}}}$ across the entire GPU"
    #         ]).format(
    #             task=task_type,
    #         )
    #     g.fig.suptitle(title)
    #     g.fig.subplots_adjust(top=0.90)
    #
    #     save_plot(df, _j(direc, 'per_sm_efficiency.svg'))
    #
    # def _plot_multithread_per_sm_efficiency(self):
    #     if self.multithread_df is None:
    #         return
    #     self._plot_multitask_per_sm_efficiency(self.args['multithread_dir'], self.multithread_df, 'thread')
    #
    # def _plot_multiprocess_per_sm_efficiency(self):
    #     if self.multiprocess_df is None:
    #         return
    #     self._plot_multitask_per_sm_efficiency(self.args['multiprocess_dir'], self.multiprocess_df, 'process')


    def run(self):
        self.read_df()
        self.plot_df()


def test_plot_grouped_bar():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
    # if self.width is not None and self.height is not None:
    #     figsize = (self.width, self.height)
    #     logging.info("Setting figsize = {fig}".format(fig=figsize))
    #     # sns.set_context({"figure.figsize": figsize})
    # else:
    # figsize = None
    # fig = plt.figure(figsize=figsize)
    def _plot(plot_path):
        speed = [0.1, 17.5, 40, 48, 52, 69, 88]
        lifespan = [2, 8, 70, 1.5, 25, 12, 28]
        animal = ['snail', 'pig', 'elephant',
                  'rabbit', 'giraffe', 'coyote', 'horse']
        # ax = fig.axes()
        df = pd.DataFrame({
            'animal': animal,
            'speed': speed,
            'lifespan': lifespan})
        df = df.set_index(['animal'])
        ax = df.plot.bar(rot=0)
        save_plot(plot_path)
        # plt.show()

    _plot('./test_plot_grouped_bar.01.svg')
    _plot('./test_plot_grouped_bar.02.svg')

from iml_profiler.profiler import iml_logging
def main():
    iml_logging.setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-plot-grouped-bar', action='store_true')
    parser.add_argument('--plot-type', choices=['gpu_util_experiment', 'trtexec'])
    parser.add_argument('--debug', action='store_true')
    args, argv = parser.parse_known_args()
    logging.info(pprint_msg({'argv': argv}))

    all_args = copy.deepcopy(vars(args))
    def _main():
        if args.test_plot_grouped_bar:
            test_plot_grouped_bar()
            return

        if args.plot_type == 'gpu_util_experiment':
            sub_parser = argparse.ArgumentParser()
            sub_parser.add_argument('--sm-efficiency-dir')
            sub_parser.add_argument('--achieved-occupancy-dir')
            sub_parser.add_argument('--rlscope-dir')
            sub_parser.add_argument('--util-dir')
            sub_parser.add_argument('--multithread-dir')
            sub_parser.add_argument('--multiprocess-dir')
            sub_args, sub_argv = sub_parser.parse_known_args(argv)
            logging.info(pprint_msg({'sub_argv': sub_argv}))

            all_args.update(vars(sub_args))
            plot = GpuUtilExperiment(args=all_args)
            plot.run()
        if args.plot_type == 'trtexec':
            sub_parser = argparse.ArgumentParser()
            sub_parser.add_argument('--trtexec-dir', required=True)
            sub_args, sub_argv = sub_parser.parse_known_args(argv)
            logging.info(pprint_msg({'sub_argv': sub_argv}))

            all_args.update(vars(sub_args))
            plot = TrtexecExperiment(args=all_args)
            plot.run()
        else:
            parser.error("Need --plot-type")

    try:
        _main()
    except Exception as e:
        if not args.debug or type(e) == bdb.BdbQuit:
            raise
        print(f"> Detected exception ({type(e).__name__}):")
        traceback.print_exc()
        print("> Entering pdb:")
        import ipdb
        ipdb.post_mortem()
        raise


def save_plot(df, plot_path, tee=True):
    dataframe_txt_path = re.sub(r'\.\w+$', '.dataframe.txt', plot_path)
    assert dataframe_txt_path != plot_path

    dataframe_csv_path = re.sub(r'\.\w+$', '.dataframe.csv', plot_path)
    assert dataframe_csv_path != plot_path

    with open(dataframe_txt_path, 'w') as f:
        f.write(DataFrame.dataframe_string(df))
    if tee:
        logging.info("{plot_path} dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(df), indent=1),
            plot_path=plot_path,
        ))

    df.to_csv(dataframe_csv_path, index=False)

    logging.info("Output plot @ {path}".format(path=plot_path))
    plt.savefig(
        plot_path,
        bbox_inches='tight',
        pad_inches=0)
    plt.close()

def keep_cupti_metric(df, cupti_metric_name):
    prof_metric_name = METRIC_NAME_CUPTI_TO_PROF[cupti_metric_name]
    prof_metric_name_re = re.compile(re.escape(prof_metric_name))
    def _is_metric(metric_name):
        # Ignore the weird +/& symbols
        ret = bool(re.search(prof_metric_name_re, metric_name))
        return ret
        # return metric_name == prof_metric_name
    df = df[df['metric_name'].apply(_is_metric)].copy()
    df['cupti_metric_name'] = cupti_metric_name
    return df


if __name__ == '__main__':
    main()
