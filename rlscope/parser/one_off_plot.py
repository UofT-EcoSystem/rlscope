"""
Plotting code for GPU hardware metrics (i.e., SM occupancy, SM efficiency),
and miscellaneous experiments with GPU utilization.
"""
from rlscope.profiler.rlscope_logging import logger
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

from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib
import matplotlib.ticker
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from rlscope.profiler.util import pprint_msg
from rlscope.parser.stacked_bar_plots import get_x_env, get_x_algo, xfields_from_xtick_expression, get_capsize, OverlapStackedBarPlot, add_repetition, group_numeric_cols
from rlscope.parser.dataframe import UtilDataframeReader, RLScopeConfig

from rlscope import py_config
from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.parser.plot_utils import is_pdf, pdf2png
from rlscope.py_config import yes_as_bool

from typing import *

class IMLInvaidArgument(Exception):
    pass


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

def parse_path_attrs(
    path : str,
    attrs : Iterable[str],
    dflt_attrs : Optional[Dict[str, Any]] = None,
    attr_types : Optional[Dict[str, Any]] = None,
    debug : bool = False,
    ):

    attr_name_regex = r'(?:{regex})'.format(
        regex='|'.join(sorted(attrs, key=lambda attr: (-1*len(attr), attr)))
    )

    attr_string_regex = r'(?P<attr_name>{attr_name})_(?P<attr_value>[^\.]*)\b'.format(
        attr_name=attr_name_regex
    )
    # e.g.
    # path = 'GPUHwCounterSampler.thread_blocks_68.thread_block_size_1024.csv'

    if debug:
        logger.info(f"attr_name_regex = {attr_name_regex}")

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
            Couldn't find all required attributes in {path}.
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
PROF_TO_METRIC_NAME_CUPTI = dict((v, k) for k, v in METRIC_NAME_CUPTI_TO_PROF.items())

# HACK: number of total SMs on the RTX 2080 GPU on the "eco" cluster machines
NUM_SMS = 68

SM_OCCUPANCY_TITLE = "SM occupancy: average percent of warps\nthat are in use within an SM"
SM_EFFICIENCY_TITLE = "SM efficiency: percent of SMs\nthat are in use across the entire GPU"

SM_EFFICIENCY_Y_LABEL = f"SM efficiency (%)\n# SMs = {NUM_SMS}"
SM_OCCUPANCY_Y_LABEL = "SM occupancy (%)\nmax threads per block = 1024"
SAMPLE_THROUGHPUT_Y_LABEL = "Throughput (samples/second)"
SAMPLE_LATENCY_Y_LABEL = "Minibatch latency (ms)"
CUPTI_METRIC_Y_LABEL = {
    'sm_efficiency': SM_EFFICIENCY_Y_LABEL,
    'achieved_occupancy': SM_OCCUPANCY_Y_LABEL,
}
CUPTI_METRIC_Y_LABEL_SHORT = {
    'sm_efficiency': "SM efficiency (%)",
    'achieved_occupancy': "SM occupancy (%)",
}
TRT_METRIC_YLABELS = {
    'host_latency_throughput_qps': SAMPLE_THROUGHPUT_Y_LABEL,
    'gpu_compute_mean_ms': "Mean GPU compute time (ms)",
    'gpu_compute_percentile_99_ms': "99%-tile GPU compute time (ms)",
}


BATCH_SIZE_X_LABEL = "Batch size"
STREAMS_X_LABEL = "# of CUDA streams"

SIMULATOR_X_LABEL = "Simulator"
STEP_THROUGHPUT_Y_LABEL = "Simulation throughput (samples/sec)"
STEP_LATENCY_Y_LABEL = "Simulation latency (ms)"

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
GPU_UTIL_EXPERIMENT_ATTR_TYPES = {
    'thread_blocks': maybe_number,
    'thread_block_size': maybe_number,
    'n_launches': maybe_number,
    'iterations': maybe_number,
    'num_threads': maybe_number,
    'processes': yes_as_bool,
    'hw_counters': yes_as_bool,
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
MULTI_TASK_ATTR_TYPES = dict(GPU_UTIL_EXPERIMENT_ATTR_TYPES)
MULTI_TASK_ATTR_TYPES.update({
    ## From directory attrs
    # 'thread_blocks',
    # 'thread_block_size',
    # 'n_launches',
    # 'iterations',
    # 'num_threads',
    'iterations_per_sched_sample': maybe_number,
    # 'processes',
    # 'hw_counters',

    ## GPUComputeSchedInfoKernel.thread_id_9.stream_id_9.trace_id_0.json
    'thread_id': maybe_number,
    'stream_id': maybe_number,
    'trace_id': maybe_number,
})

MULTI_TASK_RAW_ATTR_TYPES = dict(MULTI_TASK_ATTR_TYPES)
MULTI_TASK_RAW_ATTR_TYPES.update({
    'num_sms': maybe_number,
    'sms_allocated': maybe_number,
    'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': maybe_number,
})
# MULTI_TASK_RAW_ATTR_DFLTS = dict(MULTI_TASK)
MULTI_TASK_RAW_ATTR_DFLTS = {
    'num_sms': None,
    'sms_allocated': None,
    'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': None,
}
MULTI_TASK_RAW_ATTRS = MULTI_TASK_ATTRS.union(MULTI_TASK_RAW_ATTR_TYPES.keys()).difference({
    'stream_id',
    'thread_id',
    'trace_id',
})
# suffix=".num_sms_${NUM_SMS}.sms_allocated_${sms_allocated}.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE_${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE}"

# all_cycles:
#   the metric is computed over all cycles on the GPU, including cycles where the GPU
#   is idle and not executing any kernels.
# active_cycles:
#   the metric is computed over active GPU cycles.
#   Measurement periods where the GPU is idle result in a metric value of "0".
MEASUREMENT_PERIOD_ACTIVE_CYCLES = 'active_cycles'
MEASUREMENT_PERIOD_ALL_CYCLES = 'all_cycles'
CUPTI_METRIC_MEASUREMENT_PERIOD = {
    'achieved_occupancy': MEASUREMENT_PERIOD_ACTIVE_CYCLES,
    'sm_efficiency': MEASUREMENT_PERIOD_ALL_CYCLES,
    'inst_executed': MEASUREMENT_PERIOD_ACTIVE_CYCLES,
    'active_cycles': MEASUREMENT_PERIOD_ACTIVE_CYCLES,
    'active_warps': MEASUREMENT_PERIOD_ACTIVE_CYCLES,
    'elapsed_cycles_sm': MEASUREMENT_PERIOD_ALL_CYCLES,
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
        self._read_tf_inference_df()
        self._read_simulator_df()
        self._read_mps_df()
        """
        TODO: merge trtexec_df and tf_inference_df
        trtexec_field               tf_inference_field
        host_latency_throughput_qps throughput_qps
        
        
        """

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
            def filter_tensorflow(plot_df):
                plot_df = plot_df[plot_df['config'] == 'TF']
                return plot_df
            self._plot_batch_size_vs_throughput(
                title="Throughput with increasing batch size",
                streams=streams,
                filter_df=filter_tensorflow,
                suffix=f"{or_empty(suffix)}.just_tensorflow")

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
            def _title(title):
                return f"{title}:\n(batch size = {batch_size})"
            trt_metric_title = {
                'host_latency_throughput_qps': _title("Throughput with increasing streams"),
                'gpu_compute_mean_ms': _title("Mean GPU compute time with increasing streams"),
                'gpu_compute_percentile_99_ms': _title("99-%tile GPU compute time with increasing streams"),
            }
            cuda_graph_dict = {
                'host_latency_throughput_qps': None,
                'gpu_compute_mean_ms': None,
                'gpu_compute_percentile_99_ms': None,
            }
            for trt_metric in trt_metric_title.keys():
                self._plot_streams_vs_trt_metric(
                    trt_metric, batch_size,
                    title=trt_metric_title[trt_metric],
                    cuda_graph=cuda_graph_dict.get(trt_metric, None))
            # self._plot_streams_vs_throughput(
            #     title="Throughput with increasing streams\n(batch size = {batch_size})".format(batch_size=batch_size),
            #     batch_size=batch_size,
            #     suffix=suffix)
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
        if self.trtexec_df is not None:
            """
            batch_size = (best batch size for streams == 1)
            streams = 1, 2, 3, ..., 8
            plot:
              throughput
              sm_efficiency
              sm_occupancy
            """
            best_batch_size = self._compute_best_batch_size()
            _plot_streams_vs(batch_size=best_batch_size, suffix='.best_batch_size')

        self._plot_simulator_vs_steptime()
        self._plot_simulator_vs_throughput()

        def _plot_multiprocess_inference(df, throughput_title=None, inference_title=None, filter_df=None, suffix=None):
            # if throughput_title is None:
            #     throughput_title = 'Increasing inference throughput when slicing SMs with CUDA MPS processes'
            # if inference_title is None:
            #     inference_title = 'Inference latency when slicing SMs with CUDA MPS processes'
            self._plot_mps_batch_size_vs_metric_by_num_tasks(
                df=self.mps_df,
                metric='throughput_qps',
                title=throughput_title,
                xlabel=BATCH_SIZE_X_LABEL,
                ylabel=SAMPLE_THROUGHPUT_Y_LABEL,
                filter_df=filter_df,
                suffix=suffix,
                global_ymax=True,
            )
            self._plot_mps_batch_size_vs_metric_by_num_tasks(
                df=self.mps_raw_df,
                metric='inference_time_ms',
                title=inference_title,
                xlabel=BATCH_SIZE_X_LABEL,
                ylabel=SAMPLE_LATENCY_Y_LABEL,
                filter_df=filter_df,
                suffix=suffix,
                global_ymax=False,
            )
        """
        3 different graphs for multi-process experiment:
        - Multi-process (CPU) / config_cpu
          row['cpu']
          assert not row['mps']
        - Multi-process MPS (GPU) / config_mps_gpu_evenly
          row['mps'] and row['sm_alloc_strategy'] == 'evenly' 
          assert not row['cpu']
        - Multi-process MPS (GPU) / config_mps_gpu_evenly_x2
          row['mps'] and row['sm_alloc_strategy'] == 'evenly_x2' 
          assert not row['cpu']
        - Multi-process (GPU, no MPS) / config_gpu
          not row['mps'] and not row['cpu']
        """

        def is_config_cpu(row):
            is_cpu = row['cpu']
            if is_cpu:
                assert not row['mps']
            return is_cpu

        # def is_config_mps_gpu_evenly(row):
        #     is_mps = row['mps']
        #     if is_mps:
        #         assert not row['cpu']
        #     return is_mps and row['sm_alloc_strategy'] == 'evenly'
        #
        # def is_config_mps_gpu_evenly_x2(row):
        #     is_mps = row['mps']
        #     if is_mps:
        #         assert not row['cpu']
        #     return is_mps and row['sm_alloc_strategy'] == 'evenly_x2'

        def is_config_mps_gpu(row):
            is_mps = row['mps']
            if is_mps:
                assert not row['cpu']
            return is_mps

        def is_config_gpu(row):
            return not row['mps'] and not row['cpu']

        def as_row_filter_func(is_config):
            def row_filter_func(df):
                df = df[df.apply(is_config, axis=1)]
                return df
            return row_filter_func

        # throughput_ymax = self.mps_df['']

        sm_alloc_strategies = self.mps_df[self.mps_df['mps']]['sm_alloc_strategy'].unique().tolist()
        for sm_alloc_strategy in sm_alloc_strategies:
            def _is_config(row):
                return is_config_mps_gpu(row) and row['sm_alloc_strategy'] == sm_alloc_strategy
            _plot_multiprocess_inference(
                self.mps_df,
                throughput_title='Inference throughput:\nmulti-process TF scripts (GPU) + CUDA MPS',
                inference_title='Inference latency:\nmulti-process TF scripts (GPU) + CUDA MPS',
                filter_df=as_row_filter_func(_is_config),
                suffix=f".config_mps_gpu_{sm_alloc_strategy}")
        # _plot_multiprocess_inference(self.mps_df, filter_df=as_row_filter_func(is_config_mps_gpu_evenly), suffix='.config_mps_gpu_evenly')
        # _plot_multiprocess_inference(self.mps_df, filter_df=as_row_filter_func(is_config_mps_gpu_evenly_x2), suffix='.config_mps_gpu_evenly_x2')
        _plot_multiprocess_inference(
            self.mps_df,
            throughput_title='Inference throughput:\nmulti-process TF scripts (CPU)',
            inference_title='Inference latency:\nmulti-process TF scripts (CPU)',
            filter_df=as_row_filter_func(is_config_cpu),
            suffix='.config_cpu')
        _plot_multiprocess_inference(
            self.mps_df,
            throughput_title='Inference throughput:\nmulti-process TF scripts (GPU)',
            inference_title='Inference latency:\nmulti-process TF scripts (GPU)',
            filter_df=as_row_filter_func(is_config_gpu),
            suffix='.config_gpu')

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
        add_gpu_hw_fields(df)
        df = self._add_config(df, df_type='trtexec')

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="streams", y="metric_value",
                        hue="config",
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

        save_plot(df, _j(self.args['trtexec_dir'], f'streams_vs_{cupti_metric}.batch_size_{batch_size}{suffix}.svg'))

    def _plot_batch_size_vs_metric(self, title, cupti_metric, streams, ylabel=None, suffix=None):
        if self.trtexec_gpu_hw_df is None:
            return
        """
        WANT:
        x_field: batch_size
        y_field: metric_value
        group_field: num_threads
        """

        plot_df = pd.DataFrame(columns=['batch_size', 'metric_value', 'config'])

        if self.trtexec_gpu_hw_df is not None:
            df = copy.copy(self.trtexec_gpu_hw_df)
            df = df[df['streams'] == streams]
            df = keep_cupti_metric(df, cupti_metric)
            add_gpu_hw_fields(df)
            df = self._add_config(df, df_type='trtexec')
            plot_df = plot_df.append(df[plot_df.columns])

        if self.tf_inference_gpu_hw_df is not None:
            df = copy.copy(self.tf_inference_gpu_hw_df)
            df = df[df['range_name'] == 'inference_loop/inference']
            df = keep_cupti_metric(df, cupti_metric)
            add_gpu_hw_fields(df)
            df = self._add_config(df, df_type='tf_inference')
            plot_df = plot_df.append(df[plot_df.columns])

        plot_df.sort_values(by=['config', 'batch_size'], inplace=True)

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="batch_size", y="metric_value",
                        hue="config",
                        data=plot_df,
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

        save_plot(plot_df, _j(self.args['trtexec_dir'], f'batch_size_vs_{cupti_metric}.streams_{streams}{suffix}.svg'))

    def _plot_streams_vs_trt_metric(self, trt_metric, batch_size, title=None, ylabel=None, alias=None, cuda_graph=None, suffix=None):
        if self.trtexec_df is None:
            return
        if alias is None:
            alias = trt_metric
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

        df = self._add_config(df, df_type='trtexec')

        sns.set(style="whitegrid")
        plot_kwargs = dict(
            x="streams",
            y=trt_metric,
            kind="bar",
            palette="muted",
        )
        if cuda_graph is None:
            plot_kwargs.update(dict(
                hue="config",
            ))
        elif cuda_graph:
            df = df[df['cuda_graph']]
        else:
            df = df[~ df['cuda_graph']]
        plot_kwargs.update(dict(
            data=df,
        ))
        g = sns.catplot(**plot_kwargs)
        g.despine(left=True)
        if ylabel is None:
            ylabel = TRT_METRIC_YLABELS[trt_metric]
        g.set_ylabels(ylabel)
        # if xlabel is not None:
        g.set_xlabels(STREAMS_X_LABEL)
        if title is not None:
            g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        ss = StringIO()

        if cuda_graph is None:
            pass
        elif cuda_graph:
            ss.write(f".cuda_graph_yes")
        else:
            ss.write(f".cuda_graph_no")

        if suffix is not None:
            ss.write(f".{suffix}")

        ss = ss.getvalue()

        save_plot(df, _j(self.args['trtexec_dir'], f'streams_vs_{alias}.batch_size_{batch_size}{ss}.svg'))

    def _plot_mps_batch_size_vs_metric_by_num_tasks(self, df, metric, title=None, xlabel=None, ylabel=None, filter_df=None, suffix=None, global_ymax=False):
        """
        Throughput graph:
            Y-axis = throughput
            X-axis (major) = batch-size (larger impact on throughput)
            X-axis (minor) = num_tasks (lesser impact on throughput)

        Latency graph:
            Y-axis = latency samples (mean/std across all processes)
            X-axis (major) = batch-size (larger impact on latency)
            X-axis (minor) = num_tasks (lesser impact on latency)
        """
        if df is None:
            return
        df = copy.copy(df)
        assert metric in df

        # df = self._add_config(df, df_type='trtexec')

        global_df = df
        if filter_df is not None:
            df = filter_df(df)

        sns.set(style="whitegrid")
        g = sns.catplot(x="batch_size",
                        y=metric,
                        # data=df,
                        hue="config",
                        data=df,
                        # hue="num_threads", data=df,
                        # hue=col_titles["num_threads"], data=titled_df,
                        # height=6,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        if ylabel is not None:
            g.set_ylabels(ylabel)
        if xlabel is not None:
            g.set_xlabels(xlabel)
        if title is not None:
            g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)

        if global_ymax:
            new_ymax = global_df[metric].max()
            ymin, ymax = g.ax.get_ylim()
            g.ax.set_ylim((ymin, max(ymax, new_ymax)))


        if suffix is None:
            suffix = ""

        save_plot(df, _j(self.args['mps_dir'], f'mps_batch_size_vs_{metric}_by_num_tasks{suffix}.svg'))


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

        df = self._add_config(df, df_type='trtexec')

        sns.set(style="whitegrid")
        g = sns.catplot(x="streams", y="host_latency_throughput_qps",
                        # data=df,
                        hue="config", data=df,
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

        save_plot(df, _j(self.args['trtexec_dir'], f'streams_vs_throughput.batch_size_{batch_size}{suffix}.svg'))

    def _add_config(self, df, df_type):
        assert df_type in {'trtexec', 'tf_inference'}

        if df_type == 'trtexec':
            def _config(row):
                if row['cuda_graph']:
                    return 'TensorRT - CUDA graph ON'
                return 'TensorRT'
            df['config'] = df.apply(_config, axis=1)
        elif df_type == 'tf_inference':
            def _config(row):
                if row['xla']:
                    return 'TF - XLA ON'
                return 'TF'
            df['config'] = df.apply(_config, axis=1)
        else:
            raise NotImplementedError()
        return df

    def _plot_batch_size_vs_throughput(self, title, streams, filter_df=None, suffix=None):
        if self.trtexec_df is None:
            return
        """
        WANT:
        x_field: batch_size
        y_field: metric_value
        group_field: num_threads
        """


        plot_df = pd.DataFrame(columns=['batch_size', 'throughput_qps', 'config'])

        if self.trtexec_df is not None:
            df = copy.copy(self.trtexec_df)
            df = df[df['streams'] == streams]
            df.rename(columns={
                'host_latency_throughput_qps': 'throughput_qps',
            }, inplace=True)
            df = self._add_config(df, df_type='trtexec')
            plot_df = plot_df.append(df[plot_df.columns])

        if self.tf_inference_result_df is not None:
            df = copy.copy(self.tf_inference_result_df)
            df = self._add_config(df, df_type='tf_inference')
            plot_df = plot_df.append(df[plot_df.columns])

        plot_df.sort_values(by=['config', 'batch_size'], inplace=True)

        if filter_df is not None:
            plot_df = filter_df(plot_df)

        # df = keep_cupti_metric(df, cupti_metric)

        # titled_df = copy.copy(df)
        # col_titles = {
        #     'num_threads': 'Number of threads',
        # }
        # titled_df.rename(columns=col_titles, inplace=True)

        sns.set(style="whitegrid")
        # df = df[["thread_blocks", "metric_value", "num_threads"]]
        g = sns.catplot(x="batch_size", y="throughput_qps",
                        # data=df,
                        hue="config", data=plot_df,
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

        save_plot(plot_df, _j(self.args['trtexec_dir'], f'batch_size_vs_throughput.streams_{streams}{suffix}.svg'))

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
                    logger.info("Skip {path}:{lineno}: {line}".format(
                        path=trtexec_log_path,
                        lineno=lineno,
                        line=line,
                    ))

            return data

    @property
    def debug(self):
        return self.args['debug']

    def _read_mps_df(self):
        self.mps_df = None
        self.mps_raw_df = None
        if self.args['mps_dir'] is None:
            return
        """
        /home/jgleeson/clone/rlscope/output/microbench_inference_multiprocess/batch_size_128.num_tasks_1.env_id_BreakoutNoFrameskip-v4.num_sms_68.sms_allocated_68.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE_100.0
        
        
        """
        mps_dflt_attrs = {
            'num_sms': None,
            'sms_allocated': None,
            'sm_alloc_strategy': None,
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': None,
        }
        mps_attr_types = {
            'mps': yes_as_bool,
            'cpu': yes_as_bool,
            'batch_size': maybe_number,
            'num_tasks': maybe_number,
            'env_id': str,
            'num_sms': maybe_number,
            'sms_allocated': maybe_number,
            'sm_alloc_strategy': str,
            'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE': maybe_number,
        }
        mps_attrs = set(mps_attr_types.keys())

        dfs = []
        raw_dfs = []
        for path in each_file_recursive(self.args['mps_dir']):
            if not re.search(r'^mode_microbench_inference_multiprocess\.merged\.json$', _b(path)):
                continue

            js = load_json(path)
            df = pd.DataFrame(
                dict((k, [v]) for k, v in js['summary_metrics'].items())
            )
            attr_dict = parse_path_attrs(
                path,
                mps_attrs,
                mps_dflt_attrs,
                mps_attr_types,
            )
            for attr_name, attr_value in attr_dict.items():
                df[attr_name] = attr_value
            dfs.append(df)

            # Q: Should we discard outliers...?
            raw_df = pd.DataFrame(data=js['raw_samples'])
            for attr_name, attr_value in attr_dict.items():
                raw_df[attr_name] = attr_value
            raw_dfs.append(raw_df)

        self.mps_df = pd.concat(dfs)
        self.mps_raw_df = pd.concat(raw_dfs)

        def _add_config(df):
            def _config(row):
                if row['mps']:
                    assert row['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] is not None
                    return multitask_title('process MPS', 'processes', n_tasks=row['num_tasks'], sep=' ')
                assert row['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] is None
                return multitask_title('process', 'processes', n_tasks=row['num_tasks'], sep=' ')
            df['config'] = df.apply(_config, axis=1)
            return df

        def _sort(df):
            df = df.sort_values(by=['batch_size', 'num_tasks'])
            return df

        def _prepare_df(df):
            df = _add_config(df)
            df = _sort(df)
            return df

        self.mps_df = _prepare_df(self.mps_df)
        self.mps_raw_df = _prepare_df(self.mps_raw_df)
        self.mps_raw_df['inference_time_ms'] = self.mps_raw_df['inference_time_sec'] * 1000

        logger.info("mps dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.mps_df), indent=1),
        ))
        logger.info("mps_raw dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.mps_raw_df), indent=1),
        ))


    def _read_simulator_df(self):
        self.simulator_df = None
        self.simulator_raw_df = None
        if self.args['simulator_dir'] is None:
            return
        """
        /home/jgleeson/clone/rlscope/output/simulator/batch_size_8.xla_no/GPUHwCounterSampler.csv
        """
        simulator_dflt_attrs = {
        }
        simulator_attrs = {
            'env_id',
        }
        simulator_attr_types = {
            'env_id': str,
        }

        dfs = []
        raw_dfs = []
        for path in each_file_recursive(self.args['simulator_dir']):
            if not re.search(r'^mode_microbench_simulator\.json$', _b(path)):
                continue

            js = load_json(path)
            df = pd.DataFrame(
                dict((k, [v]) for k, v in js['summary_metrics'].items())
            )
            sm_attrs = parse_path_attrs(
                path,
                simulator_attrs,
                simulator_dflt_attrs,
                simulator_attr_types,
            )
            for attr_name, attr_value in sm_attrs.items():
                df[attr_name] = attr_value
            dfs.append(df)

            # Q: Should we discard outliers...?
            raw_df = pd.DataFrame(data={
                'step_time_sec': js['raw_samples']['step_time_sec']
            })
            for attr_name, attr_value in sm_attrs.items():
                raw_df[attr_name] = attr_value
            raw_dfs.append(raw_df)

        self.simulator_df = pd.concat(dfs)
        self.simulator_raw_df = pd.concat(raw_dfs)

        self.simulator_df = self._add_simulator(self.simulator_df)
        self.simulator_raw_df = self._add_simulator(self.simulator_raw_df)

        logger.info("simulator dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.simulator_df), indent=1),
        ))
        logger.info("simulator_raw dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.simulator_raw_df), indent=1),
        ))

    def _plot_simulator_vs_steptime(self):
        """
        x = simulator
        y = mean step time (seconds)

        :return:
        """
        if self.simulator_raw_df is None:
            return

        plot_df = pd.DataFrame(columns=['simulator', 'step_time_ms'])

        if self.simulator_raw_df is not None:
            df = copy.copy(self.simulator_raw_df)
            df['step_time_ms'] = df['step_time_sec'] * 1000
            plot_df = plot_df.append(df[plot_df.columns])

        plot_df.sort_values(by=['simulator', 'step_time_ms'], inplace=True)

        sns.set(style="whitegrid")
        # Boxplot?
        # CDF?
        g = sns.catplot(x="simulator", y="step_time_ms",
                        data=plot_df,
                        kind="bar",
                        palette="muted")
        g.despine(left=True)
        g.set_ylabels(STEP_LATENCY_Y_LABEL)
        g.set_xlabels(SIMULATOR_X_LABEL)
        title = "Simulation latency"
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=self.arg('rotation', 15),
        )

        save_plot(plot_df, _j(self.args['simulator_dir'], f'simulator_vs_latency.svg'))

    def arg(self, name, default=None):
        if name not in self.args or self.args[name] is None:
            return default
        return self.args[name]

    def _add_simulator(self, df):
        def _simulator(row):
            return get_x_env(row['env_id'])
        df['simulator'] = df.apply(_simulator, axis=1)
        return df

    def _plot_simulator_vs_throughput(self):
        """
        x = simulator
        y = steps per second (samples/second)

        :return:
        """
        if self.simulator_df is None:
            return

        plot_df = pd.DataFrame(columns=['simulator', 'throughput_step_per_sec'])

        if self.simulator_df is not None:
            df = copy.copy(self.simulator_df)
            plot_df = plot_df.append(df[plot_df.columns])

        plot_df.sort_values(by=['simulator', 'throughput_step_per_sec'], inplace=True)

        sns.set(style="whitegrid")
        g = sns.catplot(x="simulator", y="throughput_step_per_sec",
                        data=plot_df,
                        kind="bar",
                        palette="muted"
                        )
        g.despine(left=True)
        g.set_ylabels(STEP_THROUGHPUT_Y_LABEL)
        g.set_xlabels(SIMULATOR_X_LABEL)
        title = "Simulation throughput"
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=15,
        )

        save_plot(plot_df, _j(self.args['simulator_dir'], f'simulator_vs_throughput.svg'))

    def _read_tf_inference_df(self):
        self.tf_inference_df = None
        if self.args['tf_inference_dir'] is None:
            return
        """
        /home/jgleeson/clone/rlscope/output/tf_inference/batch_size_8.xla_no/GPUHwCounterSampler.csv
        """
        tf_inference_dflt_attrs = {
        }
        tf_inference_attrs = {
            'batch_size',
            'xla',
        }
        tf_inference_attr_types = {
            'batch_size': maybe_number,
            'xla': yes_as_bool,
        }

        dfs = []
        for path in each_file_recursive(self.args['tf_inference_dir']):
            if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                continue
            sm_attrs = parse_path_attrs(
                path,
                tf_inference_attrs,
                tf_inference_dflt_attrs,
                tf_inference_attr_types)
            df = pd.read_csv(path, comment='#')
            for attr_name, attr_value in sm_attrs.items():
                df[attr_name] = attr_value
            dfs.append(df)
        self.tf_inference_gpu_hw_df = pd.concat(dfs)

        dfs = []
        for path in each_file_recursive(self.args['tf_inference_dir']):
            if not re.search(r'^mode_microbench_inference\.json$', _b(path)):
                continue
            inference_js = load_json(path)
            df = pd.DataFrame(
                dict((k, [v]) for k, v in inference_js['summary_metrics'].items())
            )
            sm_attrs = parse_path_attrs(
                path,
                tf_inference_attrs,
                tf_inference_dflt_attrs,
                tf_inference_attr_types,
            )
            # trt_data = self.parse_tf_inference_log(path)
            # df = self.parse_tf_inference_logs_as_df([trt_data])
            # logger.info("TRT DATA @ {path}\n{msg}".format(
            #     path=path,
            #     msg=textwrap.indent(pprint.pformat(trt_data), prefix='  ')))
            for attr_name, attr_value in sm_attrs.items():
                df[attr_name] = attr_value
            dfs.append(df)
        self.tf_inference_result_df = pd.concat(dfs)

        join_attrs = sorted(tf_inference_attrs)
        self.tf_inference_df = self.tf_inference_gpu_hw_df.set_index(join_attrs).join(self.tf_inference_result_df.set_index(join_attrs))

        logger.info("tf_inference_gpu_hw dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.tf_inference_gpu_hw_df), indent=1),
        ))
        logger.info("tf_inference_result dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.tf_inference_result_df), indent=1),
        ))
        logger.info("tf_inference dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.tf_inference_df), indent=1),
        ))

    def _read_trtexec_df(self):
        self.trtexec_df = None
        self.trtexec_gpu_hw_df = None
        if self.args['trtexec_dir'] is None:
            return
        """
        /home/jgleeson/clone/rlscope/output/trtexec7/batch_size_1.streams_1.threads_no.cuda_graph_no.hw_counters_yes
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
            # logger.info("TRT DATA @ {path}\n{msg}".format(
            #     path=path,
            #     msg=textwrap.indent(pprint.pformat(trt_data), prefix='  ')))
            for attr_name, attr_value in sm_attrs.items():
                df[attr_name] = attr_value
            dfs.append(df)
        self.trtexec_df = pd.concat(dfs)

        logger.info("trtexec_gpu_hw dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.trtexec_gpu_hw_df), indent=1),
        ))
        logger.info("trtexec dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.trtexec_df), indent=1),
        ))

class CompositeOp:
    def __init__(self, add, subtract=None):
        self.add = sorted(set(add))
        if subtract is None:
            subtract = []
        self.subtract = sorted(set(subtract))

    def __repr__(self):
        return "{klass}(add={add}, subtract={subtract})".format(
            klass=self.__class__.__name__,
            add=self.add,
            subtract=self.subtract,
        )


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
        self._read_multiprocess_mps_df()
        self._read_multitask_df()
        self._read_multitask_raw_df()

    def _read_rlscope_df(self):
        self.rlscope_df = None
        self.gpu_hw_df = None
        self.op_gpu_hw_df = None

        overlap_plot = OverlapStackedBarPlot(
            rlscope_directories=self.arg('time_breakdown_dir'),
            unins_rlscope_directories=None,
            directory=None,
            overlap_type='CategoryOverlap',
            detailed=True,
            # directory=self.rlscope_output_dir(),
        )
        self.time_breakdown_df = overlap_plot.read_ins_df()
        self.time_breakdown_df = self._add_x_field(self.time_breakdown_df)
        assert 'repetition' in self.time_breakdown_df

        if self.arg('rlscope_dir') is None or self.arg('time_breakdown_dir') is None:
            return
        rlscope_attrs = {
            'algo',
            'env',
        }
        dfs = []
        for rlscope_dir in self.arg('rlscope_dir'):
            for path in each_file_recursive(rlscope_dir):
                if not re.search(r'^GPUHwCounterSampler.*\.csv$', _b(path)):
                    continue
                # rlscope_dflt_attrs = {
                #     # 'algo': None,
                #     # 'env': None,
                # }
                rlscope_config = RLScopeConfig(rlscope_dir)
                algo = rlscope_config.algo(allow_none=True)
                env = rlscope_config.env(allow_none=True)
                # if algo is not None:
                #     rlscope_dflt_attrs['algo'] = algo
                # if env is not None:
                #     rlscope_dflt_attrs['env'] = env
                # sm_attrs = parse_path_attrs(
                #     path,
                #     rlscope_attrs,
                #     rlscope_dflt_attrs)
                if is_empty_file(path):
                    # NOTE: this happens if there are no GPUHwCounterSampleProto.proto files (is_gpu_hw_file)
                    continue
                df = pd.read_csv(path, comment='#')
                df['algo'] = algo
                df['env'] = env
                # for attr_name, attr_value in sm_attrs.items():
                #     assert attr_name not in df
                #     df[attr_name] = maybe_number(attr_value)
                df['rlscope_directory'] = rlscope_dir
                dfs.append(df)
        if len(dfs) == 0:
            logger.warning("There were no gpu-hw protobuf files to parse; skipping GPU hw analysis.")
            return
        self.gpu_hw_df = pd.concat(dfs)
        add_repetition(self.gpu_hw_df)
        self.gpu_hw_df = self._add_x_field(self.gpu_hw_df)
        logger.info("gpu_hw dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.gpu_hw_df), indent=1),
        ))
        add_gpu_hw_fields(self.gpu_hw_df)

        # operations = self.operation_tree.leaf_operations()
        # def _get_operations():
        #     # range_node = RangeNode()
        #     # for range_name in self.gpu_hw_df['range_name'].unique():
        #     #     range_node.add(range_name)
        #     operation_tree = OperationTree(self.gpu_hw_df['range_name'].unique())
        #     operations = operation_tree.leaf_operations()
        #     return operations
        # operations = _get_operations()

        # def _is_operation(range_name):
        #     operation = _operation(range_name)
        #     return operation in operations
        # self.gpu_hw_df = self.gpu_hw_df[self.gpu_hw_df['range_name'].apply(_is_operation)]

        def _operation(range_name):
            components = re.split(r'/', range_name)
            return components[-1]
        self.gpu_hw_df['operation'] = self.gpu_hw_df['range_name'].apply(_operation)

        # GOAL:
        # get the "percent of total execution" that an operation makes up.
        # Group this calculation by "rlscope_directory".
        # Join it to self.rlscope_df on ["repetition", ]

        # Get "percent of total execution" that an operation makes up (for a given repetition run).

        # # NOTE: Need to join on x_field since it may be use_tf_functions vs no_use_tf_functions.
        # bdf = self.time_breakdown_df[['rlscope_directory', 'repetition', 'algo', 'env', 'x_field', 'operation', 'percent']]
        # bdf_cols = group_numeric_cols(bdf)
        # groupby_cols = sorted(set(bdf_cols.non_numeric_cols).union({'repetition'}))
        # self.op_df = bdf.groupby(groupby_cols).agg(pd.DataFrame.sum, skipna=False).reset_index()
        #
        # # Avoid conflicts; keep rlscope_directory of instrumented run.
        # op_df_copy = copy.copy(self.op_df)
        # if 'rlscope_directory' in op_df_copy:
        #     del op_df_copy['rlscope_directory']
        # join_on = ['algo', 'env', 'x_field', 'operation', 'repetition']
        # # Time-breakdown and SM metrics should be available for every (rlscope_directory, operation)
        # assert len(op_df_copy[join_on].drop_duplicates()) == len(self.gpu_hw_df[join_on].drop_duplicates())
        # self.rlscope_df = self.gpu_hw_df.merge(op_df_copy, on=join_on)

        self.op_gpu_hw_df = self.get_op_mapping_df()

        # TODO:
        # - "percent" is missing "evaluate" but it is empty in gpu_hw_df anyways.
        # - "training_loop" takes up non-zero time, BUT...
        #   - training_loop in time-breakdown includes time spent NOT in leaves.
        #   - training_loop in gpu-hw INCLUDES metrics for time spent in leaves.
        #   - hence, doing a weighted average based on training_loop percent would NOT make sense...
        #   - we want:
        #     training_loop.weight_percent = training_loop.percent + [leaf.percent for leaf in leaves(training_loop)]

        # import pdb; pdb.set_trace()
        # print("HI")

    def _parse_op_mapping(self):
        assert self.arg('op_mapping') is not None

        mapping = None
        my_locals = locals()
        exec(self.arg('op_mapping'), globals(), my_locals)
        mapping = my_locals['mapping']
        if mapping is None:
            raise IMLInvaidArgument("--mapping must be a python expression that defines a callable function mapping(algo) that returns a Dict[OpNameString -> CompositeOp(add=[OpNameString], subtract=[OpNameString])")
        self.mapping = mapping

    def _get_op_mapping(self, op_kwargs, operations):

        def check_operation(algo, op):
            if op not in operations:
                raise IMLInvaidArgument("--mapping is invalid for algo={algo}; no operation \"{op}\" was profiled; choices = {ops}".format(
                    algo=algo,
                    op=op,
                    ops=operations,
                ))

        algo = op_kwargs['algo']

        # algo_mapping = dict()
        op_mapping = self.mapping(**op_kwargs)
        # algo_mapping[algo] = self.mapping(algo)
        for new_op, composite_op in list(op_mapping.items()):
            if type(composite_op) == CompositeOp:
                for op in composite_op.add:
                    check_operation(algo, op)
                for op in composite_op.subtract:
                    check_operation(algo, op)
                common_ops = set(composite_op.add).intersection(set(composite_op.subtract))
                if len(common_ops) > 0:
                    raise IMLInvaidArgument("CompositeOp for algo={algo} has overlapping operations in \"add\" and \"subtract\": {intersect}".format(
                        intersect=common_ops,
                        algo=algo,
                    ))
            elif type(composite_op) == str:
                op = composite_op
                check_operation(algo, op)
                composite_op = CompositeOp(add=[op])
                op_mapping[new_op] = composite_op
            else:
                raise IMLInvaidArgument("mapping(**op_kwargs) function must return Dict[OpNameString -> CompositeOp], but dict value for op={op} was {type} ({value})".format(
                    type=type(composite_op).__name__,
                    value=composite_op,
                    algo=algo,
                    op=new_op,
                ))
        return op_mapping

    def get_op_mapping_df(self):
        """
        1. Get total time per operation, which INCLUDES nested operations;
           if active_cycles: total_time = has GPU (GPU, or CPU+GPU)
           if all_cycles: total_time = CPU and/or GPU
                   ○ PROBLEM: nested operations may happen in multiple places… (e.g., MCTS search), but it's wrong to add all MCTS time to the parent annotations…only parent annotations they were launched in.
                   ○ How can we know what it was launched in?
                       § For non minigo workloads it's fine.
                       § Otherwise ASSUME conflicting use of operation belongs to different phases, and that nesting is always the same within a phase.
        2. Compute total_metric = total_time * metric_value
        3. For each composite "operation" we wish to plot:
           for composite_op:
             time_left[composite_op] = 0
             adjusted_metric[composite_op] = 0
             for op in add_op[composite_op]:
               adjusted_metric[composite_op] += total_metric[op]
               time_left[composite_op] += total_time[op]
             for op in minus_op[composite_op]
               adjusted_metric[composite_op] -= total_metric[op]
               time_left[composite_op] -= total_time[op]
             metric_value[composite_op] = adjusted_metric[composite_op] / time_left[composite_op]

        def mapping(algo):
            if algo == 'ddpg':
              {
              "Backpropagation": ComposeOp(add=["training_loop"], subtract=["sample_action", "step"]),
              "Inference": ComposeOp(add=["sample_action"]) # or just "sample_action"
              "Simulator": "step",
              }
            elif ...:

        :param df:
        :return:
        """
        if self.arg('op_mapping') is None:
            return None

        self._parse_op_mapping()

        # PROBLEM: How to represent different "configurations"?
        # A: 'x_field'...should be captured by rlscope_directory.
        #
        # Need to add "phase" to work with minigo?

        def _check_missing(rlscope_df, missing_name, ops_missing):
            rlscope_directory = rlscope_df['rlscope_directory'].unique()[0]
            if len(ops_missing) > 0:
                logger.warning("Missing {missing_name} for operations={ops} for @ {path}; did you run for enough --rlscope-max-passes to see all the annotations?".format(
                    ops=ops_missing,
                    path=rlscope_directory,
                    missing_name=missing_name,
                ))

        dfs = []
        for cupti_metric_name in self.gpu_hw_df['cupti_metric_name'].unique():

            measurement_period = CUPTI_METRIC_MEASUREMENT_PERIOD[cupti_metric_name]
            def _keep_time(row):
                if measurement_period == MEASUREMENT_PERIOD_ACTIVE_CYCLES:
                    return constants.CATEGORY_GPU in row['region']
                elif measurement_period == MEASUREMENT_PERIOD_ALL_CYCLES:
                    return True
                else:
                    raise NotImplementedError(f"Not sure whether to keep time for measurement_period={measurement_period}")

            for (x_field, repetition), rlscope_df in self.time_breakdown_df.groupby(['x_field', 'repetition']):
                rlscope_gpu_hw_df = self.gpu_hw_df[
                    (self.gpu_hw_df['x_field'] == x_field)
                    & (self.gpu_hw_df['repetition'] == repetition)
                    & (self.gpu_hw_df['cupti_metric_name'] == cupti_metric_name)
                ]
                gpu_hw_operations = set(rlscope_gpu_hw_df['operation'].unique())
                time_breakdown_operations = set(rlscope_df['operation'].unique())
                ops_missing_time_breakdown = gpu_hw_operations.difference(time_breakdown_operations)
                ops_missing_gpu_hw = time_breakdown_operations.difference(gpu_hw_operations)
                _check_missing(rlscope_df, 'time-breakdown', ops_missing_time_breakdown)
                _check_missing(rlscope_df, 'gpu-hw', ops_missing_gpu_hw)
                # Should have BOTH SM metrics and time-breakdown for all operations.
                operations = gpu_hw_operations.intersection(time_breakdown_operations)

                operation_tree = OperationTree(rlscope_gpu_hw_df['range_name'].unique())

                algo = rlscope_df['algo'].unique()
                assert len(algo) == 1
                algo = algo[0]

                env = rlscope_df['env'].unique()
                assert len(env) == 1
                env = env[0]

                time_df = rlscope_df[rlscope_df.apply(_keep_time, axis=1)]

                # Sum up time over operation and child operations.
                total_time = dict()
                metric_value = dict()
                total_metric = dict()
                for op in operations:
                    all_operations = operation_tree.all_operations(op)
                    assert op in all_operations
                    def _in_all_operations(operation):
                        return operation in all_operations
                    total_time[op] = time_df[time_df['operation'].apply(_in_all_operations)]['percent'].sum()
                    assert not np.isnan(total_time[op])
                    assert len(rlscope_gpu_hw_df[rlscope_gpu_hw_df['operation'] == op]) == 1
                    metric_value[op] = rlscope_gpu_hw_df[rlscope_gpu_hw_df['operation'] == op]['metric_value'].values[0]
                    assert not np.isnan(metric_value[op])
                    assert metric_value[op] >= 0
                    assert total_time[op] >= 0
                    total_metric[op] = metric_value[op] * total_time[op]
                    assert total_metric[op] >= 0

                rlscope_directories = rlscope_df['rlscope_directory'].unique()
                assert len(rlscope_directories) == 1
                rlscope_directory = rlscope_directories[0]

                op_kwargs = {
                    'algo': algo,
                    'rlscope_directory': rlscope_directory,
                    'x_field': x_field,
                }
                op_mapping = self._get_op_mapping(op_kwargs, operations)
                time_left = dict()
                adjusted_metric = dict()
                new_metric_value = dict()
                for new_op, composite_op in op_mapping.items():
                    time_left[new_op] = 0.
                    adjusted_metric[new_op] = 0.
                    for op in composite_op.add:
                        adjusted_metric[new_op] += total_metric[op]
                        time_left[new_op] += total_time[op]
                    for op in composite_op.subtract:
                        adjusted_metric[new_op] -= total_metric[op]
                        time_left[new_op] -= total_time[op]

                    # If this fails, composite operation must be subtracting operations that aren't nested.
                    assert time_left[new_op] >= 0
                    # This can be negative apparently...BUG?  I suspect it's just noise based on error bars.
                    # assert adjusted_metric[new_op] >= 0

                    if adjusted_metric[new_op] <= 0.:
                        # e.g., achieved_occupancy for Simulator will end up being:
                        # new_metric_value[new_op] = (0 total metric) / (0 seconds)
                        # Since the metric is 0 when GPU is idle, and GPU idle means no GPU time.
                        # We want new_metric_value[new_op] = 0 for this case.
                        new_metric_value[new_op] = 0.
                    else:
                        assert time_left[new_op] != 0
                        new_metric_value[new_op] = adjusted_metric[new_op] / time_left[new_op]
                        assert not np.isnan(new_metric_value[new_op])

                # Keep existing columns from gpu_hw_df that are present in every row,
                # but use the new operation name and metric_value.
                keep_cols = sorted(set(self.gpu_hw_df.keys()).difference({
                    'operation',
                    'range_name',
                    'metric_name',
                    'metric_value',
                    'cupti_metric_name',
                }))
                new_gpu_hw = copy.copy(rlscope_gpu_hw_df[keep_cols])
                new_gpu_hw.drop_duplicates(inplace=True)
                assert len(new_gpu_hw) == 1
                df_rows = []
                for op, metric_value in new_metric_value.items():
                    row = copy.copy(new_gpu_hw)
                    row['operation'] = op
                    row['metric_value'] = metric_value
                    row['cupti_metric_name'] = cupti_metric_name
                    row['metric_name'] = METRIC_NAME_CUPTI_TO_PROF[cupti_metric_name]
                    assert set(self.gpu_hw_df.keys()).difference({'range_name'}) == set(row.keys())
                    df_rows.append(row)
                new_df = pd.concat(df_rows)
                dfs.append(new_df)

        op_gpu_hw_df = pd.concat(dfs)
        return op_gpu_hw_df



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

        save_plot(util_df, _j(self.arg('util_dir'), 'util.svg'))

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

    def _read_multitask_raw_df(self):
        # self.multitask_raw_df = None
        # self.multiprocess_raw_df = self._read_multi_raw_df()

        multitask_raw_df = None
        # for argname, argval in self.args.items():
            # if re.search(r'^multi\w+_dir$', argname):
            #     multitask_raw_df = self._append_multi_df(multitask_raw_df, self._read_multi_raw_df(self.args[argname]), 'thread', 'threads')
        multitask_raw_df = self._append_multi_df(multitask_raw_df, self._read_multi_raw_df(self.arg('multithread_dir')), 'thread', 'threads')
        multitask_raw_df = self._append_multi_df(multitask_raw_df, self._read_multi_raw_df(self.arg('multiprocess_dir')), 'process', 'processes')
        "Multi-process MPS (...)"
        multitask_raw_df = self._append_multi_df(multitask_raw_df, self._read_multi_raw_df(self.arg('multiprocess_mps_dir'), debug=True), 'process MPS', 'processes', debug=True)

        self.multitask_raw_df = multitask_raw_df

    def _append_multi_df(self, multitask_df, df, task_type, task_type_plural, debug=False):

        def _add_title(df, task_type, task_type_plural, n_tasks):
            df['expr'] = multitask_title(task_type, task_type_plural, n_tasks, sep=' ')
            df['expr_x'] = multitask_title(task_type, task_type_plural, n_tasks, sep='\n')

        # def _append(multitask_df, df, task_type, task_type_plural):
        if df is not None:
            df = copy.copy(df)
            if 'thread_id' in df.keys():
                n_threads = len(df['thread_id'].unique())
                df['num_tasks'] = n_threads
            elif 'num_threads' in df.keys():
                df['num_tasks'] = df['num_threads']
                assert len(df['num_threads'].unique()) == 1
                n_threads = df['num_threads'].unique()[0]
            else:
                raise NotImplementedError("Not sure how to obtain num_tasks; choices are {choices}".format(
                    choices=sorted(df.keys()),
                ))
            _add_title(df, task_type, task_type_plural, n_threads)
            if multitask_df is None:
                multitask_df = df
            else:
                multitask_df = multitask_df.append(df)
        return multitask_df

    def _read_multitask_df(self):
        multitask_df = None
        multitask_df = self._append_multi_df(multitask_df, self.multithread_df, 'thread', 'threads')
        multitask_df = self._append_multi_df(multitask_df, self.multiprocess_df, 'process', 'processes')
        "Multi-process MPS (...)"
        multitask_df = self._append_multi_df(multitask_df, self.multiprocess_mps_df, 'process MPS', 'processes')

        self.multitask_df = multitask_df

    def _read_multi_raw_df(self, direc, debug=False):
        if direc is None:
            return None

        raw_dfs = []
        for path in each_file_recursive(direc):
            if not re.search(r'^gpu_util_experiment\.json$', _b(path)):
                continue

            js = load_json(path)
            raw_df = pd.DataFrame(data=js['raw_samples'])

            attr_dict = parse_path_attrs(
                path,
                MULTI_TASK_RAW_ATTRS,
                MULTI_TASK_RAW_ATTR_DFLTS,
                MULTI_TASK_RAW_ATTR_TYPES,
                debug=debug)
            for attr_name, attr_value in attr_dict.items():
                raw_df[attr_name] = attr_value

            raw_dfs.append(raw_df)
        multi_raw_df = pd.concat(raw_dfs)

        return multi_raw_df

    def _read_multithread_df(self):
        self.multithread_df = None
        if self.arg('multithread_dir') is None:
            return
        self.multithread_df = self._read_multi_df(self.arg('multithread_dir'))

    def _read_multiprocess_df(self):
        self.multiprocess_df = None
        if self.arg('multiprocess_dir') is None:
            return
        self.multiprocess_df = self._read_multi_df(self.arg('multiprocess_dir'))

    def _read_multiprocess_mps_df(self):
        self.multiprocess_mps_df = None
        if self.arg('multiprocess_mps_dir') is None:
            return
        self.multiprocess_mps_df = self._read_multi_df(self.arg('multiprocess_mps_dir'))

    def _read_util_data(self):
        self.util_data = None
        if self.arg('util_dir') is None:
            return

        util_dflt_attrs = None
        util_attrs = GPU_UTIL_EXPERIMENT_ATTRS

        gpu_hw_csv_paths = self.gpu_hw_csv_paths(self.arg('util_dir'))
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
            self.arg('util_dir'),
            debug=self.debug)
        util_df = util_df_reader.read()
        util_df = util_df.sort_values(['machine_name', 'device_name', 'start_time_us'])
        util_df['start_time_sec'] = util_df['start_time_us']/constants.MICROSECONDS_IN_SECOND
        util_df['thread_block_size'] = thread_block_size
        util_df['thread_blocks'] = thread_blocks

        self.util_data = dict(
            util_df=util_df,
            gpu_hw_df=gpu_hw_df,
        )

    @property
    def debug(self):
        return self.arg('debug')

    def arg(self, name, default=None):
        if name not in self.args or self.args[name] is None:
            return default
        return self.args[name]

    def _read_sm_efficiency_df(self):
        self.sm_df = None
        if self.arg('sm_efficiency_dir') is None:
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
        for path in each_file_recursive(self.arg('sm_efficiency_dir')):
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
        logger.info("sm_efficiency dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.sm_df), indent=1),
        ))

    def _read_achieved_occupancy_df(self):
        self.occupancy_df = None
        if self.arg('achieved_occupancy_dir') is None:
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
        for path in each_file_recursive(self.arg('achieved_occupancy_dir')):
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
        logger.info("achieved_occupancy dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(self.occupancy_df), indent=1),
        ))

    def plot_df(self):
        self._plot_sm_efficiency()
        self._plot_achieved_occupancy()
        self._plot_rlscope_sm_metrics(self.gpu_hw_df, 'gpu_hw')
        self._plot_rlscope_sm_metrics(self.op_gpu_hw_df, 'op_gpu_hw')
        # self._plot_rlscope_sm_efficiency()
        # self._plot_rlscope_achieved_occupancy()
        self._plot_util_data()
        self._plot_multitask_sched_info()
        self._plot_multitask_sm_efficiency()
        self._plot_multitask_kernel_time()

    def _pretty_algo(self, algo):
        return algo.upper()

    def _pretty_env(self, env):
        return env.upper()

    def _set_x_field(self, df):
        df = copy.copy(df)
        if self.arg('xtick_expression') is None:
            df['x_field'] = df['algo_env']
            return df
        df['pretty_algo'] = df['algo'].apply(get_x_algo)
        df['short_env'] = df['env'].apply(get_x_env)
        x_fields = xfields_from_xtick_expression(df, self.arg('xtick_expression'), debug=self.debug)
        df['x_field'] = x_fields
        return df

    def _add_x_field(self, df):
        df = self._add_algo_env(df)
        df = self._set_x_field(df)
        return df

    def _plot_rlscope_sm_efficiency(self):
        if self.rlscope_df is None:
            return
        df = copy.copy(self.rlscope_df)
        df = keep_cupti_metric(df, 'sm_efficiency')
        add_gpu_hw_fields(df)

        df = self._add_x_field(df)

        titled_df = copy.copy(df)
        col_titles = {
            'range_name': 'Operation',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        plot_kwargs = dict()
        self._add_plot_kwargs(plot_kwargs)

        sns.set(style="whitegrid")
        def _do_plot(**kwargs):
            g = sns.catplot(x="x_field", y="metric_value", hue=col_titles["range_name"], data=titled_df,
                            kind="bar",
                            palette="muted",
                            **kwargs,
                            **plot_kwargs,
                            )
            return g
        g = _do_plot()
        capsize = self._get_capsize(g)
        g = _do_plot(capsize=capsize)
        g.despine(left=True)
        g.set_ylabels(SM_EFFICIENCY_Y_LABEL)
        x_title = self.arg('x_title', RLSCOPE_X_LABEL)
        g.set_xlabels(x_title)
        title = self.arg('title', SM_EFFICIENCY_TITLE)
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=self.arg('rotation', 15),
        )

        save_plot(df, _j(self.rlscope_output_dir(), 'rlscope_sm_efficiency.svg'))

    def _add_algo_env(self, df):
        df = copy.copy(df)
        def _algo_env(row):
            return "({algo}, {env})".format(
                algo=get_x_algo(row['algo']),
                env=get_x_env(row['env']),
            )
        df['algo_env'] = df.apply(_algo_env, axis=1)
        return df

    def _get_capsize(self, g):
        if type(g) == sns.FacetGrid:
            # ASSUME: bar width is the same across all plots (assumes same x-axis and groups).
            ax = g.axes[0][0]
        else:
            ax = g.ax
        return get_capsize(ax)
        # # num_bars = len(g.ax.patches)
        # bar_width = g.ax.patches[0].get_width()
        # # Sanity check.
        # assert all(patch.get_width() == bar_width for patch in g.ax.patches)
        # # Make error bar width 25% of bar width.
        # return bar_width/4

    def _add_plot_kwargs(self, plot_kwargs):
        if self.arg('width') is not None and self.arg('height') is not None:
            plot_kwargs['height'] = self.arg('height')
            plot_kwargs['aspect'] = self.arg('width')/self.arg('height')

    def _remap_df(self, orig_df):
        if self.arg('remap_df') is None:
            return orig_df

        # not_regions = [key for key in orig_df.keys() if not self._is_region(key)]

        # eval context:
        # TODO: limit locals/globals to these? Don't want to limit numpy/pandas access though.
        df = copy.copy(orig_df)
        # regions = self._regions(df)
        # new_df = df[not_regions]
        new_df = copy.copy(df)
        # algo
        # env

        df_transformation = self.arg('remap_df')
        # e.g.
        # new_df[('other',)] = df[('compute_advantage_estimates',)] +
        #                      df[('optimize_surrogate',)]
        if self.debug:
            logger.info("--remap-df:\n{trans}".format(trans=textwrap.indent(df_transformation, prefix='  ')))
        exec(df_transformation)

        # Make sure they didn't modify df; they SHOULD be modifying new_df
        # (i.e. adding regions to a "fresh" slate)
        # assert np.all(df == orig_df)
        # Assume NaN's in the same place => equality.
        assert df.equals(orig_df)

        if self.debug:
            buf = StringIO()
            DataFrame.print_df(orig_df, file=buf)
            logger.info("Old dataframe:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))

            buf = StringIO()
            DataFrame.print_df(new_df, file=buf)
            logger.info("New dataframe after --remap-df:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))

        return new_df

    def _plot_rlscope_sm_metrics(self, df, suffix):
        if df is None:
            return
        # if self.rlscope_df is None:
        #     return
        df = copy.copy(df)
        achieved_occupancy_df = keep_cupti_metric(df, 'achieved_occupancy')
        sm_efficiency_df = keep_cupti_metric(df, 'sm_efficiency')
        df = pd.concat([achieved_occupancy_df, sm_efficiency_df])

        # df = self._add_x_field(df)
        # df['x_field'] = df['x_field'].astype('category')

        titled_df = copy.copy(df)
        col_titles = {
            # 'range_name': 'Operation',
            'operation': 'Operation',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        plot_kwargs = dict()
        self._add_plot_kwargs(plot_kwargs)

        sns.set(style="whitegrid")
        logger.debug("Current sns.axes_style\n{msg}".format(
            msg=pprint_msg(sns.axes_style()),
        ))
        # {'axes.axisbelow': True,
        #  'axes.edgecolor':fig =  '.8',
        #  'axes.facecolor': 'white',
        #  'axes.grid': True,
        #  'axes.labelcolor': '.15',
        #  'axes.spines.bottom': True,
        #  'axes.spines.left': True,
        #  'axes.spines.right': True,
        #  'axes.spines.top': True,
        #  'figure.facecolor': 'white',
        #  'font.family': ['sans-serif'],
        #  'font.sans-serif': ['Arial',
        #                      'DejaVu Sans',
        #                      'Liberation Sans',
        #                      'Bitstream Vera Sans',
        #                      'sans-serif'],
        #  'grid.color': '.8',
        #  'grid.linestyle': '-',
        #  'image.cmap': 'rocket',
        #  'lines.solid_capstyle': 'round',
        #  'patch.edgecolor': 'w',
        #  'patch.force_edgecolor': True,
        #  'text.color': '.15',
        #  'xtick.bottom': False,
        #  'xtick.color': '.15',
        #  'xtick.direction': 'out',
        #  'xtick.top': False,
        #  'ytick.color': '.15',
        #  'ytick.direction': 'out',
        #  'ytick.left': False,
        #  'ytick.right': False}


        #  'xtick.color': '.15',
        #  'ytick.color': '.15',

        # plt.rcParams['figure.edgecolor'] = 'black'
        # plt.rcParams['legend.edgecolor'] = 'black'
        # plt.rcParams['patch.edgecolor'] = 'black'
        # plt.rcParams["axes.edgecolor"] = "0.15"
        # plt.rcParams["axes.linewidth"] = 1.25

        # plt.rcParams["legend.loc"] = 'upper left'

        fig = plt.figure(linewidth=2)
        def _do_plot(**kwargs):
            # sns.set(style={
            #     'figure.edgecolor': 'red',
            # })
            # sns.set_style('figure', {
            #     'figure.edgecolor': 'red',
            # })
            # with sns.axes_style({
            #     # 'axes.facecolor': 'red',
            #     'axes.edgecolor': 'red',
            #     # 'grid.color': 'red',
            #     # 'figure.edgecolor': 'red',
            # }):
            g = sns.catplot(x="x_field", y="metric_value", hue=col_titles["operation"], data=titled_df,
                            row='metric_name',
                            kind="bar",
                            palette="muted",
                            sharey=False,
                            legend=False,
                            # capsize=capsize,
                            # subplot_kws={
                            #     'figure.edgecolor': 'black',
                            #     'legend.edgecolor': 'black',
                            #     'patch.edgecolor': 'black',
                            #     "axes.edgecolor": "0.15",
                            #     "axes.linewidth": 1.25,
                            # },
                            **kwargs,
                            **plot_kwargs,
                            )
            return g
        g = _do_plot()
        capsize = self._get_capsize(g)
        g = _do_plot(capsize=capsize)
        # import pdb; pdb.set_trace()
        # g.add_legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

        # xticks = titled_df['x_field'].unique()
        # g.set_xticklabels(xticks)
        # import pdb; pdb.set_trace()

        axes = g.axes.flatten()
        metric_names = titled_df['metric_name'].unique()
        cupti_metric_to_ax = dict()
        for metric_name, ax in zip(metric_names, axes):
            metric_name = metric_name.rstrip('+')
            cupti_metric_name = PROF_TO_METRIC_NAME_CUPTI[metric_name]
            cupti_metric_to_ax[cupti_metric_name] = ax

        for metric_name, ax in zip(metric_names, axes):
            metric_name = metric_name.rstrip('+')
            cupti_metric_name = PROF_TO_METRIC_NAME_CUPTI[metric_name]
            cupti_metric_to_ax[cupti_metric_name] = ax
            ylabel = CUPTI_METRIC_Y_LABEL_SHORT[cupti_metric_name]
            ax.set_ylabel(ylabel)

        # legend_ax = cupti_metric_to_ax['sm_efficiency']
        top_ax = axes[0]
        legend_ax = top_ax

        handles, labels = legend_ax.get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc='upper right')
        # import pdb; pdb.set_trace()
        # handles, labels = top_ax.legend()

        # for ax in axes:
        #     # import pdb; pdb.set_trace()
        #     # ax.spines['top'].set_color('0.5')
        #     # ax.spines['top'].set_linewidth(2)
        #
        #     ax.spines['bottom'].set_color('0.5')
        #     ax.spines['top'].set_color(None)
        #     ax.spines['right'].set_color('0.5')
        #     ax.spines['left'].set_color(None)
        #     # ax.patch.set_facecolor('0.1')
        #
        #     # ax.margins(0.5)

        # metric_names = titled_df['metric_name'].unique()
        # for ax in axes:
        #     metric_name = metric_name.rstrip('+')
        #     cupti_metric_name = PROF_TO_METRIC_NAME_CUPTI[metric_name]
        #     ylabel = CUPTI_METRIC_Y_LABEL_SHORT[cupti_metric_name]
        #     ax.set_ylabel(ylabel)

        # import pdb; pdb.set_trace()

        g.set_titles("")

        # axes[-1].set_xticklabels()

        g.despine(left=True)

        # g.set_ylabels(SM_OCCUPANCY_Y_LABEL)

        x_title = self.arg('x_title', RLSCOPE_X_LABEL)
        g.set_xlabels(x_title)

        # title = self.arg('title', SM_OCCUPANCY_TITLE)
        # if title is not None:
        #     g.fig.suptitle(title)

        # g.fig.subplots_adjust(top=0.90)
        # g.fig.axes[0].set_xticklabels(
        #     g.fig.axes[0].get_xticklabels(),
        #     rotation=self.arg('rotation', 15),
        # )
        rotate_xticks(axes[-1], self.arg('rotation', 15))

        save_plot(df, _j(self.rlscope_output_dir(), f"rlscope_sm_metrics.{suffix}.svg"))

    def _plot_rlscope_achieved_occupancy(self):
        if self.rlscope_df is None:
            return
        df = copy.copy(self.rlscope_df)
        df = keep_cupti_metric(df, 'achieved_occupancy')

        df = self._add_x_field(df)

        titled_df = copy.copy(df)
        col_titles = {
            'range_name': 'Operation',
        }
        titled_df.rename(columns=col_titles, inplace=True)

        plot_kwargs = dict()
        self._add_plot_kwargs(plot_kwargs)

        sns.set(style="whitegrid")
        def _do_plot(**kwargs):
            g = sns.catplot(x="x_field", y="metric_value", hue=col_titles["range_name"], data=titled_df,
                            kind="bar",
                            palette="muted",
                            # capsize=capsize,
                            **kwargs,
                            **plot_kwargs,
                            )
            return g
        g = _do_plot()
        capsize = self._get_capsize(g)
        g = _do_plot(capsize=capsize)

        g.despine(left=True)
        g.set_ylabels(SM_OCCUPANCY_Y_LABEL)
        x_title = self.arg('x_title', RLSCOPE_X_LABEL)
        g.set_xlabels(x_title)
        title = self.arg('title', SM_OCCUPANCY_TITLE)
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=0.90)
        g.fig.axes[0].set_xticklabels(
            g.fig.axes[0].get_xticklabels(),
            rotation=self.arg('rotation', 15),
        )

        save_plot(df, _j(self.rlscope_output_dir(), 'rlscope_achieved_occupancy.svg'))

    def rlscope_output_dir(self):
        if self.arg('output_directory') is not None:
            return self.arg('output_directory')
        if len(self.arg('rlscope_dir')) == 1:
            return self.arg('rlscope_dir')[0]
        raise NotImplementedError("Not sure where to output rlscope plots:\n{msg}".format(
            msg=txt_indent({
                'args':self.args,
            }, indent=1)),
        )

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
            rotation=self.arg('rotation', 45),
        )

        save_plot(df, _j(self.arg('achieved_occupancy_dir'), 'achieved_occupancy.svg'))

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

        save_plot(df, _j(self.arg('sm_efficiency_dir'), 'sm_efficiency.svg'))

    def _plot_multitask_kernel_time(self):
        if self.multitask_raw_df is None:
            return

        df = self.multitask_raw_df.copy()

        sns.set(style="whitegrid")
        g = sns.catplot(x="expr_x", y="total_kernel_run_time_sec",
                        kind='bar',
                        data=df)
        # x_min, x_max = ax.get_xlim()
        # -1 so we can see markers at x=0 without them being cut off.
        # ax.set_xlim(-1, NUM_SMS)
        g.despine(left=True)
        g.set_xlabels('Configuration')
        # g.set_ylabels(f"SM efficiency (%)\n(normalized by max # SMs = {max_num_sm})")
        g.set_ylabels(f"Total kernel execution time (sec)")
        g.fig.suptitle("Total kernel execution time\nunder different multi-task configurations")
        g.fig.subplots_adjust(top=0.90)

        save_plot(df, _j(self._multitask_dir(), 'multitask_kernel_time.svg'))

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
        plot_df = sched_df[['sm_id', 'expr_x', 'num_tasks']].drop_duplicates().reset_index()
        del plot_df['index']

        # Max num of SM ids this configuration could possibly use.
        # == num_threads
        # NOTE:
        # - for MPS, max should be how many SMs are usable based on CUDA_MPS_ACTIVE_THREAD_PERCENTAGE and
        #   the number of SMs on the device:
        #   max_num_sm = math.ceil( (1/CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) * NUM_SMS )
        # - Perhaps normalize by the number of "CPU threads" the configuration uses?

        def _sm_efficiency(row):
            sm_ids = sched_df[(sched_df['expr_x'] == row['expr_x'])] \
                ['sm_id'].unique()
            ret = 100 * len(sm_ids) / row['num_tasks']
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
        g.set_ylabels(f"SM efficiency (%)\n(normalized by max # SMs = $N_{{tasks}}$)")
        g.fig.suptitle(SM_EFFICIENCY_TITLE)
        g.fig.subplots_adjust(top=0.90)

        save_plot(plot_df, _j(self._multitask_dir(), 'multitask_sm_efficiency.svg'))

    def _multitask_dir(self):

        def _directory(multitask_dir, opt):
            if self.args[opt] is not None and multitask_dir is None:
                multitask_dir = _d(self.args[opt])
            return multitask_dir

        multitask_dir = None
        multitask_dir = _directory(multitask_dir, 'multithread_dir')
        multitask_dir = _directory(multitask_dir, 'multiprocess_dir')
        multitask_dir = _directory(multitask_dir, 'multiprocess_mps_dir')
        return multitask_dir

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
                logger.info(f"line[{i}].set_markersize({larger_markersize})")
                line.set_markersize(larger_markersize)

        save_plot(plot_df, _j(self._multitask_dir(), 'multitask_sched_info.svg'))

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
    #     self._plot_multitask_per_sm_efficiency(self.arg('multithread_dir'), self.multithread_df, 'thread')
    #
    # def _plot_multiprocess_per_sm_efficiency(self):
    #     if self.multiprocess_df is None:
    #         return
    #     self._plot_multitask_per_sm_efficiency(self.arg('multiprocess_dir'), self.multiprocess_df, 'process')


    def run(self):
        self.read_df()
        self.plot_df()


def plot_grouped_bar():
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
    # if self.width is not None and self.height is not None:
    #     figsize = (self.width, self.height)
    #     logger.info("Setting figsize = {fig}".format(fig=figsize))
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

from rlscope.profiler import rlscope_logging
from rlscope.profiler.rlscope_logging import logger
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-plot-grouped-bar', action='store_true')
    parser.add_argument('--plot-type', choices=['gpu_util_experiment', 'trtexec'])
    parser.add_argument('--line-numbers', action='store_true', help=textwrap.dedent("""\
        Show line numbers and timestamps in RL-Scope logging messages.
        """))
    parser.add_argument('--debug', action='store_true')
    # default=15,
    parser.add_argument('--rotation', type=int, help='x-axis xtick rotation')
    parser.add_argument('--remap-df', help='Transform dataframe')
    parser.add_argument('--output-directory', help="where to output plots")
    args, argv = parser.parse_known_args()
    logger.info(pprint_msg({'argv': argv}))

    rlscope_logging.setup_logger(
        debug=args.debug,
        line_numbers=args.debug or args.line_numbers or py_config.is_development_mode(),
    )

    all_args = copy.deepcopy(vars(args))
    def _main():
        if args.test_plot_grouped_bar:
            plot_grouped_bar()
            return

        if args.plot_type == 'gpu_util_experiment':
            sub_parser = argparse.ArgumentParser()
            sub_parser.add_argument('--sm-efficiency-dir')
            sub_parser.add_argument('--achieved-occupancy-dir')
            sub_parser.add_argument('--rlscope-dir', action='append')
            sub_parser.add_argument('--util-dir')
            sub_parser.add_argument('--multithread-dir')
            sub_parser.add_argument('--multiprocess-dir')
            sub_parser.add_argument('--multiprocess-mps-dir')
            sub_args, sub_argv = sub_parser.parse_known_args(argv)
            logger.info(pprint_msg({'sub_argv': sub_argv}))

            all_args.update(vars(sub_args))
            plot = GpuUtilExperiment(args=all_args)
            plot.run()
        elif args.plot_type == 'trtexec':
            sub_parser = argparse.ArgumentParser()
            sub_parser.add_argument('--trtexec-dir')
            sub_parser.add_argument('--tf-inference-dir')
            sub_parser.add_argument('--simulator-dir')
            sub_parser.add_argument('--mps-dir')
            sub_args, sub_argv = sub_parser.parse_known_args(argv)
            logger.info(pprint_msg({'sub_argv': sub_argv}))

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
        import pdb
        pdb.post_mortem()
        raise


def save_plot(df, plot_path, tee=True, crop_margin=True, savefig_kwargs=None):
    if savefig_kwargs is None:
        savefig_kwargs = dict()
    dataframe_txt_path = re.sub(r'\.\w+$', '.dataframe.txt', plot_path)
    assert dataframe_txt_path != plot_path

    dataframe_csv_path = re.sub(r'\.\w+$', '.dataframe.csv', plot_path)
    assert dataframe_csv_path != plot_path

    with open(dataframe_txt_path, 'w') as f:
        f.write(DataFrame.dataframe_string(df))
    if tee:
        logger.info("{plot_path} dataframe:\n{msg}".format(
            msg=txt_indent(DataFrame.dataframe_string(df), indent=1),
            plot_path=plot_path,
        ))

    df.to_csv(dataframe_csv_path, index=False)

    logger.info("Output plot @ {path}".format(path=plot_path))
    if crop_margin:
        plt.savefig(
            plot_path,
            bbox_inches='tight',
            pad_inches=0, **savefig_kwargs)
    else:
        plt.savefig(plot_path, **savefig_kwargs)
    if is_pdf(plot_path):
        pdf2png(plot_path)
    plt.close()

def rotate_xticks(ax, rotation):
    if rotation is None:
        return
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=rotation,
    )

def keep_cupti_metric(df, cupti_metric_name):
    prof_metric_name = METRIC_NAME_CUPTI_TO_PROF[cupti_metric_name]
    prof_metric_name_re = re.compile(re.escape(prof_metric_name))
    def _is_metric(metric_name):
        # Ignore the weird +/& symbols
        ret = bool(re.search(prof_metric_name_re, metric_name))
        return ret
        # return metric_name == prof_metric_name
    df = df[df['metric_name'].apply(_is_metric)].copy()
    # df['cupti_metric_name'] = cupti_metric_name
    return df

def add_gpu_hw_fields(df):
    def _cupti_metric_name(metric_name):
        metric_name = metric_name.rstrip('+&')
        return PROF_TO_METRIC_NAME_CUPTI[metric_name]
    df['cupti_metric_name'] = df['metric_name'].apply(_cupti_metric_name)

def multitask_title(task_type, task_type_plural, n_tasks, sep=' '):
    return sep.join([
        "Multi-{task}",
        "($N_{{{tasks}}}$ = {n_tasks})",
    ]).format(
        task=task_type,
        tasks=task_type_plural,
        n_tasks=n_tasks,
    )

def or_empty(x):
    if x is None:
        return ""
    return x


class OperationTree:
    def __init__(self, range_names):
        self.node = RangeNode()
        for range_name in range_names:
            self.node.add(range_name)
        # memoize.
        self._all_operations = dict()

    def leaf_operations(self):
        return set(node.name for node in self.node.leaves())

    def all_operations(self, name):
        if name in self._all_operations:
            return self._all_operations[name]
        nodes = list(self.node.find(name))
        assert len(nodes) == 1
        node = nodes[0]
        all_nodes = list(node.all_nodes())
        ret = set(node.name for node in all_nodes)
        self._all_operations[name] = ret
        return ret

class RangeNode:
    def __init__(self, name="[ROOT]", parent=None):
        # self.node =
        # name -> RangeNode
        self.name = name
        self.children = dict()
        self.parent = parent

    def find(self, name):
        if self.name == name:
            yield self
        for child in self.children.values():
            for node in child.find(name):
                yield node

    def all_nodes(self):
        yield self
        for child in self.children.values():
            for node in child.all_nodes():
                yield node

    def add(self, range_name):
        components = re.split(r'/', range_name)
        # self._add_components(components)

        node = self
        i = 0
        while i < len(components):
            if components[i] not in node.children:
                new_node = RangeNode(name=components[i], parent=node)
                node.children[components[i]] = new_node
                node = new_node
            else:
                node = node.children[components[i]]
            i += 1

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def leaves(self):
        for child in self.children.values():
            if child.is_leaf:
                yield child
            for child in child.leaves():
                yield child

    def _add_components(self, components):
        if len(components) == 0:
            return
        name = components[0]
        if name not in self.children:
            self.children[name] = RangeNode()
        child = self.children[name]
        child._add_components(components[1:])


def is_empty_file(path):
    return os.stat(path).st_size == 0

if __name__ == '__main__':
    main()
