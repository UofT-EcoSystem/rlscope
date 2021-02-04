"""
matplotlib code for creating the "stacked bar plot" figure used for CPU/GPU time breakdowns.
Also contains code for generating LaTeX macros for cited metrics in RL-Scope paper.
"""
from rlscope.profiler.rlscope_logging import logger
import argparse
import subprocess
import warnings
import re
import copy
import importlib
import sys
import textwrap
import copy
import collections
from collections import OrderedDict
import itertools

from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib.patches as mpatches
import matplotlib.patches as mpl_patches
from matplotlib import ticker as mpl_ticker
import matplotlib
# NOTE: If we don't do this, then with ForwardX11 enabled in ~/.ssh/config we get an error on python script exit:
#   XIO:  fatal IO error 0 (Success) on X server "localhost:10.0"
#         after 348 requests (348 known processed) with 1 events remaining.
# matplotlib.use('agg')

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec

import shutil
import os
import os.path
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.util import pprint_msg
from rlscope.parser.db import SQLCategoryTimesReader, sql_get_source_files, sql_input_path
from rlscope.parser.plot_index import _DataIndex
from rlscope.parser import plot_index
from rlscope.parser.overlap_result import from_js, CategoryKey as CategoryKeyJS
from rlscope.parser.dataframe import VennData, get_training_durations_df, read_rlscope_config_metadata, get_total_timesteps, get_end_num_timesteps, RLScopeConfig, extrap_total_training_time
from rlscope.parser import dataframe as rlscope_dataframe
from rlscope.parser.plot import LegendMaker, HATCH_STYLES, HATCH_STYLE_EMPTY, Y_LABEL_TRAINING_TIME_SEC
from rlscope.parser.exceptions import RLScopeConfigurationError
from rlscope.parser.plot_utils import pdf2png, crop_pdf

from rlscope.parser.common import *
from rlscope.parser import constants
# Make it so exec(...) has CATEGORY_* in-scope for things like --remap-df used in run_bench.sh
from rlscope.parser.constants import *

class OverlapStackedBarPlot:
    """
    Plot overlap data output by RL-Scope across SEVERAL runs where
    a run is an (algo, env_id) pair.

    Overlap data of the same "overlap_type" can be merged into a
    single stacked bar plot.

    For example, for overlap_type=ResourceOverlap
    (i.e. the CPU/GPU utilization plot), there would be a
    bar along the x-axis for each (algo, env_id) combination, and
    each bar would consits of the stacks [CPU, GPU].

    Some overlap_type's have data that doesn't overlap;
    for example OperationOverlap.

    However, some overlap_type's have data that DOES overlap:
    - ResourceOverlap: "CPU" subsumes "GPU"
    - CategoryOverlap: "Framework API C" subsumes "CUDA API C"
    For these cases where [CPU subsumes GPU] we'd like to show:

    ________________
    |     GPU      |
    |______________|
    |  CPU - GPU   |
    |______________|

     (algo, env_id)


    TODO: we can guess the (algo, env_id) pair name from the process_name or phase_name.
    May nee to add --algos --env-ids option in the future though.

    """
    SUPPORTED_OVERLAP_TYPES = ['ResourceOverlap', 'OperationOverlap', 'CategoryOverlap']
    SUPPORTED_X_TYPES = ['algo-comparison', 'env-comparison', 'rl-comparison']
    SUPPORTED_Y_TYPES = ['percent', 'seconds']

    def __init__(self,
                 rlscope_directories,
                 unins_rlscope_directories,
                 directory,
                 overlap_type,
                 rlscope_config_directories=None,
                 resource_overlap=None,
                 operation=None,
                 training_time=False,
                 extrapolated_training_time=False,
                 detailed=False,
                 remap_df=None,
                 xtick_expression=None,
                 y2_logscale=False,
                 hack_upper_right_legend_bbox_x=None,
                 ignore_inconsistent_overlap_regions=False,
                 skip_plot=False,
                 title=None,
                 x_title=None,
                 x_order_by=None,
                 rotation=None,
                 x_type='rl-comparison',
                 y_type='percent',
                 y_lim_scale_factor=None,
                 show_title=True,
                 show_legend=True,
                 width=None,
                 height=None,
                 long_env=False,
                 keep_zero=True,
                 suffix=None,
                 host=None,
                 user=None,
                 password=None,
                 debug=False,
                 debug_single_thread=False,
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
        assert overlap_type in OverlapStackedBarPlot.SUPPORTED_OVERLAP_TYPES
        assert x_type in OverlapStackedBarPlot.SUPPORTED_X_TYPES
        assert y_type in OverlapStackedBarPlot.SUPPORTED_Y_TYPES
        if len(rlscope_directories) == 0:
            raise ValueError("OverlapStackedBarPlot expects at least 1 trace-file directory for rlscope_directories")
        self._initialized = False
        self.rlscope_directories = rlscope_directories
        self.unins_rlscope_directories = unins_rlscope_directories
        if rlscope_config_directories is None:
            rlscope_config_directories = []
        self.rlscope_config_directories = rlscope_config_directories
        logger.info("{klass}:\n{msg}".format(
            klass=self.__class__.__name__,
            msg=pprint_msg({
                'rlscope_directories': self.rlscope_directories,
                'unins_rlscope_directories': self.unins_rlscope_directories,
            })))
        self.training_time = training_time
        self.extrapolated_training_time = extrapolated_training_time
        self.detailed = detailed
        # self.should_add_training_time = training_time
        self.directory = directory
        self.overlap_type = overlap_type
        self.resource_overlap = resource_overlap
        self.operation = operation
        self.y2_logscale = y2_logscale
        self.remap_df = remap_df
        self.xtick_expression = xtick_expression
        self.hack_upper_right_legend_bbox_x = hack_upper_right_legend_bbox_x
        self.ignore_inconsistent_overlap_regions = ignore_inconsistent_overlap_regions
        self.skip_plot = skip_plot
        if self.resource_overlap is not None:
            # Normalize ('GPU', 'CPU') into ('CPU', 'GPU') for equality checks,
            self.resource_overlap = tuple(sorted(self.resource_overlap))
        self.title = title
        self.x_title = x_title
        self.x_order_by = x_order_by
        self.rotation = rotation
        self.x_type = x_type
        self.y_type = y_type
        self.y_lim_scale_factor = y_lim_scale_factor
        if self.title is not None:
            self.show_title = True
        else:
            self.show_title = show_title
        self.show_legend = show_legend
        self.width = width
        self.height = height
        self.long_env = long_env
        self.keep_zero = keep_zero
        self.suffix = suffix
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def _get_plot_path(self, ext, extra_suffix):
        if self.suffix is not None:
            suffix_str = '.{suffix}'.format(suffix=self.suffix)
        else:
            suffix_str = ''

        if extra_suffix is not None:
            extra_suffix_str = '.{extra_suffix}'.format(extra_suffix=extra_suffix)
        else:
            extra_suffix_str = ''

        return _j(self.directory, "OverlapStackedBarPlot.overlap_type_{ov}{extra_suffix}{suffix}.{ext}".format(
            ov=self.overlap_type,
            suffix=suffix_str,
            extra_suffix=extra_suffix_str,
            ext=ext,
        ))

    def _plot_data_path(self, extra_suffix=None):
        return self._get_plot_path(ext='txt', extra_suffix=extra_suffix)

    def _plot_csv_path(self, extra_suffix=None):
        return self._get_plot_path(ext='csv', extra_suffix=extra_suffix)

    def _plot_path(self, extra_suffix=None):
        return self._get_plot_path(ext='pdf', extra_suffix=extra_suffix)

    # def _json_path(self, device_id, device_name):
    #     return _j(self.directory, "util_scale{dev}.js_path.json".format(
    #         dev=device_id_suffix(device_id, device_name),
    #     ))

    def _get_algo_or_env(self, algo_or_env):
        assert algo_or_env in {'env', 'algo'}

        def _get_value(rlscope_dir):
            algo, env = self._get_algo_env_from_dir(rlscope_dir)
            if algo_or_env == 'algo':
                value = algo
            else:
                value = env
            return value

        if algo_or_env == 'algo':
            field = 'algorithm'
        else:
            field = 'environment'

        values = set()
        for rlscope_dir in self.rlscope_directories:
            value = _get_value(rlscope_dir)
            if len(values) == 1 and value not in values:
                raise RuntimeError("Expected {field}={expect} but saw {field}={saw} for --rlscope-directory {dir}".format(
                    field=field,
                    expect=list(values)[0],
                    saw=value,
                    dir=rlscope_dir,
                ))
            values.add(value)
        assert len(values) == 1
        return list(values)[0]

    @property
    def algorithm(self):
        return self._get_algo_or_env('algo')

    @property
    def environment(self):
        return self._get_algo_or_env('env')

    @property
    def plot_x_axis_label(self):
        if self.x_title is not None:
            return self.x_title

        if self.x_type == 'rl-comparison':
            return "(RL algorithm, Simulator)"
        elif self.x_type == 'env-comparison':
            return "Simulator"
        elif self.x_type == 'algo-comparison':
            return "RL algorithm"
        raise NotImplementedError

    @property
    def plot_y_axis_label(self):
        # if self.y_type == 'percent':
        #     return "Time breakdown (percent)"
        # elif self.y_type == 'seconds':
        #     return "Time breakdown (seconds)"
        # raise NotImplementedError
        return "Percent (%)"

    @property
    def plot_y2_axis_label(self):
        if self.y2_logscale:
            return "Total training time (log2 sec)"
        else:
            return "Total training time (sec)"

        return "Total training time"

    @property
    def plot_title(self):
        if not self.show_title:
            return None

        if self.title is not None:
            return self.title

        if self.x_type == 'rl-comparison':
            title = "Comparing RL workloads"
        elif self.x_type == 'env-comparison':
            title = "Comparing environments when training {algo}".format(algo=self.algorithm)
        elif self.x_type == 'algo-comparison':
            title = "Comparing algorithms when training {env}".format(env=self.get_x_env(self.environment))
        else:
            raise NotImplementedError

        return title

    # def _get_algo_env_id(self, rlscope_dir):
    #     sql_reader = self.sql_readers[rlscope_dir]
    #     procs = sql_reader.process_names()
    #     assert len(procs) == 1
    #     proc = procs[0]
    #     m = re.search(r'(?P<algo>[^_]+)_(?P<env_id>.+)', proc)
    #     algo = m.group('algo')
    #     env_id = m.group('env_id')
    #     env_id = self._reduce_env(env_id)
    #     return (algo, env_id)

    def _get_algo_env_from_dir(self, rlscope_dir):
        # .../<algo>/<env_id>
        path = os.path.normpath(rlscope_dir)
        components = path.split(os.sep)
        env_id = components[-1]
        algo = components[-2]
        env_id = self._reduce_env(env_id)
        return (algo, env_id)

    def _reduce_env(self, env_id):
        """
        We want to treat these two environments as the same during analysis:
        LunarLander-v2
        LunarLanderContinuous-v2

        HACK: just remove "Continuous"

        :param env_id:
        :return:
        """
        new_env = env_id
        new_env = re.sub(r'LunarLanderContinuous-v2', 'LunarLander-v2', new_env)
        new_env = re.sub(r'MountainCarContinuous-v0', 'MountainCar-v0', new_env)
        return new_env

    def get_index(self, rlscope_dir):

        def _del_module(import_path):
            if import_path in sys.modules:
                del sys.modules[import_path]

        def _del_index_module():
            _del_module('rlscope_plot_index')
            _del_module('rlscope_plot_index_data')


        _del_index_module()

        sys.path.insert(0, rlscope_dir)
        rlscope_plot_index = importlib.import_module("rlscope_plot_index")
        index = rlscope_plot_index.DataIndex
        del sys.path[0]

        _del_index_module()

        # Check that env_id's match; otherwise, warn the user.
        # This can happen when generating trace files on one machine, and changing the directory structure on another.
        if _b(rlscope_dir) != _b(index.directory):
            logger.warning("rlscope_dir={rlscope_dir} != index.directory={index_dir}; make sure these paths use the same (algo, env)!".format(
                rlscope_dir=rlscope_dir,
                index_dir=index.directory,
            ))

        # return index

        # NOTE: if trace-files were processed on a different machine, the _DataIndex.directory will be different;
        # handle this by re-creating the _DataIndex with rlscope_dir.
        # my_index = _DataIndex(index.index, rlscope_dir, debug=self.debug)
        my_index = _DataIndex(index.index, rlscope_dir)
        return my_index

    def read_unins_df(self):
        if len(self.unins_rlscope_directories) == 0:
            return pd.DataFrame({
                'unins_total_training_time_us': [],
                'training_duration_us': [],
                'extrap_total_training_time': [],
                'algo': [],
                'env': [],
            })
        unins_df = pd.concat(get_training_durations_df(
            self.unins_rlscope_directories,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread))
        unins_df['unins_total_training_time_us'] = unins_df['training_duration_us']

        add_repetition(unins_df)
        self._add_df_fields(unins_df)
        return unins_df

    def _add_repetition(self, df):
        def _repetition(rlscope_directory):
            m = re.search(r'repetition_(?P<repetition>\d+)', os.path.basename(rlscope_directory))
            if m:
                return int(m.group('repetition'))
            return None
        df['repetition'] = df['rlscope_directory'].apply(_repetition)

    def _init_directories(self):
        """
        Initialize SQL / DataIndex needed for reading plot-data from rls-run'd --rlscope-directory's.

        :return:
        """
        if self._initialized:
            return

        self.data_index = dict()
        # self.sql_readers = dict()

        for rlscope_dir in self.rlscope_directories:
            # self.sql_readers[rlscope_dir] = SQLCategoryTimesReader(self.db_path(rlscope_dir), host=self.host, user=self.user, password=self.password)
            index = self.get_index(rlscope_dir)
            self.data_index[rlscope_dir] = index

        self._initialized = True

    def _add_or_suggest_selector_field(self, idx, selector, field_name, can_ignore=False, allow_many=False):
        """
        For e.g. field_name = 'resource_overlap' (['CPU'], or ['CPU', 'GPU']).

        If they provided --resource-overlap:
          Use that as selector['resource_overlap']
        Else If there's only available choice for --resource-overlap
          Use that.
        Else:
          Tell the user about the available choices for --resource-overlap.

        :param field_name:
            e.g. 'resource_overlap' for --resource-overlap
        """

        def optname(option):
            return "--{s}".format(s=re.sub(r'_', '-', option))

        # Oops, forgot to add self.resource_overlap in constructor.
        assert hasattr(self, field_name)

        value = getattr(self, field_name)
        if value is not None:
            selector[field_name] = value
        else:
            choices = idx.available_values(selector, field_name, can_ignore=can_ignore, skip_missing_fields=True)
            if len(choices) > 1 and not allow_many:
                raise RuntimeError("Please provide {opt}: choices = {choices}".format(
                    opt=optname(field_name),
                    choices=choices,
                ))
            # NOTE: if there's only one choice, we don't need to add it to selector.
            # selector[field_name] = choices[0]

    def each_vd(self):

        for rlscope_dir, rlscope_config_dir in itertools.zip_longest(self.rlscope_directories, self.rlscope_config_directories):
            if rlscope_config_dir is None:
                rlscope_config_dir = rlscope_dir

            idx = self.data_index[rlscope_dir]

            # If rlscope_config.json is present in 'algo' and 'env' are defined, get them from there.
            # Else, guess algo/env from directory path: assume rlscope_dir looks like <algo>/<env>

            rlscope_config_path = rlscope_dataframe.get_rlscope_config_path(rlscope_config_dir, allow_none=True)
            algo = None
            env = None
            if rlscope_config_path is not None:
                rlscope_config = RLScopeConfig(rlscope_config_path=rlscope_config_path)
                # with open(rlscope_config_path, 'r') as f:
                #     rlscope_config = json.load(f)
                algo = rlscope_config.algo(allow_none=True)
                env = rlscope_config.env(allow_none=True)

            if algo is None or env is None:
                algo_from_dir, env_from_dir = self._get_algo_env_from_dir(rlscope_dir)
                if algo is None:
                    algo = algo_from_dir
                if env is None:
                    env = env_from_dir

            # Q: There's only one ResourceOverlap plot...
            # However, there are many OperationOverlap plots; how can we select
            # among them properly?
            selector = {
                'overlap_type': self.overlap_type,
            }

            # 'CategoryOverlap': ['process', 'phase', 'resource_overlap', 'operation'],
            # 'ResourceOverlap': ['process', 'phase'],
            # 'ResourceSubplot': ['process', 'phase'],
            # 'OperationOverlap': ['process', 'phase', 'resource_overlap'],
            # 'HeatScale': ['device_name'],

            # TODO: support multi-process stuff.
            # NOTE: we should really add more fields here for multi-process support (e.g. 'process' and 'phase');
            # For now this just support single-process results.
            if not self.detailed:
                self._add_or_suggest_selector_field(idx, selector, 'resource_overlap', can_ignore=True)
            # self._add_or_suggest_selector_field(idx, selector, 'operation', can_ignore=self.overlap_type not in ['CategoryOverlap'])

            for md, entry, ident in idx.get_files(selector=selector, skip_missing_fields=True):
                vd = VennData(entry['venn_js_path'])
                yield (algo, env), vd

            # md, entry, ident = idx.get_file(selector=selector, skip_missing_fields=True)
            # vd = VennData(entry['venn_js_path'])
            # yield (algo, env), vd

    def each_df(self):
        for (algo, env), vd in self.each_vd():
            stacked_dict = vd.stacked_bar_dict()
            md = vd.metadata()
            path = vd.path
            rlscope_dir = _d(path)

            rlscope_metadata = read_rlscope_config_metadata(_d(path))

            def as_list(v):
                if type(v) == list:
                    return v
                return [v]
            new_stacked_dict = dict(stacked_dict)
            new_stacked_dict['algo'] = algo
            new_stacked_dict['env'] = env

            # if self.training_time or 'HACK_total_timesteps' in rlscope_metadata['metadata']:
            #     if 'percent_complete' in md:
            #         # Q: Will this handle scaling phases?  I think so... basically, each phase-file will just have a
            #         # different 'percent_complete'. However, I think we need to make OverlapStackedBarPlot have a phase argument,
            #         # or run for each phase.
            #         total_size = vd.total_size()
            #         # Extrapolate the total training time using percent_complete
            #         total_training_time = extrap_total_training_time(total_size, md['percent_complete'])
            #         new_stacked_dict['extrap_total_training_time'] = total_training_time
            #     else:
            #         new_stacked_dict['extrap_total_training_time'] = np.NAN

            extrap_dict = dict()
            extrap_dict['algo'] = algo
            extrap_dict['env'] = env
            extrap_dict['path'] = path

            if ( self.extrapolated_training_time or 'HACK_total_timesteps' in rlscope_metadata['metadata'] ) and 'percent_complete' in md:
                total_size = vd.total_size()
                if all(col in rlscope_metadata['metadata'] for col in {'HACK_unins_end_training_time_us',
                                                                   'HACK_unins_start_training_time_us',
                                                                   'HACK_unins_end_num_timesteps',
                                                                   'HACK_unins_start_num_timesteps',
                                                                   'HACK_total_timesteps'}):
                    # Extrapolate the total training time using uninstrumented run
                    logger.info("HACK: ({algo}, {env}) @ {path} -- Extrapolate the total training time using uninstrumented run".format(
                        algo=algo,
                        env=env,
                        path=path,
                    ))

                    """
                    PID=24865/MainProcess @ dump_proto_txt, dump_proto.py:19 :: 2020-01-12 19:55:38,607 INFO: > DUMP: IncrementalTrainingProgress @ training_progress.trace_61.proto
                    content_code: TP_HAS_PROGRESS
                    process_name: "airlearning-rlscope"
                    phase: "airlearning-rlscope"
                    machine_name: "eco-15"
                    total_timesteps: 3000
                    start_trace_time_us: 1578618824694440
                    start_percent_complete: 0.3336666524410248
                    start_num_timesteps: 1001
                    start_training_time_us: 1578618824694492
                    end_percent_complete: 0.9990000128746033
                    end_training_time_us: 1578619511199796
                    end_num_timesteps: 2997
                    """

                    unins_train_time_us = rlscope_metadata['metadata']['HACK_unins_end_training_time_us'] - rlscope_metadata['metadata']['HACK_unins_start_training_time_us']
                    unins_timesteps = rlscope_metadata['metadata']['HACK_unins_end_num_timesteps'] - rlscope_metadata['metadata']['HACK_unins_start_num_timesteps']
                    total_timesteps = rlscope_metadata['metadata']['HACK_total_timesteps']
                    total_training_time = (unins_train_time_us / unins_timesteps) * total_timesteps
                    new_stacked_dict['extrap_total_training_time'] = total_training_time

                    extrap_dict['unins_train_time_us'] = unins_train_time_us
                    extrap_dict['unins_timesteps'] = unins_timesteps
                    extrap_dict['total_timesteps'] = total_timesteps
                    extrap_dict['total_training_time'] = total_training_time
                elif 'HACK_total_timesteps' in rlscope_metadata['metadata']:
                    # Extrapolate the total training time using percent_complete
                    logger.info("HACK: ({algo}, {env}) @ {path} -- Override total number of training timesteps to be {t}".format(
                        algo=algo,
                        env=env,
                        path=path,
                        t=rlscope_metadata['metadata']['HACK_total_timesteps'],
                    ))
                    end_time_timesteps = get_end_num_timesteps(rlscope_dir)
                    percent_complete = end_time_timesteps / rlscope_metadata['metadata']['HACK_total_timesteps']
                    total_training_time = extrap_total_training_time(total_size, percent_complete)
                    new_stacked_dict['extrap_total_training_time'] = total_training_time
                    extrap_dict['end_time_timesteps'] = end_time_timesteps
                    extrap_dict['HACK_total_timesteps'] = rlscope_metadata['metadata']['HACK_total_timesteps']
                else:
                    percent_complete = md['percent_complete']
                    total_training_time = extrap_total_training_time(total_size, percent_complete)
                    new_stacked_dict['extrap_total_training_time'] = total_training_time
                # extrap_dict['percent_complete'] = percent_complete
                # extrap_dict['total_training_time'] = total_training_time
                # extrap_dict['total_size'] = total_size
            else:
                new_stacked_dict['extrap_total_training_time'] = np.NAN
                extrap_dict['isnan'] = True
            logger.info("debug extrap:\n{msg}".format(msg=pprint_msg(extrap_dict)))

            new_stacked_dict = dict((k, as_list(v)) for k, v in new_stacked_dict.items())
            df = pd.DataFrame(new_stacked_dict)
            df['rlscope_directory'] = rlscope_dir

            df = self._HACK_remove_process_operation_df(df)
            df = self._remap_df(df, algo, env)

            yield (algo, env), self._regions(df), path, df

    def _category_join_plus(self, categories):
        return join_plus(
            categories,
            # check_non_empty=True,
        )

    def _join_plus_field(self, df, field):
        def _join_plus(row):
            return join_plus(
                row[field],
                # check_non_empty=True,
            )
        return df.apply(_join_plus, axis=1)

    def new_each_df(self):

        def _add_region_fields(df):
            # Keeps regions so we can "reduce"
            df['resource_overlap_regions'] = df['resource_overlap'].apply(tuple)
            df['region'] = df['region'].apply(tuple)

        def _add_region_plot_fields(df):
            df['category_regions'] = df['region']
            df['resource_overlap'] = self._join_plus_field(df, 'resource_overlap')
            df['category'] = self._join_plus_field(df, 'region')

        for (algo, env), vd in self.each_vd():
            path = vd.path
            md = vd.metadata()
            rlscope_dir = _d(path)

            rlscope_metadata = read_rlscope_config_metadata(rlscope_dir)
            df = vd.as_df(keep_metadata_fields=['machine', 'process', 'phase', 'operation', 'resource_overlap'])
            if len(df['operation']) > 0 and type(df['operation'][0]) != str:
                df['operation'] = df['operation'].apply(join_plus)

            _add_region_fields(df)
            df['time_sec'] = df['size'] / constants.MICROSECONDS_IN_SECOND
            # stacked_dict = vd.stacked_bar_dict()
            # md = vd.metadata()
            path = vd.path
            df['algo'] = algo
            df['env'] = env
            df['rlscope_directory'] = rlscope_dir
            # if 'percent_complete' in md:
            if all(col in rlscope_metadata['metadata'] for col in {'HACK_unins_end_training_time_us',
                                                               'HACK_unins_start_training_time_us',
                                                               'HACK_unins_end_num_timesteps',
                                                               'HACK_unins_start_num_timesteps',
                                                               'HACK_total_timesteps'}):
                # Extrapolate the total training time using uninstrumented run
                logger.info("HACK: ({algo}, {env}) @ {path} -- Extrapolate the total training time using uninstrumented run".format(
                    algo=algo,
                    env=env,
                    path=path,
                ))

                """
                PID=24865/MainProcess @ dump_proto_txt, dump_proto.py:19 :: 2020-01-12 19:55:38,607 INFO: > DUMP: IncrementalTrainingProgress @ training_progress.trace_61.proto
                content_code: TP_HAS_PROGRESS
                process_name: "airlearning-rlscope"
                phase: "airlearning-rlscope"
                machine_name: "eco-15"
                total_timesteps: 3000
                start_trace_time_us: 1578618824694440
                start_percent_complete: 0.3336666524410248
                start_num_timesteps: 1001
                start_training_time_us: 1578618824694492
                end_percent_complete: 0.9990000128746033
                end_training_time_us: 1578619511199796
                end_num_timesteps: 2997
                """

                unins_train_time_us = rlscope_metadata['metadata']['HACK_unins_end_training_time_us'] - rlscope_metadata['metadata']['HACK_unins_start_training_time_us']
                unins_timesteps = rlscope_metadata['metadata']['HACK_unins_end_num_timesteps'] - rlscope_metadata['metadata']['HACK_unins_start_num_timesteps']
                total_timesteps = rlscope_metadata['metadata']['HACK_total_timesteps']
                total_training_time = (unins_train_time_us / unins_timesteps) * total_timesteps
                # new_stacked_dict['extrap_total_training_time'] = total_training_time
                df['extrap_total_training_time'] = total_training_time
                assert not np.isnan(total_training_time)

                # extrap_dict['unins_train_time_us'] = unins_train_time_us
                # extrap_dict['unins_timesteps'] = unins_timesteps
                # extrap_dict['total_timesteps'] = total_timesteps
                # extrap_dict['total_training_time'] = total_training_time
            elif ( self.extrapolated_training_time or 'HACK_total_timesteps' in rlscope_metadata['metadata'] ) and 'percent_complete' in md:
                total_size = vd.total_size()
                # Extrapolate the total training time using percent_complete
                if 'HACK_total_timesteps' in rlscope_metadata['metadata']:
                    logger.info("HACK: ({algo}, {env}) @ {path} -- Override total number of training timesteps to be {t}".format(
                        algo=algo,
                        env=env,
                        path=path,
                        t=rlscope_metadata['metadata']['HACK_total_timesteps'],
                    ))
                    end_time_timesteps = get_end_num_timesteps(rlscope_dir)
                    percent_complete = end_time_timesteps / rlscope_metadata['metadata']['HACK_total_timesteps']
                else:
                    percent_complete = md['percent_complete']
                total_training_time = extrap_total_training_time(total_size, percent_complete)
                df['extrap_total_training_time'] = total_training_time
            # Just don't add it?
            # else:
            #     df['extrap_total_training_time'] = np.NAN

            df = self._new_HACK_remove_process_operation_df(df)
            df = self._new_remap_df(df, algo, env)
            # Re-add region fields, since remap_df may have changed them.
            _add_region_fields(df)
            _add_region_plot_fields(df)

            yield (algo, env), self._regions(df), path, df

    def _check_can_add_training_time(self):
        if not self.extrapolated_training_time:
            return
        for (algo, env), vd in self.each_vd():
            if 'percent_complete' not in vd.md:
                raise RuntimeError((
                    "Didn't find percent_complete attribute in metadata of overlap plot data @ {path}; "
                    "remove --training-time to plot without training time").format(
                    path=vd.path,
                ))
        return True

    def _check_overlap_json_files(self):
        regions_to_paths = dict()
        for (algo, env), regions, path, df in self.each_df():
            regions = frozenset(regions)
            if regions not in regions_to_paths:
                regions_to_paths[regions] = []
            regions_to_paths[regions].append(path)
        for k in list(regions_to_paths.keys()):
            regions_to_paths[k].sort()

        if self.debug:
            logger.info(pprint_msg({'regions_to_paths': regions_to_paths}))

        if len(regions_to_paths) > 1:
            """
            Output an error message:
            
            ERROR: *.venn_js.json files have inconsistent overlap regions:
            - overlap-regions (1):
              - regions: [...]
              - *.venn_js.json files:
                path/to/file_01.json
                path/to/file_02.json
            - overlap-regions (2):
              ...
            """
            err_lines = []

            if self.ignore_inconsistent_overlap_regions:
                msg_type = 'WARNING'
            else:
                msg_type = 'ERROR'

            err_lines.append("{msg_type}: *.venn_js.json files have inconsistent overlap regions:".format(
                msg_type=msg_type,
            ))
            for i, (regions, paths) in enumerate(regions_to_paths.items(), start=1):
                err_lines.append("- overlap region ({i})".format(i=i))
                err_lines.append("  - regions: {regions}".format(regions=sorted(regions)))
                err_lines.append("  - *.venn_js.json files:")
                for path in paths:
                    err_lines.append("    {path}".format(path=path))
            msg = '\n'.join(err_lines)
            if self.ignore_inconsistent_overlap_regions:
                raise RuntimeError(msg)
            else:
                logger.info(msg)

        regions_by_num_files = sorted(
            regions_to_paths.keys(),
            key=lambda regions: (len(regions_to_paths[regions]), regions_to_paths[regions]))
        use_regions = regions_by_num_files[-1]
        if self.debug:
            logger.info(pprint_msg({
                'regions_to_paths': regions_to_paths,
                'use_regions': use_regions}))
        return use_regions

    def _is_region(self, region):
        return type(region) == tuple

    def _regions(self, obj):
        assert type(obj) in [pd.DataFrame, dict]
        return set(key for key in obj.keys() if self._is_region(key))

    def _remap_df(self, orig_df, algo, env):
        if self.remap_df is None:
            return orig_df

        not_regions = [key for key in orig_df.keys() if not self._is_region(key)]

        # eval context:
        # TODO: limit locals/globals to these? Don't want to limit numpy/pandas access though.
        df = copy.copy(orig_df)
        regions = self._regions(df)
        new_df = df[not_regions]
        # algo
        # env

        df_transformation = self.remap_df
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

        # if self.debug:
        #     logger.info("--remap-df complete")
        #     logger.info("Old dataframe; regions={regions}".format(regions=self._regions(orig_df)))
        #     logger.info(pprint_msg(orig_df))
        #
        #     logger.info("New dataframe after --remap-df; regions={regions}".format(regions=self._regions(new_df)))
        #     logger.info(pprint_msg(new_df))

        if self.debug:
            buf = StringIO()
            DataFrame.print_df(orig_df, file=buf)
            logger.info("Old dataframe; regions={regions}:\n{msg}".format(
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
                regions=self._regions(orig_df),
            ))

            buf = StringIO()
            DataFrame.print_df(new_df, file=buf)
            logger.info("New dataframe after --remap-df; regions={regions}:\n{msg}".format(
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
                regions=self._regions(orig_df),
            ))

        return new_df

    def _new_remap_df(self, orig_df, algo, env):
        return apply_remap_df(self.remap_df, orig_df, debug=self.debug)

    def _num_algo_env_combos(self):
        return len(self.df[['algo', 'env']].unique())

    def _read_df(self):
        # TODO: merge these on (algo, env)
        # Only keep rows that have both.
        # (Would be nice to warn which rows are missing what)

        self.unins_df = self.read_unins_df()
        buf = StringIO()
        DataFrame.print_df(self.unins_df, file=buf)
        logger.info("unins_df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))

        self.ins_df = self.read_ins_df()
        buf = StringIO()
        DataFrame.print_df(self.ins_df, file=buf)
        logger.info("ins_df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))


        # Keep (algo, env) even if we don't have total uninstrumented training time for it.
        # Use extrapolated time instead.
        # TODO: if extrapolated_time is present in bot ins_df and unins_df, it will "conflict" creating
        # extrapolated_time_x and extrapolated_time_y

        # Avoid conflicts; keep rlscope_directory of instrumented run.
        unins_df_copy = copy.copy(self.unins_df)
        if 'rlscope_directory' in unins_df_copy:
            del unins_df_copy['rlscope_directory']
        # WRONG:
        merge_df = self.ins_df.merge(unins_df_copy, on=['algo', 'env', 'x_field', 'repetition'], how='left')
        self.df = merge_df

        def get_total_training_time(row):
            # Prefer actual uninstrumented training time over extrapolated training time.
            # Q: isn't extrapolated time including overheads...?
            if self.extrapolated_training_time or np.isnan(row['unins_total_training_time_us']):
                # If this goes off, extrapolated_time is present in bot ins_df and unins_df...
                # need to handle that still...
                assert 'extrap_total_training_time' in row
                return row['extrap_total_training_time']
            return row['unins_total_training_time_us']
        self.df['full_unins_training_time'] = self.df.apply(get_total_training_time, axis=1)
        self.df = self.df[~np.isnan(self.df['full_unins_training_time'])]
        if not self.detailed:
            self.df['total_training_time'] = self.df['full_unins_training_time']
        else:
            self.df['total_training_time'] = self.df['full_unins_training_time'] * self.df['percent']

        buf = StringIO()
        DataFrame.print_df(self.df, file=buf)
        logger.info("df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))

        self.df = self._reduce_df(self.df)

        buf = StringIO()
        DataFrame.print_df(self.df, file=buf)
        logger.info("reduced df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))


    def _reduce_df(self, df):

        def remove_common_category_resource(df):
            """
            GOAL: prepare hatch/hue labels in overlap plot; we want 'CPU' and 'GPU' categories to
            only appear in 'resource_overlap' column, NOT in 'category' column.
            NOTE: That means if ONLY GPU is running, it's going to show up as category=''.
            - Turn ["Backend", "GPU"] into ["Backend"].
            - Turn ["CUDA", "GPU"] into ["CUDA"]
            - Turn ["GPU"] into ''
            - Rationale:
                - hue='resource_overlap' which is "GPU"
                - hatch='category' which is ""
                    This is what empty_hatch_value="" is for.
            """
            cmap = dict()
            categories = set()
            for category_regions in df['category_regions'].unique():
                category_regions = frozenset(category_regions)
                cmap[category_regions] = category_regions
                for category in category_regions:
                    categories.add(category)

            resources = set()
            for resource_overlap_regions in df['resource_overlap_regions'].unique():
                resource_overlap_regions = frozenset(resource_overlap_regions)
                for resource in resource_overlap_regions:
                    resources.add(resource)

            common_resource_category = categories.intersection(resources)
            for category_regions in list(cmap.keys()):
                if len(category_regions.intersection(common_resource_category)) > 0:
                    new_category_regions = cmap[category_regions].difference(common_resource_category)
                    logger.info("cmap[category_regions:{category_regions}] - categories:{categories} = {new_c}".format(
                        category_regions=category_regions,
                        categories=common_resource_category,
                        new_c=new_category_regions,
                    ))
                    cmap[category_regions] = new_category_regions

            def _cmap(c):
                new_c = tuple(sorted(cmap[frozenset(c)]))
                return new_c
            df = copy.copy(df)
            df['category_regions'] = df['category_regions'].apply(_cmap)
            return df

        def shorten_category(df):

            def _short_category_region(category_region):
                return tuple(short_category(category) for category in category_region)

            df = copy.copy(df)
            df['category_regions'] = df['category_regions'].apply(_short_category_region)
            return df

        df = remove_common_category_resource(df)
        df = shorten_category(df)
        df['category'] = self._join_plus_field(df, 'category_regions')
        return df



    def read_ins_df(self):
        """
        Read venn_js data of several --rlscope-directory's into a single data-frame.

        :return:
        """
        self._init_directories()
        self.data = {
            'algo':[],
            'env':[],
        }
        self.regions = None

        # Process each df separately, since they have DIFFERENT keys.
        # Then, check that the groups for the df match.

        if not self.detailed:
            use_regions = self._check_overlap_json_files()

        dfs = []
        if self.detailed:
            df_iter = self.new_each_df()
        else:
            df_iter = self.each_df()
        for (algo, env), regions, path, df in df_iter:
            # rlscope_dir = _d(path)
            buf = StringIO()
            DataFrame.print_df(df)
            logger.info("({algo}, {env}) @ path={path}:\n{msg}".format(
                algo=algo,
                env=env,
                path=path,
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
            ))

            if not self.detailed and regions != use_regions:
                logger.info(
                    textwrap.dedent("""\
                    Skipping {path} (--ignore-inconsistent-overlap-regions)
                      regions = {regions}
                      use_regions = {use_regions}
                    """).lstrip().format(
                        path=path,
                        regions=regions,
                        use_regions=use_regions,
                    ))
                continue

            if self.regions is None:
                self.regions = set(regions)

            # if self.debug:
            #     logger.info(pprint_msg({
            #         'path': path,
            #         'df': df}))

            dfs.append(df)

        if self.debug:
            logger.info("ins_df before concat:\n{msg}".format(msg=pprint_msg(dfs)))
        ins_df = pd.concat(dfs)
        if self.debug:
            logger.info("ins_df after concat:\n{msg}".format(msg=pprint_msg(ins_df)))

        old_ins_df = ins_df
        all_cols = set(old_ins_df.keys())
        numeric_cols = set([colname for colname in old_ins_df.keys() if np.issubdtype(old_ins_df[colname].dtype, np.number)])
        non_numeric_cols = all_cols.difference(numeric_cols)
        # old_ins_df.groupby(['machine', 'operation', 'phase', 'process', 'region', 'resource_overlap', 'category', 'algo', 'env', 'x_field']).sum().reset_index()
        # WARNING:  df.groupby(...).sum(skipna=False)
        #   This IGNORES NaN's in aggregated columns resulting in NaN's not being forwarded (e.g. zero columns appear instead of NaN's)
        # SOLUTION: df.groupby(...).agg(pd.DataFrame.sum, skipna=False)
        # SEE: https://github.com/pandas-dev/pandas/issues/28787
        # ins_df = old_ins_df.groupby(list(non_numeric_cols)).sum().reset_index()
        ins_df = old_ins_df.groupby(list(non_numeric_cols)).agg(pd.DataFrame.sum, skipna=False).reset_index()

        if self.debug:
            logger.info("ins_df after groupby.sum():\n{msg}".format(msg=pprint_msg(ins_df)))

        if self.detailed:
            # NOTE: must calculate percent within an (algo, env)
            group_dfs = []
            for group, group_df in ins_df.groupby(['rlscope_directory', 'phase', 'algo', 'env']):
                total_size = group_df['size'].sum()
                # Q: does this update ins_df...?
                group_df['percent'] = group_df['size']/total_size
                group_df['percent_y'] = 100*group_df['percent']
                group_dfs.append(group_df)
            ins_df = pd.concat(group_dfs)

        add_repetition(ins_df)
        self._add_df_fields(ins_df)

        return ins_df

    def _HACK_remove_process_operation_df(self, df):
        """
        HACK: BUG: not sure why, but operation overlap contains an operation that looks like the process_name:
        e.g.
        {
            "label": "[ppo2_Walker2DBulletEnv-v0]",
            "sets": [
                0
            ],
            "size": 45344.0
        },
        Likely reason: we used to have code that checked if a Event.event_name looked like a "process_name"...
        for some reason that code check has been disabled during analysis.
        Fix: the time is REALLY small compared to everything else, so we can either:
        1. add it to the largest time (e.g. step)
        2. ignore it (should be safe if its an absolute time...not safe if its a percent already)

        :return:
        """
        remove_cols = set()
        for colname, coldata in df.iteritems():
            if self._is_region(colname):
                op_name = colname[0]
                if is_op_process_event(op_name, constants.CATEGORY_OPERATION):
                    # logger.info("HACK: remove process_name={proc} from operation dataframe".format(
                    #     proc=op_name,
                    # ))
                    remove_cols.add(colname)
        for colname in remove_cols:
            del df[colname]
        return df

    def _new_HACK_remove_process_operation_df(self, df):
        """
        HACK: BUG: not sure why, but operation overlap contains an operation that looks like the process_name:
        e.g.
        {
            "label": "[ppo2_Walker2DBulletEnv-v0]",
            "sets": [
                0
            ],
            "size": 45344.0
        },
        Likely reason: we used to have code that checked if a Event.event_name looked like a "process_name"...
        for some reason that code check has been disabled during analysis.
        Fix: the time is REALLY small compared to everything else, so we can either:
        1. add it to the largest time (e.g. step)
        2. ignore it (should be safe if its an absolute time...not safe if its a percent already)

        :return:
        """
        def is_process_event(op_name):
            return bool(is_op_process_event(op_name, constants.CATEGORY_OPERATION))
        new_df = df[~df['operation'].apply(is_process_event)]
        return new_df

        return new_df

    def _normalize_df(self, df):
        """
        Transform raw venn_js file into appropriate units for plotting
        (i.e. convert us to seconds).
        """
        def transform_usec_to_sec(df, group):
            return df[group]/constants.USEC_IN_SEC

        def transform_usec_to_percent(df):
            """
            GOAL:
            CPU = CPU / ([CPU] + [CPU, GPU] + [GPU])

            :param df:
            :param group:
            :return:
            """

            # e.g.
            # summation = [CPU] + [CPU, GPU] + [GPU]
            summation = None
            for g in self.regions:
                if summation is None:
                    summation = df[g]
                else:
                    summation = summation + df[g]

            ret = dict()
            for g in self.regions:
                # e.g.
                # CPU = CPU / ([CPU] + [CPU, GPU] + [GPU])
                ret[g] = 100 * df[g]/summation

            return ret

        if 'total_training_time' in df:
            df['total_training_time'] = transform_usec_to_sec(df, 'total_training_time')

        if not self.detailed:

            if self.y_type == 'seconds':
                for group in self.regions:
                    df[group] = transform_usec_to_sec(df, group)
            elif self.y_type == 'percent':
                ret = transform_usec_to_percent(df)
                for group in ret.keys():
                    df[group] = ret[group]
            else:
                raise NotImplementedError

    def get_x_env(self, env):
        return get_x_env(env, long_env=self.long_env)

    def get_x_field(self, algo, env, human_readable=False):
        return get_x_field(algo, env, self.x_type, human_readable=human_readable)

    def _plot_df(self):
        logger.info("Dataframe:\n{df}".format(df=self.df))

        if self.training_time:
            y2_field = 'total_training_time'
        else:
            y2_field = None

        stacked_bar_plot = StackedBarPlot(
            data=self.df,
            path=self._plot_path(),
            groups=sorted(self.regions),
            x_field='x_field',
            y2_field=y2_field,
            y2_logscale=self.y2_logscale,
            x_axis_label=self.plot_x_axis_label,
            y_axis_label=self.plot_y_axis_label,
            y2_axis_label=self.plot_y2_axis_label,
            title=self.plot_title,
            show_legend=self.show_legend,
            width=self.width,
            height=self.height,
            keep_zero=self.keep_zero,
            rotation=self.rotation,
            # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
            group_to_label=self.group_to_label,
        )
        stacked_bar_plot.plot()

    def _detailed_plot_df(self):
        buf = StringIO()
        DataFrame.print_df(self.df)
        logger.info("Dataframe:\n{df}".format(
            df=textwrap.indent(buf.getvalue(), prefix='  ')))

        # if self.training_time:
        #     y2_field = 'total_training_time'
        # else:
        #     y2_field = None


        orig_df_training_time = self.df[['x_field', 'algo', 'env', 'rlscope_directory', 'total_training_time']]

        if self.debug:
            ss = StringIO()
            DataFrame.print_df(orig_df_training_time, file=ss)
            logger.debug(f"orig_df_training_time\n{ss.getvalue()}")

        cols = group_numeric_cols(orig_df_training_time)
        df_training_time = orig_df_training_time.groupby(cols.non_numeric_cols).agg(pd.DataFrame.sum, skipna=False).reset_index()

        if self.debug:
            ss = StringIO()
            DataFrame.print_df(df_training_time, file=ss)
            logger.debug(f"df_training_time = orig_df_training_time.groupby({cols.non_numeric_cols}).agg(sum)\n{ss.getvalue()}")

        # if self.debug:
        #     ss = StringIO()
        #     DataFrame.print_df(self.unins_df, file=ss)
        #     logger.debug(f"data2 = self.unins_df\n{ss.getvalue()}")

        # def bar_label_func(i, x_group):
        #     if x_group == 'Backpropagation':
        #         return "BP"
        #     elif x_group == 'Inference':
        #         return "Inf"
        #     elif x_group == 'Simulation':
        #         return "Sim"
        #     raise NotImplementedError("Not sure what bar-label to use for x_group=\"{x_group}\"".format(
        #         x_group=x_group))


        # log_y_scale = True
        # log_y_scale = False

        def ax_func(stacked_bar_plot):
            if stacked_bar_plot.ax2 is not None:
                stacked_bar_plot.ax2.grid(b=True, axis='y')
                stacked_bar_plot.ax2.set_axisbelow(True)

                if self.y2_logscale:
                    stacked_bar_plot.ax2.set_yscale('log', basey=2)
                    # stacked_bar_plot.ax2.minorticks_on()
                    # stacked_bar_plot.ax2.yaxis.set_minor_locator(mpl_ticker.AutoMinorLocator())
                    # stacked_bar_plot.ax2.yaxis.set_tick_params(which='minor', right='off')

            if self.rotation is not None:
                stacked_bar_plot.ax.set_xticklabels(
                    stacked_bar_plot.ax.get_xticklabels(),
                    rotation=self.rotation)


        def mk_plot(y_field, ylabel, path, **kwargs):
            rls_paper_fontsizes()
            stacked_bar_plot = DetailedStackedBarPlot(
                data=self.df,
                path=path,
                x_field='x_field',
                y_field=y_field,
                x_group='operation',
                x_order_by=self.x_order_by,

                y_lim_scale_factor=self.y_lim_scale_factor,

                hues_together=True,
                hatch='category',
                hatch_styles=RLS_SHORT_CATEGORY_HATCH_STYLE,
                empty_hatch_value="",
                # bar_label_func=bar_label_func,
                hue='resource_overlap',
                hack_upper_right_legend_bbox_x=self.hack_upper_right_legend_bbox_x,

                # hatches_together=True,
                # hatch='resource_overlap',
                # hue='category',

                xlabel=self.plot_x_axis_label,
                # xlabel='(RL algorithm, Simulator)',

                width=self.width,
                height=self.height,

                ylabel=ylabel,
                title=self.plot_title,
                func=ax_func,

                #
                # y2 data: total training time (black bar on top)
                #
                y2_field='total_training_time',
                y2label=Y_LABEL_TRAINING_TIME_SEC,
                # HACK: adjust y-position of y-label by adding empty-whitespace at the end.
                # NOTE: DOESN'T work... autocrop fails to remove text area it occupies.
                # y2label='Training time (sec)        ',
                data2=df_training_time,
                n_y2_ticks=4,

                debug=self.debug,
                # pdb=self.debug,
                **kwargs,
            )
            stacked_bar_plot.plot()
        def _suffix(suffix, yerr):
            if yerr:
                suffix += '.yerr'
            return suffix
        for yerr in [True, False]:
            percent_suffix = _suffix('percent', yerr)
            mk_plot(
                y_field='percent_y', ylabel='Percent (%)', path=self._plot_path(percent_suffix), yerr=yerr,
            )
            # NOTE: showing "Total training time (sec)" on top is redundant.
            # TODO: add y-grid lines! Do it in a simple isolated test since I had issues before...
            time_suffix = _suffix('operation_training_time', yerr)
            mk_plot(
                y_field='total_training_time', ylabel=Y_LABEL_TRAINING_TIME_SEC, path=self._plot_path(time_suffix), yerr=yerr,
            )

        # stacked_bar_plot = DetailedStackedBarPlot(
        #     data=self.df,
        #     path=self._plot_path('plot_category'),
        #     x_field='x_field',
        #     # y_field='time_sec',
        #     y_field='total_training_time',
        #     hatch='operation',
        #     hue='category',
        #     xlabel='(RL algorithm, Simulator)',
        #     ylabel='Total training time (seconds)',
        #     title=self.plot_title,
        #     debug=self.debug,
        # )
        # stacked_bar_plot.plot()
        #
        # ignore_cols = ['category', 'region']
        # all_cols = set(self.df.keys())
        # numeric_cols = set([
        #     colname for colname in self.df.keys()
        #     if np.issubdtype(self.df[colname].dtype, np.number)])
        # non_numeric_cols = all_cols.difference(numeric_cols)
        #
        # groupby_cols = non_numeric_cols.difference(ignore_cols)
        # keep_cols = all_cols.difference(ignore_cols)
        # groupby = self.df[list(keep_cols)].groupby(by=list(groupby_cols))
        # resource_df = groupby.sum().reset_index()
        #
        # stacked_bar_plot = DetailedStackedBarPlot(
        #     data=resource_df,
        #     path=self._plot_path('plot_resource'),
        #     x_field='x_field',
        #     # y_field='time_sec',
        #     y_field='total_training_time',
        #     hatch='operation',
        #     hue='resource_overlap',
        #     xlabel='(RL algorithm, Simulator)',
        #     ylabel='Total training time (seconds)',
        #     title=self.plot_title,
        #     debug=self.debug,
        # )
        # stacked_bar_plot.plot()

    def _add_df_fields(self, df, human_readable=False):
        """
        Add any additional fields to the data-frame.

        In particular 'x_field' is a string used for the "x-tick" labels of the plot.
        :param human_readable
            If True, make it csv friendly (i.e. don't use newlines in string)
        :return:
        """
        def row_func(row):
            x_field = self.get_x_field(row['algo'], row['env'], human_readable=human_readable)
            return x_field
        add_df_xfield(df, xtick_expression=self.xtick_expression, row_func=row_func, debug=self.debug)

    def group_to_label(self, group):
        label = ' + '.join(group)
        return label

    def run(self):
        if self.debug:
            algo_env_pairs = [self._get_algo_env_from_dir(rlscope_dir) for rlscope_dir in self.rlscope_directories]
            logger.info("{klass}: {msg}".format(
                klass=self.__class__.__name__,
                msg=pprint_msg({
                    'rlscope_directories': self.rlscope_directories,
                    'algo_env_pairs': algo_env_pairs,
                })))
        self._init_directories()
        # self._check_can_add_training_time()
        self._read_df()
        self._normalize_df(self.df)
        self._normalize_df(self.unins_df)
        self.dump_plot_data()
        if self.skip_plot:
            logger.info("Skipping plotting {path} (--skip-plot)".format(path=self._plot_path))
        else:
            if self.detailed:
                self._detailed_plot_df()
            else:
                self._plot_df()

        # self.dump_js_data(norm_samples, device, start_time_sec)
        # plotter.add_data(norm_samples)
        # print("> HeatScalePlot @ {path}".format(path=png))
        # plotter.plot()

        # TODO: read data into this format:
        # data = {
        #     'algo':     [algo_1,          algo_2         ],
        #     'env':      [env_1,           env_2          ],
        #     'CPU':      [25,              50             ],
        #     'GPU':      [75,              50             ],
        # }
        # df = pandas.DataFrame(data)
        # # Transform df to handle the different "subsumes" relationships between categories.

    # def dump_js_data(self, norm_samples, device, start_time_sec):
    #     js = {
    #         'metadata': {
    #             'plot_type': 'OverlappedStackedBar',
    #             # 'device_id': device.device_id,
    #             # 'device_name': device.device_name,
    #             # 'start_time_usec': float(start_time_sec)*constants.MICROSECONDS_IN_SECOND,
    #             # 'step_usec': self.step_sec*constants.MICROSECONDS_IN_SECOND,
    #             # 'decay': self.decay,
    #         },
    #         'data': {
    #             'util': norm_samples['util'],
    #             'start_time_sec': norm_samples['start_time_sec'],
    #         },
    #     }
    #     path = self._json_path(device.device_id, device.device_name)
    #     print("> HeatScalePlot @ plot data @ {path}".format(path=path))
    #     do_dump_json(js, path)

    def dump_plot_data(self):
        human_df = self.human_df()

        logger.info("> {name} @ human readable plot data @ {path}".format(
            name=self.__class__.__name__,
            path=self._plot_data_path()))
        with open(self._plot_data_path(), 'w') as f:
            DataFrame.print_df(human_df, file=f)

        logger.info("> {name} @ csv plot data @ {path}".format(
            name=self.__class__.__name__,
            path=self._plot_csv_path()))

        human_df.to_csv(self._plot_csv_path(), index=False)

        # Print human readable plot data to stdout
        logger.info(pprint_msg(human_df))

    def human_df(self):
        """
        Convert tuple keys in data-frame to user-visible labels
        ('CPU', 'GPU') => "CPU + GPU"
        :return:
        """
        human_df = copy.copy(self.df)

        self._add_df_fields(human_df, human_readable=True)

        groups = [g for g in human_df.keys() if self._is_region(g)]
        for group in groups:
            human_group = self.group_to_label(group)
            assert human_group not in human_df
            human_df[human_group] = human_df[group]
            del human_df[group]
        return human_df

    def db_path(self, rlscope_dir):
        return sql_input_path(rlscope_dir)


def attempt_stacked_bar():
    df = pd.DataFrame(dict(
        A=[1, 2, 3, 4],
        B=[2, 3, 4, 5],
        C=[3, 4, 5, 6],
        D=[4, 5, 6, 7]))

    fig = plt.figure(figsize=(20, 10))

    ab_bar_list = [plt.bar([0, 1, 2, 3], df.B, align='edge', width= 0.2),
                   plt.bar([0, 1, 2, 3], df.A, align='edge', width= 0.2)]

    cd_bar_list = [plt.bar([0, 1, 2, 3], df.D, align='edge',width= -0.2),
                   plt.bar([0, 1, 2, 3], df.C, align='edge',width= -0.2)]

    path = './test_stacked_bar.png'
    logger.info('Save figure to {path}'.format(path=path))
    plt.savefig(path)

class StackedBarPlot:
    def __init__(self,
                 data, path,
                 groups=None,
                 x_field=None,
                 y2_field=None,
                 y2_logscale=False,
                 x_axis_label="X-axis label",
                 y_axis_label="Y-axis label",
                 y2_axis_label=None,
                 title=None,
                 show_legend=True,
                 width=None,
                 height=None,
                 keep_zero=True,
                 rotation=None,
                 # fontsize=16,
                 fontsize=None,
                 group_to_label=None):
        assert groups is not None
        assert x_field is not None
        self.data = pd.DataFrame(data)
        self.path = path
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.y2_axis_label = y2_axis_label
        self.title = title
        self.show_legend = show_legend
        self.width = width
        self.height = height
        if ( self.width is not None and self.height is None ) or \
                (self.width is None and self.height is not None ):
            raise ValueError("You must provide both --width and --height")
        self.keep_zero = keep_zero
        self.rotation = rotation
        self.fontsize = fontsize
        self.groups = groups
        self.x_field = x_field
        self.y2_field = y2_field
        self.y2_logscale = y2_logscale
        self.fontsize = fontsize
        self.group_to_label = group_to_label

    def _group_to_label(self, group):
        if self.group_to_label is not None:
            label = self.group_to_label(group)
            return label
        label = group
        return label

    def _all_zero(self, group):
        return (self.data[group] == 0.).all()

    def _add_legend(self, fig, loc, bbox_to_anchor):
        legend_handles, legend_labels = self._legend_handles_labels()
        leg = fig.legend(legend_handles, legend_labels,
                         bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
        leg.draw_frame(False)

    def _legend_handles_labels(self):
        legend_handles = []
        def mk_rect(color):
            legend_rect = plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none')
            return legend_rect
        for i, group in enumerate(self.groups):
            if not self.keep_zero and self._all_zero(group):
                # If a stacked-bar element is zero in all the bar-charts, don't show it in the legend (--keep-zero false).
                continue
            legend_rect = mk_rect(self.colors[i])
            legend_handles.append(legend_rect)
        legend_labels = [self._group_to_label(group) for group in self.groups]
        if self.y2_field is not None and self.y2_axis_label is not None:
            # Last color is for y2-field bar.
            assert len(self.colors) == len(self.groups) + 1
            legend_rect = mk_rect(self.colors[-1])
            legend_handles.append(legend_rect)
            legend_labels.append(self.y2_axis_label)
        return legend_handles, legend_labels

    def plot(self):

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)
        ax2 = None
        if self.y2_field is not None:
            ax2 = ax.twinx()
            # Need to do this, otherwise, training time bar is ABOVE gridlines from ax.
            ax.set_zorder(ax2.get_zorder()+1)
            # Need to do this, otherwise training time bar is invisible.
            ax.patch.set_visible(False)


        #Set general plot properties

        sns.set_style("white")

        # ax = plt.subplot()
        # ax_list = fig.axes
        # plt.subplot()
        # fig, ax = plt.subplots()
        # ax.set_xs
        # sns.set_context({"figure.figsize": (24, 10)})

        if self.fontsize is not None:
            sns.set_style('font', {
                'size': self.fontsize,
            })

        # plt.rc('xtick', rotation=40)
        # sns.set_style('xtick', {
        #     'rotation': 40,
        # })

        # TODO:
        # - Make it so plot legends appear to right of the plot
        # - Make it so we can choose NOT to show plot legend (ideally just make it invisible...)
        # - All fonts should be same size

        if self.y2_field is not None:
            # Total training time bar gets its own color.
            num_colors = len(self.groups) + 1
        else:
            num_colors = len(self.groups)
        self.colors = sns.color_palette("hls", num_colors)

        if self.y2_field is not None:
            bar_width = 0.25
        else:
            bar_width = 0.5

        ind = np.arange(len(self.data[self.x_field]))
        ax.set_xticks(ind + bar_width/2)
        ax.set_xticklabels(self.data[self.x_field])

        n_bars = len(self.data[self.groups[0]])
        accum_ys = np.zeros(n_bars)
        barplot_kwargs = []
        bar_zorder = 0
        # bar_zorder = -1
        grid_zorder = 1
        for i, group in enumerate(self.groups):
            accum_ys += self.data[group]
            ys = copy.copy(accum_ys)
            if self.y2_field is not None:
                xs = ind
            else:
                xs = ind + bar_width/2
            bar_kwargs = {
                'x': xs,
                # 'y': ys,
                'height': ys,
                'color': self.colors[i],
                # 'ax': ax,
                # 'position': 0,
                'zorder': bar_zorder,
            }
            if bar_width is not None:
                bar_kwargs['width'] = bar_width
            barplot_kwargs.append(bar_kwargs)

        if self.y2_field is not None:
            # TODO: we need to group rows and sum them based on matching df[group]...?
            # for i, group in enumerate(self.groups):
            y_color = self.colors[-1]
            bar_kwargs = {
                # 'x': self.data[self.x_field],
                'x': ind + bar_width,
                'height': self.data[self.y2_field],
                # 'y': self.data[self.y2_field],
                'color': y_color,
                # 'ax': ax2,
                # 'position': 1,
                'zorder': bar_zorder,
            }
            if bar_width is not None:
                bar_kwargs['width'] = bar_width
            # sns.barplot(**bar_kwargs)
            # plt.bar(**bar_kwargs)
            ax2.bar(**bar_kwargs)

        barplots = []
        for kwargs in reversed(barplot_kwargs):
            # TODO: color?
            # barplot = sns.barplot(**kwargs)
            # barplot = plt.bar(**kwargs)
            barplot = ax.bar(**kwargs)
            barplots.append(barplot)
        barplots.reverse()

        if self.y2_field is not None and self.y2_logscale:
            # ax2.set_yscale('log')
            ax2.set_yscale('log', basey=2)

            # ax2.set_yscale('log')
            # ax2.set_yticks([1,10,100] + [max(y)])
            # from matplotlib.ticker import FormatStrFormatter

            # ax2.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter('%.d'))

            # Use (days, hours, minutes, seconds)
            # ax2.yaxis.set_major_formatter(DaysHoursMinutesSecondsFormatter())

        # #Plot 1 - background - "total" (top) series
        # sns.barplot(x = self.data.Group, y = self.data.total, color = "red")
        #
        # #Plot 2 - overlay - "bottom" series
        # bottom_plot = sns.barplot(x = self.data.Group, y = self.data.Series1, color = "#0000A3")

        bottom_plot = barplots[-1]

        figlegend = plt.figure()
        self._add_legend(
            figlegend,
            loc='center',
            bbox_to_anchor=None,
        )

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
        legend_handles, legend_labels = self._legend_handles_labels()
        logger.info({
            # 'handles': handles,
            # 'labels': labels,
            'legend_handles': legend_handles,
            'legend_labels': legend_labels,
        })
        ax.legend(legend_handles, legend_labels,
                  bbox_to_anchor=(0, 1.02, 1, 0.2),
                  loc='lower left',
                  mode='expand',
                  # ncol=len(legend_labels) - 1,
                  ncol=2,
                  # nrow=2,
                  fancybox=True, framealpha=0.5)

        # , prop={'size': self.fontsize}

        # topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
        # bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
        # l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})

        #Optional code - Make plot look nicer
        sns.despine(fig=fig, left=True)
        # bottom_plot.set_ylabel(self.y_axis_label)
        # bottom_plot.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)
        ax.set_xlabel(self.x_axis_label)
        if self.title is not None:
            # bottom_plot.set_title(self.title)
            ax.set_title(self.title)

        if self.rotation is not None:
            # ax = bottom_plot.axes
            ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)

        # Suggestions about how to prevent x-label overlap with matplotlib:
        #
        # https://stackoverflow.com/questions/42528921/how-to-prevent-overlapping-x-axis-labels-in-sns-countplot

        # #Set fonts to consistent 16pt size
        # for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
        #              bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        #     if self.fontsize is not None:
        #         item.set_fontsize(self.fontsize)

        ax.grid(zorder=grid_zorder)
        # Turn off x-axis grid-lines (just want y-axis grid-lines)
        ax.grid(b=False, axis='x')
        if self.y2_field is not None:
            # ax2.grid(True)

            if self.y2_axis_label is not None:
                ax2.set_ylabel(self.y2_axis_label)

            # Align training time against percent.
            # (weird training time labels).
            #
            # l = ax.get_ylim()
            # l2 = ax2.get_ylim()
            # f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
            # ticks = f(ax.get_yticks())
            # ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

            # Align percent against training time.
            # (weird percent labels).
            #
            # l = ax2.get_ylim()
            # l2 = ax.get_ylim()
            # f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
            # ticks = f(ax2.get_yticks())
            # ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))

        logger.info('Save figure to {path}'.format(path=self.path))
        fig.tight_layout()
        fig.savefig(self.path)
        plt.close(fig)

        figlegend.tight_layout()
        figlegend.savefig(self.legend_path, bbox_inches='tight', pad_inches=0)
        plt.close(figlegend)

    @property
    def legend_path(self):
        return re.sub(r'(\.[^.]+)$', r'.legend\1', self.path)


class DetailedStackedBarPlot:

    ERRBAR_COLOR = '#969696'
    ERRBAR_CAPSIZE_POINTS = 20

    def __init__(self,
                 data, path,
                 x_field,
                 y_field,
                 x_group,
                 # operation
                 hatch,
                 # category
                 # If True AND > 1 repetitions, draw error bars.
                 yerr=False,
                 hue=None,
                 x_order_by=None,
                 hues_together=False,
                 hatch_styles=None,
                 hatches_together=False,
                 y2_field=None,
                 n_y2_ticks=None,
                 data2=None,
                 empty_hatch_value=None,
                 bar_label_func=None,
                 y_lim_scale_factor=None,
                 hack_upper_right_legend_bbox_x=None,
                 xlabel=None,
                 ylabel=None,
                 y2label=None,
                 title=None,
                 width=None,
                 height=None,
                 bar_width=0.33,
                 func=None,
                 debug=False,
                 pdb=False,
                 # Debug this class specifically.
                 verbose_debug=False,
                 ):


        self.data = data
        self.path = path

        self.x_field = x_field
        self.y_field = y_field
        self.x_group = x_group

        self.yerr = yerr
        self.hatch = hatch
        self.hue = hue
        if x_order_by is None:
            x_order_by = self.x_field
        self.x_order_by = x_order_by
        self.hatch_styles = hatch_styles
        # Must provide at least one of these as true.
        assert ( hues_together or hatches_together )
        # Must provide EXACTLY ON of these as true.
        assert not( hues_together and hatches_together )
        self.hues_together = hues_together
        self.hatches_together = hatches_together

        self.bar_width = bar_width

        self.func = func
        self.debug = debug
        self.pdb = pdb
        self.verbose_debug = verbose_debug

        # Either provide both or neither.
        assert ( y2_field is None and data2 is None ) or \
               ( y2_field is not None and data2 is not None )
        # BOTH data and data2 should contain x_field since they will share an x-axis.
        if data2 is not None:
            assert x_field in data2
        assert x_field in data
        self.y2_field = y2_field
        self.n_y2_ticks = n_y2_ticks
        self.data2 = data2

        self.empty_hatch_value = empty_hatch_value
        self.bar_label_func = bar_label_func
        self.y_lim_scale_factor = y_lim_scale_factor
        self.hack_upper_right_legend_bbox_x = hack_upper_right_legend_bbox_x
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.y2label = y2label
        self.title = title
        self.width = width
        self.height = height

        self._init_x_offsets()

        self.hatch_map = self.as_hatch_map(self.hatches(self.data), hatch_styles=self.hatch_styles)
        # if self.debug:
        #     logger.debug("hatch_map = {msg}".format(
        #         msg=pprint_msg(self.hatch_map),
        #     ))
        # if self.pdb:
        #     import pdb; pdb.set_trace()
        if self.hue is None:
            self.hue_map = {
                None: 'white',
            }
        else:
            self.hue_map = self.as_color_map(self.hues(self.data))

        # Perform pre-computation of plot data aggregated into mean value
        self._agg_data = self._compute_agg_data()

    def _init_x_offsets(self):
        x_groups = self.x_groups(self.data)

        total_bar_width = 2/3
        # width per bar:
        # bar_width = 0.33
        if len(x_groups) > 0:
            self.bar_width = total_bar_width/len(x_groups)
        else:
            # Prevent division by zero for empty dataframes.
            self.bar_width = 0.33

        bar_xloc = self._calc_bar_xloc(len(x_groups), self.bar_width)
        x_group_offset = dict()
        for x_group, offset in zip(x_groups, bar_xloc):
            x_group_offset[x_group] = offset
        self.x_group_offset = x_group_offset

        x_field_offset = dict()
        for offset, x_field in enumerate(self.x_fields(self.data)):
            x_field_offset[x_field] = offset
        self.x_field_offset = x_field_offset

    def _xtick(self, x_field, x_group):
        return self.x_field_offset[x_field] + self.x_group_offset[x_group]

    @property
    def _plot_fields(self):
        """
        All the fields used in the plot (including y_field and y2_field).
        """
        fields = self._label_fields
        fields.append(self.y_field)
        if self.y2_field is not None and self.y2_field != self.y_field:
            fields.append(self.y2_field)
        return fields

    @property
    def _label_fields(self):
        """
        All the fields used in the plot except y_field and y2_field.
        """
        fields = []
        if self.hatch is not None:
            fields.append(self.hatch)
        if self.hue is not None:
            fields.append(self.hue)
        fields.append(self.x_field)
        fields.append(self.x_group)
        return fields

    def _compute_agg_data(self):
        fields = self._plot_fields
        label_fields = self._label_fields
        df = self.data[fields]
        # Find the mean of each "slice" of each "bar" in the plot across the repetitions.
        mean_df = df.groupby(label_fields).agg('mean').reset_index()
        # Aggregate within each x_field and x_group to find the height of each "bar"
        agg_df = mean_df.groupby([self.x_field, self.x_group]).agg('sum').reset_index()
        return agg_df

    def _bar_height(self, x_field, x_group):
        # Select a specific "bar"
        agg_df = self._agg_data
        row = agg_df[
            (agg_df[self.x_field] == x_field) &
            (agg_df[self.x_group] == x_group)
        ]
        # Get the height of the "bar"
        ys = row[self.y_field].values
        if ys.shape == (0,):
            # This can happen when adding bar labels.
            # If we cannot find a particular (x_field, x_group), default to zero.
            # NOTE: if we re-use this for filling in plot data, we should probably
            # change the behaviour for plot data to "error out"
            height = 0
            return height
        assert ys.shape == (1,)
        height = ys[0]
        return height

    @staticmethod
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

    def x_fields(self, data):
        if self.x_field == self.x_order_by:
            x_fields = sorted(data[self.x_field].unique())
        else:
            sorted_df = data[[self.x_field, self.x_order_by]].drop_duplicates().sort_values(by=[self.x_order_by])
            x_fields = sorted_df[self.x_field].unique()
        return x_fields

    def x_groups(self, data):
        x_groups = sorted(data[self.x_group].unique())
        return x_groups

    def hatches(self, data):
        hatches = sorted(data[self.hatch].unique())
        return hatches

    def hues(self, data):
        if self.hue is None:
            return [None]
        hues = sorted(data[self.hue].unique())
        return hues

    def as_hatch_map(self, xs, hatch_styles=None):


        # Need enough distinct hash-styles to fit categories.
        # assert len(xs) <= len(ALL_HATCH_STYLES)
        if len(xs) > len(ALL_HATCH_STYLES):
            raise RuntimeError("ERROR: We only have {h} HATCH_STYLES, but there are {x} category-overlap labels".format(
                h=len(ALL_HATCH_STYLES),
                x=len(xs),
            ))

        hatch_map = dict()
        if hatch_styles is not None:
            for x in set(xs).intersection(hatch_styles.keys()):
                hatch_map[x] = hatch_styles[x]

        h = 0
        for x in xs:
            if x in hatch_map:
                continue

            if self.empty_hatch_value is not None and x == self.empty_hatch_value:
                hatch_map[x] = HATCH_STYLE_EMPTY
            else:
                while h < len(ALL_HATCH_STYLES) and ALL_HATCH_STYLES[h] in hatch_map.values():
                    # Find the next available hatch style
                    h += 1
                hatch_map[x] = ALL_HATCH_STYLES[h]
                h += 1
        return hatch_map

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

    def plot(self):
        figsize = None
        if self.width is not None or self.height is not None:
            figsize = (self.width, self.height)
        fig = plt.figure(figsize=figsize)
        self.fig = fig
        # Q: What's this affect?
        # ax = plt.add_subplot(111)

        ax2 = None
        if self.y2_field is None:
            ax = fig.add_subplot(1, 1, 1)
            self.ax = ax
        else:

            # # bottom (bigger)
            # ax = fig.add_subplot(2, 1, 2)
            # # top (smaller)
            # ax2 = fig.add_subplot(2, 1, 1, sharex=ax)

            # gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
            gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 2.5])
            # top
            ax_0 = plt.subplot(gs[0])
            # bottom
            ax_1 = plt.subplot(gs[1], sharex=ax_0)

            # top
            ax2 = ax_0
            # bottom
            ax = ax_1

            plt.setp(ax2.get_xticklabels(), visible=False)
            fig.subplots_adjust(hspace=0.04)

            if self.n_y2_ticks is not None:
                ax2.yaxis.set_major_locator(plt.MaxNLocator(self.n_y2_ticks))

            # ax2.grid(b=False, axis='x')
            self.ax = ax
        self.ax2 = ax2


        sort_by = []
        if self.hatch is not None:
            sort_by.append(self.hatch)
        if self.hue is not None:
            sort_by.append(self.hue)
        # if self.x_field is not None:
        if self.x_order_by not in self.data.keys():
            raise RuntimeError("Didn't find --x-order-by={x_order_by} in dataframe; columns are {cols}".format(
                x_order_by=self.x_order_by,
                cols=sorted(self.data.keys()),
            ))
        sort_by.append(self.x_order_by)
        data = self.data.sort_values(by=sort_by)

        # PSEUDOCODE:
        # bottom = zeroes(len(ys))
        # data.sort_values(by=[self.hatch, self.hue]
        # for hatch, hatch_df in data.groupby(by=[self.hatch]):
        #   for hue, hue_df in hatch_df.groupby(by=[self.hue])
        #     xs = hue_df[self.x]
        #     ys = hue_df[self.y]
        #     ax.bar(xs=xs, ys=ys, bottom=bottom,
        #       color=self.color_map[hue],
        #       hatch=self.hatch_map[hatch)
        #     bottom += ys

        if self.hues_together:
            self._plot_hues_together(data, fig, ax)
        else:
            raise NotImplementedError()
            assert self.hatches_together
            self._plot_hatches_together(data, fig, ax)

        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.title is not None:
            if ax2 is not None:
                ax2.set_title(self.title)
            else:
                ax.set_title(self.title)

        if self.y2_field is not None:
            self._add_legend(ax2)
        else:
            self._add_legend(ax)
        self._add_xgroup_legend(ax, ax2)

        if self.y2_field is not None:

            if self.debug:
                ss = StringIO()
                DataFrame.print_df(self.data2, file=ss)
                logger.debug(f"data2\n{ss.getvalue()}")

            all_x_fields = np.array(self.x_fields(data))
            set_all_x_fields = set(all_x_fields)

            keep_data2 = self.data2[
                self.data2[self.x_field].apply(lambda x_field: x_field in set_all_x_fields)
            ]

            if self.debug:
                ss = StringIO()
                DataFrame.print_df(keep_data2, file=ss)
                logger.debug(f"data2[{self.x_field} == {all_x_fields}]\n{ss.getvalue()}")

            # TODO: transform into "xs", "ys", and "yerr" ; treat different for multi vs single repetition?

            ys_mean = []
            ys_std = []
            has_repetitions = False
            for x_field in all_x_fields:
                y_repetitions = keep_data2[keep_data2[self.x_field] == x_field][self.y2_field]
                if len(y_repetitions) > 1:
                    has_repetitions = True
                y_mean = np.mean(y_repetitions)
                y_std = np.std(y_repetitions)
                ys_mean.append(y_mean)
                ys_std.append(y_std)


            xs = np.arange(len(all_x_fields))
            plot_kwargs = dict(
                x=xs, height=ys_mean,
                width=self.bar_width,
                # Color of hatch pattern.
                edgecolor='black',
                color='black',
                # hatch=self.hatch_map[hatch]
            )

            if has_repetitions and self.yerr:
                plot_kwargs['yerr'] = ys_std
                if 'error_kw' not in plot_kwargs:
                    plot_kwargs['error_kw'] = dict()
                plot_kwargs['error_kw']['ecolor'] = DetailedStackedBarPlot.ERRBAR_COLOR
                plot_kwargs['error_kw']['capsize'] = DetailedStackedBarPlot.ERRBAR_CAPSIZE_POINTS

            if self.y2label is not None:
                ax2.set_ylabel(self.y2label)
            ax2.bar(**plot_kwargs)

            # if has_repetitions:
            #     capsize = get_capsize(ax2)
            #     if 'error_kw' not in plot_kwargs:
            #         plot_kwargs['error_kw'] = dict()
            #     plot_kwargs['error_kw']['capsize'] = capsize
            #     ax2.bar(**plot_kwargs)

        if self.func is not None:
            self.func(self)

        logger.info("Output csv @ {path}".format(path=self._csv_path))
        data.to_csv(self._csv_path, index=False)

        logger.info("Output dataframe @ {path}".format(path=self._df_path))
        with open(self._df_path, 'w') as f:
            DataFrame.print_df(data, file=f)
        DataFrame.print_df(data, file=sys.stdout)

        logger.info("Output plot @ {path}".format(path=self.path))
        # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
        #                     hspace = 0, wspace = 0)
        # plt.margins(0,0)
        fig.savefig(self.path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        crop_pdf(self.path)
        # NOTE: svg files are MB's in size (not sure why)... PNG is KB in size and resolution is fine.
        # pdf2svg(self.path)
        pdf2png(self.path)

        # self._add_lines(operation)
        # self._add_legend(operation)
        # self._add_axis_labels(operation)
        # self._show(fig, operation)

    def _plot_hatches_together(self, data, fig, ax):
        # fig = plt.figure()
        # # Q: What's this affect?
        # # ax = plt.add_subplot(111)
        # ax = fig.add_subplot(111)

        all_xs = []
        all_ys = []
        bar_kwargs = []
        # TODO: inter-change hue/hatch loops..?
        last_iter = None
        # NOTE: we need to use DIFFERNT bottom's for each x_group!
        # ... so iterate over x_group FIRST.
        # Q: Different x-fields may not have a particular x-group...?
        # We are ASSUMING that EVERY (algo,env) has (BP,Sim,Inf)

        all_x_fields = self.x_fields(data)
        ax.set_xticks(np.arange(len(all_x_fields)))
        ax.set_xticklabels(all_x_fields
                           # , rotation=rotation
                           )

        x_groups = self.x_groups(data)
        for x_group in x_groups:
            bottom = None
            xgroup_df = data[data[self.x_group] == x_group]
            hatches = self.hatches(xgroup_df)
            for hatch in hatches:
                hatch_df = xgroup_df[xgroup_df[self.hatch] == hatch]
                hues = self.hues(hatch_df)
                for hue in hues:
                    if self.hue is not None:
                        hue_df = hatch_df[hatch_df[self.hue] == hue]
                    else:
                        hue_df = hatch_df
                    x_fields = hue_df[self.x_field]
                    x_groups = hue_df[self.x_group]
                    xs = np.vectorize(self._xtick, otypes=[np.float])(hue_df[self.x_field], hue_df[self.x_group])
                    ys = hue_df[self.y_field].values
                    all_xs.append(xs)
                    all_ys.append(ys)
                    plot_kwargs = dict(
                        x=xs, height=ys, bottom=bottom,
                        width=self.bar_width,
                        # Color of hatch pattern.
                        edgecolor='black',
                        color=self.hue_map[hue],
                        hatch=self.hatch_map[hatch]
                    )
                    ax.bar(**plot_kwargs)
                    kw = dict(plot_kwargs)
                    kw.update({
                        'hue_field': hue,
                        'hatch_field': hatch,
                    })
                    bar_kwargs.append(kw)
                    if bottom is None:
                        bottom = np.zeros(len(ys))
                    assert not np.isnan(bottom).any()
                    assert not np.isnan(ys).any()
                    assert not np.isnan(bottom + ys).any()
                    bottom = bottom + ys
                    assert not np.isnan(bottom).any()

                    last_iter = dict(
                        x_fields=x_fields,
                        x_groups=x_groups,
                        xs=xs,
                        ys=ys,
                        bottom=bottom,
                    )

        logger.info("Plot debug:\n{msg}".format(
            msg=pprint_msg({
                'bar_kwargs': bar_kwargs,
            }),
        ))


    def _plot_hues_together(self, data, fig, ax):

        all_xs = []
        all_ys = []
        bar_kwargs = []
        # TODO: inter-change hue/hatch loops..?
        last_iter = None
        # NOTE: we need to use DIFFERNT bottom's for each x_group!
        # ... so iterate over x_group FIRST.
        # Q: Different x-fields may not have a particular x-group...?
        # We are ASSUMING that EVERY (algo,env) has (BP,Sim,Inf)
        x_groups = self.x_groups(data)

        all_x_fields = self.x_fields(data)
        ax.set_xticks(np.arange(len(all_x_fields)))
        ax.set_xticklabels(all_x_fields
                           # , rotation=rotation
                           )

        # PSEUDOCODE:
        # x_fields = self.x_fields(self.df)
        # for x_field in x_fields:
        #     x_field_df = self.df[self.df[self.x_field] == x_field]
        #     x_groups = self.x_groups(x_field_df)
        #     for x_group in x_groups:
        #         x_group_df = x_field_df[x_field_df[self.x_group] == x_group]
        #         hues = self.hues(x_group_df)
        #         for hue in hues:
        #             hue_df = x_group_df[x_group_df[self.hue] == hue]
        #             hatches = self.hatches(hue_df)
        #             for hatch in hatches:
        #                 hatch_df = hue_df[hue_df[self.hatch] == hatch]

        x_fields = self.x_fields(data)
        for x_field in x_fields:
            x_field_df = data[data[self.x_field] == x_field]
            x_groups = self.x_groups(x_field_df)
            for x_group in x_groups:
                bottom = None
                x_group_df = x_field_df[x_field_df[self.x_group] == x_group]
                hues = self.hues(x_group_df)
                for hue in hues:
                    if self.hue is not None:
                        hue_df = x_group_df[x_group_df[self.hue] == hue]
                    else:
                        hue_df = x_group_df
                    hatches = self.hatches(hue_df)
                    for hatch in hatches:
                        hatch_df = hue_df[hue_df[self.hatch] == hatch]
                        xs = np.vectorize(self._xtick, otypes=[np.float])(hatch_df[self.x_field], hatch_df[self.x_group])
                        ys = hatch_df[self.y_field].values

                        assert len(ys) == len(xs)
                        assert len(set(xs)) == 1

                        # color = self.hue_map[hue]
                        # if self.hue is None:
                        #     assert color is None
                        plot_kwargs = dict(
                            # x=xs, height=ys,
                            bottom=bottom,
                            width=self.bar_width,
                            # Color of hatch pattern.
                            edgecolor='black',
                            color=self.hue_map[hue],
                            hatch=self.hatch_map[hatch]
                        )

                        # assert len(ys) == 1
                        if len(ys) == 1:
                            # No error bars.
                            plot_xs = xs
                            plot_ys = ys
                            all_xs.append(xs)
                            all_ys.append(ys)
                            plot_kwargs.update(dict(
                                x=xs,
                                height=ys,
                            ))
                        else:
                            # Multiple values; use error bars.
                            x = xs[0]
                            y = np.mean(ys)
                            plot_xs = [x]
                            plot_ys = [y]
                            if self.yerr:
                                yerr = np.std(ys)
                                plot_kwargs.update(dict(
                                    # x=[x],
                                    # height=[y],
                                    yerr=[yerr],
                                    capsize=DetailedStackedBarPlot.ERRBAR_CAPSIZE_POINTS,
                                    # ecolor='gray',
                                    ecolor=DetailedStackedBarPlot.ERRBAR_COLOR,
                                    # NOTE: I cannot get dashed lines to work with the errorbar style...
                                    # error_kw=dict(
                                    #     # dash_capstyle='projecting',
                                    #     linestyle='-',
                                    #     # dashes=(0.5, 0.5),
                                    #     dashes=(6, 2),
                                    # ),
                                ))
                        plot_kwargs.update(dict(
                            x=plot_xs,
                            height=plot_ys,
                        ))
                        all_xs.append(plot_xs)
                        all_ys.append(plot_ys)

                        barplot = ax.bar(**plot_kwargs)

                        # if 'yerr' in plot_kwargs:
                        #     # errline = barplot.errorbar.get_children()[2]
                        #     # errline = barplot.errorbar.lines[1][0]
                        #     for errline in barplot.errorbar.get_children():
                        #         errline.set_linestyle('-')
                        #         # errline.set_dashes((0.5, 0.5))
                        #         # errline.set_dashes((0, (6, 2)))
                        #         # errline.set_dashes('--')
                        #         # errline.set_dashes((6, 2))
                        #         # errline.set_dashes((2, 2, 10, 2))

                        kw = dict(plot_kwargs)
                        kw.update({
                            'hue_field': hue,
                            'hatch_field': hatch,
                            'x_group': x_group,
                            'x_field': x_field,
                        })
                        bar_kwargs.append(kw)
                        if bottom is None:
                            bottom = np.zeros(len(plot_ys))
                        assert not np.isnan(bottom).any()
                        assert not np.isnan(ys).any()
                        assert not np.isnan(plot_ys).any()
                        assert not np.isnan(bottom + plot_ys).any()
                        assert len(bottom) == len(plot_ys)
                        bottom = bottom + plot_ys
                        assert not np.isnan(bottom).any()

                        last_iter = dict(
                            x_fields=x_fields,
                            x_groups=x_groups,
                            xs=xs,
                            ys=ys,
                            bottom=bottom,
                        )

        if self.verbose_debug:
            logger.debug("Plot debug:\n{msg}".format(
                msg=pprint_msg({
                    'bar_kwargs': bar_kwargs,
                }),
            ))

    def _old_plot_hues_together(self, data, fig, ax):

        all_xs = []
        all_ys = []
        bar_kwargs = []
        # TODO: inter-change hue/hatch loops..?
        last_iter = None
        # NOTE: we need to use DIFFERNT bottom's for each x_group!
        # ... so iterate over x_group FIRST.
        # Q: Different x-fields may not have a particular x-group...?
        # We are ASSUMING that EVERY (algo,env) has (BP,Sim,Inf)
        x_groups = self.x_groups(data)

        all_x_fields = self.x_fields(data)
        ax.set_xticks(np.arange(len(all_x_fields)))
        ax.set_xticklabels(all_x_fields
                           # , rotation=rotation
                           )

        for x_group in x_groups:
            bottom = None
            xgroup_df = data[data[self.x_group] == x_group]
            hues = self.hues(xgroup_df)
            for hue in hues:
                hue_df = xgroup_df[xgroup_df[self.hue] == hue]
                hatches = self.hatches(hue_df)
                for hatch in hatches:
                    hatch_df = hue_df[hue_df[self.hatch] == hatch]
                    # xs = hatch_df[self.x_field].values
                    x_fields = hatch_df[self.x_field]
                    x_groups = hatch_df[self.x_group]
                    xs = np.vectorize(self._xtick, otypes=[np.float])(hatch_df[self.x_field], hatch_df[self.x_group])
                    ys = hatch_df[self.y_field].values
                    all_xs.append(xs)
                    all_ys.append(ys)
                    plot_kwargs = dict(
                        x=xs, height=ys, bottom=bottom,
                        width=self.bar_width,
                        # Color of hatch pattern.
                        edgecolor='black',
                        color=self.hue_map[hue],
                        hatch=self.hatch_map[hatch]
                    )
                    ax.bar(**plot_kwargs)
                    kw = dict(plot_kwargs)
                    kw.update({
                        'hue_field': hue,
                        'hatch_field': hatch,
                        'x_groups': list(x_groups.values),
                        'x_fields': list(x_fields.values),
                    })
                    bar_kwargs.append(kw)
                    if bottom is None:
                        bottom = np.zeros(len(ys))
                    assert not np.isnan(bottom).any()
                    assert not np.isnan(ys).any()
                    assert not np.isnan(bottom + ys).any()
                    assert len(bottom) == len(ys)
                    bottom = bottom + ys
                    assert not np.isnan(bottom).any()

                    last_iter = dict(
                        x_fields=x_fields,
                        x_groups=x_groups,
                        xs=xs,
                        ys=ys,
                        bottom=bottom,
                    )

        logger.info("Plot debug:\n{msg}".format(
            msg=pprint_msg({
                'bar_kwargs': bar_kwargs,
            }),
        ))


    def _as_path(self, file_ext):
        path = self.path
        path = re.sub(r'\.[^.]+$', file_ext, path)
        assert path != self.path
        return path

    @property
    def _csv_path(self):
        return self._as_path('.csv')

    @property
    def _df_path(self):
        return self._as_path('.dataframe.txt')

    def get_plot_operations(self):
        if self.get_png is not None:
            all_benches = [NO_BENCH_NAME] + unique(self.mean_df['operation'])
            return all_benches

        return [NO_BENCH_NAME]

    def get_png_path(self, operation):
        if self.get_png is not None:
            return self.get_png(operation)

        return self.png

    def get_png_legend_path(self, operation):
        png_path = self.get_png_path(operation)
        legend_png_path = re.sub(r'\.png$', '.legend.png', png_path)
        return legend_png_path

        # if self.get_png_legend is not None:
        #     return self.get_png_legend(operation)
        #
        # return self.png_legend

    def get_plot_data_pt(self, operation):
        if self.get_plot_data_path is not None:
            return self.get_plot_data_path(operation)

        return self.plot_data_path

    def _show(self, fig, operation=None):
        if self.show:
            plt.show()
        else:
            print("> Save figure to {path}".format(path=self.get_png_path(operation)))
            print("> Save plot data to {path}".format(path=self.get_plot_data_pt(operation)))
            fig.savefig(self.get_png_path(operation), bbox_inches="tight")
            plt.close()

    def _add_lines(self, operation=None):

        self._bottom = None
        def _add_line(impl_name, operation):
            for category in self.category_order:
                rows = self.df[
                    (self.df['operation'] == operation)
                    & (self.df['category'] == category)
                    ]
                if len(rows) == 0:
                    continue
                xvalues = self._get_xvalues(rows['impl_name'], rows['device'])
                yvalues = rows['mean'].values
                yerr = rows['std'].values

                color = self.category_color_map[category]
                hatch = self.operation_hatch_map[operation]

                if self._bottom is None:
                    self._bottom = np.zeros_like(yvalues)

                # PROBLEM: if data is missing for step
                assert self._bottom.shape == yvalues.shape

                plot = plt.bar(xvalues, yvalues, color=color, width=self.bar_width, edgecolor='white', label=operation,
                               bottom=self._bottom,
                               hatch=hatch,
                               yerr=yerr)

                self._bottom += yvalues

        for impl_name in self.impl_name_order:
            if operation == NO_BENCH_NAME:
                for operation in self.operation_order:
                    _add_line(impl_name, operation)
            else:
                _add_line(impl_name, operation)

    def _add_xgroup_legend(self, ax, ax2, x_offset=0, y_offset=0):

        # legbox_spacer = 0.04
        legbox_spacer = 0
        legend_y_spacer = 0.025
        # Place legend at top-left of plot-area.
        legend_ax = ax
        # legend_ax = ax2
        legend_kwargs = dict(

            # loc='best',

            # loc='upper left',
            # bbox_to_anchor=(0 + legbox_spacer, 1 - legbox_spacer),

            # relative to bottom plot (ax)
            loc='lower left',
            bbox_to_anchor=(
                # 1 + legbox_spacer,
                # 0 - legbox_spacer,
                1,
                0 - legend_y_spacer,
            ),

            ncol=1,
            handlelength=0,
            handletextpad=0,
        )

        annotate_kwargs = dict(
            color='black',
            weight='bold',
        )

        patch_kwargs = dict(
            edgecolor='black',
            # facecolor='white',
        )

        def get_bar_label(i, x_group):
            if self.bar_label_func is not None:
                bar_label = self.bar_label_func(i, x_group)
            else:
                bar_label = "$({i})$".format(i=i + 1)
            return bar_label

        def _add_xgroup_bar_labels():
            x_groups = self.x_groups(self.data)
            x_fields = self.x_fields(self.data)
            # Just label the first x-field with (1), (2), (3)
            # OR: BP, Inf, Sim
            x_field = x_fields[0]

            if len(x_fields) == 1 or self.y_lim_scale_factor is not None:
                # HACK: provide enough space for the bar-labels on the first x_field.
                # When the number of x_fields is 1, it always overlap with the top plot area bounding box.
                # So, add an extra 5% of head-room

                if self.y_lim_scale_factor is not None:
                    y_min, y_max = ax.get_ylim()
                    ax.set_ylim([y_min, y_max*self.y_lim_scale_factor])
                else:
                    head_room_percent = 5
                    y_min, y_max = ax.get_ylim()
                    ax.set_ylim([y_min, y_max*(1 + head_room_percent/100)])

            df = self.data
            xy_offset = (x_offset, y_offset)
            for i, x_group in enumerate(x_groups):
                x = self._xtick(x_field, x_group)
                height = self._bar_height(x_field, x_group)
                logger.info("Add bar-label:\n{msg}".format(msg=pprint_msg({
                    'x_field': x_field,
                    'x_group': x_group,
                    'height': height,
                    'x': x,
                })))
                label = get_bar_label(i, x_group)
                ax.annotate(
                    label,
                    # xy=(rect.get_x() + rect.get_width() / 2, height),
                    xy=(x, height),
                    xytext=xy_offset,
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    **annotate_kwargs)

        def _add_xgroup_legbox():
            x_groups = self.x_groups(self.data)
            patches = []
            for i, x_group in enumerate(x_groups):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message=r"Setting the 'color' property will override the edgecolor or facecolor properties.", module=r'rlscope')
                    patch = mpatches.Patch(edgecolor='white', color='white')
                # patch = mpl_patches.Rectangle((0, 0), 0, 0, fc="white", ec="white",
                #                       lw=0, alpha=0)

                # patch = mpatches.Patch(**patch_kwargs)
                # patch = mpatches.Rectangle(**patch_kwargs)

                # label = get_bar_label(i, x_group)
                # # patch = ax.annotate(label, (0, 0),
                # #             ha='center',
                # #             va='center',
                # #             # edgecolor='black',
                # #             # xycoords='offset points'
                # #             **annotate_kwargs)
                # patch = ax.text(0, 0, label,
                #                 ha='center',
                #                 va='center',
                #                 # edgecolor='black',
                #                 # xycoords='offset points'
                #                 **annotate_kwargs)

                patches.append(patch)

            # labels = x_groups

            def _get_label(i, x_group):
                bar_label = get_bar_label(i, x_group)
                # txt = r"\textbf{{{bar_label}}} = {x_group}".format(
                #     bar_label=bar_label,
                #     x_group=x_group)
                txt = "{bar_label} {x_group}".format(
                    bar_label=bar_label,
                    x_group=x_group)
                return txt
            labels = [_get_label(i, x_group) for i, x_group in enumerate(x_groups)]

            if self.debug:
                logger.debug("xgroup legend_kwargs = {msg}".format(msg=pprint_msg(legend_kwargs)))
            legend = legend_ax.legend(handles=patches, labels=labels, **legend_kwargs)
            legend_ax.add_artist(legend)

            # for i, (x_group, patch) in enumerate(zip(x_groups, patches)):
            #     bar_label = get_bar_label(i, x_group)
            #
            #     rx, ry = patch.get_xy()
            #     cx = rx + patch.get_width()/2.0
            #     cy = ry + patch.get_height()/2.0
            #
            #     # ax.annotate(bar_label, (cx, cy),
            #     #             ha='center',
            #     #             va='center',
            #     #             **annotate_kwargs)
            #
            #     txt = r"$\textbf{bar_label}$ = {x_group}".format(
            #         bar_label=bar_label,
            #         x_group=x_group)
            #     ax.annotate(
            #         txt, (cx, cy),
            #         ha='center',
            #         va='center',
            #         **annotate_kwargs)

            return legend

        _add_xgroup_bar_labels()
        _add_xgroup_legbox()




    def _add_legend(self, axis):
        """
        https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend

        - loc tells us what part of the legend's bounding box we are positioning.
          choices:
          - 9 combinations of: 'upper/lower/""' + 'left/center/right'
          - 'best'
            Q: What does "best" refer to?
            'best' places the legend at the location, among the nine locations defined so far, with the minimum overlap with other drawn artists

        - bbox_to_anchor tells us what position in the axis/figure we are placing the "loc" of the legend.
          choices:
          - (x, y)
          - (x, y, width, height)
            Q: What do width and height do?

        loc='upper right', bbox_to_anchor=(0.5, 0.5)
        Place the 'upper right' corner of the legend at x=0.5, y=0.5.

        loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5)
        "best location" at "bottom right quadrant of the axes"

        So, to place legend outside plotting area, at the top:
            loc='upper left', bbox_to_anchor=(x=1, y=1)

            (Do this for hatches and hues)

        So, to place legend outside plotting area, at the bottom:
            loc='bottom left', bbox_to_anchor=(x=1, y=0)

            (Do this for xgroup labels)

        :param axis:
        :return:
        """
        self.legend_makers = []

        reversed_labels = False

        # Sometimes they are so many legend labels that the two separate legend boxes will overlap,
        # and it's hard to position the legend boxes "on top of each other".
        # So, we're better off making a single legend box.
        single_legend = True

        common_legend_kwargs = {
            'fancybox':True,
            # 'shadow':True,
            'handlelength': 3,
            'handleheight': 2,
            # 'borderpad': 0.0,
            'handletextpad': 0.4,
            'labelspacing': 0.35,
        }

        legend_spacer = 0
        # For some reason, I need to add this to adjust the x-axis of the upper right legend...no idea why.
        # legend_spacer = 0.365
        if self.hack_upper_right_legend_bbox_x is not None:
            legend_spacer += self.hack_upper_right_legend_bbox_x

        legend_y_spacer = 0.02
        # legend_y_spacer = 0

        # We need two groups of lines:
        #
        # 1) Hatch-type:
        #    - Should have the same color
        #    - # of hash-types = len(category_order = ['GPUTimeSec', 'CppTimeSec', 'PythonTimeSec'])
        #                      = 3
        #
        # 2) Color-type:
        #    - Should have the same hatch.
        #    - # of color-types = len(operation_order = ['q_forward', 'q_backward', 'step'])
        #                       = 3

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        if not single_legend:
            legend_kwargs = []

        hatch_map = dict(self.hatch_map)
        hatches = [hatch for hatch in self.hatches(self.data) if hatch != ""]
        for hatch in list(self.hatch_map.keys()):
            # if hatch_map[hatch] == HATCH_STYLE_EMPTY:
            if hatch == "":
                del hatch_map[hatch]
        hatch_legend = LegendMaker(attr_name='hatch',
                                   field_to_attr_map=hatch_map,
                                   field_order=hatches,
                                   # field_to_attr_map=self.hatch_map,
                                   # field_order=hatches(self.data),
                                   # labels=self.hatches,
                                   legend_kwargs={
                                       # 'labelspacing': 1.2,
                                       # 'handlelength': 3,
                                       # 'handleheight': 2,
                                   },
                                   reversed=reversed_labels)
        self.legend_makers.append(hatch_legend)
        if not single_legend:
            kwargs = dict(common_legend_kwargs)
            kwargs.update({
                    # NOTE:
                    # - Internally LegendMaker uses the figure coordinate system.
                    # - So, (1, 1) is the (right, top) of the whole figure,
                    #   so 1.04 makes it just a bit to the right of the whole figure
                    'bbox_to_anchor': (1 + legend_spacer, 1),
            })
            legend_kwargs.append(kwargs)

        if self.hue is not None:
            color_legend = LegendMaker(attr_name='facecolor',
                                       field_to_attr_map=self.hue_map,
                                       field_order=self.hues(self.data),
                                       # labels=self.hues(self.data),
                                       edgecolor='white',
                                       legend_kwargs={
                                           # 'handlelength': 3,
                                           # 'handleheight': 2,
                                       },
                                       reversed=reversed_labels)
            self.legend_makers.append(color_legend)
        if not single_legend:
            kwargs = dict(common_legend_kwargs)
            kwargs.update({
                # 'loc':'lower left',
                # 'loc':'left',
                # 'bbox_to_anchor': (0, -1),

                # Place legend beneath plot so it has room to grow when there are lots of labels,
                # without overlapping the other legend.
                # Sadly, I still don't understand how this thing works.
                # (e.g. I'm not sure how to left-align the legend beneath the plot... OH WELL).
                #
                # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
                # 'loc':'upper center',
                # 'bbox_to_anchor':(0.5, -0.05),

                'loc':'lower left',
                'bbox_to_anchor':(1 + legend_spacer, 0.0),

            })
            legend_kwargs.append(kwargs)

        if not single_legend:
            LegendMaker.add_legends_multiple(
                self.legend_makers,
                axis=axis,
                legend_kwargs=legend_kwargs)

        if single_legend:
            kwargs = dict(common_legend_kwargs)
            kwargs.update({
                'loc':'upper left',
                'bbox_to_anchor':(1 + legend_spacer, 1 + legend_y_spacer),
            })
            if self.debug:
                logger.debug("hue/hatch legend_kwargs = {msg}".format(msg=pprint_msg(kwargs)))
            LegendMaker.add_legends_single(
                self.legend_makers,
                axis=axis,
                legend_kwargs=kwargs)

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
        self._build_operation_order()

        # devices = list(data.keys())
        self.orig_df = pd.DataFrame(self.df_data)

        self.df = DataFrame.get_mean_std(self.orig_df, self.value_field)
        logger.info("> DATAFRAME BEFORE SORT:")
        logger.info(self.df)
        self.df = self.df.sort_values(by=['impl_name_order', 'operation_order', 'category_order'])
        # self.df = self.df.sort_values(by=['impl_name_order', 'operation_order', 'category_order'], ascending=False)
        # self.df = self.df.sort_values(by=['operation_order'])
        logger.info("> DATAFRAME AFTER SORT:")
        logger.info(self.df)
        # groupby_cols = DataFrame.get_groupby_cols(self.orig_df, value_field)
        self.df['std_div_mean_percent'] = 100 * self.df['std']/self.df['mean']

        self.mean_df = self.df

    def _add_axis_labels(self, operation=None):
        if self.title is not None:
            plt.title(self.title)

        if self.xlabel is not None:
            # , fontweight='bold'
            if operation == NO_BENCH_NAME:
                plt.xlabel(self.xlabel)
            else:
                plt.xlabel(get_pretty_bench(operation))

        if self.ylabel is not None:
            plt.ylabel(self.ylabel)


        n_bars = len(self.data[self.x_field].unique())
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

    def _check_has_keys(self, xs, xs_map):
        for x in xs:
            assert x in xs_map


# Unused class; just some documentation
class RegionDataFrame:
    """
    Here's what the data-frame looks like for:
    # OverlapStackedBarPlot.overlap_type_OperationOverlap.txt
    $ train_stable_baselines.sh --env-id Walker2DBulletEnv-v0

       (compute_advantage_estimates,)  (sample_action,)                                  env  algo  (optimize_surrogate,)                                      x_field
    0                        0.007635         12.188267                      AntBulletEnv-v0  ppo2              87.804098                      (ppo2, AntBulletEnv-v0)
    1                        0.004305         50.264176              HalfCheetahBulletEnv-v0  ppo2              49.731519              (ppo2, HalfCheetahBulletEnv-v0)
    2                        0.002083         29.895073                   HopperBulletEnv-v0  ppo2              70.102843                   (ppo2, HopperBulletEnv-v0)
    3                        0.004586         47.460603                 HumanoidBulletEnv-v0  ppo2              52.534812                 (ppo2, HumanoidBulletEnv-v0)
    4                        0.005242         49.675716   InvertedDoublePendulumBulletEnv-v0  ppo2              50.319042   (ppo2, InvertedDoublePendulumBulletEnv-v0)
    5                        0.005556         47.268527  InvertedPendulumSwingupBulletEnv-v0  ppo2              52.725917  (ppo2, InvertedPendulumSwingupBulletEnv-v0)
    6                        0.005479         46.588143                  ReacherBulletEnv-v0  ppo2              53.406377                  (ppo2, ReacherBulletEnv-v0)
    7                        0.004384         58.306973                 Walker2DBulletEnv-v0  ppo2              41.688643                 (ppo2, Walker2DBulletEnv-v0)

    If we add two regions together:
    e.g.
      remap['other'] = regions['sample_action'] +
                       regions['optimize_surrogate'] +
                       regions['compute_advantage_estimates']
    then we want to perform this addition for each matching pair of
    rl-workloads measured.
    i.e.
    NOTE: we could just expose pandas syntax directly to the user.
      new_df[('other',)] = df[('compute_advantage_estimates',)] +
                           df[('optimize_surrogate',)]
    - Syntactic sugar (1): auto-tuplify single-name regions
      PROBLEM: loses generality, cannot reference 'env' or 'algo' now... not a bad thing really.
      new_df['other'] = df['compute_advantage_estimates'] +
                        df['optimize_surrogate']
    - Syntactic sugar (2): remove reference to new_df and df, replace strings with barename
      PROBLEM: loses generality, cannot use python syntax now.
      new_df['other'] = df['compute_advantage_estimates'] +
                        df['optimize_surrogate']

    Region-remapping operations/syntax we wish to support:

    E.g.
        remap['step'] = step
        remap['other'] = sum(r for r in regions if r != step)

    Region-remapping may be specific to a particular algorithm.
    E.g. to obtain generic breakdown from ppo2:

        remap['inference'] = sample_action
        remap['backward_pass'] = optimize_surrogate

    :param other:
    :return:
    """
    def __init__(self, df, groups):
        self.df = df
        self.groups = groups


    # TODO: overload all the things pandas.DataFrame overloads and pass them to self.df
    # Q: Why not just extend pandas.DataFrame?
    # A: because we'd like to retain the base-class instance __setitem__ behaviour...
    #    we CANNOT convert a RegionDataFrame to a pandas.DataFrame since __setitem__ becomes forever overloaded... annoying.

    def __setitem__(self, key, value):
        # TODO: only allow region-names as keys; check if key is a region-name; if not raise a informative error
        # TODO: tuplify string key into a single-element tuple.
        pass


def attempt_stacked_bar_sns():
    """
    https://randyzwitch.com/creating-stacked-bar-chart-seaborn/

    Stacked-bar chart is really just "overlaying" bars on top of each other.

    :return:
    """

    # # Plot parameters
    # x_axis_label = "X-axis label"
    # y_axis_label = "Y-axis label"
    # # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
    # groups = ['Series1', 'Series2']
    def group_to_label(group):
        '''
        Convert a "key" representing a stack into a legend-label.

        :param group:
            e.g. ('CPU',) => 'CPU'
        :return:
        '''
        return group
    # # The value to use for the x-axis
    # x_field = 'Group'
    data = {
        'Group': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],
        'Series1':[
            3.324129347,
            3.109298649,
            3.603703815,
            5.030113742,
            6.555816091,
            7.478125262,
            8.201300407,
            8.306264399,
            6.622167472,
            4.272699487,
            2.393412671,
            1.228434178,
            0.611171616,
            0.30351888,
            0.151165323,
            0.077035502,
            0.039631096,
            0.021068686,
            0.011522874,
            0.006190799,
        ],
        'Series2': [
            8.733030097,
            5.621414891,
            4.830770027,
            4.84697594,
            4.126257144,
            3.321003992,
            2.667565198,
            1.976072963,
            1.220086585,
            0.656404386,
            0.32202138,
            0.15599814,
            0.076945636,
            0.039381467,
            0.021178522,
            0.011502903,
            0.006440428,
            0.003664553,
            0.002106869,
            0.001308056,
        ],
    }

    stacked_bar_plot = StackedBarPlot(
        data=data,
        path='./test_stacked_bar_sns.png',
        groups=['Series1', 'Series2'],
        x_field='Group',
        x_axis_label="X-axis label",
        y_axis_label="Y-axis label",
        # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
        group_to_label=group_to_label,
    )
    stacked_bar_plot.plot()

def attempt_stacked_bar_sns_old():
    """
    https://randyzwitch.com/creating-stacked-bar-chart-seaborn/

    Stacked-bar chart is really just "overlaying" bars on top of each other.

    :return:
    """
    #Read in data & create total column
    data = {
        'Group': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],
        'Series1':[
            3.324129347,
            3.109298649,
            3.603703815,
            5.030113742,
            6.555816091,
            7.478125262,
            8.201300407,
            8.306264399,
            6.622167472,
            4.272699487,
            2.393412671,
            1.228434178,
            0.611171616,
            0.30351888,
            0.151165323,
            0.077035502,
            0.039631096,
            0.021068686,
            0.011522874,
            0.006190799,
        ],
        'Series2': [
            8.733030097,
            5.621414891,
            4.830770027,
            4.84697594,
            4.126257144,
            3.321003992,
            2.667565198,
            1.976072963,
            1.220086585,
            0.656404386,
            0.32202138,
            0.15599814,
            0.076945636,
            0.039381467,
            0.021178522,
            0.011502903,
            0.006440428,
            0.003664553,
            0.002106869,
            0.001308056,
        ],
    }
    stacked_bar_data = pd.DataFrame(data)

    # stacked_bar_data = pd.read_csv("C:\stacked_bar.csv")
    stacked_bar_data["total"] = stacked_bar_data.Series1 + stacked_bar_data.Series2

    #Set general plot properties
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})

    #Plot 1 - background - "total" (top) series
    sns.barplot(x = stacked_bar_data.Group, y = stacked_bar_data.total, color = "red")

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x = stacked_bar_data.Group, y = stacked_bar_data.Series1, color = "#0000A3")


    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})
    l.draw_frame(False)

    #Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("Y-axis label")
    bottom_plot.set_xlabel("X-axis label")

    #Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(16)

    path = './test_stacked_bar_sns_old.png'
    logger.info('Save figure to {path}'.format(path=path))
    plt.savefig(path)

def attempt_double_yaxis():

    # # http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/#sec-3
    # x = np.linspace(0, 2*np.pi)
    # y1 = np.sin(x);
    # y2 = 0.01 * np.cos(x);
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(x, y1)
    # ax1.set_ylabel('y1')
    #
    # ax2 = ax1.twinx()
    # ax2.plot(x, y2, 'r-')
    # ax2.set_ylabel('y2', color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')
    #
    # plt.savefig('./test_double_yaxis.png')

    # https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis
    s = StringIO("""     amount     price
    A     40929   4066443
    B     93904   9611272
    C    188349  19360005
    D    248438  24335536
    E    205622  18888604
    F    140173  12580900
    G     76243   6751731
    H     36859   3418329
    I     29304   2758928
    J     39768   3201269
    K     30350   2867059""")

    df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)

    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.4

    df.amount.plot(kind='bar', color='red', ax=ax, width=width, position=1)
    df.price.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

    ax.set_ylabel('Amount')
    ax2.set_ylabel('Price')

    plt.savefig('./test_double_yaxis.png')


class DaysHoursMinutesSecondsFormatter(mpl_ticker.Formatter):
    """
    Use an old-style ('%' operator) format string to format the tick.

    The format string should have a single variable format (%) in it.
    It will be applied to the value (not the position) of the tick.
    """
    def __init__(self):
        pass

    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        Only the value `x` is formatted. The position is ignored.
        """
        seconds_remaining = x

        SECONDS_PER_MINUTE = 60
        SECONDS_PER_HOUR = SECONDS_PER_MINUTE*60
        SECONDS_PER_DAY = SECONDS_PER_HOUR*24

        days = int(seconds_remaining / SECONDS_PER_DAY)
        seconds_remaining -= days * SECONDS_PER_DAY

        hours = int(seconds_remaining / SECONDS_PER_HOUR)
        seconds_remaining -= hours * SECONDS_PER_HOUR

        minutes = int(seconds_remaining / SECONDS_PER_MINUTE)
        seconds_remaining -= minutes * SECONDS_PER_MINUTE

        seconds = int(seconds_remaining)

        # "5 days, 3 hours"
        # "5 days, 3 hours, 5 seconds"
        # "5 days, 5 seconds"
        values = [days, hours, minutes, seconds]
        names = ['day', 'hr', 'min', 'sec']
        components = []
        assert len(values) == len(names)
        for value, name in zip(values, names):
            if value == 0:
                continue
            string = '{value} {name}'.format(
                value=value, name=name)
            components.append(string)
        if len(components) == 0:
            label = '0'
        else:
            label = ', '.join(components)

        return label

class VdTree(_DataIndex):
    """
    Same interface as DataIndex, except that VdTree represents open instances of venn_js files
    that we can modify inplace, then persist back to disk (with an additional file-suffix added).
    """
    def __init__(self, index, directory,
                 host=None,
                 user=None,
                 password=None,
                 debug=False):
        # VdTree will mutate the index; so create a copy of it.
        index_copy = copy.copy(index)
        super.__init__(index_copy, directory, debug)
        self.host = host
        self.user = user
        self.password = password
        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

    def open_vds(self, debug=False):
        for plot_type in plot_index.OVERLAP_PLOT_TYPES:
            selector = {
                'plot_type': plot_type,
            }
            for md, entry, ident in self.each_file(selector, debug=debug):
                entry['vd'] = VennData(path=entry['venn_js_path'])

    def subtract(self, debug=False):
        pass

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    def _each_operation(self):
        for machine in self.sql_reader.machines():
            machine_name = machine.machine_name
            for process in self.sql_reader.processes(machine_name=machine_name):
                process_name = process.process_name
                for phase in self.sql_reader.phases(machine_name=machine_name,
                                                    process_name=process_name):
                    phase_name = phase.phase_name
                    for operation in self.sql_reader.operations(machine_name=machine_name,
                                                                process_name=process_name,
                                                                phase_name=phase_name):
                        operation_name = operation.operation_name
                        yield machine_name, process_name, phase_name, operation_name

    # def subtract_overhead(self,
    #                       overhead_event_count_json,
    #                       cupti_overhead_json,
    #                       LD_PRELOAD_overhead_json,
    #                       pyprof_overhead_json):
    #     """
    #     - CUDA API interception:
    #       Subtract from:
    #       [CPU, q_forward, TensorFlow C++]
    #
    #     - CUPTI overhead:
    #       Subtract from:
    #       [CPU, q_forward, CUDA API]
    #
    #     - Python -> C-library interception:
    #       Subtract from:
    #       [CPU, q_forward, Python]
    #
    #     - Python annotations:
    #       Subtract from:
    #       [CPU, q_forward, Python]
    #     """
    #
    #     # 'machine', 'process', 'phase', 'resource_overlap', 'operation'
    #
    #     # e.g. subtracting Python annotation time.
    #     # The approach will be similar for other overhead types.
    #
    #     for machine_name, process_name, phase_name, operation_name in self._each_operation():
    #
    #         # Python annotations:
    #         total_pyprof_annotation = overhead_event_count_json['pyprof_annotation'][machine_name][process_name][phase_name][operation_name]
    #         per_pyprof_annotation_sec = pyprof_overhead_json['mean_pyprof_annotation_per_call_us']/constants.USEC_IN_SEC
    #         pyprof_annotation_sec = per_pyprof_annotation_sec * total_pyprof_annotation
    #         self.subtract_from_resource(
    #             resource='CPU',
    #             selector=dict(
    #                 machine=machine_name,
    #                 process=process_name,
    #                 phase=phase_name,
    #                 operation=operation_name,
    #                 category=constants.CATEGORY_PYTHON,
    #             ),
    #             subtract_sec=pyprof_annotation_sec)
    #
    #         # Python -> C-library interception:
    #         total_pyprof_interception = overhead_event_count_json['pyprof_interception'][machine_name][process_name][phase_name][operation_name]
    #         per_pyprof_interception_sec = pyprof_overhead_json['mean_pyprof_interception_overhead_per_call_us']/constants.USEC_IN_SEC
    #         pyprof_interception_sec = per_pyprof_interception_sec * total_pyprof_interception
    #         self.subtract_from_resource(
    #             resource='CPU',
    #             selector=dict(
    #                 machine=machine_name,
    #                 process=process_name,
    #                 phase=phase_name,
    #                 operation=operation_name,
    #                 category=constants.CATEGORY_PYTHON,
    #             ),
    #             subtract_sec=pyprof_interception_sec)
    #
    #
    #         missing_cupti_overhead_cuda_api_calls = dict()
    #         # CUPTI overhead:
    #         for cuda_api_name, num_api_calls in overhead_event_count_json['cuda_api_call'][machine_name][process_name][phase_name][operation_name].items():
    #             if cuda_api_name not in cupti_overhead_json:
    #                 missing_cupti_overhead_cuda_api_calls[cuda_api_name] = missing_cupti_overhead_cuda_api_calls.get(cuda_api_name, 0) + 1
    #             else:
    #                 per_cuda_api_sec = cupti_overhead_json[cuda_api_name]['mean_cupti_overhead_per_call_us']/constants.USEC_IN_SEC
    #                 cupti_overhead_sec = per_cuda_api_sec * num_api_calls
    #                 self.subtract_from_resource(
    #                     resource='CPU',
    #                     selector=dict(
    #                         machine=machine_name,
    #                         process=process_name,
    #                         phase=phase_name,
    #                         operation=operation_name,
    #                         category=constants.CATEGORY_CUDA_API_CPU,
    #                     ),
    #                     subtract_sec=cupti_overhead_sec)
    #         if len(missing_cupti_overhead_cuda_api_calls) > 0:
    #             logger.warning("Saw CUDA API calls that we didn't have calibrated CUPTI overheads for overheads for {path}: {msg}".format(
    #                 path=self.db_path,
    #                 msg=pprint_msg(missing_cupti_overhead_cuda_api_calls),
    #             ))
    #
    #         # CUDA API interception:
    #         total_cuda_api_calls = np.sum([num_api_calls for cuda_api_name, num_api_calls in
    #                                        overhead_event_count_json['cuda_api_call'][machine_name][process_name][phase_name][operation_name].items()])
    #         per_LD_PRELOAD_sec = pyprof_overhead_json['mean_interception_overhead_per_call_us']/constants.USEC_IN_SEC
    #         LD_PRELOAD_sec = per_LD_PRELOAD_sec * total_cuda_api_calls
    #         self.subtract_from_resource(
    #             resource='CPU',
    #             selector=dict(
    #                 machine=machine_name,
    #                 process=process_name,
    #                 phase=phase_name,
    #                 operation=operation_name,
    #                 # ASSUMPTION: GPU calls are being made from inside the Tensorflow C++ API...
    #                 # this might not hold once we start measuring other libraries the use the GPU...
    #                 # NOTE: this overhead does NOT come from the CUDA API call itself; the overhead is
    #                 # "around" the CUDA API call.
    #                 category=constants.CATEGORY_TF_API,
    #             ),
    #             subtract_sec=LD_PRELOAD_sec)
    #
    # def subtract_from_resource(self, resource, selector, subtract_sec):
    #     """
    #     To handle the fact that we cannot precisely attribute profiling-overhead CPU-time to [CPU], or [CPU, GPU],
    #     we decide to perform this heuristic:
    #     - Subtract from [CPU] or [CPU, GPU], in the order of whichever is largest FIRST
    #     - If we end up subtracting all of the first resource-group,
    #       then subtract what remains from the next resource-group.
    #
    #     NOTE: if we subtract CPU-time from [CPU, GPU] the new time that remains is GPU-only time...
    #
    #     :param resource:
    #     :param selector:
    #     :param subtract_sec:
    #     :return:
    #     """
    #     resource_selector = dict(selector)
    #     resource_selector['plot_type'] = 'ResourceOverlap'
    #     resource_selector = only_selector_fields(resource_selector)
    #     resource_vd = self.get_file(resource_selector)
    #     # e.g.
    #     # resource = 'CPU'
    #     # resource_types = [['CPU'], ['CPU', 'GPU']]
    #     keys = resource_vd.keys()
    #     resource_types = [resource_type for resource_type in keys if resource in resource_type]
    #     def sort_by_time(key):
    #         return -1 * resource_vd.get_size(key)
    #     # by = {by total time spent in resource in descending order}
    #     resource_types.sort(key=sort_by_time)
    #     subtract_left_sec = subtract_sec
    #     for resource_type in resource_types:
    #         if subtract_left_sec == 0:
    #             break
    #         # vd_leaf = vd_tree.lookup(machine, process, phase, operation, category)
    #         # vd_selector = copy.copy(selector)
    #         # vd_selector['resource_overlap'] = resource_type
    #         # Q: are we selecting for ResourceOverlap or ResourceSubplot?
    #         # vd_leaf = self.get_file(selector)
    #
    #         # to_subtract = min(
    #         #     subtract_left_sec,
    #         #     vd.time_sec(resource_type, process, phase, operation, category))
    #         # key_type = plot_index.KEY_TYPE[plot_type]
    #         # key = selector[key_type]
    #         to_subtract = min(
    #             subtract_left_sec,
    #             # vd_leaf.get_size(key),
    #             resource_vd.get_size(resource_type),
    #         )
    #
    #         # We need to "propagate up" the subtraction;
    #         # vd_tree.subtract handles this.
    #         # i.e. If we are subtracting from:
    #         #   [CPU, q_forward, Python]
    #         # Then, we need to subtract from:
    #         #   [CPU, q_forward, Python]
    #         #     CategoryOverlap.machine_{...}.process_{...}.phase_{...}.ops_{...}.resources_{...}.venn_js.json
    #         #   [CPU, q_forward]
    #         #     OperationOverlap.machine_{...}.process_{...}.phase_{...}.resources_{...}.venn_js.json
    #         #   [CPU]
    #         #     ResourceOverlap.machine_{...}.process_{...}.phase_{...}.venn_js.json
    #         #     SKIP: ResourceSubplot.machine_{...}.process_{...}.phase_{...}.venn_js.json
    #         # vd_tree.subtract(machine, process, phase, resource_type, operation, category, to_subtract)
    #         self.subtract(resource_selector, to_subtract)
    #         subtract_left_sec -= to_subtract
    #
    # def subtract(self, selector, subtract_sec):
    #     # selector = {
    #     #     'machine': machine
    #     #     'process': process
    #     #     'phase': phase
    #     #     'resource_type': resource_type,
    #     #     'operation': operation
    #     #     'category': category
    #     # }
    #     for plot_type in plot_index.OVERLAP_PLOT_TYPES:
    #         # e.g. ResourceOverlap: [machine, process, phase]
    #         # selector[just keep plot_type.attributes]
    #         plot_type_selector = dict((k, v) for k, v in selector.items()
    #                                   if k in plot_index.SEL_ORDER[plot_type])
    #         plot_type_selector['plot_type'] = plot_type
    #         vd = self.get_file(plot_type_selector)
    #
    #         # def vd.key_field():
    #         #  ResourceSubplot -> ListOf[resource_type]
    #         #  OperationOverlap -> operation
    #         #  CategoryOverlap -> category
    #         #  ResourceSubplot -> resource_type
    #         key_field = plot_index.KEY_TYPE[plot_type]
    #         key = selector[key_field]
    #         vd.subtract(key, subtract_sec, inplace=True)

def get_x_env(env, long_env=False):
    short_env = env
    if not long_env:
        short_env = re.sub(r'-v\d+$', '', short_env)
        short_env = re.sub(r'BulletEnv$', '', short_env)
        short_env = re.sub(r'Env$', '', short_env)
        short_env = re.sub(r'NoFrameskip', '', short_env)
        short_env = re.sub(r'Inverted', 'Inv', short_env)
        short_env = re.sub(r'Double', 'Dbl', short_env)
        short_env = re.sub(r'Pendulum', 'Pndlm', short_env)
        short_env = re.sub(r'Swingup', 'Swing', short_env)
    return short_env

def get_x_algo(algo):
    pretty_algo = algo.upper()
    return pretty_algo

def get_x_field(algo, env, x_type, human_readable=False):
    pretty_algo = get_x_algo(algo)
    short_env = get_x_env(env)
    if x_type == 'rl-comparison':
        # if human_readable:
        x_field = "({algo}, {env})".format(
            algo=pretty_algo,
            env=short_env,
        )
        # else:
        #     x_field = "{algo}\n{env}".format(
        #         algo=pretty_algo,
        #         env=short_env,
        #     )
    elif x_type == 'env-comparison':
        x_field = short_env
    elif x_type == 'algo-comparison':
        x_field = pretty_algo
    else:
        raise NotImplementedError
    return x_field

def main():
    parser = argparse.ArgumentParser(
        textwrap.dedent("""\
        Test plots
        """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--test-stacked-bar',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot multi-bar stacked bar-chart.
    """))


    parser.add_argument('--test-stacked-bar-sns',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot multi-bar stacked bar-chart.
    """))

    parser.add_argument('--test-stacked-bar-sns-old',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot multi-bar stacked bar-chart.
    """))

    parser.add_argument('--test-double-yaxis',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot double y-axis stacked bar-chart.
    """))

    args = parser.parse_args()

    if args.test_stacked_bar:
        attempt_stacked_bar()
    elif args.test_stacked_bar_sns:
        attempt_stacked_bar_sns()
    elif args.test_stacked_bar_sns_old:
        attempt_stacked_bar_sns_old()
    elif args.test_double_yaxis:
        attempt_double_yaxis()

def only_selector_fields(selector):
    assert 'plot_type' in selector
    new_selector = dict((k, v) for k, v in selector.items())
    return new_selector

def join_plus(xs, check_non_empty=False):
    """
    Concatenate strings with '+' to form legend label.

    Parameters
    ----------
    xs : list
      List of strings
    check_non_empty : bool
      If True, assert that len(xs) > 0

    Returns
    -------
    str
        Concatenation of xs (e.g., "CPU + GPU")

    Examples
    --------
    >>> join_plus(["CPU", "GPU"])
    "CPU + GPU"
    """
    if check_non_empty:
        assert len(xs) != 0
    return ' + '.join(sorted(xs))
    # PROBLEM: this messes up the DQN plot for some reason...
    # return " +\n".join(sorted(xs))

def split_plus(string):
    xs = re.split(r'\s+\+\s+', string)
    return xs

def _test_join_plus_with(string, xs):
    assert join_plus(xs) == string
    assert split_plus(string) == xs
    assert split_plus(join_plus(xs)) == xs
    assert join_plus(split_plus(string)) == string

def test_join_plus_01():
    string = 'CPU + GPU'
    xs = ['CPU', 'GPU']
    _test_join_plus_with(string, xs)

def test_join_plus_02():
    string = 'CPU'
    xs = ['CPU']
    _test_join_plus_with(string, xs)

class ColumnGrouping:
    def __init__(self, all_cols, numeric_cols, non_numeric_cols):
        self.all_cols = all_cols
        self.numeric_cols = numeric_cols
        self.non_numeric_cols = non_numeric_cols

    def __repr__(self):
        return "{klass}(all_cols={all_cols}, numeric_cols={numeric_cols}, non_numeric_cols={non_numeric_cols})".format(
            klass=self.__class__.__name__,
            all_cols=self.all_cols,
            numeric_cols=self.numeric_cols,
            non_numeric_cols=self.non_numeric_cols,
        )

def group_numeric_cols(df):
    all_cols = set(df.keys())
    numeric_cols = set([colname for colname in df.keys() if np.issubdtype(df[colname].dtype, np.number)])
    non_numeric_cols = all_cols.difference(numeric_cols)
    cols = ColumnGrouping(
        all_cols=list(all_cols),
        numeric_cols=list(numeric_cols),
        non_numeric_cols=list(non_numeric_cols),
    )
    return cols

def regex_match(x, regex_value_pairs, allow_no_match=True):
    for regex, value in regex_value_pairs:
        if re.search(regex, x):
            return value
    if allow_no_match:
        return x
    raise RuntimeError("regex_match failed to match \"{x}\":\n{msg}".format(x=x, msg=txt_indent(regex_value_pairs)))

def xfields_from_xtick_expression(df, xtick_expression, debug=False):
    x_fields = []
    orig_df = df
    df = copy.copy(df)
    for index, row in df.iterrows():
        # if debug:
        #     logger.debug("--xtick-expression:\n{trans}".format(trans=textwrap.indent(xtick_expression, prefix='  ')))
        #     logger.debug(txt_indent({
        #         'row': row,
        #     }, indent=1))
        x_field = None
        def get_errmsg():
            return "--xtick-expression must be a python expression that defines x_field for each 'row' of dataframe 'df'; available columns in row are:\n{msg}".format(
                msg=txt_indent({'columns': sorted(df.keys())}, indent=1))
        try:
            my_locals = locals()
            exec(xtick_expression, globals(), my_locals)
            x_field = my_locals['x_field']
        except Exception as e:
            if type(e) == SyntaxError:
                lines = xtick_expression.splitlines()
                ss = StringIO()
                # 3 => " > "
                num_spaces = int(np.ceil(np.log10(len(lines))))
                ss.write(f"Syntax error in --xtick-expression at line {e.lineno}, column {e.offset}: {e.msg}: \"{e.text}\"\n")
                for lineno, line in enumerate(lines, start=1):
                    lineno_str = "{lineno: <{fill}}".format(fill=num_spaces, lineno=lineno)
                    if lineno == e.lineno:
                        ss.write("{lineno} > {line}\n".format(fill=num_spaces, lineno=lineno_str, line=line))
                    else:
                        ss.write("{lineno}   {line}\n".format(fill=num_spaces, lineno=lineno_str, line=line))
                ss.write(get_errmsg())
                errmsg = ss.getvalue()
            else:
                errmsg = get_errmsg()
            raise RuntimeError("Saw exception in --xtick-expression: {klass}({e}).\n{errmsg}".format(
                klass=type(e).__name__,
                e=e,
                errmsg=errmsg))
        if x_field is None:
            raise RuntimeError("x_field wasn't set in --xtick-expression; {errmsg}".format(
                errmsg=get_errmsg()))
        x_fields.append(x_field)
    return x_fields

def get_capsize(ax):
    # num_bars = len(ax.patches)
    bar_width = ax.patches[0].get_width()

    # Sanity check.
    # NOTE: This can fail... I've seen 3 bars with widths:
    #   0.26666666666666666, 0.26666666666666666, 0.2666666666666667
    # assert all(patch.get_width() == bar_width for patch in ax.patches)
    # Make error bar width 25% of bar width.

    return bar_width/4

def add_repetition(df):
    df['repetition'] = df['rlscope_directory'].apply(get_repetition)

def get_repetition(rlscope_directory):
    m = re.search(r'repetition_(?P<repetition>\d+)', os.path.basename(rlscope_directory))
    if m:
        return int(m.group('repetition'))
    return None

class CategoryTransitionPlot:
    def __init__(self,
                 time_breakdown_directories,
                 rlscope_directories,
                 directory,
                 title=None,
                 x_title=None,
                 y_title=None,
                 width=None,
                 height=None,
                 rotation=None,
                 category=None,
                 hack_upper_right_legend_bbox_x=None,
                 xtick_expression=None,
                 include_gpu=False,
                 include_python=False,
                 include_simulator=False,
                 remap_df=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.time_breakdown_directories = time_breakdown_directories
        self.rlscope_directories = rlscope_directories
        self.directory = directory
        if len(self.time_breakdown_directories) != len(self.rlscope_directories):
            raise RuntimeError("The number of --time-breakdown-directories must match the number of --rlscope-directories")
        for time_rlscope_dir, rlscope_dir in zip(self.time_breakdown_directories, self.rlscope_directories):
            assert get_repetition(time_rlscope_dir) == get_repetition(rlscope_dir)

        category_value = None
        if category is not None and re.search(r'^CATEGORY_'):
            category_value = eval(category)
        else:
            category_value = category
        self.category = category
        self.category_value = category_value

        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self._parse_js = dict()
        self._parse_rlscope_config = dict()
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.width = width
        self.height = height
        self.rotation = rotation
        self.xtick_expression = xtick_expression
        self.hack_upper_right_legend_bbox_x = hack_upper_right_legend_bbox_x
        self.include_gpu = include_gpu
        self.include_python = include_python
        self.include_simulator = include_simulator
        self.remap_df = remap_df

    def category_key_set_as_category_key(self, category_key_set):
        category_key = CategoryKeyJS()
        def _union_all(attr):
            if len(category_key_set) == 0:
                return frozenset()
            return frozenset(set.union(*[set(getattr(k, attr)) for k in category_key_set]))
        category_key.procs = _union_all('procs')
        category_key.ops = _union_all('ops')
        assert len(category_key.ops) <= 1
        category_key.non_ops = _union_all('non_ops')
        return category_key

    def parse_js(self, json_path):
        if json_path in self._parse_js:
            return self._parse_js[json_path]
        js = load_json(json_path)
        category_trans_counts = from_js(js['category_trans_counts'])
        cmap = dict()
        for pair, value in category_trans_counts.items():
            assert len(pair) == 2
            from_category_key = self.category_key_set_as_category_key(pair[0])
            to_category_key = self.category_key_set_as_category_key(pair[1])
            cmap[(from_category_key, to_category_key)] = value
        self._parse_js[json_path] = cmap
        return cmap

    def parse_rlscope_config(self, rlscope_dir):
        # rlscope_dir = _d(json_path)
        if rlscope_dir in self._parse_rlscope_config:
            return self._parse_rlscope_config[rlscope_dir]
        rlscope_config_path = rlscope_dataframe.get_rlscope_config_path(rlscope_dir, allow_none=True)
        if rlscope_config_path is not None:
            rlscope_config = RLScopeConfig(rlscope_config_path=rlscope_config_path)
        else:
            rlscope_config = None
        self._parse_rlscope_config[rlscope_dir] = rlscope_config
        return rlscope_config

    def json_paths(self, rlscope_dir):
        for path in each_file_recursive(rlscope_dir):
            if is_overlap_result_js_file(path):
                yield path

    def each_json_path(self):
        for rlscope_dir, time_rlscope_dir in zip(self.rlscope_directories, self.time_breakdown_directories):
            for json_path in self.json_paths(time_rlscope_dir):
                yield rlscope_dir, json_path

    def all_read_df(self):
        dfs = []
        for rlscope_dir, json_path in self.each_json_path():
            df = self.read_df(rlscope_dir, json_path)
            dfs.append(df)
        all_df = pd.concat(dfs)
        return all_df

    def read_df(self, rlscope_dir, json_path):
        # js = load_json(self.cross_process_overlap)
        category_trans_counts = self.parse_js(json_path)
        if self.debug:
            logger.debug("category_trans_counts @ {json_path}:\n{msg}".format(
                json_path=json_path,
                msg=pprint_msg(category_trans_counts),
            ))
        data = {
            # 'from_category': [],
            # 'to_category': [],
            'rlscope_directory': [],
            'algo': [],
            'env': [],
            'category': [],
            'operation': [],
            'trans_count': [],
            # 'max_passes': [],
        }
        for key, value in category_trans_counts.items():
            assert len(key) == 2
            # CategoryKey
            from_category = key[0]
            # CategoryKey
            to_category = key[1]
            trans_count = value

            if len(from_category.ops) == 0 or len(to_category.ops) == 0:
                continue

            # CONCERN: This is going to fail.  New categories will happen when operations change.
            # Some categories won't have any operations.
            assert len(from_category.ops) == 1
            assert len(to_category.ops) == 1
            if from_category.ops != to_category.ops:
                # assert from_category.ops == to_category.ops
                continue
            operation = next(iter(from_category.ops))

            rlscope_config = self.parse_rlscope_config(rlscope_dir)
            # rlscope_dir = _d(json_path)
            algo = rlscope_config.algo()
            env = rlscope_config.env()
            # max_passes = rlscope_config.get_int('max_passes')
            max_passes = rlscope_config.get_var('max_passes', dflt=None)
            if max_passes is None:
                # Deprecated; keep to read older trace files.
                max_passes = rlscope_config.get_var('max_training_loop_iters', dflt=None)
            if max_passes is not None:
                max_passes = int(max_passes)

            # From: [no has category] -> [has category]
            # All category where:
            #   category is in to_category
            #   category is NOT in from_category
            # to_category.difference(from_category)
            categories = to_category.non_ops.difference(from_category.non_ops)
            if len(categories) > 0:
                logger.debug(f"categories = {categories}")
            else:
                logger.debug(textwrap.dedent("""\
                no categories:
                  from_key
                    procs:   {from_procs}
                    non_ops: {from_non_ops}
                    ops:     {from_ops}
                  to_key
                    procs:   {to_procs}
                    non_ops: {to_non_ops}
                    ops:     {to_ops}
                    """.format(
                    from_procs=from_category.procs,
                    from_non_ops=from_category.non_ops,
                    from_ops=from_category.ops,
                    to_procs=to_category.procs,
                    to_non_ops=to_category.non_ops,
                    to_ops=to_category.ops,
                )).rstrip())
            for category in categories:
                data['category'].append(category)

                # Useful transitions to count:
                # Python<->C transitions
                #   # [no Python] -> [has Python]
                #   [has Python] -> [no Python]
                # CUDA API calls:
                #   [no CUDA] -> [has CUDA]
                data['operation'].append(operation)
                # data['from_category'].append(from_category)
                # data['to_category'].append(to_category)
                data['trans_count'].append(trans_count)
                data['rlscope_directory'].append(rlscope_dir)
                data['algo'].append(algo)
                data['env'].append(env)
                if max_passes is not None:
                    if 'max_passes' not in data:
                        data['max_passes'] = []
                    data['max_passes'].append(max_passes)

        df = pd.DataFrame(data=data)
        # if self.debug:
        #     logger.debug("CategoryTransitionPlot.df:\n{msg}".format(msg=textwrap.indent(DataFrame.dataframe_string(df), prefix='  ')))

        add_repetition(df)
        add_df_xfield(df, xtick_expression=self.xtick_expression, debug=self.debug)
        df['short_category'] = df['category'].apply(short_category)
        df = apply_remap_df(self.remap_df, df, debug=self.debug)

        # Not informative
        if not self.include_gpu:
            df = df[df['category'] != constants.CATEGORY_GPU]
        # Redundant: CATEGORY_PYTHON = CATEGORY_SIMULATOR + CATEGORY_FRAMEWORK
        if not self.include_python:
            df = df[df['category'] != constants.CATEGORY_PYTHON]
        if not self.include_simulator:
            # Small but non-zero Simulator calls can happen during backpropagation.
            # This is negligible and can be ignored for graphs.
            df = df[~(
                (df['category'] == constants.CATEGORY_SIMULATOR_CPP) &
                (df['operation'] != 'Simulation'))]

        # cols = group_numeric_cols(df)
        # groupby_cols = sorted(set(cols.non_numeric_cols).difference({'repetition'}))
        groupby_cols = sorted(set(df.keys()).difference({
            'trans_count',
        }))
        df = df.groupby(groupby_cols).agg(pd.DataFrame.sum, skipna=False).reset_index()

        if self.debug:
            logger.debug("CategoryTransitionPlot.df:\n{msg}".format(msg=textwrap.indent(DataFrame.dataframe_string(df), prefix='  ')))

        return df

    def plot(self):
        self.plot_df = copy.copy(self.df)
        if 'max_passes' in self.plot_df:
            self.plot_df['trans_count_per_pass'] = self.plot_df['trans_count']/self.plot_df['max_passes']
            y_field = 'trans_count_per_pass'
            y_title = "Transitions per iteration"
        else:
            y_field = 'trans_count'
            y_title = "Total transitions"

        if self.category is not None:
            if self.category not in self.plot_df['category'].unique():
                raise RuntimeError("Cannot create CategoryTransitionPlot for --category=\"{cat}\" (\"{cat_value}\") since; available categories are: {cats}".format(
                    cat=self.category,
                    cat_value=self.category_value,
                    cats=set(self.plot_df['category'].unique()),
                ))
            category_df = self.plot_df[self.plot_df['category'] == self.category]
            self.category_plot(category_df, self.category,
                               y_field=y_field,
                               y_title=y_title)
        else:
            for category, category_df in self.plot_df.groupby(['category']):
                self.category_plot(category_df, category,
                                   y_field=y_field,
                                   y_title=y_title)
            self.combined_plot(self.plot_df,
                           y_field=y_field,
                           y_title=y_title)

    def run(self):
        self.df = self.all_read_df()
        self.plot()

    def plot_title(self):
        if self.title is not None:
            return self.title
        return None

    def _path_category(self, category):
        cat = category.lower()
        cat = re.sub(' ', '_', cat)
        cat = re.sub(' ', '_', cat)
        cat = re.sub(':', '-', cat)
        return cat

    def plot_path(self, category):
        return _j(self.directory, "{klass}.category_{category}.pdf".format(
            klass=self.__class__.__name__,
            category=self._path_category(category),
        ))

    def combined_plot_path(self):
        return _j(self.directory, "{klass}.combined.pdf".format(
            klass=self.__class__.__name__,
        ))

    def _get_ylabel(self, category, y_title=None):
        if self.y_title is not None:
            return self.y_title
        if y_title is None:
            y_title="Transitions per iteration"
        return "{y_title} to:\n{category}".format(
            y_title=y_title,
            category=short_category(category),
        )

    def category_plot(self, df, category, y_field, y_title=None, **kwargs):
        def ax_func(stacked_bar_plot):
            if self.rotation is not None:
                stacked_bar_plot.ax.set_xticklabels(
                    stacked_bar_plot.ax.get_xticklabels(),
                rotation=self.rotation)
        rls_paper_fontsizes()
        stacked_bar_plot = DetailedStackedBarPlot(
            data=df,
            path=self.plot_path(category),
            x_field='x_field',
            y_field=y_field,
            x_group='operation',

            # y_lim_scale_factor=self.y_lim_scale_factor,

            hues_together=True,
            hatch='short_category',
            hatch_styles=RLS_SHORT_CATEGORY_HATCH_STYLE,
            empty_hatch_value="",
            # hue='resource_overlap',
            hack_upper_right_legend_bbox_x=self.hack_upper_right_legend_bbox_x,

            xlabel=self.x_title,

            width=self.width,
            height=self.height,

            ylabel=self._get_ylabel(category, y_title=y_title),
            title=self.plot_title(),
            func=ax_func,
            debug=self.debug,
            **kwargs,
        )
        stacked_bar_plot.plot()

    def combined_plot(self, df, y_field, y_title=None, **kwargs):
        def ax_func(stacked_bar_plot):
            if self.rotation is not None:
                stacked_bar_plot.ax.set_xticklabels(
                    stacked_bar_plot.ax.get_xticklabels(),
                    rotation=self.rotation)
        rls_paper_fontsizes()
        if y_title is None:
            y_title = "Transitions per iteration"
        stacked_bar_plot = DetailedStackedBarPlot(
            data=df,
            path=self.combined_plot_path(),
            x_field='x_field',
            y_field=y_field,
            x_group='operation',

            # y_lim_scale_factor=self.y_lim_scale_factor,

            hues_together=True,
            hatch='short_category',
            hatch_styles=RLS_SHORT_CATEGORY_HATCH_STYLE,
            empty_hatch_value="",
            # hue='resource_overlap',
            hack_upper_right_legend_bbox_x=self.hack_upper_right_legend_bbox_x,

            xlabel=self.x_title,

            width=self.width,
            height=self.height,

            ylabel=y_title,
            title=self.plot_title(),
            func=ax_func,
            debug=self.debug,
            **kwargs,
        )
        stacked_bar_plot.plot()

def apply_remap_df(remap_df, orig_df, debug=False):
    if remap_df is None:
        return orig_df

    df = copy.copy(orig_df)
    new_df = copy.copy(df)

    # e.g.
    # new_df[('other',)] = df[('compute_advantage_estimates',)] +
    #                      df[('optimize_surrogate',)]
    if debug:
        logger.info("--remap-df:\n{trans}".format(trans=textwrap.indent(remap_df, prefix='  ')))
    exec(remap_df)

    # Make sure they didn't modify df; they SHOULD be modifying new_df
    # (i.e. adding regions to a "fresh" slate)
    # assert np.all(df == orig_df)
    # Assume NaN's in the same place => equality.
    assert df.equals(orig_df)

    if debug:
        buf = StringIO()
        DataFrame.print_df(orig_df, file=buf)
        logger.info("Old dataframe:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))

        buf = StringIO()
        DataFrame.print_df(new_df, file=buf)
        logger.info("New dataframe after --remap-df:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))

    return new_df

def add_df_xfield(df, xtick_expression=None, row_func=None, debug=False):
    """
    Add any additional fields to the data-frame.

    In particular 'x_field' is a string used for the "x-tick" labels of the plot.
    :return:
    """
    df['pretty_algo'] = df['algo'].apply(get_x_algo)
    df['short_env'] = df['env'].apply(get_x_env)
    def _algo_env(row):
        return "({algo}, {env})".format(
            algo=get_x_algo(row['pretty_algo']),
            env=get_x_env(row['short_env']),
        )
    df['algo_env'] = df.apply(_algo_env, axis=1)
    if xtick_expression is None:
        x_fields = []
        for index, row in df.iterrows():
            if row_func is not None:
                x_field = row_func(row)
            else:
                pretty_algo = get_x_algo(row['algo'])
                short_env = get_x_env(row['env'])
                x_field = "({algo}, {env})".format(
                    algo=pretty_algo,
                    env=short_env,
                )
            x_fields.append(x_field)
    else:
        x_fields = xfields_from_xtick_expression(df, xtick_expression, debug=debug)
    df['x_field'] = x_fields

def short_category(category):
    if category == constants.CATEGORY_SIMULATOR_CPP:
        return "Simulator"
    elif category == constants.CATEGORY_TF_API:
        return "Backend"
        # return "Framework"
        # return "TensorFlow"
        # return "TF"
    elif category == constants.CATEGORY_CUDA_API_CPU:
        return "CUDA"
    return category

ALL_HATCH_STYLES = [
    # 7
    # '//', r'\\', '||', 'xx', 'o', '..', '*',
    # '////', r"\\\\", '|||', 'xxx', 'oo', '..', '**',
    '////', r"\\\\", '|||', '...', '**', 'oo', 'xxx',

    # 6
    '/.', '\\.', '|.', 'x.', 'o.', '*.',

    # 5
    '/o', '\\o', '|o', 'xo', '*o',

    # '/', '\\', '|', 'x', 'o', '.', '*',
    # '/', '\\', '|', 'x', 'o', '.', '*',
    # '/', '\\', '|', 'x', 'o', '.', '*',
]

RLS_CATEGORY_HATCH_STYLE = {
    # '/', '\\', '|', 'x', 'o', '.', '*',
    # '////', r"\\\\", '|||', '...', '**', 'oo', 'xxx',
    CATEGORY_TF_API: '...',
    CATEGORY_SIMULATOR_CPP: '|||',
    CATEGORY_CUDA_API_CPU: '////',
    CATEGORY_PYTHON: r'\\\\',
    CATEGORY_GPU: 'oo',
}
RLS_SHORT_CATEGORY_HATCH_STYLE = dict((short_category(category), hatch) for category, hatch in RLS_CATEGORY_HATCH_STYLE.items())

def rls_paper_fontsizes():
    SMALL_SIZE = 8
    # Default font size for matplotlib (too small for paper).
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    FONT_SIZE = BIGGER_SIZE

    # plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title


class TexMetric:
    def __init__(self, tex_template, tex_label, set_once=True, tex_variable_prefix=None):
        self.tex_template = tex_template
        self.tex_label = tex_label
        # I tend to add metrics in the order in which they appear in the statement;
        # use that to create placeholders in-order.
        self.metrics = OrderedDict()
        # Supporting data.
        self.df = None
        self.set_once = set_once
        self.tex_variable_prefix = tex_variable_prefix

    def tex_generate_defn(self, template=False):
        """
        Generate latex variables that define values of the metrics for this statement

        :return:
        """
        defns = []
        for i, metric_name in enumerate(self.metrics.keys()):
            varname = self.tex_varname(metric_name)
            if template:
                # value = r"\textit{{{letter}}}".format(letter=letter)
                # Allow user to decide how to format (e.g., \textit{...}, $...$)
                value = _texvar(i)
            else:
                value = self.metrics[metric_name]
            defn = self.tex_defn(varname, value)
            defns.append(defn)
        return '\n'.join(defns)

    def add_json(self, js):
        for metric_name, value in self.metrics.values():
            varname = self.tex_varname(metric_name)
            value = self.metrics[metric_name]
            assert varname not in js
            js[varname] = value

    def tex_varname(self, metric_name):
        """
        Variable name format is:
        <tex_label>__<metric_name>

        find:surp-autograph-inflates-python

        Find__SurpAutographInflatesPython__

        :param metric_name:
        :return:
        """
        components = []
        if self.tex_variable_prefix is not None:
            components.append(self.tex_variable_prefix)
        components.extend(self._split(self.tex_label))
        components.extend(self._split(metric_name))
        varname = self._join(components)
        if re.search(r'[0-9]', metric_name):
            old_varname = varname
            varname = re.sub(r'[0-9]', '', varname)
            logger.warning(r"Latex variable names cannot have numbers; removing numbers from \{old} -> \{new}".format(
                old=old_varname,
                new=varname,
            ))
        return varname

    def _join(self, components):
        def _camel(x):
            if re.search(r'^[a-z]', x):
                return x.capitalize()
            # If it doesn't start with a lower-case letter, assume it's already camel-case friendly.
            return x
        trans_components = [_camel(x) for x in components]
        varname = ''.join(trans_components)
        return varname

    def _split(self, string):
        split_re = r'[-:]'
        return re.split(split_re, string)

    def path_friendly_name(self):
        components = []
        components.extend(self._split(self.tex_label))
        name = self._join(components)
        return name

    def tex_defn(self, varname, value):
        return textwrap.dedent(r"""
        \newcommand{{\{varname}}}{{{value}}}
        """.format(
            varname=varname,
            value=value,
        )).lstrip().rstrip()

    def __setitem__(self, key, value):
        if self.set_once:
            # Expect metric[key] to only be set once.
            assert key not in self.metrics
        self.metrics[key] = value

    def __getitem__(self, key):
        return self.metrics[key]

    def __contains__(self, key):
        return key in self.metrics


class TexMetrics:
    def __init__(self,
                 directory,
                 framework_choice_csv=None,
                 framework_choice_ddpg_csv=None,
                 framework_choice_trans_csv=None,
                 framework_choice_ddpg_trans_csv=None,
                 framework_choice_uncorrected_csv=None,
                 framework_choice_ddpg_uncorrected_csv=None,
                 algo_choice_csv=None,
                 file_suffix=None,
                 tex_variable_prefix=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):

        def _check_all_or_none(csv_files, opt_string):
            if any([f is None for f in csv_files]) and not all([f is None for f in csv_files]):
                raise RuntimeError(f"You must provide all {opt_string} or none.")

        def _has_any(csv_files):
            return any([f is not None for f in csv_files])


        framework_choice_trans_csvs = []
        # framework_choice_csvs = []
        framework_choice_no_trans_csvs = []
        self.framework_choice_csv = framework_choice_csv
        # framework_choice_csvs.append(self.framework_choice_csv)
        framework_choice_no_trans_csvs.append(self.framework_choice_csv)
        self.framework_choice_ddpg_csv = framework_choice_ddpg_csv
        # framework_choice_csvs.append(self.framework_choice_ddpg_csv)
        framework_choice_no_trans_csvs.append(self.framework_choice_ddpg_csv)

        self.framework_choice_trans_csv = framework_choice_trans_csv
        # framework_choice_csvs.append(self.framework_choice_trans_csv)
        framework_choice_trans_csvs.append(self.framework_choice_trans_csv)
        self.framework_choice_ddpg_trans_csv = framework_choice_ddpg_trans_csv
        framework_choice_trans_csvs.append(self.framework_choice_ddpg_trans_csv)

        _check_all_or_none(
            framework_choice_no_trans_csvs,
            '--framework-choice-*-csv')

        # _check_all_or_none(framework_choice_csvs, '--framework-choice-*-csv')
        if _has_any(framework_choice_trans_csvs):
            _check_all_or_none(
                framework_choice_no_trans_csvs + framework_choice_trans_csvs,
                '--framework-choice-*-csv and --framework-choice-*-trans-csv')

        framework_choice_uncorrected_csvs = []
        self.framework_choice_uncorrected_csv = framework_choice_uncorrected_csv
        framework_choice_uncorrected_csvs.append(self.framework_choice_uncorrected_csv)
        self.framework_choice_ddpg_uncorrected_csv = framework_choice_ddpg_uncorrected_csv
        framework_choice_uncorrected_csvs.append(self.framework_choice_ddpg_uncorrected_csv)

        if _has_any(framework_choice_uncorrected_csvs):
            _check_all_or_none(
                framework_choice_no_trans_csvs + framework_choice_uncorrected_csvs,
                '--framework-choice-*-csv and --framework-choice-*-uncorrected-csv')

        algo_choice_csvs = []
        self.algo_choice_csv = algo_choice_csv
        algo_choice_csvs.append(self.algo_choice_csv)
        _check_all_or_none(algo_choice_csvs, '--algo-choice-*-csv')

        self.file_suffix = file_suffix
        self.tex_variable_prefix = tex_variable_prefix

        self.directory = directory
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self._register_all_framework_choice_metrics()
        self._register_all_framework_choice_uncorrected_metrics()
        self._register_all_algo_choice_metrics()

    def read_df(self, csv_files):
        def _mk_category(df):
            for x_field in ['category', 'operation', 'resource_overlap', 'algo', 'env']:
                if x_field in df.keys():
                    df[x_field] = df[x_field].replace(np.nan, '').astype('category')
            return df

        def _read_csv(path):
            df = pd.read_csv(path)
            df = _mk_category(df)
            return df

        def _read_csvs(paths):
            dfs = [_read_csv(path) for path in paths]
            df = pd.concat(dfs)
            df = _mk_category(df)
            return df

        df = _read_csvs(csv_files)
        return df

    def read_framework_choice_df(self):
        if self.framework_choice_csv is None:
            return
        self.framework_choice_df = self.read_df([self.framework_choice_csv, self.framework_choice_ddpg_csv])
        if self.framework_choice_trans_csv is None:
            return
        self.framework_choice_df_trans = self.read_df([self.framework_choice_trans_csv, self.framework_choice_ddpg_trans_csv])

    def read_framework_choice_uncorrected_df(self):
        if self.framework_choice_uncorrected_csv is None:
            return
        self.framework_choice_uncorrected_df = self.read_df([self.framework_choice_uncorrected_csv, self.framework_choice_ddpg_uncorrected_csv])

    def read_algo_choice_df(self):
        if self.algo_choice_csv is None:
            return
        self.algo_choice_df = self.read_df([self.algo_choice_csv])

    def read_algo_choice_uncorrected_df(self):
        if self.algo_choice_uncorrected_csv is None:
            return
        self.algo_choice_uncorrected_df = self.read_df([self.algo_choice_uncorrected_csv])

    def compute_metrics(self, metrics, tex_path):
        # Compute metrics
        for metric in metrics:
            calc_func = self.get_calc_func(metric)
            calc_func(metric)

        def _dump_tex():
            # Dump metrics
            with open(tex_path, 'w') as f:
                def _write(txt, indent=0):
                    f.write(textwrap.indent(textwrap.dedent(txt).lstrip(), prefix='  '*indent))
                _write(r"""
                %%% AUTOGENERATED - DO NOT MODIFY! %%
                %
                % Usage:
                %   To enable raw numbers:
                %     \def\{flag_var}{{1}}
                %     \input{{{tex}}}
                %
                %   To enable placeholders:
                %     % Comment out or delete any definition of {flag_var}
                %     \input{{{tex}}}
                %
                
                \ifx\{flag_var}\undefined
                """.format(
                    flag_var=self._tex_flag_varname,
                    tex=_b(tex_path),
                ))
                for metric in metrics:
                    _write("% Autogenerated template metrics for: \\ref{{{tex_label}}}\n".format(
                        tex_label=metric.tex_label,
                    ), indent=1)
                    _write(metric.tex_generate_defn(template=True), indent=1)
                    f.write('\n')
                    # if metric.df is not None:
                    #     metric_base_path = self.metric_base_path(metric)
                    #     output_csv(metric.df, metric_base_path)
                _write(r"""
                \else
                """.format(
                    flag_var=self._tex_flag_varname,
                ))
                for metric in metrics:
                    _write("% Autogenerated metrics for: \\ref{{{tex_label}}}\n".format(
                        tex_label=metric.tex_label,
                    ), indent=1)
                    _write(metric.tex_generate_defn(), indent=1)
                    f.write('\n')
                    if metric.df is not None:
                        if type(metric.df) == pd.DataFrame:
                            metric_base_path = self.metric_base_path(tex_path, metric)
                            output_csv(metric.df, metric_base_path)
                        elif type(metric.df) == dict:
                            for df_name, df in metric.df.items():
                                metric_base_path = self.metric_base_path(tex_path, metric, suffix=df_name)
                                output_csv(df, metric_base_path)
                        else:
                            raise NotImplementedError(f"Not sure how to output type(metric.df) == {type(metric.df)}")
                _write(r"""
                \fi
                """.format(
                    flag_var=self._tex_flag_varname,
                ))

        # def _dump_json():
        #     js = dict()
        #     for metric in metrics:
        #         metric.add_json(js)
        #     do_dump_json(js, json_path)

        _dump_tex()
        logger.info("Output tex @ {path}".format(path=tex_path))

        # _dump_json()
        # logger.info("Output json @ {path}".format(path=json_path))

    @property
    def _tex_flag_varname(self):
        return "{klass}Enabled".format(
            klass=self.__class__.__name__,
        )

    def run(self):
        self.read_framework_choice_df()
        self.read_framework_choice_uncorrected_df()
        self.read_algo_choice_df()
        if self.framework_choice_csv is not None and self.framework_choice_trans_csv is not None:
            self.compute_metrics(
                metrics=self.FRAMEWORK_CHOICE_METRICS,
                tex_path=self.framework_choice_tex_path())
        if self.algo_choice_csv is not None:
            self.compute_metrics(
                metrics=self.ALGO_CHOICE_METRICS,
                tex_path=self.algo_choice_tex_path())
        if self.framework_choice_uncorrected_csv is not None:
            self.compute_metrics(
                metrics=self.FRAMEWORK_CHOICE_UNCORRECTED_METRICS,
                tex_path=self.framework_choice_uncorrected_tex_path())

    def metric_base_path(self, tex_path, metric, suffix=None):
        base = re.sub(r'\.tex$', '', tex_path)
        assert base != tex_path
        return _j(self.directory, '{base}.{name}{suffix}'.format(
            base=base,
            name=metric.path_friendly_name(),
            suffix=maybe_suffix(suffix),
        ))

    def framework_choice_tex_path(self):
        return self._framework_choice_path('tex')

    def framework_choice_json_path(self):
        return self._framework_choice_path('json')

    def _framework_choice_path(self, ext):
        path = _j(self.directory, "FrameworkChoiceMetrics{suffix}.{ext}".format(
            suffix=maybe_suffix(self.file_suffix),
            ext=ext,
        ))
        if re.search(r'\.\.', path):
            import pdb; pdb.set_trace()
        return path

    def algo_choice_tex_path(self):
        return self._algo_choice_path('tex')

    def algo_choice_json_path(self):
        return self._algo_choice_path('json')

    def _algo_choice_path(self, ext):
        path = _j(self.directory, "AlgoChoiceMetrics{suffix}.{ext}".format(
            suffix=maybe_suffix(self.file_suffix),
            ext=ext,
        ))
        return path

    def framework_choice_uncorrected_tex_path(self):
        return self._framework_choice_uncorrected_path('tex')

    def _framework_choice_uncorrected_path(self, ext):
        path = _j(self.directory, "FrameworkChoiceMetricsUncorrected{suffix}.{ext}".format(
            suffix=maybe_suffix(self.file_suffix),
            ext=ext,
        ))
        if re.search(r'\.\.', path):
            import pdb; pdb.set_trace()
        return path


    def get_calc_func(self, metric):
        metric_calc_func = re.sub(r'[-:]', '_', metric.tex_label)
        txt_calc_func = f"calc_{metric_calc_func}"
        if not hasattr(self, txt_calc_func):
            raise RuntimeError("Didn't find function {klass}.{calc_func} Not sure how to calculate metric for tex_label={tex_label}, tex_template=\n{tex_template}".format(
                klass=self.__class__.__name__,
                calc_func=txt_calc_func,
                tex_label=metric.tex_label,
                tex_template=textwrap.indent(metric.tex_template, prefix='  '),
            ))
        calc_func = getattr(self, txt_calc_func)
        return calc_func


    def mean_df(self, field, df, groupby_fields=None, debug=False):
        """
        Compute mean of <field> by aggregating across repetitions.
        :return:
        """
        config_fields = ['algo', 'env', 'x_field']
        if groupby_fields is not None:
            list_maybe_extend(config_fields, groupby_fields)
            # config_fields.extend(groupby_fields)
        value_fields = [field]
        # Config = collections.namedtuple('Config', config_fields)

        # PROBLEM: I don't know why, but this is generated NaN's in total_training_time...
        # In particular, for category = "" and resource_overlap="GPU".
        # Q: Make simple unit test to recreates weird behaviour?
        cols = list(config_fields)
        list_maybe_extend(cols, ['repetition'])
        list_maybe_extend(cols, value_fields)
        groupby_cols = list(config_fields)
        list_maybe_extend(groupby_cols, ['repetition'])

        if cols != groupby_cols:
            df_agg_sum = df[cols].groupby(groupby_cols).agg(['sum'])
            df_agg_sum = df_agg_sum.reset_index()
            # Why do I need to do this?
            df_agg_sum.columns = df_agg_sum.columns.get_level_values(0)
            df_agg_sum = df_agg_sum[~df_agg_sum[field].isnull()]
        else:
            df_agg_sum = df[cols]

        agg_cols = list(config_fields)
        list_maybe_extend(agg_cols, value_fields)

        if agg_cols != config_fields:
            df_agg_mean = df_agg_sum[agg_cols].groupby(config_fields).agg(['mean'])
            df_agg_mean = df_agg_mean.reset_index()
            df_agg_mean.columns = df_agg_mean.columns.get_level_values(0)
            df_agg_mean = df_agg_mean[~df_agg_mean[field].isnull()]
        else:
            df_agg_mean = df_agg_sum[agg_cols]

        return df_agg_mean

    def _config_op_inference(self, row):
        return row['operation'] == 'Inference'

    def _config_op_backprop(self, row):
        return row['operation'] == 'Backpropagation'

    def _config_op_neural_network(self, row):
        return row['operation'] in {'Inference', 'Backpropagation'}

    def _config_has_gpu(self, row):
        return bool(re.search(r'gpu', row['resource_overlap'].lower()))

    def _config_algo_ddpg(self, row):
        return re.search(r'ddpg', row['algo'].lower())

    def _config_algo_td3(self, row):
        return re.search(r'td3', row['algo'].lower())

    def _config_category_framework(self, row):
        if 'short_category' in row:
            category = row['short_category']
        else:
            category = row['category']
        # return bool(re.search(r'framework', category.lower()))
        return bool(re.search(r'backend', category.lower()))

    def calc_find_qual_eager_more_trans(self, metric):
        r"""
        \begin{rlscope-finding-qual}{find:qual-eager-more-trans}
        Eager execution is between $x\times$ and $y\times$ as bad as Graph/Autograph execution, and slowdown is highly correlated with how well a framework implementation is optimized to minimize Framework transitions.
        \end{rlscope-finding-qual}

        total_eager_time[algo, env, repetition:r] = Compute total training time for eager execution for repetition r
        df['eager_slowdown', algo, env, repetition=r] = df['total_training_time', algo, env, repetition=r] / total_eager_time[algo, env, r]
        report df['eager_slowdown'].mean()
        report df['eager_slowdown'].std()
        """
        df = self.framework_choice_df
        df_trans = self.framework_choice_df_trans

        df_mean = self.mean_df('total_training_time', df=df)

        df_mean_eager_rows = df_mean[df_mean.apply(self._config_is_eager, axis=1)]
        df_mean_non_eager_rows = df_mean[~df_mean.apply(self._config_is_eager, axis=1)]
        join_on = ['algo', 'env']
        df = df_mean_non_eager_rows.set_index(join_on).join(df_mean_eager_rows.set_index(join_on), rsuffix='_eager')
        df['eager_slowdown'] = df['total_training_time_eager'] / df['total_training_time']

        # This will give us average total training time for each configuration (across repetitions)
        # df_avg = df_sum.reset_index()[config_fields + value_fields].groupby(config_fields).agg(['min', 'max', 'mean', 'std'])

        metric.df = dict()
        metric['MinEagerSlowdown'] = df['eager_slowdown'].min()
        metric['MaxEagerSlowdown'] = df['eager_slowdown'].max()
        metric.df['training_time'] = df

        def _add_pytorch_to_tf_ratio_metrics(df, op_name, is_op, category_name, is_category, ratio_field, ratio_name):
            df_op = df[df.apply(is_op, axis=1)]
            df_op_category = df_op[df_op.apply(is_category, axis=1)]
            def get_alias(prefix):
                return f"{prefix}Ratio{op_name}{category_name}{ratio_name}"
            self._add_metric_stats(metric, df_op_category, get_alias, ratio_field)

        groupby_fields = ['operation', 'short_category']
        df_trans_mean = self.mean_df('trans_count_per_pass', df=df_trans, groupby_fields=groupby_fields)
        ratio_name = 'TransTFToPyTorch'
        ratio_field = 'ratio_trans_tf_to_pytorch'
        # GOAL: want to know ratio of trans_count_per_pass between identical execution models (PyTorch eager vs TensorFlow eager)
        df_trans_eager_rows = df_trans_mean[df_trans_mean.apply(self._config_is_eager, axis=1)]
        df_trans_tf_eager_rows = df_trans_eager_rows[df_trans_eager_rows.apply(self._config_is_tensorflow, axis=1)]
        df_trans_pytorch_eager_rows = df_trans_eager_rows[df_trans_eager_rows.apply(self._config_is_pytorch, axis=1)]
        join_on = ['algo', 'env'] + groupby_fields
        df_trans = df_trans_tf_eager_rows.set_index(join_on).join(df_trans_pytorch_eager_rows.set_index(join_on), how='inner', lsuffix='_tf', rsuffix='_pytorch').reset_index()
        df_trans[ratio_field] = df_trans['trans_count_per_pass_tf'] / df_trans['trans_count_per_pass_pytorch']
        # df_trans['ratio_trans_pytorch_to_tf'] = df_trans['trans_count_per_pass_pytorch'] / df_trans['trans_count_per_pass_tf']
        _add_pytorch_to_tf_ratio_metrics(df_trans, 'Inference', self._config_op_inference, 'Framework', self._config_category_framework, ratio_field, ratio_name)
        _add_pytorch_to_tf_ratio_metrics(df_trans, 'Backpropagation', self._config_op_backprop, 'Framework', self._config_category_framework, ratio_field, ratio_name)
        metric.df['trans_count'] = df_trans

        groupby_fields = ['operation', 'category']
        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df, groupby_fields=groupby_fields)
        ratio_name = 'TFToPyTorch'
        ratio_field = 'ratio_tf_to_pytorch'
        # GOAL: want to know ratio of total_training_time between identical execution models (PyTorch eager vs TensorFlow eager)
        df_eager_rows = df_mean[df_mean.apply(self._config_is_eager, axis=1)]
        df_tf_eager_rows = df_eager_rows[df_eager_rows.apply(self._config_is_tensorflow, axis=1)]
        df_pytorch_eager_rows = df_eager_rows[df_eager_rows.apply(self._config_is_pytorch, axis=1)]
        join_on = ['algo', 'env'] + groupby_fields
        df = df_tf_eager_rows.set_index(join_on).join(df_pytorch_eager_rows.set_index(join_on), how='inner', lsuffix='_tf', rsuffix='_pytorch').reset_index()
        df[ratio_field] = df['total_training_time_tf'] / df['total_training_time_pytorch']
        # df['ratio_pytorch_to_tf'] = df['total_training_time_pytorch'] / df['total_training_time_tf']
        _add_pytorch_to_tf_ratio_metrics(df, 'Inference', self._config_op_inference, 'Framework', self._config_category_framework, ratio_field, ratio_name)
        _add_pytorch_to_tf_ratio_metrics(df, 'Backpropagation', self._config_op_backprop, 'Framework', self._config_category_framework, ratio_field, ratio_name)
        metric.df['operation.category.training_time'] = df

        # Supporting csv file.
        # Tells us which two eager/non-eager configurations are being compared to obtain the min/max slowdowns.
        # (e.g., max slowdown comes from TensorFlow Eager, min slowdown comes from PyTorch eager).
        # metric.df = df

    def calc_find_qual_autograph_reduces_python(self, metric):
        r"""
        \begin{rlscope-finding}{find:qual-autograph-reduces-python}
        By removing Framework transitions, \autograph substantially reduces Python time from $x\%$ in Graph
        to at most \FCAsPercent{\FindQualAutographReducesPythonMaxAutographPythonPercentOfOp}
        for Inference/Backpropagation.
        \end{rlscope-finding}

        where x = \FCAsPercent{\FindQualAutographReducesPythonMaxGraphPythonPercentOfOp}


        for operation in df.operations.unique():
            for each config(algo, env):
                df['python_percent_of_op'] =
                  df[algo, env, operation, category=CATEGORY_PYTHON]['total_training_time']/
                  df[algo, env, operation]['total_training_time'].sum()
        """
        df = self.framework_choice_df
        df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['operation', 'category'])

        metric.df = dict()

        def category_percent_op(group_df):
            # Q: Need to make copy?
            group_df['category_percent_op'] = group_df['total_training_time']/group_df['total_training_time'].sum()
            return group_df
        percent_df = df_mean.groupby(['algo', 'env', 'x_field', 'operation']).apply(category_percent_op)
        python_percent_df = percent_df[percent_df['category'] == constants.CATEGORY_PYTHON]
        def is_op(operation):
            return operation in {'Inference', 'Backpropagation'}
        op_python_percent_df = python_percent_df[python_percent_df['operation'].apply(is_op)]
        # def is_autograph(x_field):
        #     return bool(re.search(r'autograph', x_field.lower()))
        autograph_op_python_percent_df = op_python_percent_df[op_python_percent_df['x_field'].apply(self._xfield_is_autograph)]
        # def is_graph(x_field):
        #     self._config_is_graph()
        #     return bool(re.search(r'graph', x_field.lower()))
        graph_op_python_percent_df = op_python_percent_df[op_python_percent_df['x_field'].apply(self._xfield_is_graph)]

        Config = collections.namedtuple('Config', ['Name', 'df'])
        configs = [
            Config(Name='Autograph', df=autograph_op_python_percent_df),
            Config(Name='Graph', df=graph_op_python_percent_df),
        ]
        for config in configs:
            metric[f"Min{config.Name}PythonPercentOfOp"] = config.df['category_percent_op'].min()
            metric[f"Max{config.Name}PythonPercentOfOp"] = config.df['category_percent_op'].max()

        metric.df['operation.category.python_percent_of_op'] = op_python_percent_df

        # WANT: per-operation (Backprop, Inference) per-algorithm (DDPG, TD3) breakdown of Python reduction.
        # RatioPython{Operation}{Algo}
        # Just compute the Mean/Geomean?

        groupby_fields = ['operation', 'category']
        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df, groupby_fields=groupby_fields)
        ratio_field = 'ratio_graph_to_autograph'
        df_autograph_rows = df_mean[df_mean.apply(self._config_is_autograph, axis=1)]
        df_graph_rows = df_mean[df_mean.apply(self._config_is_graph, axis=1)]
        join_on = ['algo', 'env'] + groupby_fields
        df = df_autograph_rows.set_index(join_on).join(df_graph_rows.set_index(join_on), how='inner', lsuffix='_autograph', rsuffix='_graph').reset_index()
        df[ratio_field] = df['total_training_time_graph'] / df['total_training_time_autograph']
        metric.df['operation.category.ratio_python'] = df
        # NOTE: groupby order determines order of tex statements.
        for (algo, operation, category), df_group in df.groupby(['algo', 'operation', 'category']):
            if not self._is_gpu_operation(operation) or \
                not self._is_python_category(category):
                continue
            def get_alias(prefix):
                return "{Prefix}Ratio{Category}{Operation}{Algo}".format(
                    Prefix=prefix,
                    Category=category,
                    Operation=operation.capitalize(),
                    Algo=algo.upper(),
                )
            self._add_metric_stats(metric, df_group, get_alias, ratio_field)

    def _is_gpu_operation(self, operation):
        return operation in {'Inference', 'Backpropagation'}

    def _is_python_category(self, category):
        return re.search(r'python', category.lower())

    def _is_cuda_category(self, category):
        return re.search(r'cuda', category.lower())

    def _xfield_is_eager(self, x_field):
        return bool(re.search(r'eager', x_field.lower()))

    def _xfield_is_graph(self, x_field):
        return not self._xfield_is_autograph(x_field) and bool(re.search(r'\s+graph', x_field.lower()))

    def _xfield_is_autograph(self, x_field):
        return bool(re.search(r'autograph', x_field.lower()))

    def _xfield_is_pytorch(self, x_field):
        return bool(re.search(r'pytorch', x_field.lower()))

    def _xfield_is_tensorflow(self, x_field):
        return bool(re.search(r'tensorflow', x_field.lower()))

    def _config_is_eager(self, row):
        return self._xfield_is_eager(row['x_field'])

    def _config_is_graph(self, row):
        return self._xfield_is_graph(row['x_field'])

    def _config_is_autograph(self, row):
        return self._xfield_is_autograph(row['x_field'])

    def _config_is_on_policy(self, row):
        return self._config_policy_type(row) == 'On-policy'

    def _config_is_off_policy(self, row):
        return self._config_policy_type(row) == 'Off-policy'

    def _config_policy_type(self, row):
        algo = row['algo'].lower()
        if re.search(r'a2c|ppo', algo):
            return 'On-policy'
        elif re.search(r'dqn|ddpg|sac|td3', algo):
            return 'Off-policy'
        raise NotImplementedError(f"Not sure whether algo={row['algo']} is on/off-policy for rlscope_dir={row['rlscope_directory']}")

    def _config_is_pytorch(self, row):
        return self._xfield_is_pytorch(row['x_field'])

    def _config_is_tensorflow(self, row):
        return self._xfield_is_tensorflow(row['x_field'])

    def calc_find_qual_pytorch_eager_better(self, metric):
        r"""
        \begin{rlscope-finding-qual}{find:qual-pytorch-eager-better}
        PyTorch eager is $x\times$ faster than TensorFlow eager since it minimizes Framework transitions more effectively in Inference.
        \end{rlscope-finding-qual}

        We want to compute total training time comparison between PyTorch eager and TensorFlow eager.
        df['total_training_time']
        """
        df = self.framework_choice_df
        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df)

        df_mean_eager_rows = df_mean[df_mean.apply(self._config_is_eager, axis=1)]
        df_mean_eager_tf_rows = df_mean_eager_rows[df_mean_eager_rows.apply(self._config_is_tensorflow, axis=1)]
        df_mean_eager_pytorch_rows = df_mean_eager_rows[df_mean_eager_rows.apply(self._config_is_pytorch, axis=1)]
        join_on = ['algo', 'env']
        df = df_mean_eager_tf_rows.set_index(join_on).join(df_mean_eager_pytorch_rows.set_index(join_on), how='inner', rsuffix='_pytorch')
        df['pytorch_eager_speedup'] = df['total_training_time'] / df['total_training_time_pytorch']

        metric['MinPyTorchEagerSpeedup'] = df['pytorch_eager_speedup'].min()
        metric['MaxPyTorchEagerSpeedup'] = df['pytorch_eager_speedup'].max()
        metric.df = df

    def calc_find_qual_autograph_graph_similar(self, metric):
        r"""
        \begin{rlscope-finding-qual}{find:qual-autograph-graph-similar}
        \autograph and \graph execution are on-par with one another, performing within
        \FCAsPercent{\FindQualAutographGraphSimilarMaxSpeedup} of one another on a given
        RL algorithm; \autograph does \textit{not} always outperform \graph.
        \end{rlscope-finding-qual}

            Q: How to get the "within" percent?
            percent_diff =
                 abs(A - B)
                 ----------
                 max(A, B)

            ===

            if A < B:
              (B - A)
              -------
                 B
            else:
              (A - B)
              -------
                 A

        Moreover, even \autograph does not always outperform the older \graph API.
        In particular, for TD3 (Figure~\ref{fig:framework_choice}) \autograph is \asPercent{0.088176968037688} faster than \graph;
        conversely, for DDPG (Figure~\ref{fig:framework_choice_ddpg}) \graph is \asPercent{0.164893055987829} faster than \autograph.

        """
        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df)

        df_mean_autograph_rows = df_mean[df_mean.apply(self._config_is_autograph, axis=1)]
        df_mean_graph_rows = df_mean[df_mean.apply(self._config_is_graph, axis=1)]
        join_on = ['algo', 'env']
        df = df_mean_autograph_rows.set_index(join_on).join(
            df_mean_graph_rows.set_index(join_on),
            how='inner',
            lsuffix='_autograph',
            rsuffix='_graph')
        autograph = 'total_training_time_autograph'
        graph = 'total_training_time_graph'
        # def abs_percent_diff(row):
        #     return abs(row[autograph] - row[graph]) / max(row[autograph], row[graph])
        def autograph_percent_speedup(row):
            return (row[graph] - row[autograph]) / row[autograph]
        def graph_percent_speedup(row):
            return (row[autograph] - row[graph]) / row[graph]
        # df['abs_percent_diff'] = df.apply(abs_percent_diff, axis=1)
        df['autograph_percent_speedup'] = df.apply(autograph_percent_speedup, axis=1)
        df['graph_percent_speedup'] = df.apply(graph_percent_speedup, axis=1)

        # metric['MinSpeedup'] = df['abs_percent_diff'].min()
        # metric['MaxSpeedup'] = df['abs_percent_diff'].max()

        metric['MaxAutographSpeedup'] = df['autograph_percent_speedup'].max()
        metric['MaxGraphSpeedup'] = df['graph_percent_speedup'].max()
        metric['MaxSpeedup'] = max(metric['MaxAutographSpeedup'], metric['MaxGraphSpeedup'])

        metric.df = df


    def calc_find_surp_exec_model_comparison(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-exec-model-comparison}
        Eager execution vs Graph/Autograph: the best Graph/Autograph implementation is $y\times$ faster than PyTorch eager.
            OR
        Eager execution vs Graph/Autograph: PyTorch eager is $y\times$ slower than the best Graph/Autograph implementation.
        \end{rlscope-finding-surp}

        df[pytorch, algo, env] / max(df[autograph, algo, env], df[graph, algo, env])
        """
        df = self.framework_choice_df
        df_mean = self.mean_df('total_training_time', self.framework_choice_df)

        # JOIN: "PyTorch eager" rows with "autograph or graph" rows.

        def _config_is_tf_graph_or_autograph(row):
            return self._config_is_tensorflow(row) and ( self._config_is_autograph(row) or self._config_is_graph(row) )
        df_mean_eager_tf_rows = df_mean[df_mean.apply(_config_is_tf_graph_or_autograph, axis=1)]
        df_mean_eager_pytorch_rows = df_mean[df_mean.apply(self._config_is_pytorch, axis=1)]
        join_on = ['algo', 'env']
        df = df_mean_eager_tf_rows.set_index(join_on).join(df_mean_eager_pytorch_rows.set_index(join_on), how='inner', rsuffix='_pytorch')
        # df['pytorch_eager_speedup'] = df['total_training_time'] / df['total_training_time_pytorch']
        df['tf_speedup'] = df['total_training_time_pytorch'] / df['total_training_time']

        metric['MinTFSpeedup'] = df['tf_speedup'].min()
        metric['MaxTFSpeedup'] = df['tf_speedup'].max()

        metric.df = df

    def calc_find_surp_total_gpu_time(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-total-gpu-time}
        Total GPU time is similar across all RL frameworks regardless of DL back-end, and consistently low across all RL frameworks, making up at most $x\%$ of total training time.
        \end{rlscope-finding-surp}

        df['total_training_time', category=CATEGORY_GPU] / df['total_training_time']
        """

        df = copy.copy(self.framework_choice_df)
        df['has_gpu'] = df.apply(self._config_has_gpu, axis=1)
        # 'category',
        df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['has_gpu'], debug=True)

        # JOIN: "PyTorch eager" rows with "autograph or graph" rows.

        def category_percent_op(group_df):
            # Q: Need to make copy?
            group_df['category_percent_op'] = group_df['total_training_time']/group_df['total_training_time'].sum()
            return group_df
        group_by = ['algo', 'env', 'x_field']
        percent_df = df_mean.groupby(group_by).apply(category_percent_op)
        gpu_percent_df = percent_df[percent_df['has_gpu']]

        metric['MinGPUPercent'] = gpu_percent_df['category_percent_op'].min()
        metric['MaxGPUPercent'] = gpu_percent_df['category_percent_op'].max()
        metric.df = percent_df

    def _total_gpu_time_df(self):
        df = copy.copy(self.framework_choice_df)
        df['has_gpu'] = df.apply(self._config_has_gpu, axis=1)
        # 'category',
        df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['has_gpu'])

        # JOIN: "PyTorch eager" rows with "autograph or graph" rows.

        def category_percent_op(group_df):
            # Q: Need to make copy?
            group_df['category_percent_op'] = group_df['total_training_time']/group_df['total_training_time'].sum()
            return group_df
        group_by = ['algo', 'env', 'x_field']
        percent_df = df_mean.groupby(group_by).apply(category_percent_op)
        return percent_df
        # gpu_percent_df = percent_df[percent_df['has_gpu']]


    def calc_find_surp_cuda_api_dominates(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-cuda-api-dominates}
        In all RL frameworks, CPU-side CUDA API time dominates total GPU kernel execution time, taking up between $x\times$ and $y\times$ as much time as GPU kernel execution.
        \end{rlscope-finding-surp}

        df['total_cuda_api_time'] / df['total_gpu_time']
        """

        df = self.framework_choice_df
        total_gpu_df = self._total_gpu_time_df()
        gpu_percent_df = total_gpu_df[total_gpu_df['has_gpu']]

        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df, groupby_fields=['category'])
        def category_percent_op(group_df):
            # Q: Need to make copy?
            group_df['category_percent_op'] = group_df['total_training_time']/group_df['total_training_time'].sum()
            return group_df
        group_by = ['algo', 'env', 'x_field']
        percent_df = df_mean.groupby(group_by).apply(category_percent_op)
        def config_is_cuda_api(row):
            return bool(re.search(r'cuda', row['category'].lower()))
        cuda_percent_df = percent_df[percent_df.apply(config_is_cuda_api, axis=1)]
        # gpu_percent_df = percent_df[percent_df['has_gpu']]

        join_on = ['algo', 'env']
        df = gpu_percent_df.set_index(join_on).join(cuda_percent_df.set_index(join_on), how='inner', lsuffix='_gpu', rsuffix='_cuda')
        df['ratio_cuda_to_gpu'] = df['total_training_time_cuda'] / df['total_training_time_gpu']

        from scipy.stats.mstats import gmean

        metric['MinRatioCUDAToGPU'] = df['ratio_cuda_to_gpu'].min()
        metric['MaxRatioCUDAToGPU'] = df['ratio_cuda_to_gpu'].max()
        metric['MeanRatioCUDAToGPU'] = df['ratio_cuda_to_gpu'].mean()
        metric['GeomeanRatioCUDAToGPU'] = gmean(df['ratio_cuda_to_gpu'])
        metric['StdRatioCUDAToGPU'] = df['ratio_cuda_to_gpu'].std()

        metric.df = df

    def calc_find_surp_autograph_no_gpu(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-autograph-no-gpu}
        For RL workloads, even though Autograph converts RL training code to in-graph TensorFlow operators, Autograph has no perceivable increase in total GPU training time.
        \end{rlscope-finding-surp}

        Increase in GPU time from autograph:
            df['total_gpu_time', config=autograph] / max_{config != autograph} df['total_gpu_time']
        """

        df = self.framework_choice_df
        total_gpu_df = self._total_gpu_time_df()
        total_gpu_df = total_gpu_df[total_gpu_df['has_gpu']]

        autograph_rows = total_gpu_df[total_gpu_df.apply(self._config_is_autograph, axis=1)]
        nonautograph_rows = total_gpu_df[total_gpu_df.apply(lambda row: not self._config_is_autograph(row), axis=1)]

        join_on = ['algo', 'env']
        df = autograph_rows.set_index(join_on).join(nonautograph_rows.set_index(join_on), how='inner', lsuffix='_autograph', rsuffix='_nonautograph')
        ratio_field = 'ratio_gpu_autograph_to_nonautograph'
        df[ratio_field] = df['total_training_time_autograph'] / df['total_training_time_nonautograph']

        from scipy.stats.mstats import gmean

        def _alias(prefix):
            return f"{prefix}RatioGPUAutographToNonautograph"
        metric[_alias('Min')] = df[ratio_field].min()
        metric[_alias('Max')] = df[ratio_field].max()
        metric[_alias('Mean')] = df[ratio_field].mean()
        metric[_alias('Geomean')] = gmean(df[ratio_field])
        metric[_alias('Std')] = df[ratio_field].std()

        metric.df = df

    def calc_find_surp_autograph_inflates_inference(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-autograph-inflates-inference}
        Inference time in Autograph is inflated by Framework time ($x\times$ compared to Graph), and this inflation is \textit{not} due to extra Framework transitions, and is instead a performance anomaly within the DL back-end itself.
        \end{rlscope-finding-surp}

        df[operation='Inference', category='Framework', config=Autograph] / df[operation='Inference', category='Framework', config=Graph]
        """

        df = self.framework_choice_df
        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df, groupby_fields=['operation', 'category'])
        autograph_rows = df_mean[df_mean.apply(self._config_is_autograph, axis=1)]
        graph_rows = df_mean[df_mean.apply(self._config_is_graph, axis=1)]

        join_on = ['algo', 'env', 'operation', 'category']
        df = autograph_rows.set_index(join_on).join(graph_rows.set_index(join_on), how='inner', lsuffix='_autograph', rsuffix='_graph')
        df = df.reset_index()
        ratio_field = 'ratio_autograph_to_graph'
        df[ratio_field] = df['total_training_time_autograph'] / df['total_training_time_graph']

        df_filter = df

        def is_op(operation):
            return operation == 'Inference'
        df_filter = df_filter[df_filter['operation'].apply(is_op)]

        # def is_category(category):
        #     return bool(re.search(r'framework', category.lower()))
        # df_filter = df_filter[df_filter['category'].apply(is_category)]
        df_filter = df_filter[df_filter.apply(self._config_category_framework, axis=1)]

        def get_alias(prefix):
            return f"{prefix}RatioInferenceFrameworkAutographToGraph"
        self._add_metric_stats(metric, df_filter, get_alias, ratio_field)

        # WANT:
        # FindSurpAutographInflatesInferenceMeanRatioInferenceFrameworkDDPGAutographToGraph
        #   Specific value of ratio_autograph_to_graph where algo == 'DDPG'
        # FindSurpAutographInflatesInferenceMeanRatioInferenceFrameworkTDAutographToGraph
        #   Specific value of ratio_autograph_to_graph where algo == 'TD3'
        for algo, algo_df in df_filter.groupby(['algo']):
            def get_algo_alias(prefix):
                return "{Prefix}RatioInferenceFramework{Algo}AutographToGraph".format(
                    Prefix=prefix,
                    Algo=algo.upper(),
                )
            # One row for this algo.
            assert len(algo_df) == 1
            self._add_metric_stats(metric, algo_df, get_algo_alias, ratio_field, metrics={'Mean'})

        metric.df = df

    def _autograph_vs_eager_operation_category_ratio_df(self):
        df_mean = self.mean_df('total_training_time', df=self.framework_choice_df, groupby_fields=['operation', 'category'])
        autograph_rows = df_mean[df_mean.apply(self._config_is_autograph, axis=1)]
        eager_rows = df_mean[df_mean.apply(self._config_is_eager, axis=1)]
        tf_eager_rows = eager_rows[eager_rows.apply(self._config_is_tensorflow, axis=1)]

        join_on = ['algo', 'env', 'operation', 'category']
        df = autograph_rows.set_index(join_on).join(tf_eager_rows.set_index(join_on), how='inner', lsuffix='_autograph', rsuffix='_eager')
        df = df.reset_index()
        ratio_field = 'ratio_autograph_to_eager'
        df[ratio_field] = df['total_training_time_autograph'] / df['total_training_time_eager']
        return df

    def calc_find_surp_autograph_inflates_python(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-autograph-inflates-python}
        Autograph can inflate Python time by as much as $2\times$ during Simulation; training times (not just model performance) are highly sensitive to small differences in hyperparameter choices.
        \end{rlscope-finding-surp}
        """

        df = self._autograph_vs_eager_operation_category_ratio_df()
        ratio_field = 'ratio_autograph_to_eager'

        df_filter = df

        def is_op(operation):
            return operation == 'Simulation'
        df_filter = df_filter[df_filter['operation'].apply(is_op)]

        def is_category(category):
            return bool(re.search(r'python', category.lower()))
        df_filter = df_filter[df_filter['category'].apply(is_category)]

        def get_alias(prefix):
            return f"{prefix}RatioSimulationPythonAutographToEager"
        self._add_metric_stats(metric, df_filter, get_alias, ratio_field, metrics={'Max'})

        for algo, df_algo in df_filter.groupby(['algo']):
            def algo_get_alias(prefix):
                return "{Prefix}RatioSimulationPythonAutographToEager{Algo}".format(
                    Prefix=prefix,
                    Algo=algo.upper(),
                )
            self._add_metric_stats(metric, df_algo, algo_get_alias, ratio_field, metrics={'Mean'})

        metric.df = df

    def calc_find_surp_ddpg_backprop_slow(self, metric):
        r"""
        \begin{rlscope-finding-surp}{find:surp-ddpg-backprop-slow}
        RLScope's detailed metrics identify subtle performance differences in the DDPG Graph API
        RL algorithm implementation rooted in inefficient abstractions in high-level code
        responsible for a $x\times$ inflation in total Graph API Backpropagation time compared
        to Autograph.
        \end{rlscope-finding-surp}

        % Description text metrics:
        DDPG Backpropagation in Autograph is $x\times$ faster than Graph (Figure~\ref{fig:framework_choice_ddpg}),
        whereas the TD3 Backpropagation in Autograph is only $y\times$ faster than Graph
        (Figure~\ref{fig:framework_choice}); these inefficiencies in Backpropagation for DDPG Graph
        are correlated with high CUDA API inflation ($z\times$) and high Python inflation ($a\times$)
        relative to DDPG Autograph.

        WANT:
        df['ratio_autograph_to_graph', operation] =
            df['total_training_time_autograph', operation] /
            df['total_training_time_graph', operation]

        # Same thing broken down by category for Python and CUDA.
        df['ratio_autograph_to_graph', operation, category] =
            df['total_training_time_autograph', operation, category] /
            df['total_training_time_graph', operation, category]
        """

        df = self.framework_choice_df
        ratio_name = 'RatioAutographToGraph'
        join_add_metrics_kwargs = dict(
            ratio_field='ratio_graph_to_autograph',
            lsuffix='_graph',
            rsuffix='_autograph',
            is_lsuffix=self._config_is_graph,
            is_rsuffix=self._config_is_autograph,
        )

        def add_operation_metrics():
            def get_group_alias(prefix, group_dict):
                return "{Prefix}{Ratio}{Operation}{Algo}".format(
                    Prefix=prefix,
                    Ratio=ratio_name,
                    Operation=group_dict['operation'].capitalize(),
                    Algo=group_dict['algo'].upper(),
                )
            def keep_group(group_dict):
               return self._is_gpu_operation(group_dict['operation'])
            self._join_add_metrics(
                df=df,
                metric=metric,
                metrics={'Mean'},
                get_group_alias=get_group_alias,
                groupby_fields=['algo', 'operation'],
                keep_group=keep_group,
                **join_add_metrics_kwargs,
            )
        add_operation_metrics()

        def add_operation_category_metrics():
            def get_group_alias(prefix, group_dict):
                return "{Prefix}{Ratio}{Category}{Operation}{Algo}".format(
                    Prefix=prefix,
                    Ratio=ratio_name,
                    Category=group_dict['category'],
                    Operation=group_dict['operation'].capitalize(),
                    Algo=group_dict['algo'].upper(),
                )
            def keep_group(group_dict):
                return ( self._is_python_category(group_dict['category']) or \
                         self._is_cuda_category(group_dict['category']) ) and \
                       self._is_gpu_operation(group_dict['operation'])
            self._join_add_metrics(
                df=df,
                metric=metric,
                metrics={'Mean'},
                get_group_alias=get_group_alias,
                groupby_fields=['algo', 'operation', 'category'],
                keep_group=keep_group,
                **join_add_metrics_kwargs,
            )
        add_operation_category_metrics()

    #
    # Algo choice metrics
    #

    def _as_Resource(self, resource_overlap_regions):
        """
        ('CPU', 'GPU') => CpuGpu
        """
        return ''.join([x.capitalize() for x in sorted(resource_overlap_regions)])

    def calc_find_algo_choice(self, metric):
        r"""
        add_policy_metrics

            \begin{rlscope-finding}{find:algo-choice}
            On-policy algorithms are substantially more simulation-bound than off-policy algorithms,
            spending [on average / at least / up to] [$x\times$ more time in Simulation] than off-policy algorithms.
            %spending at least \asPercent{0.706128808} in simulation, [compared to at most XX\% for off policy]
            \end{rlscope-finding}

                df['ratio_on_policy_to_off_policy', operation] =
                    df['total_training_time_on_policy', operation]['percent'] /
                    df['total_training_time_off_policy', operation]['percent']

                For: operation = Simulation

        add_operation_metrics

            A2C and PPO spend a majority of their execution in Simulation, with A2C spending \asPercent{0.667435179}
            and PPO spending \asPercent{0.707735848}.
            On the other hand, DDPG and SAC spend a majority of their execution in Backpropagation,
            \textbf{[Mickey: this should talk about Simulation time, not backprop!]}
            with DDPG spending \asPercent{0.706128808}. and SAC spending \asPercent{0.811856197}.

            # FROM: calc_find_qual_autograph_reduces_python

                for operation in df.operations.unique():
                    for each config(algo, env):
                        df['percent_of_op'] =
                          df[algo, env, operation]['total_training_time']/
                          df[algo, env, operation]['total_training_time'].sum()

                For: operation = Simulation, Backpropagation (maybe?)

        add_resource_metrics

            Of the surveyed RL algorithms, DDPG has the largest por-tion of time spent GPU-bound, with
            10.8% of total trainingtime spent executing GPU kernels; conversely, PPO spends
            the least time on the GPU with only2.8% total training timespent executing GPU kernels.

            for operation in df.operations.unique():
                for each config(algo, env):
                    df['percent_of_resource'] =
                      df[algo, env, resource_overlap]['total_training_time']/
                      df[algo, env, resource_overlap]['total_training_time'].sum()



        add_operation_resource_metrics

            However, if we just consider Backpropagation time by itself, at most11.9% percent ofa Backprogation
            operation is spent executing GPU kernels(for A2C). Similarly, at most14.1% of an Inference oper-ation
            is spent executing GPU kernels (for SAC).

            for operation in df.operations.unique():
                for each config(algo, env):
                    df['percent_of_resource'] =
                      df[algo, env, operation, resource_overlap]['total_training_time']/
                      df[algo, env, operation, resource_overlap]['total_training_time'].sum()

        add_category_metrics

            [Of the CPU time spent in training loop across all algorithms,
            up to XX% is spent in high-level code and simulation,while the rest is in CUDA and framework.]

            # Lets JUST look at pure CPU (it's what people expect me to talk about here...)
            # or 'CPU + GPU'
            df_cpu = df[df[resource_overlap]='CPU']
            for operation in df_cpu.operations.unique():
                for each config(algo, env):
                    df_cpu['percent_of_resource'] =
                      df_cpu[algo, env]['total_training_time']/
                      df_cpu[algo, env]['total_training_time'].sum()


        """

        def add_policy_metrics():
            df = self.algo_choice_df
            ratio_name = 'RatioPercentOnPolicyToOffPolicy'
            join_add_metrics_kwargs = dict(
                ratio_field='ratio_percent_on_policy_to_off_policy',
                lsuffix='_on_policy',
                rsuffix='_off_policy',
                is_lsuffix=self._config_is_on_policy,
                is_rsuffix=self._config_is_off_policy,
            )
            field = 'percent'
            def get_group_alias(prefix, group_dict):
                return "{Prefix}{Ratio}{Operation}".format(
                    Prefix=prefix,
                    Ratio=ratio_name,
                    Operation=group_dict['operation'].capitalize(),
                    # Algo=group_dict['algo'].upper(),
                )
            # Q: What's this for...?
            # def keep_group(group_dict):
            #     return self._is_gpu_operation(group_dict['operation'])
            self._join_add_metrics(
                df=df,
                field=field,
                metric=metric,
                metrics={'Min', 'Max', 'Mean'},
                get_group_alias=get_group_alias,
                # We want to join all pairs of (on-policy, off-policy) with the same environment.
                join_on=['env'],
                groupby_fields=['operation'],
                # keep_group=keep_group,
                debug=True,
                **join_add_metrics_kwargs,
            )
        add_policy_metrics()

        def add_operation_metrics():
            field = 'op_percent'
            df = self.algo_choice_df
            df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['operation'])

            def category_percent_op(group_df):
                # Q: Need to make copy?
                group_df[field] = group_df['total_training_time']/group_df['total_training_time'].sum()
                return group_df
            percent_df = df_mean.groupby(['algo', 'env', 'x_field']).apply(category_percent_op)

            # Only one simulator to look at.
            # DDPGBackpropagationOpPercentMean
            for (algo, operation), group_percent_df in percent_df.groupby(['algo', 'operation']):
                # Expect only only env.
                assert len(group_percent_df['env'].unique()) == 1
                # Expect only one (algo, env) row.
                assert len(group_percent_df) == 1
                metric_name = "{Algo}{Op}OpPercentMean".format(
                    Algo=algo.upper(),
                    Op=operation.capitalize(),
                )
                assert metric_name not in metric
                metric[metric_name] = group_percent_df[field].mean()
            metric.df[f"operation.{field}"] = percent_df
        add_operation_metrics()

        def has_gpu_as_Resource(has_gpu):
            if has_gpu:
                Resource = 'Gpu'
            else:
                Resource = 'Cpu'
            return Resource

        def add_resource_metrics():
            field = 'resource_percent'
            df = copy.copy(self.algo_choice_df)
            df['has_gpu'] = df.apply(self._config_has_gpu, axis=1)
            df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['has_gpu'])

            def category_percent_op(group_df):
                # Q: Need to make copy?
                group_df[field] = group_df['total_training_time']/group_df['total_training_time'].sum()
                return group_df
            percent_df = df_mean.groupby(['algo', 'env', 'x_field']).apply(category_percent_op)

            # Only one simulator to look at.
            # DDPGGpuResourcePercentMean
            for (algo, has_gpu), group_percent_df in percent_df.groupby(['algo', 'has_gpu']):
                # resource_overlap_regions = split_plus(resource_overlap)
                # Resource = self._as_Resource(resource_overlap_regions)
                # Expect only only env.
                assert len(group_percent_df['env'].unique()) == 1
                # Expect only one (algo, env) row.
                assert len(group_percent_df) == 1
                Resource = has_gpu_as_Resource(has_gpu)
                metric_name = "{Algo}{Resource}ResourcePercentMean".format(
                    Algo=algo.upper(),
                    Resource=Resource,
                )
                assert metric_name not in metric
                metric[metric_name] = group_percent_df[field].mean()
            metric.df[f"resource.{field}"] = percent_df
        add_resource_metrics()

        def add_operation_resource_metrics():
            field = 'resource_percent'
            df = copy.copy(self.algo_choice_df)
            df['has_gpu'] = df.apply(self._config_has_gpu, axis=1)
            df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['operation', 'has_gpu'])

            def category_percent_op(group_df):
                # Q: Need to make copy?
                group_df[field] = group_df['total_training_time']/group_df['total_training_time'].sum()
                return group_df
            percent_df = df_mean.groupby(['algo', 'env', 'x_field', 'operation']).apply(category_percent_op)

            # Ignore simulation.
            percent_df = percent_df[percent_df.apply(self._config_op_neural_network, axis=1)]

            # DDPGBackpropagationGpuResourcePercentMean
            for (algo, operation, has_gpu), group_percent_df in percent_df.groupby(['algo', 'operation', 'has_gpu']):
                # resource_overlap_regions = split_plus(resource_overlap)
                # Resource = self._as_Resource(resource_overlap_regions)
                # Expect only only env.
                assert len(group_percent_df['env'].unique()) == 1
                # Expect only one (algo, env) row.
                assert len(group_percent_df) == 1
                Resource = has_gpu_as_Resource(has_gpu)
                metric_name = "{Algo}{Op}{Resource}ResourcePercentMean".format(
                    Algo=algo.upper(),
                    Op=operation.capitalize(),
                    Resource=Resource,
                )
                assert metric_name not in metric
                metric[metric_name] = group_percent_df[field].mean()
            metric.df[f"operation.resource.{field}"] = percent_df
        add_operation_resource_metrics()

        def add_category_metrics():
            field = 'category_percent'
            df = copy.copy(self.algo_choice_df)
            df['has_gpu'] = df.apply(self._config_has_gpu, axis=1)
            # Only CPU time.
            df = df[~df['has_gpu']]
            df_mean = self.mean_df('total_training_time', df=df, groupby_fields=['category'])

            def category_percent_op(group_df):
                # Q: Need to make copy?
                group_df[field] = group_df['total_training_time']/group_df['total_training_time'].sum()
                return group_df
            percent_df = df_mean.groupby(['algo', 'env', 'x_field']).apply(category_percent_op)

            # DDPGBackpropagationGpuResourcePercentMean
            for (algo, category), group_percent_df in percent_df.groupby(['algo', 'category']):
                # resource_overlap_regions = split_plus(resource_overlap)
                # Resource = self._as_Resource(resource_overlap_regions)
                # Expect only only env.
                assert len(group_percent_df['env'].unique()) == 1
                # Expect only one (algo, env) row.
                assert len(group_percent_df) == 1
                metric_name = "{Algo}{Category}CategoryPercentMean".format(
                    Algo=algo.upper(),
                    Category=category,
                )
                assert metric_name not in metric
                metric[metric_name] = group_percent_df[field].mean()
            metric.df[f"category.{field}"] = percent_df
        add_category_metrics()

    def _join_add_metrics(self,
                          metric,
                          ratio_field,
                          lsuffix, rsuffix,
                          is_lsuffix, is_rsuffix,
                          get_group_alias,
                          df,
                          field='total_training_time',
                          metrics=None,
                          groupby_fields=None,
                          join_on=None,
                          keep_group=None,
                          debug=False):
        if groupby_fields is None:
            groupby_fields = []
        lfield = f"{field}{lsuffix}"
        rfield = f"{field}{rsuffix}"
        df_mean = self.mean_df(field, df=df, groupby_fields=groupby_fields)
        df_l = df_mean[df_mean.apply(is_lsuffix, axis=1)]
        df_r = df_mean[df_mean.apply(is_rsuffix, axis=1)]
        if join_on is None:
            join_on = ['algo', 'env']
        else:
            join_on = list(join_on)
        list_maybe_extend(join_on, groupby_fields)
        df_join = df_l.set_index(join_on).join(df_r.set_index(join_on), how='inner', lsuffix=lsuffix, rsuffix=rsuffix).reset_index()
        # if debug:
        #     import pdb; pdb.set_trace()
        df_join[ratio_field] = df_join[lfield] / df_join[rfield]
        metric_name = "{groupby}.{ratio_field}".format(
            groupby='.'.join(groupby_fields),
            ratio_field=ratio_field,
        )
        if metric.df is None:
            metric.df = dict()
        metric.df[metric_name] = df_join
        # NOTE: groupby order determines order of tex statements.
        # Q: Why is algo part of the group-by...?
        # groupby = ['algo'] + groupby_fields
        for group_tupl, df_group in df_join.groupby(groupby_fields):
            if len(groupby_fields) == 1:
                # Pandas API quirk: groupby keys are NOT always a tuple...
                # In particular, if only one groupby field, it is a scalar.
                group_tupl = (group_tupl,)
            group_dict = dict(zip(groupby_fields, group_tupl))
            if keep_group is not None and not keep_group(group_dict):
                continue
            # if not self._is_gpu_operation(operation) or \
            #     not self._is_python_category(category):
            #     continue
            def wrapper_get_alias(prefix):
                return get_group_alias(prefix, group_dict)
                # return "{Prefix}Ratio{Category}{Operation}{Algo}".format(
                #     Prefix=prefix,
                #     Category=category,
                #     Operation=operation.capitalize(),
                #     Algo=algo.upper(),
                # )
            self._add_metric_stats(metric, df_group, wrapper_get_alias, ratio_field, metrics=metrics)

    def _add_metric_stats(self, metric, df, get_alias, field, metrics=None):
        from scipy.stats.mstats import gmean

        def should_add_metric(prefix):
            return metrics is None or prefix in metrics

        if should_add_metric('Min'):
            metric[get_alias('Min')] = df[field].min()
        if should_add_metric('Max'):
            metric[get_alias('Max')] = df[field].max()
        if should_add_metric('Mean'):
            metric[get_alias('Mean')] = df[field].mean()
        if should_add_metric('Geomean'):
            metric[get_alias('Geomean')] = gmean(df[field])
        if should_add_metric('Std'):
            metric[get_alias('Std')] = df[field].std()

    def _register_all_algo_choice_metrics(self):
        tex_stmts = self._dedent_tex_stmts([
            # r"""
            # \begin{rlscope-finding}{find:software-overhead}
            # Most of the training time is spent in CPU, executing the software stack -- CUDA API calls, framework code, high-level code -- suggesting it is poorly optimized for the RL use case.
            # Even Inference and Backpropagation, which are GPU-heavy in SL workloads, spend at most \asPercent{0.141163009} executing GPU kernels.
            # \end{rlscope-finding}
            # """,

            r"""
            \begin{rlscope-finding}{find:algo-choice}
            On-policy algorithms are substantially more simulation-bound than off-policy algorithms, spending [$\times$????? more time in Simulation] than off-policy algorithms.
            %spending at least \asPercent{0.706128808} in simulation, [compared to at most XX\% for off policy]
            \end{rlscope-finding}
            """,
        ])
        self.ALGO_CHOICE_METRICS = self._register_metrics(tex_stmts)

    def _dedent_tex_stmts(self, tex_stmts):
        return [textwrap.dedent(stmt).lstrip().rstrip() for stmt in tex_stmts]

    def _register_metrics(self, tex_stmts):
        metrics = []
        for tex_stmt in tex_stmts:
            tex_lines = tex_stmt.splitlines()
            m = re.search(r'\{(?P<tex_label>find:[^}]+)\}', tex_lines[0])
            tex_label = m.group('tex_label')
            metrics.append(TexMetric(tex_stmt, tex_label, tex_variable_prefix=self.tex_variable_prefix))
        return metrics

    def _register_all_framework_choice_metrics(self):
        tex_stmts = self._dedent_tex_stmts([
            r"""
            \begin{rlscope-finding-qual}{find:qual-eager-more-trans}
            Eager execution is between $x\times$ and $y\times$ as bad as Graph/Autograph execution, and slowdown is highly correlated with how well a framework implementation is optimized to minimize Framework transitions.
            \end{rlscope-finding-qual}
            """,

            r"""
            \begin{rlscope-finding}{find:qual-autograph-reduces-python}
            By removing Framework transitions, \autograph substantially reduces Python time so 
            that is makes up at most \FCAsPercent{\FindQualAutographReducesPythonMaxAutographPythonPercentOfOp} 
            of Inference/Backpropagation time.
            \end{rlscope-finding}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-ddpg-backprop-slow}
            RLScope's detailed metrics identify subtle performance differences in the DDPG Graph API 
            RL algorithm implementation rooted in inefficient abstractions in high-level code 
            responsible for a $x\times$ inflation in total Graph API Backpropagation time compared 
            to Autograph.
            \end{rlscope-finding-surp}
            """,

            r"""
            \begin{rlscope-finding-qual}{find:qual-pytorch-eager-better}
            PyTorch eager is $x\times$ faster than TensorFlow eager since it minimizes Framework transitions more effectively in Inference.
            \end{rlscope-finding-qual}
            """,

            r"""
            \begin{rlscope-finding-qual}{find:qual-autograph-graph-similar}
            \autograph and \graph execution are on-par with one another, performing within 
            \FCAsPercent{\FindQualAutographGraphSimilarMaxSpeedup} of one another on a given 
            RL algorithm; \autograph does \textit{not} always outperform \graph.
            \end{rlscope-finding-qual}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-exec-model-comparison}
            Eager execution vs Graph/Autograph: PyTorch eager is $y\times$ faster than the best Graph/Autograph implementation.
            \end{rlscope-finding-surp}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-total-gpu-time}
            Total GPU time is similar across all RL frameworks regardless of DL back-end, and consistently low across all RL frameworks, making up at most $x\%$ of total training time.
            \end{rlscope-finding-surp}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-cuda-api-dominates}
            In all RL frameworks, CPU-side CUDA API time dominates total GPU kernel execution time, taking up between $x\times$ and $y\times$ as much time as GPU kernel execution.
            \end{rlscope-finding-surp}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-autograph-no-gpu}
            For RL workloads, even though Autograph converts RL training code to in-graph TensorFlow operators, Autograph has no perceivable increase in total GPU training time.
            \end{rlscope-finding-surp}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-autograph-inflates-inference}
            Inference time in Autograph is inflated by Framework time ($x\times$ compared to Graph), and this inflation is \textit{not} due to extra Framework transitions, and is instead a performance anomaly within the DL back-end itself.
            \end{rlscope-finding-surp}
            """,

            r"""
            \begin{rlscope-finding-surp}{find:surp-autograph-inflates-python}
            Autograph can inflate Python time by as much as $2\times$ during Simulation; training times (not just model performance) are highly sensitive to small differences in hyperparameter choices.
            \end{rlscope-finding-surp}
            """,
        ])
        self.FRAMEWORK_CHOICE_METRICS = self._register_metrics(tex_stmts)

    def _register_all_framework_choice_uncorrected_metrics(self):
        tex_stmts = self._dedent_tex_stmts([
            r"""
            \begin{rlscope-finding}{find:uncorrected-training-time-inflation}
            Total training time of the workloads we measured becomes inflated 
            from $x \times$ up to $y \times$, and $z \times$ on average.
            \end{rlscope-finding}
            """,
        ])
        self.FRAMEWORK_CHOICE_UNCORRECTED_METRICS = self._register_metrics(tex_stmts)

    def calc_find_uncorrected_training_time_inflation(self, metric):
        r"""
        \begin{rlscope-finding}{find:uncorrected-training-time-inflation}
        Total training time of the workloads we measured becomes inflated
        from $x \times$ up to $y \times$, and $z \times$ on average.
        \end{rlscope-finding}

        ratio_corrected_to_uncorrected =
            df['total_training_time', algo, env, x_field] /
            df_uncorrected['total_training_time', algo, env, x_field]
        """
        df = self.framework_choice_df
        field = 'total_training_time'
        df_mean = self.mean_df(field, df=self.framework_choice_df)
        df_uncorrected_mean = self.mean_df(field, df=self.framework_choice_uncorrected_df)

        # Join on the exact same (algo, env, DL backend).
        join_on = ['algo', 'env', 'x_field']
        df = df_mean.set_index(join_on).join(df_uncorrected_mean.set_index(join_on), how='inner',
                                             # lsuffix='',
                                             rsuffix='_uncorrected')
        df['ratio'] = df['total_training_time_uncorrected'] / df['total_training_time']

        metric['MinRatio'] = df['ratio'].min()
        metric['MaxRatio'] = df['ratio'].max()
        metric['MeanRatio'] = df['ratio'].mean()
        metric.df = df

def output_csv(plot_df, base_path, sort_by=None):
    if type(plot_df.index) == pd.MultiIndex:
        plot_df = plot_df.reset_index()

    if sort_by is not None:
        plot_df = plot_df.sort_values(sort_by)

    csv_path = f"{base_path}.csv"
    df_path = f"{base_path}.dataframe.txt"

    logger.info("Output csv @ {path}".format(path=csv_path))
    plot_df.to_csv(csv_path, index=False)

    logger.info("Output dataframe @ {path}".format(path=df_path))
    with open(df_path, 'w') as f:
        DataFrame.print_df(plot_df, file=f)
    DataFrame.print_df(plot_df, file=sys.stdout)

def maybe_suffix(suffix):
    if suffix is None or suffix == "":
        return ""
    if re.search(r'^\.', suffix):
        return suffix
    return f".{suffix}"

def _texvar(i):
    alphabet_size = 26*2
    length = int(np.ceil((i + 1) / alphabet_size))
    if (i % alphabet_size) < 26:
        letter = chr(ord('a') + (i % 26))
    else:
        # elif (i % alphabet_size) < 26*2:
        letter = chr(ord('A') + (i % 26))
    # double-up letters:
    # a,  b,  c,  ..., z
    # A,  B,  C,  ..., Z
    # aa, bb, cc, ..., cc
    # AA, BB, CC, ..., ZZ
    # ...
    return letter * length

def list_maybe_extend(lst, append_lst):
    for elem in append_lst:
        if elem not in lst:
            lst.append(elem)

if __name__ == '__main__':
    main()
