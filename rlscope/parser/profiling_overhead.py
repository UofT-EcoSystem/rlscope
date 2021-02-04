"""
Calibrating for average book-keeping durations needed to
correct for RL-Scope profiling overhead.
"""
from rlscope.profiler.rlscope_logging import logger
import copy
import warnings
import numpy as np
import itertools
import argparse
from contextlib import contextmanager
from decimal import Decimal

from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization, DeviceUtilization, UtilizationSample
from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.profiler import experiment
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
import pandas as pd
from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
import matplotlib
# matplotlib.use('agg')
import seaborn as sns

from matplotlib import pyplot as plt

from rlscope.profiler.util import pprint_msg
from rlscope.parser.dataframe import TrainingProgressDataframeReader, CUDAAPIStatsDataframeReader, PyprofDataframeReader, read_rlscope_config, DataframeMapper, RLScopeConfig, VennData, map_readers, get_training_durations_df
from rlscope.parser.readers import OpStackReader, CUDAAPIStatsReader
from rlscope.parser.stacked_bar_plots import StackedBarPlot
from rlscope.parser import stacked_bar_plots
from rlscope.parser.plot_utils import is_pdf, pdf2png

from rlscope.experiment import expr_config

from rlscope.parser import stacked_bar_plots
from rlscope.parser.db import SQLCategoryTimesReader, CSVInserter, \
    sql_input_path, \
    sql_create_connection, \
    sql_exec_query, \
    sql_operator_in, \
    sql_fetch_rows, \
    sql_count_from, \
    ConnectionPoolManager, \
    get_sql_connection, \
    RowIterator

from rlscope.parser.stats import KernelTime

from rlscope.profiler.rlscope_logging import logger
from rlscope.parser.plot import get_sns_kwargs, get_plt_kwargs, add_grouped_stacked_bars, add_bar_labels

class CalibrationJSON:
    def __init__(self, path, as_df=None):
        self.path = path
        self.as_df = as_df

        if self.path is not None:
            self.js = load_json(path)

            if self.as_df is not None:
                self.df = self.as_df(self.js)

    def is_given(self):
        return self.path is not None

    def __getitem__(self, item):
        return self.js[item]

class CalibrationJSONs:
    def __init__(self,
                 cupti_overhead_json=None,
                 LD_PRELOAD_overhead_json=None,
                 python_annotation_json=None,
                 python_clib_interception_tensorflow_json=None,
                 python_clib_interception_simulator_json=None,
                 # Swallow extra kwargs
                 **kwargs):

        self.cupti_overhead_json = CalibrationJSON(cupti_overhead_json, as_df=self._cupti_as_df)
        self.LD_PRELOAD_overhead_json = CalibrationJSON(LD_PRELOAD_overhead_json)
        self.python_annotation_json = CalibrationJSON(python_annotation_json)
        self.python_clib_interception_tensorflow_json = CalibrationJSON(python_clib_interception_tensorflow_json)
        self.python_clib_interception_simulator_json = CalibrationJSON(python_clib_interception_simulator_json)

    @staticmethod
    def add_argparse(parser, required=False):
        parser.add_argument(
            '--cupti-overhead-json',
            help="Calibration: mean per-CUDA API CUPTI overhead when GPU activities are recorded (see: CUPTIOverheadTask)",
            required=required,
        )
        parser.add_argument(
            '--LD-PRELOAD-overhead-json',
            help="Calibration: mean overhead for intercepting CUDA API calls with LD_PRELOAD  (see: CallInterceptionOverheadTask)",
            required=required,
        )
        parser.add_argument(
            '--python-clib-interception-tensorflow-json',
            help="Calibration: means for TensorFlow Python->C++ interception overhead (see: PyprofOverheadTask)",
            required=required,
        )
        parser.add_argument(
            '--python-clib-interception-simulator-json',
            help="Calibration: means for Simulator Python->C++ interception overhead (see: PyprofOverheadTask)",
            required=required,
        )
        parser.add_argument(
            '--python-annotation-json',
            help="Calibration: means for operation annotation overhead (see: PyprofOverheadTask)",
            required=required,
        )

    @staticmethod
    def from_obj(obj):
        calibration_jsons = CalibrationJSONs(**vars(obj))
        return calibration_jsons

    def _cupti_as_df(self, js):
        cupti_overhead_cols = dict()
        for api_name, cupti_data in js.items():
            add_col(cupti_overhead_cols, 'api_name', api_name)
            for field, value in cupti_data.items():
                add_col(cupti_overhead_cols, field, value)
        df = pd.DataFrame(cupti_overhead_cols)
        return df

    def _as_df(self, js):
        cols = dict()
        for field, value in js.items():
            add_col(cols, field, value)
        df = pd.DataFrame(cols)
        return df

    def argv(self):
        argv = []
        def _add(opt, json):
            if json.is_given():
                argv.extend([opt, json.path])
        _add('--python-clib-interception-tensorflow-json', self.python_clib_interception_tensorflow_json)
        _add('--python-clib-interception-simulator-json', self.python_clib_interception_tensorflow_json)
        _add('--python-annotation-json', self.python_annotation_json)
        _add('--cupti-overhead-json', self.cupti_overhead_json)
        _add('--LD-PRELOAD-overhead-json', self.LD_PRELOAD_overhead_json)
        return argv

class CorrectedTrainingTimeParser:
    """
    Compute total training time, after "subtracting" various sources of profiling overhead.
    """
    def __init__(self,
                 cupti_overhead_json,
                 LD_PRELOAD_overhead_json,
                 python_annotation_json,
                 python_clib_interception_tensorflow_json,
                 python_clib_interception_simulator_json,
                 rlscope_directories,
                 uninstrumented_directories,
                 directory,
                 rlscope_prof_config,
                 width=None,
                 height=None,
                 debug=False,
                 debug_single_thread=False,
                 debug_memoize=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """

        self.calibration_jsons = CalibrationJSONs(
            cupti_overhead_json=cupti_overhead_json,
            LD_PRELOAD_overhead_json=LD_PRELOAD_overhead_json,
            python_annotation_json=python_annotation_json,
            python_clib_interception_tensorflow_json=python_clib_interception_tensorflow_json,
            python_clib_interception_simulator_json=python_clib_interception_simulator_json,
        )

        # NOTE: we don't need "should_subtract_pyprof"

        self.rlscope_directories = rlscope_directories
        self.uninstrumented_directories = uninstrumented_directories
        self.directory = directory
        self.rlscope_prof_config = rlscope_prof_config

        self.width = width
        self.height = height
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.debug_memoize = debug_memoize

        rlscope_config = RLScopeConfig(self.rlscope_directories[0])
        if self.debug:
            logger.info("RLScopeConfig.rlscope_prof_args: {msg}".format(
                msg=pprint_msg({
                    'rlscope_config.rlscope_prof_args': rlscope_config.rlscope_prof_args,
                    # 'rlscope_config.rlscope_config': rlscope_config.rlscope_config,
                })))

        def check_all_true(var):
            for directory in self.rlscope_directories:
                rlscope_config = RLScopeConfig(directory)
                assert rlscope_config.get_env_bool(var)

        self.should_subtract_cupti = False
        self.should_subtract_LD_PRELOAD = False
        self.should_subtract_pyprof_annotation = False
        self.should_subtract_pyprof_interception = False

        if rlscope_config.get_env_bool('cuda_activities'):
            # $ rls-prof --cuda-activities
            check_all_true('cuda_activities')
            self.should_subtract_cupti = True

        if rlscope_config.get_env_bool('cuda_api_calls') and rlscope_config.get_env_bool('cuda_api_events'):
            # $ rls-prof --cuda-api-calls --cuda-api-events
            check_all_true('cuda_api_calls')
            check_all_true('cuda_api_events')
            self.should_subtract_LD_PRELOAD = True

        if rlscope_config.get_env_bool('cuda_api_calls') and not rlscope_config.get_env_bool('cuda_api_events'):

            # $ rls-prof --config gpu-activities --cuda-api-calls --cuda-api-events
            # ===
            # $ rls-prof --cuda-api-calls --cuda-activities
            logger.info(textwrap.dedent("""\
                WARNING: we cannot correct for runs like this, so there will be positive overhead (%):
                    $ rls-prof --cuda-api-calls --cuda-activities
                In particular, LD_PRELOAD overhead is measured using "--cuda-api-calls --cuda-api-events", but 
                we currently dont ever just measure "--cuda-api-calls".
                """))

        if not rlscope_config.get_bool('disable') and \
            not rlscope_config.get_bool('disable_pyprof') and \
            not rlscope_config.get_bool('disable_pyprof_annotations'):
            self.should_subtract_pyprof_annotation = True

        if not rlscope_config.get_bool('disable') and \
            not rlscope_config.get_bool('disable_pyprof') and \
            not rlscope_config.get_bool('disable_pyprof_interceptions'):
            self.should_subtract_pyprof_interception = True

        should_subtract_attrs = dict((attr, val) for attr, val in self.__dict__.items() \
                                     if re.search(r'should_subtract', attr))
        logger.info("Correction configuration: {msg}".format(
            msg=pprint_msg(should_subtract_attrs)))

        # NOTE: for pyprof overhead we SHOULD just be able to subtract regardless...
        # if there's no pyprof activity, then there should be no events present.

    # def _init_cupti_overhead_df(self):
    #     cupti_overhead_cols = dict()
    #     for api_name, cupti_data in self.cupti_overhead_json.items():
    #         add_col(cupti_overhead_cols, 'api_name', api_name)
    #         for field, value in cupti_data.items():
    #             add_col(cupti_overhead_cols, field, value)
    #     self.cupti_overhead_df = pd.DataFrame(cupti_overhead_cols)
    #
    # def _init_pyprof_overhead_df(self):
    #     pyprof_overhead_cols = dict()
    #     for field, value in self.pyprof_overhead_json.items():
    #         add_col(pyprof_overhead_cols, field, value)
    #     self.pyprof_overhead_df = pd.DataFrame(pyprof_overhead_cols)

    @property
    def cupti_overhead_json(self):
        return self.calibration_jsons.cupti_overhead_json
    @property
    def LD_PRELOAD_overhead_json(self):
        return self.calibration_jsons.LD_PRELOAD_overhead_json
    @property
    def python_annotation_json(self):
        return self.calibration_jsons.python_annotation_json
    @property
    def python_clib_interception_tensorflow_json(self):
        return self.calibration_jsons.python_clib_interception_tensorflow_json
    @property
    def python_clib_interception_simulator_json(self):
        return self.calibration_jsons.python_clib_interception_simulator_json

    def run(self):
        """
        Sources of overhead:
        - CategoryEventsProto overhead = sum(Event.duration_profiling_overhead_us If Event.start_profiling_overhead_us != 0)
          - Need to parse pyprof Event files
        - CUPTI overhead = sum(api.n_calls * api.mean_cupti_per_call_overhead_us)
          - parse api.n_calls using CUDAAPIStatsDataframeReader
          - parse api.mean_cupti_per_call_overhead_us using cupti_overhead_json
        - Interception overhead = sum(total_n_calls * api.mean_interception_per_call_overhead_us)
          - parse total_n_calls using CUDAAPIStatsDataframeReader
          - parse api.mean_cupti_per_call_overhead_us using LD_PRELOAD_overhead_json
        - "Subtracted" time = total_training_time - [ … sum of overheads … ]
          - parse total_training_time using TrainingProgressDataframeReader

        Stats to save:
            # Overheads:
            CategoryEventsProto overhead (absolute time, and percent)
            CUPTI overhead (absolute time, and percent)
            Interception overhead (absolute time, and percent)

            # Training time:
            Total training time (before subtraction correction)
            "Subtracted" total training time

        csv output:
            overhead_pyprof_sec/perc,
            overhead_cupti_sec/perc,
            overhead_interception_sec/perc,

            total_training_time_sec,
            corrected_total_training_time_sec,

            Q: Should we include data from json files used for computing overheads?

        json output:
        {
            [num/std/mean] for all fields in the csv
            e.g.
            [num/std/mean]_total_training_time_sec:
        }

        output directory:
            --rlscope-directory containing profiling data.
        """

        # capsize = 5
        # plt.style.use('seaborn')
        # plt.rcParams.update({
        #     "lines.markeredgewidth" : 1,
        #     "errorbar.capsize": capsize,
        # })

        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        PLOT_GROUP_ALGO_CHOICE = "Algorithm choice"
        PLOT_GROUP_ENV_CHOICE = "Environment choice"
        PLOT_GROUP_RL_WORKLOAD = "Misc"
        def as_plot_group(algo, env):
            if expr_config.is_fig_algo_comparison_med_complexity(algo, env):
                return PLOT_GROUP_ALGO_CHOICE
            elif expr_config.is_fig_env_comparison(algo, env):
                return PLOT_GROUP_ENV_CHOICE
            else:
                return PLOT_GROUP_RL_WORKLOAD


        def add_x_field(df):
            # df = copy.copy(df)
            def _x_type(plot_group):
                if plot_group == PLOT_GROUP_ENV_CHOICE:
                    return 'env-comparison'
                elif plot_group == PLOT_GROUP_ALGO_CHOICE:
                    return 'algo-comparison'
                return 'rl-comparison'

            def _x_field(plot_group, algo, env):
                x_type = _x_type(plot_group)
                return stacked_bar_plots.get_x_field(
                    algo, env,
                    x_type=x_type,
                    human_readable=True)
            df['plot_group'] = np.vectorize(as_plot_group, otypes=[str])(df['algo'], df['env'])
            df['x_field'] = np.vectorize(_x_field, otypes=[str])(df['plot_group'], df['algo'], df['env'])
            # return df

        def add_fields(df, rlscope_config):
            add_rlscope_config(df, rlscope_config)

        def load_dfs():
            memoize_path = _j(self.directory, "{klass}.load_dfs.pickle".format(
                klass=self.__class__.__name__))

            if should_load_memo(self.debug_memoize, memoize_path):
                ret = load_memo(self.debug_memoize, memoize_path)
                return ret

            # corrected_training_time.per_api.raw.csv
            # per-api csv output:
            #     api_name,
            #     n_calls,
            #     total_cupti_overhead_us,
            #     total_interception_overhead_us,
            #     total_overhead_us,
            per_api_dfs = []
            # total csv output (summed over all api calls over the entire training script):
            #     total_pyprof_overhead_us
            #     total_cupti_overhead_us
            #     total_interception_overhead_us
            #     total_training_duration_us
            #     total_overhead_us
            #     total_training_duration_us
            #     corrected_total_training_duration_us
            total_dfs = []

            if self.debug:
                logger.info("load_dfs: {msg}".format(
                    msg=pprint_msg({
                        'rlscope_directories':self.rlscope_directories,
                    })
                ))

            # for directory in self.rlscope_directories:
            for directory in progress(self.rlscope_directories,
                                 desc="load_dfs.rlscope_directories",
                                 show_progress=self.debug):

                if self.debug:
                    logger.info("rlscope_directory = {dir}".format(
                        dir=directory,
                    ))

                rlscope_config = read_rlscope_config(directory)

                total_training_duration_us = get_training_durations(directory, debug=self.debug, debug_single_thread=self.debug_single_thread)

                """
                - CUPTI overhead = sum(api.n_calls * api.mean_cupti_per_call_overhead_us for each api)
                  - parse api.n_calls using CUDAAPIStatsDataframeReader
                  - parse api.mean_cupti_per_call_overhead_us using cupti_overhead_json
                """
                # TODO: make it work for multiple self.directories
                # Could be nice to see a stacked-bar graph that "breaks down" total training time by its overhead sources...
                # probably a pretty busy plot though.

                # TODO: if there are NO per-api stats, make get_per_api_stats return a dataframe with zero rows:
                # api_name, num_calls, ...
                per_api_stats = get_per_api_stats(directory, debug=self.debug, debug_single_thread=self.debug_single_thread)
                per_api_df = copy.copy(per_api_stats.reset_index())
                if self.should_subtract_LD_PRELOAD:
                    per_api_df['total_interception_overhead_us'] = per_api_df['num_calls'] * self.LD_PRELOAD_overhead_json['mean_interception_overhead_per_call_us']
                else:
                    per_api_df['total_interception_overhead_us'] = 0
                    logger.info("SKIP LD_PRELOAD overhead (total_interception_overhead_us = 0)")
                add_fields(per_api_df, rlscope_config)

                # per_api_df = pd.DataFrame({
                #     'api_name': per_api_stats_df['api_name'],
                #     'num_calls': per_api_stats_df['num_calls'],
                #     # 'total_cupti_overhead_us': per_api_stats_df['num_calls'] * self.cupti_overhead_json['mean_cupti_overhead_per_call_us'],
                #     'total_interception_overhead_us': per_api_stats_df['num_calls'] * self.LD_PRELOAD_overhead_json['mean_interception_overhead_per_call_us'],
                # })

                # - make json a df
                # - join on api_name, make column 'mean_cupti_overhead_per_call_us'
                # - multiply mean by num_calls

                per_api_df = per_api_df.merge(self.cupti_overhead_json.df, on=['api_name'])
                if self.should_subtract_cupti:
                    per_api_df['total_cupti_overhead_us'] = per_api_df['num_calls'] * per_api_df['mean_cupti_overhead_per_call_us']
                else:
                    per_api_df['total_cupti_overhead_us'] = 0
                    logger.info("SKIP CUPTI overhead (total_cupti_overhead_us = 0)")

                per_api_df['total_overhead_us'] = per_api_df['total_cupti_overhead_us'] + per_api_df['total_interception_overhead_us']
                per_api_dfs.append(per_api_df)

                total_cupti_overhead_us = np.sum(per_api_df['total_cupti_overhead_us'])

                """
                Pyprof overhead:
                - Python->C++ overhead = pyprof.num_intercepted_calls * json['mean_pyprof_interception_per_call_overhead_us']
                - Operation annotation overhead = pyprof.num_annotations * json['mean_pyprof_annotation_per_call_overhead_us']
                - total_pyprof_overhead_us = [Python->C++ overhead] + [Operation annotation overhead]
                """

                # NOTE: we'd LIKE to have a plot with a breakdown into the different pyprof overheads...
                pyprof_mapper = DataframeMapper(PyprofDataframeReader, directories=[directory], debug=self.debug)

                if self.should_subtract_pyprof_interception:
                    total_tensorflow_intercepted_calls = pyprof_mapper.map_one(lambda reader: reader.total_intercepted_tensorflow_calls())
                    total_python_tensorflow_interception_overhead_us = total_tensorflow_intercepted_calls * self.python_clib_interception_tensorflow_json['mean_pyprof_interception_overhead_per_call_us']
                else:
                    total_python_tensorflow_interception_overhead_us = 0

                if self.should_subtract_pyprof_interception:
                    total_simulator_intercepted_calls = pyprof_mapper.map_one(lambda reader: reader.total_intercepted_simulator_calls())
                    total_python_simulator_interception_overhead_us = total_simulator_intercepted_calls * self.python_clib_interception_simulator_json['mean_pyprof_interception_overhead_per_call_us']
                else:
                    total_python_simulator_interception_overhead_us = 0

                if self.should_subtract_pyprof_annotation:
                    total_python_annotations = pyprof_mapper.map_one(lambda reader: reader.total_annotations())
                    total_python_annotation_overhead_us = total_python_annotations * self.python_annotation_json['mean_pyprof_annotation_overhead_per_call_us']
                else:
                    total_python_annotation_overhead_us = 0

                """
                - Interception overhead = sum(total_n_calls * api.mean_interception_per_call_overhead_us)
                  - parse total_n_calls using CUDAAPIStatsDataframeReader
                  - parse api.mean_cupti_per_call_overhead_us using LD_PRELOAD_overhead_json
                """
                total_interception_overhead_us = np.sum(per_api_df['total_interception_overhead_us'])

                """
                - "Subtracted" time = total_training_time - [ … sum of overheads … ]
                  - parse total_training_time using TrainingProgressDataframeReader
                """

                # total_training_duration_us = get_training_durations(directory, debug=self.debug, debug_single_thread=self.debug_single_thread)

                total_overhead_data = {
                    'total_cupti_overhead_us': [total_cupti_overhead_us],
                    'total_interception_overhead_us': [total_interception_overhead_us],

                    'total_python_tensorflow_interception_overhead_us': [total_python_tensorflow_interception_overhead_us],
                    'total_python_simulator_interception_overhead_us': [total_python_simulator_interception_overhead_us],

                    'total_python_annotation_overhead_us': [total_python_annotation_overhead_us],

                    'total_training_duration_us': [total_training_duration_us],
                }
                total_df = pd.DataFrame(total_overhead_data)
                add_fields(total_df, rlscope_config)

                # WARNING: protect against bug where we create more than one row unintentionally.
                # If we mix scalars/lists, pd.DataFrame will "duplicate" the scalars to match the list length.
                # This can happen if we accidentally include per-api times instead of summing across times.
                assert len(total_df) == 1

                overhead_colnames = [col for col in total_df.keys() if is_total_overhead_column(col)]
                total_df['total_overhead_us'] = 0
                for col in overhead_colnames:
                    total_df['total_overhead_us'] += total_df[col]

                total_df['corrected_total_training_duration_us'] = total_df['total_training_duration_us'] - total_df['total_overhead_us']
                total_dfs.append(total_df)

            total_df = pd.concat(total_dfs)
            per_api_df = pd.concat(per_api_dfs)
            ret = (total_df, per_api_df)

            maybe_memoize(self.debug_memoize, ret, memoize_path)

            return ret

        total_df, per_api_df = load_dfs()
        add_x_field(total_df)
        add_x_field(per_api_df)

        # PROBLEM: we need to plot the uninstrumented runtime as well to verify the "corrected" time is correct.
        # TODO: output format that ProfilingOverheadPlot can read, reuse that plotting code.

        # def mean_colname(col):
        #     return "mean_{col}".format(col=col)
        # def std_colname(col):
        #     return "std_{col}".format(col=col)
        # def num_colname(col):
        #     return "num_{col}".format(col=col)
        #
        # json_data = dict()
        # for name, values in df.iteritems():
        #     json_data[mean_colname(name)] = np.mean(values)
        #     json_data[std_colname(name)] = np.std(values)
        #     json_data[num_colname(name)] = np.len(values)
        # logger.info("Output json @ {path}".format(path=self._raw_json_path))
        # do_dump_json(json_data, self._raw_json_path)

        def pretty_config(config):
            if config == 'uninstrumented':
                # return 'CUPTI GPU activities enabled'
                return 'Uninstrumented'
            elif config == 'instrumented':
                # return 'CUPTI GPU activities disabled'
                return 'Instrumented (uncorrected)'
            elif config == 'corrected':
                return 'Instrumented (corrected)'
            else:
                raise NotImplementedError()

        def get_per_api_plot_data(per_api_df):
            overhead_cols = ['total_cupti_overhead_us', 'total_interception_overhead_us']
            # keep_cols = ['api_name', 'num_calls', 'algo', 'env', 'x_field'] + overhead_cols
            # keep_cols = ['api_name', 'num_calls'] + overhead_cols
            # Keep these columns "intact" (remove 'total_overhead_us')
            id_vars = ['api_name', 'num_calls', 'algo', 'env', 'x_field']
            keep_cols = id_vars + overhead_cols
            keep_df = per_api_df[keep_cols]
            per_api_plot_data = pd.melt(keep_df, id_vars=id_vars, var_name='overhead_type', value_name='total_overhead_us')
            per_api_plot_data['pretty_overhead_type'] = per_api_plot_data['overhead_type'].apply(
                lambda overhead_type: pretty_overhead_type(overhead_type, 'us'))
            per_api_plot_data['total_overhead_sec'] = per_api_plot_data['total_overhead_us'] / constants.USEC_IN_SEC
            del per_api_plot_data['total_overhead_us']
            return per_api_plot_data

        # if self.width is not None and self.height is not None:
        #     figsize = (self.width, self.height)
        #     logger.info("Setting figsize = {fig}".format(fig=figsize))
        #     # sns.set_context({"figure.figsize": figsize})
        # else:
        #     figsize = None


        # figsize = (7, 5)
        # figsize = (7, 8)
        # if self.plot_group == 'environment_choice':
        #     figsize = (9, 7)
        # elif self.plot_group == 'algorithm_choice':
        #     figsize = (7, 6)
        # else:
        figsize = None

        logger.info("Plot dimensions (inches) = {figsize}".format(
            figsize=figsize))



        per_api_plot_data = get_per_api_plot_data(per_api_df)
        output_csv(per_api_plot_data, self._per_api_csv_path, sort_by=['total_overhead_sec'])
        if len(per_api_plot_data) > 0:


            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            sns.barplot(
                x='api_name', y='total_overhead_sec', hue='pretty_overhead_type', data=per_api_plot_data, ax=ax,
                **sns_kwargs)
            ax.legend().set_title(None)
            ax.set_ylabel('Total training time (sec)')
            ax.set_xlabel('CUDA API call')
            ax.set_title("Breakdown of profiling overhead by CUDA API call")
            logger.info("Output plot @ {path}".format(path=self._per_api_png_path))
            fig.savefig(self._per_api_png_path, bbox_inches="tight", pad_inches=0)
            if is_pdf(self._per_api_png_path):
                pdf2png(self._per_api_png_path)
            plt.close(fig)

        def get_total_plot_data(total_df):
            #     total_cupti_overhead_us
            #     total_interception_overhead_us
            #     total_training_duration_us
            #     corrected_total_training_duration_us
            #
            #     total_overhead_us
            #     total_training_duration_us
            total_plot_data = copy.copy(total_df)
            # Ideally, (algo, env), but we don't have that luxury.
            # total_plot_data['x_field'] = ""

            # Melt overhead columns, convert to seconds.

            # We don't have any unique column values for this data-frame, so we cannot use id_vars...
            id_vars = [
                # 'total_training_duration_us',
                "x_field",
                'algo',
                'env',
            ]
            overhead_cols = [
                col for col in total_df.keys()
                if is_total_overhead_column(col) and col != 'total_overhead_us']
            value_vars = overhead_cols + ['corrected_total_training_duration_us']

            # Keep these columns "intact" (remove 'total_overhead_us')
            keep_cols = id_vars + value_vars
            keep_df = total_plot_data[keep_cols]
            # total_plot_data = pd.melt(keep_df, id_vars=id_vars, var_name='overhead_type', value_name='total_overhead_us')
            total_plot_data = pd.melt(keep_df, id_vars=id_vars, var_name='overhead_type', value_name='total_overhead_us')
            total_plot_data['total_overhead_sec'] = total_plot_data['total_overhead_us'] / constants.USEC_IN_SEC
            del total_plot_data['total_overhead_us']
            total_plot_data['pretty_overhead_type'] = total_plot_data['overhead_type'].apply(
                lambda overhead_type: pretty_overhead_type(overhead_type, 'us'))

            # Q: if we sort do we get consistent order?
            # total_plot_data = total_plot_data.sort_values(['x_field', 'pretty_overhead_type'])

            overhead_type_map = as_order_map(reversed(value_vars))
            total_plot_data['overhead_type_order'] = total_plot_data['overhead_type'].apply(
                lambda overhead_type: overhead_type_map[overhead_type])
            total_plot_data = total_plot_data.sort_values(['x_field', 'overhead_type_order'])

            return total_plot_data

        total_plot_data = get_total_plot_data(total_df)
        output_csv(total_plot_data, self._total_csv_path)
        if len(total_plot_data) > 0:
            fig = plt.figure(figsize=figsize)
            # plot_data['field'] = "Per-API-call interception overhead"
            ax = fig.add_subplot(111)
            # sns.barplot(x='x_field', y='training_duration_sec', hue='pretty_config', data=training_duration_plot_data, ax=ax)
            add_stacked_bars(
                x='x_field',
                y='total_overhead_sec',
                hue='overhead_type_order',
                label='pretty_overhead_type',
                data=total_plot_data, ax=ax,
                debug=self.debug,
                **plt_kwargs)
            # ax.legend().set_title(None)
            ax.set_ylabel('Total training time (sec)')
            ax.set_xlabel('(algo, env)')
            ax.set_title("Breakdown of profiling overhead")
            logger.info("Output plot @ {path}".format(path=self._total_png_path))
            fig.savefig(self._total_png_path, bbox_inches="tight", pad_inches=0)
            if is_pdf(self._total_png_path):
                pdf2png(self._total_png_path)
            plt.close(fig)

        # JAMES: uninstrumented training time
        def load_unins_training_durations_df():
            memoize_path = _j(self.directory, "{klass}.load_unins_training_durations_df.pickle".format(
                klass=self.__class__.__name__))

            if should_load_memo(self.debug_memoize, memoize_path):
                ret = load_memo(self.debug_memoize, memoize_path)
                return ret

            ret = pd.concat(get_training_durations_df(self.uninstrumented_directories, debug=self.debug, debug_single_thread=self.debug_single_thread))

            maybe_memoize(self.debug_memoize, ret, memoize_path)

            return ret


        unins_df = load_unins_training_durations_df()
        unins_df['config'] = 'uninstrumented'

        ins_df = pd.DataFrame({
            'training_duration_us': total_df['total_training_duration_us'],
            'algo': total_df['algo'],
            'env': total_df['env'],
        })
        ins_df['config'] = 'instrumented'

        corrected_df = pd.DataFrame({
            'training_duration_us': total_df['corrected_total_training_duration_us'],
            'algo': total_df['algo'],
            'env': total_df['env'],
        })
        corrected_df['config'] = 'corrected'

        def get_figsize(plot_group):

            SMALL_SIZE = 8
            # Default font size for matplotlib (too small for paper).
            MEDIUM_SIZE = 10
            BIGGER_SIZE = 12

            FONT_SIZE = None

            # if plot_group == PLOT_GROUP_ALGO_CHOICE:
            #     figsize = (10, 6.5)
            #     FONT_SIZE = BIGGER_SIZE + 4
            # elif plot_group == PLOT_GROUP_ENV_CHOICE:
            #     figsize = (7, 5.5)
            #     FONT_SIZE = BIGGER_SIZE + 2
            # else:
            #     # PLOT_GROUP_RL_WORKLOAD
            #     figsize = None

            if plot_group in {PLOT_GROUP_ALGO_CHOICE, PLOT_GROUP_ENV_CHOICE}:
                figsize = (10, 9.5)
                FONT_SIZE = BIGGER_SIZE + 8
            else:
                # PLOT_GROUP_RL_WORKLOAD
                figsize = None

            if FONT_SIZE is not None:
                plt.rc('font', size=FONT_SIZE)          # controls default text sizes
                plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
                plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
                plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
                plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

            return figsize

        def get_xlabel(plot_group):
            if plot_group == PLOT_GROUP_ALGO_CHOICE:
                return "RL algorithm"
            elif plot_group == PLOT_GROUP_ENV_CHOICE:
                return "Simulator"
            else:
                return "(RL algorithm, Simulator)"

        XGROUP_OVERHEAD = 0
        XGROUP_UNINS = 1
        def get_low_bias_overhead_correction_plot_data(total_plot_data, unins_df):
            """
            Get data in csv format needed for "Figure 8: Low-bias overhead correction".

            PSEUDOCODE:
            Total_df[config] = "instrumented"
            Total_df_melt = melt total_df into [algo, env, config, overhead_type, total_overhead_us]
            Unins_df["overhead_type"] = "uninstrumented_total_training_duration_us"
            Unins_df_melt = Change cols of Unins_df to  match [algo, env, config, overhead_type, total_overhead_us]
            df = Pd.concat(total_df_melt, unins_df_melt)
            Df[bar_label] = ""
            For algo, env in df:
                Df[config==uninstrumented, overhead_type=uninstrumented_total_training_duration_us, algo==algo, env==env][bar_label] =
                    ( Df[config==uninstrumented, overhead_type=uninstrumented_total_training_duration_us, algo==algo, env==env][training_duration_us] - Df[config==instrumented, overhead_type=corrected_total_training_time, algo==algo, env==env][training_duration_us] ) /
            """
            total_plot_data = copy.copy(total_plot_data)

            # unins_df
            #    training_duration_us  algo                                  env          config
            # 0           60486911091   dqn                   PongNoFrameskip-v4  uninstrumented
            # 0           47663230442  ddpg                 Walker2DBulletEnv-v0  uninstrumented
            # 0            2403215259   a2c                 Walker2DBulletEnv-v0  uninstrumented
            # 0           12817289121  ppo2                 HumanoidBulletEnv-v0  uninstrumented

            # total_plot_data
            #                            x_field                          overhead_type  total_overhead_sec                         pretty_overhead_type  overhead_type_order
            # 107             (a2c, LunarLander)   corrected_total_training_duration_us          140.016430                      Corrected training time                    0
            # 85              (a2c, LunarLander)    total_pyprof_annotation_overhead_us            4.261625                   Python annotation overhead                    1
            # 63              (a2c, LunarLander)  total_pyprof_interception_overhead_us            9.673592  Python $\rightarrow$ C-library interception                    2
            # 41              (a2c, LunarLander)         total_interception_overhead_us           18.424823             LD_PRELOAD interception overhead                    3
            # 19              (a2c, LunarLander)                total_cupti_overhead_us            6.890283                               CUPTI overhead                    4


            # WANT:
            # algo, env, x_field, duration_name, duration_pretty_name, config, config_pretty_name, duration_sec
            keep_cols = ['algo', 'env', 'x_field', 'duration_name', 'duration_pretty_name', 'duration_name_order', 'config', 'config_pretty_name', 'duration_sec']

            def as_config_pretty_name(config):
                if config == 'uninstrumented':
                    return 'Uninstrumented'
                elif config == 'instrumented':
                    return 'Instrumented'
                else:
                    raise NotImplementedError()

            def as_x_group(duration_name):
                if duration_name in {
                    'corrected_total_training_duration_us',
                    'total_python_annotation_overhead_us',
                    'total_python_simulator_interception_overhead_us',
                    'total_python_tensorflow_interception_overhead_us',
                    'total_interception_overhead_us',
                    'total_cupti_overhead_us',
                }:
                    return XGROUP_OVERHEAD

                if duration_name in {
                    'uninstrumented_total_training_duration_us',
                }:
                    return XGROUP_UNINS

                raise NotImplementedError("Not sure what x_group duration_name={name} belongs to".format(
                    name=duration_name))

            # Transform unins_df
            # max_duration_name_order = np.max(total_plot_data['overhead_type_order'])
            min_duration_name_order = np.min(total_plot_data['overhead_type_order'])
            unins_df['duration_us'] = unins_df['training_duration_us']
            add_x_field(unins_df)
            unins_df['duration_name'] = 'uninstrumented_total_training_duration_us'
            unins_df['config_pretty_name'] = unins_df['config'].apply(as_config_pretty_name)
            unins_df['duration_pretty_name'] = unins_df['duration_name'].apply(
                lambda overhead_type: pretty_overhead_type(overhead_type, 'us'))
            # unins_df['duration_name_order'] = max_duration_name_order + 1
            unins_df['duration_name_order'] = min_duration_name_order - 1
            # , colnames=['duration_us']
            unins_df = dataframe_replace_us_with_sec(unins_df)
            unins_df = unins_df[keep_cols]

            total_plot_data['config'] = 'instrumented'
            total_plot_data['config_pretty_name'] = total_plot_data['config'].apply(as_config_pretty_name)
            total_plot_data['duration_name'] = total_plot_data['overhead_type']
            total_plot_data['duration_name'] = total_plot_data['overhead_type']
            total_plot_data['duration_pretty_name'] = total_plot_data['pretty_overhead_type']
            total_plot_data['duration_sec'] = total_plot_data['total_overhead_sec']
            total_plot_data['duration_name_order'] = total_plot_data['overhead_type_order']
            total_plot_data = total_plot_data[keep_cols]

            low_bias_df = pd.concat([unins_df, total_plot_data])

            low_bias_df['x_group'] = low_bias_df['duration_name'].apply(as_x_group)
            low_bias_df['plot_group'] = np.vectorize(as_plot_group, otypes=[str])(low_bias_df['algo'], low_bias_df['env'])

            # Calculate bar-label: 100*(corrected - unins)/unins
            low_bias_df['bar_label'] = ''
            low_bias_df['percent_wrong'] = 0.
            groupby = low_bias_df.groupby(['algo', 'env'])
            for group, df_group in groupby:
                algo, env = group
                corrected = df_group[df_group['duration_name']=='corrected_total_training_duration_us']['duration_sec']
                assert len(corrected) == 1
                corrected = corrected.values[0]
                unins = df_group[df_group['duration_name']=='uninstrumented_total_training_duration_us']['duration_sec']
                assert len(unins) == 1
                unins = unins.values[0]
                percent_wrong = 100*(corrected - unins)/unins
                bar_label = '{percent:.1f}%'.format(percent=percent_wrong)

                # df_group[df_group['duration_name']=='uninstrumented_total_training_duration_us']['percent_wrong'] = percent_wrong
                # df_group[df_group['duration_name']=='uninstrumented_total_training_duration_us']['bar_label'] = bar_label

                selection = (low_bias_df['algo'] == algo) & \
                            (low_bias_df['env'] == env) & \
                            (low_bias_df['duration_name'] == 'uninstrumented_total_training_duration_us')

                low_bias_df.loc[selection, 'percent_wrong'] = percent_wrong
                low_bias_df.loc[selection, 'bar_label'] = bar_label


            # Only keep (algo, env) pairs that we want in the paper.
            is_in_paper = np.vectorize(expr_config.is_paper_env, otypes=[np.bool])(low_bias_df['algo'], low_bias_df['env'])
            low_bias_df = low_bias_df[is_in_paper]

            # low_bias_df = low_bias_df.sort_values(['algo', 'env', 'config', 'duration_name_order'])
            low_bias_df = low_bias_df.sort_values(['algo', 'env', 'duration_name_order'])

            return low_bias_df

        low_bias_df = get_low_bias_overhead_correction_plot_data(total_plot_data, unins_df)
        output_csv(low_bias_df, self._overhead_correction_csv_path)
        if len(low_bias_df) > 0:
            def get_xfield_order_map(plot_group, df):
                df = df[df['x_group'] == XGROUP_OVERHEAD]
                df = df[['x_field', 'duration_sec']].groupby(['x_field']).sum().reset_index()
                # x_fields = sorted(set(low_bias_group_df.sort_values(['duration_sec'])['x_field'].values))
                x_fields = df.sort_values(['duration_sec'])['x_field'].unique()
                xfield_to_idx = as_order_map(x_fields)
                idx_to_xfield = reverse_dict(xfield_to_idx)
                return xfield_to_idx, idx_to_xfield
            for plot_group, low_bias_group_df in low_bias_df.groupby(['plot_group']):
                figsize = get_figsize(plot_group)
                fig = plt.figure(figsize=figsize)
                xfield_to_idx, idx_to_xfield = get_xfield_order_map(plot_group, low_bias_group_df)
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
                def xfield_label_func(idx):
                    xfield = idx_to_xfield[idx]
                    return xfield
                low_bias_group_df = copy.copy(low_bias_group_df)
                low_bias_group_df['x_field_idx'] = low_bias_group_df['x_field'].apply(lambda x_field: xfield_to_idx[x_field])
                xgroup_barplot = add_grouped_stacked_bars(
                    # x='x_field',
                    x='x_field_idx',
                    x_group='x_group',
                    # x_group='x_group_idx',
                    y='duration_sec',
                    hue='duration_name_order',
                    label='duration_pretty_name',
                    label_order='duration_name_order',
                    bar_label='bar_label',
                    bar_label_x_offset=14,
                    # bar_label_kwargs=dict(
                    #     fontsize=9
                    # ),
                    data=low_bias_group_df,
                    ax=ax,
                    xfield_label_func=xfield_label_func,
                    rotation=10,
                    # rotation=None,
                    debug=self.debug,
                    **plt_kwargs)
                # HACK: make it so percent labels will fit in plot area.
                x_min, x_max = ax.get_xlim()
                ax.set_xlim([x_min, x_max*1.02])

                # for xgroup, barplot in xgroup_barplot.items():
                #     if xgroup == XGROUP_UNINS:
                #         add_ax_bar_labels(ax, barplot)
                # ax.legend().set_title(None)
                ax.set_ylabel('Total training time (sec)')
                xlabel = get_xlabel(plot_group)
                ax.set_xlabel(xlabel)
                # ax.set_title("Breakdown of profiling overhead")
                logger.info("Output plot @ {path}".format(path=self._overhead_correction_png_path(plot_group)))
                # fig.tight_layout()
                fig.savefig(self._overhead_correction_png_path(plot_group), bbox_inches="tight", pad_inches=0)
                if is_pdf(self._overhead_correction_png_path(plot_group)):
                    pdf2png(self._overhead_correction_png_path(plot_group))
                plt.close(fig)

        # unins_training_duration_us = get_training_durations(self.uninstrumented_directories, debug=self.debug)
        # unins_df = pd.DataFrame({
        #     'training_duration_us': unins_training_duration_us,
        # })
        # unins_df['config'] = 'uninstrumented'
        #
        # ins_df = pd.DataFrame({
        #     'training_duration_us': total_df['total_training_duration_us'],
        # })
        # ins_df['config'] = 'instrumented'
        #
        # corrected_df = pd.DataFrame({
        #     'training_duration_us': total_df['corrected_total_training_duration_us'],
        # })
        # corrected_df['config'] = 'corrected'

        training_duration_df = pd.concat([unins_df, ins_df, corrected_df])
        training_duration_df['pretty_config'] = training_duration_df['config'].apply(pretty_config)
        # add_fields(training_duration_df, rlscope_config)
        add_x_field(training_duration_df)

        def get_training_duration_plot_data(training_duration_df):
            training_duration_plot_data = copy.copy(training_duration_df)
            # TODO: (algo, env)
            # training_duration_plot_data['x_field'] = ""

            training_duration_plot_data['training_duration_sec'] = training_duration_plot_data['training_duration_us'] / constants.USEC_IN_SEC
            del training_duration_plot_data['training_duration_us']
            return training_duration_plot_data

        training_duration_plot_data = get_training_duration_plot_data(training_duration_df)

        def get_percent_bar_labels(df):
            # NOTE: need .values otherwise we get NaN's
            unins_time_sec = df[df['pretty_config'] == 'Uninstrumented']['training_duration_sec'].values
            df['perc'] = ( df['training_duration_sec'] - unins_time_sec ) / unins_time_sec
            def get_label(row):
                if row['pretty_config'] == 'Uninstrumented':
                    assert row['perc'] == 0.
                    return ""
                return "{perc:.1f}%".format(perc=100*row['perc'])

            bar_labels = df.apply(get_label, axis=1)
            df['bar_labels'] = bar_labels
            return bar_labels

        output_csv(training_duration_plot_data, self._training_time_csv_path, sort_by=['training_duration_sec'])
        # Fails on some runs... ignore.
        # if len(training_duration_plot_data) > 0:
        #     fig = plt.figure(figsize=figsize)
        #     ax = fig.add_subplot(111)
        #     sns.barplot(
        #         x='x_field', y='training_duration_sec', hue='pretty_config', data=training_duration_plot_data, ax=ax,
        #         **sns_kwargs)
        #     add_bar_labels(y='training_duration_sec', hue='pretty_config', ax=ax,
        #                    get_bar_labels=get_percent_bar_labels)
        #     ax.legend().set_title(None)
        #     ax.set_ylabel('Total training time (sec)')
        #     ax.set_xlabel('(algo, env)')
        #     ax.set_title("Correcting training time by subtracting profiling overhead")
        #     logger.info("Output plot @ {path}".format(path=self._training_time_png_path))
        #     fig.savefig(self._training_time_png_path)
        #     plt.close(fig)

    @property
    def _total_csv_path(self):
        return _j(self.directory, "corrected_training_time.total.raw.csv")

    @property
    def _per_api_csv_path(self):
        return _j(self.directory, "corrected_training_time.per_api.raw.csv")

    @property
    def _total_json_path(self):
        return _j(self.directory, "corrected_training_time.total.json")

    @property
    def _per_api_json_path(self):
        return _j(self.directory, "corrected_training_time.per_api.json")

    @property
    def _total_png_path(self):
        return _j(self.directory, "corrected_training_time.total.pdf")

    @property
    def _per_api_png_path(self):
        return _j(self.directory, "corrected_training_time.per_api.pdf")

    @property
    def _training_time_png_path(self):
        return _j(self.directory, "corrected_training_time.training_time.pdf")

    @property
    def _training_time_csv_path(self):
        return _j(self.directory, "corrected_training_time.training_time.csv")

    @property
    def _overhead_correction_csv_path(self):
        return _j(self.directory, "corrected_training_time.overhead_correction.csv")

    def _overhead_correction_png_path(self, plot_group):
        suffix = plot_group.lower()
        suffix = re.sub(r'\s+', '_', suffix)
        return _j(self.directory, "corrected_training_time.overhead_correction.{suffix}.pdf".format(
            suffix=suffix,
        ))

class CallInterceptionOverheadParser:
    """
    config_interception
    Run with interception enabled.
    $ rls-prof --config interception python train.py --rlscope-disable-pyprof
    # --config interception: --cuda-api-calls --cuda-api-events
    # We want to know how much time is spent just intercepting API calls (NOT GPU activities)

    config_uninstrumented
    Run with interception disabled, and pyprof disabled (uninstrumented).
    $ rls-prof --config uninstrumented python train.py --rlscope-disable-pyprof
    # Time spent without ANY interception / profiling overhead at all.

    config_pyprof
    Run with pyprof enabled (but interception disabled).
    $ rls-prof uninstrumented ... python train.py
    # Time spent without ANY interception / profiling overhead at all.
    """
    def __init__(self,
                 interception_directory,
                 uninstrumented_directory,
                 directory,
                 # ignore_phase=False,
                 # algo_env_from_dir=False,
                 # baseline_config=None,
                 width=None,
                 height=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.interception_directory = interception_directory
        self.uninstrumented_directory = uninstrumented_directory
        self.directory = directory
        self.width = width
        self.height = height
        # self.rlscope_directories = rlscope_directories
        # self.ignore_phase = ignore_phase
        # self.algo_env_from_dir = algo_env_from_dir
        # self.baseline_config = baseline_config
        self.debug = debug
        self.debug_single_thread = debug_single_thread

        # self.added_fields = set()

    @property
    def _raw_csv_path(self):
        return _j(self.directory, "LD_PRELOAD_overhead.raw.csv")

    @property
    def _raw_json_path(self):
        return _j(self.directory, "LD_PRELOAD_overhead.json")

    @property
    def _agg_csv_path(self):
        return _j(self.directory, "LD_PRELOAD_overhead.agg.csv")

    @property
    def _png_path(self):
        return _j(self.directory, "LD_PRELOAD_overhead.png")

    def run(self):
        """
        Read total time that the program ran for from training_progress.

        Grab the "last" IncrementalTrainingProgress file that was written.
        duration_us = IncrementalTrainingProgress.end_training_time_us -
                      IncrementalTrainingProgress.start_training_time_us

        Calculate the number of TOTAL intercepted API call of ANY type
        (i.e. we assume the cost for intercepting an API call is the same)
        n_total_calls = sum( CUDAAPIThreadStatsProto.num_calls across all
                             CUDAAPIThreadStatsProto files )

        Create double-bar graph:
            left bar: no-interception (uninstrumented)
            right bar: interception (instrumented)

        Stats to save:
            # NOTE: this assumes the same number of repetitions for each.
            API interception overhead per call (us) =
                mean(
                    [time-per-call for interception] - [time-per-call for no-interception]
                )
                also compute std.

            # NOTE: we don't know these
            API interception time per call (us) =
                mean([time-per-call for interception])
                also compute std.
            API no-interception time per call (us) =
                ...

        :return:
        """

        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        int_training_duration_us = get_training_durations(self.interception_directory, debug=self.debug, debug_single_thread=self.debug_single_thread)
        no_int_training_duration_us = get_training_durations(self.uninstrumented_directory, debug=self.debug, debug_single_thread=self.debug_single_thread)
        if len(int_training_duration_us) != len(no_int_training_duration_us):
            raise RuntimeError("You need to run the same number of repetitions for both config_interception and config_uninstrumented")

        int_total_calls = get_n_total_calls(self.interception_directory, debug=self.debug, debug_single_thread=self.debug_single_thread)
        # no_int_total_calls = get_n_total_calls(self.uninstrumented_directory, debug=self.debug)
        # 'no_int_total_calls': no_int_total_calls,

        df = pd.DataFrame({
            'int_training_duration_us': int_training_duration_us,
            'no_int_training_duration_us': no_int_training_duration_us,
            'int_total_calls': int_total_calls,
        })
        # 3 bars to show:
        # df['int_per_call_us'] = df['int_training_duration_us'] / df['int_total_calls']
        # df['no_int_per_call_us'] = df['no_int_training_duration_us'] / df['int_total_calls']
        df['interception_overhead_per_call_us'] = (df['int_training_duration_us'] - df['no_int_training_duration_us']) / df['int_total_calls']
        df.to_csv(self._raw_csv_path, index=False)
        data = {
            'mean_interception_overhead_per_call_us': np.mean(df['interception_overhead_per_call_us']),
            'std_interception_overhead_per_call_us': np.std(df['interception_overhead_per_call_us']),
            'num_interception_overhead_per_call_us': len(df),
        }
        do_dump_json(data, self._raw_json_path)
        check_calibration(data, 'mean_interception_overhead_per_call_us', self._raw_json_path)

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        plot_data = copy.copy(df)
        plot_data['field'] = "Per-API-call interception overhead"
        ax = fig.add_subplot(111)
        ax.set_ylabel('Time (us)')
        sns.barplot(x='field', y='interception_overhead_per_call_us', data=plot_data, ax=ax,
                    **sns_kwargs)
        ax.set_ylabel('Time per call (us)')
        ax.set_xlabel(None)
        ax.set_title("Overhead from intercepting CUDA API calls using LD_PRELOAD")
        fig.savefig(self._png_path)
        if is_pdf(self._png_path):
            pdf2png(self._png_path)
        plt.close(fig)

        logger.info("Output plot @ {path}".format(path=self._png_path))
        logger.info("Output csv @ {path}".format(path=self._raw_csv_path))
        logger.info("Output json @ {path}".format(path=self._raw_json_path))


class PyprofOverheadCalculation:
    def __init__(self, df, json, num_field,
                 mean_per_call_field=None,
                 std_per_call_field=None,
                 num_per_call_field=None):
        self.df = df
        self.json = json
        self.num_field = num_field
        self.mean_per_call_field = mean_per_call_field
        self.std_per_call_field = std_per_call_field
        self.num_per_call_field = num_per_call_field

        def num_calls(self):
            return self.df[self.num_field]

class MicrobenchmarkOverheadJSON:
    def __init__(self, path):
        self.path = path
        self._load_json()

    def _load_json(self):
        js = load_json(self.path)
        for k in list(js.keys()):
            if type(js[k]) == list:
                js[k] = np.array(js[k])
        self.js = js

    def dump(self):
        js = dict()
        for key, value in self.js.items():
            if isinstance(value, np.ndarray) or isinstance(value, pd.Series):
                js[key] = list(value)
            else:
                js[key] = value

        do_dump_json(js, self.path)

    def __getitem__(self, item):
        return self.js[item]

    def __setitem__(self, key, value):
        self.js[key] = value

    def __contains__(self, item):
        return item in self.js

    def __len__(self, item):
        return len(self.js)

def parse_microbench_overhead_js(field_name, uninstrumented_json_path, instrumented_json_path):
    """
    Summarized format:
    {
        "mean_pyprof_annotation_overhead_per_call_us": 23.5408,
        "num_pyprof_annotation_overhead_per_call_us": 5,
        "std_pyprof_annotation_overhead_per_call_us": 0.5474806261412359
    }
    Raw iteration format:
    {
        "pyprof_annotation_overhead_per_call_us": [23, 21, 22, ...],
        "num_pyprof_annotation_overhead_per_call_us": 5,
    }
    """
    # mean_overhead_us = ins['time_sec_per_iteration']*constants.USEC_IN_SEC - unins['time_sec_per_iteration']*constants.USEC_IN_SEC
    uninstrumented_json = MicrobenchmarkOverheadJSON(uninstrumented_json_path)
    instrumented_json = MicrobenchmarkOverheadJSON(instrumented_json_path)

    # instrumented_json['overhead_per_call_us'] -
    # uninstrumented_json['overhead_per_call_us']
    overhead_per_call_us = \
        ( instrumented_json['iterations_total_sec'] - uninstrumented_json['iterations_total_sec'] )*constants.USEC_IN_SEC / \
        instrumented_json['iterations_total_num_calls']
    num_field = "num_{field}".format(field=field_name)
    mean_field = "mean_{field}".format(field=field_name)
    std_field = "std_{field}".format(field=field_name)
    js = {
        field_name: list(overhead_per_call_us),
        num_field: len(overhead_per_call_us),
        mean_field: np.mean(overhead_per_call_us),
        std_field: np.std(overhead_per_call_us),
    }
    return js

class PyprofOverheadParser:
    """
    Compute extra time spent in the CUDA API as a result of turning on CUPTI's GPU activity recording feature.

    TODO: We should have

    # Run with tfprof disabled, pyprof disabled, AND op-events disabled;
    # NOTE: we should make --rlscope-disable do THIS by default.
    uninstrumented_directory
    $ rls-prof train.py --rlscope-disable --rlscope-disable-ops

    # Run with ONLY pyprof events enabled (nothing else).
    # i.e. intercept C++ methods and record Python/C++ events.
    pyprof_interceptions_directory
    $ rls-prof train.py --rlscope-disable-tfprof --rlscope-disable-ops

    # Run with ONLY op-events enabled (nothing else).
    # i.e. only rlscope.prof.operation(...) calls are added code.
    pyprof_annotations_directory

    """
    def __init__(self,
                 uninstrumented_directory,
                 pyprof_annotations_directory=None,
                 pyprof_interceptions_directory=None,
                 directory=None,
                 width=None,
                 height=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.uninstrumented_directory = uninstrumented_directory
        self.pyprof_annotations_directory = pyprof_annotations_directory
        self.pyprof_interceptions_directory = pyprof_interceptions_directory
        assert directory is not None
        self.directory = directory
        self.width = width
        self.height = height
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.filename_prefix = 'category_events'

        self._training_duration_colnames = ['training_duration_us', 'training_duration_us_unins']


    @staticmethod
    def get_total_intercepted_calls(reader):
        return reader.total_intercepted_calls()

    @staticmethod
    def get_total_annotations(reader):
        return reader.total_annotations()

    @staticmethod
    def get_overhead_field(reader):
        return reader.total_op_events()

    @staticmethod
    def compute_overhead(config,
                         uninstrumented_directory,
                         instrumented_directory,
                         name, get_num_field,
                         mapper_cb=None,
                         debug=False,
                         debug_single_thread=False,
                         ):
        """
        Given an instrumented_directory and uninstrumented_directory, join row-by-row,
        and compute the delta in training_duration_us to get "total profiling overhead".
        Then, use get_num_field to get # of "calls", and divide by this to get "per call profiling overhead".
        """

        # assert name in ['op', 'event']

        unins_df = pd.concat(get_training_durations_df(uninstrumented_directory, debug=debug, debug_single_thread=debug_single_thread))
        unins_df['config'] = 'uninstrumented'

        # e.g. num_<pyprof_interception>s
        num_field = "num_{name}s".format(name=name)
        overhead_field = overhead_colname(name)
        per_overhead_field = overhead_per_call_colname(name)

        ins_dfs = []
        for directory in instrumented_directory:
            training_duration_us = get_training_durations(directory, debug=debug, debug_single_thread=debug_single_thread)

            pyprof_mapper = DataframeMapper(PyprofDataframeReader, directories=[directory], debug=debug)
            if mapper_cb:
                mapper_cb(pyprof_mapper)

            num_field_values = pyprof_mapper.map(get_num_field)
            assert len(num_field_values) == 1

            # overhead_field_values = pyprof_mapper.map(get_overhead_field)
            # assert len(overhead_field_values) == 1

            ins_df = pd.DataFrame({
                'training_duration_us': [training_duration_us],
                num_field: num_field_values,
                # overhead_field: overhead_field_values,
            })
            ins_df['config'] = config

            rlscope_config = read_rlscope_config(directory)
            add_rlscope_config(ins_df, rlscope_config)

            assert len(ins_df) == 1
            ins_dfs.append(ins_df)

        ins_df = pd.concat(ins_dfs)

        unins_join_df = unins_df['training_duration_us']
        check_cols_eq(ins_df, unins_df, colnames=['algo', 'env'])
        # TODO: ideally we would join row-by-row on groups with equal (algo, env) to support
        # multiple different algo/env combinations.
        df = join_row_by_row(
            ins_df, unins_df[['training_duration_us']],
            suffixes=("", "_unins"))
        add_x_field(df)
        df[overhead_field] = df['training_duration_us'] - df['training_duration_us_unins']
        df[per_overhead_field] = df[overhead_field] / df[num_field]
        per_overheads = []
        for overhead, num_events in zip(df[overhead_field], df[num_field]):
            if num_events == 0:
                continue
            per_overheads.append(overhead/num_events)

        json = dict()

        if len(per_overheads) == 0:
            logger.warning(textwrap.dedent("""
            Saw 0 overhead events for {config} when processing:
              instrumented_directory   = {ins}
              uninstrumented_directory = {unins}
            This could mean one of two things:
            1. You aren't wrapping the simulator / DL library properly.
            2. You're measuring something that doesn't make simulator / DL library calls.
            We simply won't correct for {config} in this case.
            """).format(
                config=config,
                ins=instrumented_directory,
                unins=uninstrumented_directory,
            ))
            json[mean_per_call_colname(name)] = 0.
            json[std_per_call_colname(name)] = 0.
            json[num_per_call_colname(name)] = 0.
        else:
            json[mean_per_call_colname(name)] = np.mean(per_overheads)
            json[std_per_call_colname(name)] = np.std(per_overheads)
            json[num_per_call_colname(name)] = len(per_overheads)

        # mean_pyprof_interception_overhead_per_call_us
        # mean_pyprof_annotation_overhead_per_call_us

        calc = PyprofOverheadCalculation(
            df, json,
            num_field=num_field,
            mean_per_call_field=mean_per_call_colname(name),
            std_per_call_field=std_per_call_colname(name),
            num_per_call_field=num_per_call_colname(name),
        )
        return calc

    @staticmethod
    def check_no_interceptions(pyprof_mapper):
        total_intercepted_calls = pyprof_mapper.map_one(PyprofOverheadParser.get_total_intercepted_calls)
        assert total_intercepted_calls == 0

    @staticmethod
    def check_no_annotations(pyprof_mapper):
        total_annotations = pyprof_mapper.map_one(PyprofOverheadParser.get_total_annotations)
        assert total_annotations == 0

    @staticmethod
    def compute_interception_overhead(
        uninstrumented_directory,
        instrumented_directory,
        debug=False,
        debug_single_thread=False):
        pyprof_interceptions_calc = PyprofOverheadParser.compute_overhead(
            'pyprof_interceptions',
            uninstrumented_directory,
            instrumented_directory,
            'pyprof_interception',
            PyprofOverheadParser.get_total_intercepted_calls,
            mapper_cb=PyprofOverheadParser.check_no_annotations,
            debug=debug,
            debug_single_thread=debug_single_thread,
        )
        return pyprof_interceptions_calc

    def run_interceptions(self):
        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        pyprof_interceptions_calc = PyprofOverheadParser.compute_interception_overhead(
            self.uninstrumented_directory,
            self.pyprof_interceptions_directory,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )

        pyprof_interceptions_df = dataframe_replace_us_with_sec(pyprof_interceptions_calc.df, colnames=self._training_duration_colnames)
        output_csv(pyprof_interceptions_df, self._pyprof_interception_csv_path, sort_by=['pyprof_interception_overhead_us'])

        fig, ax = self._plot(x='x_field', y=overhead_per_call_colname('pyprof_interception'), data=pyprof_interceptions_df, sns_kwargs=sns_kwargs)
        ax.set_title(r'Python $\rightarrow$ C-library interception overhead')
        ax.set_ylabel('Time per interception (us)')
        ax.set_xlabel('(algo, env)')
        save_plot(fig, ax, png_path=self._pyprof_interception_png_path)

        return pyprof_interceptions_calc

    @staticmethod
    def compute_annotation_overhead(
        uninstrumented_directory,
        instrumented_directory,
        debug=False,
        debug_single_thread=False):
        pyprof_annotations_calc = PyprofOverheadParser.compute_overhead(
            'pyprof_annotations',
            uninstrumented_directory,
            instrumented_directory,
            'pyprof_annotation',
            PyprofOverheadParser.get_total_annotations,
            mapper_cb=PyprofOverheadParser.check_no_interceptions,
            debug=debug,
            debug_single_thread=debug_single_thread,
        )
        return pyprof_annotations_calc

    def run_annotations(self):
        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        pyprof_annotations_calc = PyprofOverheadParser.compute_annotation_overhead(
            self.uninstrumented_directory,
            self.pyprof_annotations_directory,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread,
        )

        pyprof_annotations_df = dataframe_replace_us_with_sec(pyprof_annotations_calc.df, colnames=self._training_duration_colnames)
        output_csv(pyprof_annotations_df, self._pyprof_annotation_csv_path, sort_by=['pyprof_annotation_overhead_us'])

        fig, ax = self._plot(x='x_field', y=overhead_per_call_colname('pyprof_annotation'), data=pyprof_annotations_df, sns_kwargs=sns_kwargs)
        ax.set_title(r'Python annotation overhead')
        ax.set_ylabel('Time per annotation (us)')
        ax.set_xlabel('(algo, env)')
        save_plot(fig, ax, png_path=self._pyprof_annotation_png_path)

        return pyprof_annotations_calc

    def run(self):
        """
        PSEUDOCODE:
        unins_df = Read data-frame from uninstrumented data:
            config            training_duration_us
            uninstrumented    ...
        ins_df = Read data-frame from instrumented (e.g. pyprof_annotations_directory) data:
            config          training_duration_us    num_ops
            pyprof_annotations      ...                     ...

        Assert len(unins_df) == len(ins_df)
        df = Join row-by-row:
                unins_df['training_duration_us' as
                         'uninstrumented_training_duration_us']
            with
                and ins_df

        df['<op>_overhead_us'] = df['uninstrumented_training_duration_us'] - df['training_duration_us']
        df['per_<op>_overhead_us'] = df['<op>_overhead_us'] / df['num_<op>s']

        Add to json:
            json['(mean/std/len)_<op>_overhead_us'] = np.mean/std/len(df['<op>_overhead_us'])

        Stats to save:
            # We need this for "subtraction"
            Per-op profiling overhead (us) =
            {
                mean/std/len: ...,
            }

            Per-event profiling overhead (us) =
            {
                mean/std/len: ...,
            }

        csv output:
        # pyprof_annotation_overhead.csv
        config, training_duration_us, uninstrumented_training_duration_us, num_ops, per_pyprof_annotation_overhead_us

        ...
        # pyprof_interception_overhead.csv
        config, training_duration_us, uninstrumented_training_duration_us, num_events, per_pyprof_interception_overhead_us

        json output:
        # pyprof_overhead.json
        api_name: {
            # pyprof_annotation_overhead.json
            (mean/std/len)_pyprof_annotation_overhead_us:
            # pyprof_interception_overhead.json
            (mean/std/len)_pyprof_interception_overhead_us:
        }
        """

        pyprof_annotations_calc = None
        pyprof_interceptions_calc = None

        def has_files(opt):
            return opt is not None

        jsons = []
        if has_files(self.pyprof_annotations_directory):
            pyprof_annotations_calc = self.run_annotations()
            jsons.append(pyprof_annotations_calc.json)

            path = self._python_annotation_json_path
            logger.info("Output json @ {path}".format(path=path))
            do_dump_json(pyprof_annotations_calc.json, path)
            check_calibration(pyprof_annotations_calc.json, pyprof_annotations_calc.mean_per_call_field, path)

        if has_files(self.pyprof_interceptions_directory):
            pyprof_interceptions_calc = self.run_interceptions()
            jsons.append(pyprof_interceptions_calc.json)

            path = self._clib_interception_json_path
            check_calibration(pyprof_interceptions_calc.json, pyprof_interceptions_calc.mean_per_call_field, path)
            logger.info("Output json @ {path}".format(path=path))
            do_dump_json(pyprof_interceptions_calc.json, path)

        if pyprof_interceptions_calc is not None and pyprof_interceptions_calc is not None:
            json = merge_jsons(jsons)
            logger.info("Output json @ {path}".format(path=self._json_path))
            do_dump_json(json, self._json_path)

    def _plot(self, x='x_field', y=None, data=None, sns_kwargs=dict()):
        assert y is not None
        assert data is not None
        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)
        sns.barplot(x=x, y=y, data=data, ax=ax,
                    **sns_kwargs)
        ax.legend().set_title(None)
        return fig, ax

    @property
    def _pyprof_annotation_csv_path(self):
        return _j(self.directory, "{prefix}.pyprof_annotation.raw.csv".format(
            prefix=self.filename_prefix))

    @property
    def _pyprof_interception_csv_path(self):
        return _j(self.directory, "{prefix}.pyprof_interception.raw.csv".format(
            prefix=self.filename_prefix))

    @property
    def _pyprof_annotation_png_path(self):
        return _j(self.directory, "{prefix}.pyprof_annotation.png".format(
            prefix=self.filename_prefix))

    @property
    def _pyprof_interception_png_path(self):
        return _j(self.directory, "{prefix}.pyprof_interception.png".format(
            prefix=self.filename_prefix))

    @property
    def _json_path(self):
        return _j(self.directory, "{prefix}.json".format(
            prefix=self.filename_prefix))

    @property
    def _clib_interception_json_path(self):
        return _j(self.directory, "{prefix}.python_clib_interception.json".format(
            prefix=self.filename_prefix))

    @property
    def _python_annotation_json_path(self):
        return _j(self.directory, "{prefix}.python_annotation.json".format(
            prefix=self.filename_prefix))

class TotalTrainingTimeParser:
    """
    Plot total training time of uninstrumented run.

    # Run with tfprof disabled, pyprof disabled, AND op-events disabled;
    # NOTE: we should make --rlscope-disable do THIS by default.
    uninstrumented_directory
    $ rls-prof train.py --rlscope-disable --rlscope-disable-ops

    # Run with ONLY pyprof events enabled (nothing else).
    # i.e. intercept C++ methods and record Python/C++ events.
    pyprof_interceptions_directory
    $ rls-prof train.py --rlscope-disable-tfprof --rlscope-disable-ops

    # Run with ONLY op-events enabled (nothing else).
    # i.e. only rlscope.prof.operation(...) calls are added code.
    pyprof_annotations_directory

    """
    def __init__(self,
                 uninstrumented_directory,
                 directory,
                 width=None,
                 height=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.uninstrumented_directory = uninstrumented_directory
        self.directory = directory
        self.width = width
        self.height = height
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.filename_prefix = 'total_training_time'

    # @staticmethod
    # def get_total_intercepted_calls(reader):
    #     return reader.total_intercepted_calls()

    @staticmethod
    def get_total_annotations(reader):
        return reader.total_annotations()

    @staticmethod
    def get_overhead_field(reader):
        return reader.total_op_events()

    def compute_overhead(self, config, instrumented_directory,
                         name, get_num_field,
                         mapper_cb=None):
        """
        Given an instrumented_directory and uninstrumented_directory, join row-by-row,
        and compute the delta in training_duration_us to get "total profiling overhead".
        Then, use get_num_field to get # of "calls", and divide by this to get "per call profiling overhead".
        """

        # assert name in ['op', 'event']

        unins_df = pd.concat(get_training_durations_df(self.uninstrumented_directory, debug=self.debug, debug_single_thread=self.debug_single_thread))
        unins_df['config'] = 'uninstrumented'

        # e.g. num_<pyprof_interception>s
        num_field = "num_{name}s".format(name=name)
        overhead_field = overhead_colname(name)
        per_overhead_field = overhead_per_call_colname(name)

        ins_dfs = []
        for directory in instrumented_directory:
            training_duration_us = get_training_durations(directory, debug=self.debug, debug_single_thread=self.debug_single_thread)

            pyprof_mapper = DataframeMapper(PyprofDataframeReader, directories=[directory], debug=self.debug)
            if mapper_cb:
                mapper_cb(pyprof_mapper)

            num_field_values = pyprof_mapper.map(get_num_field)
            assert len(num_field_values) == 1

            # overhead_field_values = pyprof_mapper.map(get_overhead_field)
            # assert len(overhead_field_values) == 1

            ins_df = pd.DataFrame({
                'training_duration_us': [training_duration_us],
                num_field: num_field_values,
                # overhead_field: overhead_field_values,
            })
            ins_df['config'] = config

            rlscope_config = read_rlscope_config(directory)
            add_rlscope_config(ins_df, rlscope_config)

            assert len(ins_df) == 1
            ins_dfs.append(ins_df)

        ins_df = pd.concat(ins_dfs)

        unins_join_df = unins_df['training_duration_us']
        check_cols_eq(ins_df, unins_df, colnames=['algo', 'env'])
        # TODO: ideally we would join row-by-row on groups with equal (algo, env) to support
        # multiple different algo/env combinations.
        df = join_row_by_row(
            ins_df, unins_df[['training_duration_us']],
            suffixes=("", "_unins"))
        add_x_field(df)
        df[overhead_field] = df['training_duration_us'] - df['training_duration_us_unins']
        df[per_overhead_field] = df[overhead_field] / df[num_field]

        json = dict()
        json[mean_per_call_colname(name)] = np.mean(df[per_overhead_field])
        json[std_per_call_colname(name)] = np.std(df[per_overhead_field])
        json[num_per_call_colname(name)] = len(df[per_overhead_field])

        # mean_pyprof_interception_overhead_per_call_us
        # mean_pyprof_annotation_overhead_per_call_us

        calc = PyprofOverheadCalculation(
            df, json,
            num_field=num_field,
            mean_per_call_field=mean_per_call_colname(name),
            std_per_call_field=std_per_call_colname(name),
            num_per_call_field=num_per_call_colname(name),
        )
        return calc

    def run(self):
        """
        PSEUDOCODE:
        unins_df = Read data-frame from uninstrumented data:
            config            training_duration_us
            uninstrumented    ...
        ins_df = Read data-frame from instrumented (e.g. pyprof_annotations_directory) data:
            config          training_duration_us    num_ops
            pyprof_annotations      ...                     ...

        Assert len(unins_df) == len(ins_df)
        df = Join row-by-row:
                unins_df['training_duration_us' as
                         'uninstrumented_training_duration_us']
            with
                and ins_df

        df['<op>_overhead_us'] = df['uninstrumented_training_duration_us'] - df['training_duration_us']
        df['per_<op>_overhead_us'] = df['<op>_overhead_us'] / df['num_<op>s']

        Add to json:
            json['(mean/std/len)_<op>_overhead_us'] = np.mean/std/len(df['<op>_overhead_us'])

        Stats to save:
            # We need this for "subtraction"
            Per-op profiling overhead (us) =
            {
                mean/std/len: ...,
            }

            Per-event profiling overhead (us) =
            {
                mean/std/len: ...,
            }

        csv output:
        # pyprof_annotation_overhead.csv
        config, training_duration_us, uninstrumented_training_duration_us, num_ops, per_pyprof_annotation_overhead_us

        ...
        # pyprof_interception_overhead.csv
        config, training_duration_us, uninstrumented_training_duration_us, num_events, per_pyprof_interception_overhead_us

        json output:
        # pyprof_overhead.json
        api_name: {
            # pyprof_annotation_overhead.json
            (mean/std/len)_pyprof_annotation_overhead_us:
            # pyprof_interception_overhead.json
            (mean/std/len)_pyprof_interception_overhead_us:
        }
        """

        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        dfs = []
        for directory in self.uninstrumented_directory:
            training_mapper = DataframeMapper(TrainingProgressDataframeReader, directories=[directory], debug=self.debug)
            df = training_mapper.map_one(lambda reader: reader.training_duration_df())
            assert len(df) == 1
            dfs.append(df)
        # algo, env, training_duration_us
        df = pd.concat(dfs)
        # algo, env, x_field, training_duration_us
        add_x_field(df)
        # algo, env, x_field, training_duration_sec
        dataframe_replace_us_with_sec(df)
        # algo, env, short_env, x_field, training_duration_sec
        add_short_env(df)

        def _plot():
            if self.width is not None and self.height is not None:
                figsize = (self.width, self.height)
                logger.info("Setting figsize = {fig}".format(fig=figsize))
                # sns.set_context({"figure.figsize": figsize})
            else:
                figsize = None
            # This is causing XIO error....
            # fig = plt.figure(figsize=figsize)

            # per_iteration_df = read_per_iteration_df(df)

            plot_data = copy.copy(df)
            # plot_data['field'] = "Per-API-call interception overhead"
            # ax = fig.add_subplot(111)

            sns.set_style('whitegrid')

            # We can create a "grouped" plot with seaborn using FacetGrid where we split on the "group name";
            # e.g. group name can be
            # Algorithm: ppo2, sac, ddpg, a2c

            g = sns.FacetGrid(data=plot_data, col="algo",
                              palette=sns.color_palette('muted'))
            with ignore_seaborn_warnings():
                g.map(sns.barplot, 'short_env', 'training_duration_sec', **sns_kwargs)
            g.add_legend()
            g.set_ylabels('Total training time (sec)')
            g.set_xlabels('Environment')
            # g.set_titles("CUDA API time with increased training loop iterations")
            g.fig.subplots_adjust(top=0.8)
            g.fig.suptitle(r"Total training time of all $(algo, env)$ workloads")
            for ax in g.axes.ravel():
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

            # TODO: I have no idea how to create a shared x-label when using FacetGrid :(
            #
            # g.fig.subplots_adjust(bottom=0.2)
            # left, width = .25, .5
            # bottom, height = .25, .5

            # left, width = 0., 1.
            # bottom, height = 0., 1.
            # right = left + width
            # top = bottom + height
            # ax = g.fig.add_axes([0, 0, 1, 1])
            # ax.text(left, bottom, 'center bottom',
            #         horizontalalignment='center',
            #         verticalalignment='bottom',
            #         transform=ax.transAxes)
            # g.fig.text()

            logger.info("Output plot @ {path}".format(path=self._png_path))
            g.savefig(self._png_path)
            if is_pdf(self._png_path):
                pdf2png(self._png_path)

        output_csv(df, self._csv_path, sort_by=['algo', 'env', 'training_duration_sec'])
        _plot()

    @property
    def _csv_path(self):
        return _j(self.directory, "{prefix}.raw.csv".format(
            prefix=self.filename_prefix))

    @property
    def _png_path(self):
        return _j(self.directory, "{prefix}.png".format(
            prefix=self.filename_prefix))

class CUPTIScalingOverheadCalculation:
    def __init__(self, df, json):
        self.df = df
        self.json = json

class CUPTIScalingOverheadParser:
    """
    See how CUPTI per-api-call time varies as we scale the number of traced training-loop iterations.
    """
    def __init__(self,
                 gpu_activities_api_time_directory,
                 interception_directory,
                 directory,
                 width=None,
                 height=None,
                 debug=False,
                 debug_single_thread=False,
                 debug_memoize=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.gpu_activities_api_time_directory = gpu_activities_api_time_directory
        self.interception_directory = interception_directory
        self.directory = directory
        self.width = width
        self.height = height
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.debug_memoize = debug_memoize
        self.filename_prefix = 'cupti_scaling_overhead'

    def colname(self, col, config):
        return "{col}_{config}".format(
            col=col,
            config=re.sub('-', '_', config))

    def load_df(self):
        def read_df():
            memoize_path = _j(self.directory, "{klass}.read_df.pickle".format(
                klass=self.__class__.__name__))
            if should_load_memo(self.debug_memoize, memoize_path):
                ret = load_memo(self.debug_memoize, memoize_path)
                return ret
            df = cupti_read_cuda_api_stats(config_directories_pairs=[
                ('interception', self.interception_directory),
                ('gpu-activities-api-time', self.gpu_activities_api_time_directory),
            ], debug=self.debug,
                debug_single_thread=self.debug_single_thread)
            ret = df
            maybe_memoize(self.debug_memoize, ret, memoize_path)
            return ret

        df = read_df()

        dataframe_replace_us_with_sec(df, colnames=['per_iteration_us'])

        df_csv = df.sort_values(['config', 'api_name', 'us_per_call'])
        df_csv.to_csv(self._raw_csv_path, index=False)

        joined_df = cupti_join_config_rows(df)
        joined_df.to_csv(self._raw_pairs_csv_path, index=False)
        joined_df['cupti_overhead_per_call_us'] = joined_df[self.colname('us_per_call', 'gpu-activities-api-time')] - joined_df[self.colname('us_per_call', 'interception')]
        json_data = dict()
        for api_name, df_api_name in joined_df.groupby('api_name'):
            assert api_name not in json_data
            json_data[api_name] = dict()
            json_data[api_name]['mean_cupti_overhead_per_call_us'] = np.mean(df_api_name['cupti_overhead_per_call_us'])
            assert not np.isnan(json_data[api_name]['mean_cupti_overhead_per_call_us'])
            json_data[api_name]['std_cupti_overhead_per_call_us'] = np.std(df_api_name['cupti_overhead_per_call_us'])
            assert not np.isnan(json_data[api_name]['std_cupti_overhead_per_call_us'])
            json_data[api_name]['num_cupti_overhead_per_call_us'] = len(df_api_name['cupti_overhead_per_call_us'])
            assert not np.isnan(json_data[api_name]['num_cupti_overhead_per_call_us'])

        def pretty_config(config):
            if config == 'gpu-activities-api-time':
                return 'CUPTI enabled'
            elif config == 'interception':
                return 'CUPTI disabled'
            else:
                raise NotImplementedError()
        df['pretty_config'] = df['config'].apply(pretty_config)

        logger.info("Output csv @ {path}".format(path=self._raw_csv_path))
        df.to_csv(self._raw_csv_path, index=False)

        logger.info("Output json @ {path}".format(path=self._raw_json_path))
        do_dump_json(json_data, self._raw_json_path)
        for api_name, df_api_name in joined_df.groupby('api_name'):
            check_calibration(json_data[api_name], 'mean_cupti_overhead_per_call_us', self._raw_json_path)

        return df, joined_df

    def run(self):
        sns_kwargs = get_sns_kwargs()
        # sns_kwargs['xticks.rotation'] = 45
        plt_kwargs = get_plt_kwargs()

        # TODO: we want to read files for EACH number of iterations,
        # need to add a "training_iterations" column.

        df, joined_df = self.load_df()

        def read_per_iteration_df(df):
            groupby_cols = ['config', 'api_name', 'algo', 'env']
            keep_cols = ['training_duration_us', 'training_iterations', 'per_iteration_sec', 'pretty_config']
            per_iteration_dfs = []
            configs = list(set((df['config'])))
            for config in configs:
                groupby = list(df[df['config'] == config].groupby(groupby_cols))
                group = groupby[0][0]
                group_df = groupby[0][1]
                group_df = group_df[groupby_cols + keep_cols]
                per_iteration_dfs.append(group_df)
            per_iteration_df = pd.concat(per_iteration_dfs)
            add_x_field(per_iteration_df)
            return per_iteration_df

        def _plot():
            if self.width is not None and self.height is not None:
                figsize = (self.width, self.height)
                logger.info("Setting figsize = {fig}".format(fig=figsize))
                # sns.set_context({"figure.figsize": figsize})
            else:
                figsize = None
            # This is causing XIO error....
            # fig = plt.figure(figsize=figsize)

            per_iteration_df = read_per_iteration_df(df)

            plot_data = copy.copy(df)
            # plot_data['field'] = "Per-API-call interception overhead"
            # ax = fig.add_subplot(111)

            sns.set_style('whitegrid')


            g = sns.FacetGrid(data=plot_data, col="training_iterations",
                              palette=sns.color_palette('muted'))
            with ignore_seaborn_warnings():
                g.map(sns.barplot, 'api_name', 'us_per_call', 'pretty_config', **sns_kwargs)
            g.add_legend()
            g.set_ylabels('Time per call (us)')
            g.set_xlabels('CUDA API call')
            # g.set_titles("CUDA API time with increased training loop iterations")
            g.fig.subplots_adjust(top=0.8)
            g.fig.suptitle("CUDA API time with increased training loop iterations")
            for ax in g.axes.ravel():
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            logger.info("Output plot @ {path}".format(path=self._png_path))
            g.savefig(self._png_path)
            if is_pdf(self._png_path):
                pdf2png(self._png_path)

            g = sns.FacetGrid(data=per_iteration_df, col="training_iterations",
                              palette=sns.color_palette('muted'))
            with ignore_seaborn_warnings():
                g.map(sns.barplot, 'x_field', 'per_iteration_sec', 'pretty_config', **sns_kwargs)
            g.add_legend()
            g.set_ylabels('Time per iteration (us)')
            g.set_xlabels('(algo, env)')
            # g.set_titles("Training loop iteration time with increased training loop iterations")
            g.fig.subplots_adjust(top=0.8)
            g.fig.suptitle("Training loop iteration time with increased training loop iterations")
            for ax in g.axes.ravel():
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            logger.info("Output plot @ {path}".format(path=self._training_loop_png_path))
            g.savefig(self._training_loop_png_path)
            if is_pdf(self._training_loop_png_path):
                pdf2png(self._training_loop_png_path)

        _plot()

    @property
    def _json_path(self):
        return _j(self.directory, "{prefix}.json".format(
            prefix=self.filename_prefix))

    @property
    def _raw_csv_path(self):
        return _j(self.directory, "{prefix}.raw.csv".format(
            prefix=self.filename_prefix,
        ))

    @property
    def _raw_pairs_csv_path(self):
        return _j(self.directory, "{prefix}.pairs.raw.csv".format(
            prefix=self.filename_prefix,
        ))

    @property
    def _raw_json_path(self):
        return _j(self.directory, "{prefix}.json".format(
            prefix=self.filename_prefix,
        ))

    @property
    def _png_path(self):
        return _j(self.directory, "{prefix}.png".format(
            prefix=self.filename_prefix,
        ))

    @property
    def _training_loop_png_path(self):
        return _j(self.directory, "{prefix}.training_loop.png".format(
            prefix=self.filename_prefix,
        ))



def cupti_read_cuda_api_stats(config_directories_pairs,
                              debug=False,
                              debug_single_thread=False):
    """
    Read CUDAAPIPhaseStatsProto.CUDAAPIThreadStatsProto data into a dataframe.
    NOTE: CUDAAPIThreadStatsProto contains accumulated CUDA API time and number of calls,
    NOT individual CUDA API call time/duration.

    Outputs data like this:
      algo, env, config, api_name, total_num_calls, total_api_time_us, us_per_call

    :param config_directories_pairs:
            [('config_name', [rlscope-directories])]
    :return:
    """
    csv_data = dict()

    # Read (config, directories) into dataframe:
    #   algo, env, config, api_name, total_num_calls, total_api_time_us, us_per_call
    for (config, directories) in config_directories_pairs:
        for directory in directories:

            training_mapper = DataframeMapper(TrainingProgressDataframeReader, directories=[directory], debug=debug)
            training_iterations = training_mapper.map_one(lambda reader: reader.training_iterations())
            training_duration_us = training_mapper.map_one(lambda reader: reader.training_duration_us())

            per_api_stats = get_per_api_stats(directory,
                                              # debug=debug,
                                              debug=True,
                                              debug_single_thread=debug_single_thread)
            per_api_stats = per_api_stats.reset_index()
            logger.debug("directory = {directory}".format(directory=directory))
            logger.debug("per_api_stats: " + pprint_msg(per_api_stats))

            if len(per_api_stats) == 0:
                logger.error(
                    textwrap.dedent("""\
                    RL-Scope: the program you ran (--rlscope-directory={directory}) didn't contain any 
                    CUDA API calls, making it impossible to perform calibration.  
                    This can happen if TensorFlow cannot locate DNN shared libraries (e.g., libcudnn.so), 
                    and (silently) falls back to CPU-only execution.
                    If you are using TensorFlow, make sure it can use the GPU by querying available devices 
                    by calling tf.config.list_physical_devices()
                      See: https://www.tensorflow.org/api_docs/python/tf/config/list_physical_devices
                    """))
                sys.exit(1)

            for i, row in per_api_stats.iterrows():
                total_api_time_us = row['total_time_us']
                total_num_calls = row['num_calls']

                add_col(csv_data, 'config', config)
                add_col(csv_data, 'api_name', row['api_name'])
                add_col(csv_data, 'algo', row['algo'])
                add_col(csv_data, 'env', row['env'])
                add_col(csv_data, 'total_num_calls', total_num_calls)
                add_col(csv_data, 'total_api_time_us', total_api_time_us)
                assert total_num_calls > 0
                us_per_call = total_api_time_us / float(total_num_calls)
                assert not np.isnan(us_per_call)
                add_col(csv_data, 'us_per_call', us_per_call)
                add_col(csv_data, 'training_iterations', training_iterations)
                add_col(csv_data, 'training_duration_us', training_duration_us)

    df = pd.DataFrame(csv_data)
    df['per_iteration_us'] = df['training_duration_us'] / df['training_iterations']
    return df

def cupti_join_config_rows(df, debug=False):
    """
    Given (config, directories) data that looks like:

    join_index
                  algo, env, config, api_name, total_num_calls, total_api_time_us, us_per_call
             1               conf1       api1
             2               conf1       api2
             3               conf1       api1
             4               conf1       api2

             1               conf2       api1
             2               conf2       api2
             3               conf2       api1
             4               conf2       api2

    Group dataframe by 'config', then join each group row-by-row, renaming the columns so their suffix
    is the 'config' value.


    :param df:
    :param groupby_cols:
        Columns to group-by, in addition to 'config' and 'api_name'.
    :return:
    """
    def get_suffix(group_dict):
        config = group_dict['config']
        return "_{config}".format(config=re.sub('-', '_', config))

    groupby_cols = ['config', 'api_name']
    non_config_groupby_cols = list(set(groupby_cols).difference({'config'}))

    if debug:
        logger.debug("dataframe:\n{msg}".format(msg=textwrap.indent(DataFrame.dataframe_string(df), prefix='  ')))
        logger.debug(f"groupby_cols = {groupby_cols}")
        logger.debug(f"non_config_groupby_cols = {non_config_groupby_cols}")

    groupby = df.groupby(non_config_groupby_cols)
    dfs = []
    for group, group_df in groupby:
        config_df = join_groups_row_by_row(
            group_df,
            join_cols=['algo', 'env'],
            groupby_cols=groupby_cols,
            get_suffix=get_suffix)
        assert not config_df.isnull().values.any()
        dfs.append(config_df)
    new_df = pd.concat(dfs)
    # NOTE: If this fails, then it's because the dataframes in concat had different non-overlapping columns.
    # This could happen if there's only one "config" row in the groupby.
    assert not new_df.isnull().values.any()
    return new_df

class CUPTIOverheadParser:
    """
    Compute extra time spent in the CUDA API as a result of turning on CUPTI's GPU activity recording feature.
    """
    def __init__(self,
                 gpu_activities_directory,
                 no_gpu_activities_directory,
                 directory,
                 width=None,
                 height=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.gpu_activities_directory = gpu_activities_directory
        self.no_gpu_activities_directory = no_gpu_activities_directory
        self.directory = directory
        self.width = width
        self.height = height
        # self.rlscope_directories = rlscope_directories
        # self.ignore_phase = ignore_phase
        # self.algo_env_from_dir = algo_env_from_dir
        # self.baseline_config = baseline_config
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def colname(self, col, config):
        return "{col}_{config}".format(
            col=col,
            config=re.sub('-', '_', config))

    def run(self):
        """
        For config in 'gpu-activities', 'no-gpu-activities':
            For api_call in 'cudaKernelLaunch', …:
                Total_api_time_us = Sum up totall time spent in a specific API call (e.g. cudaLaunchKernel) across ALL calls (all threads)
                Total_api_calls = Sum up total number of times a specific API call (e.g. cudaLaunchKernel) was made
                Us_per_call = Total_api_time_us / Total_api_calls
                data[config][api_call].append(us_per_call)

        Stats to save:
            # We need this for "subtraction"
            Per-CUDA API CUPTI-induced profiling overhead (us) =
                cudaLaunchKernel: {
                    mean: ...,
                    std: ...,
                }
                ...

            # Useful stat for sanity check:
            # Is the total time spent in the CUDA API similar across repetitions of the same configuration?
            Total time spent in CUDA API call:
                config: {
                    cudaLaunchKernel: {
                        total_time_us: ...
                    }
                }
                ...

            # Useful stat for sanity check:
            # are number of CUDA API calls between runs the same/very similar?
            Total number of CUDA API calls:
                config: {
                    cudaLaunchKernel: {
                        n_calls: ...
                    }
                }
                ...

        csv output:
        api_name, config, total_num_calls, total_api_time_us, us_per_call
        ...

        json output:
        api_name: {
            mean_us_per_call:
            std_us_per_call:
        }
        """

        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        # api_name -> {
        #   mean_us_per_call: ...,
        #   std_us_per_call: ...,
        # }
        df = cupti_read_cuda_api_stats(config_directories_pairs=[
            ('gpu-activities', self.gpu_activities_directory),
            ('no-gpu-activities', self.no_gpu_activities_directory),
        ], debug=self.debug,
            debug_single_thread=self.debug_single_thread)

        df_csv = df.sort_values(['config', 'api_name', 'us_per_call'])
        df_csv.to_csv(self._raw_csv_path, index=False)

        joined_df = cupti_join_config_rows(df)
        joined_df.to_csv(self._raw_pairs_csv_path, index=False)
        joined_df['cupti_overhead_per_call_us'] = joined_df[self.colname('us_per_call', 'gpu-activities')] - joined_df[self.colname('us_per_call', 'no-gpu-activities')]
        json_data = dict()
        for api_name, df_api_name in joined_df.groupby('api_name'):
            assert api_name not in json_data
            json_data[api_name] = dict()
            json_data[api_name]['mean_cupti_overhead_per_call_us'] = np.mean(df_api_name['cupti_overhead_per_call_us'])
            check_calibration(json_data[api_name], 'mean_cupti_overhead_per_call_us', self._raw_json_path)
            assert not np.isnan(json_data[api_name]['mean_cupti_overhead_per_call_us'])
            json_data[api_name]['std_cupti_overhead_per_call_us'] = np.std(df_api_name['cupti_overhead_per_call_us'])
            assert not np.isnan(json_data[api_name]['std_cupti_overhead_per_call_us'])
            json_data[api_name]['num_cupti_overhead_per_call_us'] = len(df_api_name['cupti_overhead_per_call_us'])
            assert not np.isnan(json_data[api_name]['num_cupti_overhead_per_call_us'])

        def pretty_config(config):
            if config == 'gpu-activities':
                # return 'CUPTI GPU activities enabled'
                return 'CUPTI enabled'
            elif config == 'no-gpu-activities':
                # return 'CUPTI GPU activities disabled'
                return 'CUPTI disabled'
            else:
                raise NotImplementedError()
        df['pretty_config'] = df['config'].apply(pretty_config)

        logger.info("Output csv @ {path}".format(path=self._raw_csv_path))
        df.to_csv(self._raw_csv_path, index=False)
        logger.info("Output json @ {path}".format(path=self._raw_json_path))
        do_dump_json(json_data, self._raw_json_path)

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        plot_data = copy.copy(df)
        # Typically, there will be one big cudaFree/cudaMalloc; ignore it.
        # Just show cudaLaunchKernel and cudaMemcpyAsync.
        plot_data = plot_data[plot_data['api_name'].isin(constants.CUDA_API_ASYNC_CALLS)]
        # plot_data['field'] = "Per-API-call interception overhead"
        ax = fig.add_subplot(111)
        sns.barplot(x='api_name', y='us_per_call', hue='pretty_config', data=plot_data, ax=ax,
                    **sns_kwargs)
        ax.legend().set_title(None)
        ax.set_ylabel('Time per call (us)')
        ax.set_xlabel('CUDA API call')
        ax.set_title("CUPTI induced profiling overhead per CUDA API call")
        logger.info("Output plot @ {path}".format(path=self._png_path))
        fig.savefig(self._png_path)
        if is_pdf(self._png_path):
            pdf2png(self._png_path)
        plt.close(fig)

    @property
    def _raw_csv_path(self):
        return _j(self.directory, "cupti_overhead.raw.csv")

    @property
    def _raw_pairs_csv_path(self):
        return _j(self.directory, "cupti_overhead.pairs.raw.csv")

    @property
    def _raw_json_path(self):
        return _j(self.directory, "cupti_overhead.json")

    @property
    def _png_path(self):
        return _j(self.directory, "cupti_overhead.png")

def get_training_durations(directories,
                           debug=False,
                           debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.training_duration_us()
    return map_readers(TrainingProgressDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def get_n_total_calls(directories,
                      debug=False,
                      debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.n_total_calls()
    return map_readers(CUDAAPIStatsDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def get_per_api_stats(directories,
                      debug=False,
                      debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.per_api_stats()
    return map_readers(CUDAAPIStatsDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def get_pyprof_overhead_us(directories,
                           debug=False,
                           debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.total_pyprof_overhead_us()
    return map_readers(PyprofDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def get_pyprof_overhead_df(directories,
                           debug=False,
                           debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.total_pyprof_overhead_df()
    return map_readers(PyprofDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def add_col(data, colname, value):
    if colname not in data:
        data[colname] = []
    data[colname].append(value)

def pretty_overhead_type(overhead_type, unit='us'):

    def with_unit(col):
        return "{col}_{us}".format(
            col=col, us=unit)

    if overhead_type in {with_unit('total_pyprof_annotation_overhead'), with_unit('total_python_annotation_overhead')}:
        return "Python annotation"
    elif overhead_type == with_unit('total_pyprof_interception_overhead'):
        return r"Python$\leftrightarrow$C"
    elif overhead_type == with_unit('total_python_simulator_interception_overhead'):
        return r"Python$\leftrightarrow$Simulator"
    elif overhead_type == with_unit('total_python_tensorflow_interception_overhead'):
        return r"Python$\leftrightarrow$TensorFlow"
    elif overhead_type == with_unit('corrected_total_training_duration'):
        return "Corrected training time"
    elif overhead_type == with_unit('total_cupti_overhead'):
        # return 'CUPTI GPU activities enabled'
        return 'CUPTI'
    elif overhead_type == with_unit('total_interception_overhead'):
        # return 'CUPTI GPU activities disabled'
        return 'CUDA API interception'
    elif overhead_type == with_unit('uninstrumented_total_training_duration'):
        return 'Uninstrumented training time'
    else:
        return overhead_type


def add_stacked_bars(x, y, hue, label=None, data=None, ax=None, debug=False, **kwargs):
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

    xtick_labels = means[groups[0]][x]
    xticks = np.arange(len(xtick_labels))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    for group in groups:
        xs = means[group][x]
        assert (xs == xtick_labels).all()
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
        barplot = ax.bar(x=xticks, height=ys, yerr=std, label=label_str, bottom=bottom, **kwargs)

        if bottom is None:
            bottom = ys
        else:
            bottom += ys

    # Reverse legend label order (when making stacked bar its in reverse order)
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title=None, loc='upper left')
    ax.legend(handles[::-1], labels[::-1])


def output_csv(plot_df, csv_path, sort_by=None):
    if sort_by is not None:
        plot_df = plot_df.sort_values(sort_by)
    plot_df.to_csv(csv_path, index=False)
    logger.info("{path}: {msg}".format(path=csv_path, msg=pprint_msg(plot_df)))
    logger.info("Output total csv @ {path}".format(path=csv_path))


def get_short_env(env):
    short_env = stacked_bar_plots.get_x_env(env)
    return short_env

def add_short_env(df):
    df['short_env'] = df['env'].apply(get_short_env)

def add_x_field(df):
    """
    Add (algo, env) as x_field.
    """
    # def get_x_field(algo, env):
    #     return "({algo}, {env})".format(
    #         algo=algo, env=env)

    def get_x_field(algo, env):
        return stacked_bar_plots.get_x_field(
            algo, env,
            x_type='rl-comparison',
            human_readable=True)

    df['x_field'] = np.vectorize(get_x_field, otypes=[str])(df['algo'], df['env'])

def add_rlscope_config(df, rlscope_config):
    if 'metadata' not in rlscope_config:
        return
    def _add(col):
        df[col] = rlscope_config['metadata'].get(col, '')
    # Q: should we just set ALL the metadata?
    _add('algo')
    _add('env')

def get_unique(xs):
    set_xs = set(xs)
    assert len(set(xs)) == 1
    return next(iter(set_xs))

def join_single_rows(dfs, allow_zero=True):
    all_df = None
    for df in dfs:
        assert allow_zero or len(df) == 1
        df = copy.copy(df)
        assert 'join_idx' not in df.keys()
        df['join_idx'] = 0
        if all_df is None or ( len(all_df) == 0 and len(df) != 0 ):
            all_df = df
        elif len(df) == 0:
            continue
        else:
            all_df = all_df.merge(df)

    assert 0 <= len(all_df) <= 1
    del all_df['join_idx']
    return all_df

def check_cols_eq(df1, df2, colnames):
    for colname in colnames:
        assert np.all(np.equal(df1[colname].values, df2[colname].values))

def join_row_by_row(df1, df2, on=None, **kwargs):
    # Q: how to account for left/right suffix?

    df1 = df1.copy()
    df2 = df2.copy()

    join_cols = None
    if on is None:
        join_cols = ['join_idx']
    else:
        join_cols = ['join_idx'] + on

    assert 'join_idx' not in df1
    assert 'join_idx' not in df2
    assert len(df1) == len(df2)
    df1['join_idx'] = list(range(len(df1)))
    df2['join_idx'] = list(range(len(df1)))
    df = df1.merge(df2, on=join_cols, **kwargs)
    del df['join_idx']
    del df1['join_idx']
    del df2['join_idx']
    assert 'join_idx' not in df
    assert 'join_idx' not in df1
    assert 'join_idx' not in df2
    return df

def join_groups_row_by_row(df, join_cols, groupby_cols, get_suffix):
    """
    Use groupby_cols to group the dataframe, then for each group, join them row-by-row.
    EXPECTS: each group should have the exact same number of rows.
    """
    groupby = df.groupby(groupby_cols)
    # Simplification: only allow two "groups".
    # assert len(groupby) == 2
    group_df_sizes = set([len(group_df) for group, group_df in groupby])
    # Size of all the groups should be identical.
    assert len(group_df_sizes) == 1

    groups = [group for group, group_df in groupby]

    def as_group_dict(groupby_cols, group):
        return dict((field, value) for field, value in zip(groupby_cols, group))

    all_df = None
    for group, group_df in groupby:
        group_dict = as_group_dict(groupby_cols, group)
        suffix = get_suffix(group_dict)
        replace_cols = set(group_df.columns).difference(groupby_cols).difference(join_cols)
        if all_df is None:
            all_df = dataframe_add_suffix(group_df, suffix, replace_cols)
        else:
            keep_cols = set(group_df.columns).difference(groupby_cols)
            new_df = join_row_by_row(
                all_df,
                dataframe_add_suffix(group_df[keep_cols], suffix, replace_cols),
                on=join_cols,
            )
            all_df = new_df

    return all_df

def overhead_colname(col):
    return "{col}_overhead_us".format(col=col)

def overhead_per_call_colname(col):
    return "{col}_overhead_per_call_us".format(col=col)

def mean_per_call_colname(col):
    return "mean_{col}".format(col=overhead_per_call_colname(col))

def std_per_call_colname(col):
    return "std_{col}".format(col=overhead_per_call_colname(col))

def num_per_call_colname(col):
    return "num_{col}".format(col=overhead_per_call_colname(col))

def mean_colname(col):
    return "mean_{col}".format(col=col)

def std_colname(col):
    return "std_{col}".format(col=col)

def num_colname(col):
    return "num_{col}".format(col=col)

def merge_jsons(jsons, allow_overwrite=False):
    all_json = dict()
    for json in jsons:
        if not allow_overwrite:
            for key in json.keys():
                if key in all_json:
                    raise RuntimeError("merge_jsons got json-dictionaries with overlapping keys: key={key}".format(
                        key=key))
        all_json.update(json)
    return all_json

def dataframe_replace_us_with_sec(df, colnames=None):
    """
    Replace all the *_us columns with *_sec columns.

    :param df:
    :return:
    """

    if colnames is None:
        # Replace all *_us columns with *_sec.
        colnames = [col for col in df.keys() if is_usec_column(col)]

    for colname in colnames:
        assert colname in df.keys()
        assert is_usec_column(colname)

    for colname in colnames:
        df[sec_colname(colname)] = df[colname] / constants.USEC_IN_SEC
        del df[colname]

    return df

def is_usec_column(colname):
    return re.search(r'(_us$|_us_)', colname)

def sec_colname(us_colname):
    m = re.search(r'(?P<col>.*)_us$', us_colname)
    if m:
        return "{col}_sec".format(
            col=m.group('col'))

    m = re.search(r'^(?P<col_prefix>.*)_us_(?P<col_suffix>.*)$', us_colname)
    if m:
        return "{col_prefix}_sec_{col_suffix}".format(
            col_prefix=m.group('col_prefix'),
            col_suffix=m.group('col_suffix'),
        )

    assert False

def save_plot(fig, ax, png_path):
    logger.info("Output plot @ {path}".format(path=png_path))
    fig.savefig(png_path)
    if is_pdf(png_path):
        pdf2png(png_path)
    plt.close(fig)

def is_total_overhead_column(colname):
    return re.search(r'^total\b.*\boverhead.*\bus', re.sub(r'_+', ' ', colname))

def dataframe_add_suffix(df, suffix, cols):
    """
    Return a new dataframe where each column in <cols> has <suffix> appended to it.
    """
    new_df = pd.DataFrame()
    old_colnames = list(df.columns)
    for col in old_colnames:
        if col in cols:
            new_col = "{col}{suffix}".format(
                col=col, suffix=suffix)
            new_df[new_col] = df[col]
        else:
            new_df[col] = df[col]
    return new_df

class OverheadEventCountParser:
    """
    Compute the data needed to subtract from the venn.js files generated by rls-run.
    In particular:

    CUDA API interception:
        Subtract from:
        [CPU, q_forward, TensorFlow C++]
        Need to know: which operation is active at top-of-stack.

        Add CUDAAPIEvent.active_operation.

    CUPTI overhead:
        Subtract from:
        [CPU, q_forward, CUDA API]
        Need to know: which operation is active at top-of-stack.

        Already have enough information from CUDA API interception info.

    Python -> C-library interception:
        Subtract from:
        [CPU, q_forward, Python]
        Need to know: which operation is active at top-of-stack.

        Add pyprof.Event.active_operation.

    Python annotations:
        Subtract from:
        [CPU, q_forward, Python]
        Need to know: which operation is active at top-of-stack.

    PSEUDOCODE:

        json = dict()

        cuda_api_event_reader = CUDAAPIEventReader(directory)
        for event in cuda_api_event_reader.events():
            cuda_api_call_count[phase][operation][event.api_name] += 1

        json['cuda_api_call'] = cuda_api_call_count

        op_stack_reader = OpStackReader(directory)
        for overhead_type, count in op_stack_reader.overhead_types():
            json[overhead_type][phase][operation] = count


    overhead_event_count.json:
    {

        # overhead_types = ["CUDA API interception",
        #                   "CUPTI overhead"]
        #
        # phase-name -> operation-name -> # of overhead events
        'cuda_api_call': ...,

        # overhead_types = ["Python -> C-library interception",
        #                   "Python annotations"]
        #
        # overhead-type -> phase-name -> operation-name -> # of overhead events
        "Python -> C-library interception"
        'overhead_type_count': ...,

        # OR:
        # Need to make sure there are no conflicts
        'pyprof_annotation': ...,
        'pyprof_interception': ...,
    }
    """
    def __init__(self,
                 directory,
                 width=None,
                 height=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.directory = directory
        self.width = width
        self.height = height
        self.debug = debug
        self.filename_prefix = 'overhead_event_count'

    def run(self):
        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

        json = dict()

        def mk_dict(dic, key):
            if key not in dic:
                dic[key] = dict()

        def mk_default(dic, key, dflt):
            if key not in dic:
                dic[key] = dflt

        cuda_api_call_count = dict()
        cuda_api_stats_files = [path for path in each_file_recursive(self.directory) if is_cuda_api_stats_file(path)]
        def add_call_counts(cuda_api_stats_file):
            cuda_api_event_reader = CUDAAPIStatsReader(cuda_api_stats_file)
            phase = cuda_api_event_reader.phase
            machine_name = cuda_api_event_reader.machine_name
            process_name = cuda_api_event_reader.process_name
            for event in cuda_api_event_reader.cuda_api_call_events():
                mk_dict(cuda_api_call_count[machine_name], process_name)
                mk_dict(cuda_api_call_count[machine_name][process_name], phase)
                mk_dict(cuda_api_call_count[machine_name][process_name][phase], event.active_operation)
                mk_default(cuda_api_call_count[machine_name][process_name][phase][event.active_operation], event.api_name, 0)
                cuda_api_call_count[machine_name][process_name][phase][event.active_operation][event.api_name] += 1
        for cuda_api_stats_file in cuda_api_stats_files:
            add_call_counts(cuda_api_stats_file)
        json['cuda_api_call'] = cuda_api_call_count

        op_stack_files = [path for path in each_file_recursive(self.directory) if is_op_stack_file(path)]
        def add_op_stack_counts(op_stack_file):
            op_stack_reader = OpStackReader(op_stack_file)
            machine_name = op_stack_reader.machine_name
            process_name = op_stack_reader.process_name
            for overhead_type, phase, operation_name, num_overhead_events in op_stack_reader.all_events():
                mk_dict(json, overhead_type)
                mk_dict(json[overhead_type], machine_name)
                mk_dict(json[overhead_type][machine_name], process_name)
                mk_dict(json[overhead_type][machine_name][process_name], phase)
                assert operation_name not in json[overhead_type][machine_name][process_name][phase]
                json[overhead_type][machine_name][process_name][phase][operation_name] =\
                    json[overhead_type][machine_name][process_name][phase].get(operation_name, 0) + \
                    num_overhead_events
        for op_stack_file in op_stack_files:
            add_op_stack_counts(op_stack_file)

        logger.info("Output json @ {path}".format(path=self._json_path))
        do_dump_json(json, self._json_path)

    @property
    def _json_path(self):
        return _j(self.directory, "{prefix}.json".format(
            prefix=self.filename_prefix))

# class CorrectedVennParser:
#     """
#     Correct the time reported in venn.js files by subtracting overheads from various categories.
#     """
#     def __init__(self,
#                  overhead_event_count_json,
#                  venn_js_paths,
#                  width=None,
#                  height=None,
#                  debug=False,
#                  # Swallow any excess arguments
#                  **kwargs):
#         """
#         :param directories:
#         :param debug:
#         """
#         self.overhead_event_count_json_path = overhead_event_count_json
#         self.overhead_event_count_json = load_json(self.overhead_event_count_json_path)
#
#         self.venn_js_paths = venn_js_paths
#
#         self.width = width
#         self.height = height
#         self.debug = debug
#         # self.filename_prefix = 'overhead_event_count'
#
#     def directory(self, venn_js):
#         return _d(venn_js)
#
#     def run(self):
#         sns_kwargs = get_sns_kwargs()
#         plt_kwargs = get_plt_kwargs()
#
#         for venn_js_path in self.venn_js_paths:
#             corrected_venn_js_path = as_corrected_venn_js_file(venn_js_path)
#             vd = VennData(venn_js_path)
#             # Q: what do we want to do?
#             """
#             e.g. for CUPTI:
#             CUPTI overhead:
#                 Subtract from:
#                 [CPU, q_forward, CUDA API]
#                 Need to know: which operation is active at top-of-stack.
#             overhead_event_count_json[overhead_type]
#             """
#
#             vd.subtract_overhead(
#                 self.overhead_event_count_json,
#                 self.cupti_overhead_json,
#                 self.LD_PRELOAD_overhead_json,
#                 self.pyprof_overhead_json,
#             )
#
#         json = dict()
#
#         cuda_api_call_count = dict()
#         cuda_api_stats_files = [path for path in each_file_recursive(self.directory) if is_cuda_api_stats_file(path)]
#         def add_call_counts(cuda_api_stats_file):
#             cuda_api_event_reader = CUDAAPIStatsReader(cuda_api_stats_file)
#             phase = cuda_api_event_reader.phase
#             for event in cuda_api_event_reader.cuda_api_call_events():
#                 cuda_api_call_count[phase][event.active_operation][event.api_name] += 1
#         for cuda_api_stats_file in cuda_api_stats_files:
#             add_call_counts(cuda_api_stats_file)
#         json['cuda_api_call'] = cuda_api_call_count
#
#         op_stack_files = [path for path in each_file_recursive(self.directory) if is_op_stack_file(path)]
#         def add_op_stack_counts(op_stack_file):
#             op_stack_reader = OpStackReader(op_stack_file)
#             for overhead_type, phase, operation_name, num_overhead_events in op_stack_reader.all_events():
#                 if overhead_type not in json:
#                     json[overhead_type] = dict()
#                 if phase not in json[overhead_type]:
#                     json[overhead_type][phase] = dict()
#                 assert operation_name not in json[overhead_type][phase]
#                 json[overhead_type][phase][operation_name] = json[overhead_type][phase].get(operation_name, 0) + num_overhead_events
#         for op_stack_file in op_stack_files:
#             add_op_stack_counts(op_stack_file)
#
#         logger.info("Output json @ {path}".format(path=self._json_path))
#         do_dump_json(json, self._json_path)
#
#     @property
#     def _json_path(self):
#         return _j(self.directory, "{prefix}.json".format(
#             prefix=self.filename_prefix))

class SQLOverheadEventsParser:
    """
    Insert "profiling overhead events":
    - CUPTI and LD_PRELOAD:
      - CUDA API interception:
        Subtract from:
        [CPU, q_forward, TensorFlow C++]

      - CUPTI overhead:
        Subtract from:
        [CPU, q_forward, CUDA API]

                      T1                                    T2
                      |                                     |
      [intercept.call][  cudaLaunchKernel   <   CUPTI   >   ][intercept.return]

      Approximation choices:

                      T1                                    T2
                      |                                     |
      [intercept.call][  cudaLaunchKernel   <   CUPTI   >   ][    intercept.return    ]
                      [ CUPTI.overhead ]                     [intercept.overhead]

      NOTE:
      - Regarding effect of putting intercept.overhead during call/return:
        - The aggregate affect on shrunken categories is the same:
          TensorFlow C time will shrink, NOT the CUDA API calls.
          NOTE: technically NOT true... if you have high frequency CUDA API events then you may end up overlapping
          intercept.overhead with a CUDA API call...
          IDEALLY: intercept.overhead should NOT overlap with a CUDA API call.
        - validation: empirically see if it has a significant affect.
      - similar argument for where to put intercept.overhead...

      Insert:
          # CUPTI:
          Event(
              cuda_api.start_time,
              duration=mean_cupti_overhead_us[cuda_api])
          # LD_PRELOAD:
          Event(
              cuda_api.end_time,
              duration=mean_LD_PRELOAD_interception_overhead_us)

    - Python -> C-library interception:
      Subtract from:
      [CPU, q_forward, Python]

      These happen at call/return transition to/from Python/C++:
          before: [ Python ][overhead][   C++  ][overhead][ Python ]

      Since we take timestamps immediately before/after C++ call,
      most overhead is incurred by Python.

      Inaccuracy: hard to know how much of overhead is attributed to call vs return...
      Approximation choice: call-side overhead is worse [ i.e. (1) ]

      Approximation choices:

          (1) Call-biased:

                                           T1       T2
                                           |        |
              [        Python.call        ][ C.call ][       Python.return         ]
                      [Python interception]

          (2) Equal call/return overhead:

                                           T1       T2
                                           |        |
              [        Python.call        ][ C.call ][       Python.return         ]
                  [1/2 Python interception]          [1/2 Python interception]

          (3) Return-biased:
              Similar to call-biased.

          Q: How can we verify these assumptions...?
          A: Create a program that records operation events at high-frequency,
             and aims to make operation events a fixed duration (e.g. sleep(100us)).
             Vary the operation duration, and see how it affects accuracy of recorded events.
             Suspicion: at typical (long) operation durations, it doesn't matter,
             since aggregate affect is the same.

      Options:
      - Subtract from before.Python
      - Subtract equally from before.Python and after.Python
      - Net effect should be the same...
        ASSERT: python_event.duration >= subtraction

      Insert:
          # Python -> C-library interception
          Event(
              start=Python.call.end - mean_pyprof_interception_overhead_us,
              duration=mean_pyprof_interception_overhead_us)

    - Python annotations:
      Subtract from:
      [CPU, q_forward, Python]

      NOTE: we COULD make a microbenchmark to measure precise set_operation and end_operation time...
      Only difference is that current benchmarking is "tailored to a workload", which could work better with
      hardware concurrency.

      Insert:
          # Python annotations
          Event(
              start=op.end,
              duration=mean_pyprof_annotation_overhead_us)

      set_operation(op1)                                                                         end_operation(op1)
      [                                          op1                                             ]
                  [                              op2                               ]
                  set_operation(op2)                                               end_operation(op2)

      Approximation choices:
          NOTE: [overhead] is the total overhead-time added to by a pair of set_operation/end_operation calls.

          (1) Child-biased:

          set_operation(op1)                                                                         end_operation(op1)
          [                                          op1                                             ]
                                                                                            [overhead]
                      [                              op2                               ]
                      set_operation(op2)                                              end_operation(op2)
                                                                              [overhead]

          [ set_operation(op1) ][ set_operation(op2) ][ end_operation(op2) ][ end_operation(op1) ]
                                                       [annotation overhead] [annotation overhead]

          (2) Parent-biased:

          set_operation(op1)                                                                         end_operation(op1)
          [                                          op1                                             ]
                                                                                                      [overhead]
                      [                              op2                               ]
                      set_operation(op2)                                              end_operation(op2)
                                                                                        [overhead]

      These overheads happen in Python, but the operation to attribute the overheads to is tricky;
      difficulties:
      (1) overhead is distributed between parent/child operations;
          seems like most overhead is added to parent though based on timestamp locations.
          Hard to attribute precisely; attribute to parent.
      (2) overhead is counts set_operation/end_operation calls together;
          cannot separate overhead of each.


    "Subtracting" overhead:

        We handle "subtracting" profiling overhead using event-overlap computation abstraction.
        In particular, later on after we have compute the overlap, we will simply "discard" the CPU-side time that is
        due to profiling, being careful to keep [CPU, GPU] time but instead count it as GPU-only time:
            For Type Overlap.CategoryKey:
            - Discard CPU-only profiling overhead time:
                CategoryKey(
                    non_ops={constants.CATEGORIES_PROF, CPU},
                    ...
                )
                -> DISCARD
            - For [CPU, GPU] overlap, turn it into GPU-only time and merge with existing GPU-time:
                CategoryKey(
                    non_ops={constants.CATEGORIES_PROF, CPU, GPU},
                    ...
                )
                ->
                CategoryKey(
                    non_ops={constants.CATEGORIES_PROF,      GPU},
                    ...
                )
    """

    def __init__(self, directory,
                 cupti_overhead_json,
                 LD_PRELOAD_overhead_json,
                 pyprof_overhead_json,
                 host=None,
                 user=None,
                 password=None,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory

        # self.cupti_overhead_json = cupti_overhead_json
        # self.LD_PRELOAD_overhead_json = LD_PRELOAD_overhead_json
        # self.pyprof_overhead_json = pyprof_overhead_json

        raise NotImplementedError("Use cpp code, not old python implementation... (not maintained anymore)")


        self.cupti_overhead_json_path = cupti_overhead_json
        self.cupti_overhead_json = load_json(self.cupti_overhead_json_path)

        self.LD_PRELOAD_overhead_json_path = LD_PRELOAD_overhead_json
        self.LD_PRELOAD_overhead_json = load_json(self.LD_PRELOAD_overhead_json_path)

        self.pyprof_overhead_json_path = pyprof_overhead_json
        self.pyprof_overhead_json = load_json(self.pyprof_overhead_json_path)

        self.host = host
        self.user = user
        self.password = password
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.conn = get_sql_connection(db_path=self.db_path, host=self.host, user=self.user, password=self.password, debug=self.debug)

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    def run(self):
        """
        PSEUDOCODE:

            # Idempotent overhead-event insertion:
            #   delete overhead-events from previous run of SQLOverheadEventsParser.
            delete all from Event where e.category in constants.CATEGORIES_PROF

            #
            # Insert overhead events.
            #

            # Insert: Python annotations
            op_events = SQL:
                select all from Event where e.category == constants.CATEGORY_OPERATION
            for op in op_events:
                insert Event(
                    start=op.end,
                    duration=mean_pyprof_annotation_overhead_us,
                    category=constants.CATEGORY_PROF_PYTHON_ANNOTATION)

            # Insert: Python -> C-library interception
            python_events = SQL:
                select all from Event where e.category == constants.CATEGORY_PYTHON
            for python_event python_events:
                insert Event(
                    start=Python.call.end - mean_pyprof_interception_overhead_us,
                    duration=mean_pyprof_interception_overhead_us,
                    category=constants.CATEGORY_PROF_PYTHON_INTERCEPTION)

            # Insert: CUPTI & LD_PRELOAD
            cuda_api_events = SQL:
                select all from Event where e.category == constants.CATEGORY_CUDA_API_CPU
            for cuda_api_event cuda_api_events:
                # Insert: CUPTI
                insert Event(
                    cuda_api_event.start_time,
                    duration=mean_cupti_overhead_us[cuda_api_event.name],
                    category=constants.CATEGORY_PROF_CUPTI)
                # Insert: LD_PRELOAD
                insert Event(
                    cuda_api.end_time,
                    duration=mean_LD_PRELOAD_interception_overhead_us,
                    category=constants.CATEGORY_PROF_LD_PRELOAD)
        """
        c = self.conn.cursor

        # Idempotent overhead-event insertion:
        #   delete overhead-events from previous run of SQLOverheadEventsParser.
        self.delete_profiling_events(c)

        event_insert_cursor = self.conn.get_cursor()
        self.csv_inserter = CSVInserter(
            db_path=self.db_path, table='Event',
            host=self.host,
            user=self.user,
            password=self.password,
            cursor=event_insert_cursor,
            debug=self.debug)

        # TODO: we want our queries to use a SEPARATE cursor, and we DON'T want to fetchall rows (we want to iterate over the cursor).
        # Refactor:
        # - we DON'T want to query all the rows at once (too much event data)
        # - instead, we will iteratively fetch events from a cursor in "chunks"
        # - we will push chunks onto a queue
        # - workers will pull chunks from the queue, and process them
        # - workers are responsible for inserting events
        # Single-threaded version:
        # - the same thread that pushes chunks, also needs to process chunks;
        #   SOLUTION:
        #   - special code-path for single-threaded mode (test with this first)
        # Q: How to handle multiple Event insertions at the same time?
        # A: We already did this in SQL parser; follow whatever it did, except
        # now we send it a chunk of events instead of a path to a file to read from.

        with self.conn.with_cursors(), self.csv_inserter:
            #
            # Insert overhead events.
            #
            event_iter_cursor = self.conn.get_cursor()

            # Insert: Python annotations
            """
            op_events = SQL:
                select all from Event where e.category == constants.CATEGORY_OPERATION
            for op in op_events:
                insert Event(
                    start=op.end,
                    duration=mean_pyprof_annotation_overhead_us,
                    category=constants.CATEGORY_PROF_PYTHON_ANNOTATION)
            """
            op_events = self.query_op_events(event_iter_cursor)
            desc = "(1) Insert overhead events: Python annotations"
            logger.info("{desc}, num_events={n}".format(
                desc=desc,
                n=len(op_events),
            ))
            per_pyprof_annotation_overhead_us = self.pyprof_overhead_json['mean_pyprof_annotation_overhead_per_call_us']

            for op in progress(op_events.each_row(), desc=desc, total=len(op_events), show_progress=True):
                self.insert_overhead_event(
                    from_event=op,
                    # Parent biased.
                    start_time_us=op.end_time_usec,
                    duration_us=per_pyprof_annotation_overhead_us,
                    prof_category=constants.CATEGORY_PROF_PYTHON_ANNOTATION)

            # Insert: Python -> C-library interception
            """
            python_events = SQL:
                select all from Event where e.category == constants.CATEGORY_PYTHON
            for python_event python_events:
                insert Event(
                    start=Python.call.end - mean_pyprof_interception_overhead_us,
                    duration=mean_pyprof_interception_overhead_us,
                    category=constants.CATEGORY_PROF_PYTHON_INTERCEPTION)
            """
            c_events = self.query_c_events(event_iter_cursor)
            desc = "(2) Insert overhead events: Python -> C-library interception"
            logger.info("{desc}, num_events={n}".format(
                desc=desc,
                n=len(c_events),
            ))
            per_pyprof_interception_us = self.pyprof_overhead_json['mean_pyprof_interception_overhead_per_call_us']
            for event in progress(c_events.each_row(), desc=desc, total=len(c_events), show_progress=True):
                self.insert_overhead_event(
                    from_event=event,
                    start_time_us=Decimal(event.end_time_usec) - Decimal(per_pyprof_interception_us),
                    duration_us=per_pyprof_interception_us,
                    prof_category=constants.CATEGORY_PROF_PYTHON_INTERCEPTION)

            # Insert: CUPTI & LD_PRELOAD
            """
            cuda_api_events = SQL:
                select all from Event where e.category == constants.CATEGORY_CUDA_API_CPU
            for cuda_api_event cuda_api_events:
                # Insert: CUPTI
                insert Event(
                    cuda_api_event.start_time,
                    duration=mean_cupti_overhead_us[cuda_api_event.name],
                    category=constants.CATEGORY_PROF_CUPTI)
                # Insert: LD_PRELOAD
                insert Event(
                    cuda_api.end_time,
                    duration=mean_LD_PRELOAD_interception_overhead_us,
                    category=constants.CATEGORY_PROF_LD_PRELOAD)
            """
            cuda_api_events = self.query_cuda_api_events(event_iter_cursor)
            desc = "(3) Insert overhead events: CUPTI & LD_PRELOAD"
            logger.info("{desc}, num_events={n}".format(
                desc=desc,
                n=len(cuda_api_events),
            ))
            per_LD_PRELOAD_interception_us = self.LD_PRELOAD_overhead_json['mean_interception_overhead_per_call_us']
            missing_cupti_overhead_cuda_api_calls = dict()
            for event in progress(cuda_api_events.each_row(), desc=desc, total=len(cuda_api_events), show_progress=True):
                cuda_api_name = event.name
                if cuda_api_name not in self.cupti_overhead_json:
                    missing_cupti_overhead_cuda_api_calls[cuda_api_name] = missing_cupti_overhead_cuda_api_calls.get(cuda_api_name, 0) + 1
                else:
                    cupti_overhead_us = self.cupti_overhead_json[cuda_api_name]['mean_cupti_overhead_per_call_us']
                    # Insert: CUPTI
                    self.insert_overhead_event(
                        from_event=event,
                        start_time_us=event.start_time_usec,
                        duration_us=cupti_overhead_us,
                        prof_category=constants.CATEGORY_PROF_CUPTI)
                # Insert: LD_PRELOAD
                self.insert_overhead_event(
                    from_event=event,
                    start_time_us=event.end_time_usec,
                    duration_us=per_LD_PRELOAD_interception_us,
                    prof_category=constants.CATEGORY_PROF_LD_PRELOAD)

            if len(missing_cupti_overhead_cuda_api_calls) > 0:
                logger.warning("Saw CUDA API calls that we didn't have calibrated CUPTI overheads for overheads for in {path}: {msg}".format(
                    path=self.cupti_overhead_json_path,
                    msg=pprint_msg(missing_cupti_overhead_cuda_api_calls),
                ))

            self.conn.put_cursor(event_insert_cursor)
            self.conn.put_cursor(event_iter_cursor)

    def insert_overhead_event(self, from_event, start_time_us, duration_us, prof_category):
        assert prof_category in constants.CATEGORIES_PROF
        return self.csv_inserter.insert_event(
            device_id=from_event.device_id, process_id=from_event.process_id, phase_id=from_event.process_id,
            category=prof_category,
            start_time_us=start_time_us,
            duration_us=duration_us,
            # Inherit event name...?
            name=from_event.name,
            thread_id=from_event.thread_id,
        )


    def query_op_events(self, c):
        return self._query_category_events(c, category=constants.CATEGORY_OPERATION)

    def query_cuda_api_events(self, c):
        return self._query_category_events(c, category=constants.CATEGORY_CUDA_API_CPU)

    def _query_category_events(self, c, category):
        # TODO: if we wish to parallelize this, we can provide (event_id_lower_bound, event_id_upper_bound)
        # to limit the "paginate" the full query.  Note however that lower/upper bounds would be formed using a
        # single serial query; you can approximate the split using the min/max(event_id) or, precisely split by querying
        # all event_id's (RISK: could be large in memory size).
        select_query = textwrap.dedent("""
            SELECT
                {event_fields}
            FROM 
                Event AS e
                NATURAL JOIN Category AS c
            WHERE 
                c.category_name = '{category}' 
            """).format(
            category=category,
            event_fields=self.sql_event_fields(event_alias='e', indents=1),
        )
        rows = self._query_event_rows(c, select_query)
        return rows

    def query_c_events(self, c):
        """
        NOTE: we CANNOT simply use _query_category_events(constants.CATEGORY_PYTHON) since
        there are a couple of python events we insert that are NOT interception events (e.g. "Finishing ..." during end_operation).
        To figure out EXACTLY which python event to use, look at which python event is inserted during an interception, OR
        use the C++ event (PROBLEM: the category for those varies between simulator/tensorflow...)

        See: PyprofDataframeReader.total_intercepted_calls()

        :param c:
        :return:
        """
        select_query = textwrap.dedent("""
            SELECT
                {event_fields}
            FROM 
                Event AS e
                NATURAL JOIN Category AS c
            WHERE 
                {category_clause}
            """).format(
            event_fields=self.sql_event_fields(event_alias='e', indents=1),
            category_clause=sql_operator_in(expr='c.category_name', values=sorted(constants.CATEGORIES_C_EVENTS), indents=1),
        )
        rows = self._query_event_rows(c, select_query)
        return rows

    def sql_event_fields(self, event_alias, indents=None):
        """
        SELECT e.field1, e.field2, ...
               -----------------------
               Return this part of a select statement.
               Need at least these fields to generate "overhead events"
        FROM Event AS e
        WHERE
            ...

        :param c:
        :param event_alias:
        :return:
        """
        keep_Event_fields = [
            'thread_id',
            'process_id',
            'phase_id',
            'device_id',
        ]
        sql_events = textwrap.dedent("""
            -- Use KernelTime.field_names
            {e}.event_name as name,
            {e}.start_time_us as start_usec,
            {e}.duration_us as time_usec,
            -- Keep Event field for re-inserting overhead-event; 
            --   We need these fields so that we can create a Event(category=constants.CATEGORY_PROF_ANNOTATION, ...)
            {keep_Event_fields}
            """).format(
            keep_Event_fields=', '.join(["{e}.{field}".format(e=event_alias, field=field) for field in keep_Event_fields]),
            e=event_alias,
        )
        sql_events = maybe_indent(sql_events, indents)
        return sql_events

    def _query_event_rows(self, c, select_query, fetchall=False):
        row_iter = RowIterator(select_query, cursor=c, RowKlass=KernelTime, debug=self.debug)
        return row_iter

    def delete_profiling_events(self, c):
        prof_clause = sql_operator_in('c.category_name', values=constants.CATEGORIES_PROF, indents=1)
        select_query = textwrap.dedent("""
            SELECT COUNT(*) as num_rows
            FROM 
                Event AS e
                NATURAL JOIN Category AS c
            WHERE 
                {prof_clause}
            """).format(
            prof_clause=prof_clause,
        )
        sql_exec_query(c, select_query, klass=self.__class__, debug=self.debug)
        rows = c.fetchall()
        assert len(rows) == 1
        num_rows = rows[0]['num_rows']
        logger.info("Idempotent overhead-event insertion: deleting {n} profiling-overhead-events".format(
            n=num_rows,
        ))
        # delete_query = textwrap.dedent("""
        #     DELETE e.*
        #     FROM
        #         Event AS e
        #         NATURAL JOIN Category AS c
        #     WHERE
        #         {prof_clause}
        #     """).format(
        #         prof_clause=prof_clause,
        #     )
        delete_query = textwrap.dedent("""
            DELETE FROM Event AS e
            USING
                Category AS c
            WHERE 
                e.category_id = c.category_id AND
                {prof_clause}
            """).format(
            prof_clause=prof_clause,
        )
        sql_exec_query(c, delete_query, klass=self.__class__, debug=self.debug)


def check_calibration(js, field, path):
    if js[field] < 0:
        # raise RuntimeError(textwrap.dedent("""
        logger.warning(textwrap.dedent("""
            Saw negative overhead calibration for \"{field}\" = {value} @ {path}
              To fix this, run additional repetitions in your experiments.
            """).format(
            field=field,
            value=js[field],
            path=path,
        ))

@contextmanager
def ignore_seaborn_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Using the barplot.*without.*order', category=UserWarning, module=r'seaborn')
        yield
