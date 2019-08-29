import logging
import copy
import itertools
import argparse

from iml_profiler.protobuf.pyprof_pb2 import Pyprof, MachineUtilization, DeviceUtilization, UtilizationSample
from iml_profiler.parser.common import *
from iml_profiler.profiler import experiment
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from iml_profiler.parser.dataframe import TrainingProgressDataframeReader, CUDAAPIStatsDataframeReader, PyprofDataframeReader
from iml_profiler.parser.stacked_bar_plots import StackedBarPlot

from iml_profiler.parser import stacked_bar_plots
from iml_profiler.parser.db import SQLCategoryTimesReader, sql_input_path

from iml_profiler.profiler import iml_logging

class CorrectedTrainingTimeParser:
    """
    Compute total training time, after "subtracting" various sources of profiling overhead.
    """
    def __init__(self,
                 cupti_overhead_json,
                 call_interception_overhead_json,
                 iml_directories,
                 uninstrumented_directories,
                 directory,
                 width=None,
                 height=None,
                 debug=False,
                 debug_memoize=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.cupti_overhead_json_path = cupti_overhead_json
        self.cupti_overhead_json = load_json(self.cupti_overhead_json_path)

        self.call_interception_overhead_json_path = call_interception_overhead_json
        self.interception_overhead_json = load_json(self.call_interception_overhead_json_path)

        self.iml_directories = iml_directories
        self.uninstrumented_directories = uninstrumented_directories
        self.directory = directory
        self.width = width
        self.height = height
        self.debug = debug
        self.debug_memoize = debug_memoize

    def run(self):
        """
        Sources of overhead:
        - Pyprof overhead = sum(Event.duration_profiling_overhead_us If Event.start_profiling_overhead_us != 0)
          - Need to parse pyprof Event files
        - CUPTI overhead = sum(api.n_calls * api.mean_cupti_per_call_overhead_us)
          - parse api.n_calls using CUDAAPIStatsDataframeReader
          - parse api.mean_cupti_per_call_overhead_us using cupti_overhead_json
        - Interception overhead = sum(total_n_calls * api.mean_interception_per_call_overhead_us)
          - parse total_n_calls using CUDAAPIStatsDataframeReader
          - parse api.mean_cupti_per_call_overhead_us using call_interception_overhead_json
        - "Subtracted" time = total_training_time - [ … sum of overheads … ]
          - parse total_training_time using TrainingProgressDataframeReader

        Stats to save:
            # Overheads:
            Pyprof overhead (absolute time, and percent)
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
            --iml-directory containing profiling data.
        """

        # capsize = 5
        # plt.style.use('seaborn')
        # plt.rcParams.update({
        #     "lines.markeredgewidth" : 1,
        #     "errorbar.capsize": capsize,
        # })

        sns_kwargs = get_sns_kwargs()
        plt_kwargs = get_plt_kwargs()

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
            #     total_training_time_us
            #     total_overhead_us
            #     total_training_time_us
            #     corrected_total_training_time_us
            total_dfs = []
            for directory in self.iml_directories:
                """
                - Pyprof overhead = sum(Event.duration_profiling_overhead_us If Event.start_profiling_overhead_us != 0)
                  - Need to parse pyprof Event files
                """
                total_pyprof_overhead_us = get_pyprof_overhead_us(directory, self.debug)

                """
                - CUPTI overhead = sum(api.n_calls * api.mean_cupti_per_call_overhead_us for each api)
                  - parse api.n_calls using CUDAAPIStatsDataframeReader
                  - parse api.mean_cupti_per_call_overhead_us using cupti_overhead_json
                """
                # TODO: make it work for multiple self.directories
                # Could be nice to see a stacked-bar graph that "breaks down" total training time by its overhead sources...
                # probably a pretty busy plot though.
                per_api_stats = get_per_api_stats(directory, self.debug)
                per_api_stats_df = per_api_stats.reset_index()

                per_api_df = pd.DataFrame({
                    'api_name': per_api_stats_df['api_name'],
                    'num_calls': per_api_stats_df['num_calls'],
                    # 'total_cupti_overhead_us': per_api_stats_df['num_calls'] * self.cupti_overhead_json['mean_cupti_overhead_per_call_us'],
                    'total_interception_overhead_us': per_api_stats_df['num_calls'] * self.interception_overhead_json['mean_interception_overhead_per_call_us'],
                })
                # - make json a df
                # - join on api_name, make column 'mean_cupti_overhead_per_call_us'
                # - multiply mean by num_calls
                cupti_overhead_cols = dict()
                for api_name, cupti_data in self.cupti_overhead_json.items():
                    add_col(cupti_overhead_cols, 'api_name', api_name)
                    for field, value in cupti_data.items():
                        add_col(cupti_overhead_cols, field, value)
                cupti_overhead_df = pd.DataFrame(cupti_overhead_cols)
                per_api_df = per_api_df.merge(cupti_overhead_df, on=['api_name'])
                per_api_df['total_cupti_overhead_us'] = per_api_df['num_calls'] * per_api_df['mean_cupti_overhead_per_call_us']

                per_api_df['total_overhead_us'] = per_api_df['total_cupti_overhead_us'] + per_api_df['total_interception_overhead_us']
                per_api_dfs.append(per_api_df)
                total_cupti_overhead_us = np.sum(per_api_df['total_cupti_overhead_us'])

                # Q: output csv/json with breakdown of overhead per-api call?
                # per_api_cupti_overhead_us = dict()
                # for api_name, row in per_api_stats.iterrows():
                #     per_api_cupti_overhead_us[api_name] = row['num_calls'] * self.cupti_overhead_json['mean_cupti_overhead_per_call_us']
                # total_cupti_overhead_us = np.sum(per_api_cupti_overhead_us.values())

                """
                - Interception overhead = sum(total_n_calls * api.mean_interception_per_call_overhead_us)
                  - parse total_n_calls using CUDAAPIStatsDataframeReader
                  - parse api.mean_cupti_per_call_overhead_us using call_interception_overhead_json
                """
                # per_api_interception_overhead_us = dict()
                # for api_name, row in per_api_stats.iterrows():
                #     per_api_interception_overhead_us[api_name] = row['num_calls'] * self.interception_overhead_json['mean_interception_overhead_per_call_us']
                # total_interception_overhead_us = np.sum(per_api_interception_overhead_us.values())

                # per_api_df['total_interception_overhead_us'] = per_api_df['num_calls'] * self.interception_overhead_json['mean_interception_overhead_per_call_us']
                total_interception_overhead_us = np.sum(per_api_df['total_interception_overhead_us'])

                """
                - "Subtracted" time = total_training_time - [ … sum of overheads … ]
                  - parse total_training_time using TrainingProgressDataframeReader
                """

                total_training_time_us = get_training_durations(directory, self.debug)
                total_df = pd.DataFrame({
                    'total_pyprof_overhead_us': [total_pyprof_overhead_us],
                    'total_cupti_overhead_us': [total_cupti_overhead_us],
                    'total_interception_overhead_us': [total_interception_overhead_us],
                    'total_training_time_us': [total_training_time_us],
                })
                # WARNING: protect against bug where we create more than one row unintentionally.
                # If we mix scalars/lists, pd.DataFrame will "duplicate" the scalars to match the list length.
                # This can happen if we accidentally include per-api times instead of summing across times.
                assert len(total_df) == 1
                total_df['total_overhead_us'] = total_df['total_pyprof_overhead_us'] + total_df['total_cupti_overhead_us'] + total_df['total_interception_overhead_us']
                total_df['corrected_total_training_time_us'] = total_df['total_training_time_us'] - total_df['total_overhead_us']
                total_dfs.append(total_df)

            total_df = pd.concat(total_dfs)
            per_api_df = pd.concat(per_api_dfs)
            ret = (total_df, per_api_df)

            maybe_memoize(self.debug_memoize, ret, memoize_path)

            return ret

        total_df, per_api_df = load_dfs()


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
        # logging.info("Output json @ {path}".format(path=self._raw_json_path))
        # do_dump_json(json_data, self._raw_json_path)

        def get_per_api_plot_data(per_api_df):
            overhead_cols = ['total_cupti_overhead_us', 'total_interception_overhead_us']
            keep_cols = ['api_name', 'num_calls'] + overhead_cols
            # Keep these columns "intact" (remove 'total_overhead_us')
            id_vars = ['api_name', 'num_calls']
            keep_df = per_api_df[keep_cols]
            per_api_plot_data = pd.melt(keep_df, id_vars=id_vars, var_name='overhead_type', value_name='total_overhead_us')
            per_api_plot_data['pretty_overhead_type'] = per_api_plot_data['overhead_type'].apply(
                lambda overhead_type: pretty_overhead_type(overhead_type, 'us'))
            per_api_plot_data['total_overhead_sec'] = per_api_plot_data['total_overhead_us'] / USEC_IN_SEC
            del per_api_plot_data['total_overhead_us']
            return per_api_plot_data

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logging.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None

        fig = plt.figure(figsize=figsize)
        per_api_plot_data = get_per_api_plot_data(per_api_df)
        output_csv(per_api_plot_data, self._per_api_csv_path, sort_by=['total_overhead_sec'])
        ax = fig.add_subplot(111)
        sns.barplot(
            x='api_name', y='total_overhead_sec', hue='pretty_overhead_type', data=per_api_plot_data, ax=ax,
            **sns_kwargs)
        ax.legend().set_title(None)
        ax.set_ylabel('Total training time (sec)')
        ax.set_xlabel('CUDA API call')
        ax.set_title("Breakdown of profiling overhead by CUDA API call")
        logging.info("Output plot @ {path}".format(path=self._per_api_png_path))
        fig.savefig(self._per_api_png_path)
        plt.close(fig)

        def get_total_plot_data(total_df):
            #     total_pyprof_overhead_us
            #     total_cupti_overhead_us
            #     total_interception_overhead_us
            #     total_training_time_us
            #     corrected_total_training_time_us
            #
            #     total_overhead_us
            #     total_training_time_us
            total_plot_data = copy.copy(total_df)
            # Ideally, (algo, env), but we don't have that luxury.
            total_plot_data['x_field'] = ""

            # Melt overhead columns, convert to seconds.

            # We don't have any unique column values for this data-frame, so we cannot use id_vars...
            id_vars = [
                # 'total_training_time_us',
                "x_field",
            ]
            overhead_cols = [
                'total_pyprof_overhead_us',
                'total_cupti_overhead_us',
                'total_interception_overhead_us',
                # 'total_training_time_us',
                'corrected_total_training_time_us',
            ]

            # Keep these columns "intact" (remove 'total_overhead_us')
            keep_cols = id_vars + overhead_cols
            keep_df = total_plot_data[keep_cols]
            # total_plot_data = pd.melt(keep_df, id_vars=id_vars, var_name='overhead_type', value_name='total_overhead_us')
            total_plot_data = pd.melt(keep_df, id_vars=id_vars, var_name='overhead_type', value_name='total_overhead_us')
            total_plot_data['total_overhead_sec'] = total_plot_data['total_overhead_us'] / USEC_IN_SEC
            del total_plot_data['total_overhead_us']
            total_plot_data['pretty_overhead_type'] = total_plot_data['overhead_type'].apply(
                lambda overhead_type: pretty_overhead_type(overhead_type, 'us'))

            # Q: if we sort do we get consistent order?
            # total_plot_data = total_plot_data.sort_values(['x_field', 'pretty_overhead_type'])

            overhead_type_map = as_order_map(reversed(overhead_cols))
            total_plot_data['overhead_type_order'] = total_plot_data['overhead_type'].apply(
                lambda overhead_type: overhead_type_map[overhead_type])
            total_plot_data = total_plot_data.sort_values(['x_field', 'overhead_type_order'])

            return total_plot_data

        total_plot_data = get_total_plot_data(total_df)
        # total_plot_data_groupby = total_plot_data.groupby(['overhead_type'])

        fig = plt.figure(figsize=figsize)
        # plot_data['field'] = "Per-API-call interception overhead"
        ax = fig.add_subplot(111)
        # sns.barplot(x='x_field', y='training_time_sec', hue='pretty_config', data=training_time_plot_data, ax=ax)
        output_csv(total_plot_data, self._total_csv_path)
        add_stacked_bars(x='x_field', y='total_overhead_sec',
                         hue='overhead_type_order',
                         label='pretty_overhead_type',
                         data=total_plot_data, ax=ax,
                         **plt_kwargs)
        # ax.legend().set_title(None)
        ax.set_ylabel('Total training time (sec)')
        ax.set_xlabel('(algo, env)')
        ax.set_title("Breakdown of profiling overhead")
        logging.info("Output plot @ {path}".format(path=self._total_png_path))
        fig.savefig(self._total_png_path)
        plt.close(fig)

        unins_training_time_us = get_training_durations(self.uninstrumented_directories, self.debug)
        unins_df = pd.DataFrame({
            'training_time_us': unins_training_time_us,
        })
        unins_df['config'] = 'uninstrumented'

        ins_df = pd.DataFrame({
            'training_time_us': total_df['total_training_time_us'],
        })
        ins_df['config'] = 'instrumented'

        corrected_df = pd.DataFrame({
            'training_time_us': total_df['corrected_total_training_time_us'],
        })
        corrected_df['config'] = 'corrected'

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
        training_time_df = pd.concat([unins_df, ins_df, corrected_df])
        training_time_df['pretty_config'] = training_time_df['config'].apply(pretty_config)

        def get_training_time_plot_data(training_time_df):
            training_time_plot_data = copy.copy(training_time_df)
            # TODO: (algo, env)
            training_time_plot_data['x_field'] = ""

            training_time_plot_data['training_time_sec'] = training_time_plot_data['training_time_us'] / USEC_IN_SEC
            del training_time_plot_data['training_time_us']
            return training_time_plot_data

        training_time_plot_data = get_training_time_plot_data(training_time_df)

        def get_percent_bar_labels(df):
            # NOTE: need .values otherwise we get NaN's
            unins_time_sec = df[df['pretty_config'] == 'Uninstrumented']['training_time_sec'].values
            df['perc'] = ( df['training_time_sec'] - unins_time_sec ) / unins_time_sec
            def get_label(row):
                if row['pretty_config'] == 'Uninstrumented':
                    assert row['perc'] == 0.
                    return ""
                return "{perc:.1f}%".format(perc=100*row['perc'])

            bar_labels = df.apply(get_label, axis=1)
            df['bar_labels'] = bar_labels
            return bar_labels

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        output_csv(training_time_plot_data, self._training_time_csv_path, sort_by=['training_time_sec'])
        sns.barplot(
            x='x_field', y='training_time_sec', hue='pretty_config', data=training_time_plot_data, ax=ax,
            **sns_kwargs)
        add_bar_labels(y='training_time_sec', hue='pretty_config', ax=ax,
                       get_bar_labels=get_percent_bar_labels)
        ax.legend().set_title(None)
        ax.set_ylabel('Total training time (sec)')
        ax.set_xlabel('(algo, env)')
        ax.set_title("Correcting training time by subtracting profiling overhead")
        logging.info("Output plot @ {path}".format(path=self._training_time_png_path))
        fig.savefig(self._training_time_png_path)
        plt.close(fig)

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
        return _j(self.directory, "corrected_training_time.total.png")

    @property
    def _per_api_png_path(self):
        return _j(self.directory, "corrected_training_time.per_api.png")

    @property
    def _training_time_png_path(self):
        return _j(self.directory, "corrected_training_time.training_time.png")

    @property
    def _training_time_csv_path(self):
        return _j(self.directory, "corrected_training_time.training_time.csv")

class CallInterceptionOverheadParser:
    """
    config_interception
    Run with interception enabled.
    $ iml-prof --debug --cuda-api-calls --cuda-api-events --iml-disable
    # We want to know how much time is spent just intercepting API calls (NOT GPU activities)

    config_no_interception
    Run with interception disabled.
    $ iml-prof --debug --iml-disable
    # Time spent without ANY interception / profiling overhead at all.
    """
    def __init__(self,
                 interception_directory,
                 no_interception_directory,
                 directory,
                 # ignore_phase=False,
                 # algo_env_from_dir=False,
                 # baseline_config=None,
                 width=None,
                 height=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.interception_directory = interception_directory
        self.no_interception_directory = no_interception_directory
        self.directory = directory
        self.width = width
        self.height = height
        # self.iml_directories = iml_directories
        # self.ignore_phase = ignore_phase
        # self.algo_env_from_dir = algo_env_from_dir
        # self.baseline_config = baseline_config
        self.debug = debug

        # self.added_fields = set()

    @property
    def _raw_csv_path(self):
        return _j(self.directory, "call_interception_overhead.raw.csv")

    @property
    def _raw_json_path(self):
        return _j(self.directory, "call_interception_overhead.json")

    @property
    def _agg_csv_path(self):
        return _j(self.directory, "call_interception_overhead.agg.csv")

    @property
    def _png_path(self):
        return _j(self.directory, "call_interception_overhead.png")

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

        int_training_duration_us = get_training_durations(self.interception_directory, debug=self.debug)
        no_int_training_duration_us = get_training_durations(self.no_interception_directory, debug=self.debug)
        if len(int_training_duration_us) != len(no_int_training_duration_us):
            raise RuntimeError("You need to run the same number of repetitions for both config_interception and config_no_interception")

        int_total_calls = get_n_total_calls(self.interception_directory, debug=self.debug)
        # no_int_total_calls = get_n_total_calls(self.no_interception_directory, debug=self.debug)
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

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logging.info("Setting figsize = {fig}".format(fig=figsize))
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
        plt.close(fig)

        logging.info("Output plot @ {path}".format(path=self._png_path))
        logging.info("Output csv @ {path}".format(path=self._raw_csv_path))
        logging.info("Output json @ {path}".format(path=self._raw_json_path))

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
        # self.iml_directories = iml_directories
        # self.ignore_phase = ignore_phase
        # self.algo_env_from_dir = algo_env_from_dir
        # self.baseline_config = baseline_config
        self.debug = debug

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

        csv_data = dict()

        # api_name -> {
        #   mean_us_per_call: ...,
        #   std_us_per_call: ...,
        # }
        for (config, directories) in [
            ('gpu-activities', self.gpu_activities_directory),
            ('no-gpu-activities', self.no_gpu_activities_directory),
        ]:
            for directory in directories:
                per_api_stats = get_per_api_stats(directory, debug=self.debug)
                logging.info("per_api_stats: " + pprint_msg(per_api_stats))
                get_per_api_stats(directory, debug=self.debug)
                for api_name, row in per_api_stats.iterrows():
                    total_api_time_us = row['total_time_us']
                    total_num_calls = row['num_calls']

                    add_col(csv_data, 'config', config)
                    add_col(csv_data, 'api_name', api_name)
                    add_col(csv_data, 'total_num_calls', total_num_calls)
                    add_col(csv_data, 'total_api_time_us', total_api_time_us)
                    us_per_call = total_api_time_us / float(total_num_calls)
                    add_col(csv_data, 'us_per_call', us_per_call)

        def merge_rows_in_order(df):
            def get_suffix(config):
                return "_{config}".format(config=re.sub('-', '_', config))

            groupby_config = df.groupby(['config'])
            assert len(groupby_config) == 2
            config_to_dfs = dict()
            for config, config_df in groupby_config:
                groupby_api_name = config_df.groupby(['api_name'])
                config_to_dfs[config] = []
                for api_name, api_name_df in groupby_api_name:
                    assert 'join_row_index' not in api_name_df
                    api_name_df['join_row_index'] = list(range(len(api_name_df)))
                    config_to_dfs[config].append(api_name_df)

            config_to_df = dict((config, pd.concat(dfs)) for config, dfs in config_to_dfs.items())
            assert len(config_to_df) == 2
            configs = sorted(config_to_df.keys())

            join_cols = ['api_name', 'join_row_index']

            config1 = configs[0]
            df1 = config_to_df[config1]
            # df1 = df1.set_index(join_cols)

            config2 = configs[1]
            df2 = config_to_df[config2]
            # df2 = df2.set_index(join_cols)

            joined_df = df1.merge(df2, on=join_cols, suffixes=(get_suffix(config1), get_suffix(config2)))
            del joined_df['join_row_index']

            return joined_df

        def colname(col, config):
            return "{col}_{config}".format(
                col=col,
                config=re.sub('-', '_', config))

        df = pd.DataFrame(csv_data)
        df_csv = df.sort_values(['config', 'api_name', 'us_per_call'])
        df_csv.to_csv(self._raw_csv_path, index=False)
        logging.info("per_api_stats: " + pprint_msg(per_api_stats))

        joined_df = merge_rows_in_order(df)
        joined_df.to_csv(self._raw_pairs_csv_path, index=False)
        joined_df['cupti_overhead_per_call_us'] = joined_df[colname('us_per_call', 'gpu-activities')] - joined_df[colname('us_per_call', 'no-gpu-activities')]
        json_data = dict()
        for api_name, df_api_name in joined_df.groupby('api_name'):
            assert api_name not in json_data
            json_data[api_name] = dict()
            json_data[api_name]['mean_cupti_overhead_per_call_us'] = np.mean(df_api_name['cupti_overhead_per_call_us'])
            json_data[api_name]['std_cupti_overhead_per_call_us'] = np.std(df_api_name['cupti_overhead_per_call_us'])
            json_data[api_name]['num_cupti_overhead_per_call_us'] = len(df_api_name['cupti_overhead_per_call_us'])

        # api_names = set(df['api_name'])
        # for api_name in api_names:
        #     assert api_name not in json_data
        #     json_data[api_name] = dict()
        #     df_api_name = df[df['api_name'] == api_name]
        #     json_data[api_name]['mean_us_per_call'] = np.mean(df_api_name['us_per_call'])
        #     json_data[api_name]['std_us_per_call'] = np.std(df_api_name['us_per_call'])

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

        logging.info("Output csv @ {path}".format(path=self._raw_csv_path))
        df.to_csv(self._raw_csv_path, index=False)
        logging.info("Output json @ {path}".format(path=self._raw_json_path))
        do_dump_json(json_data, self._raw_json_path)

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logging.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        plot_data = copy.copy(df)
        # plot_data['field'] = "Per-API-call interception overhead"
        ax = fig.add_subplot(111)
        sns.barplot(x='api_name', y='us_per_call', hue='pretty_config', data=plot_data, ax=ax,
                    **sns_kwargs)
        ax.legend().set_title(None)
        ax.set_ylabel('Time per call (us)')
        ax.set_xlabel('CUDA API call')
        ax.set_title("CUPTI induced profiling overhead per CUDA API call")
        logging.info("Output plot @ {path}".format(path=self._png_path))
        fig.savefig(self._png_path)
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

def map_readers(DataframeReaderKlass, directories, func, debug=False):
    xs = []

    if type(directories) == str:
        dirs = [directories]
    else:
        dirs = list(directories)

    for directory in dirs:
        df_reader = DataframeReaderKlass(
            directory,
            # add_fields=self.maybe_add_algo_env,
            debug=debug)
        x = func(df_reader)
        xs.append(x)

    if type(directories) == str:
        assert len(xs) == 1
        return xs[0]
    return xs

def get_training_durations(directories, debug):
    def get_value(df_reader):
        return df_reader.training_duration_us()
    return map_readers(TrainingProgressDataframeReader, directories, get_value, debug=debug)

def get_n_total_calls(directories, debug):
    def get_value(df_reader):
        return df_reader.n_total_calls()
    return map_readers(CUDAAPIStatsDataframeReader, directories, get_value, debug=debug)

def get_per_api_stats(directories, debug):
    def get_value(df_reader):
        return df_reader.per_api_stats()
    return map_readers(CUDAAPIStatsDataframeReader, directories, get_value, debug=debug)

def get_pyprof_overhead_us(directories, debug):
    def get_value(df_reader):
        return df_reader.total_pyprof_overhead_us()
    return map_readers(PyprofDataframeReader, directories, get_value, debug=debug)

def add_col(data, colname, value):
    if colname not in data:
        data[colname] = []
    data[colname].append(value)

def pretty_overhead_type(overhead_type, unit='us'):

    def with_unit(col):
        return "{col}_{us}".format(
            col=col, us=unit)

    if overhead_type == with_unit('total_pyprof_overhead'):
        return "Python overhead"
    elif overhead_type == with_unit('corrected_total_training_time'):
        return "Corrected training time"
    elif overhead_type == with_unit('total_cupti_overhead'):
        # return 'CUPTI GPU activities enabled'
        return 'CUPTI overhead'
    elif overhead_type == with_unit('total_interception_overhead'):
        # return 'CUPTI GPU activities disabled'
        return 'LD_PRELOAD interception overhead'
    else:
        return overhead_type


def add_stacked_bars(x, y, hue, label=None, data=None, ax=None, **kwargs):
    # sns.barplot(x=.., y=.., hue=..)

    # Q: Does order of "data" affect groups returned by groupby?
    if label is not None:
        groupby_cols = [hue, label]
    else:
        groupby_cols = [hue]
    data_groupby = data.groupby(groupby_cols)
    groups = [pair[0] for pair in list(data_groupby)]
    logging.info("groups: " + pprint_msg(groups))
    means = dict()
    stds = dict()
    for group, group_df in data_groupby:
        means[group] = group_df.groupby([x]).mean().reset_index()
        stds[group] = group_df.groupby([x]).std().reset_index()
    bottom = None
    for group in groups:
        xs = means[group][x]
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
        barplot = ax.bar(x=xs, height=ys, yerr=std, label=label_str, bottom=bottom, **kwargs)

        if bottom is None:
            bottom = ys
        else:
            bottom += ys

    # Reverse legend label order (when making stacked bar its in reverse order)
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title=None, loc='upper left')
    ax.legend(handles[::-1], labels[::-1])

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

def output_csv(plot_df, csv_path, sort_by=None):
    if sort_by is not None:
        plot_df = plot_df.sort_values(sort_by)
    plot_df.to_csv(csv_path, index=False)
    logging.info("{path}: {msg}".format(path=csv_path, msg=pprint_msg(plot_df)))
    logging.info("Output total csv @ {path}".format(path=csv_path))

def get_sns_kwargs():
    sns_kwargs = dict()
    sns_kwargs['capsize'] = 0.04
    sns_kwargs['errwidth'] = 1.25
    return sns_kwargs

def get_plt_kwargs():
    plt_kwargs = dict()
    plt_kwargs['capsize'] = 5
    return plt_kwargs

