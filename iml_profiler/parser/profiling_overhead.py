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

from iml_profiler.parser.dataframe import TrainingProgressDataframeReader, CUDAAPIStatsDataframeReader

from iml_profiler.parser import stacked_bar_plots
from iml_profiler.parser.db import SQLCategoryTimesReader, sql_input_path

from iml_profiler.profiler import iml_logging

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
        sns.barplot(x='field', y='interception_overhead_per_call_us', data=plot_data, ax=ax)
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
        sns.barplot(x='api_name', y='us_per_call', hue='pretty_config', data=plot_data, ax=ax)
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

def add_col(data, colname, value):
    if colname not in data:
        data[colname] = []
    data[colname].append(value)
