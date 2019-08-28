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

    def get_training_durations(self, directories):
        training_duration_us = []
        if type(directories) == str:
            directories = [directories]
        for directory in directories:
            df_reader = TrainingProgressDataframeReader(
                directory,
                # add_fields=self.maybe_add_algo_env,
                debug=self.debug)
            # df = df_reader.read()
            # df = df_reader.last_progress()
            duration_us = df_reader.training_duration_us()
            training_duration_us.append(duration_us)
        return training_duration_us

    def get_n_total_calls(self, directories):
        n_total_calls = []
        if type(directories) == str:
            directories = [directories]
        for directory in directories:
            df_reader = CUDAAPIStatsDataframeReader(
                directory,
                # add_fields=self.maybe_add_algo_env,
                debug=self.debug)
            n_total_calls.append(df_reader.n_total_calls())
        return n_total_calls

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
        int_training_duration_us = self.get_training_durations(self.interception_directory)
        no_int_training_duration_us = self.get_training_durations(self.no_interception_directory)
        if len(int_training_duration_us) != len(no_int_training_duration_us):
            raise RuntimeError("You need to run the same number of repetitions for both config_interception and config_no_interception")

        int_total_calls = self.get_n_total_calls(self.interception_directory)
        # no_int_total_calls = self.get_n_total_calls(self.no_interception_directory)
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


