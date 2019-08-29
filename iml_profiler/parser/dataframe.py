import pandas as pd
import os
import pprint
import logging

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.parser.common import *

class BaseDataframeReader:
    """
    Read training_progress.trace_*.proto files into data-frame.

    Q: How should read experiment_config.json?
       Can we add something that read additional dict-attributes for each directory?
    """
    def __init__(self, directory, add_fields=None, debug=False):
        self.directory = directory
        # column-name -> [values]
        self.data = None
        self.df = None
        # add_fields(iml_directory) -> dict of fields to add to data-frame
        self.add_fields = add_fields
        self.added_fields = set()
        self.debug = debug

    def _add_col(self, colname, value):
        if colname not in self.data:
            self.data[colname] = []
        self.data[colname].append(value)

    def _add_columns(self, colnames, proto):
        for col in colnames:
            val = getattr(proto, col)
            self._add_col(col, val)

    def _check_cols(self):
        col_to_length = dict((col, len(self.data[col])) for col in self.data.keys())
        if len(set(col_to_length.values())) > 1:
            raise RuntimeError("Detected inconsistent column lengths:\n{dic}\n{data}".format(
                dic=pprint.pformat(col_to_length),
                data=pprint.pformat(self.data),
            ))

    def is_proto_file(self, path):
        raise NotImplementedError()

    def add_proto_cols(self, path):
        raise NotImplementedError()

    def _maybe_add_fields(self, path):
        if self.add_fields is not None:
            extra_fields = self.add_fields(path)
            if extra_fields is not None:
                self.added_fields.update(extra_fields.keys())
                for key, value in extra_fields.items():
                    self._add_col(key, value)

    # def _read_df(self):
    #     self.df = pd.DataFrame(self.data)

    def _read_data(self):
        self.data = dict()
        proto_paths = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                # if self.debug:
                #     logging.info("{klass}.is_proto_file(path={path}) = {ret}".format(
                #         klass=self.__class__.__name__,
                #         path=path,
                #         ret=self.is_proto_file(path),
                #     ))
                if self.is_proto_file(path):
                    proto_paths.append(path)
                    # if self.debug:
                    #     logging.info("{klass}: Add proto cols: path = {path}".format(
                    #         klass=self.__class__.__name__,
                    #         path=path))
                    self.add_proto_cols(path)
                    self._check_cols()
        if len(proto_paths) == 0:
            raise RuntimeError("Saw 0 proto paths rooted at {dir}".format(
                dir=self.directory,
            ))

    def read(self):
        if self.df is None:
            self._read_data()
            self.df = pd.DataFrame(self.data)
        return self.df

class UtilDataframeReader(BaseDataframeReader):

    def __init__(self, directory, add_fields=None, debug=False):
        super().__init__(directory, add_fields=add_fields, debug=debug)

    def is_proto_file(self, path):
        return is_machine_util_file(path)

    def add_proto_cols(self, path):
        machine_util = read_machine_util_file(path)
        if self.debug:
            logging.info("Read MachineUtilization from {path}".format(path=path))
        for device_name, device_utilization in machine_util.device_util.items():
            for sample in device_utilization.samples:
                self._add_col('machine_name', machine_util.machine_name)
                self._add_col('device_name', device_name)

                self._add_col('util', sample.util)
                self._add_col('start_time_us', sample.start_time_us)
                self._add_col('total_resident_memory_bytes', sample.total_resident_memory_bytes)

                self._maybe_add_fields(path)

class TrainingProgressDataframeReader(BaseDataframeReader):

    def __init__(self, directory, add_fields=None, debug=False):
        super().__init__(directory, add_fields=add_fields, debug=debug)

    def is_proto_file(self, path):
        return is_training_progress_file(path)

    def add_proto_cols(self, path):
        training_progress = read_training_progress_file(path)
        colnames = [
            'process_name',
            'phase',
            'machine_name',
            'total_timesteps',
            'start_trace_time_us',
            'start_percent_complete',
            'start_num_timesteps',
            'start_training_time_us',
            'end_percent_complete',
            'end_training_time_us',
            'end_num_timesteps',
        ]
        # if self.debug:
        #     logging.info("Read {name} from {path}".format(
        #         name=training_progress.__class__.__name__,
        #         path=path))
        self._add_columns(colnames, training_progress)
        self._maybe_add_fields(path)

    def last_progress(self):
        df = self.read()
        last_df = ( df[df['end_training_time_us'] == np.max(df['end_training_time_us'])] )
        return last_df

    def training_duration_us(self):
        df = self.last_progress()
        training_duration_us = df['end_training_time_us'] - df['start_training_time_us']
        assert len(training_duration_us) == 1
        training_duration_us = training_duration_us.values[0]
        return training_duration_us

class CUDAAPIStatsDataframeReader(BaseDataframeReader):
    def __init__(self, directory, add_fields=None, debug=False):
        super().__init__(directory, add_fields=add_fields, debug=debug)

    def is_proto_file(self, path):
        return is_cuda_api_stats_file(path)

    def add_proto_cols(self, path):
        proto = read_cuda_api_stats_file(path)
        # if self.debug:
        #     logging.info("Read CUDAAPIPhaseStatsProto from {path}".format(path=path))

        for api_thread_stats in proto.stats:
            self._add_col('process_name', proto.process_name)
            self._add_col('machine_name', proto.machine_name)
            self._add_col('phase_name', proto.phase)

            self._add_col('tid', api_thread_stats.tid)
            self._add_col('api_name', api_thread_stats.api_name)
            self._add_col('total_time_us', api_thread_stats.total_time_us)
            self._add_col('num_calls', api_thread_stats.num_calls)

            self._maybe_add_fields(path)

    def n_total_calls(self):
        df = self.read()
        n_total_calls = np.sum(df['num_calls'])
        return n_total_calls

    def per_api_stats(self):
        """
        Group all CUDA API calls by CUDA API name.
        # e.g. cudaKernelLaunch
        - api_name
        # How much total time (us) was spent in this API call over the course of the program?
        - total_time_us
        # How many times was this API call over the course of the program?
        - num_calls
        :return:
        """
        df = self.read()
        groupby_cols = ['api_name']
        keep_cols = sorted(set(groupby_cols + ['total_time_us', 'num_calls']))
        df_keep = df[keep_cols]
        groupby = df_keep.groupby(groupby_cols)
        df_sum = groupby.sum()
        return df_sum

class PyprofDataframeReader(BaseDataframeReader):
    def __init__(self, directory, add_fields=None, debug=False):
        super().__init__(directory, add_fields=add_fields, debug=debug)

    def is_proto_file(self, path):
        return is_pyprof_file(path)

    def add_proto_cols(self, path):
        proto = read_pyprof_file(path)
        # if self.debug:
        #     logging.info("Read Pyprof from {path}".format(path=path))

        # Event from pyprof.proto
        event_colnames = [
            'thread_id',
            'start_time_us',
            'duration_us',
            'start_profiling_overhead_us',
            'duration_profiling_overhead_us',
            'name',
        ]

        def add_event(category, event):
            self._add_col('process_name', proto.process_name)
            self._add_col('machine_name', proto.machine_name)
            self._add_col('phase_name', proto.phase)

            self._add_col('category', category)

            self._add_columns(event_colnames, event)

            self._maybe_add_fields(path)

        for step, python_events in proto.python_events.items():
            for event in python_events.events:
                add_event(CATEGORY_PYTHON, event)

        for step, clibs in proto.clibs.items():
            for category, category_clibs in clibs.clibs.items():
                for event in category_clibs.events:
                    add_event(category, event)

    def total_pyprof_overhead_us(self):
        df = self.read()
        # Filter for events that have profiling overhead recorded.
        df = df[df['start_profiling_overhead_us'] != 0]
        total_pyprof_overhead_us = np.sum(df['duration_profiling_overhead_us'])
        return total_pyprof_overhead_us

