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
        self.data = dict()
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

    def _read_df(self):
        self.df = pd.DataFrame(self.data)

    def read(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if self.is_proto_file(path):
                    self.add_proto_cols(path)
                    self._check_cols()
        self._read_df()
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
        if self.debug:
            logging.info("Read {name} from {path}".format(
                name=training_progress.__class__.__name__,
                path=path))
        self._add_columns(colnames, training_progress)
        self._maybe_add_fields(path)
