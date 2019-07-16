from iml_profiler.protobuf.pyprof_pb2 import Pyprof, MachineUtilization, DeviceUtilization, UtilizationSample
from iml_profiler.parser.common import *
from iml_profiler.profiler import experiment
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
import pandas as pd

def protobuf_to_dict(pb):
    return dict((field.name, value) for field, value in pb.ListFields())

class UtilDataframeReader:
    """
    Read machine_util.trace_*.proto files into data-frame.

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

    def _check_cols(self):
        col_to_length = dict((col, len(self.data[col])) for col in self.data.keys())
        if len(set(col_to_length.values())) > 1:
            raise RuntimeError("Detected inconsistent column lengths:\n{dic}\n{data}".format(
                dic=pprint.pformat(col_to_length),
                data=pprint.pformat(self.data),
            ))

    def _read_machine_util(self, path):
        machine_util = read_machine_util_file(path)
        if self.debug:
            logging.info("Read MachineUtilization from {path}".format(path=path))
        for device_name, device_utilization in machine_util.device_util.items():
            for sample in device_utilization.samples:
                self._add_col('machine_name', machine_util.machine_name)
                self._add_col('device_name', device_name)

                self._add_col('util', sample.util)
                self._add_col('total_resident_memory_bytes', sample.total_resident_memory_bytes)

                # sample_fields = protobuf_to_dict(sample)
                # for key, value in sample_fields.items():
                #     self._add_col(key, value)

                if self.add_fields is not None:
                    extra_fields = self.add_fields(path)
                    if extra_fields is not None:
                        self.added_fields.update(extra_fields.keys())
                        for key, value in extra_fields.items():
                            self._add_col(key, value)
        self._check_cols()

    def _read_df(self):
        self.df = pd.DataFrame(self.data)

    def read(self):
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if is_machine_util_file(path):
                    self._read_machine_util(path)
        self._read_df()
        return self.df

class UtilParser:
    """
    Given a directory containing machine_util.trace_*.proto files, output a
    csv file containing CPU/GPU/memory utilization info useful for plotting.

    GOAL: Show that single-machines cannot scale to utilize the entire GPU.

    1. Memory utilization:
       - Show that memory gets used up before GPU is fully utilized
         - X-axis = number of total bytes memory used, OR
                    % of total machine memory used
         - Y-axis = Average GPU utilization
         - Plot: y_gpu_util.x_mem_util.png


    2. CPU utilization:
       - Show that CPU gets used up before GPU is fully utilized:
         - X-axis = average CPU utilization (a number in 0..1), OR
                    absolute CPU utilization (2 cores would be [0%...200%])
         - Y-axis = Average GPU utilization
       - Plot: y_gpu_util.x_cpu_util.png

                      Training {env} using {algo}

                      100 |
                          |
                          |              ----
      GPU utilization (%) |      -----...
                          | -----
                        0 |--------------------
                            1   2   3 ... N

                            Number of parallel
                             inference workers

    IDEALLY:
    - (algo, env) would be something citable that people have scaled up in the past
    - e.g. nature papers
      - (minigo, DQN)
      - (Atari, DQN)

    Data-frame should be like:

    1. Memory utilization:
       - algo, env, num_workers, gpu_util, mem_util
       - NOTE: For gpu_util we include:
         - gpu_util_min
         - gpu_util_max
         - gpu_util_avg
       - Same for mem_util

    2. CPU utilization:
       - algo, env, num_workers, gpu_util, cpu_util

    In addition to utilization stuff, each directory should record:
    - algo
    - env
    - num_workers

    We can record this in a JSON file experiment_config.json.
    {
        'expr_type': 'OverallMachineUtilization',
        'algo': ...
        'env': ...
        'num_workers': ...
    }

    TODO:
    1. Read/output raw data-frame with all the data:
       - algo, env, num_workers, gpu_util, mem_util
    2. Read/output pre-processed data-frame (min/max/avg), 1 for each plot.
    """
    def __init__(self,
                 directory,
                 iml_directories,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.directory = directory
        self.iml_directories = iml_directories
        self.debug = debug

        self.added_fields = set()

    @property
    def _raw_csv_path(self):
        return _j(self.directory, "overall_machine_util.raw.csv")

    @property
    def _agg_csv_path(self):
        return _j(self.directory, "overall_machine_util.agg.csv")

    def _json_path(self, device_id, device_name):
        return _j(self.directory, "util_scale{dev}.js_path.json".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def add_experiment_config(self, machine_util_path):
        """
        add_fields(machine_util_path)

        We expect to find experiment_config.json where the machine_util.*.proto files live.

        :param machine_util_path:
        :return:
        """
        assert is_machine_util_file(machine_util_path)
        directory = _d(machine_util_path)
        path = experiment.experiment_config_path(directory)
        if not _e(path):
            if self.debug:
                logging.info("Didn't find {path}; skip adding experiment columns to csv".format(path=path))
            return None
        data = experiment.load_experiment_config(directory)
        return data

    def flattened_agg_df(self, df):
        """
        :param df:
            The result of a df.groupby([...]).agg([...])
        :return:
        """
        # https://stackoverflow.com/questions/19078325/naming-returned-columns-in-pandas-aggregate-function
        df = df.reset_index()
        old_cols = df.columns.ravel()
        def get_new_col(col_agg):
            col, agg = col_agg
            if agg == '':
                return col
            return '{col}_{agg}'.format(col=col, agg=agg)
        new_cols = [get_new_col(col_agg) for col_agg in df.columns.ravel()]
        new_df_data = dict()
        for old_col, new_col in zip(old_cols, new_cols):
            new_df_data[new_col] = df[old_col]
        new_df = pd.DataFrame(new_df_data)
        return new_df

    def run(self):
        dfs = []
        for directory in self.iml_directories:
            df_reader = UtilDataframeReader(
                directory,
                add_fields=self.add_experiment_config,
                debug=self.debug)
            df = df_reader.read()
            self.added_fields.update(df_reader.added_fields)
            dfs.append(df)
        df = pd.concat(dfs)

        logging.info("Output raw un-aggregated machine utilization data @ {path}".format(path=self._raw_csv_path))
        df.to_csv(self._raw_csv_path, index=False)

        # 1. Memory utilization:
        # 2. CPU utilization:
        groupby_cols = sorted(self.added_fields) + ['machine_name', 'device_name']
        df_agg = df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        flat_df_agg = self.flattened_agg_df(df_agg)
        # import ipdb; ipdb.set_trace()
        logging.info("Output min/max/std aggregated machine utilization data @ {path}".format(path=self._agg_csv_path))
        flat_df_agg.to_csv(self._agg_csv_path, index=False)
