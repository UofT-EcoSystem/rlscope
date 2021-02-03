"""
Plotting GPU utilization over time.
"""
from rlscope.profiler.rlscope_logging import logger
import copy
import itertools
import subprocess
import argparse
import csv


from rlscope.profiler.util import pprint_msg
from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization, DeviceUtilization, UtilizationSample
from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.parser.overlap_result import from_js
from rlscope.profiler import experiment
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
import pandas as pd
import seaborn as sns

from rlscope.parser.dataframe import UtilDataframeReader

from rlscope.parser.plot_utils import setup_matplotlib
setup_matplotlib()
from matplotlib import pyplot as plt
import matplotlib.gridspec

from rlscope.parser import stacked_bar_plots

from rlscope.profiler.rlscope_logging import logger

from rlscope.experiment import expr_config
from rlscope.parser.plot import CUDAEventCSVReader, fix_seaborn_legend

import multiprocessing
from rlscope.profiler.concurrent import map_pool
from concurrent.futures import ProcessPoolExecutor

FLOAT_RE = r'(?:[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'

def protobuf_to_dict(pb):
    return dict((field.name, value) for field, value in pb.ListFields())

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
                 rlscope_directories,
                 algo_env_from_dir=False,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.directory = directory
        self.rlscope_directories = rlscope_directories
        self.algo_env_from_dir = algo_env_from_dir
        self.debug = debug

        self.added_fields = set()

    @staticmethod
    def is_cpu(device_name):
        if re.search(r'\b(Intel|Xeon|CPU|AMD)\b', device_name):
            return True
        return False

    @staticmethod
    def is_gpu(device_name):
        return not UtilParser.is_cpu(device_name)

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
                logger.info("Didn't find {path}; skip adding experiment columns to csv".format(path=path))
            return None
        data = experiment.load_experiment_config(directory)
        return data

    def maybe_add_algo_env(self, machine_util_path):
        assert is_machine_util_file(machine_util_path)

        rlscope_directory = _d(machine_util_path)

        if self.algo_env_from_dir:
            return self.add_algo_env_from_dir(machine_util_path)
        if not _e(experiment.experiment_config_path(rlscope_directory)):
            return self.add_experiment_config(machine_util_path)

        # Not sure what (algo, env) is; don't add those columns.
        return None

    def add_algo_env_from_dir(self, machine_util_path):
        assert is_machine_util_file(machine_util_path)
        rlscope_dir = _d(machine_util_path)

        path = os.path.normpath(rlscope_dir)
        components = path.split(os.sep)
        env_id = components[-1]
        algo = components[-2]
        fields = {
            'algo': algo,
            'env_id': env_id,
            'env': env_id,
        }
        return fields

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
        for directory in self.rlscope_directories:
            df_reader = UtilDataframeReader(
                directory,
                add_fields=self.maybe_add_algo_env,
                debug=self.debug)
            df = df_reader.read()
            self.added_fields.update(df_reader.added_fields)
            dfs.append(df)
        df = pd.concat(dfs)

        # 1. Memory utilization:
        # 2. CPU utilization:
        groupby_cols = sorted(self.added_fields) + ['machine_name', 'device_name']

        # df_agg = df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        # flat_df_agg = self.flattened_agg_df(df_agg)

        # - Use start_time_us timestamp to assign each utilization sample an "index" number from [0...1];
        #   this is trace_percent: the percent time into the collected trace
        # - Group by (algo, env)
        #   - Reduce:
        #     # for each group member, divide by group-max
        #     - max_time = max(row['start_time_us'])
        #     - min_time = min(row['start_time_us'])
        #     - row['trace_percent'] = (row['start_time_us'] - min_time)/max_time
        # TODO: debug this to see if it works.
        dfs = []
        groupby = df.groupby(groupby_cols)
        for group, df_group in groupby:

            max_time = max(df_group['start_time_us'])
            start_time = min(df_group['start_time_us'])
            length_time = max_time - start_time
            df_group['trace_percent'] = (df_group['start_time_us'] - start_time) / length_time
            dfs.append(df_group)

            logger.info(pprint_msg({
                'group': group,
                'start_time': start_time,
                'max_time': max_time,
            }))
            logger.info(pprint_msg(df_group))


        new_df = pd.concat(dfs)
        def cpu_or_gpu(device_name):
            if UtilParser.is_cpu(device_name):
                return 'CPU'
            return 'GPU'
        new_df['device_type'] = new_df['device_name'].apply(cpu_or_gpu)
        def used_by_tensorflow(CUDA_VISIBLE_DEVICES, device_id, device_type):
            if device_type == 'CPU':
                return True
            if device_type == 'GPU':
                return device_id in CUDA_VISIBLE_DEVICES
            # Not handled.
            raise NotImplementedError()
        new_df['used_by_tensorflow'] = np.vectorize(used_by_tensorflow, otypes=[np.bool])(
            new_df['CUDA_VISIBLE_DEVICES'],
            new_df['device_id'],
            new_df['device_type'])

        # OUTPUT raw thing here.
        logger.info("Output raw un-aggregated machine utilization data @ {path}".format(path=self._raw_csv_path))
        new_df.to_csv(self._raw_csv_path, index=False)

        df_agg = new_df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        flat_df_agg = self.flattened_agg_df(df_agg)

        logger.info("Output min/max/std aggregated machine utilization data @ {path}".format(path=self._agg_csv_path))
        flat_df_agg.to_csv(self._agg_csv_path, index=False)

        # Q: Which (algo, env) have at least one utilization readings > 0 a GPU whose device_id > 0?

        util_plot = UtilPlot(
            csv=self._raw_csv_path,
            directory=self.directory,
            x_type='rl-comparison',
            debug=self.debug,
        )
        util_plot.run()


class NvprofCSVParser:
    """
    Parse output from:
    $ nvprof --csv -i profile.nvprof ...
    """
    def __init__(self,
                 nvprof_file,
                 csv_file,
                 nvprof_csv_opts,
                 debug=False,
                 debug_single_thread=False,
                 ):
        """

        :param nvprof_file:
            Raw nvprof file from running:
            $ nvprof -o profile.nvprof
        :param nvprof_csv_opts:
            Command line parameters to pass to:
                $ nvprof --csv -i <csv_file> <nvprof_csv_opts...>
        :param csv_file:
            CSV file from processing nvprof file using:
            $ nvprof --csv -i profile.nvprof ...
        # :param skip_line_regexes
        #     List of regexes for lines we should ignore at the start of the "nvprof --csv" output.
        # :param warning_line_regexes
        #     List of regexes for lines we should warn about at the start of the "nvprof --csv" output (but skip)

            NOTE: any other lines will cause an ERROR in pandas.read_csv, since we'll assume they're input.
        """
        self.nvprof_file = nvprof_file
        self.csv_file = csv_file
        self.nvprof_csv_opts = nvprof_csv_opts
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def _read_row(self, x):
        line = None
        if type(x) == str:
            line = x
        else:
            line = x.readline().rstrip()
        reader = csv.reader([line])
        rows = [row for row in reader]
        assert len(rows) == 1
        row = rows[0]
        return row

    def _is_empty_file(self, path):
        filesize = os.path.getsize(path)
        return filesize == 0

    def write_csv(self, skip_if_exists=True):
        if not skip_if_exists or not _e(self.csv_file) or self._is_empty_file(self.csv_file):
            with open(self.csv_file, 'w') as f:
                # This takes really long...
                # For a relatively short 325MB file collected from 32 games in minigo, it takes 565 seconds (10 minutes!).
                # real    565.79s
                # user    564.22s
                # sys     1.03s
                start_nvprof_t = time.time()
                proc = subprocess.Popen(
                    ["nvprof", "-i", self.nvprof_file, "--csv", "--normalized-time-unit", "us"] + self.nvprof_csv_opts,
                    stdout=f,
                    stderr=subprocess.STDOUT)
                ret = proc.wait()
                end_nvprof_t = time.time()
                assert ret == 0
                logger.info("Running \"nvprof --csv -i {path}\" took {sec} sec".format(
                    path=self.nvprof_file,
                    sec=end_nvprof_t - start_nvprof_t))

    def parse(self, read_csv_kwargs):
        """
        "nvprof --csv" output looks like ():

            1. ======== Warning: 376746 records have invalid timestamps due to insufficient device buffer space. You can configure the buffer space using the option --device-buffer-size.
            2. ======== Profiling result:
            3. "Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","SrcMemType","DstMemType","Device","Context","Stream","Name","Correlation_ID"
            4. us,us,,,,,,,,KB,KB,KB,GB/s,,,,,,,
            5. 0.000000,1.568000,,,,,,,,,,1.003906,0.610586,"Device",,"GeForce RTX 2080 Ti (0)","1","7","[CUDA memset]",274

        NOTE: line numbers above aren't present in output

        NOTE:
        - line 1 is a warning line
        - line 2 is skipped
        - line 3 is a header line (first unskipped line)
        - line 4 is a units line (second unskipped line)
        - line >= 5 are input lines

        :param read_csv_kwargs:
            Extra arguments to pass to pandas.read_csv
        :return:
        """
        self.write_csv()
        with open(self.csv_file, 'r') as f:
            num_skip_lines = 0
            num_other_lines = 0
            for lineno, line in enumerate(f):
                line = line.rstrip()
                m = re.search(r'^=+\s+(?P<msg>.*)', line)
                if m:
                    msg = m.group('msg')
                    if re.search(r'Warning', msg, flags=re.IGNORECASE):
                        logger.warning("Saw WARNING in {path} at line {lineno}:\n{warning}".format(
                            path=self.csv_file,
                            lineno=lineno,
                            warning=textwrap.indent(line, prefix="  ")))
                else:
                    if num_other_lines == 0:
                        self.header = self._read_row(line)
                    elif num_other_lines == 1:
                        self.units = self._read_row(line)
                    else:
                        # We've encountered the first data line.
                        break
                    num_other_lines += 1

                num_skip_lines += 1

        with open(self.csv_file, 'r') as f:
            for i in range(num_skip_lines):
                f.readline()
            start_read_csv_t = time.time()
            # usecols=[0, 1, 18],
            # header=0,
            # names=["start_us", "duration_us", "name"])
            read_csv_kwargs = dict(read_csv_kwargs)
            if 'names' not in read_csv_kwargs:
                # If user doesn't specify custom header names,
                # use nvprof's header names as labels.
                read_csv_kwargs['names'] = list(self.header)
                if 'usecols' in read_csv_kwargs:
                    # Only keep nvprof header names for the corresponding columns the user is selecting.
                    read_csv_kwargs['names'] = [read_csv_kwargs['names'][i] for i in read_csv_kwargs['usecols']]
            df = pd.read_csv(
                f,
                index_col=False,
                **read_csv_kwargs)
            end_read_csv_t = time.time()
            logger.info("Reading {path} took {sec} sec".format(
                path=self.csv_file,
                sec=end_read_csv_t - start_read_csv_t))
            if self.debug:
                logger.info(
                    textwrap.dedent("""\
                    Read nvprof file {path} into dataframe:
                      Header: {header}
                      Units: {header}
                      Dataframe:
                    {df}
                    """).format(
                        path=self.nvprof_file,
                        header=read_csv_kwargs['names'],
                        units=self.units,
                        df=textwrap.indent(str(df), prefix="    "),
                    ))
        return df

NVPROF_HIST_DEFAULT_FMT = "{y:.2f}"
# NVPROF_HIST_INT_FMT = "{y:d}"

def nvprof_fmt_int(y):
    return "{y:d}".format(y=int(np.round(y)))



class CrossProcessOverlapHistogram:
    def __init__(self,
                 cross_process_overlap,
                 # discard_top_percentile=None,
                 # discard_bottom_percentile=None,
                 # discard_iqr_whiskers=False,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.cross_process_overlap = cross_process_overlap
        # self.discard_top_percentile = discard_top_percentile
        # self.discard_bottom_percentile = discard_bottom_percentile
        # self.discard_iqr_whiskers = discard_iqr_whiskers
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def parse_js(self):
        js = load_json(self.cross_process_overlap)
        overlap_result_obj = from_js(js)
        overlap_result = dict()
        for key, value in overlap_result_obj['overlap'].items():
            overlap_result[frozenset(key)] = value
        return overlap_result

    def read_df(self):
        # js = load_json(self.cross_process_overlap)
        overlap_result = self.parse_js()
        pprint.pprint(overlap_result)
        gpu_time_sec = dict()
        for overlap_key, time_sec in overlap_result.items():
            kernels_running = 0
            for category_key in overlap_key:
                if constants.CATEGORY_GPU in category_key.non_ops:
                    kernels_running += 1
            if kernels_running not in gpu_time_sec:
                gpu_time_sec[kernels_running] = 0
            gpu_time_sec[kernels_running] += time_sec
        data = {
            'time_sec': [],
            'kernels_running': [],
        }
        for kernels_running, time_sec in gpu_time_sec.items():
            data['time_sec'].append(time_sec)
            data['kernels_running'].append(kernels_running)
        df = pd.DataFrame(data=data)
        df['percent'] = df['time_sec'] / df['time_sec'].sum()
        DataFrame.print_df(df)

        """
        We want to read a dataframe of this format:
        
            kernels_running  time_sec  percent_time
            0                ...       ...
            1
            2
            ...
            16
            
        When 2 kernels are running, that means we have an overlap key that looks like:
          {
              CategoryKey(procs={1234}, non_ops={constants.CATEGORY_GPU},
              CategoryKey(procs={4567}, non_ops={constants.CATEGORY_GPU},
          } 
        """

    def run(self):
        self.read_df()

class NvprofTraces:
    UNIT_RE = r'ns|ms|sec|s'

    def __init__(self,
                 directory,
                 n_workers=None,
                 force=False,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.n_workers = n_workers
        self.force = force
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    @staticmethod
    def nvprof_print_gpu_trace(self, nvprof_file):
        # raise RuntimeError("FAIL1")
        nvprof_parser = NvprofCSVParser(
            nvprof_file=nvprof_file,
            csv_file=self._nvprof_gpu_trace_csv(nvprof_file),
            nvprof_csv_opts=["--print-gpu-trace"],
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        nvprof_parser.write_csv(skip_if_exists=not self.force)

    @staticmethod
    def nvprof_print_api_trace(self, nvprof_file):
        # raise RuntimeError("FAIL2")
        nvprof_parser = NvprofCSVParser(
            nvprof_file=nvprof_file,
            csv_file=self._nvprof_api_trace_csv(nvprof_file),
            nvprof_csv_opts=["--print-api-trace"],
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        nvprof_parser.write_csv(skip_if_exists=not self.force)

    def run(self):
        nvprof_files = [path for path in each_file_recursive(self.directory) if is_nvprof_file(path)]
        logger.info("nvprof_files = {nvprof_files}".format(nvprof_files=pprint_msg(nvprof_files)))
        with ProcessPoolExecutor() as pool:
            results = []
            for nvprof_file in nvprof_files:
                if not self.debug_single_thread:
                    results.append(pool.submit(self.nvprof_print_gpu_trace, self, nvprof_file))
                    results.append(pool.submit(self.nvprof_print_api_trace, self, nvprof_file))
                else:
                    self.nvprof_print_gpu_trace(nvprof_file)
                    self.nvprof_print_api_trace(nvprof_file)
            for result in results:
                # If exception was raised, it will be re-raised here.
                result.result()

    def _nvprof_api_trace_csv(self, nvprof_file):
        return "{path}.api_trace.csv".format(path=nvprof_file)

    def _nvprof_api_summary_csv(self, nvprof_file):
        return "{path}.api_summary.csv".format(path=nvprof_file)

    def _nvprof_gpu_trace_csv(self, nvprof_file):
        return "{path}.gpu_trace.csv".format(path=nvprof_file)

    def _nvprof_gpu_summary_csv(self, nvprof_file):
        return "{path}.gpu_summary.csv".format(path=nvprof_file)

class NvprofKernelHistogram:
    UNIT_RE = r'ns|ms|sec|s'

    def __init__(self,
                 nvprof_file,
                 discard_top_percentile=None,
                 discard_bottom_percentile=None,
                 discard_iqr_whiskers=False,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.nvprof_file = nvprof_file
        self.discard_top_percentile = discard_top_percentile
        self.discard_bottom_percentile = discard_bottom_percentile
        self.discard_iqr_whiskers = discard_iqr_whiskers
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def _parse_human_time_us(self, txt, expect_match=True):
        m = re.search(r'^(?P<value>{float})(?P<unit>{unit})$'.format(
            float=FLOAT_RE,
            unit=NvprofKernelHistogram.UNIT_RE),
            txt)
        if not m:
            assert not expect_match
            return None
        value = float(m.group('value'))
        unit = m.group('unit')
        if unit == 'ns':
            return value / 1e3
        elif unit == 'us':
            return value
        elif unit == 'ms':
            return value * 1e3
        elif unit in {'s', 'sec'}:
            return value * 1e6
        else:
            raise NotImplementedError()

    def _nvprof_api_trace_csv(self, nvprof_file):
        return "{path}.api_trace.csv".format(path=nvprof_file)

    def _nvprof_api_summary_csv(self, nvprof_file):
        return "{path}.api_summary.csv".format(path=nvprof_file)

    def _nvprof_gpu_trace_csv(self, nvprof_file):
        return "{path}.gpu_trace.csv".format(path=nvprof_file)

    def _nvprof_gpu_summary_csv(self, nvprof_file):
        return "{path}.gpu_summary.csv".format(path=nvprof_file)

    def add_bar_labels(self, ax, scale='linear', fmt=NVPROF_HIST_DEFAULT_FMT):
        """
        :param ax:
        :param scale:
        :param fmt:
            Format string, where "y" is the variable to format.
            Default: "{y:.2f}"
            i.e. 2 decimal places:
            2.123123 => 2.12
        :return:
        """
        # set individual bar lables using above list
        max_height = max(i.get_height() for i in ax.patches)
        min_y, max_y = ax.get_ylim()
        if scale == 'log':
            ax.set_ylim(min_y, 1.10*max_y)
        else:
            ax.set_ylim(min_y, 1.025*max_y)
        for i in ax.patches:
            if i.get_height() == 0:
                continue
            # get_x pulls left or right; get_height pushes up or down
            if scale == 'log':
                x = i.get_x()
                y = i.get_height() + 0.10*i.get_height()
            else:
                x = i.get_x()
                y = i.get_height() + 0.025*max_height
            if type(fmt) == str:
                label = fmt.format(y=i.get_height())
            else:
                label = fmt(i.get_height())
            ax.text(x, y, label)

    def plot_binned_df(self, binned_df, y, xlabel, ylabel, img_path=None, fmt=NVPROF_HIST_DEFAULT_FMT):
        ax = binned_df.plot.bar(x='hist_label', y=y, rot=45, logy=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_ylabel("Number of kernel invocations")
        # ax.set_title("Histogram of minigo worker kernel durations")
        ax.get_legend().remove()
        self.add_bar_labels(ax, scale='log', fmt=fmt)
        if img_path is not None:
            plt.tight_layout()
            plt.savefig(img_path)
        return ax


    def plot_binned_gpu_kernels(self, binned_df, suffix, fmt=NVPROF_HIST_DEFAULT_FMT):
        xlabel = r"Kernel duration ($\mu s$)"
        self.plot_binned_df(binned_df, y='total_sec', xlabel=xlabel, ylabel="Total GPU time (sec)", img_path=self.get_img_path(suffix, "total_sec"), fmt=fmt)
        self.plot_binned_df(binned_df, y='percent_time', xlabel=xlabel, ylabel="Percent total GPU time (%)", img_path=self.get_img_path(suffix, "percent_time"), fmt=fmt)
        self.plot_binned_df(binned_df, y='total_invocations', xlabel=xlabel, ylabel="Kernel invocations", img_path=self.get_img_path(suffix, "total_invocations"), fmt=nvprof_fmt_int)
        self.plot_binned_df(binned_df, y='percent_invocations', xlabel=xlabel, ylabel="Percent total kernel invocations (%)", img_path=self.get_img_path(suffix, "percent_invocations"), fmt=fmt)

    def plot_binned_cudaLaunchKernel(self, binned_df, suffix, fmt=NVPROF_HIST_DEFAULT_FMT):
        xlabel = r"Duration of cudaLaunchKernel calls ($\mu s$)"
        self.plot_binned_df(binned_df, y='total_sec', xlabel=xlabel, ylabel="Total GPU time (sec)", img_path=self.get_img_path(suffix, "total_sec"), fmt=fmt)
        self.plot_binned_df(binned_df, y='percent_time', xlabel=xlabel, ylabel="Percent total GPU time (%)", img_path=self.get_img_path(suffix, "percent_time"), fmt=fmt)
        self.plot_binned_df(binned_df, y='total_invocations', xlabel=xlabel, ylabel="Kernel invocations", img_path=self.get_img_path(suffix, "total_invocations"), fmt=nvprof_fmt_int)
        self.plot_binned_df(binned_df, y='percent_invocations', xlabel=xlabel, ylabel="Percent total kernel invocations (%)", img_path=self.get_img_path(suffix, "percent_invocations"), fmt=fmt)

    def plot_binned_delay(self, binned_df, suffix, fmt=NVPROF_HIST_DEFAULT_FMT):
        xlabel = r"Delay between cudaLaunchKernel calls ($\mu s$)"
        self.plot_binned_df(binned_df, y='total_sec', xlabel=xlabel, ylabel="Total CUDA API time (sec)", img_path=self.get_img_path(suffix, "total_sec"), fmt=fmt)
        self.plot_binned_df(binned_df, y='percent_time', xlabel=xlabel, ylabel="Percent total CUDA API time (%)", img_path=self.get_img_path(suffix, "percent_time"), fmt=fmt)
        self.plot_binned_df(binned_df, y='total_invocations', xlabel=xlabel, ylabel="cudaLaunchKernel invocations", img_path=self.get_img_path(suffix, "total_invocations"), fmt=nvprof_fmt_int)
        self.plot_binned_df(binned_df, y='percent_invocations', xlabel=xlabel, ylabel="Percent total cudaLaunchKernel invocations (%)", img_path=self.get_img_path(suffix, "percent_invocations"), fmt=fmt)

    def get_img_path(self, suffix, y):
        return "{path}.minigo_kernel_histograms.{suffix}.{y}.svg".format(
            suffix=suffix,
            path=self.nvprof_file,
            y=y)

    def hist_df_from_summary(self, summary_df, duration_field='avg_duration_us', calls_field='num_calls'):
        hist_df = pd.DataFrame({
            duration_field: np.repeat(summary_df[duration_field], summary_df[calls_field])
        })
        return hist_df

    def hist_df_from_trace(self, trace_df, duration_field='avg_duration_us'):
        """
        Just keep the 'duration_us' column form the raw trace dataframe.

        :param trace_df:
            Raw trace dataframe
        :param duration_field:
            e.g. 'duration_us'
        :return:
        """
        return trace_df[[duration_field]]

    def delay_df_from_trace(self, trace_df):
        """
        Just keep the 'duration_us' column form the raw trace dataframe.

        :param trace_df:
            Raw trace dataframe
        :param duration_field:
            e.g. 'duration_us'
        :return:
        """
        trace_df['end_us'] = trace_df['start_us'] + trace_df['duration_us']
        trace_df.sort_values(['start_us', 'end_us'], inplace=True)
        delay_us = trace_df['start_us'][1:].values - trace_df['end_us'][:len(trace_df)-1].values
        delay_non_neg_us = delay_us[delay_us >= 0]
        # report_delay_us = delay_non_neg_us
        # Q: What does this mean...?  The number of consecutive kernels that overlap.
        delay_neg_us = delay_us[delay_us < 0]
        # For a single process/thread, as long as cudaLaunchKernel is done from a single thread,
        # overlapping cudaLaunchKernel's shouldn't happen.
        assert len(delay_neg_us) == 0
        # num_overlapped = len(delay_neg_us)
        # mean_delay_neg_us =  delay_neg_us.mean()
        # std_delay_neg_us = delay_neg_us.std()
        #
        # mean_delay_us =  report_delay_us.mean()
        # std_delay_us = report_delay_us.std()
        # num_delay_us = len(report_delay_us)
        # mean_duration_us = trace_df['duration_us'].mean()
        # std_duration_us = trace_df['duration_us'].std()
        delay_df = pd.DataFrame({
            'delay_us': delay_non_neg_us,
        })
        return delay_df

    def binned_df_from_hist(self, hist_df, n_bins=10, duration_field='avg_duration_us'):
        """
        Given a dataframe with 1 column ('duration_us'), divide the range of durations
        into equally sized bins (n_bins), and compute aggregated values for each bin:

        - total_us:
          sum of 'duration_us'
        - total_invocations:
          total number of kernel calls
        - total_sec :
          sum of 'duration_us', in sec
        - percent_time :
          percent of the total GPU time that this bin makes up.
        - percent_invocations :
          percent of the total kernel invocations that this bin makes up.

        :param hist_df:
            Dataframe containing just a 'duration_us' column.
        :param n_bins:
        :param duration_field:
        :return:
        """
        total_time_us = hist_df[duration_field].sum()
        max_dur_us = hist_df[duration_field].max()
        min_dur_us = hist_df[duration_field].min()
        bin_size = int(np.ceil((max_dur_us - min_dur_us)/n_bins))

        binned_dfs = []
        for i in range(n_bins):
            bin_start = bin_size*i
            bin_end = bin_size*(i+1)
            # [..)
            bin_df = hist_df[(bin_start <= hist_df[duration_field]) & (hist_df[duration_field] < bin_end)]
            # bin_df = hist_df[bin_start <= hist_df[duration_field] < bin_end]

            binned_df = pd.DataFrame({
                'total_us': [bin_df[duration_field].sum()],
                'total_invocations': [len(bin_df)],
                'hist_label': "[{start},{end})".format(start=bin_start, end=bin_end),
            })
            binned_dfs.append(binned_df)
        binned_df = pd.concat(binned_dfs)
        binned_df['total_sec'] = binned_df['total_us']/1e6
        binned_df['percent_time'] = 100*binned_df['total_us']/binned_df['total_us'].sum()
        binned_df['percent_invocations'] = 100*binned_df['total_invocations']/binned_df['total_invocations'].sum()
        return binned_df

    def parse_api_trace(self, debug=False):
        """
        Convert CUDA API call times into dataframe:
        start_us,duration_us,name

        # Read (start, duration, api)
        $ nvprof --print-api-trace -i profile.nvprof 2>&1

        ======== Profiling result:
           Start  Duration  Name
             0ns  2.4150us  cuDeviceGetPCIBusId
        38.829ms  2.4050us  cuDeviceGetCount
        38.833ms     180ns  cuDeviceGetCount
        39.293ms     621ns  cuDeviceGet
        39.295ms     912ns  cuDeviceGetAttribute
        39.303ms     241ns  cuDeviceGetAttribute
        39.303ms     521ns  cuDeviceGetAttribute
        39.317ms     441ns  cuDeviceGetCount
        ...

        :return:
        """
        nvprof_parser = NvprofCSVParser(
            nvprof_file=self.nvprof_file,
            csv_file=self._nvprof_api_trace_csv(self.nvprof_file),
            nvprof_csv_opts=["--print-api-trace"],
            debug=self.debug or debug,
            debug_single_thread=self.debug_single_thread)
        df = nvprof_parser.parse(
            read_csv_kwargs=dict(
                header=0,
                names=["start_us", "duration_us", "name"],
            ))
        return df

    def parse_api_summary(self):
        """
        Convert CUDA API call times into dataframe:
        start_us,duration_us,name

        # Read (start, duration, api)
        $ nvprof --print-api-summary -i profile.nvprof 2>&1
        ======== Profiling result:
        "Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
        ,%,us,,us,us,us,
        "API calls",78.020031,7570567.921000,11,688233.447000,684120.984000,693194.046000,"cudaMemcpyAsync"
        "API calls",15.389094,1493259.895000,8,186657.486000,1.313000,1493246.020000,"cudaStreamCreateWithFlags"
        "API calls",3.686877,357751.097000,4,89437.774000,0.541000,357741.117000,"cudaFree"
        "API calls",1.904952,184844.416000,1,184844.416000,184844.416000,184844.416000,"cuDevicePrimaryCtxRetain"
        ...

        :return:
        """

        nvprof_parser = NvprofCSVParser(
            nvprof_file=self.nvprof_file,
            csv_file=self._nvprof_api_summary_csv(self.nvprof_file),
            nvprof_csv_opts=["--print-api-summary"],
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        df = nvprof_parser.parse(
            read_csv_kwargs=dict(
                header=0,
                # "Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
                names=["api_type", "time_percent", "total_time_us", "num_calls", "avg_duration_us", "min_duration_us", "max_duration_us", "name"],
        ))
        return df

    def parse_gpu_trace(self):
        """
        Convert CUDA kernel times into dataframe:
        start_us,duration_us,name

        # Read (start, duration, api)
        $ nvprof --print-gpu-trace -i /home/jgleeson/clone/rlscope/output/minigo/nvprof.debug/minigo_base_dir/nvprof --csv
        ======== Profiling result:
        "Start","Duration","Grid X","Grid Y","Grid Z","Block X","Block Y","Block Z","Registers Per Thread","Static SMem","Dynamic SMem","Size","Throughput","SrcMemType","DstMemType","Device","Context","Stream","Name","Correlation_ID"
        us,us,,,,,,,,KB,KB,KB,GB/s,,,,,,,
        0.000000,1.888000,,,,,,,,,,1.003906,0.507097,"Device",,"GeForce RTX 2080 Ti (0)","1","7","[CUDA memset]",272
        4421.878000,1.408000,,,,,,,,,,0.125000,0.084666,"Pinned","Device","GeForce RTX 2080 Ti (0)","1","22","[CUDA memcpy HtoD]",305

        :return:
        """

        nvprof_parser = NvprofCSVParser(
            nvprof_file=self.nvprof_file,
            csv_file=self._nvprof_gpu_trace_csv(self.nvprof_file),
            nvprof_csv_opts=["--print-gpu-trace"],
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        df = nvprof_parser.parse(
            read_csv_kwargs=dict(
                usecols=[0, 1, 18],
                header=0,
                names=["start_us", "duration_us", "name"],
        ))
        return df

    def parse_gpu_summary(self):
        """
        Convert CUDA GPU call times into dataframe:
        start_us,duration_us,name

        # Read (start, duration, name)
        $ nvprof --print-gpu-summary -i profile.nvprof 2>&1
        ======== Profiling result:
        "Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
        ,%,us,,us,us,us,
        "GPU activities",29.820746,6659.683000,510,13.058000,10.944000,22.048000,"volta_sgemm_128x64_nn"
        "GPU activities",8.283205,1849.837000,458,4.038000,3.904000,5.952000,"void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)"
        "GPU activities",7.780943,1737.670000,458,3.794000,3.712000,5.504000,"void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)"
        ...

        :return:
        """

        nvprof_parser = NvprofCSVParser(
            nvprof_file=self.nvprof_file,
            csv_file=self._nvprof_gpu_summary_csv(self.nvprof_file),
            nvprof_csv_opts=["--print-gpu-summary"],
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        df = nvprof_parser.parse(
            read_csv_kwargs=dict(
                header=0,
                # "Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
                names=["api_type", "time_percent", "total_time_us", "num_calls", "avg_duration_us", "min_duration_us", "max_duration_us", "name"],
        ))
        return df

    @property
    def directory(self):
        return _d(self.nvprof_file)

    def run(self):
        gpu_trace = self.parse_gpu_trace()
        gpu_trace_hist_df = self.hist_df_from_trace(gpu_trace, duration_field='duration_us')
        binned_df = self.binned_df_from_hist(gpu_trace_hist_df, duration_field='duration_us')
        self.plot_binned_gpu_kernels(binned_df, 'gpu_trace')

        # gpu_summary = self.parse_gpu_summary()
        # gpu_summary_hist_df = self.hist_df_from_summary(gpu_summary, duration_field='avg_duration_us')
        # binned_df = self.binned_df_from_hist(gpu_summary_hist_df, duration_field='avg_duration_us')
        # self.plot_binned(binned_df, 'gpu_summary')

        api_trace = self.parse_api_trace(debug=True)
        cudaLaunchKernel_api_trace_df = api_trace[api_trace['name'] == 'cudaLaunchKernel']
        all_cudaLaunchKernel_api_trace_hist_df = self.hist_df_from_trace(cudaLaunchKernel_api_trace_df, duration_field='duration_us')
        cudaLaunchKernel_api_trace_hist_df = self.discard_outliers(all_cudaLaunchKernel_api_trace_hist_df, duration_field='duration_us')
        cudaLaunchKernel_api_trace_binned_df = self.binned_df_from_hist(cudaLaunchKernel_api_trace_hist_df, duration_field='duration_us')
        self.plot_binned_cudaLaunchKernel(cudaLaunchKernel_api_trace_binned_df, 'api_trace.cudaLaunchKernel_duration_us')
        # self.plot_binned_gpu_kernels(cudaLaunchKernel_delay_df, '')

        # Q: How to ignore outliers...?
        cudaLaunchKernel_delay_df = self.delay_df_from_trace(cudaLaunchKernel_api_trace_df)
        all_cudaLaunchKernel_delay_hist_df = self.hist_df_from_trace(cudaLaunchKernel_delay_df, duration_field='delay_us')
        cudaLaunchKernel_delay_hist_df = self.discard_outliers(all_cudaLaunchKernel_delay_hist_df, duration_field='delay_us')
        cudaLaunchKernel_delay_binned_df = self.binned_df_from_hist(cudaLaunchKernel_delay_hist_df, n_bins=10, duration_field='delay_us')
        self.plot_binned_delay(cudaLaunchKernel_delay_binned_df, 'api_trace.cudaLaunchKernel_delay_us')

        # api_summary = self.parse_api_summary()
        # api_summary_hist_df = self.hist_df_from_summary(api_summary, duration_field='avg_duration_us')
        # binned_df = self.binned_df_from_hist(api_summary_hist_df, duration_field='avg_duration_us')
        # self.plot_binned(binned_df, 'api_summary')

    def discard_outliers(self, df, duration_field):
        """
        Discard outliers the same way a boxplot does with its whiskers.
        https://www.purplemath.com/modules/boxwhisk3.htm

        :param df:
        :param duration_field:
        :return:
        """
        # quartiles
        Q = np.percentile(df[duration_field], [25, 50, 75])
        IQR = Q[-1] - Q[0]
        # Q[0] - 1.5*(Q[2] - Q[0])
        # == Q[0] - 1.5*Q[2] + 1.5*Q[0]
        # == 2.5*Q[0] - 1.5*Q[2]
        keep_df = df[(Q[0] - 1.5*IQR <= df[duration_field]) &
                     (df[duration_field] <= Q[-1] + 1.5*IQR)]
        return keep_df

class GPUUtilOverTimePlot:
    """
    Legend label:
        "Kernels (delay=mean +/- std us, duration=mean +/- std us)"

    xs:
      Time (in seconds) since first GPU sample (for the same run)
    ys:
      GPU utilization in (%).
    """
    def __init__(self,
                 directory,
                 rlscope_directories,
                 show_std=False,
                 debug=False,
                 debug_single_thread=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.rlscope_directories = rlscope_directories
        self.show_std = show_std
        self.debug = debug
        self.debug_single_thread = debug_single_thread

    def _human_time(self, usec):
        units = ['us', 'ms', 'sec']
        # conversion_factor[i+1] = value you must divide by to convert units[i] to units[i+1]
        # conversion_factor[0] = value you must divide by to convert us to us
        conversion_factor = [1, 1000, 1000]

        value = usec
        unit = 'us'
        for i in range(len(units)):
            if (value / conversion_factor[i]) >= 1:
                value = (value / conversion_factor[i])
                unit = units[i]
            else:
                break
        return value, unit

    def _human_time_str(self, usec):
        value, unit = self._human_time(usec)
        return "{value:.1f} {unit}".format(
            value=value,
            unit=unit,
        )

    def read_data(self, rlscope_directory):
        df_reader = UtilDataframeReader(
            rlscope_directory,
            # add_fields=self.maybe_add_algo_env,
            debug=self.debug)
        df = df_reader.read()
        df = df[
            df['used_by_tensorflow'] &
            (df['device_type'] == 'GPU')]
        df['time_sec'] = (df['start_time_us'] - df['start_time_us'].min())/constants.MICROSECONDS_IN_SECOND
        df['util_percent'] = df['util'] * 100
        df.sort_values(by=['start_time_us'], inplace=True)

        event_reader = CUDAEventCSVReader(rlscope_directory, debug=self.debug)
        event_df = event_reader.read_df()
        # WANT:
        #   k[1].delay = k[1].start - k[0].end
        #   k[2].delay = k[2].start - k[1].end
        #   k[3].delay = k[3].start - k[2].end
        #   ...
        #   k[n].delay = k[n].start - k[n-1].end
        #                ----------   ----------
        #                k[1:].start  k[:n-1].end
        #
        # Q: Should we keep NEGATIVE delay?
        # A: depends what we want to know... I think NO... we want to know
        # "When there are gaps where no kernel is running, how large are they?"
        # Negative delay is nice to have a yes/no verdict of whether kernels overlap in time.
        #
        # NOTE: this is the difference between the START of consecutive kernel launches...
        # [    K1    ]      [      K2      ]
        # |                 |
        # K1.start          K2.start
        #
        # delay = K2.start - K1.start
        # This explains why delay is 10sec for 10-sec duration kernels...
        #
        # I think we would actually want:
        #   delay = max(0, K2.start - K1.end)

        # delay_us = event_df['start_time_us'].diff()[1:]
        # report_delay_us = delay_us

        event_df['end_time_us'] = event_df['start_time_us'] + event_df['duration_us']
        event_df.sort_values(['start_time_us', 'end_time_us'], inplace=True)
        delay_us = event_df['start_time_us'][1:].values - event_df['end_time_us'][:len(event_df)-1].values
        delay_non_neg_us = delay_us[delay_us >= 0]
        report_delay_us = delay_non_neg_us
        # Q: What does this mean...?  The number of consecutive kernels that overlap.
        delay_neg_us = delay_us[delay_us < 0]
        num_overlapped = len(delay_neg_us)
        mean_delay_neg_us =  delay_neg_us.mean()
        std_delay_neg_us = delay_neg_us.std()

        mean_delay_us =  report_delay_us.mean()
        std_delay_us = report_delay_us.std()
        num_delay_us = len(report_delay_us)
        mean_duration_us = event_df['duration_us'].mean()
        std_duration_us = event_df['duration_us'].std()

        data = {
            # 'util_df': util_df,
            # 'df': df,
            'mean_delay_neg_us': mean_delay_neg_us,
            'std_delay_neg_us': std_delay_neg_us,
            'num_overlapped': num_overlapped,

            'directory': rlscope_directory,
            'num_delay_us': num_delay_us,
            'mean_delay_us': mean_delay_us,
            'std_delay_us': std_delay_us,
            'mean_duration_us': mean_duration_us,
            'std_duration_us': std_duration_us,
        }

        df['label'] = self.legend_label(data)
        data['df'] = df

        return data

    def legend_label(self, data):
        def _mean_std(mean, std):
            if self.show_std:
                return "{mean} +/- {std}".format(
                    mean=self._human_time_str(usec=mean),
                    std=self._human_time_str(usec=std),
                )
            else:
                return "{mean}".format(
                    mean=self._human_time_str(usec=mean),
                )
        unit = 'us'
        # return "Kernels (delay={delay}, duration={duration}".format(
        base = _b(data['directory'])
        base = re.sub(r'^gpu_util_experiment\.', '', base)
        return "{base}: delay={delay}, duration={duration}".format(
            base=base,
            delay=_mean_std(data['mean_delay_us'], data['std_delay_us']),
            duration=_mean_std(data['mean_duration_us'], data['std_duration_us']),
        )

    @staticmethod
    def Worker_GPUUtilOverTimePlot_read_data(kwargs):
        # for var, value in kwargs.items():
        #     locals()[var] = value
        self = kwargs['self']
        rlscope_directory = kwargs['rlscope_directory']
        data = self.read_data(rlscope_directory)
        return rlscope_directory, data

    def run(self):
        dir_to_data = dict()

        # for rlscope_directory in self.rlscope_directories:
        #     dir_to_data[rlscope_directory] = self.read_data(rlscope_directory)

        if self.debug_single_thread:
            n_workers = 1
        else:
            n_workers = multiprocessing.cpu_count()
        def Args_GPUUtilOverTimePlot_read_data(rlscope_directory):
            return dict(
                self=self,
                rlscope_directory=rlscope_directory,
            )

        with ProcessPoolExecutor(n_workers) as pool:
            # splits = split_list(proto_paths, n_workers)
            kwargs_list = [Args_GPUUtilOverTimePlot_read_data(rlscope_directory) for rlscope_directory in self.rlscope_directories]
            data_list = map_pool(
                pool, GPUUtilOverTimePlot.Worker_GPUUtilOverTimePlot_read_data, kwargs_list,
                desc="GPUUtilOverTimePlot.read_data",
                show_progress=True,
                sync=self.debug_single_thread)
            for rlscope_directory, data in data_list:
                dir_to_data[rlscope_directory] = data

        df = pd.concat([
            dir_to_data[rlscope_directory]['df']
            for rlscope_directory in self.rlscope_directories], sort=True)

        fig, ax = plt.subplots()
        # df = pd.DataFrame({'A':26, 'B':20}, index=['N'])
        # df.plot(kind='bar', ax=ax)
        # ax.legend(["AAA", "BBB"]);
        # df.plot(kind='scatter', x='time_sec', y='gpu_util_percent', ax=ax)
        # sns.scatterplot(x='time_sec', y='util_percent', hue='label', data=df, ax=ax)
        sns.lineplot(x='time_sec', y='util_percent', hue='label', data=df, ax=ax)
        fix_seaborn_legend(ax)
        ax.set_ylabel("GPU utilization (%)")
        ax.set_xlabel("Total runtime (seconds)")

        os.makedirs(self.directory, exist_ok=True)
        plot_path = self._get_path('pdf')
        csv_path = self._get_path('csv')
        json_path = self._get_path('json')

        metadata_js = dict()
        for rlscope_directory in self.rlscope_directories:
            metadata_js[rlscope_directory] = dict()
            for k, v in dir_to_data[rlscope_directory].items():
                if k != 'df':
                    metadata_js[rlscope_directory][k] = v
        # logger.info('Dump metadata js to {path}'.format(path=json_path))
        do_dump_json(metadata_js, json_path)

        df.to_csv(csv_path, index=False)
        logger.info('Save figure to {path}'.format(path=plot_path))
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


    def _get_path(self, ext):
        return _j(
            self.directory,
            "GPUUtilOverTimePlot.{ext}".format(ext=ext),
        )

class UtilPlot:
    def __init__(self,
                 csv,
                 directory,
                 x_type,
                 y_title=None,
                 suffix=None,
                 rotation=45.,
                 width=None,
                 height=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.csv = csv
        self.directory = directory
        self.x_type = x_type
        self.y_title = y_title
        self.rotation = rotation
        self.suffix = suffix
        self.width = width
        self.height = height
        self.debug = debug

    def _read_df(self):
        self.df = pd.read_csv(self.csv)

        def _x_field(algo, env):
            return stacked_bar_plots.get_x_field(algo, env, self.x_type)
        self.df['x_field'] = np.vectorize(_x_field, otypes=[str])(
            self.df['algo'],
            self.df['env'])

        self.all_df = copy.copy(self.df)

        self.df = self.df[
            (self.df['used_by_tensorflow']) &
            (self.df['device_type'] == 'GPU')]

        keep_cols = ['machine_name', 'algo', 'env', 'device_name']
        # df_count = self.df[keep_cols].groupby(keep_cols).reset_index()
        df_count = self.df[keep_cols].groupby(keep_cols).size().reset_index(name="counts")[keep_cols]
        groupby_cols = ['machine_name', 'algo', 'env']
        df_count = df_count[keep_cols].groupby(groupby_cols).size().reset_index(name='counts')
        df_count_more_than_1_gpu = df_count[df_count['counts'] > 1]
        if len(df_count_more_than_1_gpu) > 0:
            buf = StringIO()
            DataFrame.print_df(df_count_more_than_1_gpu, file=buf)
            logger.info("Saw > 1 GPU being using for at least one (algo, env) experiment; not sure which GPU to show:\n{msg}".format(
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
            ))
            assert len(df_count_more_than_1_gpu) == 0

    def run(self):
        self._read_df()
        self._plot()

    def _get_plot_path(self, ext):
        if self.suffix is not None:
            suffix_str = '.{suffix}'.format(suffix=self.suffix)
        else:
            suffix_str = ''
        return _j(self.directory, "UtilPlot{suffix}.{ext}".format(
            suffix=suffix_str,
            ext=ext,
        ))

    def legend_path(self, ext):
        return re.sub(r'(?P<ext>\.[^.]+)$', r'.legend\g<ext>', self._get_plot_path(ext))

    def _plot(self):
        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None

        # a4_width_px = 983
        # textwidth_px = 812
        # a4_width_inches = 8.27
        # plot_percent = 5/6
        # plot_width_inches = (a4_width_inches * (textwidth_px / a4_width_px) * plot_percent)
        # plot_height_inches = 3
        # figsize = (plot_width_inches, plot_height_inches)

        figsize = (10, 2.5)

        logger.info("Plot dimensions (inches) = {figsize}".format(
            figsize=figsize))

        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        SMALL_SIZE = 8
        # Default font size for matplotlib (too small for paper).
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        FONT_SIZE = MEDIUM_SIZE

        plt.rc('font', size=FONT_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

        # gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[5, 4, 1])
        ax_0 = plt.subplot(gs[0])
        ax_1 = plt.subplot(gs[1])
        ax_2 = plt.subplot(gs[2])
        axes = [ax_0, ax_1, ax_2]

        # plt.setp(ax2.get_xticklabels(), visible=False)
        fig.subplots_adjust(wspace=0.4)

        self.df_gpu = self.df

        logger.info(pprint_msg(self.df_gpu))
        # ax = sns.boxplot(x=self.df_gpu['x_field'], y=100*self.df_gpu['util'],
        #                  showfliers=False,
        #                  )

        # algo_env_group_title = {
        #     'environment_choice': "Simulator choice\n(RL algorithm = PPO)",
        #     'algorithm_choice_1a_med_complexity': "Algorithm choice\n(Simulator = Walker2D)",
        #     'scaleup_rl': 'Scale-up RL workload',
        # }
        UNKNOWN_ALGO_ENV = "UNKNOWN_ALGO_ENV"
        # def is_scaleup_rl(algo, env):
        #     return (algo == 'MCTS' and env == 'GoEnv')
        def as_algo_env_group(algo, env):
            if expr_config.is_fig_algo_comparison_med_complexity(algo, env):
                return 'algorithm_choice_1a_med_complexity'
            elif expr_config.is_fig_env_comparison(algo, env):
                # HACK: Match (algo, env) used in the "Simulator choice" figure in paper.
                if not re.search(r'Ant|HalfCheetah|Hopper|Pong|Walker2D|AirLearning', env):
                    return UNKNOWN_ALGO_ENV
                return 'environment_choice'
            elif expr_config.is_mcts_go(algo, env):
                # is_scaleup_rl(algo, env)
                return 'scaleup_rl'

            return UNKNOWN_ALGO_ENV

        def get_plot_x_axis_label(algo_env_group):
            if algo_env_group == 'scaleup_rl':
                return "(RL algorithm, Simulator)"
            elif algo_env_group == 'environment_choice':
                return "Simulator"
            elif algo_env_group == 'algorithm_choice_1a_med_complexity':
                return "RL algorithm"
            raise NotImplementedError()

        def get_plot_title(algo_env_group):
            if algo_env_group == 'scaleup_rl':
                return "Scale-up RL workload"
            elif algo_env_group == 'environment_choice':
                return "Simulator choice\n(RL algorithm = PPO)"
            elif algo_env_group == 'algorithm_choice_1a_med_complexity':
                return "Algorithm choice\n(Simulator = Walker2D)"
            raise NotImplementedError()

        def as_x_type(algo_env_group):
            if algo_env_group == 'scaleup_rl':
                return 'rl-comparison'
            if algo_env_group == 'environment_choice':
                return 'env-comparison'
            if algo_env_group == 'algorithm_choice_1a_med_complexity':
                return 'algo-comparison'
            raise NotImplementedError()

        def _mk_boxplot(i, ax, algo_env_group, df_gpu):
            # xs = df_gpu['x_field']
            def _x_field(algo, env):
                x_type = as_x_type(algo_env_group)
                return stacked_bar_plots.get_x_field(algo, env, x_type)
            xs = np.vectorize(_x_field, otypes=[str])(
                df_gpu['algo'],
                df_gpu['env'])
            ys = 100*df_gpu['util']
            logger.info("Plot algo_env_group:\n{msg}".format(
                msg=pprint_msg({
                    'i': i,
                    'algo_env_group': algo_env_group,
                    'df_gpu': df_gpu,
                    'xs': xs,
                    'ys': ys,
                }),
            ))
            # https://python-graph-gallery.com/33-control-colors-of-boxplot-seaborn/
            # https://matplotlib.org/examples/color/named_colors.html
            boxplot_color = 'tan'
            sns.boxplot(
                xs, ys,
                color=boxplot_color,
                ax=ax,
                showfliers=False,
                medianprops={'color': 'black'},
            )
            x_label = get_plot_x_axis_label(algo_env_group)
            ax.set_xlabel(x_label)
            if i == 0:
                ax.set_ylabel('GPU Utilization (%)')
            else:
                # i > 0
                ax.set_ylabel(None)
            title = get_plot_title(algo_env_group)
            ax.set_title(title)

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0., ymax)

            if self.rotation is not None:
                # ax = bottom_plot.axes
                ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)

        self.df_gpu['algo_env_group'] = np.vectorize(as_algo_env_group, otypes=[np.str])(
            self.df_gpu['algo'],
            self.df_gpu['env'],
        )
        self.df_gpu = self.df_gpu[self.df_gpu['algo_env_group'] != UNKNOWN_ALGO_ENV]
        algo_env_groups = [
            'environment_choice',
            'algorithm_choice_1a_med_complexity',
            'scaleup_rl',
        ]
        for i, algo_env_group in enumerate(algo_env_groups):
            df_group = self.df_gpu[self.df_gpu['algo_env_group'] == algo_env_group]
            if len(df_group) == 0:
                logger.warning("Found no GPU utilization data for algo_env_group={group}, SKIP plot".format(
                    group=algo_env_group))
                continue
            ax = axes[i]
            _mk_boxplot(i, ax, algo_env_group, df_group)

        # groupby_cols = ['algo', 'env_id']
        # # label_df = self.df_gpu[list(set(groupby_cols + ['x_field', 'util']))]
        # label_df = self.df_gpu.groupby(groupby_cols).mean()
        # add_hierarchical_labels(fig, ax, self.df_gpu, label_df, groupby_cols)

        # if self.rotation is not None:
        #     # ax = bottom_plot.axes
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)
        #
        # # Default ylim for violinplot is slightly passed bottom/top of data:
        # #   ipdb> ax.get_ylim()
        # #   (-2.3149999976158147, 48.614999949932105)
        # #   ipdb> np.min(100*self.df['util'])
        # #   0.0
        # #   ipdb> np.max(100*self.df['util'])
        # #   46.29999995231629
        # ymin, ymax = ax.get_ylim()
        # ax.set_ylim(0., ymax)
        #
        # ax.set_xlabel(self.x_axis_label)
        # if self.y_title is not None:
        #     ax.set_ylabel(self.y_title)

        png_path = self._get_plot_path('pdf')
        logger.info('Save figure to {path}'.format(path=png_path))
        # fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def _plot_all(self):

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logger.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)


        self.df_gpu = self.df

        logger.info(pprint_msg(self.df_gpu))

        ax = sns.boxplot(x=self.df_gpu['x_field'], y=100*self.df_gpu['util'],
                         showfliers=False,
                         )

        # groupby_cols = ['algo', 'env_id']
        # # label_df = self.df_gpu[list(set(groupby_cols + ['x_field', 'util']))]
        # label_df = self.df_gpu.groupby(groupby_cols).mean()
        # add_hierarchical_labels(fig, ax, self.df_gpu, label_df, groupby_cols)

        if self.rotation is not None:
            # ax = bottom_plot.axes
            ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)

        # Default ylim for violinplot is slightly passed bottom/top of data:
        #   ipdb> ax.get_ylim()
        #   (-2.3149999976158147, 48.614999949932105)
        #   ipdb> np.min(100*self.df['util'])
        #   0.0
        #   ipdb> np.max(100*self.df['util'])
        #   46.29999995231629
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0., ymax)

        ax.set_xlabel(self.x_axis_label)
        if self.y_title is not None:
            ax.set_ylabel(self.y_title)

        png_path = self._get_plot_path('pdf')
        logger.info('Save figure to {path}'.format(path=png_path))
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    @property
    def x_axis_label(self):
        if self.x_type == 'rl-comparison':
            return "(RL algorithm, Environment)"
        elif self.x_type == 'env-comparison':
            return "Environment"
        elif self.x_type == 'algo-comparison':
            return "RL algorithm"
        raise NotImplementedError()

# https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts-in-matplotlib


def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_len(my_index,level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k,g in itertools.groupby(labels)]

def my_label_len(label_df, col):
    # labels = my_index.get_level_values(level)
    labels = label_df[col]
    ret = [(k, sum(1 for i in g)) for k,g in itertools.groupby(labels)]
    logger.info(pprint_msg({'label_len': ret}))
    return ret

def label_group_bar_table(ax, df):
    ypos = -.1
    scale = 1./df.index.size
    for level in range(df.index.nlevels)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index, level):
            lxpos = (pos + .5 * rpos)*scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale , ypos)
        ypos -= .1

def my_label_group_bar_table(ax, label_df, df, groupby_cols):
    ypos = -.1
    # df.index.size = len(['Room', 'Shelf', 'Staple'])
    scale = 1./len(groupby_cols)
    # scale = 1./df.index.size
    # for level in range(df.index.nlevels)[::-1]:
    for level in range(len(groupby_cols))[::-1]:
        pos = 0
        col = groupby_cols[level]
        for label, rpos in my_label_len(label_df, col):
            lxpos = (pos + .5 * rpos)*scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
            add_line(ax, pos*scale, ypos)
            pos += rpos
        add_line(ax, pos*scale, ypos)
        ypos -= .1


def attempt_grouped_xlabel():
    # https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts-in-matplotlib

    def get_table():
        data_table = pd.DataFrame({'Room':['Room A']*4 + ['Room B']*4,
                                   'Shelf':(['Shelf 1']*2 + ['Shelf 2']*2)*2,
                                   'Staple':['Milk','Water','Sugar','Honey','Wheat','Corn','Chicken','Cow'],
                                   'Quantity':[10,20,5,6,4,7,2,1],
                                   'Ordered':np.random.randint(0,10,8)
                                   })
        return data_table

    sample_df = get_table()
    g = sample_df.groupby(['Room', 'Shelf', 'Staple'])
    df = g.sum()
    logger.info(pprint_msg(df))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    df.plot(kind='bar', stacked=True, ax=fig.gca())
    # sns.barplot(x=df[''])

    #Below 3 lines remove default labels
    labels = ['' for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.set_xlabel('')

    label_group_bar_table(ax, df)

    # This makes the vertical spacing between x-labels closer.
    fig.subplots_adjust(bottom=.1*df.index.nlevels)

    png_path = '{func}.png'.format(
        func=test_grouped_xlabel.__name__,
    )
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0)

def add_hierarchical_labels(fig, ax, df, label_df, groupby_cols):

    #Below 3 lines remove default labels
    labels = ['' for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    ax.set_xlabel('')

    # label_group_bar_table(ax, df)
    my_label_group_bar_table(ax, label_df, df, groupby_cols)


    # This makes the vertical spacing between x-labels closer.
    # fig.subplots_adjust(bottom=.1*df.index.nlevels)
    fig.subplots_adjust(bottom=.1*len(groupby_cols))

def main():
    parser = argparse.ArgumentParser(
        textwrap.dedent("""\
        Test plots
        """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--test-grouped-xlabel',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to group x-labels + sub-labels.
    """))

    args = parser.parse_args()

    if args.test_grouped_xlabel:
        attempt_grouped_xlabel()

if __name__ == '__main__':
    main()
