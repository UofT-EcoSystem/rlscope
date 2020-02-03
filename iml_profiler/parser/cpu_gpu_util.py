import logging
import copy
import itertools
import argparse

from iml_profiler.protobuf.pyprof_pb2 import CategoryEventsProto, MachineUtilization, DeviceUtilization, UtilizationSample
from iml_profiler.parser.common import *
from iml_profiler.profiler import experiment
from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b
import pandas as pd
import seaborn as sns

from iml_profiler.parser.dataframe import UtilDataframeReader

from matplotlib import pyplot as plt
import matplotlib.gridspec

from iml_profiler.parser import stacked_bar_plots

from iml_profiler.profiler import iml_logging

from iml_profiler.experiment import expr_config
from iml_profiler.parser.plot import CUDAEventCSVReader, fix_seaborn_legend

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
                 iml_directories,
                 algo_env_from_dir=False,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """
        :param directories:
        :param debug:
        """
        self.directory = directory
        self.iml_directories = iml_directories
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
                logging.info("Didn't find {path}; skip adding experiment columns to csv".format(path=path))
            return None
        data = experiment.load_experiment_config(directory)
        return data

    def maybe_add_algo_env(self, machine_util_path):
        assert is_machine_util_file(machine_util_path)

        iml_directory = _d(machine_util_path)

        if self.algo_env_from_dir:
            return self.add_algo_env_from_dir(machine_util_path)
        if not _e(experiment.experiment_config_path(iml_directory)):
            return self.add_experiment_config(machine_util_path)

        # Not sure what (algo, env) is; don't add those columns.
        return None

    def add_algo_env_from_dir(self, machine_util_path):
        assert is_machine_util_file(machine_util_path)
        iml_dir = _d(machine_util_path)

        path = os.path.normpath(iml_dir)
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
        for directory in self.iml_directories:
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

            logging.info(pprint_msg({
                'group': group,
                'start_time': start_time,
                'max_time': max_time,
            }))
            logging.info(pprint_msg(df_group))

            # import ipdb; ipdb.set_trace()


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
        logging.info("Output raw un-aggregated machine utilization data @ {path}".format(path=self._raw_csv_path))
        new_df.to_csv(self._raw_csv_path, index=False)

        df_agg = new_df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        flat_df_agg = self.flattened_agg_df(df_agg)

        # import ipdb; ipdb.set_trace()
        logging.info("Output min/max/std aggregated machine utilization data @ {path}".format(path=self._agg_csv_path))
        flat_df_agg.to_csv(self._agg_csv_path, index=False)

        # Q: Which (algo, env) have at least one utilization readings > 0 a GPU whose device_id > 0?

        util_plot = UtilPlot(
            csv=self._raw_csv_path,
            directory=self.directory,
            x_type='rl-comparison',
            debug=self.debug,
        )
        util_plot.run()

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
                 iml_directories,
                 show_std=False,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.iml_directories = iml_directories
        self.show_std = show_std
        self.debug = debug

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

    def read_data(self, iml_directory):
        df_reader = UtilDataframeReader(
            iml_directory,
            # add_fields=self.maybe_add_algo_env,
            debug=self.debug)
        df = df_reader.read()
        df = df[
            df['used_by_tensorflow'] &
            (df['device_type'] == 'GPU')]
        df['time_sec'] = (df['start_time_us'] - df['start_time_us'].min())/MICROSECONDS_IN_SECOND
        df['util_percent'] = df['util'] * 100
        df.sort_values(by=['start_time_us'], inplace=True)

        event_reader = CUDAEventCSVReader(iml_directory, debug=self.debug)
        event_df = event_reader.read_df()
        delay_us = event_df['start_time_us'].diff()[1:]
        mean_delay_us =  delay_us.mean()
        std_delay_us = delay_us.std()
        mean_duration_us = event_df['duration_us'].mean()
        std_duration_us = event_df['duration_us'].std()

        data = {
            # 'util_df': util_df,
            # 'df': df,
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
        return "delay={delay}, duration={duration}".format(
            delay=_mean_std(data['mean_delay_us'], data['std_delay_us']),
            duration=_mean_std(data['mean_duration_us'], data['std_duration_us']),
        )

    def run(self):
        dir_to_data = dict()
        for iml_directory in self.iml_directories:
            dir_to_data[iml_directory] = self.read_data(iml_directory)

        df = pd.concat([
            dir_to_data[iml_directory]['df']
            for iml_directory in self.iml_directories], sort=True)

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
        plot_path = self._get_path('pdf')
        csv_path = self._get_path('csv')

        df.to_csv(csv_path, index=False)
        logging.info('Save figure to {path}'.format(path=plot_path))
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
            logging.info("Saw > 1 GPU being using for at least one (algo, env) experiment; not sure which GPU to show:\n{msg}".format(
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
            logging.info("Setting figsize = {fig}".format(fig=figsize))
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

        logging.info("Plot dimensions (inches) = {figsize}".format(
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

        logging.info(pprint_msg(self.df_gpu))
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
            logging.info("Plot algo_env_group:\n{msg}".format(
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
                logging.warning("Found no GPU utilization data for algo_env_group={group}, SKIP plot".format(
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
        logging.info('Save figure to {path}'.format(path=png_path))
        # fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def _plot_all(self):

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logging.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)


        self.df_gpu = self.df

        logging.info(pprint_msg(self.df_gpu))

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
        logging.info('Save figure to {path}'.format(path=png_path))
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

def test_table():
    data_table = pd.DataFrame({'Room':['Room A']*4 + ['Room B']*4,
                               'Shelf':(['Shelf 1']*2 + ['Shelf 2']*2)*2,
                               'Staple':['Milk','Water','Sugar','Honey','Wheat','Corn','Chicken','Cow'],
                               'Quantity':[10,20,5,6,4,7,2,1],
                               'Ordered':np.random.randint(0,10,8)
                               })
    return data_table


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
    logging.info(pprint_msg({'label_len': ret}))
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


def test_grouped_xlabel():
    # https://stackoverflow.com/questions/19184484/how-to-add-group-labels-for-bar-charts-in-matplotlib
    sample_df = test_table()
    g = sample_df.groupby(['Room', 'Shelf', 'Staple'])
    df = g.sum()
    logging.info(pprint_msg(df))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    import ipdb; ipdb.set_trace()

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

    iml_logging.setup_logging()

    if args.test_grouped_xlabel:
        test_grouped_xlabel()

if __name__ == '__main__':
    main()
