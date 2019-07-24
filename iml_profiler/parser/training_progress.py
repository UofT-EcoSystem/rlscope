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

from iml_profiler.parser.dataframe import TrainingProgressDataframeReader

from iml_profiler.parser import stacked_bar_plots

from iml_profiler.profiler import iml_logging


class TrainingProgressParser:
    """
    PSEUDOCODE:
    Read all the training_progress.trace_*.proto files across all configurations
    into a single data-frame.
    IML profiling overhead:
      To compare profiling overhead, compare the last sample of iterations/second for a given configuration
      (that way we don't need the exact number of timesteps to match up between compared configurations).
      This also works if we want to compare TOTAL training time; just run training till the very end.
    IML accuracy:
      total_training_time['instrumented'] - extrap_training_time['instrumented']

        df = []
        For each config in (instrumented, uninstrumented):
            df.append: Read rows like: [
                machine, process, phase, algo, env, config,

                total_timesteps,

                start_trace_time_us,

                start_percent_complete,
                start_num_timesteps,
                start_training_time_us,

                end_percent_complete,
                end_training_time_us,
                end_num_timesteps,
            ]

        #
        # Keep the very last call to iml.prof.report_progress(...).
        # We will use this to compare "iterations-per-second".
        #
        keep_rows = []
        For each (machine, process, phase, algo, env):
            max_time = max(df[machine, process, phase, algo, env]['end_training_time_us'])
            row = df[machine, process, phase, algo, env, AND max_time]
            keep_rows.append(
                row
            )

        df['training_time_sec'] = (df['end_training_time_us'] - df['start_training_time_us']) / USEC_IN_SEC
        df['iters_per_sec'] = df['training_time_sec'] / (df['end_num_timesteps'] - df['start_num_timesteps'])
    """
    def __init__(self,
                 directory,
                 iml_directories,
                 ignore_phase=False,
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
        self.ignore_phase = ignore_phase
        self.algo_env_from_dir = algo_env_from_dir
        self.debug = debug

        self.added_fields = set()

    @property
    def _raw_csv_path(self):
        return _j(self.directory, "overall_training_progress.raw.csv")

    @property
    def _last_agg_csv_path(self):
        return _j(self.directory, "overall_training_progress.last.agg.csv")

    @property
    def _profiling_overhead_agg_csv_path(self):
        return _j(self.directory, "overall_training_progress.profiling_overhead.agg.csv")

    def add_experiment_config(self, path):
        """
        add_fields(path)

        We expect to find experiment_config.json where the machine_util.*.proto files live.

        :param path:
        :return:
        """
        assert is_training_progress_file(path)
        directory = _d(path)
        path = experiment.experiment_config_path(directory)
        if not _e(path):
            if self.debug:
                logging.info("Didn't find {path}; skip adding experiment columns to csv".format(path=path))
            return None
        data = experiment.load_experiment_config(directory)
        return data

    def maybe_add_algo_env(self, path):
        assert is_training_progress_file(path)

        iml_directory = _d(path)

        if self.algo_env_from_dir:
            return self.add_config_algo_env_from_dir(path)
        if not _e(experiment.experiment_config_path(iml_directory)):
            return self.add_experiment_config(path)

        # Not sure what (algo, env) is; don't add those columns.
        return None

    def add_config_algo_env_from_dir(self, path):
        assert is_training_progress_file(path)
        iml_dir = _d(path)
        # Path looks like:
        # '$HOME/clone/iml/output/iml_bench/all.debug/config_uninstrumented/a2c/PongNoFrameskip-v4/process/a2c_PongNoFrameskip-v4/phase/default_phase/training_progress.trace_3.proto'
        #  - $HOME
        #  - clone
        #  - iml
        #  - output
        #  - iml_bench
        #  - all.debug
        # -7 - config_uninstrumented
        # -6 - a2c
        # -5 - PongNoFrameskip-v4
        # -4 - process
        # -3 - a2c_PongNoFrameskip-v4
        # -2 - phase
        # -1 - default_phase

        norm_path = os.path.normpath(iml_dir)
        components = norm_path.split(os.sep)
        env_id = components[-5]
        algo = components[-6]
        fields = {
            'algo': algo,
            'env_id': env_id,
        }
        assert is_config_dir(os.sep.join(components[:-7 + 1]))
        # if is_config_dir(os.sep.join(components[:-7 + 1])):
        fields['config'] = components[-7]
        # else:
        #     # Default to 'instrumented'
        #     fields['config'] = 'instrumented'
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
            df_reader = TrainingProgressDataframeReader(
                directory,
                add_fields=self.maybe_add_algo_env,
                debug=self.debug)
            df = df_reader.read()
            self.added_fields.update(df_reader.added_fields)
            dfs.append(df)
        raw_df = pd.concat(dfs)

        # # 1. Memory utilization:
        # # 2. CPU utilization:
        # groupby_cols = sorted(self.added_fields) + ['machine_name', 'device_name']
        #
        # # df_agg = df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        # # flat_df_agg = self.flattened_agg_df(df_agg)
        #
        # # - Use start_time_us timestamp to assign each utilization sample an "index" number from [0...1];
        # #   this is trace_percent: the percent time into the collected trace
        # # - Group by (algo, env)
        # #   - Reduce:
        # #     # for each group member, divide by group-max
        # #     - max_time = max(row['start_time_us'])
        # #     - min_time = min(row['start_time_us'])
        # #     - row['trace_percent'] = (row['start_time_us'] - min_time)/max_time
        # # TODO: debug this to see if it works.
        # dfs = []
        # groupby = df.groupby(groupby_cols)
        # for group, df_group in groupby:
        #
        #     max_time = max(df_group['start_time_us'])
        #     start_time = min(df_group['start_time_us'])
        #     length_time = max_time - start_time
        #     df_group['trace_percent'] = (df_group['start_time_us'] - start_time) / length_time
        #     dfs.append(df_group)
        #
        #     logging.info(pprint_msg({
        #         'group': group,
        #         'start_time': start_time,
        #         'max_time': max_time,
        #     }))
        #     logging.info(pprint_msg(df_group))
        #
        #     # import ipdb; ipdb.set_trace()
        #
        # new_df = pd.concat(dfs)

        # OUTPUT raw thing here.
        logging.info("Output raw un-aggregated machine utilization data @ {path}".format(path=self._raw_csv_path))
        raw_df.to_csv(self._raw_csv_path, index=False)

        agg_df = raw_df
        agg_df['timesteps_per_sec'] = \
            ( agg_df['end_num_timesteps'] - agg_df['start_num_timesteps'] ) / \
            (( agg_df['end_training_time_us'] - agg_df['start_training_time_us'] ) / USEC_IN_SEC )
        agg_df['total_trace_time_sec'] = (1. / agg_df['timesteps_per_sec']) * agg_df['total_timesteps']


        groupby_cols = set([
            'process_name',
            'phase',
            'machine_name',
            'algo',
            'env_id',
            'config',
        ])
        if self.ignore_phase:
            groupby_cols.remove('phase')
        groupby_cols = sorted(groupby_cols)
        groupby = agg_df.groupby(groupby_cols)
        keep_dfs = []
        for group, df_group in groupby:
            # Only keep the last sample of training progress (it will average over total training time the most).
            max_time = np.max(df_group['end_training_time_us'])
            df_keep = df_group[df_group['end_training_time_us'] == max_time]
            keep_dfs.append(df_keep)
        keep_agg_df = pd.concat(keep_dfs)

        logging.info("Add 'total_training_time_sec' and only keep the 'last' sampled training progress data for each (config, algo, env) @ {path}".format(path=self._last_agg_csv_path))
        keep_agg_df.to_csv(self._last_agg_csv_path, index=False)

        join_cols = set(groupby_cols)
        join_cols.remove('config')
        join_cols = sorted(join_cols)

        # TODO: check that each index only contains 1 row
        # CONCERN: don't want to mix config=uninstrumented with config=uninstrumented_full
        ins_df = keep_agg_df[keep_agg_df['config'].apply(lambda config: bool(config_is_instrumented(config)))]
        ins_df = ins_df.set_index(join_cols)
        assert ins_df.index.is_unique
        unins_df = keep_agg_df[keep_agg_df['config'].apply(lambda config: bool(config_is_uninstrumented(config)))]
        unins_df = unins_df.set_index(join_cols)
        assert unins_df.index.is_unique

        all_df = ins_df.join(unins_df, lsuffix='_ins', rsuffix='_unins')
        all_df['profiling_overhead_percent'] = \
            100. * ( all_df['total_trace_time_sec_ins'] - all_df['total_trace_time_sec_unins'] ) / all_df['total_trace_time_sec_unins']
        # other_calc = 100. * ( all_df['timesteps_per_sec_unins'] / all_df['timesteps_per_sec_ins'] )
        other_calc = 100. * ( (1./all_df['timesteps_per_sec_ins']) - (1./all_df['timesteps_per_sec_unins']) ) / (1./all_df['timesteps_per_sec_unins'])
        assert np.all(np.isclose(all_df['profiling_overhead_percent'], other_calc))

        logging.info("Add 'profiling_overhead_percent' for each (config, algo, env) @ {path}".format(path=self._profiling_overhead_agg_csv_path))
        all_df.to_csv(self._profiling_overhead_agg_csv_path, index=True)

        # df_agg = new_df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        # flat_df_agg = self.flattened_agg_df(df_agg)

        # import ipdb; ipdb.set_trace()
        # logging.info("Output min/max/std aggregated machine utilization data @ {path}".format(path=self._agg_csv_path))
        # flat_df_agg.to_csv(self._agg_csv_path, index=False)

class ProfilingOverheadPlot:
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

        x_fields = []
        for index, row in self.df.iterrows():
            if 'env' in row:
                env = row['env']
            else:
                env = row['env_id']
            x_field = stacked_bar_plots.get_x_field(row['algo'], env, self.x_type)
            x_fields.append(x_field)
        self.df['x_field'] = x_fields

    def run(self):
        self._read_df()
        self.plot()

    def _get_plot_path(self, ext):
        if self.suffix is not None:
            suffix_str = '.{suffix}'.format(suffix=self.suffix)
        else:
            suffix_str = ''
        return _j(self.directory, "ProfilingOverheadPlot{suffix}.{ext}".format(
            suffix=suffix_str,
            ext=ext,
        ))

    def legend_path(self, ext):
        return re.sub(r'(?P<ext>\.[^.]+)$', r'.legend\g<ext>', self._get_plot_path(ext))

    def plot(self):

        # figlegend.tight_layout()
        # figlegend.savefig(self.legend_path, bbox_inches='tight', pad_inches=0)
        # plt.close(figlegend)

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logging.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)


        # ax = fig.add_subplot(111)
        # ax2 = None
        # if self.y2_field is not None:
        #     ax2 = ax.twinx()
        #     # Need to do this, otherwise, training time bar is ABOVE gridlines from ax.
        #     ax.set_zorder(ax2.get_zorder()+1)
        #     # Need to do this, otherwise training time bar is invisible.
        #     ax.patch.set_visible(False)

        # def is_cpu(device_name):
        #     if re.search(r'Intel|Xeon|CPU', device_name):
        #         return True
        #     return False
        #
        # def is_gpu(device_name):
        #     return not is_cpu(device_name)
        #
        # def should_keep(row):
        #     if row['machine_name'] == 'reddirtx-ubuntu':
        #         # Ignore 'Tesla K40c' (unused, 0 util)
        #         return row['device_name'] == 'GeForce RTX 2080 Ti'
        #     return True
        #
        # self.df_gpu = self.df
        #
        # self.df_gpu = self.df_gpu[self.df_gpu['device_name'].apply(is_gpu)]
        #
        # self.df_gpu = self.df_gpu[self.df_gpu.apply(should_keep, axis=1)]

        logging.info(pprint_msg(self.df))

        # ax = sns.violinplot(x=self.df_gpu['x_field'], y=100*self.df_gpu['util'],
        #                     inner="box",
        #                     # cut=0.,
        #                     )

        # ax = sns.boxplot(x=self.df['x_field'], y=100*self.df['util'],
        #                  showfliers=False,
        #                  )

        # We want data that looks like:
        # algo, env, config,         x_field, total_training_time_sec
        # ppo2  Pong uninstrumented      ...                      ...
        # ppo2  Pong instrumented        ...                      ...
        #
        # To achieve that, we need to:
        # - For each row, split it into two rows, one for ins, one for unins.
        def is_ins(colname):
            return bool(re.search(r'_ins$', colname))
        def is_unins(colname):
            return bool(re.search(r'_unins$', colname))
        def remove_suffix(colname):
            m = re.search(r'(?P<col>.*)_(?P<suffix>ins|unins)$', colname)
            return m.group('col')
        def is_common_row(colname):
            return not is_ins(colname) and not is_unins(colname)
        rows = []
        common_cols = [col for col in self.df.keys() if is_common_row(col)]
        ins_cols = [col for col in self.df.keys() if is_ins(col)]
        # unins_cols = [col for col in self.df.keys() if is_unins(col)]
        # keep_cols = set([remove_suffix(col) for col in ins_cols]).intersection(
        #     set([remove_suffix(col) for col in ins_cols])
        # ).union(common_cols)

        def _extract_row(row, should_keep):
            data = dict()
            for field in row.keys():
                if should_keep(field):
                    new_field = remove_suffix(field)
                    assert new_field not in data
                    data[new_field] = [row[field]]
                elif is_common_row(field):
                    assert field not in data
                    data[field] = [row[field]]
            return pd.DataFrame(data)
        for index, row in self.df.iterrows():
            ins_row = _extract_row(row, is_ins)
            unins_row = _extract_row(row, is_unins)
            rows.append(ins_row)
            rows.append(unins_row)
        df = pd.concat(rows)
        # Allows df.loc[row_index]
        df = df.reset_index()

        def config_pretty(config):
            m = re.search(r'^config_(?P<config>.*)', config)
            conf = m.group('config')
            def upper_repl_01(m):
                return ' ' + m.group(1).upper()
            conf = re.sub(r'_(\w)', upper_repl_01, conf)
            def upper_repl_02(m):
                return m.group(1).upper()
            conf = re.sub(r'^(\w)', upper_repl_02, conf)
            return conf

        df['config_pretty'] = df['config'].apply(config_pretty)

        plot_data_fields = ['x_field', 'total_trace_time_sec', 'profiling_overhead_percent', 'config_pretty']
        logging.info(pprint_msg(df[plot_data_fields]))
        ax = sns.barplot(x='x_field', y='total_trace_time_sec', hue='config_pretty', data=df)
        leg = ax.legend()
        leg.set_title(None)

        # PROBLEM: (a2c, half-cheetah) profile percent is shown as 188%, but it's actually 222...
        # 188 is the (ppo, half-cheetah) result...
        # TODO: index by x_field, retrieve x_field from plot/patches.

        def add_percent_bar_labels(df, ax):
            logging.info(pprint_msg({
                'len(patches)': len(ax.patches),
                'len(df)': len(df),
            }))
            xticklabels = ax.get_xticklabels()
            xticks = ax.get_xticks()
            ins_df = df[df['config'].apply(lambda config: bool(config_is_instrumented(config)))]
            bar_width = ax.patches[0].get_width()
            xticklabel_to_xtick = dict()
            for xtick, xticklabel in zip(xticks, xticklabels):
                xticklabel_to_xtick[xticklabel.get_text()] = xtick
            for i in range(len(ins_df)):
                row = ins_df.iloc[i]

                x_field = row['x_field']

                # Keep single decimal place.
                # bar_label = "{perc:.1f}%".format(
                #     perc=df.loc[i]['profiling_overhead_percent'])

                # Round to nearest percent.
                # bar_label = "{perc:.0f}%".format(
                #     perc=df.loc[i]['profiling_overhead_percent'])

                profiling_overhead_percent = row['profiling_overhead_percent']
                bar_label = "{perc:.0f}%".format(
                    perc=profiling_overhead_percent)

                total_trace_time_sec = row['total_trace_time_sec']

                xtick = xticklabel_to_xtick[x_field]
                pos = (xtick - bar_width / 2, total_trace_time_sec)

                logging.info(pprint_msg({
                    'bar_label': bar_label,
                    'x_field': x_field,
                    'pos': pos,
                    'total_trace_time_sec': total_trace_time_sec,
                }))

                ax.annotate(
                    bar_label,
                    pos,
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

        add_percent_bar_labels(df, ax)


        # groupby_cols = ['algo', 'env_id']
        # # label_df = self.df_gpu[list(set(groupby_cols + ['x_field', 'util']))]
        # label_df = self.df_gpu.groupby(groupby_cols).mean()
        # add_hierarchical_labels(fig, ax, self.df_gpu, label_df, groupby_cols)

        # df = self.df
        # ax = sns.violinplot(x=df['x_field'], y=100*df['util'],
        #                     # hue=df['algo'],
        #                     # hue=df['env_id'],
        #                     inner="box", cut=0.)

        if self.rotation is not None:
            # ax = bottom_plot.axes
            ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)

        # Remove legend-title that seaborn adds:
        # https://stackoverflow.com/questions/51579215/remove-seaborn-lineplot-legend-title?rq=1
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=labels[1:])

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

        png_path = self._get_plot_path('png')
        logging.info('Save figure to {path}'.format(path=png_path))
        fig.tight_layout()
        fig.savefig(png_path)
        plt.close(fig)

        # figlegend.tight_layout()
        # figlegend.savefig(self.legend_path, bbox_inches='tight', pad_inches=0)
        # plt.close(figlegend)

        return

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
            # import ipdb; ipdb.set_trace()
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
            ax2.yaxis.set_major_formatter(DaysHoursMinutesSecondsFormatter())

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

        if self.show_legend:
            self._add_legend(
                fig,
                loc='upper left',
                bbox_to_anchor=(1.05, 1),
            )

        # , prop={'size': self.fontsize}

        # topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
        # bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
        # l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})

        #Optional code - Make plot look nicer
        sns.despine(fig=fig, left=True)
        # bottom_plot.set_ylabel(self.y_axis_label)
        # bottom_plot.set_xlabel(self.x_axis_label)
        ax.set_ylabel(self.y_axis_label)
        if self.title is not None:
            # bottom_plot.set_title(self.title)
            ax.set_title(self.title)

        # Suggestions about how to prevent x-label overlap with matplotlib:
        #
        # https://stackoverflow.com/questions/42528921/how-to-prevent-overlapping-x-axis-labels-in-sns-countplot

        # #Set fonts to consistent 16pt size
        # for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
        #              bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        #     if self.fontsize is not None:
        #         item.set_fontsize(self.fontsize)

        ax.grid(zorder=grid_zorder)
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

        logging.info('Save figure to {path}'.format(path=self.path))
        fig.tight_layout()
        fig.savefig(self.path)
        plt.close(fig)

    @property
    def x_axis_label(self):
        if self.x_type == 'rl-comparison':
            return "(RL algorithm, Environment)"
        elif self.x_type == 'env-comparison':
            return "Environment"
        elif self.x_type == 'algo-comparison':
            return "RL algorithm"
        raise NotImplementedError

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

