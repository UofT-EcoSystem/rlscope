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

from iml_profiler.parser import stacked_bar_plots

from iml_profiler.profiler import iml_logging

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

        # OUTPUT raw thing here.
        logging.info("Output raw un-aggregated machine utilization data @ {path}".format(path=self._raw_csv_path))
        new_df.to_csv(self._raw_csv_path, index=False)

        df_agg = new_df.groupby(groupby_cols).agg(['min', 'max', 'mean', 'std'])
        flat_df_agg = self.flattened_agg_df(df_agg)

        # import ipdb; ipdb.set_trace()
        logging.info("Output min/max/std aggregated machine utilization data @ {path}".format(path=self._agg_csv_path))
        flat_df_agg.to_csv(self._agg_csv_path, index=False)

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
        return _j(self.directory, "UtilPlot{suffix}.{ext}".format(
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

        def is_cpu(device_name):
            if re.search(r'Intel|Xeon|CPU', device_name):
                return True
            return False

        def is_gpu(device_name):
            return not is_cpu(device_name)

        def should_keep(row):
            if row['machine_name'] == 'reddirtx-ubuntu':
                # Ignore 'Tesla K40c' (unused, 0 util)
                return row['device_name'] == 'GeForce RTX 2080 Ti'
            return True

        self.df_gpu = self.df

        self.df_gpu = self.df_gpu[self.df_gpu['device_name'].apply(is_gpu)]

        self.df_gpu = self.df_gpu[self.df_gpu.apply(should_keep, axis=1)]

        logging.info(pprint_msg(self.df_gpu))

        # ax = sns.violinplot(x=self.df_gpu['x_field'], y=100*self.df_gpu['util'],
        #                     inner="box",
        #                     # cut=0.,
        #                     )

        ax = sns.boxplot(x=self.df_gpu['x_field'], y=100*self.df_gpu['util'],
                         showfliers=False,
                         )

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
