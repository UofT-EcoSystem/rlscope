import logging
import argparse
import re
import importlib
import sys

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from iml_profiler.parser.db import SQLCategoryTimesReader, sql_get_source_files, sql_input_path
from iml_profiler.parser.plot_index import _DataIndex

from iml_profiler.parser.common import *

class OverlapStackedBarPlot:
    """
    Plot overlap data output by IML across SEVERAL runs where
    a run is an (algo, env_id) pair.

    Overlap data of the same "overlap_type" can be merged into a
    single stacked bar plot.

    For example, for overlap_type=ResourceOverlap
    (i.e. the CPU/GPU utilization plot), there would be a
    bar along the x-axis for each (algo, env_id) combination, and
    each bar would consits of the stacks [CPU, GPU].

    Some overlap_type's have data that doesn't overlap;
    for example OperationOverlap.

    However, some overlap_type's have data that DOES overlap:
    - ResourceOverlap: "CPU" subsumes "GPU"
    - CategoryOverlap: "Framework API C" subsumes "CUDA API C"
    For these cases where [CPU subsumes GPU] we'd like to show:

    ________________
    |     GPU      |
    |______________|
    |  CPU - GPU   |
    |______________|

     (algo, env_id)


    TODO: we can guess the (algo, env_id) pair name from the process_name or phase_name.
    May nee to add --algos --env-ids option in the future though.

    """
    def __init__(self,
                 iml_directories,
                 overlap_type,
                 host=None,
                 user=None,
                 password=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        """

        :param directory:
        :param debug:

        :param step_sec:
        :param pixels_per_square:
            `pixels_per_square` determines how many pixels are taken up by "step" units.
            `step_sec` determines how many seconds each utilization square represents.
            - step_sec should be bigger than the sampling frequency (500 ms)
            Q: Any reason we want to specify step and pixels_per_square separately?
            A: Yes I believe so:
            - Only increases pixels_per_square if we'd like the same "exponential average window", but just a larger generated image.
            - Only decrease/increase step_sec if we'd like to see a more fine/coarse grained "exponential average window"

        :param decay:
        :param kwargs:
        """
        self.overlap_type = overlap_type
        assert self.overlap_type in ['ResourceOverlap', 'OperationOverlap', 'CategoryOverlap']
        self.iml_directories = iml_directories
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug

    def _plot_data_path(self, device_id, device_name):
        return _j(self.directory, "util_scale.plot_data{dev}.txt".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def _json_path(self, device_id, device_name):
        return _j(self.directory, "util_scale{dev}.js_path.json".format(
            dev=device_id_suffix(device_id, device_name),
        ))

    def _get_algo_env_id(self, iml_dir):
        sql_reader = self.sql_readers[iml_dir]
        procs = sql_reader.process_names
        assert len(procs) == 1
        proc = procs[0]
        m = re.search(r'(?P<algo>[^_]+)_(?P<env_id>.+)', proc)
        algo = m.group('algo')
        env_id = m.group('env_id')
        return (algo, env_id)

    def get_index(self, iml_dir):
        sys.path.insert(0, iml_dir)
        iml_profiler_plot_index = importlib.import_module("iml_profiler_plot_index")
        index = iml_profiler_plot_index.DataIndex
        del sys.path[0]
        if 'iml_profiler_plot_index' in sys.modules:
            del sys.modules['iml_profiler_plot_index']
        return index

    def run(self):
        self.data_index = dict()
        self.sql_readers = dict()

        for iml_dir in self.iml_directories:
            self.sql_readers[iml_dir] = SQLCategoryTimesReader(self.db_path(iml_dir), host=self.host, user=self.user, password=self.password)
            index = self.get_index(iml_dir)
            idx = _DataIndex(index, iml_dir, debug=self.debug)
            self.data_index[iml_dir] = idx

        self.data = {
            'algo':[],
            'env':[],
        }
        self.all_labels = None
        for iml_dir in self.iml_directories:
            idx = self.data_index[iml_dir]
            # Q: There's only one ResourceOverlap plot...
            # However, there are many OperationOverlap plots; how can we select
            # among them properly?
            md, entry, ident = idx.get_file(selector={
                'overlap_type':self.overlap_type,
            })
            vd = VennData(entry['venn_js_path'])
            vd_dict = vd.as_dict()
            for label, size_us in vd_dict.items():
                if self.all_labels is None:
                    self.all_labels = set(vd_dict.keys())
                else:
                    # All venn_js_path files should have the same categories in them.
                    # If not, we must do something to make 1 file look like another.
                    assert label in self.all_labels
                if label not in self.data:
                    self.data[label] = []
                self.data[label].append(size_us)
        self.df = pd.DataFrame(self.data)

        # TODO: read data into this format:
        # data = {
        #     'algo':     [algo_1,          algo_2         ],
        #     'env':      [env_1,           env_2          ],
        #     'CPU':      [25,              50             ],
        #     'GPU':      [75,              50             ],
        # }
        # df = pandas.DataFrame(data)
        # # Transform df to handle the different "subsumes" relationships between categories.


        # for device_name, device_id in sql_reader.devices:
        #   samples = SELECT all utilization samples for <device_name>
        #   plotter.plot(samples) @ png=util_scale.<device_id>.png

        # The start time of all traced Events / utilization samples.
        # Use this as the "starting point" of the heat-scale.

        # TODO: How can we make things "match up" with the SummaryView?
        # Options:
        #
        # 1. Show utilization from [start_time_usec .. start_time_usec + duration_sec].
        #
        #    We currently use start_time_usec from events to decide on initial plot locations;
        #    so ideally we would sample the utilization samples running from:
        #    [start_time_usec .. start_time_usec + duration_sec] for each phase.
        #    PROBLEM:
        #    - actual time shown in subplots spans ~ 531 seconds;
        #    - total time according to utilization data is ~ 1244 seconds.
        #
        #    PRO: utilization will "appear" to match up with ResourceSubplot.
        #    CON: utilization will not actually match up with ResourceSubplot.
        #    PRO: easiest to implement.
        #
        # 2. "Condense" utilization from [subplot.start_time_usec .. subplot.end_time_usec] to fit within subplot height;
        #    don't show time-scale.
        #
        #    "Operations" may not capture all CPU/GPU activity.
        #    However, we can still show a condensed view of overall hardware utilization during that time.
        #
        #    CON: if "operations" are wrong, maybe we didn't capture high activity and so our ResourceSubplot subplot
        #         will look wrong in comparison.
        #    PRO: true hardware utilization is shown for "some" interval of time
        #
        # 3. Only keeps utilization samples that "match up" with the spans of time that make up our ResourceSubplot plots.
        #    PRO: ResourceOverlap and HeatScale will "match up".
        #    CON: We may be missing actual hardware utilization (programmer annotations are wrong)
        #    CON: time-consuming to implement.

        start_time_sec = self.sql_reader.trace_start_time_sec
        for device in self.sql_reader.util_devices:
            samples = self.sql_reader.util_samples(device)
            # png = self.get_util_scale_png(device.device_id, device.device_name)
            # plotter = HeatScale(
            #     color_value='util', y_axis='start_time_sec',
            #     png=png,
            #     pixels_per_square=self.pixels_per_square,
            #     # Anchor colormap colors using min/max utilization values.
            #     vmin=0.0, vmax=1.0,
            #     # 1 second
            #     step=1.)

            if self.debug:
                # Print the unadjusted raw utilization + timestamp data, centered @ start_time_sec.
                raw_centered_time_secs = (np.array(samples['start_time_sec']) - start_time_sec).tolist()
                raw_df = pd.DataFrame({
                    'util':samples['util'],
                    'start_time_sec':raw_centered_time_secs,
                }).astype(float)
                logging.info("> DEBUG: Unadjusted raw utilization measurements for device={dev}".format(dev=device))
                logging.info(raw_df)
            norm_time_secs, norm_utils = exponential_moving_average(
                samples['start_time_sec'], samples['util'],
                start_time_sec, self.step_sec, self.decay)
            centered_time_secs = (np.array(norm_time_secs) - start_time_sec).tolist()
            norm_samples = {
                'util':norm_utils,
                'start_time_sec':centered_time_secs,
            }
            plot_df = pd.DataFrame(norm_samples).astype(float)
            self.dump_plot_data(plot_df, device)
            self.dump_js_data(norm_samples, device, start_time_sec)
            # plotter.add_data(norm_samples)
            # print("> HeatScalePlot @ {path}".format(path=png))
            # plotter.plot()

    def dump_js_data(self, norm_samples, device, start_time_sec):
        js = {
            'metadata': {
                'plot_type': 'HeatScale',
                'device_id': device.device_id,
                'device_name': device.device_name,
                'start_time_usec': float(start_time_sec)*MICROSECONDS_IN_SECOND,
                'step_usec': self.step_sec*MICROSECONDS_IN_SECOND,
                'decay': self.decay,
            },
            'data': {
                'util': norm_samples['util'],
                'start_time_sec': norm_samples['start_time_sec'],
            },
        }
        path = self._json_path(device.device_id, device.device_name)
        print("> HeatScalePlot @ plot data @ {path}".format(path=path))
        do_dump_json(js, path)

    def dump_plot_data(self, plot_df, device):
        path = self._plot_data_path(device.device_id, device.device_name)
        print("> HeatScalePlot @ plot data @ {path}".format(path=path))
        with open(path, 'w') as f:
            DataFrame.print_df(plot_df, file=f)
        logging.info(plot_df)

    def db_path(self, iml_dir):
        return sql_input_path(self.iml_dir)

class VennData:
    def __init__(self, path):
        with open(path) as f:
            self.venn = json.load(f)
        self._build_idx_to_label()

    @property
    def total_size(self):
        total_size = 0.
        # [ size of all regions ] - [ size of overlap regions ]
        for venn_set in self.venn['venn']:
            if len(venn_set['sets']) > 1:
                # Overlap region is JUST the size of the overlap.
                total_size -= venn_set['size']
            else:
                # Single 'set' is the size of the WHOLE region (INCLUDING overlaps)
                assert len(venn_set['sets']) == 1
                total_size += venn_set['size']
        return total_size

    def _build_idx_to_label(self):
        self.idx_to_label = dict()
        for venn_data in self.venn['venn']:
            if len(venn_data['sets']) == 1:
                assert 'label' in venn_data:
                idx = venn_data['sets'][0]
                self.idx_to_label[idx] = venn_data['label']

    def _indices_to_labels(self, indices):
        return sorted(tuple(self.idx_to_label[i] for i in indices))

    def as_dict(self):
        """
        {
            ('CPU',): 135241018.0,
            ('GPU',): 3230025.0,
            ('CPU', 'GPU'): 3230025.0,
        }
        """
        d = dict()
        for venn_data in self.venn['venn']:
            labels = self._indices_to_labels(venn_data['sets'])
            size_us = venn_data['size']
            assert labels not in d
            d[labels] = size_us
        return d

def test_stacked_bar():
    df = pd.DataFrame(dict(
        A=[1, 2, 3, 4],
        B=[2, 3, 4, 5],
        C=[3, 4, 5, 6],
        D=[4, 5, 6, 7]))

    fig = plt.figure(figsize=(20, 10))

    ab_bar_list = [plt.bar([0, 1, 2, 3], df.B, align='edge', width= 0.2),
                   plt.bar([0, 1, 2, 3], df.A, align='edge', width= 0.2)]

    cd_bar_list = [plt.bar([0, 1, 2, 3], df.D, align='edge',width= -0.2),
                   plt.bar([0, 1, 2, 3], df.C, align='edge',width= -0.2)]

    path = './test_stacked_bar.png'
    logging.info('Save figure to {path}'.format(path=path))
    plt.savefig(path)

def test_stacked_bar_sns():
    """
    https://randyzwitch.com/creating-stacked-bar-chart-seaborn/

    Stacked-bar chart is really just "overlaying" bars on top of each other.

    :return:
    """
    #Read in data & create total column
    data = {
        'Group': [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ],
        'Series1':[
            3.324129347,
            3.109298649,
            3.603703815,
            5.030113742,
            6.555816091,
            7.478125262,
            8.201300407,
            8.306264399,
            6.622167472,
            4.272699487,
            2.393412671,
            1.228434178,
            0.611171616,
            0.30351888,
            0.151165323,
            0.077035502,
            0.039631096,
            0.021068686,
            0.011522874,
            0.006190799,
        ],
        'Series2': [
            8.733030097,
            5.621414891,
            4.830770027,
            4.84697594,
            4.126257144,
            3.321003992,
            2.667565198,
            1.976072963,
            1.220086585,
            0.656404386,
            0.32202138,
            0.15599814,
            0.076945636,
            0.039381467,
            0.021178522,
            0.011502903,
            0.006440428,
            0.003664553,
            0.002106869,
            0.001308056,
        ],
    }
    stacked_bar_data = pd.DataFrame(data)

    # stacked_bar_data = pd.read_csv("C:\stacked_bar.csv")
    stacked_bar_data["total"] = stacked_bar_data.Series1 + stacked_bar_data.Series2

    #Set general plot properties
    sns.set_style("white")
    sns.set_context({"figure.figsize": (24, 10)})

    #Plot 1 - background - "total" (top) series
    sns.barplot(x = stacked_bar_data.Group, y = stacked_bar_data.total, color = "red")

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x = stacked_bar_data.Group, y = stacked_bar_data.Series1, color = "#0000A3")


    topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
    bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
    l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})
    l.draw_frame(False)

    #Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("Y-axis label")
    bottom_plot.set_xlabel("X-axis label")

    #Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(16)

    path = './test_stacked_bar_sns.png'
    logging.info('Save figure to {path}'.format(path=path))
    plt.savefig(path)

def main():
    parser = argparse.ArgumentParser(
        textwrap.dedent("""\
        Test plots
        """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--test-stacked-bar',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot multi-bar stacked bar-chart.
    """))


    parser.add_argument('--test-stacked-bar-sns',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot multi-bar stacked bar-chart.
    """))

    args = parser.parse_args()

    if args.test_stacked_bar:
        test_stacked_bar()
    elif args.test_stacked_bar_sns:
        test_stacked_bar_sns()

if __name__ == '__main__':
    main()
