import logging
import argparse
import re
import copy
import importlib
import sys

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.parser.db import SQLCategoryTimesReader, sql_get_source_files, sql_input_path
from iml_profiler.parser.plot_index import _DataIndex

from iml_profiler.parser.common import *

USEC_IN_SEC = 1e6

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
    SUPPORTED_OVERLAP_TYPES = ['ResourceOverlap', 'OperationOverlap', 'CategoryOverlap']
    SUPPORTED_X_TYPES = ['algo-comparison', 'env-comparison', 'rl-comparison']
    SUPPORTED_Y_TYPES = ['percent', 'seconds']

    def __init__(self,
                 iml_directories,
                 directory,
                 overlap_type,
                 resource_overlap=None,
                 ignore_inconsistent_overlap_regions=False,
                 title=None,
                 x_type='rl-comparison',
                 y_type='percent',
                 show_title=True,
                 show_legend=True,
                 keep_zero=True,
                 suffix=None,
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
        assert overlap_type in OverlapStackedBarPlot.SUPPORTED_OVERLAP_TYPES
        assert x_type in OverlapStackedBarPlot.SUPPORTED_X_TYPES
        assert y_type in OverlapStackedBarPlot.SUPPORTED_Y_TYPES
        if len(iml_directories) == 0:
            raise ValueError("OverlapStackedBarPlot expects at least 1 trace-file directory for iml_directories")
        self.iml_directories = iml_directories
        self.directory = directory
        self.overlap_type = overlap_type
        self.resource_overlap = resource_overlap
        self.ignore_inconsistent_overlap_regions = ignore_inconsistent_overlap_regions
        if self.resource_overlap is not None:
            # Normalize ('GPU', 'CPU') into ('CPU', 'GPU') for equality checks,
            self.resource_overlap = tuple(sorted(self.resource_overlap))
        self.title = title
        self.x_type = x_type
        self.y_type = y_type
        self.show_title = show_title
        self.show_legend = show_legend
        self.keep_zero = keep_zero
        self.suffix = suffix
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug

    def _get_plot_path(self, ext):
        if self.suffix is not None:
            suffix_str = '.{suffix}'.format(self.suffix)
        else:
            suffix_str = ''
        return _j(self.directory, "OverlapStackedBarPlot.overlap_type_{ov}{suffix}.{ext}".format(
            ov=self.overlap_type,
            suffix=suffix_str,
            ext=ext,
        ))

    @property
    def _plot_data_path(self):
        return self._get_plot_path(ext='txt')

    @property
    def _plot_path(self):
        return self._get_plot_path(ext='png')

    # def _json_path(self, device_id, device_name):
    #     return _j(self.directory, "util_scale{dev}.js_path.json".format(
    #         dev=device_id_suffix(device_id, device_name),
    #     ))

    def _get_algo_or_env(self, algo_or_env):
        assert algo_or_env in {'env', 'algo'}

        def _get_value(iml_dir):
            algo, env = self._get_algo_env_from_dir(iml_dir)
            if algo_or_env == 'algo':
                value = algo
            else:
                value = env
            return value

        if algo_or_env == 'algo':
            field = 'algorithm'
        else:
            field = 'environment'

        values = set()
        for iml_dir in self.iml_directories:
            value = _get_value(iml_dir)
            if len(values) == 1 and value not in values:
                raise RuntimeError("Expected {field}={expect} but saw {field}={saw} for --iml-directory {dir}".format(
                    field=field,
                    expect=list(values)[0],
                    saw=value,
                    dir=iml_dir,
                ))
            values.add(value)
        assert len(values) == 1
        return list(values)[0]

    @property
    def algorithm(self):
        return self._get_algo_or_env('algo')

    @property
    def environment(self):
        return self._get_algo_or_env('env')

    @property
    def plot_x_axis_label(self):
        if self.x_type == 'rl-comparison':
            return "(RL algorithm, Environment)"
        elif self.x_type == 'env-comparison':
            return "Environment"
        elif self.x_type == 'algo-comparison':
            return "RL algorithm"
        raise NotImplementedError

    @property
    def plot_y_axis_label(self):
        if self.y_type == 'percent':
            return "Time breakdown (percent)"
        elif self.y_type == 'seconds':
            return "Time breakdown (seconds)"
        raise NotImplementedError

    @property
    def plot_title(self):
        if self.title is not None:
            return self.title

        if not self.show_title:
            return None

        if self.x_type == 'rl-comparison':
            title = "Comparing RL workloads"
        elif self.x_type == 'env-comparison':
            title = "Comparing environments when training {algorithm}".format(algorithm=self.algorithm)
        elif self.x_type == 'algo-comparison':
            title = "Comparing algorithms when training {environment}".format(environment=self.environment)
        else:
            raise NotImplementedError

        return title

    def _get_algo_env_id(self, iml_dir):
        sql_reader = self.sql_readers[iml_dir]
        procs = sql_reader.process_names
        assert len(procs) == 1
        proc = procs[0]
        m = re.search(r'(?P<algo>[^_]+)_(?P<env_id>.+)', proc)
        algo = m.group('algo')
        env_id = m.group('env_id')
        return (algo, env_id)

    def _get_algo_env_from_dir(self, iml_dir):
        # .../<algo>/<env_id>
        path = os.path.normpath(iml_dir)
        components = path.split(os.sep)
        env_id = components[-1]
        algo = components[-2]
        return (algo, env_id)

    def get_index(self, iml_dir):

        def _del_module(import_path):
            if import_path in sys.modules:
                del sys.modules[import_path]

        def _del_index_module():
            _del_module('iml_profiler_plot_index')
            _del_module('iml_profiler_plot_index_data')


        _del_index_module()

        sys.path.insert(0, iml_dir)
        iml_profiler_plot_index = importlib.import_module("iml_profiler_plot_index")
        index = iml_profiler_plot_index.DataIndex
        del sys.path[0]

        _del_index_module()

        # Check that env_id's match
        assert _b(iml_dir) == _b(index.directory)

        # return index

        # NOTE: if trace-files were processed on a different machine, the _DataIndex.directory will be different;
        # handle this by re-creating the _DataIndex with iml_dir.
        # my_index = _DataIndex(index.index, iml_dir, debug=self.debug)
        my_index = _DataIndex(index.index, iml_dir)
        return my_index

    def _init_directories(self):
        """
        Initialize SQL / DataIndex needed for reading plot-data from iml-analyze'd --iml-directory's.

        :return:
        """
        self.data_index = dict()
        self.sql_readers = dict()

        for iml_dir in self.iml_directories:
            self.sql_readers[iml_dir] = SQLCategoryTimesReader(self.db_path(iml_dir), host=self.host, user=self.user, password=self.password)
            index = self.get_index(iml_dir)
            self.data_index[iml_dir] = index

    def _add_or_suggest_selector_field(self, idx, selector, field_name):
        """
        For e.g. field_name = 'resource_overlap' (['CPU'], or ['CPU', 'GPU']).

        If they provided --resource-overlap:
          Use that as selector['resource_overlap']
        Else If there's only available choice for --resource-overlap
          Use that.
        Else:
          Tell the user about the available choices for --resource-overlap.

        :param field_name:
            e.g. 'resource_overlap' for --resource-overlap
        """

        def optname(option):
            return "--{s}".format(s=re.sub(r'_', '-', option))

        # Oops, forgot to add self.resource_overlap in constructor.
        assert hasattr(self, field_name)

        value = getattr(self, field_name)
        if value is not None:
            selector[field_name] = value
        else:
            choices = idx.available_values(selector, field_name)
            # NOTE: if there's only one choice, we don't need to add it to selector.
            if len(choices) > 1:
                raise RuntimeError("Please provide {opt}: choices = {choices}".format(
                    opt=optname(field_name),
                    choices=choices,
                ))
                selector[field_name] = avail[0]

    def each_stacked_dict(self):
        for iml_dir in self.iml_directories:
            idx = self.data_index[iml_dir]

            algo, env = self._get_algo_env_from_dir(iml_dir)

            # Q: There's only one ResourceOverlap plot...
            # However, there are many OperationOverlap plots; how can we select
            # among them properly?
            selector = {
                'overlap_type': self.overlap_type,
            }

            self._add_or_suggest_selector_field(idx, selector, 'resource_overlap')

            md, entry, ident = idx.get_file(selector=selector)
            vd = VennData(entry['venn_js_path'])
            stacked_dict = vd.stacked_bar_dict()
            path = entry['venn_js_path']
            yield (algo, env), path, stacked_dict

    def _check_overlap_json_files(self):
        regions_to_paths = dict()
        for (algo, env), path, stacked_dict in self.each_stacked_dict():
            regions = frozenset(stacked_dict.keys())
            if regions not in regions_to_paths:
                regions_to_paths[regions] = []
            regions_to_paths[regions].append(path)
        for k in list(regions_to_paths.keys()):
            regions_to_paths[k].sort()

        if len(regions_to_paths) > 1:
            """
            Output an error message:
            
            ERROR: *.venn_js.json files have inconsistent overlap regions:
            - overlap-regions (1):
              - regions: [...]
              - *.venn_js.json files:
                path/to/file_01.json
                path/to/file_02.json
            - overlap-regions (2):
              ...
            """
            err_lines = []

            if self.ignore_inconsistent_overlap_regions:
                msg_type = 'WARNING'
            else:
                msg_type = 'ERROR'

            err_lines.append("{msg_type}: *.venn_js.json files have inconsistent overlap regions:".format(
                msg_type=msg_type,
            ))
            for i, (regions, paths) in enumerate(regions_to_paths.items(), start=1):
                err_lines.append("- overlap region ({i})".format(i=i))
                err_lines.append("  - regions: {regions}".format(regions=sorted(regions)))
                err_lines.append("  - *.venn_js.json files:")
                for path in paths:
                    err_lines.append("    {path}".format(path=path))
            msg = '\n'.join(err_lines)
            if self.ignore_inconsistent_overlap_regions:
                raise RuntimeError(msg)
            else:
                logging.info(msg)

        regions_by_num_files = sorted(
            regions_to_paths.keys(),
            key=lambda regions: (len(regions_to_paths[regions]), regions_to_paths[regions]))
        use_overlap_regions = regions_by_num_files[-1]
        if self.debug:
            logging.info(pprint_msg({
                'regions_to_paths': regions_to_paths,
                'use_overlap_regions': use_overlap_regions}))
        return use_overlap_regions

    def _read_df(self):
        """
        Read venn_js data of several --iml-directory's into a single data-frame.

        :return:
        """
        self.data = {
            'algo':[],
            'env':[],
        }
        self.all_groups = None

        use_overlap_regions = self._check_overlap_json_files()

        for (algo, env), path, stacked_dict in self.each_stacked_dict():

            overlap_regions = set(stacked_dict.keys())
            if overlap_regions != use_overlap_regions:
                logging.info("Skipping {path} (--ignore-inconsistent-overlap-regions)".format(
                    path=path,
                ))
                continue

            if self.debug:
                logging.info(pprint_msg({
                    'path': path,
                    'stacked_dict': stacked_dict}))

            self.data['algo'].append(algo)
            self.data['env'].append(env)
            # Add each overlap region to self.data.
            # e.g. self.data[('CPU', 'GPU')].append(time in seconds)
            for group, size_us in stacked_dict.items():
                if self.all_groups is None:
                    self.all_groups = set(stacked_dict.keys())
                else:
                    # All venn_js_path files should have the same categories in them.
                    # If not, we must do something to make 1 file look like another.
                    assert group in self.all_groups
                if group not in self.data:
                    self.data[group] = []
                self.data[group].append(size_us)
        self.df = pd.DataFrame(self.data)

    def _normalize_df(self):
        """
        Transform raw venn_js file into appropriate units for plotting
        (i.e. convert us to seconds).
        """
        def transform_usec_to_sec(df, group):
            return df[group]/USEC_IN_SEC

        def transform_usec_to_percent(df):
            """
            GOAL:
            CPU = CPU / ([CPU] + [CPU, GPU] + [GPU])

            :param df:
            :param group:
            :return:
            """

            # e.g.
            # summation = [CPU] + [CPU, GPU] + [GPU]
            summation = None
            for g in self.all_groups:
                if summation is None:
                    summation = df[g]
                else:
                    summation = summation + df[g]

            ret = dict()
            for g in self.all_groups:
                # e.g.
                # CPU = CPU / ([CPU] + [CPU, GPU] + [GPU])
                ret[g] = 100 * df[g]/summation

            return ret

        if self.y_type == 'seconds':
            for group in self.all_groups:
                self.df[group] = transform_usec_to_sec(self.df, group)
        elif self.y_type == 'percent':
            ret = transform_usec_to_percent(self.df)
            for group in ret.keys():
                self.df[group] = ret[group]
        else:
            raise NotImplementedError

    def _plot_df(self):
        logging.info("Dataframe:\n{df}".format(df=self.df))

        def group_to_label(group):
            label = ' + '.join(group)
            return label
        x_fields = []
        for index, row in self.df.iterrows():
            x_field = "({algo}, {env})".format(
                algo=row['algo'],
                env=row['env'],
            )
            x_fields.append(x_field)
        self.df['x_field'] = x_fields

        stacked_bar_plot = StackedBarPlot(
            data=self.df,
            path=self._plot_path,
            groups=sorted(self.all_groups),
            x_field='x_field',
            x_axis_label=self.plot_x_axis_label,
            y_axis_label=self.plot_y_axis_label,
            title=self.plot_title,
            show_legend=self.show_legend,
            keep_zero=self.keep_zero,
            # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
            group_to_label=group_to_label,
        )
        self.dump_plot_data(self.df)
        stacked_bar_plot.plot()

    def run(self):
        self._init_directories()
        self._read_df()
        self._normalize_df()
        self._plot_df()

        # self.dump_js_data(norm_samples, device, start_time_sec)
        # plotter.add_data(norm_samples)
        # print("> HeatScalePlot @ {path}".format(path=png))
        # plotter.plot()

        # TODO: read data into this format:
        # data = {
        #     'algo':     [algo_1,          algo_2         ],
        #     'env':      [env_1,           env_2          ],
        #     'CPU':      [25,              50             ],
        #     'GPU':      [75,              50             ],
        # }
        # df = pandas.DataFrame(data)
        # # Transform df to handle the different "subsumes" relationships between categories.

    # def dump_js_data(self, norm_samples, device, start_time_sec):
    #     js = {
    #         'metadata': {
    #             'plot_type': 'OverlappedStackedBar',
    #             # 'device_id': device.device_id,
    #             # 'device_name': device.device_name,
    #             # 'start_time_usec': float(start_time_sec)*MICROSECONDS_IN_SECOND,
    #             # 'step_usec': self.step_sec*MICROSECONDS_IN_SECOND,
    #             # 'decay': self.decay,
    #         },
    #         'data': {
    #             'util': norm_samples['util'],
    #             'start_time_sec': norm_samples['start_time_sec'],
    #         },
    #     }
    #     path = self._json_path(device.device_id, device.device_name)
    #     print("> HeatScalePlot @ plot data @ {path}".format(path=path))
    #     do_dump_json(js, path)

    def dump_plot_data(self, plot_df):
        path = self._plot_data_path
        logging.info("> {name} @ plot data @ {path}".format(
            name=self.__class__.__name__,
            path=path))
        with open(path, 'w') as f:
            DataFrame.print_df(plot_df, file=f)
        logging.info(plot_df)

    def db_path(self, iml_dir):
        return sql_input_path(iml_dir)

class VennData:
    """
    Regarding venn_js format.

    The "size" of each "circle" in the venn diagram is specified in a list of regions:
    # ppo2/HumanoidBulletEnv-v0/ResourceOverlap.process_ppo2_HumanoidBulletEnv-v0.phase_ppo2_HumanoidBulletEnv-v0.venn_js.json
    self.venn['venn'] = {
        {
            "label": "CPU",
            "sets": [
                0
            ],
            # This the size of the blue "CPU" circle.
            "size": 117913583.0
        },
        {
            "label": "GPU",
            "sets": [
                1
            ],
            # This the size of the orange "GPU" circle.
            "size": 1096253.0
        },
        {
            "sets": [
                0,
                1
            ],
            # This the size of the region of intersection between the "CPU" and "GPU" circles.
            # Since CPU time SUBSUMES GPU time, this is the same as the GPU time.
            "size": 1096253.0
        }
    }

    NOTE:
    - In our current diagram, CPU-time SUBSUMES GPU-time.
    - [CPU, GPU] time is time where both the CPU AND the GPU are being used.
    - If you just want to know CPU ONLY time (i.e. no GPU in-use), you must compute:
      [CPU only] = [CPU] - [CPU, GPU]
    - Since stacked-bar charts cannot show overlap between adjacent squares, we need to
      create a separate stacked-bar "label" for each combination of resource overlaps.
    - In order to obtain this stack-bar data, we want to compute:
      - 'CPU' = [CPU only]
      - 'GPU' = [GPU only]
      - 'CPU + GPU' = [CPU-GPU only]
    - In our example, we have:
      - 'CPU' = 117913583.0 - (any group that has 'CPU' in it)
              = 117913583.0 - 1096253.0
      - 'GPU' = 1096253.0 - 1096253.0
              = 0
      - 'CPU + GPU' = 1096253.0 - (any group that has BOTH 'CPU' and 'GPU' in it, but NOT 'CPU'/'GPU' only)
    """
    def __init__(self, path):
        with open(path) as f:
            self.venn = json.load(f)
        self._build_idx_to_label()

    def stacked_bar_dict(self):
        """
        In order to obtain stack-bar data, we must compute:
        - 'CPU' = [CPU only]
        - 'GPU' = [GPU only]
        - 'CPU + GPU' = [CPU-GPU only]

        See VennData NOTE above for details.
        """
        venn_dict = self.as_dict()
        stacked_dict = dict()
        # e.g. group = ['CPU']
        for group in venn_dict.keys():
            # Currently, stacked_dic['CPU'] includes overlap time from ['CPU', 'GPU']
            stacked_dict[group] = venn_dict[group]
            # e.g. member = ['CPU']
            for other_group in venn_dict.keys():
                # e.g. ['CPU'] subset-of ['CPU', 'GPU']
                if group != other_group and set(group).issubset(set(other_group)):
                    stacked_dict[group] = stacked_dict[group] - venn_dict[other_group]
        return stacked_dict

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
                assert 'label' in venn_data
                idx = venn_data['sets'][0]
                self.idx_to_label[idx] = venn_data['label']

    def _indices_to_labels(self, indices):
        return tuple(sorted(self.idx_to_label[i] for i in indices))

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

class StackedBarPlot:
    def __init__(self,
                 data, path,
                 groups=None,
                 x_field=None,
                 x_axis_label="X-axis label",
                 y_axis_label="Y-axis label",
                 title=None,
                 show_legend=True,
                 keep_zero=True,
                 rotation=10,
                 # fontsize=16,
                 fontsize=None,
                 group_to_label=None):
        assert groups is not None
        assert x_field is not None
        self.data = pd.DataFrame(data)
        self.path = path
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.title = title
        self.show_legend = show_legend
        self.keep_zero = keep_zero
        self.rotation = rotation
        self.fontsize = fontsize
        self.groups = groups
        self.x_field = x_field
        self.fontsize = fontsize
        self.group_to_label = group_to_label

    def _group_to_label(self, group):
        if self.group_to_label is not None:
            label = self.group_to_label(group)
            return label
        label = group
        return label

    def _all_zero(self, group):
        return (self.data[group] == 0.).all()

    def _add_legend(self):
        legend_rects = []
        for i, group in enumerate(self.groups):
            if not self.keep_zero and self._all_zero(group):
                # If a stacked-bar element is zero in all the bar-charts, don't show it in the legend (--keep-zero false).
                continue
            legend_rect = plt.Rectangle((0, 0), 1, 1, fc=self.colors[i], edgecolor='none')
            legend_rects.append(legend_rect)
        legend_labels = [self._group_to_label(group) for group in self.groups]
        # leg = plt.legend(legend_rects, legend_labels, loc=1, ncol=2)
        leg = plt.legend(legend_rects, legend_labels,
                         bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        leg.draw_frame(False)

    def plot(self):

        #Set general plot properties

        sns.set_style("white")
        # fig = plt.figure()
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

        self.colors = sns.color_palette("hls", len(self.groups))

        n_bars = len(self.data[self.groups[0]])
        accum_ys = np.zeros(n_bars)
        barplot_kwargs = []
        for i, group in enumerate(self.groups):
            accum_ys += self.data[group]
            ys = copy.copy(accum_ys)
            barplot_kwargs.append(
                {
                    'x': self.data[self.x_field],
                    'y': ys,
                    'color': self.colors[i]
                }
            )

        barplots = []
        for kwargs in reversed(barplot_kwargs):
            # TODO: color?
            barplot = sns.barplot(**kwargs)
            barplots.append(barplot)
        barplots.reverse()

        # #Plot 1 - background - "total" (top) series
        # sns.barplot(x = self.data.Group, y = self.data.total, color = "red")
        #
        # #Plot 2 - overlay - "bottom" series
        # bottom_plot = sns.barplot(x = self.data.Group, y = self.data.Series1, color = "#0000A3")

        bottom_plot = barplots[-1]

        if self.show_legend:
            self._add_legend()

        # , prop={'size': self.fontsize}

        # topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
        # bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')
        # l = plt.legend([bottombar, topbar], ['Bottom Bar', 'Top Bar'], loc=1, ncol = 2, prop={'size':16})

        #Optional code - Make plot look nicer
        sns.despine(left=True)
        bottom_plot.set_ylabel(self.y_axis_label)
        bottom_plot.set_xlabel(self.x_axis_label)
        if self.title is not None:
            bottom_plot.set_title(self.title)


        if self.rotation is not None:
            ax = bottom_plot.axes
            ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)

        # Suggestions about how to prevent x-label overlap with matplotlib:
        #
        # https://stackoverflow.com/questions/42528921/how-to-prevent-overlapping-x-axis-labels-in-sns-countplot

        # #Set fonts to consistent 16pt size
        # for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
        #              bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        #     if self.fontsize is not None:
        #         item.set_fontsize(self.fontsize)

        logging.info('Save figure to {path}'.format(path=self.path))
        plt.tight_layout()
        plt.savefig(self.path)

def test_stacked_bar_sns():
    """
    https://randyzwitch.com/creating-stacked-bar-chart-seaborn/

    Stacked-bar chart is really just "overlaying" bars on top of each other.

    :return:
    """

    # # Plot parameters
    # x_axis_label = "X-axis label"
    # y_axis_label = "Y-axis label"
    # # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
    # groups = ['Series1', 'Series2']
    def group_to_label(group):
        '''
        Convert a "key" representing a stack into a legend-label.

        :param group:
            e.g. ('CPU',) => 'CPU'
        :return:
        '''
        return group
    # # The value to use for the x-axis
    # x_field = 'Group'
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

    stacked_bar_plot = StackedBarPlot(
        data=data,
        path='./test_stacked_bar_sns.png',
        groups=['Series1', 'Series2'],
        x_field='Group',
        x_axis_label="X-axis label",
        y_axis_label="Y-axis label",
        # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
        group_to_label=group_to_label,
    )
    stacked_bar_plot.plot()

def test_stacked_bar_sns_old():
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

    path = './test_stacked_bar_sns_old.png'
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

    parser.add_argument('--test-stacked-bar-sns-old',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot multi-bar stacked bar-chart.
    """))

    args = parser.parse_args()

    if args.test_stacked_bar:
        test_stacked_bar()
    elif args.test_stacked_bar_sns:
        test_stacked_bar_sns()
    elif args.test_stacked_bar_sns_old:
        test_stacked_bar_sns_old()

if __name__ == '__main__':
    main()
