import logging
import argparse
import re
import copy
import importlib
import sys
import textwrap
import copy

from matplotlib import ticker as mpl_ticker
import matplotlib
# NOTE: If we don't do this, then with ForwardX11 enabled in ~/.ssh/config we get an error on python script exit:
#   XIO:  fatal IO error 0 (Success) on X server "localhost:10.0"
#         after 348 requests (348 known processed) with 1 events remaining.
matplotlib.use('agg')

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.parser.db import SQLCategoryTimesReader, sql_get_source_files, sql_input_path
from iml_profiler.parser.plot_index import _DataIndex
from iml_profiler.parser import plot_index

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
    SUPPORTED_OVERLAP_TYPES = ['ResourceOverlap', 'OperationOverlap', 'CategoryOverlap']
    SUPPORTED_X_TYPES = ['algo-comparison', 'env-comparison', 'rl-comparison']
    SUPPORTED_Y_TYPES = ['percent', 'seconds']

    def __init__(self,
                 iml_directories,
                 directory,
                 overlap_type,
                 resource_overlap=None,
                 operation=None,
                 training_time=False,
                 remap_df=None,
                 y2_logscale=False,
                 ignore_inconsistent_overlap_regions=False,
                 skip_plot=False,
                 title=None,
                 rotation=None,
                 x_type='rl-comparison',
                 y_type='percent',
                 show_title=True,
                 show_legend=True,
                 width=None,
                 height=None,
                 long_env=False,
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
        self.training_time = training_time
        self.should_add_training_time = training_time
        self.directory = directory
        self.overlap_type = overlap_type
        self.resource_overlap = resource_overlap
        self.operation = operation
        self.y2_logscale = y2_logscale
        self.remap_df = remap_df
        self.ignore_inconsistent_overlap_regions = ignore_inconsistent_overlap_regions
        self.skip_plot = skip_plot
        if self.resource_overlap is not None:
            # Normalize ('GPU', 'CPU') into ('CPU', 'GPU') for equality checks,
            self.resource_overlap = tuple(sorted(self.resource_overlap))
        self.title = title
        self.rotation = rotation
        self.x_type = x_type
        self.y_type = y_type
        self.show_title = show_title
        self.show_legend = show_legend
        self.width = width
        self.height = height
        self.long_env = long_env
        self.keep_zero = keep_zero
        self.suffix = suffix
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug

    def _get_plot_path(self, ext):
        if self.suffix is not None:
            suffix_str = '.{suffix}'.format(suffix=self.suffix)
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
    def _plot_csv_path(self):
        return self._get_plot_path(ext='csv')

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
        # if self.y_type == 'percent':
        #     return "Time breakdown (percent)"
        # elif self.y_type == 'seconds':
        #     return "Time breakdown (seconds)"
        # raise NotImplementedError
        return "Percent (%)"

    @property
    def plot_y2_axis_label(self):
        if self.y2_logscale:
            return "Total training time (log2-scale)"
        else:
            return "Total training time (sec)"

        return "Total training time"

    @property
    def plot_title(self):
        if self.title is not None:
            return self.title

        if not self.show_title:
            return None

        if self.x_type == 'rl-comparison':
            title = "Comparing RL workloads"
        elif self.x_type == 'env-comparison':
            title = "Comparing environments when training {algo}".format(algo=self.algorithm)
        elif self.x_type == 'algo-comparison':
            title = "Comparing algorithms when training {env}".format(env=self.get_x_env(self.environment))
        else:
            raise NotImplementedError

        return title

    def _get_algo_env_id(self, iml_dir):
        sql_reader = self.sql_readers[iml_dir]
        procs = sql_reader.process_names()
        assert len(procs) == 1
        proc = procs[0]
        m = re.search(r'(?P<algo>[^_]+)_(?P<env_id>.+)', proc)
        algo = m.group('algo')
        env_id = m.group('env_id')
        env_id = self._reduce_env(env_id)
        return (algo, env_id)

    def _get_algo_env_from_dir(self, iml_dir):
        # .../<algo>/<env_id>
        path = os.path.normpath(iml_dir)
        components = path.split(os.sep)
        env_id = components[-1]
        algo = components[-2]
        env_id = self._reduce_env(env_id)
        return (algo, env_id)

    def _reduce_env(self, env_id):
        """
        We want to treat these two environments as the same during analysis:
        LunarLander-v2
        LunarLanderContinuous-v2

        HACK: just remove "Continuous"

        :param env_id:
        :return:
        """
        new_env = env_id
        new_env = re.sub(r'LunarLanderContinuous-v2', 'LunarLander-v2', new_env)
        return new_env

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

        # Check that env_id's match; otherwise, warn the user.
        # This can happen when generating trace files on one machine, and changing the directory structure on another.
        if _b(iml_dir) != _b(index.directory):
            logging.warning("iml_dir={iml_dir} != index.directory={index_dir}; make sure these paths use the same (algo, env)!".format(
                iml_dir=iml_dir,
                index_dir=index.directory,
            ))

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

    def _add_or_suggest_selector_field(self, idx, selector, field_name, can_ignore=False):
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
            choices = idx.available_values(selector, field_name, can_ignore=can_ignore, skip_missing_fields=True)
            if len(choices) > 1:
                raise RuntimeError("Please provide {opt}: choices = {choices}".format(
                    opt=optname(field_name),
                    choices=choices,
                ))
            # NOTE: if there's only one choice, we don't need to add it to selector.
            # selector[field_name] = choices[0]

    def each_vd(self):
        for iml_dir in self.iml_directories:
            idx = self.data_index[iml_dir]

            algo, env = self._get_algo_env_from_dir(iml_dir)

            # Q: There's only one ResourceOverlap plot...
            # However, there are many OperationOverlap plots; how can we select
            # among them properly?
            selector = {
                'overlap_type': self.overlap_type,
            }

            # 'CategoryOverlap': ['process', 'phase', 'resource_overlap', 'operation'],
            # 'ResourceOverlap': ['process', 'phase'],
            # 'ResourceSubplot': ['process', 'phase'],
            # 'OperationOverlap': ['process', 'phase', 'resource_overlap'],
            # 'HeatScale': ['device_name'],

            # TODO: support multi-process stuff.
            # NOTE: we should really add more fields here for multi-process support (e.g. 'process' and 'phase');
            # For now this just support single-process results.
            self._add_or_suggest_selector_field(idx, selector, 'resource_overlap', can_ignore=True)
            self._add_or_suggest_selector_field(idx, selector, 'operation', can_ignore=self.overlap_type not in ['CategoryOverlap'])

            md, entry, ident = idx.get_file(selector=selector, skip_missing_fields=True)
            vd = VennData(entry['venn_js_path'])
            yield (algo, env), vd

    def each_df(self):
        for (algo, env), vd in self.each_vd():
            stacked_dict = vd.stacked_bar_dict()
            md = vd.metadata()
            path = vd.path

            def as_list(v):
                if type(v) == list:
                    return v
                return [v]
            new_stacked_dict = dict(stacked_dict)
            new_stacked_dict['algo'] = algo
            new_stacked_dict['env'] = env
            if self.should_add_training_time:
                # Q: Will this handle scaling phases?  I think so... basically, each phase-file will just have a
                # different 'percent_complete'. However, I think we need to make OverlapStackedBarPlot have a phase argument,
                # or run for each phase.
                total_size = vd.total_size()
                # Extrapolate the total training time using percent_complete
                assert 'percent_complete' in md
                total_training_time = extrap_total_training_time(total_size, md['percent_complete'])
                new_stacked_dict['total_training_time'] = total_training_time
            new_stacked_dict = dict((k, as_list(v)) for k, v in new_stacked_dict.items())
            df = pd.DataFrame(new_stacked_dict)

            df = self._remap_df(df)

            yield (algo, env), self._regions(df), path, df

    def _check_can_add_training_time(self):
        if not self.training_time:
            return
        for (algo, env), vd in self.each_vd():
            if 'percent_complete' not in vd.md:
                raise RuntimeError((
                    "Didn't find percent_complete attribute in metadata of overlap plot data @ {path}; "
                    "remove --training-time to plot without training time").format(
                    path=vd.path,
                ))
        return True

    def _check_overlap_json_files(self):
        regions_to_paths = dict()
        for (algo, env), regions, path, df in self.each_df():
            regions = frozenset(regions)
            if regions not in regions_to_paths:
                regions_to_paths[regions] = []
            regions_to_paths[regions].append(path)
        for k in list(regions_to_paths.keys()):
            regions_to_paths[k].sort()

        if self.debug:
            logging.info(pprint_msg({'regions_to_paths': regions_to_paths}))

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
        use_regions = regions_by_num_files[-1]
        if self.debug:
            logging.info(pprint_msg({
                'regions_to_paths': regions_to_paths,
                'use_regions': use_regions}))
        return use_regions

    def _is_region(self, region):
        return type(region) == tuple

    def _regions(self, obj):
        assert type(obj) in [pd.DataFrame, dict]
        return set(key for key in obj.keys() if self._is_region(key))

    def _remap_df(self, orig_df):
        if self.remap_df is None:
            return orig_df

        not_regions = [key for key in orig_df.keys() if not self._is_region(key)]

        # eval context:
        # TODO: limit locals/globals to these? Don't want to limit numpy/pandas access though.
        df = copy.copy(orig_df)
        regions = self._regions(df)
        new_df = df[not_regions]

        for df_transformation in self.remap_df:
            # e.g.
            # new_df[('other',)] = df[('compute_advantage_estimates',)] +
            #                      df[('optimize_surrogate',)]
            if self.debug:
                logging.info("--remap-df:\n{trans}".format(trans=textwrap.indent(df_transformation, prefix='  ')))
            exec(df_transformation)
        # Make sure they didn't modify df; they SHOULD be modifying new_df
        # (i.e. adding regions to a "fresh" slate)
        assert np.all(df == orig_df)

        if self.debug:
            logging.info("--remap-df complete")
            logging.info("Old dataframe; regions={regions}".format(regions=self._regions(orig_df)))
            logging.info(pprint_msg(orig_df))

            logging.info("New dataframe after --remap-df; regions={regions}".format(regions=self._regions(new_df)))
            logging.info(pprint_msg(new_df))

        return new_df

    def _read_df(self):
        """
        Read venn_js data of several --iml-directory's into a single data-frame.

        :return:
        """
        self.data = {
            'algo':[],
            'env':[],
        }
        self.regions = None

        # Process each df separately, since they have DIFFERENT keys.
        # Then, check that the groups for the df match.

        use_regions = self._check_overlap_json_files()

        dfs = []
        for (algo, env), regions, path, df in self.each_df():

            if regions != use_regions:
                logging.info(
                    textwrap.dedent("""\
                    Skipping {path} (--ignore-inconsistent-overlap-regions)
                      regions = {regions}
                      use_regions = {use_regions}
                    """).lstrip().format(
                        path=path,
                        regions=regions,
                        use_regions=use_regions,
                    ))
                continue

            if self.regions is None:
                self.regions = set(regions)

            if self.debug:
                logging.info(pprint_msg({
                    'path': path,
                    'df': df}))

            dfs.append(df)

        self.df = pd.concat(dfs)

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
            for g in self.regions:
                if summation is None:
                    summation = df[g]
                else:
                    summation = summation + df[g]

            ret = dict()
            for g in self.regions:
                # e.g.
                # CPU = CPU / ([CPU] + [CPU, GPU] + [GPU])
                ret[g] = 100 * df[g]/summation

            return ret

        if 'total_training_time' in self.df:
            self.df['total_training_time'] = transform_usec_to_sec(self.df, 'total_training_time')

        if self.y_type == 'seconds':
            for group in self.regions:
                self.df[group] = transform_usec_to_sec(self.df, group)
        elif self.y_type == 'percent':
            ret = transform_usec_to_percent(self.df)
            for group in ret.keys():
                self.df[group] = ret[group]
        else:
            raise NotImplementedError

    def get_x_env(self, env):
        return get_x_env(env, long_env=self.long_env)

    def get_x_field(self, algo, env, human_readable=False):
        return get_x_field(algo, env, self.x_type, human_readable=human_readable)

    def _plot_df(self):
        logging.info("Dataframe:\n{df}".format(df=self.df))

        if self.training_time:
            y2_field = 'total_training_time'
        else:
            y2_field = None

        stacked_bar_plot = StackedBarPlot(
            data=self.df,
            path=self._plot_path,
            groups=sorted(self.regions),
            x_field='x_field',
            y2_field=y2_field,
            y2_logscale=self.y2_logscale,
            x_axis_label=self.plot_x_axis_label,
            y_axis_label=self.plot_y_axis_label,
            y2_axis_label=self.plot_y2_axis_label,
            title=self.plot_title,
            show_legend=self.show_legend,
            width=self.width,
            height=self.height,
            keep_zero=self.keep_zero,
            rotation=self.rotation,
            # groups: the "keys" into the data dictionary, which are the "stacks" found in each bar.
            group_to_label=self.group_to_label,
        )
        stacked_bar_plot.plot()

    def _add_df_fields(self, df, human_readable=False):
        """
        Add any additional fields to the data-frame.

        In particular 'x_field' is a string used for the "x-tick" labels of the plot.
        :param human_readable
            If True, make it csv friendly (i.e. don't use newlines in string)
        :return:
        """

        x_fields = []
        for index, row in df.iterrows():
            x_field = self.get_x_field(row['algo'], row['env'], human_readable=human_readable)
            x_fields.append(x_field)
        df['x_field'] = x_fields

    def group_to_label(self, group):
        label = ' + '.join(group)
        return label

    def run(self):
        self._init_directories()
        self._check_can_add_training_time()
        self._read_df()
        self._normalize_df()
        self._add_df_fields(self.df)
        self.dump_plot_data()
        if self.skip_plot:
            logging.info("Skipping plotting {path} (--skip-plot)".format(path=self._plot_path))
        else:
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

    def dump_plot_data(self):
        human_df = self.human_df()

        logging.info("> {name} @ human readable plot data @ {path}".format(
            name=self.__class__.__name__,
            path=self._plot_data_path))
        with open(self._plot_data_path, 'w') as f:
            DataFrame.print_df(human_df, file=f)

        logging.info("> {name} @ csv plot data @ {path}".format(
            name=self.__class__.__name__,
            path=self._plot_csv_path))

        human_df.to_csv(self._plot_csv_path, index=False)

        # Print human readable plot data to stdout
        logging.info(pprint_msg(human_df))

    def human_df(self):
        """
        Convert tuple keys in data-frame to user-visible labels
        ('CPU', 'GPU') => "CPU + GPU"
        :return:
        """
        human_df = copy.copy(self.df)

        self._add_df_fields(human_df, human_readable=True)

        groups = [g for g in human_df.keys() if self._is_region(g)]
        for group in groups:
            human_group = self.group_to_label(group)
            assert human_group not in human_df
            human_df[human_group] = human_df[group]
            del human_df[group]
        return human_df

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
        self.path = path
        with open(path) as f:
            self.venn = json.load(f)
        self._metadata = self.venn['metadata']
        self._build_idx_to_label()
        self.data = self.as_dict()

    def metadata(self):
        return copy.copy(self._metadata)

    @property
    def md(self):
        return self._metadata

    def subtract(self, subtract_sec, inplace=True):
        """
        Return a new instance of VennData, but with overhead counts subtracted.

        PSEUDOCODE:
        # subtract pyprof_annotation:
        def vd_tree.subtract(machine, process, phase, resource_type, operation, category, subtract_sec):
            selector = {
                'machine': machine
                'process': process
                'phase': phase
                'resource_type': resource_type,
                'operation': operation
                'category': category
            }
            for plot_type in plot_types:

                # e.g. ResourceOverlap: [machine, process, phase]
                plot_type_selector = selector[just keep plot_type.attributes]
                plot_type_selector['plot_type'] = plot_type
                vd = vd_tree.lookup(selector)

                # def vd.key_field():
                #  ResourceSubplot -> ListOf[resource_type]
                #  OperationOverlap -> operation
                #  CategoryOverlap -> category
                #  ResourceSubplot -> resource_type
                key = selector[vd.key_field()]

                vd.subtract(key, subtract_sec, inplace=True)

        def subtract_from_resource(resource, machine, process, phase, operation, category, subtract_sec):
            # e.g.
            # resource = 'CPU'
            # resource_types = [['CPU'], ['CPU', 'GPU']]
            resource_types = [resource_type for resource_type in vd.resource_types if resource in resource_type]
            resource_types.sort(key={by total time spent in resource})
            subtract_left_sec = subtract_sec
            for resource_type in resource_types:
                vd_leaf = vd_tree.lookup(machine, process, phase, operation, category)
                to_subtract = min(
                  subtract_left_sec,
                  vd.time_sec(resource_type, process, phase, operation, category))
                  # We need to "propagate up" the subtraction;
                  # vd_tree.subtract handles this.
                  # i.e. If we are subtracting from:
                  #   [CPU, q_forward, Python]
                  # Then, we need to subtract from:
                  #   [CPU, q_forward, Python]
                  #     CategoryOverlap.machine_{...}.process_{...}.phase_{...}.ops_{...}.resources_{...}.venn_js.json
                  #   [CPU, q_forward]
                  #     OperationOverlap.machine_{...}.process_{...}.phase_{...}.resources_{...}.venn_js.json
                  #   [CPU]
                  #     ResourceOverlap.machine_{...}.process_{...}.phase_{...}.venn_js.json
                  #     ResourceSubplot.machine_{...}.process_{...}.phase_{...}.venn_js.json
                vd_tree.subtract(machine, process, phase, resource_type, operation, category, to_subtract)
                subtract_left_sec -= to_subtract

        # Q: What's a good way to sanity check venn_js consistency?
        # Make sure the child venn_js number "add up" to those found in the parent venn_js.
        # e.g. child=OperationOverlap, parent=ResourceOverlap
        # for resource_type in [['CPU'], ['CPU', 'GPU'], ['GPU']]:
        #   assert sum[OperationOverlap[op, resource_type] for each op] == ResourceOverlap[resource_type]

        # e.g. subtracting Python annotation time.
        # The approach will be similar for other overhead types.
        for machine in machines(directory):
            for process in processes(machine, directory):
                for phase in phases(machine, process, directory):
                    for operation in operations(machine, process, phase, directory):
                        subtract_sec = (pyprof_overhead_json['mean_pyprof_annotation_per_call_us']/USEC_IN_SEC) *
                                       overhead_event_count_json[pyprof_annotation][process][phase][operation]
                        vd_tree.subtract_from_resource(resource='CPU', machine, process, phase, operation, category='Python',
                            subtract_sec)

        :param overhead_event_count_json:
        :param cupti_overhead_json:
        :param LD_PRELOAD_overhead_json:
        :param pyprof_overhead_json:
        :return:
        """
        pass

    def stacked_bar_dict(self):
        """
        In order to obtain stack-bar data, we must compute:
        - 'CPU' = [CPU only]
        - 'GPU' = [GPU only]
        - 'CPU + GPU' = [CPU-GPU only]

        See VennData NOTE above for details.
        """
        venn_dict = self.data
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

    def total_size(self):
        total_size = 0.
        # [ size of all regions ] - [ size of overlap regions ]
        for labels, size in self.data.items():
            if len(labels) > 1:
                # Overlap region is JUST the size of the overlap.
                total_size -= size
            else:
                # Single 'set' is the size of the WHOLE region (INCLUDING overlaps)
                assert len(labels) == 1
                total_size += size
        return total_size

    def get_size(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def _build_idx_to_label(self):
        self.idx_to_label = dict()
        self.label_to_idx = dict()
        for venn_data in self.venn['venn']:
            if len(venn_data['sets']) == 1:
                assert 'label' in venn_data
                idx = venn_data['sets'][0]
                self.idx_to_label[idx] = venn_data['label']
                self.idx_to_label[venn_data['label']] = idx

    def _indices_to_labels(self, indices):
        return tuple(sorted(self.idx_to_label[i] for i in indices))

    def _labels_to_indices(self, labels):
        return tuple(sorted(self.label_to_idx[label] for label in labels))

    def labels(self):
        return self.data.keys()

    def get_label(self, label):
        idx = self._label_to_idx(label)

    def as_dict(self):
        """
        {
            ('CPU',): 135241018.0,
            ('GPU',): 3230025.0,
            ('CPU', 'GPU'): 3230025.0,
        }
        """
        if self.data is not None:
            return dict(self.data)
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
                 y2_field=None,
                 y2_logscale=False,
                 x_axis_label="X-axis label",
                 y_axis_label="Y-axis label",
                 y2_axis_label=None,
                 title=None,
                 show_legend=True,
                 width=None,
                 height=None,
                 keep_zero=True,
                 rotation=None,
                 # fontsize=16,
                 fontsize=None,
                 group_to_label=None):
        assert groups is not None
        assert x_field is not None
        self.data = pd.DataFrame(data)
        self.path = path
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.y2_axis_label = y2_axis_label
        self.title = title
        self.show_legend = show_legend
        self.width = width
        self.height = height
        if ( self.width is not None and self.height is None ) or \
                (self.width is None and self.height is not None ):
            raise ValueError("You must provide both --width and --height")
        self.keep_zero = keep_zero
        self.rotation = rotation
        self.fontsize = fontsize
        self.groups = groups
        self.x_field = x_field
        self.y2_field = y2_field
        self.y2_logscale = y2_logscale
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

    def _add_legend(self, fig, loc, bbox_to_anchor):
        legend_rects = []
        for i, group in enumerate(self.groups):
            if not self.keep_zero and self._all_zero(group):
                # If a stacked-bar element is zero in all the bar-charts, don't show it in the legend (--keep-zero false).
                continue
            legend_rect = plt.Rectangle((0, 0), 1, 1, fc=self.colors[i], edgecolor='none')
            legend_rects.append(legend_rect)
        legend_labels = [self._group_to_label(group) for group in self.groups]
        leg = fig.legend(legend_rects, legend_labels,
                         bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
        leg.draw_frame(False)

    def plot(self):

        if self.width is not None and self.height is not None:
            figsize = (self.width, self.height)
            logging.info("Setting figsize = {fig}".format(fig=figsize))
            # sns.set_context({"figure.figsize": figsize})
        else:
            figsize = None
        # This is causing XIO error....
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(111)
        ax2 = None
        if self.y2_field is not None:
            ax2 = ax.twinx()
            # Need to do this, otherwise, training time bar is ABOVE gridlines from ax.
            ax.set_zorder(ax2.get_zorder()+1)
            # Need to do this, otherwise training time bar is invisible.
            ax.patch.set_visible(False)


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
        ax.set_xlabel(self.x_axis_label)
        if self.title is not None:
            # bottom_plot.set_title(self.title)
            ax.set_title(self.title)

        if self.rotation is not None:
            # ax = bottom_plot.axes
            ax.set_xticklabels(ax.get_xticklabels(), rotation=self.rotation)

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

        figlegend.tight_layout()
        figlegend.savefig(self.legend_path, bbox_inches='tight', pad_inches=0)
        plt.close(figlegend)

    @property
    def legend_path(self):
        return re.sub(r'(\.[^.]+)$', r'.legend\1', self.path)

# Unused class; just some documentation
class RegionDataFrame:
    """
    Here's what the data-frame looks like for:
    # OverlapStackedBarPlot.overlap_type_OperationOverlap.txt
    $ train_stable_baselines.sh --env-id Walker2DBulletEnv-v0

       (compute_advantage_estimates,)  (sample_action,)                                  env  algo  (optimize_surrogate,)                                      x_field
    0                        0.007635         12.188267                      AntBulletEnv-v0  ppo2              87.804098                      (ppo2, AntBulletEnv-v0)
    1                        0.004305         50.264176              HalfCheetahBulletEnv-v0  ppo2              49.731519              (ppo2, HalfCheetahBulletEnv-v0)
    2                        0.002083         29.895073                   HopperBulletEnv-v0  ppo2              70.102843                   (ppo2, HopperBulletEnv-v0)
    3                        0.004586         47.460603                 HumanoidBulletEnv-v0  ppo2              52.534812                 (ppo2, HumanoidBulletEnv-v0)
    4                        0.005242         49.675716   InvertedDoublePendulumBulletEnv-v0  ppo2              50.319042   (ppo2, InvertedDoublePendulumBulletEnv-v0)
    5                        0.005556         47.268527  InvertedPendulumSwingupBulletEnv-v0  ppo2              52.725917  (ppo2, InvertedPendulumSwingupBulletEnv-v0)
    6                        0.005479         46.588143                  ReacherBulletEnv-v0  ppo2              53.406377                  (ppo2, ReacherBulletEnv-v0)
    7                        0.004384         58.306973                 Walker2DBulletEnv-v0  ppo2              41.688643                 (ppo2, Walker2DBulletEnv-v0)

    If we add two regions together:
    e.g.
      remap['other'] = regions['sample_action'] +
                       regions['optimize_surrogate'] +
                       regions['compute_advantage_estimates']
    then we want to perform this addition for each matching pair of
    rl-workloads measured.
    i.e.
    NOTE: we could just expose pandas syntax directly to the user.
      new_df[('other',)] = df[('compute_advantage_estimates',)] +
                           df[('optimize_surrogate',)]
    - Syntactic sugar (1): auto-tuplify single-name regions
      PROBLEM: loses generality, cannot reference 'env' or 'algo' now... not a bad thing really.
      new_df['other'] = df['compute_advantage_estimates'] +
                        df['optimize_surrogate']
    - Syntactic sugar (2): remove reference to new_df and df, replace strings with barename
      PROBLEM: loses generality, cannot use python syntax now.
      new_df['other'] = df['compute_advantage_estimates'] +
                        df['optimize_surrogate']

    Region-remapping operations/syntax we wish to support:

    E.g.
        remap['step'] = step
        remap['other'] = sum(r for r in regions if r != step)

    Region-remapping may be specific to a particular algorithm.
    E.g. to obtain generic breakdown from ppo2:

        remap['inference'] = sample_action
        remap['backward_pass'] = optimize_surrogate

    :param other:
    :return:
    """
    def __init__(self, df, groups):
        self.df = df
        self.groups = groups


    # TODO: overload all the things pandas.DataFrame overloads and pass them to self.df
    # Q: Why not just extend pandas.DataFrame?
    # A: because we'd like to retain the base-class instance __setitem__ behaviour...
    #    we CANNOT convert a RegionDataFrame to a pandas.DataFrame since __setitem__ becomes forever overloaded... annoying.

    def __setitem__(self, key, value):
        # TODO: only allow region-names as keys; check if key is a region-name; if not raise a informative error
        # TODO: tuplify string key into a single-element tuple.
        pass


def extrap_total_training_time(time_unit, percent_complete):
    """
    10 * (1/0.01) => 100
    #
    10 seconds in 1%  0.01
    ->
    1000 seconds in 100% 1.0

    :param time_unit:
    :param percent_complete:
    :return:
    """
    assert 0. <= percent_complete <= 1.
    total_time_unit = time_unit * (1./percent_complete)
    return total_time_unit

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

def test_double_yaxis():

    # # http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/#sec-3
    # x = np.linspace(0, 2*np.pi)
    # y1 = np.sin(x);
    # y2 = 0.01 * np.cos(x);
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.plot(x, y1)
    # ax1.set_ylabel('y1')
    #
    # ax2 = ax1.twinx()
    # ax2.plot(x, y2, 'r-')
    # ax2.set_ylabel('y2', color='r')
    # for tl in ax2.get_yticklabels():
    #     tl.set_color('r')
    #
    # plt.savefig('./test_double_yaxis.png')

    # https://stackoverflow.com/questions/24183101/pandas-bar-plot-with-two-bars-and-two-y-axis
    s = StringIO("""     amount     price
    A     40929   4066443
    B     93904   9611272
    C    188349  19360005
    D    248438  24335536
    E    205622  18888604
    F    140173  12580900
    G     76243   6751731
    H     36859   3418329
    I     29304   2758928
    J     39768   3201269
    K     30350   2867059""")

    df = pd.read_csv(s, index_col=0, delimiter=' ', skipinitialspace=True)

    fig = plt.figure() # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.4

    df.amount.plot(kind='bar', color='red', ax=ax, width=width, position=1)
    df.price.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

    ax.set_ylabel('Amount')
    ax2.set_ylabel('Price')

    plt.savefig('./test_double_yaxis.png')


class DaysHoursMinutesSecondsFormatter(mpl_ticker.Formatter):
    """
    Use an old-style ('%' operator) format string to format the tick.

    The format string should have a single variable format (%) in it.
    It will be applied to the value (not the position) of the tick.
    """
    def __init__(self):
        pass

    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        Only the value `x` is formatted. The position is ignored.
        """
        seconds_remaining = x

        SECONDS_PER_MINUTE = 60
        SECONDS_PER_HOUR = SECONDS_PER_MINUTE*60
        SECONDS_PER_DAY = SECONDS_PER_HOUR*24

        days = int(seconds_remaining / SECONDS_PER_DAY)
        seconds_remaining -= days * SECONDS_PER_DAY

        hours = int(seconds_remaining / SECONDS_PER_HOUR)
        seconds_remaining -= hours * SECONDS_PER_HOUR

        minutes = int(seconds_remaining / SECONDS_PER_MINUTE)
        seconds_remaining -= minutes * SECONDS_PER_MINUTE

        seconds = int(seconds_remaining)

        # "5 days, 3 hours"
        # "5 days, 3 hours, 5 seconds"
        # "5 days, 5 seconds"
        values = [days, hours, minutes, seconds]
        names = ['day', 'hr', 'min', 'sec']
        components = []
        assert len(values) == len(names)
        for value, name in zip(values, names):
            if value == 0:
                continue
            string = '{value} {name}'.format(
                value=value, name=name)
            components.append(string)
        if len(components) == 0:
            label = '0'
        else:
            label = ', '.join(components)

        return label

class VdTree(_DataIndex):
    """
    Same interface as DataIndex, except that VdTree represents open instances of venn_js files
    that we can modify inplace, then persist back to disk (with an additional file-suffix added).
    """
    def __init__(self, index, directory,
                 host=None,
                 user=None,
                 password=None,
                 debug=False):
        # VdTree will mutate the index; so create a copy of it.
        index_copy = copy.copy(index)
        super.__init__(index_copy, directory, debug)
        self.host = host
        self.user = user
        self.password = password
        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

    def open_vds(self, debug=False):
        for plot_type in plot_index.OVERLAP_PLOT_TYPES:
            selector = {
                'plot_type': plot_type,
            }
            for md, entry, ident in self.each_file(selector, debug=debug):
                entry['vd'] = VennData(path=entry['venn_js_path'])

    def subtract(self, debug=False):
        pass

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    def _each_operation(self):
        for machine in self.sql_reader.machines():
            machine_name = machine.machine_name
            for process in self.sql_reader.processes(machine_name=machine_name):
                process_name = process.process_name
                for phase in self.sql_reader.phases(machine_name=machine_name,
                                                    process_name=process_name):
                    phase_name = phase.phase_name
                    for operation in self.sql_reader.operations(machine_name=machine_name,
                                                                process_name=process_name,
                                                                phase_name=phase_name):
                        operation_name = operation.operation_name
                        yield machine_name, process_name, phase_name, operation_name

    def subtract_overhead(self,
                          overhead_event_count_json,
                          cupti_overhead_json,
                          LD_PRELOAD_overhead_json,
                          pyprof_overhead_json):
        """
        - CUDA API interception:
          Subtract from:
          [CPU, q_forward, TensorFlow C++]

        - CUPTI overhead:
          Subtract from:
          [CPU, q_forward, CUDA API]

        - Python -> C-library interception:
          Subtract from:
          [CPU, q_forward, Python]

        - Python annotations:
          Subtract from:
          [CPU, q_forward, Python]
        """

        # 'machine', 'process', 'phase', 'resource_overlap', 'operation'

        # e.g. subtracting Python annotation time.
        # The approach will be similar for other overhead types.

        for machine_name, process_name, phase_name, operation_name in self._each_operation():

            # Python annotations:
            total_pyprof_annotation = overhead_event_count_json['pyprof_annotation'][machine_name][process_name][phase_name][operation_name]
            per_pyprof_annotation_sec = pyprof_overhead_json['mean_pyprof_annotation_per_call_us']/USEC_IN_SEC
            pyprof_annotation_sec = per_pyprof_annotation_sec * total_pyprof_annotation
            self.subtract_from_resource(
                resource='CPU',
                selector=dict(
                    machine=machine_name,
                    process=process_name,
                    phase=phase_name,
                    operation=operation_name,
                    category=CATEGORY_PYTHON,
                ),
                subtract_sec=pyprof_annotation_sec)

            # Python -> C-library interception:
            total_pyprof_interception = overhead_event_count_json['pyprof_interception'][machine_name][process_name][phase_name][operation_name]
            per_pyprof_interception_sec = pyprof_overhead_json['mean_pyprof_interception_overhead_per_call_us']/USEC_IN_SEC
            pyprof_interception_sec = per_pyprof_interception_sec * total_pyprof_interception
            self.subtract_from_resource(
                resource='CPU',
                selector=dict(
                    machine=machine_name,
                    process=process_name,
                    phase=phase_name,
                    operation=operation_name,
                    category=CATEGORY_PYTHON,
                ),
                subtract_sec=pyprof_interception_sec)

            # CUPTI overhead:
            for cuda_api_name, num_api_calls in overhead_event_count_json['cuda_api_call'][machine_name][process_name][phase_name][operation_name].items():
                per_cuda_api_sec = cupti_overhead_json[cuda_api_name]['mean_cupti_overhead_per_call_us']/USEC_IN_SEC
                cupti_overhead_sec = per_cuda_api_sec * num_api_calls
                self.subtract_from_resource(
                    resource='CPU',
                    selector=dict(
                        machine=machine_name,
                        process=process_name,
                        phase=phase_name,
                        operation=operation_name,
                        category=CATEGORY_CUDA_API_CPU,
                    ),
                    subtract_sec=cupti_overhead_sec)

            # CUDA API interception:
            total_cuda_api_calls = np.sum([num_api_calls for cuda_api_name, num_api_calls in
                                           overhead_event_count_json['cuda_api_call'][machine_name][process_name][phase_name][operation_name].items()])
            per_LD_PRELOAD_sec = pyprof_overhead_json['mean_interception_overhead_per_call_us']/USEC_IN_SEC
            LD_PRELOAD_sec = per_LD_PRELOAD_sec * total_cuda_api_calls
            self.subtract_from_resource(
                resource='CPU',
                selector=dict(
                    machine=machine_name,
                    process=process_name,
                    phase=phase_name,
                    operation=operation_name,
                    # ASSUMPTION: GPU calls are being made from inside the Tensorflow C++ API...
                    # this might not hold once we start measuring other libraries the use the GPU...
                    # NOTE: this overhead does NOT come from the CUDA API call itself; the overhead is
                    # "around" the CUDA API call.
                    category=CATEGORY_TF_API,
                ),
                subtract_sec=LD_PRELOAD_sec)

    def subtract_from_resource(self, resource, selector, subtract_sec):
        """
        To handle the fact that we cannot precisely attribute profiling-overhead CPU-time to [CPU], or [CPU, GPU],
        we decide to perform this heuristic:
        - Subtract from [CPU] or [CPU, GPU], in the order of whichever is largest FIRST
        - If we end up subtracting all of the first resource-group,
          then subtract what remains from the next resource-group.

        NOTE: if we subtract CPU-time from [CPU, GPU] the new time that remains is GPU-only time...

        :param resource:
        :param selector:
        :param subtract_sec:
        :return:
        """
        resource_selector = dict(selector)
        resource_selector['plot_type'] = 'ResourceOverlap'
        resource_selector = only_selector_fields(resource_selector)
        resource_vd = self.get_file(resource_selector)
        # e.g.
        # resource = 'CPU'
        # resource_types = [['CPU'], ['CPU', 'GPU']]
        keys = resource_vd.keys()
        resource_types = [resource_type for resource_type in keys if resource in resource_type]
        def sort_by_time(key):
            return -1 * resource_vd.get_size(key)
        # by = {by total time spent in resource in descending order}
        resource_types.sort(key=sort_by_time)
        subtract_left_sec = subtract_sec
        for resource_type in resource_types:
            if subtract_left_sec == 0:
                break
            # vd_leaf = vd_tree.lookup(machine, process, phase, operation, category)
            # vd_selector = copy.copy(selector)
            # vd_selector['resource_overlap'] = resource_type
            # Q: are we selecting for ResourceOverlap or ResourceSubplot?
            # vd_leaf = self.get_file(selector)

            # to_subtract = min(
            #     subtract_left_sec,
            #     vd.time_sec(resource_type, process, phase, operation, category))
            # key_type = plot_index.KEY_TYPE[plot_type]
            # key = selector[key_type]
            to_subtract = min(
                subtract_left_sec,
                # vd_leaf.get_size(key),
                resource_vd.get_size(resource_type),
            )

            # We need to "propagate up" the subtraction;
            # vd_tree.subtract handles this.
            # i.e. If we are subtracting from:
            #   [CPU, q_forward, Python]
            # Then, we need to subtract from:
            #   [CPU, q_forward, Python]
            #     CategoryOverlap.machine_{...}.process_{...}.phase_{...}.ops_{...}.resources_{...}.venn_js.json
            #   [CPU, q_forward]
            #     OperationOverlap.machine_{...}.process_{...}.phase_{...}.resources_{...}.venn_js.json
            #   [CPU]
            #     ResourceOverlap.machine_{...}.process_{...}.phase_{...}.venn_js.json
            #     SKIP: ResourceSubplot.machine_{...}.process_{...}.phase_{...}.venn_js.json
            # vd_tree.subtract(machine, process, phase, resource_type, operation, category, to_subtract)
            self.subtract(resource_selector, to_subtract)
            subtract_left_sec -= to_subtract

    def subtract(self, selector, subtract_sec):
        # selector = {
        #     'machine': machine
        #     'process': process
        #     'phase': phase
        #     'resource_type': resource_type,
        #     'operation': operation
        #     'category': category
        # }
        for plot_type in plot_index.OVERLAP_PLOT_TYPES:
            # e.g. ResourceOverlap: [machine, process, phase]
            # selector[just keep plot_type.attributes]
            plot_type_selector = dict((k, v) for k, v in selector.items()
                                      if k in plot_index.SEL_ORDER[plot_type])
            plot_type_selector['plot_type'] = plot_type
            vd = self.get_file(plot_type_selector)

            # def vd.key_field():
            #  ResourceSubplot -> ListOf[resource_type]
            #  OperationOverlap -> operation
            #  CategoryOverlap -> category
            #  ResourceSubplot -> resource_type
            key_field = plot_index.KEY_TYPE[plot_type]
            key = selector[key_field]
            vd.subtract(key, subtract_sec, inplace=True)

def get_x_env(env, long_env=False):
    short_env = env
    if not long_env:
        short_env = re.sub(r'-v\d+$', '', short_env)
        short_env = re.sub(r'BulletEnv$', '', short_env)
        short_env = re.sub(r'Env$', '', short_env)
        short_env = re.sub(r'NoFrameskip', '', short_env)
        short_env = re.sub(r'Inverted', 'Inv', short_env)
        short_env = re.sub(r'Double', 'Dbl', short_env)
        short_env = re.sub(r'Pendulum', 'Pndlm', short_env)
        short_env = re.sub(r'Swingup', 'Swing', short_env)
    return short_env

def get_x_field(algo, env, x_type, human_readable=False):
    short_env = get_x_env(env)
    if x_type == 'rl-comparison':
        if human_readable:
            x_field = "({algo}, {env})".format(
                algo=algo,
                env=short_env,
            )
        else:
            x_field = "{algo}\n{env}".format(
                algo=algo,
                env=short_env,
            )
    elif x_type == 'env-comparison':
        x_field = short_env
    elif x_type == 'algo-comparison':
        x_field = algo
    else:
        raise NotImplementedError
    return x_field

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

    parser.add_argument('--test-double-yaxis',
                        action='store_true',
                        help=textwrap.dedent("""
    Test how to plot double y-axis stacked bar-chart.
    """))

    args = parser.parse_args()

    if args.test_stacked_bar:
        test_stacked_bar()
    elif args.test_stacked_bar_sns:
        test_stacked_bar_sns()
    elif args.test_stacked_bar_sns_old:
        test_stacked_bar_sns_old()
    elif args.test_double_yaxis:
        test_double_yaxis()

def only_selector_fields(selector):
    assert 'plot_type' in selector
    new_selector = dict((k, v) for k, v in selector.items())
    return new_selector

if __name__ == '__main__':
    main()
