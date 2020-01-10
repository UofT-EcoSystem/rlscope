import logging
import argparse
import re
import copy
import importlib
import sys
import textwrap
import copy
import itertools


import matplotlib.patches as mpatches
import matplotlib.patches as mpl_patches
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
from iml_profiler.parser.dataframe import VennData, get_training_durations_df, read_iml_config_metadata, get_total_timesteps, get_end_num_timesteps
from iml_profiler.parser.plot import LegendMaker, HATCH_STYLES

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
                 unins_iml_directories,
                 directory,
                 overlap_type,
                 resource_overlap=None,
                 operation=None,
                 training_time=False,
                 extrapolated_training_time=False,
                 detailed=False,
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
                 debug_single_thread=False,
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
        self.unins_iml_directories = unins_iml_directories
        logging.info("{klass}:\n{msg}".format(
            klass=self.__class__.__name__,
            msg=pprint_msg({
                'iml_directories': self.iml_directories,
                'unins_iml_directories': self.unins_iml_directories,
            })))
        self.training_time = training_time
        self.extrapolated_training_time = extrapolated_training_time
        self.detailed = detailed
        # self.should_add_training_time = training_time
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
        self.debug_single_thread = debug_single_thread

    def _get_plot_path(self, ext, extra_suffix):
        if self.suffix is not None:
            suffix_str = '.{suffix}'.format(suffix=self.suffix)
        else:
            suffix_str = ''

        if extra_suffix is not None:
            extra_suffix_str = '.{extra_suffix}'.format(extra_suffix=extra_suffix)
        else:
            extra_suffix_str = ''

        return _j(self.directory, "OverlapStackedBarPlot.overlap_type_{ov}{extra_suffix}{suffix}.{ext}".format(
            ov=self.overlap_type,
            suffix=suffix_str,
            extra_suffix=extra_suffix_str,
            ext=ext,
        ))

    def _plot_data_path(self, extra_suffix=None):
        return self._get_plot_path(ext='txt', extra_suffix=extra_suffix)

    def _plot_csv_path(self, extra_suffix=None):
        return self._get_plot_path(ext='csv', extra_suffix=extra_suffix)

    def _plot_path(self, extra_suffix=None):
        return self._get_plot_path(ext='pdf', extra_suffix=extra_suffix)

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
            return "(RL algorithm, Simulator)"
        elif self.x_type == 'env-comparison':
            return "Simulator"
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
            return "Total training time (log2 sec)"
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

    # def _get_algo_env_id(self, iml_dir):
    #     sql_reader = self.sql_readers[iml_dir]
    #     procs = sql_reader.process_names()
    #     assert len(procs) == 1
    #     proc = procs[0]
    #     m = re.search(r'(?P<algo>[^_]+)_(?P<env_id>.+)', proc)
    #     algo = m.group('algo')
    #     env_id = m.group('env_id')
    #     env_id = self._reduce_env(env_id)
    #     return (algo, env_id)

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
        new_env = re.sub(r'MountainCarContinuous-v0', 'MountainCar-v0', new_env)
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

    def read_unins_df(self):
        unins_df = pd.concat(get_training_durations_df(
            self.unins_iml_directories,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread))
        unins_df['unins_total_training_time_us'] = unins_df['training_duration_us']
        return unins_df


    def _init_directories(self):
        """
        Initialize SQL / DataIndex needed for reading plot-data from iml-analyze'd --iml-directory's.

        :return:
        """
        self.data_index = dict()
        # self.sql_readers = dict()

        for iml_dir in self.iml_directories:
            # self.sql_readers[iml_dir] = SQLCategoryTimesReader(self.db_path(iml_dir), host=self.host, user=self.user, password=self.password)
            index = self.get_index(iml_dir)
            self.data_index[iml_dir] = index

    def _add_or_suggest_selector_field(self, idx, selector, field_name, can_ignore=False, allow_many=False):
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
            if len(choices) > 1 and not allow_many:
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
            if not self.detailed:
                self._add_or_suggest_selector_field(idx, selector, 'resource_overlap', can_ignore=True)
            # self._add_or_suggest_selector_field(idx, selector, 'operation', can_ignore=self.overlap_type not in ['CategoryOverlap'])

            for md, entry, ident in idx.get_files(selector=selector, skip_missing_fields=True):
                vd = VennData(entry['venn_js_path'])
                yield (algo, env), vd

            # md, entry, ident = idx.get_file(selector=selector, skip_missing_fields=True)
            # vd = VennData(entry['venn_js_path'])
            # yield (algo, env), vd

    def each_df(self):
        for (algo, env), vd in self.each_vd():
            stacked_dict = vd.stacked_bar_dict()
            md = vd.metadata()
            path = vd.path
            iml_dir = _d(path)

            iml_metadata = read_iml_config_metadata(_d(path))

            def as_list(v):
                if type(v) == list:
                    return v
                return [v]
            new_stacked_dict = dict(stacked_dict)
            new_stacked_dict['algo'] = algo
            new_stacked_dict['env'] = env

            # if self.training_time or 'HACK_total_timesteps' in iml_metadata['metadata']:
            #     if 'percent_complete' in md:
            #         # Q: Will this handle scaling phases?  I think so... basically, each phase-file will just have a
            #         # different 'percent_complete'. However, I think we need to make OverlapStackedBarPlot have a phase argument,
            #         # or run for each phase.
            #         total_size = vd.total_size()
            #         # Extrapolate the total training time using percent_complete
            #         total_training_time = extrap_total_training_time(total_size, md['percent_complete'])
            #         new_stacked_dict['extrap_total_training_time'] = total_training_time
            #     else:
            #         new_stacked_dict['extrap_total_training_time'] = np.NAN

            extrap_dict = dict()
            extrap_dict['algo'] = algo
            extrap_dict['env'] = env
            extrap_dict['path'] = path

            # if env == 'AirLearningEnv':
            #     import ipdb; ipdb.set_trace()

            if ( self.extrapolated_training_time or 'HACK_total_timesteps' in iml_metadata['metadata'] ) and 'percent_complete' in md:
                total_size = vd.total_size()
                # Extrapolate the total training time using percent_complete
                if 'HACK_total_timesteps' in iml_metadata['metadata']:
                    logging.info("HACK: ({algo}, {env}) @ {path} -- Override total number of training timesteps to be {t}".format(
                        algo=algo,
                        env=env,
                        path=path,
                        t=iml_metadata['metadata']['HACK_total_timesteps'],
                    ))
                    end_time_timesteps = get_end_num_timesteps(iml_dir)
                    percent_complete = end_time_timesteps / iml_metadata['metadata']['HACK_total_timesteps']
                    extrap_dict['end_time_timesteps'] = end_time_timesteps
                    extrap_dict['HACK_total_timesteps'] = iml_metadata['metadata']['HACK_total_timesteps']
                else:
                    percent_complete = md['percent_complete']
                total_training_time = extrap_total_training_time(total_size, percent_complete)
                new_stacked_dict['extrap_total_training_time'] = total_training_time
                extrap_dict['percent_complete'] = percent_complete
                extrap_dict['total_training_time'] = total_training_time
                extrap_dict['total_size'] = total_size
            else:
                new_stacked_dict['extrap_total_training_time'] = np.NAN
                extrap_dict['isnan'] = True
            logging.info("debug extrap:\n{msg}".format(msg=pprint_msg(extrap_dict)))

            new_stacked_dict = dict((k, as_list(v)) for k, v in new_stacked_dict.items())
            df = pd.DataFrame(new_stacked_dict)

            df = self._HACK_remove_process_operation_df(df)
            df = self._remap_df(df, algo, env)

            yield (algo, env), self._regions(df), path, df

    def new_each_df(self):
        for (algo, env), vd in self.each_vd():
            path = vd.path
            md = vd.metadata()
            iml_dir = _d(path)

            iml_metadata = read_iml_config_metadata(iml_dir)
            df = vd.as_df(keep_metadata_fields=['machine', 'process', 'phase', 'operation', 'resource_overlap'])
            if len(df['operation']) > 0 and type(df['operation'][0]) != str:
                df['operation'] = df['operation'].apply(join_plus)
            df['resource_overlap'] = df['resource_overlap'].apply(join_plus)
            df['category'] = df['region'].apply(join_plus)
            df['time_sec'] = df['size'] / MICROSECONDS_IN_SECOND
            # stacked_dict = vd.stacked_bar_dict()
            # md = vd.metadata()
            path = vd.path
            df['algo'] = algo
            df['env'] = env
            # if 'percent_complete' in md:
            if ( self.extrapolated_training_time or 'HACK_total_timesteps' in iml_metadata['metadata'] ) and 'percent_complete' in md:
                total_size = vd.total_size()
                # Extrapolate the total training time using percent_complete
                if 'HACK_total_timesteps' in iml_metadata['metadata']:
                    logging.info("HACK: ({algo}, {env}) @ {path} -- Override total number of training timesteps to be {t}".format(
                        algo=algo,
                        env=env,
                        path=path,
                        t=iml_metadata['metadata']['HACK_total_timesteps'],
                    ))
                    end_time_timesteps = get_end_num_timesteps(iml_dir)
                    percent_complete = end_time_timesteps / iml_metadata['metadata']['HACK_total_timesteps']
                else:
                    percent_complete = md['percent_complete']
                total_training_time = extrap_total_training_time(total_size, percent_complete)
                df['extrap_total_training_time'] = total_training_time
            else:
                df['extrap_total_training_time'] = np.NAN

            df = self._new_HACK_remove_process_operation_df(df)
            df = self._new_remap_df(df, algo, env)

            yield (algo, env), self._regions(df), path, df

    def _check_can_add_training_time(self):
        if not self.extrapolated_training_time:
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

    def _remap_df(self, orig_df, algo, env):
        if self.remap_df is None:
            return orig_df

        not_regions = [key for key in orig_df.keys() if not self._is_region(key)]

        # eval context:
        # TODO: limit locals/globals to these? Don't want to limit numpy/pandas access though.
        df = copy.copy(orig_df)
        regions = self._regions(df)
        new_df = df[not_regions]
        # algo
        # env

        for df_transformation in self.remap_df:
            # e.g.
            # new_df[('other',)] = df[('compute_advantage_estimates',)] +
            #                      df[('optimize_surrogate',)]
            if self.debug:
                logging.info("--remap-df:\n{trans}".format(trans=textwrap.indent(df_transformation, prefix='  ')))
            exec(df_transformation)
        # Make sure they didn't modify df; they SHOULD be modifying new_df
        # (i.e. adding regions to a "fresh" slate)
        # assert np.all(df == orig_df)
        # Assume NaN's in the same place => equality.
        assert df.equals(orig_df)

        # if self.debug:
        #     logging.info("--remap-df complete")
        #     logging.info("Old dataframe; regions={regions}".format(regions=self._regions(orig_df)))
        #     logging.info(pprint_msg(orig_df))
        #
        #     logging.info("New dataframe after --remap-df; regions={regions}".format(regions=self._regions(new_df)))
        #     logging.info(pprint_msg(new_df))

        if self.debug:
            buf = StringIO()
            DataFrame.print_df(orig_df, file=buf)
            logging.info("Old dataframe; regions={regions}:\n{msg}".format(
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
                regions=self._regions(orig_df),
            ))

            buf = StringIO()
            DataFrame.print_df(new_df, file=buf)
            logging.info("New dataframe after --remap-df; regions={regions}:\n{msg}".format(
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
                regions=self._regions(orig_df),
            ))

        return new_df

    def _new_remap_df(self, orig_df, algo, env):
        if self.remap_df is None:
            return orig_df

        # not_regions = [key for key in orig_df.keys() if not self._is_region(key)]

        # eval context:
        # TODO: limit locals/globals to these? Don't want to limit numpy/pandas access though.
        df = copy.copy(orig_df)
        # regions = self._regions(df)
        # new_df = df[not_regions]
        new_df = copy.copy(df)
        # algo
        # env

        for df_transformation in self.remap_df:
            # e.g.
            # new_df[('other',)] = df[('compute_advantage_estimates',)] +
            #                      df[('optimize_surrogate',)]
            if self.debug:
                logging.info("--remap-df:\n{trans}".format(trans=textwrap.indent(df_transformation, prefix='  ')))
            exec(df_transformation)

        # Make sure they didn't modify df; they SHOULD be modifying new_df
        # (i.e. adding regions to a "fresh" slate)
        # assert np.all(df == orig_df)
        # Assume NaN's in the same place => equality.
        assert df.equals(orig_df)

        if self.debug:
            buf = StringIO()
            DataFrame.print_df(orig_df, file=buf)
            logging.info("Old dataframe:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))

            buf = StringIO()
            DataFrame.print_df(new_df, file=buf)
            logging.info("New dataframe after --remap-df:\n{msg}".format(msg=textwrap.indent(buf.getvalue(), prefix='  ')))

        return new_df

    def _num_algo_env_combos(self):
        return len(self.df[['algo', 'env']].unique())

    def _read_df(self):
        # TODO: merge these on (algo, env)
        # Only keep rows that have both.
        # (Would be nice to warn which rows are missing what)

        self.unins_df = self.read_unins_df()
        buf = StringIO()
        DataFrame.print_df(self.unins_df, file=buf)
        logging.info("unins_df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))

        self.ins_df = self.read_ins_df()
        buf = StringIO()
        DataFrame.print_df(self.ins_df, file=buf)
        logging.info("ins_df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))

        # self.df = self.ins_df.merge(self.unins_df, on=['algo', 'env'])
        # if not self.detailed:
        #     self.df['total_training_time'] = self.df['unins_total_training_time_us']
        # else:
        #     self.df['total_training_time'] = self.df['unins_total_training_time_us'] * self.df['percent']

        # Keep (algo, env) even if we don't have total uninstrumented training time for it.
        # Use extrapolated time instead.
        self.df = self.ins_df.merge(self.unins_df, on=['algo', 'env'], how='left')
        def get_total_training_time(row):
            # Prefer actual uninstrumented training time over extrapolated training time.
            # Q: isn't extrapolated time including overheads...?
            if np.isnan(row['unins_total_training_time_us']):
                return row['extrap_total_training_time']
            return row['unins_total_training_time_us']
        self.df['full_unins_training_time'] = self.df.apply(get_total_training_time, axis=1)
        self.df = self.df[~np.isnan(self.df['full_unins_training_time'])]
        if not self.detailed:
            self.df['total_training_time'] = self.df['full_unins_training_time']
        else:
            self.df['total_training_time'] = self.df['full_unins_training_time'] * self.df['percent']
            # Number of (algo, env) combinations is just 1;
            # Show single training time bar, where total training time is broken down into percent's

            # Number of (algo, env) combinations is more than one;
            # Show two bars per (algo, env):
            # - Left bar: percent breakdown with high-level resource (CPU/GPU) (hue) and high-level operation (hatches).
            # - Right bar: Total training time


            # buf = StringIO()
            # DataFrame.print_df(self.df, file=buf)
            # logging.info("ins_df:\n{msg}".format(
            #     msg=textwrap.indent(buf.getvalue(), prefix='  '),
            #     # msg=pprint_msg(self.df),
            # ))

        buf = StringIO()
        DataFrame.print_df(self.df, file=buf)
        logging.info("df:\n{msg}".format(
            msg=textwrap.indent(buf.getvalue(), prefix='  '),
        ))



    def read_ins_df(self):
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

        if not self.detailed:
            use_regions = self._check_overlap_json_files()

        dfs = []
        if self.detailed:
            df_iter = self.new_each_df()
        else:
            df_iter = self.each_df()
        for (algo, env), regions, path, df in df_iter:
            buf = StringIO()
            DataFrame.print_df(df)
            logging.info("({algo}, {env}) @ path={path}:\n{msg}".format(
                algo=algo,
                env=env,
                path=path,
                msg=textwrap.indent(buf.getvalue(), prefix='  '),
            ))

            if not self.detailed and regions != use_regions:
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

            # if self.debug:
            #     logging.info(pprint_msg({
            #         'path': path,
            #         'df': df}))

            dfs.append(df)

        # if self.debug:
        #     logging.info("ins_df before concat:\n{msg}".format(msg=pprint_msg(dfs)))
        ins_df = pd.concat(dfs)
        # if self.debug:
        #     logging.info("ins_df after concat:\n{msg}".format(msg=pprint_msg(ins_df)))

        old_ins_df = ins_df
        all_cols = set(old_ins_df.keys())
        numeric_cols = set([colname for colname in old_ins_df.keys() if np.issubdtype(old_ins_df[colname].dtype, np.number)])
        non_numeric_cols = all_cols.difference(numeric_cols)
        # old_ins_df.groupby(['machine', 'operation', 'phase', 'process', 'region', 'resource_overlap', 'category', 'algo', 'env', 'x_field']).sum().reset_index()
        # WARNING:  df.groupby(...).sum(skipna=False)
        #   This IGNORES NaN's in aggregated columns resulting in NaN's not being forwarded (e.g. zero columns appear instead of NaN's)
        # SOLUTION: df.groupby(...).agg(pd.DataFrame.sum, skipna=False)
        # SEE: https://github.com/pandas-dev/pandas/issues/28787
        # ins_df = old_ins_df.groupby(list(non_numeric_cols)).sum().reset_index()
        ins_df = old_ins_df.groupby(list(non_numeric_cols)).agg(pd.DataFrame.sum, skipna=False).reset_index()

        # if self.debug:
        #     logging.info("ins_df after groupby.sum():\n{msg}".format(msg=pprint_msg(ins_df)))
        #     import ipdb; ipdb.set_trace()

        if self.detailed:
            # NOTE: must calculate percent within an (algo, env)
            group_dfs = []
            for group, group_df in ins_df.groupby(['algo', 'env']):
                total_size = group_df['size'].sum()
                # Q: does this update ins_df...?
                group_df['percent'] = group_df['size']/total_size
                group_dfs.append(group_df)
            ins_df = pd.concat(group_dfs)

        return ins_df

    def _HACK_remove_process_operation_df(self, df):
        """
        HACK: BUG: not sure why, but operation overlap contains an operation that looks like the process_name:
        e.g. 
        {
            "label": "[ppo2_Walker2DBulletEnv-v0]",
            "sets": [
                0
            ],
            "size": 45344.0
        },
        Likely reason: we used to have code that checked if a Event.event_name looked like a "process_name"...
        for some reason that code check has been disabled during analysis.
        Fix: the time is REALLY small compared to everything else, so we can either:
        1. add it to the largest time (e.g. step)
        2. ignore it (should be safe if its an absolute time...not safe if its a percent already)
        
        :return: 
        """
        remove_cols = set()
        for colname, coldata in df.iteritems():
            if self._is_region(colname):
                op_name = colname[0]
                if is_op_process_event(op_name, CATEGORY_OPERATION):
                    # logging.info("HACK: remove process_name={proc} from operation dataframe".format(
                    #     proc=op_name,
                    # ))
                    remove_cols.add(colname)
        for colname in remove_cols:
            del df[colname]
        return df

    def _new_HACK_remove_process_operation_df(self, df):
        """
        HACK: BUG: not sure why, but operation overlap contains an operation that looks like the process_name:
        e.g.
        {
            "label": "[ppo2_Walker2DBulletEnv-v0]",
            "sets": [
                0
            ],
            "size": 45344.0
        },
        Likely reason: we used to have code that checked if a Event.event_name looked like a "process_name"...
        for some reason that code check has been disabled during analysis.
        Fix: the time is REALLY small compared to everything else, so we can either:
        1. add it to the largest time (e.g. step)
        2. ignore it (should be safe if its an absolute time...not safe if its a percent already)

        :return:
        """
        def is_process_event(op_name):
            return bool(is_op_process_event(op_name, CATEGORY_OPERATION))
        new_df = df[~df['operation'].apply(is_process_event)]
        return new_df

        return new_df

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

        if not self.detailed:

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
            path=self._plot_path(),
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

    def _detailed_plot_df(self):
        buf = StringIO()
        DataFrame.print_df(self.df)
        logging.info("Dataframe:\n{df}".format(
            df=textwrap.indent(buf.getvalue(), prefix='  ')))

        # if self.training_time:
        #     y2_field = 'total_training_time'
        # else:
        #     y2_field = None

        stacked_bar_plot = DetailedStackedBarPlot(
            data=self.df,
            path=self._plot_path('plot_fancy'),
            x_field='x_field',
            y_field='percent',
            x_group='operation',
            # y2_field='total_training_time',

            hues_together=True,
            hatch='category',
            hue='resource_overlap',

            # hatches_together=True,
            # hatch='resource_overlap',
            # hue='category',

            xlabel='(RL algorithm, Simulator)',
            ylabel='Percent',
            title=self.plot_title,
            debug=self.debug,
        )
        stacked_bar_plot.plot()

        # stacked_bar_plot = DetailedStackedBarPlot(
        #     data=self.df,
        #     path=self._plot_path('plot_category'),
        #     x_field='x_field',
        #     # y_field='time_sec',
        #     y_field='total_training_time',
        #     hatch='operation',
        #     hue='category',
        #     xlabel='(RL algorithm, Simulator)',
        #     ylabel='Total training time (seconds)',
        #     title=self.plot_title,
        #     debug=self.debug,
        # )
        # stacked_bar_plot.plot()
        #
        # ignore_cols = ['category', 'region']
        # all_cols = set(self.df.keys())
        # numeric_cols = set([
        #     colname for colname in self.df.keys()
        #     if np.issubdtype(self.df[colname].dtype, np.number)])
        # non_numeric_cols = all_cols.difference(numeric_cols)
        #
        # groupby_cols = non_numeric_cols.difference(ignore_cols)
        # keep_cols = all_cols.difference(ignore_cols)
        # groupby = self.df[list(keep_cols)].groupby(by=list(groupby_cols))
        # resource_df = groupby.sum().reset_index()
        #
        # stacked_bar_plot = DetailedStackedBarPlot(
        #     data=resource_df,
        #     path=self._plot_path('plot_resource'),
        #     x_field='x_field',
        #     # y_field='time_sec',
        #     y_field='total_training_time',
        #     hatch='operation',
        #     hue='resource_overlap',
        #     xlabel='(RL algorithm, Simulator)',
        #     ylabel='Total training time (seconds)',
        #     title=self.plot_title,
        #     debug=self.debug,
        # )
        # stacked_bar_plot.plot()

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
        if self.debug:
            algo_env_pairs = [self._get_algo_env_from_dir(iml_dir) for iml_dir in self.iml_directories]
            logging.info("{klass}: {msg}".format(
                klass=self.__class__.__name__,
                msg=pprint_msg({
                    'iml_directories': self.iml_directories,
                    'algo_env_pairs': algo_env_pairs,
                })))
        self._init_directories()
        # self._check_can_add_training_time()
        self._read_df()
        self._normalize_df()
        self._add_df_fields(self.df)
        self.dump_plot_data()
        if self.skip_plot:
            logging.info("Skipping plotting {path} (--skip-plot)".format(path=self._plot_path))
        else:
            if self.detailed:
                self._detailed_plot_df()
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
            path=self._plot_data_path()))
        with open(self._plot_data_path(), 'w') as f:
            DataFrame.print_df(human_df, file=f)

        logging.info("> {name} @ csv plot data @ {path}".format(
            name=self.__class__.__name__,
            path=self._plot_csv_path()))

        human_df.to_csv(self._plot_csv_path(), index=False)

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
        legend_handles, legend_labels = self._legend_handles_labels()
        leg = fig.legend(legend_handles, legend_labels,
                         bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.)
        leg.draw_frame(False)

    def _legend_handles_labels(self):
        legend_handles = []
        def mk_rect(color):
            legend_rect = plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none')
            return legend_rect
        for i, group in enumerate(self.groups):
            if not self.keep_zero and self._all_zero(group):
                # If a stacked-bar element is zero in all the bar-charts, don't show it in the legend (--keep-zero false).
                continue
            legend_rect = mk_rect(self.colors[i])
            legend_handles.append(legend_rect)
        legend_labels = [self._group_to_label(group) for group in self.groups]
        if self.y2_field is not None and self.y2_axis_label is not None:
            # Last color is for y2-field bar.
            assert len(self.colors) == len(self.groups) + 1
            legend_rect = mk_rect(self.colors[-1])
            legend_handles.append(legend_rect)
            legend_labels.append(self.y2_axis_label)
        return legend_handles, legend_labels

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

            # Use (days, hours, minutes, seconds)
            # ax2.yaxis.set_major_formatter(DaysHoursMinutesSecondsFormatter())

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

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
        legend_handles, legend_labels = self._legend_handles_labels()
        logging.info({
            # 'handles': handles,
            # 'labels': labels,
            'legend_handles': legend_handles,
            'legend_labels': legend_labels,
        })
        ax.legend(legend_handles, legend_labels,
                  bbox_to_anchor=(0, 1.02, 1, 0.2),
                  loc='lower left',
                  mode='expand',
                  # ncol=len(legend_labels) - 1,
                  ncol=2,
                  # nrow=2,
                  fancybox=True, framealpha=0.5)

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
        # Turn off x-axis grid-lines (just want y-axis grid-lines)
        ax.grid(b=False, axis='x')
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


class DetailedStackedBarPlot:
    def __init__(self,
                 data, path,
                 x_field,
                 y_field,
                 x_group,
                 # operation
                 hatch,
                 # category
                 hue,
                 hues_together=False,
                 hatches_together=False,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 bar_width=0.33,
                 debug=False,
                 ):


        self.data = data
        self.path = path

        self.x_field = x_field
        self.y_field = y_field
        self.x_group = x_group

        self.hatch = hatch
        self.hue = hue
        # Must provide at least one of these as true.
        assert ( hues_together or hatches_together )
        # Must provide EXACTLY ON of these as true.
        assert not( hues_together and hatches_together )
        self.hues_together = hues_together
        self.hatches_together = hatches_together

        self.bar_width = bar_width

        self.debug = debug

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self._init_x_offsets()

        self.hatch_map = self.as_hatch_map(self.hatches(self.data))
        self.hue_map = self.as_color_map(self.hues(self.data))

    def _init_x_offsets(self):
        x_groups = self.x_groups()

        total_bar_width = 2/3
        # width per bar:
        # bar_width = 0.33
        if len(x_groups) > 0:
            self.bar_width = total_bar_width/len(x_groups)
        else:
            # Prevent division by zero for empty dataframes.
            self.bar_width = 0.33

        bar_xloc = self._calc_bar_xloc(len(x_groups), self.bar_width)
        x_group_offset = dict()
        for x_group, offset in zip(x_groups, bar_xloc):
            x_group_offset[x_group] = offset
        self.x_group_offset = x_group_offset

        x_field_offset = dict()
        for offset, x_field in enumerate(self.x_fields()):
            x_field_offset[x_field] = offset
        self.x_field_offset = x_field_offset

    def _xtick(self, x_field, x_group):
        return self.x_field_offset[x_field] + self.x_group_offset[x_group]

    @staticmethod
    def _calc_bar_xloc(n, w):
        """
        n = total number of bars
        w = width per bar

        :param n:
        :return:
        """
        i = np.arange(n) - (n // 2)
        if n % 2 == 0:
            # even:
            # n = 4
            # i = [0, 1, 2, 3]
            ws = i*w + w/2
        else:
            # odd:
            # n = 5
            # i = [0, 1, 2, 3, 4]
            ws = i*w
        return ws

    def x_fields(self):
        x_fields = sorted(self.data[self.x_field].unique())
        return x_fields

    def x_groups(self):
        x_groups = sorted(self.data[self.x_group].unique())
        return x_groups

    def hatches(self, data):
        hatches = sorted(data[self.hatch].unique())
        return hatches

    def hues(self, data):
        hues = sorted(data[self.hue].unique())
        return hues

    def as_hatch_map(self, xs):
        # Need enough distinct hash-styles to fit categories.
        # assert len(xs) <= len(HATCH_STYLES)
        if len(xs) > len(HATCH_STYLES):
            logging.info("> WARNING: We only have {h} HATCH_STYLES, but there are {x} category-overlap labels".format(
                h=len(HATCH_STYLES),
                x=len(xs),
            ))
        hatch_map = dict()
        for x, hatch_style in zip(xs, itertools.cycle(HATCH_STYLES)):
            hatch_map[x] = hatch_style
        return hatch_map

    def get_color_map(self):
        cmap = plt.get_cmap('Pastel1')
        return cmap

    def as_color_map(self, xs):
        # https://matplotlib.org/examples/color/colormaps_reference.html
        # https://matplotlib.org/tutorials/colors/colormaps.html
        color_map = dict()
        cmap = self.get_color_map()
        # np.arange(0,1,(1 - 0)/5)
        for i, x in enumerate(xs):
            color = cmap(i % cmap.N)
            color_map[x] = color
        return color_map

    def plot(self):
        fig = plt.figure()
        # Q: What's this affect?
        # ax = plt.add_subplot(111)
        ax = fig.add_subplot(111)

        data = self.data.sort_values(by=[self.hatch, self.hue, self.x_field])

        # PSEUDOCODE:
        # bottom = zeroes(len(ys))
        # data.sort_values(by=[self.hatch, self.hue]
        # for hatch, hatch_df in data.groupby(by=[self.hatch]):
        #   for hue, hue_df in hatch_df.groupby(by=[self.hue])
        #     xs = hue_df[self.x]
        #     ys = hue_df[self.y]
        #     ax.bar(xs=xs, ys=ys, bottom=bottom,
        #       color=self.color_map[hue],
        #       hatch=self.hatch_map[hatch)
        #     bottom += ys

        if self.hues_together:
            self._plot_hues_together(data, fig, ax)
        else:
            assert self.hatches_together
            self._plot_hatches_together(data, fig, ax)

        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.title is not None:
            ax.set_title(self.title)

        self._add_legend(ax)
        self._add_xgroup_legend(ax)

        logging.info("Output csv @ {path}".format(path=self._csv_path))
        data.to_csv(self._csv_path, index=False)

        logging.info("Output dataframe @ {path}".format(path=self._df_path))
        with open(self._df_path, 'w') as f:
            DataFrame.print_df(data, file=f)
        DataFrame.print_df(data, file=sys.stdout)

        logging.info("Output plot @ {path}".format(path=self.path))
        fig.savefig(self.path, bbox_inches="tight")
        plt.close(fig)

        # self._add_lines(operation)
        # self._add_legend(operation)
        # self._add_axis_labels(operation)
        # self._show(fig, operation)

    def _plot_hatches_together(self, data, fig, ax):
        # fig = plt.figure()
        # # Q: What's this affect?
        # # ax = plt.add_subplot(111)
        # ax = fig.add_subplot(111)

        all_xs = []
        all_ys = []
        bar_kwargs = []
        # TODO: inter-change hue/hatch loops..?
        last_iter = None
        # NOTE: we need to use DIFFERNT bottom's for each x_group!
        # ... so iterate over x_group FIRST.
        # Q: Different x-fields may not have a particular x-group...?
        # We are ASSUMING that EVERY (algo,env) has (BP,Sim,Inf)
        x_groups = self.x_groups()

        all_x_fields = self.x_fields()
        ax.set_xticks(np.arange(len(all_x_fields)))
        ax.set_xticklabels(all_x_fields
                           # , rotation=rotation
                           )

        for x_group in x_groups:
            bottom = None
            xgroup_df = data[data[self.x_group] == x_group]
            hatches = self.hatches(xgroup_df)
            for hatch in hatches:
                hatch_df = xgroup_df[xgroup_df[self.hatch] == hatch]
                hues = self.hues(hatch_df)
                for hue in hues:
                    hue_df = hatch_df[hatch_df[self.hue] == hue]
                    x_fields = hue_df[self.x_field]
                    x_groups = hue_df[self.x_group]
                    xs = np.vectorize(self._xtick, otypes=[np.float])(hue_df[self.x_field], hue_df[self.x_group])
                    ys = hue_df[self.y_field].values
                    all_xs.append(xs)
                    all_ys.append(ys)
                    plot_kwargs = dict(
                        x=xs, height=ys, bottom=bottom,
                        width=self.bar_width,
                        # Color of hatch pattern.
                        edgecolor='black',
                        color=self.hue_map[hue],
                        hatch=self.hatch_map[hatch]
                    )
                    ax.bar(**plot_kwargs)
                    kw = dict(plot_kwargs)
                    kw.update({
                        'hue_field': hue,
                        'hatch_field': hatch,
                    })
                    bar_kwargs.append(kw)
                    if bottom is None:
                        bottom = np.zeros(len(ys))
                    assert not np.isnan(bottom).any()
                    assert not np.isnan(ys).any()
                    assert not np.isnan(bottom + ys).any()
                    bottom = bottom + ys
                    assert not np.isnan(bottom).any()

                    last_iter = dict(
                        x_fields=x_fields,
                        x_groups=x_groups,
                        xs=xs,
                        ys=ys,
                        bottom=bottom,
                    )

        logging.info("Plot debug:\n{msg}".format(
            msg=pprint_msg({
                'bar_kwargs': bar_kwargs,
            }),
        ))


    def _plot_hues_together(self, data, fig, ax):

        all_xs = []
        all_ys = []
        bar_kwargs = []
        # TODO: inter-change hue/hatch loops..?
        last_iter = None
        # NOTE: we need to use DIFFERNT bottom's for each x_group!
        # ... so iterate over x_group FIRST.
        # Q: Different x-fields may not have a particular x-group...?
        # We are ASSUMING that EVERY (algo,env) has (BP,Sim,Inf)
        x_groups = self.x_groups()

        all_x_fields = self.x_fields()
        ax.set_xticks(np.arange(len(all_x_fields)))
        ax.set_xticklabels(all_x_fields
                           # , rotation=rotation
                           )

        for x_group in x_groups:
            bottom = None
            xgroup_df = data[data[self.x_group] == x_group]
            hues = self.hues(xgroup_df)
            for hue in hues:
                hue_df = xgroup_df[xgroup_df[self.hue] == hue]
                hatches = self.hatches(hue_df)
                for hatch in hatches:
                    hatch_df = hue_df[hue_df[self.hatch] == hatch]
                    # xs = hatch_df[self.x_field].values
                    x_fields = hatch_df[self.x_field]
                    x_groups = hatch_df[self.x_group]
                    xs = np.vectorize(self._xtick, otypes=[np.float])(hatch_df[self.x_field], hatch_df[self.x_group])
                    ys = hatch_df[self.y_field].values
                    all_xs.append(xs)
                    all_ys.append(ys)
                    plot_kwargs = dict(
                        x=xs, height=ys, bottom=bottom,
                        width=self.bar_width,
                        # Color of hatch pattern.
                        edgecolor='black',
                        color=self.hue_map[hue],
                        hatch=self.hatch_map[hatch]
                    )
                    ax.bar(**plot_kwargs)
                    kw = dict(plot_kwargs)
                    kw.update({
                        'hue_field': hue,
                        'hatch_field': hatch,
                    })
                    bar_kwargs.append(kw)
                    if bottom is None:
                        bottom = np.zeros(len(ys))
                    assert not np.isnan(bottom).any()
                    assert not np.isnan(ys).any()
                    assert not np.isnan(bottom + ys).any()
                    bottom = bottom + ys
                    assert not np.isnan(bottom).any()

                    last_iter = dict(
                        x_fields=x_fields,
                        x_groups=x_groups,
                        xs=xs,
                        ys=ys,
                        bottom=bottom,
                    )

        logging.info("Plot debug:\n{msg}".format(
            msg=pprint_msg({
                'bar_kwargs': bar_kwargs,
            }),
        ))


    def _as_path(self, file_ext):
        path = self.path
        path = re.sub(r'\.[^.]+$', file_ext, path)
        assert path != self.path
        return path

    @property
    def _csv_path(self):
        return self._as_path('.csv')

    @property
    def _df_path(self):
        return self._as_path('.dataframe.txt')

    def get_plot_operations(self):
        if self.get_png is not None:
            all_benches = [NO_BENCH_NAME] + unique(self.mean_df['operation'])
            return all_benches

        return [NO_BENCH_NAME]

    def get_png_path(self, operation):
        if self.get_png is not None:
            return self.get_png(operation)

        return self.png

    def get_png_legend_path(self, operation):
        png_path = self.get_png_path(operation)
        legend_png_path = re.sub(r'\.png$', '.legend.png', png_path)
        return legend_png_path

        # if self.get_png_legend is not None:
        #     return self.get_png_legend(operation)
        #
        # return self.png_legend

    def get_plot_data_pt(self, operation):
        if self.get_plot_data_path is not None:
            return self.get_plot_data_path(operation)

        return self.plot_data_path

    def _show(self, fig, operation=None):
        if self.show:
            plt.show()
        else:
            print("> Save figure to {path}".format(path=self.get_png_path(operation)))
            print("> Save plot data to {path}".format(path=self.get_plot_data_pt(operation)))
            fig.savefig(self.get_png_path(operation), bbox_inches="tight")
            plt.close()

    def _add_lines(self, operation=None):

        self._bottom = None
        def _add_line(impl_name, operation):
            for category in self.category_order:
                rows = self.df[
                    (self.df['operation'] == operation)
                    & (self.df['category'] == category)
                    ]
                if len(rows) == 0:
                    continue
                xvalues = self._get_xvalues(rows['impl_name'], rows['device'])
                yvalues = rows['mean'].values
                yerr = rows['std'].values

                color = self.category_color_map[category]
                hatch = self.operation_hatch_map[operation]

                if self._bottom is None:
                    self._bottom = np.zeros_like(yvalues)

                # PROBLEM: if data is missing for step
                assert self._bottom.shape == yvalues.shape

                plot = plt.bar(xvalues, yvalues, color=color, width=self.bar_width, edgecolor='white', label=operation,
                               bottom=self._bottom,
                               hatch=hatch,
                               yerr=yerr)

                self._bottom += yvalues

        for impl_name in self.impl_name_order:
            if operation == NO_BENCH_NAME:
                for operation in self.operation_order:
                    _add_line(impl_name, operation)
            else:
                _add_line(impl_name, operation)

    def _add_xgroup_legend(self, ax, x_offset=0, y_offset=0):

        # legbox_spacer = 0.04
        legbox_spacer = 0
        # Place legend at top-left of plot-area.
        legend_kwargs = dict(
            loc='upper left',
            bbox_to_anchor=(0 + legbox_spacer, 1 - legbox_spacer),
            ncol=1,
            handlelength=0,
            handletextpad=0,
        )

        annotate_kwargs = dict(
            color='black',
            weight='bold',
        )

        patch_kwargs = dict(
            edgecolor='black',
            # facecolor='white',
        )

        def get_bar_label(i, x_group):
            # label = "({i})".format(i=i + 1)
            # label = "{i}".format(i=i + 1)
            label = "$({i})$".format(i=i + 1)
            return label

        def _add_xgroup_bar_labels():
            x_groups = self.x_groups()
            x_fields = self.x_fields()
            # Just label the first x-field with (1), (2), (3)
            # OR: BP, Inf, Sim
            x_field = x_fields[0]

            df = self.data
            xy_offset = (x_offset, y_offset)
            for i, x_group in enumerate(x_groups):
                x = self._xtick(x_field, x_group)
                sub_df = df[
                    (df[self.x_field] == x_field) &
                    (df[self.x_group] == x_group)]
                height = sub_df[self.y_field].sum()
                logging.info("Add bar-label:\n{msg}".format(msg=pprint_msg({
                    'x_field': x_field,
                    'x_group': x_group,
                    'height': height,
                    'x': x,
                })))
                label = get_bar_label(i, x_group)
                ax.annotate(
                    label,
                    # xy=(rect.get_x() + rect.get_width() / 2, height),
                    xy=(x, height),
                    xytext=xy_offset,
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    **annotate_kwargs)

        def _add_xgroup_legbox():
            x_groups = self.x_groups()
            patches = []
            for i, x_group in enumerate(x_groups):
                patch = mpatches.Patch(edgecolor='white', color='white')
                # patch = mpl_patches.Rectangle((0, 0), 0, 0, fc="white", ec="white",
                #                       lw=0, alpha=0)

                # patch = mpatches.Patch(**patch_kwargs)
                # patch = mpatches.Rectangle(**patch_kwargs)

                # label = get_bar_label(i, x_group)
                # # patch = ax.annotate(label, (0, 0),
                # #             ha='center',
                # #             va='center',
                # #             # edgecolor='black',
                # #             # xycoords='offset points'
                # #             **annotate_kwargs)
                # patch = ax.text(0, 0, label,
                #                 ha='center',
                #                 va='center',
                #                 # edgecolor='black',
                #                 # xycoords='offset points'
                #                 **annotate_kwargs)

                patches.append(patch)

            # labels = x_groups

            def _get_label(i, x_group):
                bar_label = get_bar_label(i, x_group)
                # txt = r"\textbf{{{bar_label}}} = {x_group}".format(
                #     bar_label=bar_label,
                #     x_group=x_group)
                txt = "{bar_label} {x_group}".format(
                    bar_label=bar_label,
                    x_group=x_group)
                # import ipdb; ipdb.set_trace()
                return txt
            labels = [_get_label(i, x_group) for i, x_group in enumerate(x_groups)]

            legend = ax.legend(handles=patches, labels=labels, **legend_kwargs)
            ax.add_artist(legend)

            # for i, (x_group, patch) in enumerate(zip(x_groups, patches)):
            #     bar_label = get_bar_label(i, x_group)
            #
            #     rx, ry = patch.get_xy()
            #     cx = rx + patch.get_width()/2.0
            #     cy = ry + patch.get_height()/2.0
            #
            #     # ax.annotate(bar_label, (cx, cy),
            #     #             ha='center',
            #     #             va='center',
            #     #             **annotate_kwargs)
            #
            #     txt = r"$\textbf{bar_label}$ = {x_group}".format(
            #         bar_label=bar_label,
            #         x_group=x_group)
            #     ax.annotate(
            #         txt, (cx, cy),
            #         ha='center',
            #         va='center',
            #         **annotate_kwargs)

            return legend

        _add_xgroup_bar_labels()
        _add_xgroup_legbox()




    def _add_legend(self, axis):
        self.legend_makers = []

        reversed_labels = False

        # Sometimes they are so many legend labels that the two separate legend boxes will overlap,
        # and it's hard to position the legend boxes "on top of each other".
        # So, we're better off making a single legend box.
        single_legend = True

        common_legend_kwargs = {
            'fancybox':True,
            # 'shadow':True,
            'labelspacing': 1.2,
            'handlelength': 3,
            'handleheight': 2,
        }

        # legend_spacer = 0.04
        legend_spacer = 0

        # We need two groups of lines:
        #
        # 1) Hatch-type:
        #    - Should have the same color
        #    - # of hash-types = len(category_order = ['GPUTimeSec', 'CppTimeSec', 'PythonTimeSec'])
        #                      = 3
        #
        # 2) Color-type:
        #    - Should have the same hatch.
        #    - # of color-types = len(operation_order = ['q_forward', 'q_backward', 'step'])
        #                       = 3

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        if not single_legend:
            legend_kwargs = []

        hatch_legend = LegendMaker(attr_name='hatch',
                                   field_to_attr_map=self.hatch_map,
                                   field_order=self.hatches(self.data),
                                   # labels=self.hatches,
                                   legend_kwargs={
                                       # 'labelspacing': 1.2,
                                       # 'handlelength': 3,
                                       # 'handleheight': 2,
                                   },
                                   reversed=reversed_labels)
        self.legend_makers.append(hatch_legend)
        if not single_legend:
            kwargs = dict(common_legend_kwargs)
            kwargs.update({
                    # NOTE:
                    # - Internally LegendMaker uses the figure coordinate system.
                    # - So, (1, 1) is the (right, top) of the whole figure,
                    #   so 1.04 makes it just a bit to the right of the whole figure
                    'bbox_to_anchor': (1 + legend_spacer, 1),
            })
            legend_kwargs.append(kwargs)

        color_legend = LegendMaker(attr_name='facecolor',
                                   field_to_attr_map=self.hue_map,
                                   field_order=self.hues(self.data),
                                   # labels=self.hues(self.data),
                                   edgecolor='white',
                                   legend_kwargs={
                                       # 'handlelength': 3,
                                       # 'handleheight': 2,
                                   },
                                   reversed=reversed_labels)
        self.legend_makers.append(color_legend)
        if not single_legend:
            kwargs = dict(common_legend_kwargs)
            kwargs.update({
                # 'loc':'lower left',
                # 'loc':'left',
                # 'bbox_to_anchor': (0, -1),

                # Place legend beneath plot so it has room to grow when there are lots of labels,
                # without overlapping the other legend.
                # Sadly, I still don't understand how this thing works.
                # (e.g. I'm not sure how to left-align the legend beneath the plot... OH WELL).
                #
                # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
                # 'loc':'upper center',
                # 'bbox_to_anchor':(0.5, -0.05),

                'loc':'lower left',
                'bbox_to_anchor':(1 + legend_spacer, 0.0),

            })
            legend_kwargs.append(kwargs)

        if not single_legend:
            LegendMaker.add_legends_multiple(
                self.legend_makers,
                axis=axis,
                legend_kwargs=legend_kwargs)

        if single_legend:
            kwargs = dict(common_legend_kwargs)
            kwargs.update({
                'loc':'top left',
                'bbox_to_anchor':(1 + legend_spacer, 1.0),
            })
            LegendMaker.add_legends_single(
                self.legend_makers,
                axis=axis,
                legend_kwargs=kwargs)

        # LegendMaker.add_legends_vertically(self.legend_makers,
        #                         legend_kwargs=legend_kwargs)


    # def _get_categories(self, json_data):
    #     return list(k for k in json_data.keys() if k in self.category_order)

    @property
    def dataframe(self):
        if self.df is None:
            self._as_dataframe()
        return self.df

    def _as_dataframe(self):
        self._build_operation_order()

        # devices = list(data.keys())
        self.orig_df = pd.DataFrame(self.df_data)

        self.df = DataFrame.get_mean_std(self.orig_df, self.value_field)
        logging.info("> DATAFRAME BEFORE SORT:")
        logging.info(self.df)
        self.df = self.df.sort_values(by=['impl_name_order', 'operation_order', 'category_order'])
        # self.df = self.df.sort_values(by=['impl_name_order', 'operation_order', 'category_order'], ascending=False)
        # self.df = self.df.sort_values(by=['operation_order'])
        logging.info("> DATAFRAME AFTER SORT:")
        logging.info(self.df)
        # groupby_cols = DataFrame.get_groupby_cols(self.orig_df, value_field)
        self.df['std_div_mean_percent'] = 100 * self.df['std']/self.df['mean']

        self.mean_df = self.df

    def _add_axis_labels(self, operation=None):
        if self.title is not None:
            plt.title(self.title)

        if self.xlabel is not None:
            # , fontweight='bold'
            if operation == NO_BENCH_NAME:
                plt.xlabel(self.xlabel)
            else:
                plt.xlabel(get_pretty_bench(operation))

        if self.ylabel is not None:
            plt.ylabel(self.ylabel)


        n_bars = len(self.data[self.x_field].unique())
        xtick_xvalues = self._xtick_xvalues(self.impl_name_order, self.impl_name_order_map, n_bars)
        plt.xticks(xtick_xvalues, self.impl_name_order)


    def _get_xvalue(self, impl_name, device):
        bench_order = self.impl_name_order_map[impl_name]
        graphics_order = self.device_order_map[device]
        return bench_order + graphics_order*self.bar_width

    def _get_xvalues(self, impl_names, devices):
        return np.array([self._get_xvalue(impl_name, device) \
                         for impl_name, device in zip(impl_names, devices)])

    # Add xticks on the middle of the group bars
    def _xtick_xvalues(self, xvalues, order_map, n_bars):
        idxes = [order_map[xvalue] for xvalue in xvalues]
        all_bars_width = n_bars * self.bar_width
        # This may be wrong.
        center_width = ((n_bars - 1)*self.bar_width)/2
        return [i*all_bars_width + center_width \
                for i in idxes]

    def _check_has_keys(self, xs, xs_map):
        for x in xs:
            assert x in xs_map


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

    # def subtract_overhead(self,
    #                       overhead_event_count_json,
    #                       cupti_overhead_json,
    #                       LD_PRELOAD_overhead_json,
    #                       pyprof_overhead_json):
    #     """
    #     - CUDA API interception:
    #       Subtract from:
    #       [CPU, q_forward, TensorFlow C++]
    #
    #     - CUPTI overhead:
    #       Subtract from:
    #       [CPU, q_forward, CUDA API]
    #
    #     - Python -> C-library interception:
    #       Subtract from:
    #       [CPU, q_forward, Python]
    #
    #     - Python annotations:
    #       Subtract from:
    #       [CPU, q_forward, Python]
    #     """
    #
    #     # 'machine', 'process', 'phase', 'resource_overlap', 'operation'
    #
    #     # e.g. subtracting Python annotation time.
    #     # The approach will be similar for other overhead types.
    #
    #     for machine_name, process_name, phase_name, operation_name in self._each_operation():
    #
    #         # Python annotations:
    #         total_pyprof_annotation = overhead_event_count_json['pyprof_annotation'][machine_name][process_name][phase_name][operation_name]
    #         per_pyprof_annotation_sec = pyprof_overhead_json['mean_pyprof_annotation_per_call_us']/USEC_IN_SEC
    #         pyprof_annotation_sec = per_pyprof_annotation_sec * total_pyprof_annotation
    #         self.subtract_from_resource(
    #             resource='CPU',
    #             selector=dict(
    #                 machine=machine_name,
    #                 process=process_name,
    #                 phase=phase_name,
    #                 operation=operation_name,
    #                 category=CATEGORY_PYTHON,
    #             ),
    #             subtract_sec=pyprof_annotation_sec)
    #
    #         # Python -> C-library interception:
    #         total_pyprof_interception = overhead_event_count_json['pyprof_interception'][machine_name][process_name][phase_name][operation_name]
    #         per_pyprof_interception_sec = pyprof_overhead_json['mean_pyprof_interception_overhead_per_call_us']/USEC_IN_SEC
    #         pyprof_interception_sec = per_pyprof_interception_sec * total_pyprof_interception
    #         self.subtract_from_resource(
    #             resource='CPU',
    #             selector=dict(
    #                 machine=machine_name,
    #                 process=process_name,
    #                 phase=phase_name,
    #                 operation=operation_name,
    #                 category=CATEGORY_PYTHON,
    #             ),
    #             subtract_sec=pyprof_interception_sec)
    #
    #
    #         missing_cupti_overhead_cuda_api_calls = dict()
    #         # CUPTI overhead:
    #         for cuda_api_name, num_api_calls in overhead_event_count_json['cuda_api_call'][machine_name][process_name][phase_name][operation_name].items():
    #             if cuda_api_name not in cupti_overhead_json:
    #                 missing_cupti_overhead_cuda_api_calls[cuda_api_name] = missing_cupti_overhead_cuda_api_calls.get(cuda_api_name, 0) + 1
    #             else:
    #                 per_cuda_api_sec = cupti_overhead_json[cuda_api_name]['mean_cupti_overhead_per_call_us']/USEC_IN_SEC
    #                 cupti_overhead_sec = per_cuda_api_sec * num_api_calls
    #                 self.subtract_from_resource(
    #                     resource='CPU',
    #                     selector=dict(
    #                         machine=machine_name,
    #                         process=process_name,
    #                         phase=phase_name,
    #                         operation=operation_name,
    #                         category=CATEGORY_CUDA_API_CPU,
    #                     ),
    #                     subtract_sec=cupti_overhead_sec)
    #         if len(missing_cupti_overhead_cuda_api_calls) > 0:
    #             logging.warning("Saw CUDA API calls that we didn't have calibrated CUPTI overheads for overheads for {path}: {msg}".format(
    #                 path=self.db_path,
    #                 msg=pprint_msg(missing_cupti_overhead_cuda_api_calls),
    #             ))
    #
    #         # CUDA API interception:
    #         total_cuda_api_calls = np.sum([num_api_calls for cuda_api_name, num_api_calls in
    #                                        overhead_event_count_json['cuda_api_call'][machine_name][process_name][phase_name][operation_name].items()])
    #         per_LD_PRELOAD_sec = pyprof_overhead_json['mean_interception_overhead_per_call_us']/USEC_IN_SEC
    #         LD_PRELOAD_sec = per_LD_PRELOAD_sec * total_cuda_api_calls
    #         self.subtract_from_resource(
    #             resource='CPU',
    #             selector=dict(
    #                 machine=machine_name,
    #                 process=process_name,
    #                 phase=phase_name,
    #                 operation=operation_name,
    #                 # ASSUMPTION: GPU calls are being made from inside the Tensorflow C++ API...
    #                 # this might not hold once we start measuring other libraries the use the GPU...
    #                 # NOTE: this overhead does NOT come from the CUDA API call itself; the overhead is
    #                 # "around" the CUDA API call.
    #                 category=CATEGORY_TF_API,
    #             ),
    #             subtract_sec=LD_PRELOAD_sec)
    #
    # def subtract_from_resource(self, resource, selector, subtract_sec):
    #     """
    #     To handle the fact that we cannot precisely attribute profiling-overhead CPU-time to [CPU], or [CPU, GPU],
    #     we decide to perform this heuristic:
    #     - Subtract from [CPU] or [CPU, GPU], in the order of whichever is largest FIRST
    #     - If we end up subtracting all of the first resource-group,
    #       then subtract what remains from the next resource-group.
    #
    #     NOTE: if we subtract CPU-time from [CPU, GPU] the new time that remains is GPU-only time...
    #
    #     :param resource:
    #     :param selector:
    #     :param subtract_sec:
    #     :return:
    #     """
    #     resource_selector = dict(selector)
    #     resource_selector['plot_type'] = 'ResourceOverlap'
    #     resource_selector = only_selector_fields(resource_selector)
    #     resource_vd = self.get_file(resource_selector)
    #     # e.g.
    #     # resource = 'CPU'
    #     # resource_types = [['CPU'], ['CPU', 'GPU']]
    #     keys = resource_vd.keys()
    #     resource_types = [resource_type for resource_type in keys if resource in resource_type]
    #     def sort_by_time(key):
    #         return -1 * resource_vd.get_size(key)
    #     # by = {by total time spent in resource in descending order}
    #     resource_types.sort(key=sort_by_time)
    #     subtract_left_sec = subtract_sec
    #     for resource_type in resource_types:
    #         if subtract_left_sec == 0:
    #             break
    #         # vd_leaf = vd_tree.lookup(machine, process, phase, operation, category)
    #         # vd_selector = copy.copy(selector)
    #         # vd_selector['resource_overlap'] = resource_type
    #         # Q: are we selecting for ResourceOverlap or ResourceSubplot?
    #         # vd_leaf = self.get_file(selector)
    #
    #         # to_subtract = min(
    #         #     subtract_left_sec,
    #         #     vd.time_sec(resource_type, process, phase, operation, category))
    #         # key_type = plot_index.KEY_TYPE[plot_type]
    #         # key = selector[key_type]
    #         to_subtract = min(
    #             subtract_left_sec,
    #             # vd_leaf.get_size(key),
    #             resource_vd.get_size(resource_type),
    #         )
    #
    #         # We need to "propagate up" the subtraction;
    #         # vd_tree.subtract handles this.
    #         # i.e. If we are subtracting from:
    #         #   [CPU, q_forward, Python]
    #         # Then, we need to subtract from:
    #         #   [CPU, q_forward, Python]
    #         #     CategoryOverlap.machine_{...}.process_{...}.phase_{...}.ops_{...}.resources_{...}.venn_js.json
    #         #   [CPU, q_forward]
    #         #     OperationOverlap.machine_{...}.process_{...}.phase_{...}.resources_{...}.venn_js.json
    #         #   [CPU]
    #         #     ResourceOverlap.machine_{...}.process_{...}.phase_{...}.venn_js.json
    #         #     SKIP: ResourceSubplot.machine_{...}.process_{...}.phase_{...}.venn_js.json
    #         # vd_tree.subtract(machine, process, phase, resource_type, operation, category, to_subtract)
    #         self.subtract(resource_selector, to_subtract)
    #         subtract_left_sec -= to_subtract
    #
    # def subtract(self, selector, subtract_sec):
    #     # selector = {
    #     #     'machine': machine
    #     #     'process': process
    #     #     'phase': phase
    #     #     'resource_type': resource_type,
    #     #     'operation': operation
    #     #     'category': category
    #     # }
    #     for plot_type in plot_index.OVERLAP_PLOT_TYPES:
    #         # e.g. ResourceOverlap: [machine, process, phase]
    #         # selector[just keep plot_type.attributes]
    #         plot_type_selector = dict((k, v) for k, v in selector.items()
    #                                   if k in plot_index.SEL_ORDER[plot_type])
    #         plot_type_selector['plot_type'] = plot_type
    #         vd = self.get_file(plot_type_selector)
    #
    #         # def vd.key_field():
    #         #  ResourceSubplot -> ListOf[resource_type]
    #         #  OperationOverlap -> operation
    #         #  CategoryOverlap -> category
    #         #  ResourceSubplot -> resource_type
    #         key_field = plot_index.KEY_TYPE[plot_type]
    #         key = selector[key_field]
    #         vd.subtract(key, subtract_sec, inplace=True)

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
        # if human_readable:
        x_field = "({algo}, {env})".format(
            algo=algo,
            env=short_env,
        )
        # else:
        #     x_field = "{algo}\n{env}".format(
        #         algo=algo,
        #         env=short_env,
        #     )
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

def join_plus(xs):
    return ' + '.join(sorted(xs))

if __name__ == '__main__':
    main()
