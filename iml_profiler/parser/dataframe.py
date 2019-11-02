import pandas as pd
import copy
import os
import pprint
import progressbar
import logging
import functools
import multiprocessing

from concurrent.futures import ProcessPoolExecutor

from iml_profiler.profiler import concurrent

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.parser.common import *

def Worker_split_map_merge(kwargs):
    kwargs = dict(kwargs)
    self = kwargs['self']
    # del kwargs['self']
    map_fn = kwargs['map_fn']
    merge_fn = kwargs['merge_fn']

    # del kwargs['map_fn']
    split = kwargs['split']
    debug = kwargs['debug']

    results = []
    for proto_path in split:
        df = self.read_one_df(proto_path)
        result = map_fn(df)
        results.append(result)

    merged_result = functools.reduce(merge_fn, results)

    return merged_result

class BaseDataframeReader:
    """
    Read training_progress.trace_*.proto files into data-frame.

    Q: How should read experiment_config.json?
       Can we add something that read additional dict-attributes for each directory?
    """
    def __init__(self, directory, add_fields=None, colnames=None, debug=False, debug_single_thread=False):
        self.directory = directory
        # column-name -> [values]
        # self.data = None
        self.df = None
        # add_fields(iml_directory) -> dict of fields to add to data-frame
        self.add_fields = add_fields
        if colnames is not None:
            colnames = set(colnames)
        self._colnames = colnames
        self.added_fields = set()
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.iml_config = read_iml_config(self.directory)
        self.iml_columns = self._get_iml_columns()
        self.colnames = self._get_colnames()

    def split_map_merge(self, name, map_fn, merge_fn,
                        n_workers=None,
                        debug=False,
                        debug_single_thread=False):
        """
        :param name:
        :param n_workers:
        :param map_fn:
            Given a proto_path, run a function on it that produces (most likely) a dataframe, or
            some computation that reduces a dataframe.
        :param merge_fn:
            Given two inputs A and B, where A and/or B are one of:
            - a result from map_fn
            - a result from merge_fn
            Return the result of "merging" A and B.
        :param debug_single_thread:
        :return:
        """
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        proto_paths = self.proto_paths
        def Args_split_map_merge(split):
            return dict(
                self=self,
                split=split,
                map_fn=map_fn,
                merge_fn=merge_fn,
                debug=debug,
            )
        with ProcessPoolExecutor(n_workers) as pool:
            # splits = split_list(proto_paths, n_workers)
            splits = split_list_no_empty(proto_paths, n_workers)
            kwargs_list = [Args_split_map_merge(split) for split in splits]
            split_results = concurrent.map_pool(pool, Worker_split_map_merge, kwargs_list,
                                             desc="split_map_merge.{name}".format(name=name),
                                             show_progress=True,
                                             sync=debug_single_thread)
        merged_result = functools.reduce(merge_fn, split_results)
        # if self.debug:
        #     logging.info("split_map_merge.{name}: {msg}".format(
        #         name=name,
        #         msg=pprint_msg({
        #             'merged_result': merged_result,
        #         })
        #     ))
        return merged_result

    def merge_from_map(self, map_fn, a, b):
        # if self.debug:
        #     logging.info("merge_from_map: {msg}".format(
        #         msg=pprint_msg({
        #             'a': a,
        #             'b': b,
        #         })
        #     ))
        df = pd.concat([a, b])
        result = map_fn(df)
        # if self.debug:
        #     logging.info("merge_from_map: {msg}".format(
        #         msg=pprint_msg({
        #             'result': result,
        #         })
        #     ))
        return result

    def _get_colnames(self):
        return self._colnames.union(set(self.iml_columns))

    def _add_iml_config_columns(self, data=None):
        assert data is not None
        # if data is None:
        #     data = self.data
        if 'metadata' not in self.iml_config:
            return
        def _get(col):
            self._add_col(col, self.iml_config['metadata'].get(col, ''), data=data)
        # Q: should we just set ALL the metadata?
        _get('algo')
        _get('env')

    def _get_iml_columns(self):
        if 'metadata' not in self.iml_config:
            return []
        return ['algo', 'env']

    def _add_col(self, colname, value, data=None):
        assert data is not None
        # if data is None:
        #     data = self.data
        if self.colnames is not None:
            assert colname in self.colnames
        if colname not in data:
            data[colname] = []
        data[colname].append(value)

    def _add_col_to_data(self, colname, value, data=None):
        assert data is not None
        # if data is None:
        #     data = self.data
        if colname not in data:
            data[colname] = []
        data[colname].append(value)

    def _add_columns(self, colnames, proto, data=None):
        assert data is not None
        # if data is None:
        #     data = self.data
        for col in colnames:
            val = getattr(proto, col)
            self._add_col(col, val, data=data)

    def _add_columns_to_data(self, colnames, proto, data=None):
        for col in colnames:
            val = getattr(proto, col)
            self._add_col_to_data(col, val, data=data)

    def _check_cols(self, data=None):
        assert data is not None
        # if data is None:
        #     data = self.data
        col_to_length = dict((col, len(data[col])) for col in data.keys())
        if len(set(col_to_length.values())) > 1:
            raise RuntimeError("Detected inconsistent column lengths:\n{dic}\n{data}".format(
                dic=pprint.pformat(col_to_length),
                data=pprint.pformat(data),
            ))

    def is_proto_file(self, path):
        raise NotImplementedError()

    def empty_dataframe(self):
        if self._colnames is None:
            raise NotImplementedError(
                "If we don't find any proto files, this method should return a dataframe "
                "with all the columns we would expect to find but didn't; to do this, please set self._colnames")
        data = dict((colname, []) for colname in self.colnames)
        df = pd.DataFrame(data)
        return df

    def zero_dataframe(self, zero_colnames):
        if self._colnames is None:
            raise NotImplementedError(
                "If we don't find any proto files, this method should return a dataframe "
                "with all the columns we would expect to find but didn't; to do this, please set self._colnames")
        data = dict()
        for colname in zero_colnames:
            data[colname] = [0]
        self._add_iml_config_columns(data=data)
        df = pd.DataFrame(data)
        return df

    def add_proto_cols(self, path, data=None):
        raise NotImplementedError()

    def _maybe_add_fields(self, path, data=None):
        if self.add_fields is not None:
            extra_fields = self.add_fields(path)
            if extra_fields is not None:
                self.added_fields.update(extra_fields.keys())
                for key, value in extra_fields.items():
                    self._add_col_to_data(key, value, data=data)

        self._add_iml_config_columns(data=data)

    @property
    def proto_paths(self):
        proto_paths = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            for base in filenames:
                path = _j(dirpath, base)
                if self.is_proto_file(path):
                    proto_paths.append(path)
        return proto_paths

    def read_one_df(self, proto_path):
        data = dict()
        self.add_proto_cols(proto_path, data=data)
        self._check_cols(data=data)
        df = pd.DataFrame(data)
        return df

    def _read_df(self):
        # self.data = dict()
        data = dict()
        proto_paths = self.proto_paths
        for proto_path in progress(proto_paths, desc="{klass}.read dataframe".format(
            klass=self.__class__.__name__), show_progress=True):
            self.add_proto_cols(proto_path, data=data)
            self._check_cols(data=data)
        if len(proto_paths) == 0:
            logging.warning("{klass}: Saw 0 proto paths rooted at {dir}; returning empty dataframe".format(
                klass=self.__class__.__name__,
                dir=self.directory,
            ))
            self.df = self.empty_dataframe()
        else:
            self.df = pd.DataFrame(data)

    def read_each(self):
        proto_paths = self.proto_paths
        for proto_path in progress(proto_paths, desc="{klass}.read dataframe".format(
            klass=self.__class__.__name__), show_progress=True):
            data = dict()
            self.add_proto_cols(proto_path, data=data)
            self._check_cols(data=data)
            df = pd.DataFrame(data)
            yield df
        if len(proto_paths) == 0:
            logging.warning("{klass}: Saw 0 proto paths rooted at {dir}; returning empty dataframe".format(
                klass=self.__class__.__name__,
                dir=self.directory,
            ))
            df = self.empty_dataframe()
            yield df

    def read(self):
        if self.df is None:
            self._read_df()
        return self.df

class UtilDataframeReader(BaseDataframeReader):

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            # MachineUtilization from pyprof.proto
            'machine_name',
            # DeviceUtilization from pyprof.proto
            'device_name',
            # UtilizationSample from pyprof.proto
            'util',
            'start_time_us',
            'total_resident_memory_bytes',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_machine_util_file(path)

    def add_proto_cols(self, path, data=None):
        machine_util = read_machine_util_file(path)
        if self.debug:
            logging.info("Read MachineUtilization from {path}".format(path=path))
        for device_name, device_utilization in machine_util.device_util.items():
            for sample in device_utilization.samples:
                self._add_col('machine_name', machine_util.machine_name, data=data)
                self._add_col('device_name', device_name, data=data)

                self._add_col('util', sample.util, data=data)
                self._add_col('start_time_us', sample.start_time_us, data=data)
                self._add_col('total_resident_memory_bytes', sample.total_resident_memory_bytes, data=data)

                self._maybe_add_fields(path, data=data)

class TrainingProgressDataframeReader(BaseDataframeReader):

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False, add_algo_env=False):

        colnames = [
            # IncrementalTrainingProgress from pyprof.proto
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

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)
        self.add_algo_env = add_algo_env

    def is_proto_file(self, path):
        return is_training_progress_file(path)

    def add_proto_cols(self, path, data=None):
        training_progress = read_training_progress_file(path)
        self._add_columns(self._colnames, training_progress, data=data)
        self._maybe_add_fields(path, data=data)

    def last_progress(self):
        # TODO: could be improved by avoiding reading entire df into memory;
        # Instead, we want to keep track of the row that has np.max(df['end_training_time_us']) seen so far.
        # NOTE: I can't think of an elegant way to perform this type of think efficiently automatically.
        # We can also think of how to [split, map, merge, memoize] this by taking the max across each file.
        # Map the same thing across each file (spark)
        # Merge, a.k.a. aggregate all the mapped results to a single place
        # NOTE: I should've used Spark for this!

        # df = self.read()
        # last_df = ( df[df['end_training_time_us'] == np.max(df['end_training_time_us'])] )
        # return last_df

        if len(self.proto_paths) == 0:
            df = self.empty_dataframe()
            return df
        split_map_merge = SplitMapMerge_TrainingProgressDataframeReader__last_progress(obj=self)
        last_df = self.split_map_merge(
            'last_progress', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return last_df

    def training_duration_us(self):
        df = self.last_progress()
        training_duration_us = df['end_training_time_us'] - df['start_training_time_us']
        assert len(training_duration_us) == 1
        training_duration_us = training_duration_us.values[0]
        return training_duration_us

    def training_iterations(self):
        df = self.last_progress()
        training_iterations = df['end_num_timesteps'] - df['start_num_timesteps']
        assert len(training_iterations) == 1
        training_iterations = training_iterations.values[0]
        return training_iterations

    def training_duration_df(self):
        df = copy.copy(self.last_progress())
        df['training_duration_us'] = df['end_training_time_us'] - df['start_training_time_us']
        keep_cols = [
            'training_duration_us',
            'algo',
            'env',
        ]
        df = df[keep_cols]
        return df

class SplitMapMerge_CUDAAPIStatsDataframeReader__per_api_stats:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        per_api_stats_df = self.obj._compute_per_api_stats(df)
        return per_api_stats_df

    def merge_fn(self, a, b):
        return self.obj.merge_from_map(self.map_fn, a, b)


class SplitMapMerge_PyprofDataframeReader__total_intercepted_calls:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        was_intercepted_call_event = df['category'].apply(self.is_intercepted_call_event)
        intercepted_call_event_df = df[was_intercepted_call_event]
        total_intercepted_calls = len(intercepted_call_event_df)
        return total_intercepted_calls

    def merge_fn(self, a, b):
        return a + b

    def is_intercepted_call_event(self, category):
        # PROBLEM: If we add new categories of events, this will break.
        # We already DID add new event categories: CATEGORY_PROF_...
        # All we can really do keep track of a set of "CATEGORIES_C_EVENTS" that represent "Python -> C" interceptions.
        # However, if we ever add 'custom categories' in the future, that approach will break.
        # We should check for that somehow...
        # return category not in {CATEGORY_OPERATION, CATEGORY_PYTHON}
        return category in CATEGORIES_C_EVENTS

class SplitMapMerge_PyprofDataframeReader__total_op_events:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        is_op_event = np.vectorize(PyprofDataframeReader.is_op_event, otypes=[np.bool])(df['name'], df['category'])
        op_df = df[is_op_event]
        total_op_events = len(op_df)
        return total_op_events

    def merge_fn(self, a, b):
        return a + b

class SplitMapMerge_PyprofDataframeReader__total_op_process_events:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        is_op_proc_event = np.vectorize(is_op_process_event, otypes=[np.bool])(df['name'], df['category'])
        op_proc_df = df[is_op_proc_event]
        total_op_proc_events = len(op_proc_df)
        return total_op_proc_events

    def merge_fn(self, a, b):
        return a + b

class SplitMapMerge_PyprofDataframeReader__len_df:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        return len(df)

    def merge_fn(self, a, b):
        return a + b

class SplitMapMerge_PyprofDataframeReader__total_pyprof_overhead_us:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        # Filter for events that have profiling overhead recorded.
        df = df[df['start_profiling_overhead_us'] != 0]
        total_pyprof_overhead_us = np.sum(df['duration_profiling_overhead_us'])
        return total_pyprof_overhead_us

    def merge_fn(self, a, b):
        return a + b

class SplitMapMerge_PyprofDataframeReader__total_pyprof_overhead_df:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        groupby_cols = self.obj.iml_columns
        agg_cols = ['duration_profiling_overhead_us']
        keep_cols = sorted(set(groupby_cols + agg_cols))
        df_keep = df[keep_cols]
        groupby = df_keep.groupby(groupby_cols)
        df = groupby.sum().reset_index()
        return df

    def merge_fn(self, a, b):
        return self.obj.merge_from_map(self.map_fn, a, b)

class SplitMapMerge_TrainingProgressDataframeReader__last_progress:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        last_df = ( df[df['end_training_time_us'] == np.max(df['end_training_time_us'])] )
        # NOTE: sometimes we have duplicate rows, presumably because their exist two dumps of
        # TrainingProgressDataframeReader with the same data (not sure why)...
        last_df = last_df.drop_duplicates()
        return last_df

    def merge_fn(self, a, b):
        return self.obj.merge_from_map(self.map_fn, a, b)

class SplitMapMerge_CUDAAPIStatsDataframeReader__n_total_calls:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        n_total_calls = np.sum(df['num_calls'])
        return n_total_calls

    def merge_fn(self, a, b):
        return a + b

class CUDAAPIStatsDataframeReader(BaseDataframeReader):

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            # CUDAAPIPhaseStatsProto from iml_prof.proto
            'process_name',
            'machine_name',
            'phase_name',

            # CUDAAPIThreadStatsProto from iml_prof.proto
            'tid',
            'api_name',
            'total_time_us',
            'num_calls',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_cuda_api_stats_file(path)

    def add_proto_cols(self, path, data=None):
        proto = read_cuda_api_stats_file(path)
        # if self.debug:
        #     logging.info("Read CUDAAPIPhaseStatsProto from {path}".format(path=path))

        for api_thread_stats in proto.stats:
            self._add_col('process_name', proto.process_name, data=data)
            self._add_col('machine_name', proto.machine_name, data=data)
            self._add_col('phase_name', proto.phase, data=data)

            self._add_col('tid', api_thread_stats.tid, data=data)
            self._add_col('api_name', api_thread_stats.api_name, data=data)
            self._add_col('total_time_us', api_thread_stats.total_time_us, data=data)
            self._add_col('num_calls', api_thread_stats.num_calls, data=data)

            self._maybe_add_fields(path, data=data)

    def n_total_calls(self):
        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMerge_CUDAAPIStatsDataframeReader__n_total_calls(obj=self)
        n_total_calls = self.split_map_merge(
            'n_total_calls', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return n_total_calls
        # df = self.read()
        # n_total_calls = np.sum(df['num_calls'])
        # return n_total_calls

    def per_api_stats(self):
        if len(self.proto_paths) == 0:
            df = self.empty_dataframe()
            return df
        split_map_merge = SplitMapMerge_CUDAAPIStatsDataframeReader__per_api_stats(obj=self)
        per_api_stats_df = self.split_map_merge(
            'per_api_stats', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return per_api_stats_df

        # def map_fn(df):
        #     per_api_stats_df = self._compute_per_api_stats(df)
        #     return per_api_stats_df
        # def merge_fn(a, b):
        #     return self.merge_from_map(map_fn, a, b)
        # per_api_stats_df = self.split_map_merge(
        #     'per_api_stats', map_fn, merge_fn,
        #     debug=self.debug,
        #     debug_single_thread=self.debug_single_thread)
        # return per_api_stats_df

        # def merge(df1, df2):
        #     if df1 is None:
        #         return df2
        #     if df2 is None:
        #         return df1
        #     return self._compute_per_api_stats(pd.concat([df1, df2]))
        #
        # per_api_stats_df = None
        # groupby_cols = self._groupby_cols()
        # keep_cols = self._keep_cols()
        # for df in self.read_each():
        #     df_sum = self._df_group_sum(df, groupby_cols, keep_cols)
        #     # Q: How do we merge the per api stats?
        #     # df looks like:
        #     # api_name, algo, env, total_time_us, num_calls
        #     # Easiest option: concat and re-group.
        #     per_api_stats_df = merge(per_api_stats_df, df_sum)

    def _groupby_cols(self):
        groupby_cols = ['api_name'] + self.iml_columns
        return groupby_cols

    def _agg_cols(self):
        agg_cols = ['total_time_us', 'num_calls']
        return agg_cols

    def _keep_cols(self):
        keep_cols = sorted(set(self._groupby_cols() + self._agg_cols()))
        return keep_cols

    def _df_group_sum(self, df, groupby_cols, keep_cols):
        """
        Output looks like:
        api_name, algo, env, total_time_us, num_calls

        :param df:
        :param keep_cols:
        :param groupby_cols:
        :return:
        """
        df_keep = df[keep_cols]
        groupby = df_keep.groupby(groupby_cols)
        df_sum = groupby.sum().reset_index()
        return df_sum

    def _compute_per_api_stats(self, df):
        """
        Group all CUDA API calls by CUDA API name.
        # e.g. cudaKernelLaunch
        - api_name
        # How much total time (us) was spent in this API call over the course of the program?
        - total_time_us
        # How many times was this API call over the course of the program?
        - num_calls
        :return:
        """
        # df = self.read()
        # groupby_cols = ['api_name'] + self.iml_columns
        # agg_cols = ['total_time_us', 'num_calls']
        # keep_cols = sorted(set(groupby_cols + agg_cols))

        groupby_cols = self._groupby_cols()
        keep_cols = self._keep_cols()
        df_sum = self._df_group_sum(df, groupby_cols, keep_cols)

        # if self.debug:
        #     logging.info("_compute_per_api_stats: {msg}".format(
        #         msg=pprint_msg({
        #             'df_sum': df_sum,
        #         })
        #     ))

        return df_sum

class PyprofDataframeReader(BaseDataframeReader):

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            # CategoryEventsProto from pyprof.proto
            'process_name',
            'machine_name',
            'phase_name',
            'category',
            # Event from pyprof.proto
            'thread_id',
            'start_time_us',
            'duration_us',
            'name',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_pyprof_file(path)

    def add_proto_cols(self, path, data=None):
        proto = read_pyprof_file(path)
        # if self.debug:
        #     logging.info("Read CategoryEventsProto from {path}".format(path=path))

        # Event from pyprof.proto
        event_colnames = [
            'thread_id',
            'start_time_us',
            'duration_us',
            'name',
        ]

        def add_event(category, event):
            self._add_col_to_data('process_name', proto.process_name, data=data)
            self._add_col_to_data('machine_name', proto.machine_name, data=data)
            self._add_col_to_data('phase_name', proto.phase, data=data)

            self._add_col_to_data('category', category, data=data)

            self._add_columns_to_data(event_colnames, event, data=data)

            self._maybe_add_fields(path, data=data)

        # if self.debug:
        #     num_events = 0
        #     for category, event_list in proto.category_events.items():
        #         num_events += len(event_list.events)
        #     logging.info("{klass}.add_proto_cols path={path}".format(
        #         path=path,
        #         klass=self.__class__.__name__,
        #     ))
        #     # bar = progressbar.ProgressBar(
        #     #     prefix="{klass}.add_proto_cols".format(
        #     #         klass=self.__class__.__name__),
        #     #     max_value=num_events)

        try:
            i = 0
            for category, event_list in proto.category_events.items():
                for event in event_list.events:
                    add_event(category, event)
                    # if self.debug:
                    #     bar.update(i)
                    i += 1
        finally:
            pass
            # if self.debug:
            #     bar.finish()

    def total_intercepted_calls(self):
        """
        How many times did we perform interception:
        CATEGORY_PYTHON: Python -> C++
        <clib category>: C++ -> Python

        where <clib category> could be:
        - CATEGORY_SIMULATOR_CPP
        - CATEGORY_TF_API

        Everytime we have a C++ call, we record two events:

        :return:
        """

        # df = self.read()
        # def is_intercepted_call_event(category):
        #     # PROBLEM: If we add new categories of events, this will break.
        #     # We already DID add new event categories: CATEGORY_PROF_...
        #     # All we can really do keep track of a set of "CATEGORIES_C_EVENTS" that represent "Python -> C" interceptions.
        #     # However, if we ever add 'custom categories' in the future, that approach will break.
        #     # We should check for that somehow...
        #     # return category not in {CATEGORY_OPERATION, CATEGORY_PYTHON}
        #     return category in CATEGORIES_C_EVENTS
        # was_intercepted_call_event = df['category'].apply(is_intercepted_call_event)
        # intercepted_call_event_df = df[was_intercepted_call_event]
        # total_intercepted_calls = len(intercepted_call_event_df)
        # return total_intercepted_calls

        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMerge_PyprofDataframeReader__total_intercepted_calls(obj=self)
        total_intercepted_calls = self.split_map_merge(
            'total_intercepted_calls', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return total_intercepted_calls

    @staticmethod
    def is_op_event(event_name, category):
        return not is_op_process_event(event_name, category) and category == CATEGORY_OPERATION

    def total_annotations(self):
        return self.total_op_events()

    def total_op_events(self):
        """
        How many times did we record op-events using "with iml.prof.operation(...)"?

        Every time we execute this annotation, we record the following pyprof events:

        Here are some of the events that get recorded during pyprof.
        - Op-events:
          Category: CATEGORY_OPERATION, Event(name='sample_action')
          Operation annotation for the duration of the operation.

        - Process-events:
          Category: CATEGORY_OPERATION, Event(name='[process_name]')
          Single operation annotation recorded once at termination that measures
          the total duration of the process.

        - Remaining python time at end of operation:
          Category: CATEGORY_PYTHON, Event(name='Finish python benchmark')
          The amount of time in between returning from the TensorFlow C++ API,
          and finishing an operation annotation.

        - Python-events:
          Category: CATEGORY_PYTHON, Event(name=<CLIB__*>)
          Time spent in python before making a call into a wrapped C library (i.e. simulator, TensorFlow C++).

        - C-events:
          Category: CATEGORY_TF_API/CATEGORY_SIMULATOR_CPP, Event(name=<CLIB__*>)
          Time spent in the simulator / TensorFlow C++ code during an API call.
          NOTE: new categories could be created in the future for other "types" of C-libraries.
          For that reason, we recommend counting C-events by subtracting all other categories of events
          (i.e. at the time of writing this: CATEGORY_OPERATION, CATEGORY_PYTHON).

        def total_op_events(self):
            # PSEUDOCODE:
            If we want to count the number of iml.prof.operation calls, then we wish to count the number of "Op-events":
                Number of events where:
                    event.category == CATEGORY_OPERATION and
                    not is_op_process_event(event_name, category)

        def total_intercepted_calls(self):
            # PSEUDOCODE:
            If we want to count the number of intercepted Python->C++ calls, then we wish to count the number of "C-events":
                Number of events where:
                    event.category not in {CATEGORY_OPERATION, CATEGORY_PYTHON}

        :return:
        """
        # df = self.read()
        # is_op_event = np.vectorize(PyprofDataframeReader.is_op_event, otypes=[np.bool])(df['name'], df['category'])
        # op_df = df[is_op_event]
        # total_op_events = len(op_df)
        # return total_op_events

        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMerge_PyprofDataframeReader__total_op_events(obj=self)
        total_op_events = self.split_map_merge(
            'total_op_events', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return total_op_events

    def total_op_process_events(self):
        # df = self.read()
        # is_op_proc_event = np.vectorize(PyprofDataframeReader.is_op_process_event, otypes=[np.bool])(df['name'], df['category'])
        # op_proc_df = df[is_op_proc_event]
        # total_op_proc_events = len(op_proc_df)
        # return total_op_proc_events

        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMerge_PyprofDataframeReader__total_op_process_events(obj=self)
        total_op_proc_events = self.split_map_merge(
            'total_op_process_events', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return total_op_proc_events


    # def total_interception_overhead_us(self):
    #     """
    #     Overhead from Python->C++ interception.
    #
    #     :return:
    #     """
    #     total_intercepted_calls = self.total_intercepted_calls()
    #     total_pyprof_interception_overhead_us = total_intercepted_calls *
    #     # df = self.read()
    #     # # Filter for events that have profiling overhead recorded.
    #     # df = df[df['start_profiling_overhead_us'] != 0]
    #     # total_pyprof_overhead_us = np.sum(df['duration_profiling_overhead_us'])
    #     # return total_pyprof_overhead_us

    def len_df(self):
        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMerge_PyprofDataframeReader__len_df(obj=self)
        len_df = self.split_map_merge(
            'len(df)', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return len_df

    def check_events(self):
        # df = self.read()

        len_df = self.len_df()

        total_intercepted_calls = self.total_intercepted_calls()
        total_op_events = self.total_op_events()
        total_op_proc_events = self.total_op_process_events()

        # Total category_events recorded:
        # - Each CATEGORY_PYTHON event should have a corresponding C++ event.
        # - Then we have op-events
        # - Plus, we may have some "extra" events like the
        #   process event Event(category=CATEGORY_OPERATION, name="[<PROC>ppo2_HalfCheetah]")
        assert 2*total_intercepted_calls + total_op_events + total_op_proc_events == len_df

    def total_pyprof_overhead_us(self):
        # df = self.read()
        # # Filter for events that have profiling overhead recorded.
        # df = df[df['start_profiling_overhead_us'] != 0]
        # total_pyprof_overhead_us = np.sum(df['duration_profiling_overhead_us'])
        # return total_pyprof_overhead_us

        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMerge_PyprofDataframeReader__total_pyprof_overhead_us(obj=self)
        total_pyprof_overhead_us = self.split_map_merge(
            'total_pyprof_overhead_us', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return total_pyprof_overhead_us

    def total_pyprof_overhead_df(self):
        # df = copy.copy(self.read())
        # if len(df) == 0:
        #     zero_df = self.zero_dataframe(['total_pyprof_overhead_us'])
        #     return zero_df
        #
        # # Filter for events that have profiling overhead recorded.
        # # df = df[df['start_profiling_overhead_us'] != 0]
        # groupby_cols = self.iml_columns
        # agg_cols = ['duration_profiling_overhead_us']
        # keep_cols = sorted(set(groupby_cols + agg_cols))
        # df_keep = df[keep_cols]
        # groupby = df_keep.groupby(groupby_cols)
        # df = groupby.sum().reset_index()
        # df['total_pyprof_overhead_us'] = df['duration_profiling_overhead_us']
        # del df['duration_profiling_overhead_us']
        # return df

        # df = copy.copy(self.read())
        # if len(df) == 0:
        #     zero_df = self.zero_dataframe(['total_pyprof_overhead_us'])
        #     return zero_df

        # Filter for events that have profiling overhead recorded.
        # df = df[df['start_profiling_overhead_us'] != 0]

        if len(self.proto_paths) == 0:
            zero_df = self.zero_dataframe(['total_pyprof_overhead_us'])
            return zero_df
        split_map_merge = SplitMapMerge_PyprofDataframeReader__total_pyprof_overhead_df(obj=self)
        df = self.split_map_merge(
            'total_pyprof_overhead_df', split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)

        if len(df) == 0:
            zero_df = self.zero_dataframe(['total_pyprof_overhead_us'])
            return zero_df

        df['total_pyprof_overhead_us'] = df['duration_profiling_overhead_us']
        del df['duration_profiling_overhead_us']
        return df

class DataframeMapper:
    def __init__(self, DataframeReaderKlass, directories, debug=False, debug_single_thread=False):
        self.DataframeReaderKlass = DataframeReaderKlass
        if type(directories) == str:
            dirs = [directories]
        else:
            dirs = list(directories)
        self.directories = dirs
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self._init_readers()

    def _init_readers(self):
        self.readers = []
        for directory in self.directories:
            df_reader = self.DataframeReaderKlass(directory, debug=self.debug)
            self.readers.append(df_reader)

    def map(self, func):
        return [func(reader) for reader in self.readers]

    def map_one(self, func):
        assert len(self.readers) == 1
        reader = self.readers[0]
        return func(reader)

def get_iml_config_path(directory):
    """
    Add (algo, env) from iml_config.json, if they were set by the training script using iml.prof.set_metadata(...).

    :return:
    """
    iml_config_paths = [
        path for path in each_file_recursive(directory)
        if is_iml_config_file(path) and _b(_d(path)) != DEFAULT_PHASE]
    # There should be exactly one iml_config.json file.
    # Q: Couldn't there be multiple for multi-process scripts like minigo?
    if len(iml_config_paths) != 1:
        logging.info("Expected 1 iml_config.json but saw {len} within iml_directory={dir}: {msg}".format(
            dir=directory,
            len=len(iml_config_paths),
            msg=pprint_msg(iml_config_paths)))
        assert len(iml_config_paths) == 1
    iml_config_path = iml_config_paths[0]
    return iml_config_path

def read_iml_config(directory):
    iml_config_path = get_iml_config_path(directory)
    iml_config = load_json(iml_config_path)
    return iml_config

class IMLConfig:
    def __init__(self, directory=None, iml_config_path=None):
        assert directory is not None or iml_config_path is not None
        self.directory = directory
        if iml_config_path is not None:
            self.iml_config_path = iml_config_path
        else:
            self.iml_config_path = get_iml_config_path(directory)
        self.iml_config = load_json(self.iml_config_path)
        self.init_iml_prof_args()

    def _get_metadata(self, var, allow_none=False):
        if 'metadata' in self.iml_config and var in self.iml_config['metadata']:
            return self.iml_config['metadata'][var]
        assert allow_none
        return None

    def algo(self, allow_none=False):
        return self._get_metadata('algo', allow_none=allow_none)

    def env(self, allow_none=False):
        return self._get_metadata('env', allow_none=allow_none)

    def init_iml_prof_args(self):
        self.iml_prof_args = dict()
        if 'env' in self.iml_config:
            for var, value in self.iml_config['env'].items():
                if not is_iml_prof_env(var):
                    continue
                self.iml_prof_args[iml_prof_varname(var)] = iml_prof_value(var, value)

    def get_env_bool(self, var, dflt=False):
        return self.iml_prof_args.get(var, dflt)

    def get_env_var(self, var, dflt=None):
        return self.iml_prof_args.get(var, dflt)

    def must_get_env(self, var):
        return self.iml_prof_args.get[var]

    def get_bool(self, var, dflt=False):
        return self.iml_config.get(var, dflt)

    def get_var(self, var, dflt=None):
        return self.iml_config.get(var, dflt)

    def must_get(self, var):
        return self.iml_config[var]

def is_iml_prof_env(var):
    return re.search(r'^IML_', var)

def iml_prof_varname(env_var):
    var = env_var
    var = re.sub(r'^IML_', '', var)
    var = var.lower()
    return var

def iml_prof_value(var, value):
    if value == 'yes':
        return True
    if value == 'no':
        return False

    try:
        num = int(value)
        return num
    except ValueError:
        pass

    try:
        num = float(value)
        return num
    except ValueError:
        pass

    return value

class LinearRegressionSampleReader:
    def __init__(self, directory, debug=False, debug_single_thread=False):
        self.directory = directory
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.cuda_reader = CUDAAPIStatsDataframeReader(self.directory, debug=self.debug)
        self.pyprof_reader = PyprofDataframeReader(self.directory, debug=self.debug)

    def feature_names(self):
        pass

    def feature_df(self):
        pass

class LinearRegressionReader:
    """
    Read a "feature matrix" X to be used in training a linear regression model
    for predicting the total overhead y.

    The features of the matrix are all of the attributes we manually use for
    computing average overheads when perform ad-hoc profiling overhead subtraction.

    CUDAAPIStatsDataframeReader
    CUPTI profiling: CUDA API calls:
    - # of cudaMemcpy calls
    - # of cudaLaunchKernel calls

    PyprofDataframeReader
    Python profiling:
    - # of intercepted Python->C++ calls.
    - # of Python "operation" annotations

    Read matrix like:

    FEATURES --->
    SAMPLES
    |         [ <# of cudaMemcpy calls> <# of cudaLaunchKernel calls> <# of intercepted Python->C++ calls> ... ]
    |         [  ...                                                                                           ]
    |         [  ...                                                                                           ]
    V         [  ...                                                                                           ]

    We could make the "samples" by the stats aggregated over a single training-loop iteration,
    or multiple training-loop iteration.

    Our labels will be the "observed" overhead" of the application, which we measure be running an uninstrumented run,
    and compute the actual "extra time" compared to an instrumented run.

    The easiest thing to do right now in our setup is to just run multiple iterations.
    """
    def __init__(self, directory, debug=False, debug_single_thread=False):
        self.directory = directory
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.cuda_reader = CUDAAPIStatsDataframeReader(self.directory, debug=self.debug)
        self.pyprof_reader = PyprofDataframeReader(self.directory, debug=self.debug)

    def feature_names(self):
        pass

    def feature_df(self):
        pass

