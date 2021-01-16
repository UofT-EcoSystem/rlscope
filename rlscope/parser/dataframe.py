"""
Read RL-Scope data into pandas dataframes.

This modules defines various ``*Reader`` classes that can read RL-Scope trace files into a single dataframe.
These readers are used by plotting scripts.
"""
import pandas as pd
import decimal
import copy
import os
import pprint
import progressbar
from rlscope.profiler.rlscope_logging import logger
import functools
import multiprocessing

from concurrent.futures import ProcessPoolExecutor

from rlscope.profiler import concurrent

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from rlscope.profiler.util import pprint_msg
from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.protobuf import rlscope_prof_pb2

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
    Base functionality shared by readers, including parallelizing reading of trace files,
    and walking the directory tree to find matching trace files.
    """
    def __init__(self, directory, add_fields=None, colnames=None, debug=False, debug_single_thread=False):
        self.directory = directory
        # column-name -> [values]
        # self.data = None
        self.df = None
        # add_fields(rlscope_directory) -> dict of fields to add to data-frame
        self.add_fields = add_fields
        if colnames is not None:
            colnames = set(colnames)
        self._colnames = colnames
        self.added_fields = set()
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        # self.rlscope_config = read_rlscope_config(self.directory, allow_union=True)
        self.rlscope_metadata = read_rlscope_config_metadata(self.directory)
        self.rlscope_columns = self._get_rlscope_columns()
        self.colnames = self._get_colnames()

    def split_map_merge(self, name, map_fn, merge_fn,
                        n_workers=None,
                        debug=False,
                        debug_single_thread=False):
        """
        Maps map_fn across each trace file.
        map_fn should produce a dataframe.
        merge_fn is run pair-wise on all the results using functools.reduce().

        Arguments
        ---------
        name: str
            For debugging output.
        n_workers: int
            Parallelism of "map" stage.
        map_fn: func
            Given a proto_path, run a function on it that produces (most likely) a dataframe, or
            some computation that reduces a dataframe.
        merge_fn: func
            Given two inputs A and B, where A and/or B are one of:
            - a result from map_fn
            - a result from merge_fn
            Return the result of "merging" A and B.
        debug_single_thread: bool
            Allow debugging with pdb.
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
        #     logger.info("split_map_merge.{name}: {msg}".format(
        #         name=name,
        #         msg=pprint_msg({
        #             'merged_result': merged_result,
        #         })
        #     ))
        return merged_result

    def merge_from_map(self, map_fn, a, b):
        # if self.debug:
        #     logger.info("merge_from_map: {msg}".format(
        #         msg=pprint_msg({
        #             'a': a,
        #             'b': b,
        #         })
        #     ))
        df = pd.concat([a, b])
        result = map_fn(df)
        # if self.debug:
        #     logger.info("merge_from_map: {msg}".format(
        #         msg=pprint_msg({
        #             'result': result,
        #         })
        #     ))
        return result

    def _get_colnames(self):
        return self._colnames.union(set(self.rlscope_columns))

    def _add_rlscope_config_columns(self, data=None, skip_fields=None):
        assert data is not None
        def _get_meta(col):
            if col not in self.rlscope_metadata['metadata'] or \
                (skip_fields is not None and col in skip_fields):
                return
            self._add_col(col, self.rlscope_metadata['metadata'][col], data=data)
        # Q: should we just set ALL the metadata?
        _get_meta('algo')
        _get_meta('env')
        if 'env' in self.rlscope_metadata and 'CUDA_VISIBLE_DEVICES' in self.rlscope_metadata['env']:
            devs = self.rlscope_metadata['env']['CUDA_VISIBLE_DEVICES']
            if type(devs) == int:
                dev_ids = frozenset([devs])
            elif type(devs) == str:
                dev_ids = frozenset([int(d) for d in re.split(r',\s*', devs)])
            else:
                raise NotImplementedError(
                    ("Not sure how to parse "
                     "rlscope_config['env']['CUDA_VISIBLE_DEVICES'] = {obj}, "
                     "type={type}").format(
                        obj=devs,
                        type=type(devs)))
            self._add_col('CUDA_VISIBLE_DEVICES', dev_ids, data=data)

    def _has_rlscope_columns(self):
        return 'algo' in self.rlscope_metadata['metadata'] and 'env' in self.rlscope_metadata['metadata']

    def _get_rlscope_columns(self):
        if not self._has_rlscope_columns():
            return []
        return ['algo', 'env']

    def _add_col(self, colname, value, data=None):
        assert data is not None
        # if data is None:
        #     data = self.data

        # if self.colnames is not None:
        #     assert colname in self.colnames

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
        self._add_rlscope_config_columns(data=data)
        df = pd.DataFrame(data)
        return df

    def add_proto_cols(self, path, data=None):
        raise NotImplementedError()

    def _maybe_add_fields(self, path, data=None):
        skip_fields = set()
        if self.add_fields is not None:
            extra_fields = self.add_fields(path)
            if extra_fields is not None:
                self.added_fields.update(extra_fields.keys())
                for key, value in extra_fields.items():
                    self._add_col_to_data(key, value, data=data)
            # self._check_cols(data=data)
            # Avoid adding same columns twice.
            skip_fields.update(extra_fields.keys())

        self._add_rlscope_config_columns(data=data, skip_fields=skip_fields)
        self._check_cols(data=data)

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
            if self.debug:
                logger.info("Read proto_path={path}".format(path=proto_path))
            self.add_proto_cols(proto_path, data=data)
            self._check_cols(data=data)
        if len(proto_paths) == 0:
            logger.warning("{klass}: Saw 0 proto paths rooted at {dir}; returning empty dataframe".format(
                klass=self.__class__.__name__,
                dir=self.directory,
            ))
            self.df = self.empty_dataframe()
        else:
            self.df = pd.DataFrame(data)

        self.df = self.add_to_dataframe(self.df)

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
            logger.warning("{klass}: Saw 0 proto paths rooted at {dir}; returning empty dataframe".format(
                klass=self.__class__.__name__,
                dir=self.directory,
            ))
            df = self.empty_dataframe()
            yield df

    def read(self):
        if self.df is None:
            self._read_df()
        return self.df

    def add_to_dataframe(self, df):
        return df


class OverlapDataframeReader(BaseDataframeReader):
    """
    Read *.venn_js.json CPU/GPU time breakdown files output by ``rls-analyze``.
    """

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            'machine_name',
            'phase_name',
            'process_name',
            'start_time_usec',
            'end_time_usec',
            'overlap_type',
            'overlap_label',
            # 'overlap_usec',
            'CUDA_VISIBLE_DEVICES',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_venn_js_file(path)

    def _label_string(self, tpl):
        # if len(tpl) == 1:
        #     return tpl[0]
        return ', '.join([str(x) for x in tpl])

    def add_proto_cols(self, path, data=None):
        if self.debug:
            logger.info("Read OverlapDataframeReader from {path}".format(path=path))

        venn_js = VennData(path)

        def add_key(col_key, venn_js_key=None):
            if venn_js_key is None:
                venn_js_key = col_key
            self._add_col(col_key, venn_js.md[venn_js_key], data=data)


        # WARNING: this returns venn_sizes which we probably DON'T want to work with...
        overlap_dict = venn_js.as_dict()
        for overlap_label, overlap_usec in overlap_dict.items():
            add_key('machine_name', 'machine')
            add_key('process_name', 'process')
            add_key('phase_name', 'phase')

            add_key('start_time_usec')
            add_key('end_time_usec')
            add_key('overlap_type')

            label_str = self._label_string(overlap_label)
            self._add_col('overlap_label', label_str, data=data)
            # self._add_col('overlap_usec', overlap_usec, data=data)

            self._maybe_add_fields(path, data=data)

class CUDADeviceEventsReader(BaseDataframeReader):
    """
    .. deprecated:: 1.0.0
       Old TensorFlow-specific profiling code.
    """

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            'machine_name',
            'process_name',
            'phase_name',
            'device_name',
            'event_name',
            'cuda_event_type',
            'start_time_us',
            'duration_us',
            'CUDA_VISIBLE_DEVICES',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_cuda_device_events_file(path)

    # # override
    # def add_to_dataframe(self, df):
    #     # new_df = pd.concat(dfs)
    #     df['device_type'] = df['device_name'].apply(self.cpu_or_gpu)
    #     if 'CUDA_VISIBLE_DEVICES' in df:
    #         df['used_by_tensorflow'] = np.vectorize(self.used_by_tensorflow, otypes=[np.bool])(
    #             df['CUDA_VISIBLE_DEVICES'],
    #             df['device_id'],
    #             df['device_type'])
    #     return df

    def get_event_name(self, event):
        if event.name != "" and event.name is not None:
            return event.name

    def get_event_type(self, event):
        if event.cuda_event_type == rlscope_prof_pb2.UNKNOWN:
            return "UNKNOWN"
        elif event.cuda_event_type == rlscope_prof_pb2.KERNEL:
            return "KERNEL"
        elif event.cuda_event_type == rlscope_prof_pb2.MEMCPY:
            return "MEMCPY"
        elif event.cuda_event_type == rlscope_prof_pb2.MEMSET:
            return "MEMSET"
        else:
            raise NotImplementedError("Not sure what Event.name to use for event.cuda_event_type == {code}".format(
                code=event.cuda_event_type,
            ))

    def add_proto_cols(self, path, data=None):
        if self.debug:
            logger.info("Read CUDADeviceEventsReader from {path}".format(path=path))

        proto = read_cuda_device_events_file(path)
        category = constants.CATEGORY_GPU
        for device_name, dev_events_proto in proto.dev_events.items():
            for event in dev_events_proto.events:
                event_name = self.get_event_name(event)
                event_type = self.get_event_type(event)

                # string name = 1;
                # CudaEventType cuda_event_type = 2;
                # int64 start_time_us = 3;
                # int64 duration_us = 4;

                self._add_col('machine_name', proto.machine_name, data=data)
                self._add_col('process_name', proto.process_name, data=data)
                self._add_col('phase_name', proto.phase, data=data)
                self._add_col('device_name', device_name, data=data)
                self._add_col('event_name', event_name, data=data)
                self._add_col('cuda_event_type', event_type, data=data)
                self._add_col('start_time_us', event.start_time_us, data=data)
                self._add_col('duration_us', event.duration_us, data=data)
                # self._check_cols(data=data)

                self._maybe_add_fields(path, data=data)
                self._check_cols(data=data)


class UtilDataframeReader(BaseDataframeReader):
    """
    Read ``nvidia-smi`` GPU utilization into dataframe.
    """

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            # MachineUtilization from pyprof.proto
            'machine_name',
            # DeviceUtilization from pyprof.proto
            'device_name',
            'device_id',
            # UtilizationSample from pyprof.proto
            'util',
            'start_time_us',
            'total_resident_memory_bytes',
            'CUDA_VISIBLE_DEVICES',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_machine_util_file(path)

    @staticmethod
    def is_cpu(device_name):
        if re.search(r'\b(Intel|Xeon|CPU|AMD)\b', device_name):
            return True
        return False

    @staticmethod
    def cpu_or_gpu(device_name):
        if UtilDataframeReader.is_cpu(device_name):
            return 'CPU'
        return 'GPU'

    @staticmethod
    def used_by_tensorflow(CUDA_VISIBLE_DEVICES, device_id, device_type):
        if device_type == 'CPU':
            return True
        if device_type == 'GPU':
            return device_id in CUDA_VISIBLE_DEVICES
        # Not handled.
        raise NotImplementedError()

    # override
    def add_to_dataframe(self, df):
        # new_df = pd.concat(dfs)
        df['device_type'] = df['device_name'].apply(self.cpu_or_gpu)
        if 'CUDA_VISIBLE_DEVICES' in df:
            df['used_by_tensorflow'] = np.vectorize(self.used_by_tensorflow, otypes=[np.bool])(
                df['CUDA_VISIBLE_DEVICES'],
                df['device_id'],
                df['device_type'])
        return df

    def add_proto_cols(self, path, data=None):
        machine_util = read_machine_util_file(path)
        if self.debug:
            logger.info("Read MachineUtilization from {path}".format(path=path))
        for device_name, device_utilization in machine_util.device_util.items():
            last_start_time_us = None
            device_id = 0
            # TODO: once we verify this works, limit output devices to CUDA_VISIBLE_DEVICES
            for sample in device_utilization.samples:
                # HACK: rlscope-sample-util will record 1 sample at the same timestamp for each GPU, regardless
                # of CUDA_VISIBLE_DEVICES.
                # Often a machine has multiple GPUs with the same device_name.
                # To ensure we don't incorrectly group multiple samples into the same device,
                # add a device_id suffix to the device_name.
                # We can dissambiguate the samples since we assume:
                # - Devices are samples in the same order each time
                # - Consecutive device samples appear next to each other in the proto file
                if last_start_time_us is not None:
                    if last_start_time_us == sample.start_time_us:
                        # Sample taken at same time as last sample.
                        # This sample is for another GPU.
                        # Increment device_id.
                        device_id += 1
                    else:
                        # New sample (not taken at same time).
                        # Reset device_id.
                        device_id = 0
                self._add_col('machine_name', machine_util.machine_name, data=data)
                dev = "{name}.{id:02}".format(name=device_name, id=device_id)
                self._add_col('device_name', dev, data=data)
                self._add_col('device_id', device_id, data=data)

                self._add_col('util', sample.util, data=data)
                self._add_col('start_time_us', sample.start_time_us, data=data)
                self._add_col('total_resident_memory_bytes', sample.total_resident_memory_bytes, data=data)
                # self._check_cols(data=data)

                self._maybe_add_fields(path, data=data)
                self._check_cols(data=data)
                last_start_time_us = sample.start_time_us

class TrainingProgressDataframeReader(BaseDataframeReader):
    """
    Read training progress (training_progress*.proto) into dataframe.
    """

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False, add_algo_env=False):

        colnames = [
            # IncrementalTrainingProgress from pyprof.proto
            # 'rlscope_directory',
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
            # 'CUDA_VISIBLE_DEVICES',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)
        self.add_algo_env = add_algo_env

    def is_proto_file(self, path):
        return is_training_progress_file(path)

    def add_proto_cols(self, path, data=None):
        training_progress = read_training_progress_file(path)
        self._add_columns(self._colnames, training_progress, data=data)
        self._add_col_to_data('rlscope_directory', self.directory, data=data)
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

    def percent_complete(self):
        df = self.last_progress()
        percent_complete = (df['end_num_timesteps'] - df['start_num_timesteps'])/df['total_timesteps']
        # from_percent = df['end_percent_complete'] - df['start_percent_complete']
        # assert percent_complete == from_percent
        assert len(percent_complete) == 1
        return percent_complete.values[0]

    def extrap_total_training_time_us(self):
        training_duration_us = self.training_duration_us()
        percent_complete = self.percent_complete()
        extrap_total_training_time_us = extrap_total_training_time(training_duration_us, percent_complete)
        return extrap_total_training_time_us

    def training_iterations(self):
        df = self.last_progress()
        training_iterations = df['end_num_timesteps'] - df['start_num_timesteps']
        assert len(training_iterations) == 1
        training_iterations = training_iterations.values[0]
        return training_iterations

    def total_timesteps(self):
        df = self.last_progress()
        total_timesteps = df['total_timesteps']
        assert len(total_timesteps) == 1
        total_timesteps = total_timesteps.values[0]
        return total_timesteps

    def end_num_timesteps(self):
        df = self.last_progress()
        end_num_timesteps = df['end_num_timesteps']
        assert len(end_num_timesteps) == 1
        end_num_timesteps = end_num_timesteps.values[0]
        return end_num_timesteps

    def training_duration_df(self):
        df = copy.copy(self.last_progress())
        df['training_duration_us'] = df['end_training_time_us'] - df['start_training_time_us']
        df['extrap_total_training_time'] = self.extrap_total_training_time_us()
        keep_cols = [
            'training_duration_us',
            'extrap_total_training_time',
            'algo',
            'env',
            'rlscope_directory',
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
        # We already DID add new event categories: constants.CATEGORY_PROF_...
        # All we can really do keep track of a set of "constants.CATEGORIES_C_EVENTS" that represent "Python -> C" interceptions.
        # However, if we ever add 'custom categories' in the future, that approach will break.
        # We should check for that somehow...
        # return category not in {constants.CATEGORY_OPERATION, constants.CATEGORY_PYTHON}
        return category in constants.CATEGORIES_C_EVENTS

class SplitMapMerge_PyprofDataframeReader__total_intercepted_tensorflow_calls:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        was_intercepted_call_event = df['category'].apply(self.is_intercepted_tensorflow_call_event)
        intercepted_call_event_df = df[was_intercepted_call_event]
        total_intercepted_tensorflow_calls = len(intercepted_call_event_df)
        return total_intercepted_tensorflow_calls

    def merge_fn(self, a, b):
        return a + b

    def is_intercepted_tensorflow_call_event(self, category):
        return category == constants.CATEGORY_TF_API

class SplitMapMerge_PyprofDataframeReader__total_intercepted_simulator_calls:
    def __init__(self, obj):
        self.obj = obj

    def map_fn(self, df):
        was_intercepted_call_event = df['category'].apply(self.is_intercepted_simulator_call_event)
        intercepted_call_event_df = df[was_intercepted_call_event]
        total_intercepted_simulator_calls = len(intercepted_call_event_df)
        return total_intercepted_simulator_calls

    def merge_fn(self, a, b):
        return a + b

    def is_intercepted_simulator_call_event(self, category):
        return category == constants.CATEGORY_SIMULATOR_CPP

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
        groupby_cols = self.obj.rlscope_columns
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
    """
    Read CUDA API call count and duration (cuda_api_stats*.proto) into dataframe.
    Used for measuring CUPTI profiling overhead.
    """

    def __init__(self, directory, add_fields=None, debug=False, debug_single_thread=False):

        colnames = [
            # CUDAAPIPhaseStatsProto from rlscope_prof.proto
            'process_name',
            'machine_name',
            'phase_name',

            # CUDAAPIThreadStatsProto from rlscope_prof.proto
            'tid',
            'api_name',
            'total_time_us',
            'num_calls',
            'CUDA_VISIBLE_DEVICES',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_cuda_api_stats_file(path)

    def add_proto_cols(self, path, data=None):
        proto = read_cuda_api_stats_file(path)
        # if self.debug:
        #     logger.info("Read CUDAAPIPhaseStatsProto from {path}".format(path=path))

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
        groupby_cols = ['api_name'] + self.rlscope_columns
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
        # groupby_cols = ['api_name'] + self.rlscope_columns
        # agg_cols = ['total_time_us', 'num_calls']
        # keep_cols = sorted(set(groupby_cols + agg_cols))

        groupby_cols = self._groupby_cols()
        keep_cols = self._keep_cols()
        df_sum = self._df_group_sum(df, groupby_cols, keep_cols)

        # if self.debug:
        #     logger.info("_compute_per_api_stats: {msg}".format(
        #         msg=pprint_msg({
        #             'df_sum': df_sum,
        #         })
        #     ))

        return df_sum

class PyprofDataframeReader(BaseDataframeReader):
    """
    Read start/end timestamps for various event types (e.g., user operation annotations)
    from category_events*.proto into dataframe.
    """

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
            'CUDA_VISIBLE_DEVICES',
        ]

        super().__init__(directory, add_fields=add_fields, colnames=colnames, debug=debug, debug_single_thread=debug_single_thread)

    def is_proto_file(self, path):
        return is_pyprof_file(path)

    def add_proto_cols(self, path, data=None):
        proto = read_pyprof_file(path)
        # if self.debug:
        #     logger.info("Read CategoryEventsProto from {path}".format(path=path))

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
        #     logger.info("{klass}.add_proto_cols path={path}".format(
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
        return self._total_intercepted_calls(
            'total_intercepted_calls',
            SplitMapMerge_PyprofDataframeReader__total_intercepted_calls)

    def total_intercepted_tensorflow_calls(self):
        return self._total_intercepted_calls(
            'total_intercepted_tensorflow_calls',
            SplitMapMerge_PyprofDataframeReader__total_intercepted_tensorflow_calls)

    def total_intercepted_simulator_calls(self):
        return self._total_intercepted_calls(
            'total_intercepted_simulator_calls',
            SplitMapMerge_PyprofDataframeReader__total_intercepted_simulator_calls)

    def _total_intercepted_calls(self, name, SplitMapMergeKlass):
        """
        How many times did we perform interception:
        constants.CATEGORY_PYTHON: Python -> C++
        <clib category>: C++ -> Python

        where <clib category> could be:
        - constants.CATEGORY_SIMULATOR_CPP
        - constants.CATEGORY_TF_API

        Everytime we have a C++ call, we record two events:

        :return:
        """
        if len(self.proto_paths) == 0:
            return 0
        split_map_merge = SplitMapMergeKlass(obj=self)
        total_intercepted_calls = self.split_map_merge(
            name, split_map_merge.map_fn, split_map_merge.merge_fn,
            debug=self.debug,
            debug_single_thread=self.debug_single_thread)
        return total_intercepted_calls

    @staticmethod
    def is_op_event(event_name, category):
        return not is_op_process_event(event_name, category) and category == constants.CATEGORY_OPERATION

    def total_annotations(self):
        return self.total_op_events()

    def total_op_events(self):
        """
        How many times did we record op-events using "with rlscope.prof.operation(...)"?

        Every time we execute this annotation, we record the following pyprof events:

        Here are some of the events that get recorded during pyprof.
        - Op-events:
          Category: constants.CATEGORY_OPERATION, Event(name='sample_action')
          Operation annotation for the duration of the operation.

        - Process-events:
          Category: constants.CATEGORY_OPERATION, Event(name='[process_name]')
          Single operation annotation recorded once at termination that measures
          the total duration of the process.

        - Remaining python time at end of operation:
          Category: constants.CATEGORY_PYTHON, Event(name='Finish python benchmark')
          The amount of time in between returning from the TensorFlow C++ API,
          and finishing an operation annotation.

        - Python-events:
          Category: constants.CATEGORY_PYTHON, Event(name=<CLIB__*>)
          Time spent in python before making a call into a wrapped C library (i.e. simulator, TensorFlow C++).

        - C-events:
          Category: constants.CATEGORY_TF_API/constants.CATEGORY_SIMULATOR_CPP, Event(name=<CLIB__*>)
          Time spent in the simulator / TensorFlow C++ code during an API call.
          NOTE: new categories could be created in the future for other "types" of C-libraries.
          For that reason, we recommend counting C-events by subtracting all other categories of events
          (i.e. at the time of writing this: constants.CATEGORY_OPERATION, constants.CATEGORY_PYTHON).

        def total_op_events(self):
            # PSEUDOCODE:
            If we want to count the number of rlscope.prof.operation calls, then we wish to count the number of "Op-events":
                Number of events where:
                    event.category == constants.CATEGORY_OPERATION and
                    not is_op_process_event(event_name, category)

        def total_intercepted_calls(self):
            # PSEUDOCODE:
            If we want to count the number of intercepted Python->C++ calls, then we wish to count the number of "C-events":
                Number of events where:
                    event.category not in {constants.CATEGORY_OPERATION, constants.CATEGORY_PYTHON}

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
        """
        Total category_events recorded:
        - Each constants.CATEGORY_PYTHON event should have a corresponding C++ event.
        - Then we have op-events
        - Plus, we may have some "extra" events like the
          process event Event(category=constants.CATEGORY_OPERATION, name="[<PROC>ppo2_HalfCheetah]")
        """
        len_df = self.len_df()

        total_intercepted_tensorflow_calls = self.total_intercepted_tensorflow_calls()
        total_intercepted_simulator_calls = self.total_intercepted_simulator_calls()
        total_intercepted_calls = total_intercepted_tensorflow_calls + total_intercepted_simulator_calls
        total_op_events = self.total_op_events()
        total_op_proc_events = self.total_op_process_events()

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
        # groupby_cols = self.rlscope_columns
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

def get_rlscope_config_path(directory, allow_many=False, allow_none=False):
    """
    Add (algo, env) from rlscope_config.json, if they were set by the training script using rlscope.prof.set_metadata(...).

    :return:
    """
    rlscope_config_paths = [
        path for path in each_file_recursive(directory)
        if is_rlscope_config_file(path) and _b(_d(path)) != constants.DEFAULT_PHASE]
    # There should be exactly one rlscope_config.json file.
    # Q: Couldn't there be multiple for multi-process scripts like minigo?
    if len(rlscope_config_paths) != 1 and not allow_many:
        raise NoRLScopeConfigFound("Expected 1 rlscope_config.json but saw {len} within rlscope_directory={dir}: {msg}".format(
            dir=directory,
            len=len(rlscope_config_paths),
            msg=pprint_msg(rlscope_config_paths)))

    if len(rlscope_config_paths) == 0 and allow_none:
        if allow_none:
            return None
        else:
            raise NoRLScopeConfigFound("Didn't find any rlscope_config.json in rlscope_directory={dir}".format(
                dir=directory,
            ))

    if allow_many:
        return rlscope_config_paths

    rlscope_config_path = rlscope_config_paths[0]
    return rlscope_config_path

def read_rlscope_config(directory):
    rlscope_config_path = get_rlscope_config_path(directory)
    rlscope_config = load_json(rlscope_config_path)
    return rlscope_config

def read_rlscope_config_metadata(directory):
    rlscope_metadata = {
        'metadata': dict()
    }
    rlscope_config_paths = get_rlscope_config_path(directory, allow_many=True)
    for rlscope_config_path in rlscope_config_paths:
        rlscope_config = load_json(rlscope_config_path)
        if 'metadata' in rlscope_config:
            rlscope_metadata['metadata'].update(rlscope_config['metadata'])

        if 'env' in rlscope_config and 'CUDA_VISIBLE_DEVICES' in rlscope_config['env']:
            if 'env' not in rlscope_metadata:
                rlscope_metadata['env'] = dict()

            if 'CUDA_VISIBLE_DEVICES' in rlscope_metadata['env']:
                assert rlscope_metadata['env']['CUDA_VISIBLE_DEVICES'] == rlscope_config['env']['CUDA_VISIBLE_DEVICES']
            else:
                rlscope_metadata['env']['CUDA_VISIBLE_DEVICES'] = rlscope_config['env']['CUDA_VISIBLE_DEVICES']

    return rlscope_metadata

class NoRLScopeConfigFound(Exception):
    """
    Raised if we cannot locate a rlscope_config.json file

    See also
    ---------
    RLScopeConfig : reading RL-Scope configuration files.
    """
    pass

class RLScopeConfig:
    """
    Read rlscope_config.json file, which contains RL-Scope profiler configuration information that the training script
    was run with (e.g., `--rlscope-*` command-line options, algo, env)
    """
    def __init__(self, directory=None, rlscope_config_path=None):
        assert directory is not None or rlscope_config_path is not None
        self.directory = directory
        if rlscope_config_path is not None:
            self.rlscope_config_path = rlscope_config_path
        else:
            self.rlscope_config_path = get_rlscope_config_path(directory)
        self.rlscope_config = load_json(self.rlscope_config_path)
        self.init_rlscope_prof_args()

    def _get_metadata(self, var, allow_none=False):
        if 'metadata' in self.rlscope_config and var in self.rlscope_config['metadata']:
            return self.rlscope_config['metadata'][var]
        assert allow_none
        return None

    def algo(self, allow_none=False):
        return self._get_metadata('algo', allow_none=allow_none)

    def env(self, allow_none=False):
        return self._get_metadata('env', allow_none=allow_none)

    def init_rlscope_prof_args(self):
        self.rlscope_prof_args = dict()
        if 'env' in self.rlscope_config:
            for var, value in self.rlscope_config['env'].items():
                if not is_rlscope_prof_env(var):
                    continue
                self.rlscope_prof_args[rlscope_prof_varname(var)] = rlscope_prof_value(var, value)

    def get_env_bool(self, var, dflt=False):
        return self.rlscope_prof_args.get(var, dflt)

    def get_env_var(self, var, dflt=None):
        return self.rlscope_prof_args.get(var, dflt)

    def must_get_env(self, var):
        return self.rlscope_prof_args.get[var]

    def get_bool(self, var, dflt=False):
        return self.rlscope_config.get(var, dflt)

    def get_int(self, var, dflt=None):
        return int(self.rlscope_config.get(var, dflt))

    def get_var(self, var, dflt=None):
        return self.rlscope_config.get(var, dflt)

    def must_get(self, var):
        return self.rlscope_config[var]

def is_rlscope_prof_env(var):
    return re.search(r'^RLSCOPE_', var)

def rlscope_prof_varname(env_var):
    var = env_var
    var = re.sub(r'^RLSCOPE_', '', var)
    var = var.lower()
    return var

def rlscope_prof_value(var, value):
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


class VennData:
    """
    Read *.venn_js.json files.

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
        self.data = None
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
                        subtract_sec = (pyprof_overhead_json['mean_pyprof_annotation_per_call_us']/constants.USEC_IN_SEC) *
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
        if self._is_vennbug_version():
            return self._old_vennbug_total_size()
        return self._new_total_size()

    def _is_vennbug_version(self):
        return 'version' not in self._metadata

    def _new_total_size(self):
        total_size = 0.
        overlap = self.as_overlap_dict()
        for labels, size in overlap.items():
            total_size += size
        return total_size

    def _old_vennbug_total_size(self):
        # NOTE: this may have been correct with our broken venn_js format...
        # but it's no longer correct.
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
        NOTE: This returns venn_js sizes.
        i.e.
        ('CPU',) contains the size of the CPU region, INCLUDING possible overlap with GPU.
        ('CPU', 'GPU',) contains only the overlap across CPU and GPU.
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

    def as_overlap_dict(self):
        if self._is_vennbug_version():
            return self._old_vennbug_as_overlap_dict()
        return self._new_as_overlap_dict()


    def _new_as_overlap_dict(self):
        """
        ##
        ## To calcuate V[…] from O[…]:
        ##
        # Add anything from O which is subset-eq of [0,1,2]
        V[0,1,2] = O[0,1,2]
        # Add anything from O which is subset-eq of [0,2]
        V[0,2] = O[0,2] + O[0,1,2]
        V[1,2] = O[1,2] + O[0,1,2]
        V[0,1] = O[0,1] + O[0,1,2]
        # Add anything from O which is subset-eq of [0]
        V[0] = O[0] + O[0,1] + O[0,2] + O[0,1,2]
        V[1] = O[1] + O[0,1] + O[1,2] + O[0,1,2]
        V[2] = O[2] + O[0,2] + O[1,2] + O[0,1,2]

        ##
        ## To calcuate O[…] from V[…]:
        ##
        O[0,1,2] = V[0,1,2]
        O[0,2] = V[0,1] - O[0,1,2]
        O[1,2] = V[1,2] - O[0,1,2]
        O[0,1] = V[0,1] - O[0,1,2]
        O[0] = V[0] - O[0,1] - O[0,2] - O[0,1,2]
        O[1] = V[1] - O[0,1] - O[1,2] - O[0,1,2]
        O[2] = V[2] - O[0,2] - O[1,2] - O[0,1,2]

        PSEUDOCODE:
        INPUT: V[region] -> size
        OUTPUT: O[region] -> size

        For region in sorted(V.keys(), key=lambda region: len( region)):
            O[region] = V[region]
            # Subtract all keys we've added so far to O that are a subset of region.
            Sub_regions = set([k for k in O.keys() if k != region and k.issubset(region)])
            For k in sub_regions:
                O[region] -= O[k]
        """
        V = self.as_dict()
        O = venn_as_overlap_dict(V)
        return O

    def _old_vennbug_as_overlap_dict(self):
        """
        NOTE: This returns "overlap" sizes.
        i.e.
        ('CPU',) contains the exclusive size of the CPU region, NOT INCLUDING possible overlap with GPU.
        ('CPU', 'GPU',) contains only the overlap across CPU and GPU.
        {
            ('CPU',): 132010993,
            ('GPU',): 0,
            ('CPU', 'GPU'): 3230025.0,
        }
        PSEUDOCODE:
        overlap = dict()
        for region, size_us in venn_sizes.keys():
            if len(region) == 1:
                overlap[region] += size_us
            else:
                overlap[region] = size_us
                for r in region:
                    overlap[(r,)] -= size_us
        """
        venn_sizes = self.as_dict()
        overlap = dict()
        def mk_region(region):
            if region not in overlap:
                overlap[region] = 0.
        for region, size_us in venn_sizes.items():
            if len(region) == 1:
                mk_region(region)
                overlap[region] += size_us
            else:
                mk_region(region)
                overlap[region] = size_us
                for r in region:
                    mk_region((r,))
                    overlap[(r,)] -= size_us

        for region in overlap.keys():
            # Convert things like this to zero to prevent negatives causing false assertions.
            # ('GPU',): -2.9103830456733704e-11,
            if np.isclose(overlap[region], 0):
                overlap[region] = 0

        for region, size_us in overlap.items():
            assert size_us >= 0

        # NOTE: if a region only exists in overlap with another region
        # (e.g. CUDA API CPU + Framework API C),
        # we DON'T want to have a legend-label for it.
        del_regions = set()
        for region, size_us in overlap.items():
            if size_us == 0:
                del_regions.add(region)
        for region in del_regions:
            del overlap[region]

        return overlap

    def as_df(self, keep_metadata_fields):
        overlap = self.as_overlap_dict()
        data = dict()

        def mk_col(col):
            if col not in data:
                data[col] = []

        def append_col(col, value):
            mk_col(col)
            data[col].append(value)

        for region, size in overlap.items():
            append_col('region', region)
            append_col('size', size)
            for field in keep_metadata_fields:
                append_col(field, self._metadata[field])

        df = pd.DataFrame(data)
        return df


def get_training_durations_df(directories,
                              debug=False,
                              debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.training_duration_df()
    return map_readers(TrainingProgressDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def get_total_timesteps(directories,
                        debug=False,
                        debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.total_timesteps()
    return map_readers(TrainingProgressDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)

def get_end_num_timesteps(directories,
                          debug=False,
                          debug_single_thread=False):
    def get_value(df_reader):
        return df_reader.end_num_timesteps()
    return map_readers(TrainingProgressDataframeReader, directories, get_value,
                       debug=debug,
                       debug_single_thread=debug_single_thread)


def map_readers(DataframeReaderKlass, directories, func,
                debug=False,
                debug_single_thread=False):
    xs = []

    if type(directories) == str:
        dirs = [directories]
    else:
        dirs = list(directories)

    for directory in dirs:
        df_reader = DataframeReaderKlass(
            directory,
            # add_fields=self.maybe_add_algo_env,
            debug=debug,
            debug_single_thread=debug_single_thread)
        x = func(df_reader)
        xs.append(x)

    if type(directories) == str:
        assert len(xs) == 1
        return xs[0]
    return xs

def overlap_as_venn_dict(O):
    """
    ##
    ## To calcuate V[…] from O[…]:
    ##
    # Add anything from O which is subset-eq of [0,1,2]
    V[0,1,2] = O[0,1,2]
    # Add anything from O which is subset-eq of [0,2]
    V[0,2] = O[0,2] + O[0,1,2]
    V[1,2] = O[1,2] + O[0,1,2]
    V[0,1] = O[0,1] + O[0,1,2]
    # Add anything from O which is subset-eq of [0]
    V[0] = O[0] + O[0,1] + O[0,2] + O[0,1,2]
    V[1] = O[1] + O[0,1] + O[1,2] + O[0,1,2]
    V[2] = O[2] + O[0,2] + O[1,2] + O[0,1,2]

    PSEUDOCODE:
    INPUT: O[region] -> size
    OUTPOUT: V[region] -> size

    For region in O.keys():
        V[region] = 0
        add_keys = set([k for k in O.keys() if region.issubset(k)])
        For k in add_keys:
            V[region] += O[k]

    # Add any "single regions" that are missing.
    labels = set()
    For region in O.keys():
        for label in region:
            labels.insert(label)
    single_regions = set({l} for l in labels)
    for region in single_regions:
        if region not in V:
            V[region] = 0
            for k in O.keys():
                if region.issubset(k):
                    V[region] += O[k]


    :param O:
        overlap dict
    :return:
    """
    V = dict()
    for region in O.keys():
        V[region] = 0
        add_keys = set([k for k in O.keys() if set(region).issubset(set(k))])
        for k in add_keys:
            V[region] += O[k]

    # Add any "single regions" that are missing.
    labels = set()
    for region in O.keys():
        for label in region:
            labels.add(label)
    single_regions = set((l,) for l in labels)
    for region in single_regions:
        if region in V:
            continue
        V[region] = 0
        for k in O.keys():
            if set(region).issubset(set(k)):
                V[region] += O[k]

    return V

def venn_as_overlap_dict(V):
    """
    ##
    ## To calcuate O[…] from V[…]:
    ##
    O[0,1,2] = V[0,1,2]
    O[0,2] = V[0,1] - O[0,1,2]
    O[1,2] = V[1,2] - O[0,1,2]
    O[0,1] = V[0,1] - O[0,1,2]
    O[0] = V[0] - O[0,1] - O[0,2] - O[0,1,2]
    O[1] = V[1] - O[0,1] - O[1,2] - O[0,1,2]
    O[2] = V[2] - O[0,2] - O[1,2] - O[0,1,2]

    PSEUDOCODE:
    INPUT: V[region] -> size
    OUTPUT: O[region] -> size

    For region in sorted(V.keys(), key=lambda region: len( region)):
        O[region] = V[region]
        # Subtract all keys we've added so far to O that are a subset of region.
        Sub_regions = set([k for k in O.keys() if k != region and k.issubset(region)])
        For k in sub_regions:
            O[region] -= O[k]

    :param V:
        venn_js dict
    :return:
    """
    O = dict()
    for region in reversed(sorted(V.keys(), key=lambda region: len(region))):
        O[region] = V[region]
        # Subtract all keys we've added so far to O that are a subset of region.
        sub_regions = set([k for k in O.keys() if k != region and set(region).issubset(set(k))])
        for k in sub_regions:
            O[region] -= O[k]

    del_regions = set()
    for region, size in O.items():
        if size == 0:
            del_regions.add(region)
    for region in del_regions:
        del O[region]

    return O

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

class TestVennAsOverlapDict:

    def check_eq(self, name, got, expect):
        result = (got == expect)
        if not result:
            logger.info(pprint_msg({
                'name': name,
                'got': got,
                'expect': expect,
            }))
        assert got == expect

    def check_from_venn(self, name, V, expect_O):
        got_O = venn_as_overlap_dict(V)
        self.check_eq("{name}.venn_as_overlap_dict".format(name=name), got_O, expect_O)
        got_V = overlap_as_venn_dict(expect_O)
        expect_V = V
        self.check_eq("{name}.overlap_as_venn_dict".format(name=name), got_V, expect_V)

    def check_from_overlap(self, name, O, expect_V):
        got_V = overlap_as_venn_dict(O)
        self.check_eq("{name}.overlap_as_venn_dict".format(name=name), got_V, expect_V)

        V = expect_V
        expect_O = O
        got_O = venn_as_overlap_dict(V)
        self.check_eq("{name}.test_venn_as_overlap_dict_06.overlap_as_venn_dict".format(name=name), got_O, expect_O)


    def test_venn_as_overlap_dict_01(self):
        V = {
            (0,): 6,
            (1,): 6,
            (2,): 6,
            (0,1): 2,
            (0,2): 2,
            (1,2): 2,
            (0,1,2): 1,
        }
        expect_O = {
            (0,): 3,
            (1,): 3,
            (2,): 3,
            (0,1): 1,
            (0,2): 1,
            (1,2): 1,
            (0,1,2): 1,
        }
        self.check_from_venn("test_venn_as_overlap_dict_01", V, expect_O)

    def test_venn_as_overlap_dict_02(self):
        V = {
            (0,): 5,
            (1,): 5,
            (0,1): 2,
        }
        expect_O = {
            (0,): 3,
            (1,): 3,
            (0,1): 2,
        }
        self.check_from_venn("test_venn_as_overlap_dict_02", V, expect_O)

    def test_venn_as_overlap_dict_03(self):
        V = {
            (0,): 3,
            (1,): 3,
            (2,): 3,
            (0,1): 1,
            (0,2): 1,
        }
        expect_O = {
            (0,): 1,
            (1,): 2,
            (2,): 2,
            (0,1): 1,
            (0,2): 1,
        }
        self.check_from_venn("test_venn_as_overlap_dict_03", V, expect_O)

    # def test_venn_as_overlap_dict_04(self):
    #     # FROM : /mnt/data/james/clone/rlscope/output/rlscope_bench/all/config_instrumented_repetition_01/ppo2/Walker2DBulletEnv-v0/CategoryOverlap.machine_2420f5fc91b8.process_ppo2_Walker2DBulletEnv-v0.phase_ppo2_Walker2DBulletEnv-v0.ops_sample_action.resources_CPU_GPU.venn_js.json
    #     # Generated using old SQL code.
    #     # "venn": [
    #     #     {
    #     #         "label": "CUDA API CPU",
    #     #         "sets": [
    #     #             0
    #     #         ],
    #     #         "size": 1157825.0156115906
    #     #     },
    #     #     {
    #     #         "label": "Framework API C",
    #     #         "sets": [
    #     #             1
    #     #         ],
    #     #         "size": 1242334.3759821171
    #     #     },
    #     #     {
    #     #         "label": "GPU",
    #     #         "sets": [
    #     #             2
    #     #         ],
    #     #         "size": 1242334.3759821171
    #     #     },
    #     #     {
    #     #         "sets": [
    #     #             1,
    #     #             2
    #     #         ],
    #     #         "size": 84509.36037052666
    #     #     },
    #     #     {
    #     #         "sets": [
    #     #             0,
    #     #             1,
    #     #             2
    #     #         ],
    #     #         "size": 1157825.0156115906
    #     #     }
    #     # ]
    #     V = {
    #         # Fails due to floating point weirdness.
    #         # (0,): decimal.Decimal(1157825.0156115906),
    #         # (1,): decimal.Decimal(1242334.3759821171),
    #         # (2,): decimal.Decimal(1242334.3759821171),
    #         # (1,2): decimal.Decimal(84509.36037052666),
    #         # # Q: Why isn't there a (0,2) / (0,1)?
    #         # (0,1,2): decimal.Decimal(1157825.0156115906),
    #         (0,): 1157825,
    #         (1,): 1242334,
    #         (2,): 1242334,
    #         (1,2): 84509,
    #         # Q: Why isn't there a (0,2) / (0,1)?
    #         (0,1,2): 1157825,
    #     }
    #     expect_O = {
    #         (0,): V[(0,)] - (V[(0,1,2)]),
    #         (1,): V[(1,)] - (V[(1,2)] - V[(0,1,2)]) - (V[(0,1,2)]),
    #         (2,): V[(2,)] - (V[(1,2)] - V[(0,1,2)]) - (V[(0,1,2)]),
    #         (1,2): V[(1,2)] - V[(0,1,2)],
    #         # Q: Why isn't there a (0,2) / (0,1)?
    #         (0,1,2): V[(0,1,2)],
    #     }
    #     check_from_venn("test_venn_as_overlap_dict_04", V, expect_O)
    # test_venn_as_overlap_dict_04()

    def test_venn_as_overlap_dict_05(self):
        O = {
            (0,): 5,
            (0,1): 5,
        }
        expect_V = {
            (0,): 10,
            (1,): 5,
            (0,1): 5,
        }
        got_V = overlap_as_venn_dict(O)
        self.check_eq("test_venn_as_overlap_dict_05.overlap_as_venn_dict", got_V, expect_V)

    def test_venn_as_overlap_dict_06(self):
        O = {
            (0,1): 5,
        }
        expect_V = {
            (0,): 5,
            (1,): 5,
            (0,1): 5,
        }
        self.check_from_overlap("test_venn_as_overlap_dict_06", O, expect_V)

        # got_V = overlap_as_venn_dict(O)
        # check_eq("test_venn_as_overlap_dict_06.overlap_as_venn_dict", got_V, expect_V)
        #
        # V = expect_V
        # expect_O = O
        # got_O = venn_as_overlap_dict(V)
        # check_eq("test_venn_as_overlap_dict_06.overlap_as_venn_dict", got_O, expect_O)
