"""
Python / numba implementation of event overlap computation needed for
scoping CPU/GPU time to high-level user operation annotations.

.. deprecated:: 1.0.0
    We no longer compute event overlap in Python/numba.
    Instead, we implement event overlap in C++ for performance.
"""
from rlscope.profiler.rlscope_logging import logger
import itertools
from collections import namedtuple
import functools
from os.path import join as _j, dirname as _d
import copy

import types

from rlscope import py_config

if py_config.USE_NUMBA:
    import numba
    from numba import njit

from rlscope.profiler.util import pprint_msg
from rlscope.parser.common import *
from rlscope.parser import constants
# from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from rlscope.protobuf.pyprof_pb2 import CategoryEventsProto, ProcessMetadata

from rlscope.profiler import proto_util

from rlscope.parser.stats import KernelTime

from rlscope.parser.trace_events import TraceEventsDumper, dump_category_times
from rlscope.profiler.concurrent import ForkedProcessPool, FailedProcessException

from rlscope.parser.dataframe import venn_as_overlap_dict, overlap_as_venn_dict

from rlscope.profiler import concurrent

from concurrent.futures import ProcessPoolExecutor

from rlscope.parser.db import SQLCategoryTimesReader, sql_input_path, sql_get_source_files, \
    Machine, Process, Phase, EventSplit, \
    GetConnectionPool, \
    EventsAsEOTimes, AsNumbaEOTimes, category_to_idx_maps

from rlscope.parser.readers import TFProfCategoryTimesReader, \
   DEFAULT_group_by_device, \
   DEFAULT_ignore_categories, \
   DEFAULT_debug


if py_config.USE_NUMBA:
    from rlscope.scripts.unique_intervals import UniqueSplits, PlotOutput, ShowOrSave, \
        bitset_add, \
        bitset_contains, \
        bitset_empty_set, \
        bitset_full_set, \
        bitset_indices, \
        bitset_is_empty, \
        bitset_remove, \
        bitset_union, \
        bitset_np_bool_vector

class ComputeOverlap:
    # DEBUG = True
    DEBUG = False
    """
    Want to compute:
    [ CUDA CPU / GPU | GPU | both ]

    In the future, we could end up with finer grained categories for CPU.
    i.e.
    Currently CUDA CPU time is the whole TF operator time.
    Maybe we want to add TF framework time on the main thread.
    Q: Would we just group that time into CPU? Or, would we add a 3rd category?
    Then we may want to see all combinations of overlaps between things.
    The strategy to compute that is the same regardless;
    (1) for each category, group times into single contiguous group of overlapping time (TimeGroup)
    (2) When looking at a single pair of categories, we look for overlaps between time groups
    (3) To look at overlap between say [CUDA CPU + Framework CPU] and [GPU], we need to
        group the CUDA CPU and framework CPU TimeGroups, then again run the pairwise overlap detector.

        PROBLEM: what about TimeGroup that is JUST GPU... we don't want to count an overlap twice,
        since we wish to create a stacked bar graph; i.e.
        We don't want to count time twice in
        - [CUDA CPU] overlap-with [GPU]
        - [CUDA CPU + Framework CPU] overlap-with [GPU]

        To prevent this, when we form TimeGroup from [CUDA CPU + Framework CPU],
        we need to be sure that EXCLUSIVELY those overlap, and no other categories.
        Better algo:
        Compute TimeGroup's for each category
        for TimeGroups (c1, c2, ..., cn) in next TimeGroups from category c<i>:
          We want to find a divisions of (c1, ..., cn) such that ci and cj are
          for each group of overlapping segments set(ci, cj, ..., ck),
              such that there is no cm not in set(ci, cj, ..., ck) with cm.overlaps(any ci in set(...))

              time[set(ci, cj, ..., ck)] += overlap(ci, cj, ..., ck)

    Perhaps we want to subdivide CUDA CPU time

    category_times = {
        'CUDA CPU': [
            [[start, end], ...],
            [[start, end], ...],
            ...
        ]
        'Framework CPU': [
            [[start, end], ...],
            [[start, end], ...],
            ...
        ]
        'GPU': [
            [[start, end], ...],
            [[start, end], ...],
            ...
        ]
    }
    merged_category_times = {
        'CUDA CPU': [[start, end], ...],
        'Framework CPU': [[start, end], ...],
        'GPU': [[start, end], ...],
    }

    :param overlaps_with
        Only keep times whose category_key contains at least one of these categories.

        Typically overlaps_with = [constants.CATEGORY_OPERATION], so we only keep execution time
        that happened during an operation.
    """
    def __init__(self,
                 # category_times,
                 eo_times,
                 overlaps_with=None,
                 keep_empty_time=False,
                 check_key=None,
                 timer=None,
                 debug=False,
                 show_progress=False):
        self.timer = timer
        self.debug = ( ComputeOverlap.DEBUG or py_config.RLSCOPE_DEBUG_UNIT_TESTS ) and debug
        self.check_key = check_key
        self.show_progress = show_progress
        self.overlaps_with = overlaps_with
        self.keep_empty_time = keep_empty_time

        self.eo_times = eo_times

        # if self.overlaps_with is not None:
        #     self.overlaps_with = set(self.overlaps_with)
        #     for category in self.overlaps_with:
        #         assert category in category_times.keys()
        #
        # self.category_times = category_times
        # self.category_times = self._sort_category_times(self.category_times)
        # if self.timer is not None:
        #     self.timer.end_operation('ComputerOverlap._sort_category_times()')

        # Sanity check: no self-overlap
        # for category_key, events in self.category_times.items():
        #     for e1, e2 in zip(events, events[1:]):
        #         # [ e1 ]
        #         #   [ e2 ]
        #         if e2.start_time_usec < e1.end_time_usec:
        #             logger.info("Saw overlap within event list for same category_key: {msg}".format(msg=pprint_msg({
        #                 'category_key': category_key,
        #                 'e1': e1,
        #                 'e2': e2,
        #                 })))
        #             assert not( e2.start_time_usec < e1.end_time_usec )
        # if self.timer is not None:
        #     self.timer.end_operation('ComputerOverlap: sanity check - no self-overlap')


    def compute(self):
        # start_merge_t = time.time()
        # self.compute_merge()
        # if self.timer is not None:
        #     self.timer.end_operation('ComputerOverlap.compute_merge()')
        # end_merge_t = time.time()
        # sec_merge = end_merge_t - start_merge_t
        # if self.debug:
        #     logger.info("> {klass}.compute_merge took {sec} seconds".format(
        #         klass=self.__class__.__name__,
        #         sec=end_merge_t))
        start_compute_t = time.time()
        self.compute_times()
        end_compute_t = time.time()
        sec_compute = end_compute_t - start_compute_t
        if self.debug:
            logger.info("> {klass}.compute_times took {sec} seconds".format(
                klass=self.__class__.__name__,
                sec=sec_compute))

    # def compute_merge(self):
    #     self.merged_category_times = self._merge_category_times(self.category_times)

    def compute_times(self):
        # set(c1, ..., cn) -> time in seconds
        self.times, self.overlap_metadata = self._compute_overlap(self.eo_times)

    def get_category_times(self):
        return self.times

    def get_overlap_metadata(self):
        return self.overlap_metadata

    def get_merged_categories(self):
        return self.merged_category_times

    # def _flatten_category_times(self, category_times):
    #     new_category_times = dict()
    #     for category in category_times.keys():
    #         all_times = []
    #         for times in category_times[category]:
    #             all_times.extend(times)
    #         new_category_times[category] = all_times
    #     return new_category_times

    def _sort_category_times(self, category_times):
        new_category_times = dict()
        for category in category_times:
            new_category_times[category] = sorted(category_times[category], key=lambda ktime: ktime.start_time_usec)
            new_category_times[category] = merge_adjacent_events(new_category_times[category])
        return new_category_times

    def _merge_category_times(self, category_times):
        merged_category_times = dict()
        for category in category_times.keys():
            times = category_times[category]
            merged_category_times[category] = self._merge_times(times)
        return merged_category_times

    def _merge_times(self, times):
        """
        Given a list of sorted KernelTime's, merge adjacent
        overlapping KernelTime's into a single KernelTime.

        :param times: List[KernelTime]
            times is in sorted order by event.start_time_usec.
        :return:
        """
        new_times = [times[0]]
        i = 1
        while i < len(times):
            curtime = new_times[-1]
            if curtime.overlaps(times[i]):
                new_times[-1] = curtime.merge(times[i])
            else:
                new_times.append(times[i])
            i += 1
        return new_times

    def _compute_overlap(self, eo_times_dict):
        if py_config.USE_NUMBA:
            overlap, overlap_metadata = compute_overlap_single_thread_numba(
                # category_times,
                eo_times_dict,
                self.overlaps_with,
                self.check_key,
                self.debug,
                self.show_progress,
                timer=self.timer,
            )
        else:
            overlap, overlap_metadata = compute_overlap_single_thread(
                eo_times_dict,
                self.overlaps_with,
                self.check_key,
                self.debug,
                self.show_progress,
                timer=self.timer,
            )

        if self.overlaps_with is not None:
            del_keys = []
            for categories_key in overlap.keys():
                if len(self.overlaps_with.intersection(categories_key)) == 0:
                    del_keys.append(categories_key)

            for categories_key in del_keys:
                del overlap[categories_key]

        if not self.keep_empty_time:
            # Delete empty keys; these result from blank space between events. e.g.
            #   frozenset(): 1000000
            del_keys = []
            for categories_key in overlap.keys():
                if len(categories_key) == 0:
                    del_keys.append(categories_key)

            for categories_key in del_keys:
                del overlap[categories_key]

        if self.timer is not None:
            self.timer.end_operation('ComputerOverlap._compute_overlap(): fixup results')

        return overlap, overlap_metadata


def split_category_times(category_times, n):
    pass

def _sort_category_times_by(category_times, key):
    new_category_times = dict()
    for category in category_times:
        new_category_times[category] = sorted(category_times[category], key=key)
    return new_category_times

class ListWrapper:
    """
    Wrap pop() so it doesn't change the underlying list,
    but still makes it look like the list has shrunk.
    """
    def __init__(self, xs):
        self.xs = xs
        self._len = len(self.xs)
        self._orig_len = self._len

    def pop(self):
        self._len -= 1

    def __len__(self):
        return self._len

    def __getitem__(self, key):
        if type(key) == int:
            if key < 0:
                # Index relative to end of list.
                # Need to take into account any .pop() calls.
                # e.g. if they called .pop() once,  when they reference xs[-1], give them xs[-2]
                #                            twice,                     xs[-1],           xs[-3]
                #                            ...
                idx = (self._len - self._orig_len) + key
                return self.xs[idx]
            elif key > self._len:
                raise IndexError("list index out of range: len={len}, index={key}".format(
                    len=self._len, key=key))
        return self.xs[key]

    def __setitem__(self, key, value):
        self.xs[key] = value

class CategoryTimesWrapper:
    # enum
    START = 0
    END = 1
    NAME_TO_TYPE_CODE = {
        'start':START,
        'end':END,
    }

    def __init__(self, ctimes, key, reversed_key, name):
        self.key = key
        self.reversed_key = reversed_key
        self.name = name
        self.by_key = _sort_category_times_by(ctimes, key=self.reversed_key)
        self.type_code = CategoryTimesWrapper.NAME_TO_TYPE_CODE[self.name]
        self._num_events = self._count_times_left()

    def min_time(self):

        min_category = None
        min_ktime = None
        for category in self.by_key.keys():
            ktimes = self.by_key[category]
            if len(ktimes) > 0:
                ktime = ktimes[-1]
                if min_category is None or \
                    self.key(ktime) < self.key(min_ktime):
                    min_category = category
                    min_ktime = ktime

        return min_category, min_ktime

    def has_times_left(self):
        return self._num_events > 0

    def count_times_left(self):
        return self._num_events

    def _count_times_left(self):
        return sum(len(ctimes_for_category) for ctimes_for_category in self.by_key.values())

    def pop_time(self, category):
        self.by_key[category].pop()
        self._num_events -= 1

    def get_time(self, ktime):
        return self.key(ktime)

    @staticmethod
    def min_time_start_end(by_start, by_end):
        if not by_end.has_times_left():
            start_category, start_ktime = by_start.min_time()
            return start_category, start_ktime, by_start
        elif not by_start.has_times_left():
            end_category, end_ktime = by_end.min_time()
            return end_category, end_ktime, by_end

        start_category, start_ktime = by_start.min_time()
        end_category, end_ktime = by_end.min_time()
        if by_start.key(start_ktime) <= by_end.key(end_ktime):
            return start_category, start_ktime, by_start
        return end_category, end_ktime, by_end

    @staticmethod
    def total_left(by_start, by_end):
        return by_start.count_times_left() + by_end.count_times_left()

class RegionMetadata:
    def __init__(self, category_key=None, start_time_usec=None, end_time_usec=None, num_events=0, time_unit='us'):
        self.category_key = category_key
        self.start_time_usec = us_from_unit(start_time_usec, time_unit=time_unit)
        self.end_time_usec = us_from_unit(end_time_usec, time_unit=time_unit)
        self.num_events = num_events

    def add_event(self, event):
        if self.start_time_usec is None or event.start_time_usec < self.start_time_usec:
            self.start_time_usec = event.start_time_usec

        if self.end_time_usec is None or event.end_time_usec > self.end_time_usec:
            self.end_time_usec = event.end_time_usec

        self.num_events += 1

    def merge_inplace(self, region2):
        region1 = self
        region1.start_time_usec = min([
            x for x in [region1.start_time_usec, region2.start_time_usec]
            if x is not None],
            default=None)
        region1.end_time_usec = max([
            x for x in [region1.end_time_usec, region2.end_time_usec]
            if x is not None],
            default=None)
        region1.num_events = region1.num_events + region2.num_events
        return region1

    @staticmethod
    def from_NumbaRegionMetadata(numba_region_metadata, category_to_idx, idx_to_category, time_unit):
        category_key = category_key_from_bitset(
            numba_region_metadata.category_key,
            category_to_idx,
            idx_to_category)
        region_metadata = RegionMetadata(
            category_key=category_key,
            start_time_usec=numba_region_metadata.start_time_usec,
            end_time_usec=numba_region_metadata.end_time_usec,
            num_events=numba_region_metadata.num_events,
            time_unit=time_unit,
        )
        return region_metadata

    def __str__(self):
        return "(key={key}, start_usec={start}, end_usec={end}, num_events={num_events})".format(
            key=self.category_key,
            start=self.start_time_usec,
            end=self.end_time_usec,
            num_events=self.num_events)

class OverlapMetadata:
    """
    When computing overlap between categories,
    compute some useful statistics about each overlap region:

    - region.start_time_usec:
      The earliest event.start_time_usec that has contributed to this region.

    - region.end_time_usec:
      The latest event.end_time_usec that has contributed to this region.
    """
    def __init__(self):
        # CategoryKey -> RegionMetadata
        self.regions = dict()

    def add_event(self, category_key, event):
        if category_key not in self.regions:
            self.regions[category_key] = RegionMetadata(category_key)

        self.regions[category_key].add_event(event)

    @staticmethod
    def merge_regions(regions):
        region = RegionMetadata()
        for r in regions:
            region.merge_inplace(r)
        return region

    @staticmethod
    def merge_overlap_metadata_pair(overlap_metadata1, overlap_metadata2):
        assert type(overlap_metadata1) == OverlapMetadata
        assert type(overlap_metadata2) == OverlapMetadata

        merged_od = OverlapMetadata()
        for od in [overlap_metadata1, overlap_metadata2]:
            for category_key, region in od.items():
                merged_od.merge_region(category_key, region)
        return merged_od

    @staticmethod
    def merge_overlap_metadata(overlap_metadatas):
        overlap_metadata = functools.reduce(
            OverlapMetadata.merge_overlap_metadata_pair,
            overlap_metadatas,
            OverlapMetadata())
        return overlap_metadata

    def items(self):
        return self.regions.items()

    def merge_keys(self, category_key1, category_key2, new_category_key):
        region1 = self.regions[category_key1]
        region2 = self.regions[category_key2]
        region = region1.merge(region2, new_category_key)
        del self.regions[category_key1]
        del self.regions[category_key2]

        # Q: Perhaps we're merging into an existing region though...
        assert new_category_key not in self.regions
        self.regions[new_category_key] = region

    def get_region(self, category_key):
        return self.regions[category_key]

    def merge_region(self, category_key, region):
        if category_key not in self.regions:
            self.regions[category_key] = RegionMetadata(category_key)
        self.regions[category_key].merge_inplace(region)

    def _add_region(self, category_key, region_metadata):
        assert category_key not in self.regions
        self.regions[category_key] = region_metadata

    # @staticmethod
    # def from_NumbaOverlapMetadata(numba_overlap_metadata, category_to_idx, idx_to_category):
    #     overlap_metadata = OverlapMetadata()
    #     for category_key_bitset, numba_region_metadata in numba_overlap_metadata.regions.items():
    #         category_key = category_key_from_bitset(category_key_bitset, category_to_idx, idx_to_category)
    #         region_metadata = RegionMetadata.from_NumbaRegionMetadata(numba_region_metadata, category_to_idx, idx_to_category)
    #         overlap_metadata._add_region(category_key, region_metadata)
    #     return overlap_metadata

    @staticmethod
    def from_NumbaOverlapMetadata(numba_overlap_metadata, category_to_idx, idx_to_category, time_unit):
        overlap_metadata = OverlapMetadata()
        for category_key_bitset, numba_region_metadata in numba_overlap_metadata.items():
            category_key = category_key_from_bitset(category_key_bitset, category_to_idx, idx_to_category)
            region_metadata = RegionMetadata.from_NumbaRegionMetadata(numba_region_metadata, category_to_idx, idx_to_category, time_unit)
            overlap_metadata._add_region(category_key, region_metadata)
        return overlap_metadata

    def __str__(self):
        return "OverlapMetadata(regions={regions})".format(regions=self.regions)

if py_config.USE_NUMBA:
    @numba.njit
    def best_by_index_start(lle, cindices):
        """
        Find next minimum start-time.

        :param lle: [[Events]]
            Category times, represented as a List-of-List-of-Events (lle).
        :param cindices:
            Next category time to consider, for each Category categories[i].
        :return:
        """
        best = -1
        best_time = sys.maxsize
        for i in range(len(cindices)):
            if cindices[i] < len(lle[i]) and \
                lle[i][cindices[i]].start_time_usec <= best_time:
                    best_time = lle[i][cindices[i]].start_time_usec
                    best = i
        return best, best_time

    @numba.njit
    def best_by_index_end(lle, cindices):
        """
        Same as best_by_index_start, but find next minimum end-time.
        """
        best = -1
        best_time = sys.maxsize
        for i in range(len(cindices)):
            if cindices[i] < len(lle[i]) and \
                lle[i][cindices[i]].end_time_usec <= best_time:
                    best_time = lle[i][cindices[i]].end_time_usec
                    best = i
        return best, best_time




def category_key_from_bitset(bitset, category_to_idx, idx_to_category, freeze=True):
    category_key = set()
    indices = bitset_indices(bitset)
    for idx in indices:
        category = idx_to_category[idx]
        category_key.add(category)
    if freeze:
        category_key = frozenset(category_key)
    return category_key

class Overlap:
    """
    Numbafied version of OverlapMetadata.
    """
    def __init__(self):
        # CategoryKey -> Int64
        # self.overlap_regions = dict()
        raise NotImplementedError()

    # @staticmethod
    # def from_NumbaOverlap(numba_overlap, category_to_idx, idx_to_category):
    #     overlap = dict()
    #     for category_key_bitset, time_usec in numba_overlap.overlap_regions.items():
    #         category_key = category_key_from_bitset(category_key_bitset, category_to_idx, idx_to_category)
    #         overlap[category_key] = time_usec
    #     return overlap

    @staticmethod
    def from_NumbaOverlap(numba_overlap_dict, category_to_idx, idx_to_category, time_unit):
        overlap = dict()
        for category_key_bitset, time_usec in numba_overlap_dict.items():
            category_key = category_key_from_bitset(category_key_bitset, category_to_idx, idx_to_category)
            overlap[category_key] = us_from_unit(time_usec, time_unit)
        return overlap

if py_config.USE_NUMBA:
    @numba.njit
    def numba_compute_overlap(
        by_start, by_end,
        show_progress=False,
        debug=False):
        """
        Convert event overlap computation into something that is optimizable by numba.
        In particular:
            - avoid classes and complicated Python data-structures (dicts, lists, sets, combinations thereof)
            - use numpy operations/arrays (e.g. broadcasting adds)
        To get an idea of what numba can optimize and cannot, see the following:
            https://numba.pydata.org/numba-doc/dev/user/5minguide.html

        :param by_start:
            Category -> [Event]
            Represented as [[Event]]
            In particular, by_start[i][...] are all the Event's that belong to Category=categories[i]
            Events in each list are sorted by start time.
        :param by_end:
            Category -> [Event]
            Represented as [[Event]]
            Same as by_start, except Events in each list are sorted by end time.
        :return: overlap:
             { Category }     ->          Int64
             ------------                 -----
            Overlap region        Duration of overlap

            Mapping from a set of Categories (a.k.a. overlap region) to total time (in microseconds).
            An "overlap region" is a set of overlapping categories, and the total duration of the overlap.

        # :param categories
        #     List of strings representing category names:
        #     e.g. {"CPU", "GPU", ...}
        """

        # NOTE: assertions cause unsupported opcode error:
        #   Use of unknown opcode 'IMPORT_NAME'
        # Work-around: push the assertions up into our python caller.
        # assert len(by_start) == len(by_end)
        k = len(by_start)

        overlap = NumbaOverlap()

        # How many events are in each Category.
        lengths = np.array([len(l) for l in by_start], dtype=int)

        if debug:
            # NOTE: we use print instead of logger.info so that
            # these will print when running Numba JIT compiled code.
            print("(1) after lengths = ...")

        overlap_metadata = NumbaOverlapMetadata()
        if len(lengths) == 0 or np.sum(lengths) == 0:
            # Either no categories, or no
            return overlap, overlap_metadata

        if debug:
            print("(2) NumbaOverlapMetadata()")

        start_index = np.zeros(k, dtype=int)
        end_index = np.zeros(k, dtype=int)

        # cur_categories = set()
        cur_categories = bitset_empty_set()
        min_by_start, min_time_by_start = best_by_index_start(by_start, start_index)
        if debug:
            print("(3) after finding start time of earliest event with best_by_index_start")
        cur_time = min_time_by_start
        # cur_categories.add(min_by_start)
        cur_categories = bitset_add(cur_categories, min_by_start)

        while (start_index < lengths).any() or (end_index < lengths).any():
            min_by_start, min_time_by_start = best_by_index_start(by_start, start_index)
            if debug:
                print("(4) after best_by_index_start")
            min_by_end, min_time_by_end = best_by_index_end(by_end, end_index)
            if debug:
                print("(5) after best_by_index_end")
            # assert min_by_start >= 0 or min_by_end >= 0

            if min_time_by_start <= min_time_by_end:
                min_time = min_time_by_start
                event = by_start[min_by_start][start_index[min_by_start]]
                if debug:
                    print("(6) after by_start[min_by_start]")
                min_category = min_by_start
                is_start = True
            else:
                min_time = min_time_by_end
                event = by_end[min_by_end][end_index[min_by_end]]
                if debug:
                    print("(7) after by_end[min_by_end]")
                min_category = min_by_end
                is_start = False

            time_chunk = min_time - cur_time

            # if len(cur_categories) > 0 and time_chunk > 0:
            if not bitset_is_empty(cur_categories) and time_chunk > 0:
                # Don't bother recording empty gaps between times.
                # categories_key = frozenset(cur_categories)

                # if cur_categories not in overlap:
                #     overlap[cur_categories] = 0
                # overlap[cur_categories] += time_chunk
                overlap.add_time(cur_categories, time_chunk)
                if debug:
                    print("(8) overlap.add_time")
                overlap_metadata.add_event(cur_categories, event)
                if debug:
                    print("(8) overlap_metadata.add_event")

            if is_start:
                start_index[min_by_start] += 1
                # cur_categories.add(min_category)
                cur_categories = bitset_add(cur_categories, min_category)
                if debug:
                    print("(9) after is_start")
            else:
                end_index[min_by_end] += 1
                # cur_categories.remove(min_category)
                cur_categories = bitset_remove(cur_categories, min_category)
                if debug:
                    print("(9) after not is_start")

            # if show_progress:
            #     count_left = CategoryTimesWrapper.total_left(by_start, by_end)
            #     bar.update(total_events - count_left)

            cur_time = min_time

        # assert len(cur_categories) == 0

        return overlap, overlap_metadata


    def compute_overlap_single_thread_numba(
        # category_times,
        eo_times_dict,
        overlaps_with=None,
        check_key=None,
        debug=False,
        show_progress=False,
        timer=None):

        # Python -> Numba:
        #   Convert Python types to Numba types.
        category_to_idx, idx_to_category = category_to_idx_maps(eo_times_dict)
        eo_times = AsNumbaEOTimes(
            eo_times_dict,
            # category_times,
            category_to_idx, idx_to_category,
            # debug=debug,
        )
        if timer is not None:
            timer.end_operation('compute_overlap_single_thread_numba(...): Python -> Numba (eo_times)')

        # logger.info("{msg}".format(msg=pprint_msg({
        #     'idx_to_category': idx_to_category,
        #     # 'eo_times': eo_times,
        # })))
        # logger.info("eo_times: {msg}".format(msg=pprint_msg(eo_times)))

        # Call into Numba
        use_numba = not py_config.RLSCOPE_DISABLE_JIT
        # numba_overlap, numba_overlap_metadata = UniqueSplits(eo_times, use_numba=use_numba)
        numba_overlap, numba_overlap_metadata, outputs, output_categories = UniqueSplits(eo_times, use_numba=use_numba)
        if py_config.RLSCOPE_DEBUG_UNIQUE_SPLITS_BASE:
            cat_idx_pairs = sorted([(cat, idx) for (cat, idx) in category_to_idx.items()], key=lambda cat_idx: cat_idx[1])
            categories = [cat for cat, idx in cat_idx_pairs]
            PlotOutput(outputs, output_categories, categories)
            ShowOrSave(
                base=py_config.RLSCOPE_DEBUG_UNIQUE_SPLITS_BASE,
                interactive=False,
            )

        if timer is not None:
            timer.end_operation('compute_overlap_single_thread_numba(...): Event overlap; UniqueSplits(eo_times)')

        # Numba -> Python:
        #   Convert Numba types back to Python types.
        overlap_metadata = OverlapMetadata.from_NumbaOverlapMetadata(numba_overlap_metadata, category_to_idx, idx_to_category, time_unit='ps')
        overlap = Overlap.from_NumbaOverlap(numba_overlap, category_to_idx, idx_to_category, time_unit='ps')

        if timer is not None:
            timer.end_operation('compute_overlap_single_thread_numba(...): Numba -> Python')

        return overlap, overlap_metadata

def compute_overlap_single_thread(
    category_times,
    overlaps_with=None,
    check_key=None,
    debug=False,
    show_progress=False,
    timer=None):

    overlap_metadata = OverlapMetadata()

    start_key = lambda ktime: ktime.start_time_usec
    reversed_start_key = lambda ktime: - ktime.start_time_usec
    end_key = lambda ktime: ktime.end_time_usec
    reversed_end_key = lambda ktime: - ktime.end_time_usec

    by_start = CategoryTimesWrapper(category_times, start_key, reversed_start_key, 'start')
    if timer is not None:
        timer.end_operation('compute_overlap_single_thread(...): sort by start-time')
    by_end = CategoryTimesWrapper(category_times, end_key, reversed_end_key, 'end')
    if timer is not None:
        timer.end_operation('compute_overlap_single_thread(...): sort by end-time')

    cur_categories = set()
    cur_events = dict()
    def pop_start(category, event):
        by_start.pop_time(category)
        cur_categories.add(category)
        if check_key:
            assert category not in cur_events
            cur_events[category] = event
    def pop_end(category):
        by_end.pop_time(category)
        if category in cur_categories:
            cur_categories.remove(category)
        else:
            # # Q: Under what circumstances might this occur?  What information do we need to backtrack?
            pass
        if check_key:
            assert category in cur_events
            del cur_events[category]

    times = dict()

    if len(category_times) == 0:
        return dict(), overlap_metadata

    min_category, min_ktime = by_start.min_time()
    start_or_end = by_start
    curtime = by_start.key(min_ktime)
    pop_start(min_category, min_ktime)

    if debug:
        logger.info("> Start computing overlap; choose initial start (curtime)")
        pprint.pprint({
            'min_category': min_category,
            'min_ktime': min_ktime,
            'start_or_end': start_or_end.name,
            'curtime': curtime,
        })
        logger.info()

    total_events = CategoryTimesWrapper.total_left(by_start, by_end)
    bar = progress(desc="compute_overlap", show_progress=show_progress, total=total_events)

    # Takes 4:50.
    # Delete from back of array, now it takes: 3:12
    # ListWrapper, now it takes: 4:48
    # Avoid counting/summing, now it takes: 2:43
    # avoid intermediate ktimes list for min operation, now it takes: 2:40
    while by_start.has_times_left() or by_end.has_times_left():

        min_category, min_ktime, start_or_end = CategoryTimesWrapper.min_time_start_end(by_start, by_end)
        next_time = start_or_end.get_time(min_ktime)
        time_chunk = next_time - curtime

        # NOTE: Avoid adding time_chunk=0 overlaps.
        # These can happen when an event finishes just as the next one starts.
        # This can lead to "false positive bugs" when we
        # see operations within the same process overlapping.
        if len(cur_categories) > 0 and time_chunk > 0:
            # Don't bother recording empty gaps between times.
            categories_key = frozenset(cur_categories)

            # For debugging; useful to check for overlaps we don't expect to occur.
            # For example, overlap across constants.CATEGORY_OPERATION's within a single process.
            if check_key is not None:
                md = {
                    'curtime':curtime,
                    'next_time':next_time,
                    'ktime':min_ktime,
                    'ktime_dict':min_ktime.__dict__,
                    'cur_events':cur_events,
                    'cur_categories':cur_categories,
                }
                check_key(categories_key, md)

            if categories_key not in times:
                NumberType = type(time_chunk)
                times[categories_key] = NumberType(0.)
            overlap_metadata.add_event(categories_key, min_ktime)
            times[categories_key] += time_chunk

        if debug:
            pprint.pprint({
                'min_category': min_category,
                'min_ktime': min_ktime,
                'start_or_end': start_or_end.name,
                'update':"times[{s}] += {t}".format(s=categories_key, t=time_chunk),
            })
            logger.info()

        if start_or_end.type_code == CategoryTimesWrapper.START:
            if debug:
                pprint.pprint({'cur_categories':cur_categories,
                               'add': min_category,
                               'curtime': next_time})
                logger.info()
            pop_start(min_category, min_ktime)
        else:
            if debug:
                pprint.pprint({'cur_categories':cur_categories,
                               'remove': min_category,
                               'curtime': next_time})
                logger.info()
            # NOTE: I think this bug will occur if you have two events from the same category
            # that overlap each other (EITHER partially or fully).
            # Perhaps this is happening inadvertantly when I form the 'CPU'
            # events from the Python/C++ events.
            # - TODO:
            #   - make a test case to reproduce the bug
            #   - assert for this condition being violated in the event-data
            #   - pickle the category_times data so we can iterate quickly.
            # KeyError: frozenset({'CPU', 'PROC:loop_train_eval'})
            pop_end(min_category)

        if show_progress:
            count_left = CategoryTimesWrapper.total_left(by_start, by_end)
            bar.update(total_events - count_left)

        curtime = next_time

    bar.close()

    assert len(cur_categories) == 0

    if timer is not None:
        timer.end_operation('compute_overlap_single_thread(...): Event overlap')

    for categories_key in list(times.keys()):
        # We may get artificial overlaps even if two categories are synchronous,
        # if the next category starts exactly when the last one ends.
        if times[categories_key] == 0:
            del times[categories_key]

    return times, overlap_metadata

class CategoryTimesWalker:
    def __init__(self, category_times):
        self.category_times = category_times


class TraceEventsParser:
    """
    Usage modes:
    python3 python/profiler/analyze.py --directories ... --rules TraceEventsParser
      Default mode; just show just the FIRST "step" for EACH operation (e.g. q_forward, q_backward, ...) in the ML script.
      Each operation will generate a separate trace_events.json file.
      NOTE: a "step" = an 0-based identifier for each traced TensorFlow C++ API call (e.g. q_forward)

    python3 python/profiler/analyze.py --directories ... --rules TraceEventsParser --op-name q_forward
      Same as above, except just output a file for --op-name.

    python3 python/profiler/analyze.py --directories ... --rules TraceEventsParser --overlaps-event-id 116098
      Keep all events that overlap with a particular event-id (from the Event SQL table).
      Useful for debugging overlap computation.

    python3 python/profiler/analyze.py --directories ... --rules TraceEventsParser --start-usec <epoch_us> --end-usec <epoch_us>
      Output ALL events that fall within a certain [--start-usec, --end-usec] epoch time range.
      NOTE: We output IMLUnitTest data for this as well.

    """

    # TODO: Add the ability to parse IMLUnitTestOnce/Multiple files and add the events to the trace_events.json.
    # Row:
    # IMLUnitTest:
    #
    # We can just show the ENTIRE trace;
    # For larger traces (minigo), we probably just want to 'zoom-in' to a trouble-some area
    # (phase start/end time + IMLUnitTest start/end time for that phase)

    # def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 # Swallow any excess arguments
                 debug=False,
                 filter_op=True,
                 show_progress=False,
                 overlaps_event_id=None,
                 op_name=None,
                 process_name=None,
                 process_op_nest=False,
                 start_usec=None,
                 end_usec=None,
                 # process_op_nest=False,
                 **kwargs):

        self.directory = directory
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug
        self.filter_op = filter_op
        self.start_usec = start_usec
        self.end_usec = end_usec
        self.overlaps_event_id = overlaps_event_id
        self.op_name = op_name
        self.process_name = process_name
        self.show_progress = show_progress
        self.process_op_nest = process_op_nest
        self.td = None

        # self.dummy_times = []

        # self.category_times_readers = []

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    # Output

    def _trace_first_step_path(self, process_name, step, bench_name):
        path = _j(self.directory, 'traceEvents{proc}{step}{bench}.json'.format(
            proc=process_suffix(process_name),
            step=step_suffix(step),
            bench=bench_suffix(bench_name),
        ))
        return path

    def _trace_event_overlap_path(self, event):
        path = _j(self.directory, 'traceEvents{proc}{event}.json'.format(
            proc=process_suffix(event.process_name),
            event=event_suffix(event.event_id),
        ))
        return path

    def _trace_event_time_range_path(self):
        def usec_suffix(usec):
            assert usec is not None
            return ".{us}_usec".format(
                us=usec,
            )
        path = _j(self.directory, 'traceEvents{start}{end}.json'.format(
            start=usec_suffix(self.start_usec),
            end=usec_suffix(self.end_usec),
        ))
        return path

    def _trace_first_step(self):
        ignore_cats = list(DEFAULT_ignore_categories)
        if constants.CATEGORY_DUMMY_EVENT in ignore_cats:
            ignore_cats.remove(constants.CATEGORY_DUMMY_EVENT)

        machines = self.sql_reader.machines()
        assert len(machines) == 1
        machine = machines[0]

        logger.info("op_names = {op_names}".format(op_names=self.op_names))
        logger.info("process_names = {procs}".format(procs=self.sql_reader.process_names(machine.machine_name)))
        for op_name in self.op_names:
            logger.info("op_name = {op_name}".format(op_name=op_name))
            for process_name, step, category_times in itertools.islice(
                    self.sql_reader.each_op_instance(op_name,
                                                     machine_name=machine.machine_name,
                                                     filter_op=self.filter_op,
                                                     group_by_device=True,
                                                     ignore_categories=ignore_cats,
                                                     debug=self.debug),
                    # Just grab the very first operation from the very first process.
                    0, 1):
                logger.info("  process_name = {proc}, step = {step}".format(proc=process_name, step=step))

                logger.info("> Generate traceEvents for step={step}".format(step=step))

                trace_events_dumper = TraceEventsDumper(
                    category_times,
                    json_path=self._trace_first_step_path(process_name, step, op_name),
                    debug=self.debug)
                trace_events_dumper.dump()

    def _trace_event_overlap(self):
        event = self.sql_reader.event_by_id(self.overlaps_event_id, self.debug)
        category_times = self.sql_reader.events_that_overlap_with(
            event, event.process_name,
            show_progress=self.show_progress)
        trace_events_dumper = TraceEventsDumper(
            category_times,
            json_path=self._trace_event_overlap_path(event),
            debug=self.debug)
        trace_events_dumper.dump()

    def _add_unit_test_times(self, category_times):
        """
        td = Read IMLUnitTest

        # NOTE: We need to restrict events we add to falling between [self.start_usec, self.end_usec]

        category = "IMLUnitTest: phases"
        for process in td.processes:
            for phase in process.phases:
                category_times[category].append(
                  Event(name=phase, ...))

        category = "IMLUnitTest: profiling"
        category_times[category].append(
          Event(name='prof.start()/prof.stop()', start_us=td.start_usec, end_us=td.end_usec))

        :return:
        """
        if self.td is None:
            from rlscope.profiler.profilers import unit_test_util
            self.td = unit_test_util.TestData(debug=self.debug)
            self.td.read_directory(self.directory)

        def should_keep_event(event):
            return self.start_usec <= event.start_time_us <= proto_util.event_end_us(event) <= self.end_usec

        processes = [self.td.get_process(self.process_name)]

        # prof.set_phase(...)
        category = 'IMLUnitTest: phases'
        # for process in self.td.processes:
        for process in processes:
            for phase in process.phases:
                for event in process.events(phase):
                    if should_keep_event(event):
                        # assert event.name == phase
                        ktime = KernelTime(name=phase, start_usec=event.start_time_us, end_usec=proto_util.event_end_us(event))
                        if category not in category_times:
                            category_times[category] = []
                        category_times[category].append(ktime)

        # prof.start()/stop()
        category = 'IMLUnitTest: processes'
        # for process in self.td.processes:
        for process in processes:
            if should_keep_event(process.prof_event):
                # TODO: us prof.start()/prof.stop() times instead of first/last phase times.
                # However, we need to adjust unit-tests first.
                ktime = KernelTime(name=process.process_name, start_usec=process.prof_event.start_time_us, end_usec=proto_util.event_end_us(process.prof_event))
                if category not in category_times:
                    category_times[category] = []
                category_times[category].append(ktime)

    def _trace_event_time_range(self):
        category_times = self.sql_reader.events_by_time_range(
            self.start_usec, self.end_usec, self.process_name,
            debug=self.debug)
        # category_times = self.sql_reader.events_that_overlap_with(
        #     event, event.process_name,
        #     show_progress=self.show_progress)

        self._add_unit_test_times(category_times)

        trace_events_dumper = TraceEventsDumper(
            category_times,
            json_path=self._trace_event_time_range_path(),
            debug=self.debug)
        trace_events_dumper.dump()

    def run(self):

        self.sql_reader = SQLCategoryTimesReader(sql_input_path(self.directory), host=self.host, user=self.user, password=self.password)
        if self.op_name is not None:
            self.op_names = [self.op_name]
        else:
            self.op_names = self.sql_reader.op_names()


        if self.overlaps_event_id is not None:
            trace_func = self._trace_event_overlap
        elif self.start_usec is not None:
            process_names = self.sql_reader.process_names()
            if self.end_usec is None or self.process_name is None:
                raise RuntimeError("Need --start-usec, --end-usec, and --process-name for TraceEventsParser time-range;\n  process_names = {procs}".format(
                    procs=process_names))
            trace_func = self._trace_event_time_range
        else:
            trace_func = self._trace_first_step

        trace_func()

    def get_output_path(self):
        if self.overlaps_event_id is not None:
            event = self.sql_reader.event_by_id(self.overlaps_event_id, self.debug)
            return self._trace_event_overlap_path(event)
        elif self.start_usec is not None:
            return self._trace_event_time_range_path()
        else:
            return None
            # ignore_cats = list(DEFAULT_ignore_categories)
            # if constants.CATEGORY_DUMMY_EVENT in ignore_cats:
            #     ignore_cats.remove(constants.CATEGORY_DUMMY_EVENT)
            #
            # for op_name in self.op_names:
            #     for process_name, step, category_times in itertools.islice(
            #             self.sql_reader.each_op_instance(op_name,
            #                                              group_by_device=True,
            #                                              ignore_categories=ignore_cats,
            #                                              debug=self.debug),
            #             # Just grab the very first operation from the very first process.
            #             0, 1):
            #
            #         logger.info("> Generate traceEvents for step={step}".format(step=step))
            #
            #         trace_events_dumper = TraceEventsDumper(
            #             category_times,
            #             json_path=self._trace_first_step_path(process_name, step, op_name),
            #             debug=self.debug)
            #         trace_events_dumper.dump()

    def dump(self, bench_name):
        if self.skip:
            return


class _CPUAndGPUCategories(namedtuple('CPUAndGPUCategories', 'cpus gpus')):
    __slots__ = ()
def CPUAndGPUCategories(cpus, gpus):
    return _CPUAndGPUCategories(
        cpus=frozenset(cpus),
        gpus=frozenset(gpus))

def split_cpu_gpu_categories(category_key):
    cpus = set()
    gpus = set()
    for category in category_key.non_ops:
        if category in constants.CATEGORIES_CPU:
            cpus.add(category)
        elif category in constants.CATEGORIES_GPU:
            gpus.add(category)
        else:
            raise NotImplementedError("Not sure how to categorize category={cat} as CPU vs GPU".format(
                cat=category))
    return CPUAndGPUCategories(cpus, gpus)

def traceEvents_key_str(category_key):
    assert isinstance(category_key, _CategoryKey)
    proc_prefix = ""
    if len(category_key.procs) > 0:
        proc_prefix = "[" + \
                      ", ".join(match_proc_category(proc).group('process_name') for proc in category_key.procs) + \
                      "] : "

    exec_prefix = ""
    if len(category_key.non_ops) > 0:
        exec_prefix = ", ".join(category_key.non_ops) + " - "

    op_prefix = ""
    if len(category_key.ops) > 0:
        op_prefix = ", ".join(category_key.ops)

    return "{proc_prefix}{exec_prefix}{op_prefix}".format(
        proc_prefix=proc_prefix,
        exec_prefix=exec_prefix,
        op_prefix=op_prefix)



class EventSplitter:
    def __init__(self,
                 process_name=None,
                 phase_name=None,
                 machine_name=None,
                 start_time_us=None,
                 end_time_us=None,
                 debug=False):
        self.process_name = process_name
        self.phase_name = phase_name
        self.machine_name = machine_name
        self.start_time_us = start_time_us
        self.end_time_us = end_time_us
        self.debug = debug

    # IDEA: to make a more robust approximation, we could do some SQL queries to guesstimate
    # what time duration is needed to achieve events_per_split by looking at the number
    # of events in the first split and extrapolating:
    #
    # select max(e.end_time_usec) as end_time_usec,
    # ordered by e.start_time_usec
    # limit {events_per_split}
    #
    # Q: Will this do what I think it should?
    # i.e. Will this sort the events by start-time, then give me the maximum of just the LIMIT rows (not outside the limit)?
    # A: No... but we CAN do that with a subselect query:
    # https://stackoverflow.com/questions/1150715/how-can-i-use-max-and-limit-together-in-mysql
    #
    # select max(end_time_usec) from (
    #     select e.end_time_usec from events
    #     ordered by e.start_time_usec
    #     limit {events_per_split}
    # )
    #
    # IDEA: If we want to be precise, we can iteratively perform this same query, except we specify that
    # e.start_time_usec >= LAST_CHUNK.end_time_usec.

    def events_per_event_splits(self, sql_reader, events_per_split, min_splits=1):
        """
        Given that we want approximately events_per_split for each split,
        perform a crude approximation to return splits of equal trace-time duration.

        Crude approximation for load-balancing splits:
            ASSUMPTION: events are equally distributed across the trace.

            NOTE: this is not precisely true, but the workload is at least iterative
            so the sequence of events should repeat themselves at each iteration, which will be captured
            for a large enough events_per_split.

        :param sql_reader:
        :param events_per_split:
        :return:
        """

        # Reasonable lower bound...otherwise there's no work to be done in one split.
        # Generally, we'd like this to capture at least 10 iterations of the computation we are measuring...
        # but it's not the end of the world if it doesn't (just means splits will be unbalanced...).
        assert events_per_split > 1000

        period = sql_reader.query_trace_period(
            machine_name=self.machine_name,
            process_name=self.process_name,
            phase_name=self.phase_name)
        logger.info("Tracing period: {msg}".format(msg=pprint_msg(period)))

        # Approximation:
        # number of splits = 1/(events / split) * [total events]
        # total events = SQL: select count(*) from event-trace
        desired_splits = int(np.ceil(
            (1/events_per_split) * period.total_events
        ))
        n_splits = max(
            min_splits,
            desired_splits,
        )

        logger.info("Total events = {n}".format(
            n=period.total_events))

        logger.info("Desired splits = {splits}".format(
            splits=desired_splits))

        event_splits = self._n_splits(period, n_splits)
        return event_splits

    def equal_time_event_splits(self, sql_reader, n_splits):
        """
        Return splits computed based on equal chunks of time over the tracing-time.

        Rationale: workload is iterative, so the number
        of events per-split will be roughly balanced.

        :param n_splits:
        :return:
        """
        assert n_splits >= 1
        period = sql_reader.query_trace_period(
            machine_name=self.machine_name,
            process_name=self.process_name,
            phase_name=self.phase_name)
        logger.info("Tracing period: {msg}".format(msg=pprint_msg(period)))
        event_splits = self._n_splits(period, n_splits)
        return event_splits

    def _n_splits(self, period, n_splits):
        """
        Return n_splits of equal time-duration over the trace-period.
        """
        # E.g.
        # split = 2
        # start_time = 5
        # end_time = 10
        # duration = 5
        # duration_per_split = 2.5
        # splits = [5, 7.5] [7.5, 10]
        # splits = [
        #     [start_time + i * duration_per_split,
        #      start_time + (i + 1) * duration_per_split]
        #   for i in range(n_splits)]
        duration_us = period.duration_us

        # NumberType = type(duration_us)
        # n_splits = NumberType(n_splits)

        # Round up to nearest integer.
        duration_per_split_us = int(1 + duration_us/n_splits)
        event_splits = [
            EventSplit(
                period.start_time_us + i*duration_per_split_us,
                min(period.start_time_us + (i + 1)*duration_per_split_us,
                    period.end_time_us),
                )
            for i in range(n_splits)]
        for split in event_splits:
            check_no_decimal(split.start_time_us)
            check_no_decimal(split.end_time_us)
        assert event_splits[0].start_time_us == period.start_time_us
        assert event_splits[-1].end_time_us == period.end_time_us
        return event_splits

def split_overlap_computation_Worker(kwargs):
    kwargs = dict(kwargs)
    self = kwargs['self']
    del kwargs['self']
    if self.debug_perf:
        timer = SimpleTimer("split_overlap_computation_Worker")
        timer.reset_start_time()
    else:
        timer = None
    with GetConnectionPool(conn_kwargs=dict(
        db_path=self.db_path,
        host=self.host,
        user=self.user,
        password=self.password,
    ),
        maxconn=1,
        # forked child process worker for computing overlap split.
        new_process=True) as pool:
        ret = self._split_overlap_computation(timer=timer, **kwargs)
    if self.debug_perf:
        logger.info("[--debug-perf] Time breakdown of split_overlap_computation_Worker: {msg}".format(
            msg=pprint_msg(timer),
        ))
    return ret

class OverlapComputer:
    """
    Computes a json file containing the overlap between event categories across different steps,
    on a per-operation/bench_name basis.

    e.g.
    tfprof.q_forward.json:
        For each q_forward operation, tell me the overlap among
        the different categories measure (GPU, Python, CUDA API C, etc)
    """

    def __init__(self, db_path,
                 host=None,
                 user=None,
                 password=None,
                 timer=None,
                 debug=False,
                 debug_single_thread=False,
                 debug_perf=False,
                 debug_ops=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.db_path = db_path
        self.host = host
        self.user = user
        self.password = password
        self.timer = timer
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.debug_perf = debug_perf
        self.debug_ops = debug_ops

    @property
    def directory(self):
        return _d(self.db_path)

    # def _compute_process_timeline_stats(self, sql_reader, overlap):
    #     """
    #
    #     Q: What's the total time spent tracing that isn't captured by ANY tracing events?
    #     i.e.
    #     missing_traced_time_sec = total_trace_time_sec - time_overlap(anything)
    #                                                      ----------------------
    #                                                      traced_time_sec
    #
    #     Q: What's the total time spent tracing that ISN'T captured by operation trace events?
    #     i.e.
    #     missing_op_time_sec = total_trace_time_sec - time_overlap(with operation-type)
    #
    #     Q: What's the total time spent tracing that ISN'T captured by CPU/GPU operation trace events?
    #     i.e.
    #     missing_op_time_sec = total_trace_time_sec - time_overlap(with CPU/GPU operations)
    #                                                  -------------------------------------
    #                                                  traced_op_time_sec
    #
    #     :param overlap
    #       Output from:
    #         category_times, ... = sql_reader.parse_timeline()
    #         compute_overlap = ComputeOverlap(category_times)
    #         compute_overlap.compute()
    #         overlap = compute_overlap.get_category_times()
    #     """
    #     # NOTE: it's nicer to work with
    #     new_overlap_01 = reduce_category_keys(overlap)
    #     NumberType = float
    #     if len(new_overlap_01) > 0:
    #         NumberType = type(next(iter(new_overlap_01.values())))
    #     traced_time_sec = NumberType(0.)
    #     traced_op_time_sec = NumberType(0.)
    #     for category_key, time_us in new_overlap_01.items():
    #         traced_time_sec += time_us/NumberType(constants.MICROSECONDS_IN_SECOND)
    #         if len(category_key.ops) > 0:
    #             traced_op_time_sec += time_us/NumberType(constants.MICROSECONDS_IN_SECOND)
    #
    #     total_trace_time_sec = sql_reader.total_trace_time_sec(debug=self.debug)
    #     missing_traced_time_sec = total_trace_time_sec - as_type(traced_time_sec, type(total_trace_time_sec))
    #     missing_op_time_sec = total_trace_time_sec - as_type(traced_op_time_sec, type(total_trace_time_sec))
    #
    #     proc_stats = {
    #         # Tracing time that ISN'T covered by ANY trace-events.
    #         # Q: How to fix?
    #         # - Not sure... reduce "gaps" between traced events?
    #         'missing_traced_time_sec':missing_traced_time_sec,
    #         # Tracing time that ISN'T covered by ANY operation trace-events.
    #         # Q: How to fix?
    #         # - Add more set_operation calls to ML script.
    #         'missing_op_time_sec':missing_op_time_sec,
    #     }
    #     return proc_stats

    def reduce_overlap_resource_operation(
            self, overlap, overlap_metadata, visible_overhead,
            categories, operation_types, proc_types,
            group_self_overlap=False):
        """
        Reduce keys to pair of operation-types, or a single operation-type.
        (eliminate process, just keep operation-type and execution-type)

        We want to produce overlap that has keys like:

        1. Single-op type:
           e.g.
           <q_forward>: The q_forward operation, not overlapped with any other operation.

        2. Pair of op-types:
           e.g.
           <q_forward, q_forward>: Two or more q_forward operations running simultaneously.
           <q_forward, q_backward>: Two or more q_forward and q_backward operations running simultaneously.

        PSEUDOCODE: Reduce across the "Process<ID>" dimension:

        For keys like <q_forward, P1, P2, ..., [CPU, GPU]>:
            new_overlap[q_forward, q_forward] +=
              overlap[q_forward, P1, P2, ..., [CPU, GPU]]

        For keys like <q_forward, ..., P1, ...>:
            "Operations can only overlap cross-process, not within a single-process"
            Assert: If len(ops) > 1 then len(procs) > 1
            new_overlap[q_forward, ..., [CPU, GPU]] +=
              overlap[q_forward, ..., P1, ..., [CPU, GPU]]

        return new_overlap
        """

        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            if len(overlap_key.ops) == 1 and len(overlap_key.procs) >= 2:
                """
                Single operation type, with overlap.
                
                i.e. operation overlaps with itself across processes.
                e.g. <q_forward, q_forward>
                """
                cat = next(iter(overlap_key.ops))
                if group_self_overlap:
                    # "Group" overlap of an operation with itself across processes
                    # into the operation's time.
                    ops = [cat]
                else:
                    # "Group" overlap of an operation with itself across processes
                    # into a pair of operation-types <op, op>.
                    ops = [cat, cat]
                new_key = CategoryKey(ops=ops,
                                      non_ops=overlap_key.non_ops,
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))
                continue

            if len(overlap_key.ops) >= 1 and len(overlap_key.procs) >= 1:
                """
                Either:
                - Single operation type, no overlap (single with overlap already handled above!)
                - Multi-operation type, with overlap
                """
                new_key = CategoryKey(ops=overlap_key.ops,
                                      non_ops=overlap_key.non_ops,
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))
                continue

            if self.debug:
                pprint.pprint({
                    'overlap.keys()':list(overlap.keys()),
                })
            raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        return new_overlap, new_overlap_metadata

    # PROBLEM: CategoryKey has lost whether q_forward belonged to p1 or p2...
    def reduce_overlap_ResourceOverlap(
            self, overlap, overlap_metadata,
            categories, operation_types, proc_types):
        """
        Group keys by resource-type (non_ops).
        """

        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            new_key = CategoryKey(ops=frozenset(),
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            _add_key(new_overlap, new_key, times)
            new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

        return new_overlap, new_overlap_metadata

    def reduce_overlap_ResourceSubplot(
            self, overlap, overlap_metadata,
            categories, operation_types, proc_types):
        """
        Group keys by resource-type (non_ops).
        """
        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            for resource_type in overlap_key.non_ops:
                # NOTE: This is sort of hacky;
                # we AREN'T outputting disjoint overlap regions here;
                # instead we are outputting an entire "set" including its overlaps:
                # i.e.
                # CPU   = [CPU only time] + [CPU overlapped with GPU time]
                # GPU   = [GPU only time] + [GPU overlapped with GPU time]
                # Total = [CPU only time] + [GPU only time] + [CPU overlapped with GPU time]
                new_key = CategoryKey(ops=frozenset(),
                                      non_ops=frozenset([resource_type]),
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

                new_key = CategoryKey(ops=frozenset(),
                                      non_ops=frozenset([constants.CATEGORY_TOTAL]),
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

        return new_overlap, new_overlap_metadata

    def reduce_overlap_OperationOverlap(
            self, overlap, overlap_metadata, visible_overhead,
            categories, operation_types, proc_types):
        """
        Remove keys that don't match CPU(?).

        Group keys by operation-type (non_ops).
        """
        return self.reduce_overlap_resource_operation(
            overlap, overlap_metadata, visible_overhead,
            categories, operation_types, proc_types,
            group_self_overlap=True)

    def _debug_trace_events_path(self, process_name, phase_name):
        return _j(self.directory, "OverlapComputer.trace_events{proc}{phase}{debug}.json".format(
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            # bench=bench_suffix(bench_name),
            debug=debug_suffix(self.debug),
        ))

    def _compute_metadata(self, category_times):
        """
        Q: How can we make the start/end time's zero-based?
        A: We need to subtract the earliest time we are displaying.
           So, bootstrap's start_time_usec serves as the "zero" time for the whole visualization.
           The javascript code will have to handle that then.
        metadata = {
            start_time_usec: ...,
            end_time_usec: ...,
        }

        :param category_times:
        :return:
        """
        def get_extreme(extreme, key, default=None):
            best = default
            found = False
            for cat, events in category_times.items():
                if len(events) > 0:
                    candidate = extreme(events, key=key)
                    if best is None:
                        best = candidate
                    else:
                        best = extreme(best, candidate, key=key)
                    found = True
            if found:
                return key(best)
            return default
            # return extreme(
            #     extreme(key(event) for event in category_times[cat])
            #     for cat in category_times.keys()
            # )

        metadata = {
            'start_time_usec':get_extreme(min, lambda event: event.start_time_usec),
            'end_time_usec':get_extreme(max, lambda event: event.end_time_usec),
        }
        return metadata


    OVERLAP_TYPES = ['OperationOverlap', 'ResourceOverlap', 'ResourceSubplot', 'CategoryOverlap', 'default']
    def compute_process_timeline_overlap(self,
                                         # pre_reduce,
                                         visible_overhead=False,
                                         machine_name=None,
                                         process_name=None,
                                         phase_name=None,
                                         n_workers=1,
                                         events_per_split=10000,
                                         debug_memoize=False,
                                         # overlap_type=None,
                                         ):
        """
        Compute CPU/GPU overlap for a given process, and for a given phase.

        :param process_name:
            Limit events to a specific process.
            If not provided, consider all processes.
        :param phase_name:
            Limit events to a specific phase.
            If not provided, consider all phases.
        :param debug_memoize:
        :return:
        """

        # TODO: run this function on EACH event-split for a particular (machine, process, phase).
        sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

        # if overlap_type is None:
        #     overlap_type = 'default'
        # assert overlap_type in OverlapComputer.OVERLAP_TYPES

        event_splitter = EventSplitter(
            process_name=process_name,
            phase_name=phase_name,
            machine_name=machine_name,
            # debug=self.debug,
            debug=True,
        )
        event_splits = event_splitter.events_per_event_splits(sql_reader, events_per_split,
            # At least maximize potential parallelism.
            min_splits=n_workers)
        # if self.debug:
        logger.info("event_splits: {msg}".format(
            msg=pprint_msg(event_splits)))
        logger.info("Using n_splits = {n_splits}, n_workers = {n_workers}".format(
            n_splits=len(event_splits),
            n_workers=n_workers,
        ))
        sql_reader.close()

        def split_overlap_computation_Args(event_split):
            return dict(
                self=self,
                event_split=event_split,
                visible_overhead=visible_overhead,
                # pre_reduce=pre_reduce,
                machine_name=machine_name,
                process_name=process_name,
                phase_name=phase_name,
                debug_memoize=debug_memoize,
                # overlap_type=overlap_type,
            )
        with ProcessPoolExecutor(n_workers) as pool:
            kwargs_list = [split_overlap_computation_Args(event_split) for event_split in event_splits]
            split_overlap_results = concurrent.map_pool(pool, split_overlap_computation_Worker, kwargs_list,
                     desc="OverlapComputer.splits",
                     show_progress=True,
                     sync=self.debug_single_thread)

        overlap_results = [result[0] for result in split_overlap_results]
        overlap_metadata_results = [result[1] for result in split_overlap_results]

        merged_overlap = functools.reduce(merge_ComputeOverlap, overlap_results, dict())
        merged_overlap_metadata = OverlapMetadata.merge_overlap_metadata(overlap_metadata_results)

        return merged_overlap, merged_overlap_metadata


    def _split_overlap_computation(self, event_split,
                                   # pre_reduce,
                                   visible_overhead=False,
                                   machine_name=None,
                                   process_name=None,
                                   phase_name=None,
                                   debug_memoize=False,
                                   timer=None,
                                   # overlap_type=None,
                                   ):

        sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)
        if timer is not None:
            timer.end_operation('sql_reader = SQLCategoryTimesReader(...)')

        start_parse_timeline_t = time.time()
        category_times = sql_reader.parse_timeline(
            process_name=process_name,
            phase_name=phase_name,
            machine_name=machine_name,
            start_time_us=event_split.start_time_us,
            end_time_us=event_split.end_time_us,
            visible_overhead=visible_overhead,
            # pre_reduce=pre_reduce,
            timer=timer,
            debug=self.debug,
            debug_memoize=debug_memoize)
        sql_reader.close()

        if self.debug:
            pprint_msg({'category_times.keys()': sorted(category_times.keys())})

            # category_times_path = _j(self.directory, "category_times{over}{proc}{mach}{phase}.debug.txt".format(
            #     proc=process_suffix(process_name),
            #     mach=machine_suffix(machine_name),
            #     phase=phase_suffix(phase_name),
            #     over=overlap_type_suffix(overlap_type),
            # ))
            # logger.info('Write category_times @ {path}'.format(path=category_times_path))
            # with open(category_times_path, 'w') as f:
            #     # f.write(pprint_msg(category_times)
            #     print("> category_times data", file=f)
            #     print(pprint_msg({
            #         'process_name': process_name,
            #         'phase_name': phase_name,
            #         'machine_name': machine_name,
            #         'overlap_type': overlap_type,
            #     }))
            #     pprint.pprint(category_times, stream=f)

        end_parse_timeline_t = time.time()
        parse_timeline_sec = end_parse_timeline_t - start_parse_timeline_t
        if self.debug:
            logger.info("> parse_timeline took {sec} seconds".format(sec=parse_timeline_sec))

        check_key = None

        if self.debug:
            # Set a breakpoint if we detect overlap across operations within the same process.
            # We can inspect the event's using TraceDumper once we know where in the timeline they occur.
            # It would help to know the event_id as well...
            def check_key(overlap_key, md):
                category_key = _as_category_key(overlap_key=overlap_key)
                if len(category_key.ops) > 1:
                    # Operations can only overlap cross-process, not within a single-process
                    if not( len(category_key.procs) > 1 ):
                        logger.info("> Detected unexpected CategoryKey when computing overlap:")
                        pprint.pprint({
                            'category_key':category_key,
                            'md':md})
                    assert len(category_key.procs) > 1

        # metadata = self._compute_metadata(category_times)
        # pprint.pprint({
        #     'metadata': metadata,
        #     'process_name':process_name,
        #     'phase_name':phase_name})

        # NOTE: We can reduce across whatever dimensions we want to achieve different
        # levels/configurations of the drill-down.
        # Q: ... is that true?
        compute_overlap = ComputeOverlap(category_times,
                                         check_key=check_key,
                                         timer=timer,
                                         debug=self.debug,
                                         show_progress=self.debug)
        compute_overlap.compute()
        overlap = compute_overlap.get_category_times()
        overlap_metadata = compute_overlap.get_overlap_metadata()
        # maybe_memoize(debug_memoize, overlap, self._compute_overlap_memo_path())
        # assert len(overlap) > 0

        # pprint.pprint({
        #     'overlap_metadata': overlap_metadata,
        #     'machine_name':machine_name,
        #     'process_name':process_name,
        #     'phase_name':phase_name})

        # if self.debug:
        #     pprint.pprint({'overlap.keys()':list(overlap.keys())})

        return overlap, overlap_metadata

    def _group_by_ops_resource(self, new_overlap):
        # set(operation categories) -> set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #  <q_forward, q_backward>       <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        operation_overlap = dict()
        for combo_key, time_us in new_overlap.items():
            assert len(combo_key.ops) > 0
            assert len(combo_key.non_ops) > 0
            assert len(combo_key.procs) == 0
            if combo_key.ops not in operation_overlap:
                operation_overlap[combo_key.ops] = dict()
            operation_overlap[combo_key.ops][combo_key.non_ops] = time_us
        return operation_overlap

    def _group_by_resource(self, new_overlap):
        # set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        operation_overlap = dict()
        for combo_key, time_us in new_overlap.items():
            assert len(combo_key.ops) == 0
            assert len(combo_key.non_ops) > 0
            assert len(combo_key.procs) == 0
            assert combo_key.non_ops not in operation_overlap
            operation_overlap[combo_key.non_ops] = time_us
        return operation_overlap

    def _group_by_resource_ops(self, new_overlap):
        # set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec
        operation_overlap = dict()
        for combo_key, time_us in new_overlap.items():
            assert len(combo_key.ops) > 0
            assert len(combo_key.non_ops) > 0
            assert len(combo_key.procs) == 0
            if combo_key.non_ops not in operation_overlap:
                operation_overlap[combo_key.non_ops] = dict()
            operation_overlap[combo_key.non_ops][combo_key.ops] = time_us
        return operation_overlap


    def _debug_process_timeline_json_path(self):
        path = _j(self.directory, 'process_timeline.traceEvents.debug.json')
        return path

    def _compute_overlap_memo_path(self):
        return _j(self.directory, '{klass}.compute_overlap.pickle'.format(
            klass=self.__class__.__name__,
        ))

    # def _stats(self):
    #     return _j(self.directory, "process_timeline.stats.json")
    #
    # def _dump_stats(self):
    #     """
    #     Dump some stats useful for testing the correctness of our plot.
    #
    #     - Total time spent tracing:
    #       We expect total time spent tracing to match that total size of our bar-graph.
    #       NOTE: would be nice to measure this with time.time() separately, but oh well!
    #
    #     -
    #     :param bench_name:
    #     :return:
    #     """
    #     total_trace_time_sec = self.sql_reader.total_trace_time_sec(debug=self.debug)
    #     # EXPECT:
    #     # - total_trace_time_sec    ~ total_time_sec
    #     #   --------------------      --------------
    #     #   Anything that's traced    Stuff covered by operations
    #     # IF FAILS:
    #     # - then we aren't profiling part of the code.
    #     js_stats = {
    #         # Select min(start_time_us) as, max(end_time_us) from Event
    #         # (i.e. across all processes)
    #         'total_trace_time_sec':total_trace_time_sec,
    #     }
    #     _add_cpu_gpu_stats(js_stats, self.plotter)
    #     logger.info("> Save plot stats to {path}".format(path=self._stats()))
    #     do_dump_json(js_stats, self._stats())
    #     return js_stats

    DEBUG_COMPUTE_PER_OPERATION_OVERLAP = False
    def compute_per_operation_overlap(self, bench_name):

        # TODO: we could consider memoizing results inside of a file, but screw it shouldn't take too long.

        #
        # In order to handle overlapping events.
        # PSEUDOCODE:
        # SELECT all events that overlap with an event-type.
        #

        # PSEUDOCODE:
        # Compute [CUDA CPU time | GPU time | CUDA CPU/GPU time] overlap/non-overlap for a single step.
        #
        # We want to "group together" the 3 CPU lane thread times.
        # More when GPU time overlaps with nothing in CPU time,
        # and CPU time overlaps with nothing in GPU time.
        #
        # To get the average stats, we average ACROSS the steps measured.
        # Would be nice to know CPU is every executing anything else...
        # thread-pool threads are just blocked, but is the "main" thread doing anything?

        sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

        # Overlap, computed across different "steps".
        overlaps = []

        sql_reader.parse_debug = self.debug
        # class PerOperationOverlapWorker:
        #     def __init__(self):
        for process_name, step, category_times in sql_reader.each_op_instance(bench_name):

            if sql_reader.parse_debug:
                # Q: What do does train_loop look like, overlapped with all its fellow operation-types?
                json_path = _j(self.directory, 'OverlapComputer{proc}{step}{bench}.debug.json'.format(
                    proc=process_suffix(process_name),
                    step=step_suffix(step),
                    bench=bench_suffix(bench_name),
                ))
                logger.info("> DEBUG: dump trace events AFTER process_op_nest @ {path}".format(path=json_path))
                dump_category_times(category_times, json_path, print_log=False)

            for ktime in category_times.get(constants.CATEGORY_OPERATION, []):
                assert ktime.name == bench_name
            # JAMES TODO: We only want to compute overlap of execution time with op-events whose type is bench_name.
            # If it's just execution time without op-type overlap we should discard it.

            # JAMES TODO: remove "Operation" from plot labels
            # compute_overlap = ComputeOverlap(category_times, overlaps_with=[constants.CATEGORY_OPERATION])

            # Can take up to 1.2 seconds, often 0.011 seconds, 0.004 seconds for loop_train_eval.
            start_overlap_t = time.time()
            compute_overlap = ComputeOverlap(category_times)
            compute_overlap.compute()
            overlap = compute_overlap.get_category_times()
            end_overlap_t = time.time()
            sec_overlap = end_overlap_t - start_overlap_t
            if OverlapComputer.DEBUG_COMPUTE_PER_OPERATION_OVERLAP:
                logger.info("> compute_overlap(process={proc}, step={step}) took {sec} seconds".format(
                    proc=process_name,
                    step=step,
                    sec=sec_overlap))
            for category_key in list(overlap.keys()):
                if not( len(category_key) > 1 and constants.CATEGORY_OPERATION in category_key ):
                    del overlap[category_key]

            overlaps.append(overlap)

            # Only debug the first step of the first process (if --debug is set)
            sql_reader.parse_debug = False

        json_output = self._overlaps_as_json(overlaps)
        if self.debug:
            self._dump_per_operation_json(bench_name, json_output)

        return json_output

    def _overlaps_as_json(self, overlaps):
        pprint.pprint({
            'overlaps':overlaps,
        })

        categories = set()
        category_combinations = set()
        combo_to_id = dict()
        combo_id_pairs = []
        next_combo_id = 0
        for overlap in overlaps:
            for combo in overlap.keys():
                category_combinations.add(combo)
                combo_key = frozenset(combo)
                if combo_key not in combo_to_id:
                    combo_id = next_combo_id
                    next_combo_id += 1
                    combo_to_id[combo_key] = combo_id
                    combo_id_pairs.append((combo_id, sorted(list(combo))))
                for category in combo:
                    categories.add(category)

        combo_to_time_usec = dict()
        for overlap in overlaps:
            for combo_key in overlap.keys():
                if combo_key not in combo_to_time_usec:
                    combo_to_time_usec[combo_key] = []
                combo_to_time_usec[combo_key].append(overlap[combo_key])

        category_combo_times = []
        for combo_key in sorted(combo_to_time_usec.keys()):
            combo_times = combo_to_time_usec[combo_key]
            category_combo_times.append({
                'category_combo':sorted(combo_key),
                'times_usec':combo_times,
            })

        json_output = {
            'categories': sorted(list(categories)),
            'category_combinations': sorted(sorted(combo) for combo in category_combinations),
            'category_combo_times':category_combo_times,
        }

        return json_output

    def _dump_per_operation_json(self, bench_name, json_output):
        path = self._per_operation_json_path(bench_name)
        logger.info("> DEBUG: dump per-operation compute overlap @ {path}".format(path=path))
        do_dump_json(json_output, path)

    def _per_operation_json_path(self, bench_name):
        path = _j(self.directory, 'tfprof{bench}.json'.format(bench=bench_suffix(bench_name)))
        return path

# Debug ResourceOverlapType/OperationOverlapType/etc.
# Useful for avoiding debugging OverlapComputer (noisy debug messages).
DEBUG_OVERLAP_TYPE = True
# DEBUG_OVERLAP_TYPE = False

class OverlapTypeInterface:
    """
    Overlap computations performed during --rule=UtilizationPlot.

    --overlap-type is one of:
    ResourceOverlap / CategoryOverlap / ResourceSubplot.

    We have a sub-class for each overlap-type.
    """

    def as_js_dict(self, new_overlap):
        operation_overlap = dict()
        for category_key, value in new_overlap.items():
            new_key = self.category_key_as_strs(category_key)
            assert new_key not in operation_overlap
            operation_overlap[new_key] = value
        return operation_overlap

    def dump_json_files(self, sql_reader, new_overlap, new_overlap_metadata, directory, machine_name, process_name, phase_name):
        operation_overlap = self.as_js_dict(new_overlap)
        overlap_metadata_dict = self.as_js_dict(new_overlap_metadata.regions)
        process = sql_reader.process(machine_name=machine_name, process_name=process_name)
        machine = sql_reader.machine(machine_name=machine_name)
        phase = sql_reader.phase(machine_name=machine_name, process_name=process_name, phase_name=phase_name)
        training_progress = sql_reader.training_progress(machine_name=machine_name, process_name=process_name, phase_name=phase_name, allow_none=True)
        self.dump_overlap(
            operation_overlap, overlap_metadata_dict,
            directory=directory,
            process=process,
            machine=machine,
            phase=phase,
            training_progress=training_progress,
            path=self._overlap_json(directory, machine, process, phase),
            venn_js_path=self._overlap_venn_js_json(directory, machine, process, phase))

    def _add_process_md(self, md, process):
        if process is None:
            return

        md['process'] = process.process_name

        # if process.percent_complete is not None:
        #     md['percent_complete'] = process.percent_complete
        #
        # if process.num_timesteps is not None:
        #     md['num_timesteps'] = process.num_timesteps
        #
        # if process.total_timesteps is not None:
        #     md['total_timesteps'] = process.total_timesteps

    def _add_machine_md(self, md, machine):
        if machine is None:
            return

        md['machine'] = machine.machine_name

    def _add_phase_md(self, md, phase):
        if phase is None:
            return

        md['phase'] = phase.phase_name

    def _add_training_progress_md(self, md, training_progress):
        if training_progress is None:
            return

        # Raw training progress fields, for convenience / sanity checks:
        md['training_progress'] = dict(
            total_timesteps=training_progress.total_timesteps,
            start_trace_time_us=training_progress.start_trace_time_us,

            start_percent_complete=training_progress.start_percent_complete,
            start_num_timesteps=training_progress.start_num_timesteps,
            start_training_time_us=training_progress.start_training_time_us,

            end_percent_complete=training_progress.end_percent_complete,
            end_training_time_us=training_progress.end_training_time_us,
            end_num_timesteps=training_progress.end_num_timesteps,
        )

        # Minimal fields needed for total training time extrapolation.
        md['percent_complete'] = training_progress.end_percent_complete - training_progress.start_percent_complete
        md['num_timesteps'] = training_progress.end_num_timesteps - training_progress.start_num_timesteps
        md['total_timesteps'] = training_progress.total_timesteps

    def _add_overlap_region_metadata(
            self, overlap_metadata_dict, md,
            overlap_type=None,
            process=None, machine=None, phase=None, training_progress=None):
        overlap_region = OverlapMetadata.merge_regions(overlap_metadata_dict.values())
        md['start_time_usec'] = overlap_region.start_time_usec
        md['end_time_usec'] = overlap_region.end_time_usec
        md['num_events'] = overlap_region.num_events

        if overlap_type is not None:
            md['overlap_type'] = overlap_type

        self._add_process_md(md, process)
        self._add_machine_md(md, machine)
        self._add_phase_md(md, phase)
        self._add_training_progress_md(md, training_progress)

    def _overlap_json(self, directory, machine_name, process_name, phase_name):
        if type(machine_name) == Machine:
            machine_name = machine_name.machine_name
        if type(process_name) == Process:
            process_name = process_name.process_name
        if type(phase_name) == Phase:
            phase_name = phase_name.phase_name
        return _j(directory, "{OverlapType}{mach}{proc}{phase}.overlap_js.json".format(
            OverlapType=self.overlap_type,
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def _overlap_venn_js_json(self, directory, machine_name, process_name, phase_name):
        if type(machine_name) == Machine:
            machine_name = machine_name.machine_name
        if type(process_name) == Process:
            process_name = process_name.process_name
        if type(phase_name) == Phase:
            phase_name = phase_name.phase_name
        return _j(directory, "{OverlapType}{mach}{proc}{phase}.venn_js.json".format(
            OverlapType=self.overlap_type,
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def dump_overlap(self, operation_overlap, overlap_metadata_dict,
                     directory,
                     process=None, machine=None, phase=None, training_progress=None,
                     path=None, venn_js_path=None):
        md = dict()
        self._add_overlap_region_metadata(overlap_metadata_dict, md,
                                          overlap_type=self.overlap_type,
                                          process=process,
                                          machine=machine,
                                          phase=phase,
                                          training_progress=training_progress)
        if self.should_dump_as_is:
            return self.dump_overlap_as_is(
                operation_overlap, md,
                directory=directory,
                process=process, machine=machine, phase=phase,
                path=path)

        # self._add_overlap_region_metadata(new_overlap_metadata, md)
        return self._dump_overlap(operation_overlap, md,
                                  directory=directory,
                                  machine=machine,
                                  process=process,
                                  phase=phase,
                                  path=path, venn_js_path=venn_js_path)


    def _dump_overlap(self, operation_overlap, metadata,
                      directory=None,
                      machine=None,
                      process=None,
                      phase=None,
                      path=None, venn_js_path=None):
        if path is None:
            assert directory is not None
            path = self._overlap_json(directory, machine, process, phase)
        if venn_js_path is None:
            assert directory is not None
            venn_js_path = self._overlap_venn_js_json(directory, machine, process, phase)
        logger.info("> Dump data for {overlap_type} @ {path}".format(path=path, overlap_type=self.overlap_type))
        dumper = OverlapJSONDumper(operation_overlap, metadata)
        dumper.dump(path)

        if venn_js_path is not None:
            logger.info("> Dump data for {overlap_type} venn.js plot @ {path}".format(path=venn_js_path, overlap_type=self.overlap_type))
            # converter = OverlapJSONToVennConverter(js=js)
            converter = OverlapJSONToVennConverter(path=path)
            venn_js = converter.dump(venn_js_path)
            pprint.pprint({'venn_js':venn_js})

    def dump_overlap_as_is(self, operation_overlap, metadata,
                           directory,
                           machine=None, process=None, phase=None,
                           path=None):
        if path is None:
            path = self._overlap_json(directory, machine, process, phase)
        # if venn_js_path is None:
        #     venn_js_path = self._overlap_venn_js_json(directory, process_name, phase_name)
        logger.info("> Dump data for {overlap_type} @ {path}".format(path=path, overlap_type=self.overlap_type))
        js = js_friendly({
            'overlap': operation_overlap,
            'metadata': metadata,
        })
        do_dump_json(js, path, cls=DecimalEncoder)

    def post_reduce_category_key(self, overlap, overlap_metadata, visible_overhead):
        # Implement custom overlap reduction logic.
        # E.g. for ResourceOverlap, map the fine-grained CPU categories to just one "CPU" category.
        raise NotImplementedError("You must override this.")

    def post_reduce(self, overlap, overlap_metadata, visible_overhead):
        category_key_overlap, category_key_overlap_metadata = self.reduce_to_category_key(overlap, overlap_metadata, visible_overhead)
        new_overlap, new_overlap_metadata = self.post_reduce_category_key(category_key_overlap, category_key_overlap_metadata, visible_overhead)
        return new_overlap, new_overlap_metadata

    def reduce_category_key(self, category_key, visible_overhead, as_cpu_gpu):
        """
        Modular function to bin_events for "reducing" events to CPU/GPU BEFORE OverlapComputation.
        Also, allow ability to "filter-out" events (e.g. category=GPU; needed for CategoryOverlap).

        [Events] ->
        :return:
        """
        # visible_overhead or invisible_overhead:
        #
        #     Whether to "subtract" overhead or not.
        #     visible_overhead:
        #       make profiling overhead visible during rlscope-drill.
        #       Don't subtract; count overhead as extra CPU time.
        #     invisible_overhead:
        #       If false (and calibration files are given), then subtract overhead making it 'invisible' in rlscope-drill.
        #       Subtract; remove CPU-time that is due to CPU overhead.
        #
        #     visible_overhead is determined by a rls-run flag.
        #     However, if the user DOESN'T provide calibration files, all we do is invisible_overhead.
        #     If calibration files are provided, then invisible_overhead ought to be the default.
        #

        non_ops = set()
        for category in category_key.non_ops:
            if category in constants.CATEGORIES_CPU or ( visible_overhead and category in constants.CATEGORIES_PROF ):
                if as_cpu_gpu:
                    # NOTE: profiling types are treated as fine-grained CPU categories.
                    non_ops.add(constants.CATEGORY_CPU)
                else:
                    non_ops.add(category)
            elif category in constants.CATEGORIES_GPU:
                if as_cpu_gpu:
                    non_ops.add(constants.CATEGORY_GPU)
                else:
                    non_ops.add(category)
            elif ( not visible_overhead ) and category in constants.CATEGORIES_PROF:
                # Overhead will get removed during maybe_remove_overhead.
                # Keep the overhead category so we can "see where it is".
                non_ops.add(category)
            else:
                raise RuntimeError("Not sure how to categorize category_key: {msg}.".format(
                    msg=pprint_msg({
                        'non_ops.category': category,
                        'category_key': category_key,
                    })))

        new_category_key = CategoryKey(
            ops=category_key.ops,
            non_ops=non_ops,
            procs=category_key.procs,
        )
        return new_category_key

    # def pre_reduce_cpu_gpu(self, category, event, visible_overhead):
    #     """
    #     Modular function to bin_events for "reducing" events to CPU/GPU BEFORE OverlapComputation.
    #     Also, allow ability to "filter-out" events (e.g. category=GPU; needed for CategoryOverlap).
    #
    #     [Events] ->
    #     :return:
    #     """
    #     # visible_overhead or invisible_overhead:
    #     #
    #     #     Whether to "subtract" overhead or not.
    #     #     visible_overhead   = don't subtract; count overhead as extra CPU time.
    #     #     invisible_overhead = subtract; remove CPU-time that is due to CPU overhead.
    #     #
    #     #     visible_overhead is determined by a rls-run flag.
    #     #     However, if the user DOESN'T provide calibration files, all we do is invisible_overhead.
    #     #     If calibration files are provided, then invisible_overhead ought to be the default.
    #     #
    #     if category in constants.CATEGORIES_CPU or ( visible_overhead and category in constants.CATEGORIES_PROF ):
    #         non_ops = frozenset([constants.CATEGORY_CPU])
    #         ops = frozenset()
    #     elif ( not visible_overhead ) and category in constants.CATEGORIES_PROF:
    #         non_ops = frozenset([category])
    #         ops = frozenset()
    #     elif category == constants.CATEGORY_GPU:
    #         non_ops = frozenset([constants.CATEGORY_GPU])
    #         ops = frozenset()
    #     elif category == constants.CATEGORY_OPERATION:
    #         non_ops = frozenset()
    #         ops = frozenset([event.name])
    #     else:
    #         raise RuntimeError("Not sure how to categorize {cat} into CPU or GPU.".format(
    #             cat=category))
    #     new_key = CategoryKey(ops=ops,
    #                           non_ops=non_ops,
    #                           procs=frozenset([event.process_name]))
    #
    #     # pprint.pprint({
    #     #     'name':'pre_reduce_cpu_gpu',
    #     #     'event':event,
    #     #     'category':category,
    #     #     'new_key': new_key})
    #
    #     return new_key

    def reduce_to_category_key(self, overlap, overlap_metadata, visible_overhead):
        """
        Reduce keys across processes into a single CategoryKey.

        i.e. make it so:
        1. Remove keys that either don't have a process, or don't have a CPU/GPU
        2. Remove association between op's and resource.
           i.e. replace with
           CategoryKey(
             ops=<operations>,
             non_ops=[CPU|GPU],
             procs[p0, p1, ...])

           NOTE: with CategoryKey, it's no longer clear which op/resource belongs to which process.

        For initial overlap computation, the keys are sets-of-sets:

        e.g.
        {'category_key': frozenset({frozenset({'PROC:run_atari_1', 'CPU'}),
                                    frozenset({'PROC:run_atari_0', 'CPU'}),
                                    frozenset({'PROC:run_atari_1', 'GPU'}),
                                    frozenset({'PROC:run_atari_1', 'q_backward'})}

        We want to transform this "set"-based key into:
        CategoryKey(
            ops={q_backward},
            non_ops={CPU, GPU},
            procs={P0, P1},
        )

        NOTE: It's possible for us to need to merge keys when doing this.
        For example, the following similar key from the above example,
        maps to the same reduced key:
        {'category_key': frozenset({frozenset({'PROC:run_atari_0', 'CPU'}),
                                    frozenset({'PROC:run_atari_1', 'CPU'}),
                                    frozenset({'PROC:run_atari_0', 'GPU'}),
                                    frozenset({'PROC:run_atari_0', 'q_backward'})}
        =>
        CategoryKey(
            ops={q_backward},
            non_ops={CPU, GPU},
            procs={P0, P1},
        )

        We got this key by simply switching P0/P1, which is a likely occurrence.

        :param overlap:
        :return:
        """

        def _get(dic, key, default):
            if key not in dic:
                dic[key] = default
            return dic[key]

        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            """
            Delete any time (CPU, GPU, or both) that involves at least one process, 
            and that is NOT captured by an operation.
            
            Q: Where the heck is this un-captured CPU/GPU time coming from in the first place...?
            Seems like we're compensating for a bug earlier in the pipeline....
            """
            proc_ops = dict()
            proc_non_ops = dict()
            for category_key in overlap_key:
                assert len(category_key.procs) == 1
                # proc = category_key.procs[0]
                proc = next(iter(category_key.procs))
                _get(proc_ops, proc, set()).update(category_key.ops)
                _get(proc_non_ops, proc, set()).update(category_key.non_ops)
            skip = False
            skip_proc = None
            # If for every "process_key", the process_key
            #    involves BOTH execution: {CPU, GPU, CPU/GPU},
            #    and an operation: {q_forward, q_backward, ...}
            #   Keep.
            for proc in proc_ops.keys():
                if len(proc_ops[proc]) == 0 or len(proc_non_ops[proc]) == 0:
                    skip = True
                    skip_proc = proc
                    break
            if skip:
                # if self.debug:
                #     logger.info("> DELETE OVERLAP:")
                #     pprint.pprint({
                #         'overlap_key':overlap_key,
                #         'proc':skip_proc,
                #         'ops':proc_ops[proc],
                #         'non_ops':proc_non_ops[proc],
                #     })
                continue

            new_key = _as_category_key(overlap_key)
            if self.debug:
                pprint.pprint({
                    'name':'reduce_to_category',
                    'new_key':new_key,
                    'overlap_key':overlap_key})
            assert len(new_key.ops) > 0
            assert len(new_key.non_ops) > 0
            assert len(new_key.procs) > 0

            if len(new_key.ops) > 1 and not( len(new_key.procs) > 1 ):
                # Operations can only overlap cross-process, not within a single-process
                logger.info("Saw > 1 ops within a single process: {msg}".format(msg=pprint_msg({
                    'ops': new_key.ops,
                    'procs': new_key.procs,
                    'times': times,
                })))
                assert len(new_key.procs) > 1

            add_overlap_with_key(
                new_overlap, new_overlap_metadata, new_key,
                overlap_metadata, overlap_key,
                times, visible_overhead)

            # pprint.pprint({
            #     'overlap.keys()':overlap.keys(),
            # })
            # raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        if self.debug:
            pprint.pprint({
                'reduce_to_category_key.keys': list(new_overlap.keys()),
                'new_overlap_metadata': new_overlap_metadata,
            })

        return new_overlap, new_overlap_metadata

    def reduce_overlap_resource_operation(
        self, overlap, overlap_metadata,
        visible_overhead,
        as_cpu_gpu,
        group_self_overlap=False):
        """
        Reduce keys to pair of operation-types, or a single operation-type.
        (eliminate process, just keep operation-type and execution-type)

        We want to produce overlap that has keys like:

        1. Single-op type:
           e.g.
           <q_forward>: The q_forward operation, not overlapped with any other operation.

        2. Pair of op-types:
           e.g.
           <q_forward, q_forward>: Two or more q_forward operations running simultaneously.
           <q_forward, q_backward>: Two or more q_forward and q_backward operations running simultaneously.

        PSEUDOCODE: Reduce across the "Process<ID>" dimension:

        For keys like <q_forward, P1, P2, ..., [CPU, GPU]>:
            new_overlap[q_forward, q_forward] +=
              overlap[q_forward, P1, P2, ..., [CPU, GPU]]

        For keys like <q_forward, ..., P1, ...>:
            "Operations can only overlap cross-process, not within a single-process"
            Assert: If len(ops) > 1 then len(procs) > 1
            new_overlap[q_forward, ..., [CPU, GPU]] +=
              overlap[q_forward, ..., P1, ..., [CPU, GPU]]

        return new_overlap
        """

        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            if len(overlap_key.ops) == 1 and len(overlap_key.procs) >= 2:
                """
                Single operation type, with overlap.
                
                i.e. operation overlaps with itself across processes.
                e.g. <q_forward, q_forward>
                """
                cat = next(iter(overlap_key.ops))
                if group_self_overlap:
                    # "Group" overlap of an operation with itself across processes
                    # into the operation's time.
                    ops = [cat]
                else:
                    # "Group" overlap of an operation with itself across processes
                    # into a pair of operation-types <op, op>.
                    ops = [cat, cat]
                new_key = CategoryKey(ops=ops,
                                      non_ops=overlap_key.non_ops,
                                      procs=frozenset())
                new_key = self.reduce_category_key(
                    new_key,
                    visible_overhead=visible_overhead,
                    as_cpu_gpu=as_cpu_gpu)
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))
                continue

            if len(overlap_key.ops) >= 1 and len(overlap_key.procs) >= 1:
                """
                Either:
                - Single operation type, no overlap (single with overlap already handled above!)
                - Multi-operation type, with overlap
                """
                new_key = CategoryKey(ops=overlap_key.ops,
                                      non_ops=overlap_key.non_ops,
                                      procs=frozenset())
                new_key = self.reduce_category_key(
                    new_key,
                    visible_overhead=visible_overhead,
                    as_cpu_gpu=as_cpu_gpu)
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))
                continue

            raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        if self.debug:
            total_size = compute_total_size(overlap)
            pprint.pprint({
                'overlap':overlap,
                'new_overlap':new_overlap,
                'total_size':total_size,
                'overlap.keys()':list(overlap.keys()),
                'new_overlap_metadata':new_overlap_metadata,
            })


        return new_overlap, new_overlap_metadata

#
# Helper functions for reduce_overlap_*
#
def _reduce_new_key_like(new_overlap, key, value):
    """
    Add a non-existant key.
    """
    if key not in new_overlap:
        NumberType = type(value)
        new_overlap[key] = NumberType(0.)

def _reduce_add_key(new_overlap, key, value):
    """
    Merge an existing key.
    """
    # assert isinstance(value, numbers.Number)
    if key not in new_overlap:
        _reduce_new_key_like(new_overlap, key, value)
    new_overlap[key] += value

def maybe_remove_overhead(key, visible_overhead):
    if visible_overhead:
        # Keep the overhead and the CPU category it belongs to.
        return key
    prof_categories = constants.CATEGORIES_PROF.intersection(key.non_ops)
    non_ops = key.non_ops
    if len(prof_categories) > 0:
        # cpu_categories = constants.CATEGORIES_CPU.intersection(key.non_ops)
        # if len(cpu_categories) > 0:
        #     # Discard CPU-time that is due to profiling overhead.
        #     # NOTE: constants.CATEGORY_GPU won't get discarded.
        #     non_ops = non_ops.difference(cpu_categories)

        # Discard CPU-time that is due to profiling overhead.
        # NOTE: constants.CATEGORY_GPU won't get discarded.
        non_ops = non_ops.difference(constants.CATEGORIES_CPU)
        # Q: Should we remove the profiling category as well...? I think so yes.
        non_ops = non_ops.difference(constants.CATEGORIES_PROF)
    new_key = CategoryKey(
        ops=key.ops,
        non_ops=non_ops,
        procs=key.procs,
    )
    return new_key

def is_empty_key(category_key):
    """
    We "ignore" an overlap region if:
    - It contains an op (e.g. sample_action), but no resource category (e.g. CPU, GPU)
    - It contains a resource category (e.g. CPU, GPU), but no op (e.g. sample_action)

    :param category_key:
    :return:
    """
    # return len(category_key.ops) == 0 and \
    #        len(category_key.non_ops) == 0
    return len(category_key.ops) == 0 or \
           len(category_key.non_ops) == 0

def add_overlap_with_key(
    new_overlap, new_overlap_metadata, new_key,
    overlap_metadata, overlap_key,
    times, visible_overhead):
    no_overhead_key = maybe_remove_overhead(new_key, visible_overhead)
    if is_empty_key(no_overhead_key):
        # SKIP: no "operations" and no "categories".
        # This happens when we "subtract" CPU-overhead from a CPU-only time.
        # Only thing there may be left are processes.
        return
    _reduce_add_key(new_overlap, no_overhead_key, times)
    new_overlap_metadata.merge_region(no_overhead_key, overlap_metadata.get_region(overlap_key))

class DefaultOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'default'
        self.should_dump_as_is = True
        self.debug = debug or DEBUG_OVERLAP_TYPE

    # def pre_reduce(self, category, event, visible_overhead):
    #     return self.pre_reduce_cpu_gpu(category, event, visible_overhead)

    def as_js_dict(self, new_overlap):
        # def _group_by_ops_resource(self, new_overlap):
        # set(operation categories) -> set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #  <q_forward, q_backward>       <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        operation_overlap = dict()
        for category_key, value in new_overlap.items():
            assert len(category_key.ops) > 0
            assert len(category_key.non_ops) > 0
            assert len(category_key.procs) == 0
            if category_key.ops not in operation_overlap:
                operation_overlap[category_key.ops] = dict()
            operation_overlap[category_key.ops][category_key.non_ops] = value
        return operation_overlap

    def post_reduce_category_key(self, overlap, overlap_metadata, visible_overhead):
        return self.reduce_overlap_resource_operation(
            overlap, overlap_metadata, visible_overhead,
            as_cpu_gpu=False,
            group_self_overlap=False)

class ResourceOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'ResourceOverlap'
        self.should_dump_as_is = False
        self.debug = debug or DEBUG_OVERLAP_TYPE

    # def pre_reduce(self, category, event, visible_overhead):
    #     """
    #     Re-map CPU-like categories to constants.CATEGORY_CPU:
    #         e.g.
    #         constants.CATEGORY_PYTHON -> CategoryKey(non_ops=[constants.CATEGORY_CPU], procs=event.process_name)
    #
    #     Keep constants.CATEGORY_OPERATION events.
    #         constants.CATEGORY_PYTHON -> CategoryKey(ops=[op], procs=event.process_name)
    #
    #     :param category:
    #     :param event:
    #     :return:
    #     """
    #     return self.pre_reduce_cpu_gpu(category, event, visible_overhead)

    def post_reduce_category_key(self, overlap, overlap_metadata, visible_overhead):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        { CategoryKey(op),
          CategoryKey(CPU) }

        ->

        :return:
        """
        """
        Group keys by resource-type (non_ops).
        """
        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            new_key = CategoryKey(ops=frozenset(),
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            new_key = self.reduce_category_key(
                new_key,
                visible_overhead=visible_overhead,
                as_cpu_gpu=True)
            add_overlap_with_key(
                new_overlap, new_overlap_metadata, new_key,
                overlap_metadata, overlap_key,
                times, visible_overhead)

        if self.debug:
            pprint.pprint({
                'ResourceOverlapType.post_reduce_category_key.keys': list(new_overlap.keys()),
                'new_overlap_metadata':new_overlap_metadata,
            })

        return new_overlap, new_overlap_metadata

    def category_key_as_strs(self, category_key):
        # def _group_by_resource(self, new_overlap):
        #     # set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #     #   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        assert len(category_key.ops) == 0
        assert len(category_key.non_ops) > 0
        assert len(category_key.procs) == 0
        return category_key.non_ops

class OperationOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'OperationOverlap'
        self.should_dump_as_is = False
        self.debug = debug or DEBUG_OVERLAP_TYPE

    # def pre_reduce(self, category, event, visible_overhead):
    #     return self.pre_reduce_cpu_gpu(category, event, visible_overhead)

    def post_reduce_category_key(self, overlap, overlap_metadata, visible_overhead):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Remove keys that don't match CPU(?).

        Group keys by operation-type (non_ops).

        :return:
        """
        return self.reduce_overlap_resource_operation(
            overlap, overlap_metadata, visible_overhead,
            as_cpu_gpu=True,
            group_self_overlap=True)

    def _operation_overlap_json(self, directory, machine_name, process_name, phase_name, resources):
        if type(machine_name) == Machine:
            machine_name = machine_name.machine_name
        if type(process_name) == Process:
            process_name = process_name.process_name
        if type(phase_name) == Phase:
            phase_name = phase_name.phase_name
        return _j(directory, "{OverlapType}{mach}{proc}{phase}{resources}.overlap_js.json".format(
            OverlapType=self.overlap_type,
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            resources=resources_suffix(resources),
        ))

    def _operation_overlap_venn_js_json(self, directory, machine_name, process_name, phase_name, resources):
        if type(machine_name) == Machine:
            machine_name = machine_name.machine_name
        if type(process_name) == Process:
            process_name = process_name.process_name
        if type(phase_name) == Phase:
            phase_name = phase_name.phase_name
        return _j(directory, "{OverlapType}{mach}{proc}{phase}{resources}.venn_js.json".format(
            OverlapType=self.overlap_type,
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            resources=resources_suffix(resources),
        ))

    def dump_json_files(self, sql_reader, new_overlap, new_overlap_metadata, directory, machine_name, process_name, phase_name):
        # Q: Why don't we need to call as_js_dict here?
        # JAMES LEFT OFF
        overlap = self.as_js_dict(new_overlap)
        overlap_metadata_dict = self.as_js_dict(new_overlap_metadata.regions)
        process = sql_reader.process(machine_name=machine_name, process_name=process_name)
        machine = sql_reader.machine(machine_name=machine_name)
        phase = sql_reader.phase(machine_name=machine_name, process_name=process_name, phase_name=phase_name)
        training_progress = sql_reader.training_progress(machine_name=machine_name, process_name=process_name, phase_name=phase_name, allow_none=True)
        for resources, op_overlap in overlap.items():
            # operation_overlap = self.as_js_dict(new_overlap)
            if self.debug:
                pprint.pprint({
                    'name':'{OverlapType}.dump_json_files'.format(OverlapType=self.overlap_type),
                    'resources':resources,
                    'op_overlap':op_overlap,
                })
            md = dict()
            md.update({
                'resource_overlap': sorted(resources),
            })
            self._add_overlap_region_metadata(overlap_metadata_dict[resources], md,
                                              overlap_type=self.overlap_type,
                                              process=process,
                                              machine=machine,
                                              phase=phase,
                                              training_progress=training_progress)
            self._dump_overlap(
                op_overlap, md,
                path=self._operation_overlap_json(directory, machine, process, phase, resources),
                venn_js_path=self._operation_overlap_venn_js_json(directory, machine, process, phase, resources))

    def as_js_dict(self, new_overlap):
        # def _group_by_resource_ops(self, new_overlap):
        # set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec
        operation_overlap = dict()
        for combo_key, value in new_overlap.items():
            assert len(combo_key.ops) > 0
            assert len(combo_key.non_ops) > 0
            assert len(combo_key.procs) == 0
            if combo_key.non_ops not in operation_overlap:
                operation_overlap[combo_key.non_ops] = dict()
            operation_overlap[combo_key.non_ops][combo_key.ops] = value
        return operation_overlap

class CategoryOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'CategoryOverlap'
        self.should_dump_as_is = False
        self.debug = debug or DEBUG_OVERLAP_TYPE

    # def pre_reduce(self, category, event, visible_overhead):
    #     """
    #     Modular function to bin_events for "reducing" events to CPU/GPU BEFORE OverlapComputation.
    #     Also, allow ability to "filter-out" events (e.g. category=GPU; needed for CategoryOverlap).
    #
    #     Pre-reduce: keep categories as-is; filter out GPU stuff.
    #     """
    #     if category == constants.CATEGORY_OPERATION:
    #         non_ops = frozenset()
    #         ops = frozenset([event.name])
    #     else:
    #         non_ops = frozenset([category])
    #         ops = frozenset()
    #     new_key = CategoryKey(ops=ops,
    #                           non_ops=non_ops,
    #                           procs=frozenset([event.process_name]))
    #
    #     # if self.debug:
    #     #     pprint.pprint({
    #     #         'name':'{OverlapType}.pre_reduce'.format(OverlapType=self.overlap_type),
    #     #         'event':event,
    #     #         'category':category,
    #     #         'new_key': new_key})
    #
    #     return new_key

    def post_reduce_category_key(self, overlap, overlap_metadata, visible_overhead):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Post-reduce: I don't think we need to do anything here?
        Everything should belong to a single process:
          CategoryKey(ops=key.ops, non_ops=key.non_ops, procs=None)
        """
        # return self.reduce_overlap_resource_operation(
        #     overlap, overlap_metadata, visible_overhead,
        #     group_self_overlap=True)

        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            assert len(overlap_key.procs) == 1
            # PROBLEM: Now that we subtract profiling CPU time, we can end up with
            # non_ops being empty...
            # This SHOULD have been possible in the past...we'd usually count is as nothing I think...?
            # _CategoryKey(ops=('sample_action',), non_ops=frozenset(), procs=frozenset({'ppo2_PongNoFrameskip-v4'}))
            assert len(overlap_key.ops) == 1
            assert len(overlap_key.non_ops) >= 1
            new_key = CategoryKey(ops=overlap_key.ops,
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            new_key = self.reduce_category_key(
                new_key,
                visible_overhead=visible_overhead,
                as_cpu_gpu=False)
            add_overlap_with_key(
                new_overlap, new_overlap_metadata, new_key,
                overlap_metadata, overlap_key,
                times, visible_overhead)

        if self.debug:
            total_size = compute_total_size(overlap)
            pprint.pprint({
                'overlap':overlap,
                'new_overlap':new_overlap,
                'total_size':total_size,
                '{OverlapType}.post_reduce_category_key.keys'.format(OverlapType=self.overlap_type):
                    list(new_overlap.keys()),
                'new_overlap_metadata':new_overlap_metadata,
            })

        return new_overlap, new_overlap_metadata

    def _category_overlap_json(self, directory, machine_name, process_name, phase_name, ops, resources):
        if type(machine_name) == Machine:
            machine_name = machine_name.machine_name
        if type(process_name) == Process:
            process_name = process_name.process_name
        if type(phase_name) == Phase:
            phase_name = phase_name.phase_name
        return _j(directory, "{OverlapType}{mach}{proc}{phase}{ops}{resources}.overlap_js.json".format(
            OverlapType=self.overlap_type,
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            ops=ops_suffix(ops),
            resources=resources_suffix(resources),
        ))

    def _category_overlap_venn_js_json(self, directory, machine_name, process_name, phase_name, ops, resources):
        if type(machine_name) == Machine:
            machine_name = machine_name.machine_name
        if type(process_name) == Process:
            process_name = process_name.process_name
        if type(phase_name) == Phase:
            phase_name = phase_name.phase_name
        return _j(directory, "{OverlapType}{mach}{proc}{phase}{ops}{resources}.venn_js.json".format(
            OverlapType=self.overlap_type,
            mach=machine_suffix(machine_name),
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            ops=ops_suffix(ops),
            resources=resources_suffix(resources),
        ))

    def dump_json_files(self, sql_reader, new_overlap, new_overlap_metadata, directory, machine_name, process_name, phase_name):
        """
        For op in ops(process, phases):
          key = CategoryKey[ops=[op]]
          # SHOULD exist; all keys should have 1 op also
          # (no overlap with a process)
          yield Key -> [str(non_op) for non_op in key.non_ops]
          # CategoryOverlap{process}{phase}{op}.json

        :param new_overlap:
        :param directory:
        :param process_name:
        :param phase_name:
        :return:
        """
        overlap = self.as_js_dict(new_overlap)
        overlap_metadata_dict = self.as_js_dict(new_overlap_metadata.regions)
        process = sql_reader.process(machine_name=machine_name, process_name=process_name)
        machine = sql_reader.machine(machine_name=machine_name)
        phase = sql_reader.phase(machine_name=machine_name, process_name=process_name, phase_name=phase_name)
        if self.debug:
            pprint.pprint({
                'name':"{OverlapType}.dump_json_files".format(OverlapType=self.overlap_type),
                'overlap':overlap,
            })
        for ops, resource_to_category_overlap in overlap.items():
            assert len(ops) == 1
            op = next(iter(ops))
            for resources, category_overlap in resource_to_category_overlap.items():
                md = dict()
                self._add_overlap_region_metadata(overlap_metadata_dict[ops][resources], md,
                                                  overlap_type=self.overlap_type,
                                                  process=process,
                                                  machine=machine,
                                                  phase=phase)
                md.update({
                    # TODO: For now we only are computed category overlap for CPU;
                    # in the future we may have more fine-grained GPU categories.
                    # 'resource_overlap': sorted([constants.CATEGORY_CPU]),
                    'resource_overlap': resources,
                    'operation': op,
                })
                self._dump_overlap(
                    category_overlap, md,
                    # No directory; use paths specified by path/venn_js_path
                    path=self._category_overlap_json(directory, machine, process, phase, ops, resources),
                    venn_js_path=self._category_overlap_venn_js_json(directory, machine, process, phase, ops, resources))

    def as_js_dict(self, new_overlap):
        # def _group_by_resource_ops(self, new_overlap):
        # set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #    <CPU>, <GPU>, <CPU, GPU>       <q_forward, q_backward>           0.001 sec

        # set(non-operation categories) -> set(operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #    <CPU>, <GPU>, <CPU, GPU>      <q_forward>, <q_backward>           0.001 sec


        # non_ops={TFlow C, CUDA API CPU, GPU}
        # => CPU={TFlow C, CUDA API CPU}
        #    GPU={GPU}
        # resource_key = {CPU, GPU}
        #
        # non_ops={Python}
        # => CPU={Python}
        #    GPU={GPU}
        # resource_key = {CPU, GPU}
        operation_overlap = dict()
        for combo_key, value in new_overlap.items():
            cpus_gpus = split_cpu_gpu_categories(combo_key)
            resource_key = set()
            if len(cpus_gpus.cpus) > 0:
                resource_key.add(constants.CATEGORY_CPU)
            if len(cpus_gpus.gpus) > 0:
                resource_key.add(constants.CATEGORY_GPU)
            resource_key = frozenset(resource_key)
            assert len(combo_key.ops) > 0
            assert len(combo_key.non_ops) > 0
            assert len(combo_key.procs) == 0
            if combo_key.ops not in operation_overlap:
                operation_overlap[combo_key.ops] = dict()
            if resource_key not in operation_overlap[combo_key.ops]:
                operation_overlap[combo_key.ops][resource_key] = dict()
            categories_key = cpus_gpus.cpus.union(cpus_gpus.gpus)
            assert categories_key not in operation_overlap[combo_key.ops][resource_key]
            operation_overlap[combo_key.ops][resource_key][categories_key] = value
        return operation_overlap

class ResourceSubplotOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'ResourceSubplot'
        self.should_dump_as_is = False
        self.debug = debug or DEBUG_OVERLAP_TYPE

    # def pre_reduce(self, category, event, visible_overhead):
    #     return self.pre_reduce_cpu_gpu(category, event, visible_overhead)

    def post_reduce_category_key(self, overlap, overlap_metadata, visible_overhead):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Group keys by resource-type (non_ops).
        :return:
        """
        # def reduce_overlap_ResourceSubplot(
        #         self, overlap,
        #         categories, operation_types, proc_types):
        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            assert len(overlap_key.non_ops) > 0

            # Just {CPU}
            # Add time to CPU, add time to Total.
            #
            # Just {GPU}
            # Add time to GPU, add time to Total.
            #
            # Just {CPU, GPU}
            # Add time to CPU, add time to GPU, add time to Total.

            new_key = CategoryKey(ops=frozenset(),
                                  non_ops=frozenset([constants.CATEGORY_TOTAL]),
                                  procs=frozenset())
            _add_key(new_overlap, new_key, times)
            new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

            cpu_gpu_key = self.reduce_category_key(overlap_key, visible_overhead, as_cpu_gpu=True)
            for resource_type in cpu_gpu_key.non_ops:
                # NOTE: This is sort of hacky;
                # we AREN'T outputting disjoint overlap regions here;
                # instead we are outputting an entire "set" including its overlaps:
                # i.e.
                # CPU   = [CPU only time] + [CPU overlapped with GPU time]
                # GPU   = [GPU only time] + [GPU overlapped with GPU time]
                # Total = [CPU only time] + [GPU only time] + [CPU overlapped with GPU time]
                new_key = CategoryKey(ops=frozenset(),
                                      non_ops=frozenset([resource_type]),
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

        return new_overlap, new_overlap_metadata

    def category_key_as_strs(self, category_key):
        # def _group_by_resource(self, new_overlap):
        # set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        assert len(category_key.ops) == 0
        assert len(category_key.non_ops) > 0
        assert len(category_key.procs) == 0
        return category_key.non_ops

OVERLAP_TYPE_TO_KLASS = {
    'ResourceOverlap':ResourceOverlapType,
    'default':DefaultOverlapType,
    'CategoryOverlap':CategoryOverlapType,
    'OperationOverlap':OperationOverlapType,
    'ResourceSubplot':ResourceSubplotOverlapType,
}
def overlap_type_to_instance(overlap_type, debug=False):
    OverlapType = OVERLAP_TYPE_TO_KLASS[overlap_type]
    return OverlapType(debug)

class OverlapJSONDumper:
    def __init__(self, overlap, metadata):
        # "overlap region"     "size of overlap"
        # set([CPU, GPU])  ->  Number
        self.overlap = overlap
        self.metadata = metadata

    def dump(self, path):
        js = js_friendly({
            'overlap': self.overlap,
            'metadata': self.metadata,
        })
        do_dump_json(js, path, cls=DecimalEncoder)
        return js

class OverlapJSONToVennConverter:
    def __init__(self, js=None, path=None, overlap=None, metadata=None):
        # "overlap region"     "size of overlap"
        # set([CPU, GPU])  ->  Number
        assert js is not None or path is not None or ( overlap is not None and metadata is not None )

        self.js = None
        self.path = None
        self.overlap = None
        self.metadata = None

        if js is not None or path is not None:
            if path is not None:
                with open(path, 'r') as f:
                    js = json.load(f)
            self.js = js
            self.path = path
            self.overlap = self._reconstruct_overlap()
            self.metadata = self.js['metadata']
        else:
            self.overlap = overlap
            self.metadata = metadata

    def _reconstruct_overlap(self):
        overlap = dict()

        for pairs, size in self.js['overlap']:
            overlap[tuple(sorted(pairs))] = float(size)
        return overlap

    def _compute_set_sizes(self):
        # Convert self.overlap into venn_js sizes.
        V = overlap_as_venn_dict(self.overlap)
        return V
        # set_to_size = dict()
        # for overlap_region, size in self.overlap.items():
        #     for set_region in overlap_region:
        #         assert type(set_region) == str
        #         if set_region not in set_to_size:
        #             set_to_size[set_region] = as_type(0., type(size))
        #         set_to_size[set_region] += size
        # return set_to_size

    def convert(self):
        """
        venn_music_data = [
            {"sets": [0], "label": "Radiohead", "size": 77348},
            {"sets": [1], "label": "Thom Yorke", "size": 5621},
            ...
            {"sets": [0, 1], "size": 4832},
            ...
        ]
        """
        js = {
            'venn':[],
            'metadata':self.metadata,
        }
        venn_js = js['venn']

        set_to_size = self._compute_set_sizes()

        label_to_id = dict()
        labels = set()
        for overlap, size in self.overlap.items():
            assert type(overlap) in [list, tuple, set, frozenset]
            labels.update(overlap)
        labels = sorted(labels)
        for i, label in enumerate(labels):
            label_to_id[label] = i

        def as_sets(overlap):
            sets_ids = [label_to_id[category] for category in overlap]
            sets_ids.sort()
            return sets_ids

        for label, size in set_to_size.items():
            venn_set = {
                "sets": as_sets([label]),
                "size": size,
                "label": label,
            }
            venn_js.append(venn_set)

        for overlap, size in self.overlap.items():
            if len(overlap) == 1:
                # Single "set region" doesn't include overlap with other sets.
                # "Set region" is handled in for-loop above this one.
                continue
            venn_set = {
                "sets": as_sets(overlap),
                "size": size,
            }
            venn_js.append(venn_set)

        # Make the shorter (in particular, single-element) venn_sets appear first.
        # venn_sets within the same length are ordered based on lexicographic order.
        venn_js.sort(key=lambda venn_set: (len(venn_set['sets']), venn_set['sets']))

        return js

    def dump(self, path):
        js = self.convert()
        do_dump_json(js, path)
        return js

# From TensorFlow code base:
#
# bool CountAsAcceleratorTime(const string& device) {
# return device.find("stream:all") != device.npos;
# }
# bool CountAsCPUTime(const string& device) {
# return RE2::FullMatch(device,
#                       ".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+");
# }

def split_combo_key(combo_key, operation_types, proc_types):
    assert not isinstance(combo_key, _CategoryKey)
    op_categories = set()
    proc_categories = set()
    non_op_categories = set()
    for category in combo_key:
        if ( proc_types is not None and category in proc_types ) or match_proc_category(category):
            proc_categories.add(category)
        elif ( operation_types is not None and category in operation_types ) or not match_cpu_gpu_category(category):
            # If operation_types isn't provided, assume it's an operation (unless its CPU or GPU).
            op_categories.add(category)
        else:
            non_op_categories.add(category)
    return frozenset(op_categories), frozenset(non_op_categories), frozenset(proc_categories)

def _as_category_key(overlap_key):
    op_categories = set()
    non_op_categories = set()
    proc_categories = set()
    for proc_key in overlap_key:
        if len(proc_key.ops) > 0:
            # CPU or GPU.
            # DON'T keep proc information from this.
            # We only keep proc information that tells us about __operation__ overlap across processes,
            assert len(proc_key.non_ops) == 0
            assert len(proc_key.procs) > 0
            proc_categories.update(proc_key.procs)
        op_categories.update(proc_key.ops)
        non_op_categories.update(proc_key.non_ops)

    new_key = CategoryKey(op_categories, non_op_categories, proc_categories)
    return new_key

# def reduce_category_keys(overlap):
#
#     new_overlap = dict()
#     for overlap_key, times in overlap.items():
#         new_key = _as_category_key(overlap_key=overlap_key)
#         assert len(new_key.ops) > 0 or \
#                len(new_key.non_ops) > 0
#         # assert len(new_key.procs) > 0
#
#         if len(new_key.ops) > 1:
#             # Operations can only overlap cross-process, not within a single-process
#             assert len(new_key.procs) > 1
#
#         _reduce_add_key(new_overlap, new_key, times)
#
#         # pprint.pprint({
#         #     'overlap.keys()':overlap.keys(),
#         # })
#         # raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))
#
#     return new_overlap

#
# Helper functions for reduce_overlap_*
#
def _new_key_like(new_overlap, key, value):
    """
    Add a non-existant key.
    """
    if key not in new_overlap:
        NumberType = type(value)
        new_overlap[key] = NumberType(0.)
def _add_key(new_overlap, key, value):
    """
    Merge an existing key.
    """
    # assert isinstance(value, numbers.Number)
    if key not in new_overlap:
        _new_key_like(new_overlap, key, value)
    new_overlap[key] += value
def _merge_key(new_overlap, key, merge_values, new_value):
    """
    Merge an existing key.

    e.g.

    merge_values(key, old_value, new_value):
        # Add
        return old_value + new_value

    merge_values(key, old_value, new_value):
        # Subtract from old value
        return old_value - new_value
    """
    # assert isinstance(value, numbers.Number)
    if key not in new_overlap:
        _new_key_like(new_overlap, key, new_value)

    old_value = new_overlap[key]
    merged_value = merge_values(old_value, new_value)
    new_overlap[key] = merged_value

def compute_total_size(overlap):
    return np.sum(list(overlap.values()))

def merge_ComputeOverlap(overlap1, overlap2):
    def merge_func(set_of_category_key, overlap1_time, overlap2_time):
        return overlap1_time + overlap2_time
    return merge_dicts(overlap1, overlap2, merge_func)

def merge_dicts(dict1, dict2, merge_func):
    """
    Merge two dictionaries.
    Use merge_func to resolve key-conflicts.
    merge_func(k, value1, value2) returns the result of merging value1/value2.

    Returns a dictionary dic where dic.keys() == Union[dict1.keys(), dict2.keys()]

    :param dict1:
    :param dict2:
    :param merge_func:
        [value(dict1), value(dict2)] -> value
    :return:
    """
    dic = dict(dict1)
    for k, v in dict2.items():
        if k not in dic:
            dic[k] = v
        else:
            dic[k] = merge_func(k, dic[k], v)
    return dic

def test_merge_adjacent_events():

    from rlscope.test.test_util import sec, T, flatten_category_times as flat

    def test_01_merge_adj_events():
        events = [T(1, 6), T(2, 6), T(3, 7), T(4, 6)]
        got = merge_adjacent_events(events)
        expect = [T(1, 7)]
        assert got == expect
    test_01_merge_adj_events()


##
## Event overlap unit tests.
##
"""
Environment variables that effect unit-tests:
    RLSCOPE_DEBUG_UNIT_TESTS=[0/1]
        Default: 0
        Enable more verbose debugging information during unit-tests.
        
    RLSCOPE_USE_NUMBA=[0/1]
        Default: 0
        Use numbafied event overlap computation.
        
    NUMBA_DISABLE_JIT=[0/1]
        Default: 0
        numba library specific option; 
        when turned off, disables all JIT compilation (i.e. code runs as regular python code).
        Makes it easier to debug segfaults.
        For details: 
            https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#disabling-jit-compilation
            
Common usage:

    # cd to root directory of rlscope repo checkout.
    $ cd ~/clone/rlscope

    # To run numbaified code with JIT compilation enabled:
    $ RLSCOPE_USE_NUMBA=1 pytest -vv -s --pdb rlscope/parser/tfprof.py

    # To run numbaified code WITHOUT JIT enabled (e.g. if you segfault and don't know why):
    $ RLSCOPE_USE_NUMBA=1 NUMBA_DISABLE_JIT=1 pytest -vv -s --pdb rlscope/parser/tfprof.py
    
    # To run un-numbafied code:
    $ pytest -vv -s --pdb rlscope/parser/tfprof.py


"""

if py_config.USE_NUMBA:

    from rlscope.test import test_util
    class TestOverlap:

        def T(self, *args, **kwargs):
            return test_util.T(*args, **kwargs)

        def flat(self, *args, **kwargs):
            return test_util.flatten_category_times(*args, **kwargs)

        def sec(self, *args, **kwargs):
            return test_util.sec(*args, **kwargs)

        def test_01_complete(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            # Q: Any way to automate this by checking if a pytest is running...?
            py_config.IS_UNIT_TEST = True
            category_times = {
                'c1':[
                    [
                        T(3, 4), T(8, 10),
                    ],
                    [
                        T(3.5, 7),
                    ],
                ],
                'c2':[
                    [
                        T(1, 4), T(6, 9),
                    ],
                ],
                'c3':[
                    [
                        T(2, 3), T(4, 5), T(7, 8),
                    ],
                    [
                        T(3, 4), T(11, 12),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            # overlap = compute_overlap.get_category_times()
            # overlap_metadata = compute_overlap.get_overlap_metadata()

            # compute_overlap.compute_merge()
            # got = compute_overlap.get_merged_categories()

            expect = {
                'c1':[
                    T(3, 7), T(8, 10),
                ],
                'c2':[
                    T(1, 4), T(6, 9),
                ],
                'c3':[
                    T(2, 5), T(7, 8), T(11, 12),
                ],
            }
            assert got == expect

            # compute_overlap.compute()
            compute_overlap.compute_times()
            got = compute_overlap.get_category_times()
            expect = {
                frozenset({'c1'}):sec(2),
                frozenset({'c2'}):sec(1),
                frozenset({'c3'}):sec(1),
                frozenset({'c1', 'c2'}):sec(2),
                frozenset({'c1', 'c3'}):sec(1),
                frozenset({'c2', 'c3'}):sec(2),
                frozenset({'c1', 'c2', 'c3'}):sec(1),
            }
            assert got == expect

        def test_02_overlaps_with(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            py_config.IS_UNIT_TEST = True
            category_times = {
                'c1':[
                    [
                        T(3, 4), T(8, 10),
                    ],
                    [
                        T(3.5, 7),
                    ],
                ],
                'c2':[
                    [
                        T(1, 4), T(6, 9),
                    ],
                ],
                'c3':[
                    [
                        T(2, 3), T(4, 5), T(7, 8),
                    ],
                    [
                        T(3, 4), T(11, 12),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), overlaps_with=['c1'], debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            expect = {
                frozenset({'c1'}):sec(2),
                frozenset({'c1', 'c2'}):sec(2),
                frozenset({'c1', 'c3'}):sec(1),
                frozenset({'c1', 'c2', 'c3'}):sec(1),
            }
            assert got == expect

        def test_03_error_partial_overlap(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            py_config.IS_UNIT_TEST = True
            category_times = {
                'c1':[
                    [
                        T(3, 5), T(4, 6),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            expect = {
                frozenset({'c1'}):sec(3),
                # frozenset({'c1', 'c2'}):sec(2),
                # frozenset({'c1', 'c3'}):sec(1),
                # frozenset({'c1', 'c2', 'c3'}):sec(1),
            }
            # expect = {
            #     frozenset({'c1'}):sec(2),
            #     frozenset({'c1', 'c2'}):sec(2),
            #     frozenset({'c1', 'c3'}):sec(1),
            #     frozenset({'c1', 'c2', 'c3'}):sec(1),
            # }
            assert got == expect

        def test_04_error_full_overlap(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            py_config.IS_UNIT_TEST = True
            category_times = {
                'c1':[
                    [
                        T(3, 6), T(4, 5),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            # expect = {
            #     frozenset({'c1'}):sec(2),
            #     frozenset({'c1', 'c2'}):sec(2),
            #     frozenset({'c1', 'c3'}):sec(1),
            #     frozenset({'c1', 'c2', 'c3'}):sec(1),
            # }
            # assert got == expect

        def test_05_error_duplicate_overlap(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            py_config.IS_UNIT_TEST = True
            category_times = {
                'c1':[
                    [
                        T(3, 6), T(3, 6),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            # expect = {
            #     frozenset({'c1'}):sec(2),
            #     frozenset({'c1', 'c2'}):sec(2),
            #     frozenset({'c1', 'c3'}):sec(1),
            #     frozenset({'c1', 'c2', 'c3'}):sec(1),
            # }
            # assert got == expect

        def test_06_error_not_sorted_by_end_time(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            py_config.IS_UNIT_TEST = True
            category_times = {
                'c1':[
                    [
                        # T(3, 6), T(3, 5),
                        # T(3, 5), T(3, 6),
                        # T(2, 5), T(3, 6),
                        T(3, 6), T(2, 5),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            # expect = {
            #     frozenset({'c1'}):sec(2),
            #     frozenset({'c1', 'c2'}):sec(2),
            #     frozenset({'c1', 'c3'}):sec(1),
            #     frozenset({'c1', 'c2', 'c3'}):sec(1),
            # }
            # assert got == expect

        def test_07_overlapping_sorted_events(self):
            T = self.T
            flat = self.flat
            sec = self.sec
            py_config.IS_UNIT_TEST = True

            category_times = {
                'c1':[
                    [
                        # [1..7]
                        T(1, 6), T(2, 6), T(3, 7), T(4, 6)
                    ],
                ],
                'c2':[
                    [
                        T(4, 5),
                    ],
                ],
            }
            compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)

            compute_overlap.compute()
            got = compute_overlap.get_category_times()
            expect = {
                frozenset({'c1'}):sec(5),
                frozenset({'c1', 'c2'}):sec(1),
            }
            assert got == expect

        # def test_08_overlapping_sorted_events(self):
        #     """
        #     This test has events within a single category that overlap.
        #
        #     For example, c1:
        #             1  2  3  4  5  6  7  8  9  10
        #             |  |  |  |  |  |  |  |  |  |
        #
        #             [              ]
        #      c1        [           ]
        #                   [           ]
        #                      [     ]
        #
        #                      [][]
        #      c2                 [           ]
        #                         [        ]
        #
        #      c3                 [           ]
        #
        #     Ideally, the algorithm would process the event trace as if the event-trace has no overlaps:
        #
        #             1  2  3  4  5  6  7  8  9  10
        #             |  |  |  |  |  |  |  |  |  |
        #
        #      c1     [                 ]
        #
        #      c2              [              ]
        #
        #      c3                 [           ]
        #
        #      Technically we can prevent this type of scenario in the input by pre-processing events
        #      to eliminate self-overlap.
        #
        #     :return:
        #     """
        #     from rlscope.test.test_util import sec, T, flatten_category_times as flat
        #     py_config.IS_UNIT_TEST = True
        #     # Q: What if start times match but end times are unordered?
        #     # Q: WHY would this EVER happen in our data though...?
        #     #    It CAN if concurrent events get "shuffled" into the same category (for some reason).
        #     #    Perhaps this could happen with CPU/GPU?
        #
        #     category_times = {
        #         'c1':[
        #             [
        #                 # [1..7] 6
        #                 T(1, 6), T(2, 6), T(3, 7), T(4, 6)
        #             ],
        #         ],
        #         'c2':[
        #             [
        #                 # [4..9] 5
        #                 T(4, 4.5), T(4.5, 5), T(5, 9), T(5, 8)
        #             ],
        #         ],
        #         'c3':[
        #             [
        #                 # [5..9] 4
        #                 T(5, 9),
        #             ],
        #         ],
        #     }
        #     compute_overlap = ComputeOverlap(flat(category_times), debug=py_config.RLSCOPE_DEBUG_UNIT_TESTS)
        #
        #     compute_overlap.compute()
        #     got = compute_overlap.get_category_times()
        #     expect = {
        #         # [1..4]
        #         frozenset({'c1'}):sec(3),
        #         # [4..5]
        #         frozenset({'c1', 'c2'}):sec(1),
        #         # [5..7]
        #         frozenset({'c1', 'c2', 'c3'}):sec(2),
        #         # [7..9]
        #         frozenset({'c2', 'c3'}):sec(2),
        #     }
        #     assert got == expect

def test_split():

    xs = list(range(1, 10+1))

    actual = list(split_list(xs, 3))
    expect = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9, 10],
    ]
    assert actual == expect

    actual = list(split_list(xs, 11))
    expect = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [],
    ]
    assert actual == expect

def merge_adjacent_events(events, inplace=False, debug=False):
    """
    Given a list of events sorted by event.start_time_usec,
    merge events that "overlap"

    :param events:
    :return:
    """
    if len(events) == 0:
        return []

    def get_event(event):
        if inplace:
            return event
        else:
            return event.copy()

    new_events = [get_event(events[0])]

    if debug:
        pprint.pprint(new_events)

    for event in events[1:]:
        # In case of overlap, it may look like any of:
        #
        # case 1: subsume
        # [ new_events[-1] ]
        #       [ event ]
        #
        # case 2: partial overlap
        # [ new_events[-1] ]
        #       [ event        ]
        #
        # [ new_events[-1] ]
        #       [ event    ]
        #
        # In case of NO overlap, it may look like:
        #
        # case 3:
        # [ new_events[-1] ]
        #                   [ event ]
        #
        # [ new_events[-1] ]
        #                       [ event ]
        assert new_events[-1].start_time_usec <= event.start_time_usec

        if new_events[-1].start_time_usec <= event.start_time_usec <= event.end_time_usec <= new_events[-1].end_time_usec:
            # case 1: subsume
            if debug:
                logger.info("Subsume: {e1} subsumes {e2}".format(e1=new_events[-1], e2=event))
            new_events[-1].set_start_end(
                start_usec=new_events[-1].start_usec,
                end_usec=max(new_events[-1].end_usec, event.end_usec),
            )
        elif new_events[-1].start_time_usec <= event.start_time_usec <= new_events[-1].end_time_usec <= event.end_time_usec:
            # case 2: partial overlap
            if debug:
                logger.info("Partial: {e1} partial {e2}".format(e1=new_events[-1], e2=event))
            new_events[-1].set_start_end(
                start_usec=new_events[-1].start_usec,
                end_usec=max(new_events[-1].end_usec, event.end_usec),
            )
        else:
            # case 3: no overlap
            if debug:
                logger.info("No-overlap: {e1} no-overlap {e2}".format(e1=new_events[-1], e2=event))
            new_events.append(get_event(event))

        if debug:
            pprint.pprint(new_events)

    return new_events

