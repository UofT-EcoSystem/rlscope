import re
from decimal import Decimal
import numbers
import time
import sys
import os
import csv
import textwrap
import pprint
from io import StringIO
import itertools
import json
import codecs
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from parser.common import *
from parser.stats import Stats
# from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

from parser.stats import KernelTime, category_times_add_time

from parser.trace_events import TraceEventsDumper, dump_category_times

from parser.db import SQLCategoryTimesReader, sql_input_path, sql_get_source_files

from parser.readers import TFProfCategoryTimesReader, \
    DEFAULT_group_by_device, \
    DEFAULT_ignore_categories, \
    DEFAULT_debug \

class ComputeOverlap:
    DEBUG = True
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

        Typically overlaps_with = [CATEGORY_OPERATION], so we only keep execution time
        that happened during an operation.
    """
    def __init__(self,
                 category_times,
                 overlaps_with=None,
                 check_key=None,
                 debug=False,
                 show_progress=False):
        self.debug = ComputeOverlap.DEBUG and debug
        self.check_key = check_key
        self.show_progress = show_progress
        self.overlaps_with = overlaps_with
        if self.overlaps_with is not None:
            self.overlaps_with = set(self.overlaps_with)
            for category in self.overlaps_with:
                assert category in category_times.keys()
        category = list(category_times.keys())[0]
        if type(category_times[category][0]) == list:
            self.category_times = self._flatten_category_times(category_times)
        else:
            # It's already flattened
            self.category_times = category_times
        self.category_times = self._sort_category_times(self.category_times)

    def compute(self):
        start_merge_t = time.time()
        self.compute_merge()
        end_merge_t = time.time()
        sec_merge = end_merge_t - start_merge_t
        if self.debug:
            print("> {klass}.compute_merge took {sec} seconds".format(
                klass=self.__class__.__name__,
                sec=end_merge_t))
        start_compute_t = end_merge_t
        self.compute_times()
        end_compute_t = time.time()
        sec_compute = end_compute_t - start_compute_t
        if self.debug:
            print("> {klass}.compute_times took {sec} seconds".format(
                klass=self.__class__.__name__,
                sec=sec_compute))

    def compute_merge(self):
        self.merged_category_times = self._merge_category_times(self.category_times)

    def compute_times(self):
        # set(c1, ..., cn) -> time in seconds
        self.times = self._compute_overlap(self.merged_category_times)

    def get_category_times(self):
        return self.times

    def get_merged_categories(self):
        return self.merged_category_times

    def _flatten_category_times(self, category_times):
        new_category_times = dict()
        for category in category_times.keys():
            all_times = []
            for times in category_times[category]:
                all_times.extend(times)
            new_category_times[category] = all_times
        return new_category_times

    def _sort_category_times(self, category_times):
        new_category_times = dict()
        for category in category_times:
            new_category_times[category] = sorted(category_times[category], key=lambda ktime: ktime.start_time_usec)
        return new_category_times

    def _merge_category_times(self, category_times):
        merged_category_times = dict()
        for category in category_times.keys():
            times = category_times[category]
            # if self.debug and category == 'c3':
            #     import ipdb; ipdb.set_trace()
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

    def _compute_overlap(self, category_times):
        return compute_overlap_single_thread(
            category_times,
            self.overlaps_with,
            self.check_key,
            self.debug,
            self.show_progress)

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
        # for k in list(self.by_key.keys()):
        #     self.by_key[k] = ListWrapper(self.by_key[k])
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

        # if len(self.by_key[category]) > 0:
        #     ktimes.append((category, self.by_key[category][-1]))
        # category, ktime = min(ktimes, key=lambda time_ktime: self.key(time_ktime[1]))

        # ktimes = []
        # for category in self.by_key.keys():
        #     if len(self.by_key[category]) > 0:
        #         ktimes.append((category, self.by_key[category][-1]))
        # category, ktime = min(ktimes, key=lambda time_ktime: self.key(time_ktime[1]))
        # return category, ktime

        return min_category, min_ktime

    def has_times_left(self):
        # return any(len(ctimes_for_category) > 0 for ctimes_for_category in self.by_key.values())
        return self._num_events > 0

    def count_times_left(self):
        return self._num_events

    def _count_times_left(self):
        return sum(len(ctimes_for_category) for ctimes_for_category in self.by_key.values())

    def pop_time(self, category):
        # del self.by_key[category][0]
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

def compute_overlap_single_thread(
    category_times,
    overlaps_with=None,
    check_key=None,
    debug=False,
    show_progress=False):
    # categories = set(category_times.keys())
    # JAMES TODO: compute the progress of this function... I think it takes forever with minigo

    start_key = lambda ktime: ktime.start_time_usec
    reversed_start_key = lambda ktime: - ktime.start_time_usec
    end_key = lambda ktime: ktime.end_time_usec
    reversed_end_key = lambda ktime: - ktime.end_time_usec

    by_start = CategoryTimesWrapper(category_times, start_key, reversed_start_key, 'start')
    by_end = CategoryTimesWrapper(category_times, end_key, reversed_end_key, 'end')

    cur_categories = set()
    def pop_start(category):
        by_start.pop_time(category)
        cur_categories.add(category)
    def pop_end(category):
        by_end.pop_time(category)
        if category not in cur_categories:
            import ipdb; ipdb.set_trace()
        cur_categories.remove(category)

    times = dict()

    min_category, min_ktime = by_start.min_time()
    start_or_end = by_start
    curtime = by_start.key(min_ktime)
    pop_start(min_category)

    if debug:
        print("> Start computing overlap; choose initial start (curtime)")
        pprint.pprint({
            'min_category': min_category,
            'min_ktime': min_ktime,
            'start_or_end': start_or_end.name,
            'curtime': curtime,
        })
        print()

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

        if len(cur_categories) > 0:
            # Don't bother recording empty gaps between times.
            categories_key = frozenset(cur_categories)

            # For debugging; useful to check for overlaps we don't expect to occur.
            # For example, overlap across CATEGORY_OPERATION's within a single process.
            if check_key is not None:
                check_key(categories_key)

            if categories_key not in times:
                NumberType = type(time_chunk)
                times[categories_key] = NumberType(0.)
            times[categories_key] += time_chunk

        if debug:
            pprint.pprint({
                'min_category': min_category,
                'min_ktime': min_ktime,
                'start_or_end': start_or_end.name,
                'update':"times[{s}] += {t}".format(s=categories_key, t=time_chunk),
            })
            print()

        if start_or_end.type_code == CategoryTimesWrapper.START:
            if debug:
                pprint.pprint({'cur_categories':cur_categories,
                               'add': min_category,
                               'curtime': next_time})
                print()
            pop_start(min_category)
        else:
            if debug:
                pprint.pprint({'cur_categories':cur_categories,
                               'remove': min_category,
                               'curtime': next_time})
                print()
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

    for categories_key in list(times.keys()):
        # We may get artificial overlaps even if two categories are synchronous,
        # if the next category starts exactly when the last one ends.
        if times[categories_key] == 0:
            del times[categories_key]

    if overlaps_with is not None:
        del_keys = []
        for categories_key in times.keys():
            if len(overlaps_with.intersection(categories_key)) == 0:
                del_keys.append(categories_key)

        for categories_key in del_keys:
            del times[categories_key]

    return times

class CategoryTimesWalker:
    def __init__(self, category_times):
        self.category_times = category_times

class TotalTimeParser(ProfilerParserCommonMixin):

    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths
        self.src_files = src_files

        # When True, include pyprof timestamps; the total-time-sec we end up with is too large
        # (i.e. larger than total time measuring using time.time() around profiled code)
        #
        # When False, use only tfprof timestamps; the total-time-sec "makes sense"
        # (i.e. a bit smaller but similar to time.time() measurement)
        self.include_pyprof_timestamps = False

        self.parser = parser
        self.args = args
        self.bench_name = bench_name
        self.data = data
        self.skip = False
        self.conn = None

        self.config_path = src_files.get('config_json', bench_name, or_none=True)
        if self.config_path is not None:
            self.config = load_json(self.config_path)
            print("> Found optional config_json @ {f}".format(f=self.config_path))
        else:
            self.config = {
            }

    @staticmethod
    def required_source_basename_regexes():
        return {
            'tfprof_path': r"^profile{bench}.proto$".format(bench=BENCH_SUFFIX_RE),
            'pyprof_path': r"^Pyprof{bench}.proto$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def optional_source_basename_regexes():
        return {
            'microbenchmark_json':r"^microbenchmark.json$",
            'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return False

    @staticmethod
    def uses_multiple_dirs():
        return False

    def tfprof_path(self, bench_name, or_none=True):
        return self.src_files.get('tfprof_path', self.bench_name, or_none=or_none)

    def pyprof_path(self, bench_name):
        return self.src_files.get('pyprof_path', self.bench_name)

    # Output

    @property
    def _profile_json_path(self):
        return self.get_profile_json_path(self.src_files, self.bench_name)

    @classmethod
    def get_profile_json_path(ParseKlass, src_files, bench_name):
        path = _j(src_files.directory, 'total_time_sec{bench}.json'.format(bench=bench_suffix(bench_name)))
        return path

    def parse(self, bench_name):
        self.tf_proto = None
        if self.tfprof_path(self.bench_name) is not None:
            with open(self.tfprof_path(self.bench_name), 'rb') as f:
                self.tf_proto = ProfileProto()
                self.tf_proto.ParseFromString(f.read())

        with open(self.pyprof_path(self.bench_name), 'rb') as f:
            self.py_proto = Pyprof()
            self.py_proto.ParseFromString(f.read())

        if self.tf_proto is not None:
            steps = list(self.tf_proto.steps)
            steps_name = "TF_PROTO"
        else:
            steps = list(self.py_proto.steps)
            steps_name = "PY_PROTO"

        first_step = steps[0]
        min_start_us, max_end_us = self.min_start_max_end(first_step, self.tf_proto, self.py_proto)

        js = {
            'min_start_us': min_start_us,
            'max_end_us': max_end_us,
            'total_time_sec': (max_end_us - min_start_us)/MICROSECONDS_IN_SECOND,
        }
        do_dump_json(js, self._profile_json_path)

    def min_start_max_end(self, step, tf_proto, py_proto):
        def min_start_key(timestamp):
            start_us, end_us = self.as_start_end(timestamp)
            return start_us
        min_start = min(self.each_timestamp(step, tf_proto, py_proto),
                        key=min_start_key)

        def max_end_key(timestamp):
            start_us, end_us = self.as_start_end(timestamp)
            return end_us
        max_end = max(self.each_timestamp(step, tf_proto, py_proto),
                        key=max_end_key)

        min_start_us = self.as_start_end(min_start)[0]
        max_end_us = self.as_start_end(max_end)[1]

        return min_start_us, max_end_us

    def as_start_end(self, timestamp):
        start_us, duration_us = timestamp
        return start_us, start_us + duration_us

    def each_timestamp(self, step, tf_proto, py_proto):
        for tupl in self.each_tfprof_timestamp(step, tf_proto):
            yield tupl
        if self.include_pyprof_timestamps:
            for tupl in self.each_pyprof_timestamp(step, py_proto):
                yield tupl

    def each_tfprof_timestamp(self, step, tf_proto):
        if tf_proto is None:
            return

        for node_id, node in tf_proto.nodes.items():
            # for exec in node.execs[step]:
            for device in node.execs[step].devices:
                for tupl in node.execs[step].accelerator_execs[device].times:
                    start_us, duration_us = tupl.int64_values
                    yield start_us, duration_us
                for tupl in node.execs[step].cpu_execs[device].times:
                    start_us, duration_us = tupl.int64_values
                    yield start_us, duration_us

    def each_pyprof_timestamp(self, step, py_proto):
        if py_proto is None:
            return

        for event in py_proto.python_events[step].events:
            start_us = event.start_time_us
            duration_us = event.duration_us
            yield start_us, duration_us

        for category in py_proto.clibs[step].clibs.keys():
            for event in py_proto.clibs[step].clibs[category].events:
                start_us = event.start_time_us
                duration_us = event.duration_us
                yield start_us, duration_us

    def dump(self, bench_name):
        if self.skip:
            return

class TraceEventsParser:

    # def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
    def __init__(self, directory,
                 # Swallow any excess arguments
                 debug=False,
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
        self.debug = debug
        self.start_usec = start_usec
        self.end_usec = end_usec
        self.overlaps_event_id = overlaps_event_id
        self.op_name = op_name
        self.process_name = process_name
        self.show_progress = show_progress
        self.process_op_nest = process_op_nest

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
        if CATEGORY_DUMMY_EVENT in ignore_cats:
            ignore_cats.remove(CATEGORY_DUMMY_EVENT)

        for op_name in self.op_names:

            for process_name, step, category_times in itertools.islice(
                    self.sql_reader.each_op_instance(op_name,
                                                     group_by_device=True,
                                                     ignore_categories=ignore_cats,
                                                     debug=self.debug),
                    # Just grab the very first operation from the very first process.
                    0, 1):

                print("> Generate traceEvents for step={step}".format(step=step))

                trace_events_dumper = TraceEventsDumper(
                    category_times,
                    json_path=self._trace_first_step_path(process_name, step, op_name))
                trace_events_dumper.dump()

    def _trace_event_overlap(self):
        event = self.sql_reader.event_by_id(self.overlaps_event_id, self.debug)
        category_times = self.sql_reader.events_that_overlap_with(
            event, event.process_name,
            show_progress=self.show_progress)
        trace_events_dumper = TraceEventsDumper(
            category_times,
            json_path=self._trace_event_overlap_path(event))
        trace_events_dumper.dump()

    def _trace_event_time_range(self):
        category_times = self.sql_reader.events_by_time_range(
            self.start_usec, self.end_usec, self.process_name,
            debug=self.debug)
        # category_times = self.sql_reader.events_that_overlap_with(
        #     event, event.process_name,
        #     show_progress=self.show_progress)

        trace_events_dumper = TraceEventsDumper(
            category_times,
            json_path=self._trace_event_time_range_path())
        trace_events_dumper.dump()

    def run(self):

        self.sql_reader = SQLCategoryTimesReader(sql_input_path(self.directory))
        if self.op_name is not None:
            self.op_names = [self.op_name]
        else:
            self.op_names = self.sql_reader.op_names()


        if self.overlaps_event_id is not None:
            trace_func = self._trace_event_overlap
        elif self.start_usec is not None:
            if self.end_usec is None or self.process_name is None:
                raise RuntimeError("Need --start-usec, --end-usec, and --process-name for TraceEventsParser time-range")
            trace_func = self._trace_event_time_range
        else:
            trace_func = self._trace_first_step

        trace_func()

    def dump(self, bench_name):
        if self.skip:
            return

from collections import namedtuple
class _CategoryKey(namedtuple('CategoryKey', 'ops non_ops procs')):
    __slots__ = ()
def CategoryKey(ops, non_ops, procs):
    return _CategoryKey(
        # Allow repeated ops:
        # e.g. <q_forward, q_forward>
        ops=tuple(sorted(ops)),
        non_ops=frozenset(non_ops),
        procs=frozenset(procs))

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
                 debug=False,
                 debug_ops=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.db_path = db_path
        self.debug = debug
        self.debug_ops = debug_ops

    @property
    def directory(self):
        return _d(self.db_path)

    def _compute_process_timeline_stats(self, sql_reader, overlap):
        """

        Q: What's the total time spent tracing that isn't captured by ANY tracing events?
        i.e.
        missing_traced_time_sec = total_trace_time_sec - time_overlap(anything)
                                                         ----------------------
                                                         traced_time_sec

        Q: What's the total time spent tracing that ISN'T captured by operation trace events?
        i.e.
        missing_op_time_sec = total_trace_time_sec - time_overlap(with operation-type)

        Q: What's the total time spent tracing that ISN'T captured by CPU/GPU operation trace events?
        i.e.
        missing_op_time_sec = total_trace_time_sec - time_overlap(with CPU/GPU operations)
                                                     -------------------------------------
                                                     traced_op_time_sec

        :param overlap
          Output from:
            category_times, ... = sql_reader.parse_timeline()
            compute_overlap = ComputeOverlap(category_times)
            compute_overlap.compute()
            overlap = compute_overlap.get_category_times()
        """
        # NOTE: it's nicer to work with
        new_overlap_01 = reduce_category_keys(overlap)
        NumberType = float
        if len(new_overlap_01) > 0:
            NumberType = type(next(iter(new_overlap_01.values())))
        traced_time_sec = NumberType(0.)
        traced_op_time_sec = NumberType(0.)
        for category_key, time_us in new_overlap_01.items():
            traced_time_sec += time_us/NumberType(MICROSECONDS_IN_SECOND)
            if len(category_key.ops) > 0:
                traced_op_time_sec += time_us/NumberType(MICROSECONDS_IN_SECOND)

        total_trace_time_sec = sql_reader.total_trace_time_sec(debug=self.debug)
        missing_traced_time_sec = total_trace_time_sec - traced_time_sec
        missing_op_time_sec = total_trace_time_sec - traced_op_time_sec

        proc_stats = {
            # Tracing time that ISN'T covered by ANY trace-events.
            # Q: How to fix?
            # - Not sure... reduce "gaps" between traced events?
            'missing_traced_time_sec':missing_traced_time_sec,
            # Tracing time that ISN'T covered by ANY operation trace-events.
            # Q: How to fix?
            # - Add more set_operation calls to ML script.
            'missing_op_time_sec':missing_op_time_sec,
        }
        return proc_stats


    def reduce_overlap_resource_operation(
            self, overlap,
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
                continue

            pprint.pprint({
                'overlap.keys()':list(overlap.keys()),
            })
            raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        return new_overlap

    # PROBLEM: CategoryKey has lost whether q_forward belonged to p1 or p2...
    def reduce_overlap_ResourceOverlap(
            self, overlap,
            categories, operation_types, proc_types):
        """
        Group keys by resource-type (non_ops).
        """

        new_overlap = dict()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            new_key = CategoryKey(ops=frozenset(),
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            _add_key(new_overlap, new_key, times)

        return new_overlap

    def reduce_overlap_ResourceSubplot(
            self, overlap,
            categories, operation_types, proc_types):
        """
        Group keys by resource-type (non_ops).
        """
        new_overlap = dict()
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

                new_key = CategoryKey(ops=frozenset(),
                                      non_ops=frozenset([CATEGORY_TOTAL]),
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)

        return new_overlap

    def reduce_overlap_OperationOverlap(
            self, overlap,
            categories, operation_types, proc_types):
        """
        Remove keys that don't match CPU(?).

        Group keys by operation-type (non_ops).
        """
        return self.reduce_overlap_resource_operation(
            overlap,
            categories, operation_types, proc_types,
            group_self_overlap=True)

        return new_overlap

    def _debug_trace_events_path(self, process_name, phase_name):
        return _j(self.directory, "OverlapComputer.trace_events{proc}{phase}{debug}.json".format(
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            # bench=bench_suffix(bench_name),
            debug=debug_suffix(self.debug),
        ))

    OVERLAP_TYPES = ['OperationOverlap', 'ResourceOverlap', 'ResourceSubplot', 'CategoryOverlap', 'default']
    def compute_process_timeline_overlap(self,
                                         pre_reduce,
                                         process_name=None,
                                         phase_name=None,
                                         start_time_us=None,
                                         end_time_us=None,
                                         debug_memoize=False,
                                         overlap_type=None):
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
        sql_reader = SQLCategoryTimesReader(self.db_path)

        # Overlap, computed across different "steps".
        # overlaps = []
        if overlap_type is None:
            overlap_type = 'default'
        assert overlap_type in OverlapComputer.OVERLAP_TYPES

        start_parse_timeline_t = time.time()
        # category_times, categories, operation_types, proc_types = sql_reader.parse_timeline(
        category_times = sql_reader.parse_timeline(
            process_name=process_name,
            phase_name=phase_name,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
            pre_reduce=pre_reduce,
            debug=self.debug,
            debug_ops=self.debug_ops,
            debug_memoize=debug_memoize)
        # if self.debug:
        #     def category_as_str(category):
        #         """
        #         frozenset({'PROC:loop_train_eval', 'estimator_save_model'})
        #         ->
        #         ['estimator_save_model', 'PROC:loop_train_eval']
        #         ->
        #         'estimator_save_model + PROC:loop_train_eval'
        #         """
        #         category_str = ' + '.join(sorted(category))
        #         return category_str
        #     trace_events_dumper = TraceEventsDumper(
        #         category_times,
        #         json_path=self._debug_trace_events_path(process_name, phase_name),
        #         category_as_str=category_as_str)
        #     trace_events_dumper.dump()
        # assert len(operation_types) > 0
        # assert len(categories) > 0
        end_parse_timeline_t = time.time()
        parse_timeline_sec = end_parse_timeline_t - start_parse_timeline_t
        print("> parse_timeline took {sec} seconds".format(sec=parse_timeline_sec))

        # # This can take a while since the timeline can be large...
        # if self.debug:
        #     start_t = time.time()
        #     print("> DEBUG: write process timeline traceEvents @ {path}".format(
        #         path=self._debug_process_timeline_json_path()))
        #     new_category_times = dict((_as_category_key(operation_types, proc_types, proc_key=proc_key), value)
        #                               for proc_key, value in category_times.items())
        #     # reduce_category_keys(category_times, categories, operation_types, proc_types)
        #     dump_category_times(new_category_times, self._debug_process_timeline_json_path(),
        #                         print_log=False,
        #                         category_as_str=traceEvents_key_str)
        #     end_t = time.time()
        #     print("  Took {sec} seconds".format(sec=end_t - start_t))

        # We only want to keep CATEGORY_OPERATION times.
        # However, the operation-types have replaced CATEGORY_OPERATION.

        # if should_load_memo(debug_memoize, self._compute_overlap_memo_path()):
        #     overlap = load_memo(debug_memoize, self._compute_overlap_memo_path())
        # else:

        check_key = None

        # if self.debug:
        #     # Set a breakpoint if we detect overlap across operations within the same process.
        #     # We can inspect the event's using TraceDumper once we know where in the timeline they occur.
        #     # It would help to know the event_id as well...
        #     def check_key(overlap_key):
        #         category_key = _as_category_key(overlap_key=overlap_key)
        #         if len(category_key.ops) > 1:
        #             # Operations can only overlap cross-process, not within a single-process
        #             assert len(category_key.procs) > 1

        # NOTE: We can reduce across whatever dimensions we want to achieve different
        # levels/configurations of the drill-down.
        # Q: ... is that true?
        compute_overlap = ComputeOverlap(category_times,
                                         check_key=check_key,
                                         debug=self.debug,
                                         show_progress=self.debug)
        compute_overlap.compute()
        overlap = compute_overlap.get_category_times()
        # maybe_memoize(debug_memoize, overlap, self._compute_overlap_memo_path())
        assert len(overlap) > 0

        # if self.debug:
        #     pprint.pprint({'overlap.keys()':list(overlap.keys())})

        proc_stats = self._compute_process_timeline_stats(sql_reader, overlap)

        # return operation_overlap, proc_stats
        return overlap, proc_stats

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
    #     print("> Save plot stats to {path}".format(path=self._stats()))
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

        sql_reader = SQLCategoryTimesReader(self.db_path)

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
                print("> DEBUG: dump trace events AFTER process_op_nest @ {path}".format(path=json_path))
                dump_category_times(category_times, json_path, print_log=False)

            for ktime in category_times.get(CATEGORY_OPERATION, []):
                assert ktime.name == bench_name
            # JAMES TODO: We only want to compute overlap of execution time with op-events whose type is bench_name.
            # If it's just execution time without op-type overlap we should discard it.

            # JAMES TODO: remove "Operation" from plot labels
            # compute_overlap = ComputeOverlap(category_times, overlaps_with=[CATEGORY_OPERATION])

            # Can take up to 1.2 seconds, often 0.011 seconds, 0.004 seconds for loop_train_eval.
            start_overlap_t = time.time()
            compute_overlap = ComputeOverlap(category_times)
            compute_overlap.compute()
            overlap = compute_overlap.get_category_times()
            end_overlap_t = time.time()
            sec_overlap = end_overlap_t - start_overlap_t
            if OverlapComputer.DEBUG_COMPUTE_PER_OPERATION_OVERLAP:
                print("> compute_overlap(process={proc}, step={step}) took {sec} seconds".format(
                    proc=process_name,
                    step=step,
                    sec=sec_overlap))
            for category_key in list(overlap.keys()):
                if not( len(category_key) > 1 and CATEGORY_OPERATION in category_key ):
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
        print("> DEBUG: dump per-operation compute overlap @ {path}".format(path=path))
        do_dump_json(json_output, path)

    def _per_operation_json_path(self, bench_name):
        path = _j(self.directory, 'tfprof{bench}.json'.format(bench=bench_suffix(bench_name)))
        return path

class OverlapTypeInterface:
    """
    Overlap computations performed during --rule=UtilizationPlot.

    --overlap-type is one of:
    ResourceOverlap / CategoryOverlap / ResourceSubplot.

    We have a sub-class for each overlap-type.
    """

    def as_overlap_js(self, new_overlap):
        operation_overlap = dict()
        for category_key, time_us in new_overlap.items():
            new_key = self.category_key_as_strs(category_key)
            assert new_key not in operation_overlap
            operation_overlap[new_key] = time_us
        return operation_overlap

    def dump_json_files(self, new_overlap, directory, process_name, phase_name):
        operation_overlap = self.as_overlap_js(new_overlap)
        self.dump_overlap(operation_overlap,
                          path=self._overlap_json(directory, process_name, phase_name),
                          venn_js_path=self._overlap_venn_js_json(directory, process_name, phase_name))

    def _overlap_json(self, directory, process_name, phase_name):
        return _j(directory, "{OverlapType}{proc}{phase}.json".format(
            OverlapType=self.overlap_type,
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def _overlap_venn_js_json(self, directory, process_name, phase_name):
        return _j(directory, "{OverlapType}{proc}{phase}.venn_js.json".format(
            OverlapType=self.overlap_type,
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
        ))

    def dump_overlap(self, operation_overlap,
                     directory=None, process_name=None, phase_name=None,
                     path=None, venn_js_path=None):
        if self.should_dump_as_is:
            return self.dump_overlap_as_is(operation_overlap,
                     directory=directory, process_name=process_name, phase_name=phase_name,
                     path=path)
        return self._dump_overlap(operation_overlap,
                                  directory=directory, process_name=process_name, phase_name=phase_name,
                                  path=path, venn_js_path=venn_js_path)


    def _dump_overlap(self, operation_overlap,
                      directory=None, process_name=None, phase_name=None,
                      path=None, venn_js_path=None):
        if path is None:
            path = self._overlap_json(directory, process_name, phase_name)
        if venn_js_path is None:
            venn_js_path = self._overlap_venn_js_json(directory, process_name, phase_name)
        print("> Dump data for {overlap_type} @ {path}".format(path=path, overlap_type=self.overlap_type))
        dumper = OverlapJSONDumper(operation_overlap)
        js = dumper.dump(path)

        if venn_js_path is not None:
            print("> Dump data for {overlap_type} venn.js plot @ {path}".format(path=venn_js_path, overlap_type=self.overlap_type))
            converter = OverlapJSONToVennConverter(js=js)
            venn_js = converter.dump(venn_js_path)
            pprint.pprint({'venn_js':venn_js})

    def dump_overlap_as_is(self, operation_overlap,
                           directory=None, process_name=None, phase_name=None,
                           path=None):
        if path is None:
            path = self._overlap_json(directory, process_name, phase_name)
        # if venn_js_path is None:
        #     venn_js_path = self._overlap_venn_js_json(directory, process_name, phase_name)
        print("> Dump data for {overlap_type} @ {path}".format(path=path, overlap_type=self.overlap_type))
        js = js_friendly(operation_overlap)
        do_dump_json(js, path, cls=DecimalEncoder)

    def post_reduce(self, overlap):
        category_key_overlap = self.reduce_to_category_key(overlap)
        new_overlap = self.post_reduce_category_key(category_key_overlap)
        return new_overlap

    def pre_reduce_cpu_gpu(self, category, event, process_name):
        """
        Modular function to bin_events for "reducing" events to CPU/GPU BEFORE OverlapComputation.
        Also, allow ability to "filter-out" events (e.g. category=GPU; needed for CategoryOverlap).

        [Events] ->
        :return:
        """
        if category in CATEGORIES_CPU:
            non_ops = frozenset([CATEGORY_CPU])
            ops = frozenset()
        elif category == CATEGORY_GPU:
            non_ops = frozenset([CATEGORY_GPU])
            ops = frozenset()
        elif category == CATEGORY_OPERATION:
            non_ops = frozenset()
            ops = frozenset([event.name])
        else:
            # Q: What about category operation...?
            # We want to KEEP the operation category so we can determine
            # overlap between q_backward/q_forward across processes...
            #
            # I think all we need to do is replace "CATEGORY_OPERATION" for an event
            # with event.name (it's operation-type).
            # Then, when we go to plot the category_times data, we "remove" any operation
            # names from the category (forming an operation_key), and group the data
            # into an operation-specific dict for plotting.
            #
            # We can still test a single process trace without handling this.
            # (graph should be the same with fewer categories: CPU, GPU, CPU + GPU)
            raise RuntimeError("Not sure how to categorize {cat} into CPU or GPU.".format(
                cat=category))
        # new_category = frozenset([cat, proc_category])
        new_key = CategoryKey(ops=ops,
                              non_ops=non_ops,
                              procs=frozenset([process_name]))
        # pprint.pprint({
        #     'name':'pre_reduce_cpu_gpu',
        #     'event':event,
        #     'category':category,
        #     'new_key': new_key})
        return new_key

    def reduce_to_category_key(self, overlap):
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
                #     print("> DELETE OVERLAP:")
                #     pprint.pprint({
                #         'overlap_key':overlap_key,
                #         'proc':skip_proc,
                #         'ops':proc_ops[proc],
                #         'non_ops':proc_non_ops[proc],
                #     })
                continue

            new_key = _as_category_key(overlap_key)
            pprint.pprint({
                'name':'reduce_to_category',
                'new_key':new_key,
                'overlap_key':overlap_key})
            assert len(new_key.ops) > 0
            assert len(new_key.non_ops) > 0
            assert len(new_key.procs) > 0

            if len(new_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(new_key.procs) > 1

            _reduce_add_key(new_overlap, new_key, times)

            # pprint.pprint({
            #     'overlap.keys()':overlap.keys(),
            # })
            # raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        pprint.pprint({'reduce_to_category_key.keys': list(new_overlap.keys())})
        # import ipdb; ipdb.set_trace()

        return new_overlap

    def reduce_overlap_resource_operation(
            self, overlap,
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
                continue

            pprint.pprint({
                'overlap.keys()':list(overlap.keys()),
            })
            raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        return new_overlap

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

class DefaultOverlapType(OverlapTypeInterface):
    def __init__(self):
        self.overlap_type = 'default'
        self.should_dump_as_is = True

    def pre_reduce(self, category, event, process_name):
        return self.pre_reduce_cpu_gpu(category, event, process_name)

    def as_overlap_js(self, new_overlap):
        # def _group_by_ops_resource(self, new_overlap):
        # set(operation categories) -> set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #  <q_forward, q_backward>       <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        operation_overlap = dict()
        for category_key, time_us in new_overlap.items():
            assert len(category_key.ops) > 0
            assert len(category_key.non_ops) > 0
            assert len(category_key.procs) == 0
            if category_key.ops not in operation_overlap:
                operation_overlap[category_key.ops] = dict()
            operation_overlap[category_key.ops][category_key.non_ops] = time_us
        return operation_overlap

    def post_reduce_category_key(self, overlap):
        return self.reduce_overlap_resource_operation(
            overlap, group_self_overlap=False)

class ResourceOverlapType(OverlapTypeInterface):
    def __init__(self):
        self.overlap_type = 'ResourceOverlap'
        self.should_dump_as_is = False

    def pre_reduce(self, category, event, process_name):
        return self.pre_reduce_cpu_gpu(category, event, process_name)

    def post_reduce_category_key(self, overlap):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        :return:
        """
        # def reduce_overlap_ResourceOverlap(
        #         self, overlap,
        #         categories, operation_types, proc_types):
        """
        Group keys by resource-type (non_ops).
        """
        new_overlap = dict()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            new_key = CategoryKey(ops=frozenset(),
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            _reduce_add_key(new_overlap, new_key, times)

        pprint.pprint({'ResourceOverlapType.post_reduce_category_key.keys': list(new_overlap.keys())})
        # import ipdb; ipdb.set_trace()

        return new_overlap

    def category_key_as_strs(self, category_key):
        # def _group_by_resource(self, new_overlap):
        #     # set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #     #   <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        assert len(category_key.ops) == 0
        assert len(category_key.non_ops) > 0
        assert len(category_key.procs) == 0
        return category_key.non_ops

class OperationOverlapType(OverlapTypeInterface):
    def __init__(self):
        self.overlap_type = 'OperationOverlap'
        self.should_dump_as_is = False

    def pre_reduce(self, category, event, process_name):
        return self.pre_reduce_cpu_gpu(category, event, process_name)

    def post_reduce_category_key(self, overlap):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Remove keys that don't match CPU(?).

        Group keys by operation-type (non_ops).

        :return:
        """
        return self.reduce_overlap_resource_operation(
            overlap,
            group_self_overlap=True)

    def _operation_overlap_json(self, directory, process_name, phase_name, resources):
        return _j(directory, "{OverlapType}{proc}{phase}{resources}.json".format(
            OverlapType=self.overlap_type,
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            resources=resources_suffix(resources),
        ))

    def _operation_overlap_venn_js_json(self, directory, process_name, phase_name, resources):
        return _j(directory, "{OverlapType}{proc}{phase}{resources}.venn_js.json".format(
            OverlapType=self.overlap_type,
            proc=process_suffix(process_name),
            phase=phase_suffix(phase_name),
            resources=resources_suffix(resources),
        ))

    def dump_json_files(self, new_overlap, directory, process_name, phase_name):
        for resources, op_overlap in new_overlap.items():
            # operation_overlap = self.as_overlap_js(new_overlap)
            self._dump_overlap(
                op_overlap,
                path=self._operation_overlap_json(directory, process_name, phase_name, resources),
                venn_js_path=self._operation_overlap_venn_js_json(directory, process_name, phase_name, resources))

    def as_overlap_js(self, new_overlap):
        # def _group_by_resource_ops(self, new_overlap):
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

class ResourceSubplotOverlapType(OverlapTypeInterface):
    def __init__(self):
        self.overlap_type = 'ResourceSubplot'
        self.should_dump_as_is = False

    def pre_reduce(self, category, event, process_name):
        return self.pre_reduce_cpu_gpu(category, event, process_name)

    def post_reduce_category_key(self, overlap):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Group keys by resource-type (non_ops).
        :return:
        """
        # def reduce_overlap_ResourceSubplot(
        #         self, overlap,
        #         categories, operation_types, proc_types):
        new_overlap = dict()
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

                new_key = CategoryKey(ops=frozenset(),
                                      non_ops=frozenset([CATEGORY_TOTAL]),
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)

        return new_overlap

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
    # 'CategoryOverlap':CategoryOverlapType,
    'OperationOverlap':OperationOverlapType,
    'ResourceSubplot':ResourceSubplotOverlapType,
}
def overlap_type_to_instance(overlap_type):
    OverlapType = OVERLAP_TYPE_TO_KLASS[overlap_type]
    return OverlapType()

class OverlapJSONDumper:
    def __init__(self, overlap):
        # set([CPU, GPU]) -> np.array(...)
        self.overlap = overlap
        # self.overlap_to_id = dict()

    def dump(self, path):
        js = dict()
        overlap_to_id = dict()
        js['overlap_id_pairs'] = []
        js['overlap_id_to_values'] = dict()
        for i, (k, vs) in enumerate(self.overlap.items()):
            new_k = tuple(sorted(k))
            assert new_k not in overlap_to_id
            overlap_to_id[new_k] = i
            js['overlap_id_pairs'].append((new_k, overlap_to_id[new_k]))
            js['overlap_id_to_values'][overlap_to_id[new_k]] = vs
        do_dump_json(js, path)
        return js

class OverlapJSONToVennConverter:
    def __init__(self, js=None, path=None):
        # set([CPU, GPU]) -> np.array(...)?
        assert js is not None or path is not None
        if path is not None:
            with open(path, 'r') as f:
                js = json.load(f)

        self.js = js
        self.path = path

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
        venn_js = []
        # keys = sorted([k for overlap, k in self.js['overlap_id_pairs']])
        overlap_id_to_overlap_list = dict((k, overlap) for overlap, k in self.js['overlap_id_pairs'])

        category_to_id = dict()
        categories = set()
        for overlap, k in self.js['overlap_id_pairs']:
            assert type(overlap) in [list, tuple, set, frozenset]
            categories.update(overlap)
        categories = sorted(categories)
        for i, category in enumerate(categories):
            category_to_id[category] = i

        def as_sets(overlap_id):
            overlap_list = overlap_id_to_overlap_list[overlap_id]
            sets_ids = [category_to_id[category] for category in overlap_list]
            sets_ids.sort()
            return sets_ids

        for overlap_id, values in self.js['overlap_id_to_values'].items():
            overlap_id = int(overlap_id)
            venn_set = {
                "sets": as_sets(overlap_id),
                "size": values,
            }
            if len(venn_set['sets']) == 1:
                assert len(overlap_id_to_overlap_list[overlap_id]) == 1
                venn_set["label"] = overlap_id_to_overlap_list[overlap_id][0]
            venn_js.append(venn_set)

        # Make the shorter (in particular, single-element) venn_sets appear first.
        # venn_sets within the same length are ordered based on lexicographic order.
        venn_js.sort(key=lambda venn_set: (len(venn_set['sets']), venn_set['sets']))

        return venn_js

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

def reduce_category_keys(overlap):

    new_overlap = dict()
    for overlap_key, times in overlap.items():
        new_key = _as_category_key(overlap_key=overlap_key)
        assert len(new_key.ops) > 0 or \
               len(new_key.non_ops) > 0
        # assert len(new_key.procs) > 0

        if len(new_key.ops) > 1:
            # Operations can only overlap cross-process, not within a single-process
            assert len(new_key.procs) > 1

        _reduce_add_key(new_overlap, new_key, times)

        # pprint.pprint({
        #     'overlap.keys()':overlap.keys(),
        # })
        # raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

    return new_overlap

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

def test_compute_overlap():
    # Set to true to print info.
    # debug = False
    debug = True

    from test.test_util import sec, T

    def test_01_complete():
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
        compute_overlap = ComputeOverlap(category_times, debug=debug)
        compute_overlap.compute_merge()
        got = compute_overlap.get_merged_categories()
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
    test_01_complete()

    def test_02_overlaps_with():
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
        compute_overlap = ComputeOverlap(category_times, overlaps_with=['c1'], debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        expect = {
            frozenset({'c1'}):sec(2),
            frozenset({'c1', 'c2'}):sec(2),
            frozenset({'c1', 'c3'}):sec(1),
            frozenset({'c1', 'c2', 'c3'}):sec(1),
        }
        assert got == expect
    test_02_overlaps_with()

    def test_03_error_partial_overlap():
        category_times = {
            'c1':[
                [
                    T(3, 5), T(4, 6),
                ],
            ],
        }
        compute_overlap = ComputeOverlap(category_times, debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        # expect = {
        #     frozenset({'c1'}):sec(2),
        #     frozenset({'c1', 'c2'}):sec(2),
        #     frozenset({'c1', 'c3'}):sec(1),
        #     frozenset({'c1', 'c2', 'c3'}):sec(1),
        # }
        # assert got == expect
    test_03_error_partial_overlap()

    def test_04_error_full_overlap():
        category_times = {
            'c1':[
                [
                    T(3, 6), T(4, 5),
                ],
            ],
        }
        compute_overlap = ComputeOverlap(category_times, debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        # expect = {
        #     frozenset({'c1'}):sec(2),
        #     frozenset({'c1', 'c2'}):sec(2),
        #     frozenset({'c1', 'c3'}):sec(1),
        #     frozenset({'c1', 'c2', 'c3'}):sec(1),
        # }
        # assert got == expect
    test_04_error_full_overlap()

    def test_05_error_duplicate_overlap():
        category_times = {
            'c1':[
                [
                    T(3, 6), T(3, 6),
                ],
            ],
        }
        compute_overlap = ComputeOverlap(category_times, debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        # expect = {
        #     frozenset({'c1'}):sec(2),
        #     frozenset({'c1', 'c2'}):sec(2),
        #     frozenset({'c1', 'c3'}):sec(1),
        #     frozenset({'c1', 'c2', 'c3'}):sec(1),
        # }
        # assert got == expect
    test_05_error_duplicate_overlap()

    def test_06_error_not_sorted_by_end_time():
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
        compute_overlap = ComputeOverlap(category_times, debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        # expect = {
        #     frozenset({'c1'}):sec(2),
        #     frozenset({'c1', 'c2'}):sec(2),
        #     frozenset({'c1', 'c3'}):sec(1),
        #     frozenset({'c1', 'c2', 'c3'}):sec(1),
        # }
        # assert got == expect
    test_06_error_not_sorted_by_end_time()
