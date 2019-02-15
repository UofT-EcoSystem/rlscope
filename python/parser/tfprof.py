import re
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
    def __init__(self, category_times, overlaps_with=None, debug=False, show_progress=False):
        self.debug = debug
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
            if categories_key not in times:
                times[categories_key] = 0.
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
                 **kwargs):

        self.directory = directory
        self.debug = debug

        # self.dummy_times = []

        # self.category_times_readers = []

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    # def parse_dummy_events(self, step):
    #
    #     self.dummy_times = []
    #
    #     if self._dummy_events_path is None:
    #         return
    #
    #     timestamps = dict()
    #     with open(self._dummy_events_path) as f:
    #         cur_step = None
    #         for line in f:
    #             line = line.rstrip()
    #
    #             m = re.search(r'> RECORDING STEP = (?P<step>\d+)', line)
    #             if m:
    #                 cur_step = int(m.group('step'))
    #
    #             if cur_step is None or cur_step != step:
    #                 continue
    #
    #             m = re.search(r'> name="(?P<name>[^"]+)", timestamp = (?P<time_usec>{float}) usec'.format(
    #                 float=float_re),
    #                 line)
    #             # print("LINE = {line}".format(line=line))
    #             if m:
    #                 assert m.group('name') not in timestamps
    #                 timestamps[m.group('name')] = int(float(m.group('time_usec')))
    #                 continue
    #
    #     for name, time_usec in timestamps.items():
    #         ktime = KernelTime(start_usec=time_usec, time_usec=1, name=name)
    #         self.dummy_times.append(ktime)

    # Output

    # def _dummy_events_path(self, bench_name):
    #     path = self.get_dummy_events_path(self.src_files, bench_name)
    #     if not _e(path):
    #         return None
    #     return path
    #
    # @classmethod
    # def get_dummy_events_path(ParseKlass, src_files, bench_name):
    #     path = _j(src_files.directory, 'dummy_events{bench}.txt'.format(bench=bench_suffix(bench_name)))
    #     return path

    def _trace_events_path(self, process_name, step, bench_name):
        path = _j(self.directory, 'traceEvents{proc}{step}{bench}.json'.format(
            proc=process_suffix(process_name),
            step=step_suffix(step),
            bench=bench_suffix(bench_name),
        ))
        return path

    def run(self):

        self.sql_reader = SQLCategoryTimesReader(sql_input_path(self.directory))
        self.bench_names = self.sql_reader.bench_names()
        # self.category_times_readers.append(self.sql_reader)

        ignore_cats = list(DEFAULT_ignore_categories)
        if CATEGORY_DUMMY_EVENT in ignore_cats:
            ignore_cats.remove(CATEGORY_DUMMY_EVENT)

        for bench_name in self.bench_names:

            for process_name, step, category_times in itertools.islice(
                self.sql_reader.each_op_instance(bench_name,
                                                 group_by_device=True,
                                                 ignore_categories=ignore_cats,
                                                 debug=self.debug),
                # Just grab the very first operation from the very first process.
                0, 1):

                print("> Generate traceEvents for step={step}".format(step=step))

                trace_events_dumper = TraceEventsDumper(
                    category_times,
                    json_path=self._trace_events_path(process_name, step, bench_name))
                trace_events_dumper.dump()

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

    def _compute_process_timeline_stats(self, sql_reader, overlap,
                                        categories, operation_types, proc_types):
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
        new_overlap_01 = reduce_category_keys(overlap,
                                              categories, operation_types, proc_types)
        traced_time_sec = 0.
        traced_op_time_sec = 0.
        for category_key, time_us in new_overlap_01.items():
            traced_time_sec += time_us/MICROSECONDS_IN_SECOND
            if len(category_key.ops) > 0:
                traced_op_time_sec += time_us/MICROSECONDS_IN_SECOND

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

    def reduce_overlap_p0(self, overlap,
                          categories, operation_types, proc_types):
        """
        Reduce keys across processes.

        For initial overlap computation, the keys are sets-of-sets:

        e.g.
        {'category_key': frozenset({frozenset({'PROC:run_atari_1', 'CPU'}),
                                    frozenset({'PROC:run_atari_0', 'CPU'}),
                                    frozenset({'PROC:run_atari_1', 'GPU'}),
                                    frozenset({'PROC:run_atari_1', 'q_backward'})}

        We want to reduce this to:
        {CPU, GPU} U {q_backward} U {P0, P1}

        NOTE: It's possible for use to need to merge keys when doing this.
        For example, the following similar key from the above example,
        maps to the same reduced key:
        {'category_key': frozenset({frozenset({'PROC:run_atari_0', 'CPU'}),
                                    frozenset({'PROC:run_atari_1', 'CPU'}),
                                    frozenset({'PROC:run_atari_0', 'GPU'}),
                                    frozenset({'PROC:run_atari_0', 'q_backward'})}
        =>
        {CPU, GPU} U {q_backward} U {P0, P1}

        We got this key by simply switching P0/P1, which is a likely occurence.

        :param overlap:
        :return:
        """

        def _get(dic, key, default):
            if key not in dic:
                dic[key] = default
            return dic[key]

        def _split_combo_key(combo_key):
            return split_combo_key(combo_key, categories, operation_types, proc_types)

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
            for proc_key in overlap_key:
                ops, non_ops, procs = _split_combo_key(proc_key)
                assert len(procs) == 1
                proc = next(iter(procs))
                _get(proc_ops, proc, set()).update(ops)
                _get(proc_non_ops, proc, set()).update(non_ops)
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

            new_key = _as_category_key(categories, operation_types, proc_types,
                                       overlap_key=overlap_key)
            assert len(new_key.ops) > 0
            assert len(new_key.non_ops) > 0
            assert len(new_key.procs) > 0

            if len(new_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(new_key.procs) > 1

            _add_key(new_overlap, new_key, times)

            # pprint.pprint({
            #     'overlap.keys()':overlap.keys(),
            # })
            # raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        return new_overlap

    def reduce_overlap_p1(self, overlap,
                          categories, operation_types, proc_types):
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

        For keys like <q_forward, P1, P2, …, [CPU, GPU]>:
            new_overlap[q_forward, q_forward] +=
              overlap[q_forward, P1, P2, …, [CPU, GPU]]

        For keys like <q_forward, …, P1, …>:
            "Operations can only overlap cross-process, not within a single-process"
            Assert: If len(ops) > 1 then len(procs) > 1
            new_overlap[q_forward, …, [CPU, GPU]] +=
              overlap[q_forward, …, P1, …, [CPU, GPU]]

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
                new_key = CategoryKey(ops=[cat, cat],
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

    def compute_process_timeline_overlap(self, debug_memoize=False):
        sql_reader = SQLCategoryTimesReader(self.db_path)

        # Overlap, computed across different "steps".
        # overlaps = []

        start_parse_timeline_t = time.time()
        category_times, categories, operation_types, proc_types = sql_reader.parse_timeline(
            debug=self.debug,
            debug_ops=self.debug_ops,
            debug_memoize=debug_memoize)
        assert len(operation_types) > 0
        assert len(categories) > 0
        end_parse_timeline_t = time.time()
        parse_timeline_sec = end_parse_timeline_t - start_parse_timeline_t
        print("> parse_timeline took {sec} seconds".format(sec=parse_timeline_sec))

        # # This can take a while since the timeline can be large...
        # if self.debug:
        #     start_t = time.time()
        #     print("> DEBUG: write process timeline traceEvents @ {path}".format(
        #         path=self._debug_process_timeline_json_path()))
        #     new_category_times = dict((_as_category_key(categories, operation_types, proc_types, proc_key=proc_key), value)
        #                               for proc_key, value in category_times.items())
        #     # reduce_category_keys(category_times, categories, operation_types, proc_types)
        #     dump_category_times(new_category_times, self._debug_process_timeline_json_path(),
        #                         print_log=False,
        #                         category_as_str=traceEvents_key_str)
        #     end_t = time.time()
        #     print("  Took {sec} seconds".format(sec=end_t - start_t))

        # We only want to keep CATEGORY_OPERATION times.
        # However, the operation-types have replaced CATEGORY_OPERATION.


        if should_load_memo(debug_memoize, self._compute_overlap_memo_path()):
            overlap = load_memo(debug_memoize, self._compute_overlap_memo_path())
        else:
            compute_overlap = ComputeOverlap(category_times, show_progress=self.debug)
            compute_overlap.compute()
            overlap = compute_overlap.get_category_times()
            maybe_memoize(debug_memoize, overlap, self._compute_overlap_memo_path())
        assert len(overlap) > 0

        # if self.debug:
        #     pprint.pprint({'overlap.keys()':list(overlap.keys())})

        proc_stats = self._compute_process_timeline_stats(sql_reader, overlap,
                                                          categories, operation_types, proc_types)

        new_overlap = overlap
        assert len(new_overlap) > 0

        new_overlap = self.reduce_overlap_p0(new_overlap,
                                             categories, operation_types, proc_types)
        assert len(new_overlap) > 0

        new_overlap = self.reduce_overlap_p1(new_overlap,
                                             categories, operation_types, proc_types)
        assert len(new_overlap) > 0

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

        if self.debug:
            self._dump_process_timeline_json(operation_overlap)

        return operation_overlap, proc_stats

    def _dump_process_timeline_json(self, operation_overlap):
        path = self._process_timeline_json_path()
        print("> DEBUG: dump process timeline compute overlap @ {path}".format(path=path))

        # PROBLEM: overlap JSON file is usually for a single operation.
        # However, now we have multiple operations for a given overlap calculation.
        # NOTE: the only reason we have a JSON-specific format us because
        # JSON doesn't allow a "set" as a dictionary key.
        #
        # Conversion to JSON:
        # A dict whose keys are frozenset's should be converted to a list of key/value pairs:
        # [
        #   (key[0], value[0]),
        #   ...,
        # ]

        def js_friendly(obj):
            """
            Dict keys in json can only be numbers/strings/booleans/null, they CANNOT be lists/dicts/sets.

            So, to represent a dict whose keys are frozensets...well you just cannot do that.

            :param obj:
            :return:
            """
            if type(obj) == dict and len(obj) > 0 and \
                isinstance(
                    next(iter(obj.keys())),
                    (frozenset, tuple)
                ):
                key_values_pairs = []
                for key, value in obj.items():
                    key_values_pairs.append((
                        js_friendly(key),
                        js_friendly(value)))
                return key_values_pairs
            elif type(obj) == frozenset:
                return sorted([js_friendly(x) for x in obj])
            return obj

        js = js_friendly(operation_overlap)
        do_dump_json(js, path)

    def _process_timeline_json_path(self):
        path = _j(self.directory, 'process_timeline.json')
        return path

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

# From TensorFlow code base:
#
# bool CountAsAcceleratorTime(const string& device) {
# return device.find("stream:all") != device.npos;
# }
# bool CountAsCPUTime(const string& device) {
# return RE2::FullMatch(device,
#                       ".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+");
# }

def split_combo_key(combo_key, categories, operation_types, proc_types):
    assert not isinstance(combo_key, _CategoryKey)
    op_categories = set()
    proc_categories = set()
    non_op_categories = set()
    for category in combo_key:
        if category in operation_types:
            op_categories.add(category)
        elif category in proc_types:
            proc_categories.add(category)
        else:
            non_op_categories.add(category)
    return frozenset(op_categories), frozenset(non_op_categories), frozenset(proc_categories)

def _as_category_key(categories, operation_types, proc_types,
                     overlap_key=None,
                     proc_key=None):

    assert ( overlap_key is not None and proc_key is None ) or \
           ( overlap_key is None and proc_key is not None )

    def _split_combo_key(combo_key):
        return split_combo_key(combo_key, categories, operation_types, proc_types)

    op_categories = set()
    non_op_categories = set()
    proc_categories = set()
    def _add_proc_key(proc_key):
        ops, non_ops, procs = _split_combo_key(proc_key)
        if len(ops) > 0:
            # CPU or GPU.
            # DON'T keep proc information from this.
            # We only keep proc information that tells us about __operation__ overlap across processes,
            assert len(non_ops) == 0
            assert len(procs) > 0
            proc_categories.update(procs)
        op_categories.update(ops)
        non_op_categories.update(non_ops)

    if overlap_key is not None:
        for proc_key in overlap_key:
            _add_proc_key(proc_key)
    else:
        _add_proc_key(proc_key)

    new_key = CategoryKey(op_categories, non_op_categories, proc_categories)
    return new_key

def reduce_category_keys(overlap,
                         categories, operation_types, proc_types):

    new_overlap = dict()
    for overlap_key, times in overlap.items():
        new_key = _as_category_key(categories, operation_types, proc_types,
                                   overlap_key=overlap_key)
        assert len(new_key.ops) > 0 or \
               len(new_key.non_ops) > 0
        # assert len(new_key.procs) > 0

        if len(new_key.ops) > 1:
            # Operations can only overlap cross-process, not within a single-process
            assert len(new_key.procs) > 1

        _add_key(new_overlap, new_key, times)

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
        new_overlap[key] = 0.
def _add_key(new_overlap, key, value):
    """
    Merge an existing key.
    """
    assert type(value) in {float, int}
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
