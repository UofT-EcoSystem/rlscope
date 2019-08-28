import logging
import itertools
from os.path import join as _j, dirname as _d

from iml_profiler.parser.common import *
# from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from iml_profiler.protobuf.pyprof_pb2 import Pyprof, ProcessMetadata

from iml_profiler.profiler import proto_util

from iml_profiler.parser.stats import KernelTime

from iml_profiler.parser.trace_events import TraceEventsDumper, dump_category_times

from iml_profiler.parser.db import SQLCategoryTimesReader, sql_input_path, sql_get_source_files, \
    Machine, Process, Phase

from iml_profiler.parser.readers import TFProfCategoryTimesReader, \
   DEFAULT_group_by_device, \
   DEFAULT_ignore_categories, \
   DEFAULT_debug \

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
        # TODO: Just remove this...?
        # if len(category_times) > 0 and \
        #         type(next(iter(category_times.values()))[0]) == list:
        #     self.category_times = self._flatten_category_times(category_times)
        # else:
        # It's already flattened
        self.category_times = category_times
        self.category_times = self._sort_category_times(self.category_times)

    def compute(self):
        start_merge_t = time.time()
        self.compute_merge()
        end_merge_t = time.time()
        sec_merge = end_merge_t - start_merge_t
        if self.debug:
            logging.info("> {klass}.compute_merge took {sec} seconds".format(
                klass=self.__class__.__name__,
                sec=end_merge_t))
        start_compute_t = end_merge_t
        self.compute_times()
        end_compute_t = time.time()
        sec_compute = end_compute_t - start_compute_t
        if self.debug:
            logging.info("> {klass}.compute_times took {sec} seconds".format(
                klass=self.__class__.__name__,
                sec=sec_compute))

    def compute_merge(self):
        self.merged_category_times = self._merge_category_times(self.category_times)

    def compute_times(self):
        # set(c1, ..., cn) -> time in seconds
        self.times, self.overlap_metadata = self._compute_overlap(self.merged_category_times)

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

class RegionMetadata:
    def __init__(self, category_key=None):
        self.category_key = category_key
        self.start_time_usec = None
        self.end_time_usec = None
        self.num_events = 0

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

    def __str__(self):
        return "OverlapMetadata(regions={regions})".format(regions=self.regions)

def compute_overlap_single_thread(
    category_times,
    overlaps_with=None,
    check_key=None,
    debug=False,
    show_progress=False):
    # categories = set(category_times.keys())
    # JAMES TODO: compute the progress of this function... I think it takes forever with minigo

    overlap_metadata = OverlapMetadata()

    start_key = lambda ktime: ktime.start_time_usec
    reversed_start_key = lambda ktime: - ktime.start_time_usec
    end_key = lambda ktime: ktime.end_time_usec
    reversed_end_key = lambda ktime: - ktime.end_time_usec

    by_start = CategoryTimesWrapper(category_times, start_key, reversed_start_key, 'start')
    by_end = CategoryTimesWrapper(category_times, end_key, reversed_end_key, 'end')

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
        if category not in cur_categories:
            import ipdb; ipdb.set_trace()
        cur_categories.remove(category)
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
        logging.info("> Start computing overlap; choose initial start (curtime)")
        pprint.pprint({
            'min_category': min_category,
            'min_ktime': min_ktime,
            'start_or_end': start_or_end.name,
            'curtime': curtime,
        })
        logging.info()

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
            # For example, overlap across CATEGORY_OPERATION's within a single process.
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
            logging.info()

        if start_or_end.type_code == CategoryTimesWrapper.START:
            if debug:
                pprint.pprint({'cur_categories':cur_categories,
                               'add': min_category,
                               'curtime': next_time})
                logging.info()
            pop_start(min_category, min_ktime)
        else:
            if debug:
                pprint.pprint({'cur_categories':cur_categories,
                               'remove': min_category,
                               'curtime': next_time})
                logging.info()
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

    return times, overlap_metadata

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
            logging.info("> Found optional config_json @ {f}".format(f=self.config_path))
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
        if CATEGORY_DUMMY_EVENT in ignore_cats:
            ignore_cats.remove(CATEGORY_DUMMY_EVENT)

        machines = self.sql_reader.machines()
        assert len(machines) == 1
        machine = machines[0]

        logging.info("op_names = {op_names}".format(op_names=self.op_names))
        logging.info("process_names = {procs}".format(procs=self.sql_reader.process_names(machine.machine_name)))
        for op_name in self.op_names:
            logging.info("op_name = {op_name}".format(op_name=op_name))
            for process_name, step, category_times in itertools.islice(
                    self.sql_reader.each_op_instance(op_name,
                                                     machine_name=machine.machine_name,
                                                     filter_op=self.filter_op,
                                                     group_by_device=True,
                                                     ignore_categories=ignore_cats,
                                                     debug=self.debug),
                    # Just grab the very first operation from the very first process.
                    0, 1):
                logging.info("  process_name = {proc}, step = {step}".format(proc=process_name, step=step))

                logging.info("> Generate traceEvents for step={step}".format(step=step))

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
            from iml_profiler.profiler.profilers import unit_test_util
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
            # if CATEGORY_DUMMY_EVENT in ignore_cats:
            #     ignore_cats.remove(CATEGORY_DUMMY_EVENT)
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
            #         logging.info("> Generate traceEvents for step={step}".format(step=step))
            #
            #         trace_events_dumper = TraceEventsDumper(
            #             category_times,
            #             json_path=self._trace_first_step_path(process_name, step, op_name),
            #             debug=self.debug)
            #         trace_events_dumper.dump()

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
        if category in CATEGORIES_CPU:
            cpus.add(category)
        elif category in CATEGORIES_GPU:
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
                 debug=False,
                 debug_ops=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.db_path = db_path
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug
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
    #         traced_time_sec += time_us/NumberType(MICROSECONDS_IN_SECOND)
    #         if len(category_key.ops) > 0:
    #             traced_op_time_sec += time_us/NumberType(MICROSECONDS_IN_SECOND)
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
            self, overlap, overlap_metadata,
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
                                      non_ops=frozenset([CATEGORY_TOTAL]),
                                      procs=frozenset())
                _add_key(new_overlap, new_key, times)
                new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

        return new_overlap, new_overlap_metadata

    def reduce_overlap_OperationOverlap(
            self, overlap, overlap_metadata,
            categories, operation_types, proc_types):
        """
        Remove keys that don't match CPU(?).

        Group keys by operation-type (non_ops).
        """
        return self.reduce_overlap_resource_operation(
            overlap, overlap_metadata,
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
                                         pre_reduce,
                                         machine_name=None,
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
        sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

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
            machine_name=machine_name,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
            pre_reduce=pre_reduce,
            debug=self.debug,
            debug_ops=self.debug_ops,
            debug_memoize=debug_memoize)

        if self.debug:
            pprint_msg({'category_times.keys()': sorted(category_times.keys())})

            # category_times_path = _j(self.directory, "category_times{over}{proc}{mach}{phase}.debug.txt".format(
            #     proc=process_suffix(process_name),
            #     mach=machine_suffix(machine_name),
            #     phase=phase_suffix(phase_name),
            #     over=overlap_type_suffix(overlap_type),
            # ))
            # logging.info('Write category_times @ {path}'.format(path=category_times_path))
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
        logging.info("> parse_timeline took {sec} seconds".format(sec=parse_timeline_sec))

        # # This can take a while since the timeline can be large...
        # if self.debug:
        #     start_t = time.time()
        #     logging.info("> DEBUG: write process timeline traceEvents @ {path}".format(
        #         path=self._debug_process_timeline_json_path()))
        #     new_category_times = dict((_as_category_key(operation_types, proc_types, proc_key=proc_key), value)
        #                               for proc_key, value in category_times.items())
        #     # reduce_category_keys(category_times, categories, operation_types, proc_types)
        #     dump_category_times(new_category_times, self._debug_process_timeline_json_path(),
        #                         print_log=False,
        #                         category_as_str=traceEvents_key_str)
        #     end_t = time.time()
        #     logging.info("  Took {sec} seconds".format(sec=end_t - start_t))

        # We only want to keep CATEGORY_OPERATION times.
        # However, the operation-types have replaced CATEGORY_OPERATION.

        # if should_load_memo(debug_memoize, self._compute_overlap_memo_path()):
        #     overlap = load_memo(debug_memoize, self._compute_overlap_memo_path())
        # else:

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
                        logging.info("> Detected unexpected CategoryKey when computing overlap:")
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
                                         debug=self.debug,
                                         show_progress=self.debug)
        compute_overlap.compute()
        overlap = compute_overlap.get_category_times()
        overlap_metadata = compute_overlap.get_overlap_metadata()
        # maybe_memoize(debug_memoize, overlap, self._compute_overlap_memo_path())
        # assert len(overlap) > 0

        pprint.pprint({
            'overlap_metadata': overlap_metadata,
            'machine_name':machine_name,
            'process_name':process_name,
            'phase_name':phase_name})

        # if self.debug:
        #     pprint.pprint({'overlap.keys()':list(overlap.keys())})

        # proc_stats = self._compute_process_timeline_stats(sql_reader, overlap)
        proc_stats = None

        # return operation_overlap, proc_stats
        return overlap, proc_stats, overlap_metadata

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
    #     logging.info("> Save plot stats to {path}".format(path=self._stats()))
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
                logging.info("> DEBUG: dump trace events AFTER process_op_nest @ {path}".format(path=json_path))
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
                logging.info("> compute_overlap(process={proc}, step={step}) took {sec} seconds".format(
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
        logging.info("> DEBUG: dump per-operation compute overlap @ {path}".format(path=path))
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
        logging.info("> Dump data for {overlap_type} @ {path}".format(path=path, overlap_type=self.overlap_type))
        dumper = OverlapJSONDumper(operation_overlap, metadata)
        dumper.dump(path)

        if venn_js_path is not None:
            logging.info("> Dump data for {overlap_type} venn.js plot @ {path}".format(path=venn_js_path, overlap_type=self.overlap_type))
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
        logging.info("> Dump data for {overlap_type} @ {path}".format(path=path, overlap_type=self.overlap_type))
        js = js_friendly({
            'overlap': operation_overlap,
            'metadata': metadata,
        })
        do_dump_json(js, path, cls=DecimalEncoder)

    def post_reduce(self, overlap, overlap_metadata):
        category_key_overlap, category_key_overlap_metadata = self.reduce_to_category_key(overlap, overlap_metadata)
        new_overlap, new_overlap_metadata = self.post_reduce_category_key(category_key_overlap, category_key_overlap_metadata)
        return new_overlap, new_overlap_metadata

    def pre_reduce_cpu_gpu(self, category, event):
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
                              procs=frozenset([event.process_name]))

        # pprint.pprint({
        #     'name':'pre_reduce_cpu_gpu',
        #     'event':event,
        #     'category':category,
        #     'new_key': new_key})

        return new_key

    def reduce_to_category_key(self, overlap, overlap_metadata):
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
                #     logging.info("> DELETE OVERLAP:")
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

            if len(new_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(new_key.procs) > 1

            _reduce_add_key(new_overlap, new_key, times)
            new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

            # pprint.pprint({
            #     'overlap.keys()':overlap.keys(),
            # })
            # raise NotImplementedError("Not sure how to reduce overlap keys for overlap_key={key}".format(key=overlap_key))

        if self.debug:
            pprint.pprint({
                'reduce_to_category_key.keys': list(new_overlap.keys()),
                'new_overlap_metadata': new_overlap_metadata,
            })
            # import ipdb; ipdb.set_trace()

        return new_overlap, new_overlap_metadata

    def reduce_overlap_resource_operation(
            self, overlap, overlap_metadata,
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

class DefaultOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'default'
        self.should_dump_as_is = True
        self.debug = debug or DEBUG_OVERLAP_TYPE

    def pre_reduce(self, category, event):
        return self.pre_reduce_cpu_gpu(category, event)

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

    def post_reduce_category_key(self, overlap, overlap_metadata):
        return self.reduce_overlap_resource_operation(
            overlap, overlap_metadata, group_self_overlap=False)

class ResourceOverlapType(OverlapTypeInterface):
    def __init__(self, debug=False):
        self.overlap_type = 'ResourceOverlap'
        self.should_dump_as_is = False
        self.debug = debug or DEBUG_OVERLAP_TYPE

    def pre_reduce(self, category, event):
        return self.pre_reduce_cpu_gpu(category, event)

    def post_reduce_category_key(self, overlap, overlap_metadata):
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
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            new_key = CategoryKey(ops=frozenset(),
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            _reduce_add_key(new_overlap, new_key, times)
            new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

        if self.debug:
            pprint.pprint({
                'ResourceOverlapType.post_reduce_category_key.keys': list(new_overlap.keys()),
                'new_overlap_metadata':new_overlap_metadata,
            })
        # import ipdb; ipdb.set_trace()

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

    def pre_reduce(self, category, event):
        return self.pre_reduce_cpu_gpu(category, event)

    def post_reduce_category_key(self, overlap, overlap_metadata):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Remove keys that don't match CPU(?).

        Group keys by operation-type (non_ops).

        :return:
        """
        return self.reduce_overlap_resource_operation(
            overlap, overlap_metadata,
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

    def pre_reduce(self, category, event):
        """
        Modular function to bin_events for "reducing" events to CPU/GPU BEFORE OverlapComputation.
        Also, allow ability to "filter-out" events (e.g. category=GPU; needed for CategoryOverlap).

        Pre-reduce: keep categories as-is; filter out GPU stuff.
        """
        if category in CATEGORIES_CPU:
            # Keep ALL the CPU events, and retain the details of their category.

            non_ops = frozenset([category])
            # Q: If we change this one thing, we SHOULD get the same total_size as OperationOverlap (equivalent code)
            # A: YES, we do as expected
            #    => Something caused by using non_ops=raw_category during causes the inflation in overlap time...
            # non_ops = frozenset([CATEGORY_CPU])

            ops = frozenset()
        elif category == CATEGORY_GPU:
            # SKIP GPU events; we just want to measure CPU time.
            # Q: Won't GPU perhaps overlap with these?
            #       [ GPU ]
            # [     CPU     ]
            # No, I don't think so; GPU events are recorded from the "GPU-side".
            # "CUDA API C" is recorded from the CPU-side.
            # return None
            non_ops = frozenset([category])
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
                              procs=frozenset([event.process_name]))

        # if self.debug:
        #     pprint.pprint({
        #         'name':'{OverlapType}.pre_reduce'.format(OverlapType=self.overlap_type),
        #         'event':event,
        #         'category':category,
        #         'new_key': new_key})

        return new_key

    def post_reduce_category_key(self, overlap, overlap_metadata):
        """
        Add modular "post-reduce" function for "adding" CategoryKey's that map to the same key.

        Post-reduce: I don't think we need to do anything here?
        Everything should belong to a single process:
          CategoryKey(ops=key.ops, non_ops=key.non_ops, procs=None)
        """
        # return self.reduce_overlap_resource_operation(
        #     overlap, overlap_metadata,
        #     group_self_overlap=True)

        new_overlap = dict()
        new_overlap_metadata = OverlapMetadata()
        for overlap_key, times in overlap.items():

            if len(overlap_key.ops) > 1:
                # Operations can only overlap cross-process, not within a single-process
                assert len(overlap_key.procs) > 1

            assert len(overlap_key.procs) == 1
            assert len(overlap_key.ops) == 1
            assert len(overlap_key.non_ops) >= 1
            new_key = CategoryKey(ops=overlap_key.ops,
                                  non_ops=overlap_key.non_ops,
                                  procs=frozenset())
            _reduce_add_key(new_overlap, new_key, times)
            new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

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
            # import ipdb; ipdb.set_trace()

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
                    # 'resource_overlap': sorted([CATEGORY_CPU]),
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
                resource_key.add(CATEGORY_CPU)
            if len(cpus_gpus.gpus) > 0:
                resource_key.add(CATEGORY_GPU)
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

    def pre_reduce(self, category, event):
        return self.pre_reduce_cpu_gpu(category, event)

    def post_reduce_category_key(self, overlap, overlap_metadata):
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
                                  non_ops=frozenset([CATEGORY_TOTAL]),
                                  procs=frozenset())
            _add_key(new_overlap, new_key, times)
            new_overlap_metadata.merge_region(new_key, overlap_metadata.get_region(overlap_key))

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
    def __init__(self, js=None, path=None):
        # "overlap region"     "size of overlap"
        # set([CPU, GPU])  ->  Number
        assert js is not None or path is not None
        if path is not None:
            with open(path, 'r') as f:
                js = json.load(f)

        self.js = js
        self.path = path
        self.overlap = self._reconstruct_overlap()
        self.metadata = self.js['metadata']

    def _reconstruct_overlap(self):
        overlap = dict()

        for pairs, size in self.js['overlap']:
            overlap[tuple(sorted(pairs))] = float(size)
        return overlap

    def _compute_set_sizes(self):
        set_to_size = dict()
        for overlap_region, size in self.overlap.items():
            for set_region in overlap_region:
                assert type(set_region) == str
                if set_region not in set_to_size:
                    set_to_size[set_region] = as_type(0., type(size))
                set_to_size[set_region] += size
        return set_to_size

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

# def test_venn_js_converter():
#
#     wrong_venn_js = {
#         # Sizes of the INDIVIDUAL regions.
#         'venn_js': [{'label': 'Framework API C',
#                      'sets': [1],
#                      'size': Decimal('15174940')},
#                     {'label': 'Python', 'sets': [2], 'size': Decimal('192978')},
#                     {'sets': [0, 1], 'size': Decimal('202827')},
#                     {'sets': [0, 2], 'size': Decimal('401')},
#                     {'sets': [1, 2], 'size': Decimal('85')}]}
#
#     correct_venn_js =
#         # Sizes of the INDIVIDUAL regions.
#         'venn_js': [{'label': 'Framework API C',
#                      'sets': [1],
#                      'size': Decimal('15174940') + Decimal('202827') + Decimal('85')},
#                     {'label': 'Python', 'sets': [2], 'size': Decimal('192978') + Decimal('401') + Decimal('85')},
#                     {'sets': [0, 1], 'size': Decimal('202827')},
#                     {'sets': [0, 2], 'size': Decimal('401')},
#                     {'sets': [1, 2], 'size': Decimal('85')}]}
#
#     overlap = {('estimator_save_model',): {frozenset({'Framework API C'}): Decimal('369188'),
#                                            frozenset({'Python'}): Decimal('336883'),
#                                            frozenset({'CUDA API CPU', 'Framework API C'}): Decimal('41590')},
#                ('estimator_train',): {frozenset({'Framework API C'}): Decimal('6908831'),
#                                       frozenset({'Python'}): Decimal('17241'),
#                                       frozenset({'CUDA API CPU', 'Framework API C'}): Decimal('19335039')},
#                ('gather',): {frozenset({'Framework API C'}): Decimal('15174940'),
#                              frozenset({'CUDA API CPU', 'Framework API C'}): Decimal('202827'),
#                              frozenset({'Python'}): Decimal('192978'),
#                              frozenset({'Python', 'CUDA API CPU'}): Decimal('401'),
#                              frozenset({'Python', 'Framework API C'}): Decimal('85')},
#                ('train',): {frozenset({'Framework API C'}): Decimal('3595757'),
#                             frozenset({'Python'}): Decimal('4952793'),
#                             frozenset({'CUDA API CPU', 'Framework API C'}): Decimal('60576')}}
#
#     js = overlap[('gather',)]
#
#     venn_js = OverlapJSONToVennConverter(js)
#     assert venn_js != wrong_venn_js

def compute_total_size(overlap):
    return np.sum(list(overlap.values()))

def test_merge_adjacent_events():

    from test.test_util import sec, T, flatten_category_times as flat

    def test_01_merge_adj_events():
        events = [T(1, 6), T(2, 6), T(3, 7), T(4, 6)]
        got = merge_adjacent_events(events)
        expect = [T(1, 7)]
        assert got == expect
    test_01_merge_adj_events()

def test_compute_overlap():
    # Set to true to print info.
    # debug = False
    debug = True

    from test.test_util import sec, T, flatten_category_times as flat

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
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)
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
        compute_overlap = ComputeOverlap(flat(category_times), overlaps_with=['c1'], debug=debug)

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
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)

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
    test_03_error_partial_overlap()

    def test_04_error_full_overlap():
        category_times = {
            'c1':[
                [
                    T(3, 6), T(4, 5),
                ],
            ],
        }
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)

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
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)

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
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)

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

    def test_07_overlapping_sorted_events():

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
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        expect = {
            frozenset({'c1'}):sec(5),
            frozenset({'c1', 'c2'}):sec(1),
        }
        assert got == expect
    test_07_overlapping_sorted_events()

    def test_08_overlapping_sorted_events():
        # Q: What if start times match but end times are unordered?
        # Q: WHY would this EVER happen in our data though...?
        #    It CAN if concurrent events get "shuffled" into the same category (for some reason).
        #    Perhaps this could happen with CPU/GPU?

        category_times = {
            'c1':[
                [
                    # [1..7] 6
                    T(1, 6), T(2, 6), T(3, 7), T(4, 6)
                ],
            ],
            'c2':[
                [
                    # [4..9] 5
                    T(4, 4.5), T(4.5, 5), T(5, 9), T(5, 8)
                ],
            ],
            'c3':[
                [
                    # [5..9] 4
                    T(5, 9),
                ],
            ],
        }
        compute_overlap = ComputeOverlap(flat(category_times), debug=debug)

        compute_overlap.compute()
        got = compute_overlap.get_category_times()
        expect = {
            # [1..4]
            frozenset({'c1'}):sec(3),
            # [4..5]
            frozenset({'c1', 'c2'}):sec(1),
            # [5..7]
            frozenset({'c1', 'c2', 'c3'}):sec(2),
            # [7..9]
            frozenset({'c2', 'c3'}):sec(2),
        }
        assert got == expect
    test_08_overlapping_sorted_events()

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
                logging.info("Subsume: {e1} subsumes {e2}".format(e1=new_events[-1], e2=event))
            new_events[-1].set_start_end(
                start_usec=new_events[-1].start_usec,
                end_usec=max(new_events[-1].end_usec, event.end_usec),
            )
        elif new_events[-1].start_time_usec <= event.start_time_usec <= new_events[-1].end_time_usec <= event.end_time_usec:
            # case 2: partial overlap
            if debug:
                logging.info("Partial: {e1} partial {e2}".format(e1=new_events[-1], e2=event))
            new_events[-1].set_start_end(
                start_usec=new_events[-1].start_usec,
                end_usec=max(new_events[-1].end_usec, event.end_usec),
            )
        else:
            # case 3: no overlap
            if debug:
                logging.info("No-overlap: {e1} no-overlap {e2}".format(e1=new_events[-1], e2=event))
            new_events.append(get_event(event))

        if debug:
            pprint.pprint(new_events)

    return new_events

