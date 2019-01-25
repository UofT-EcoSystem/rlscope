import re
import sys
import os
import csv
import textwrap
import pprint
from io import StringIO
import json
import codecs
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from parser.common import *
from parser.stats import Stats
# from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

from parser.stats import KernelTime, category_times_add_time

from parser.db import SQLiteCategoryTimesReader, traces_db_path

from parser.readers import TFProfCategoryTimesReader, \
    DEFAULT_group_by_device, \
    DEFAULT_ignore_categories, \
    DEFAULT_debug \

def read_category_times(category_times_readers, *args, **kwargs):
    category_times = dict()
    for reader in category_times_readers:
        new_times = reader.parse(*args, **kwargs)
        same_categories = set(new_times.keys()).intersection(set(category_times.keys()))
        assert len(same_categories) == 0
        category_times.update(new_times)
    for category in category_times.keys():
        if type(category_times[category]) == list:
            category_times[category].sort(key=lambda ktime: ktime.start_time_usec)
        else:
            for device in category_times[category].keys():
                category_times[category][device].sort(key=lambda ktime: ktime.start_time_usec)
    return category_times

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
    """
    def __init__(self, category_times, debug=False):
        self.debug = debug
        category = list(category_times.keys())[0]
        if type(category_times[category][0]) == list:
            self.category_times = self._flatten_category_times(category_times)
        else:
            # It's already flattened
            self.category_times = category_times
        self.category_times = self._sort_category_times(self.category_times)

    def compute(self):
        self.compute_merge()
        self.compute_times()

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

    def _sort_category_times_by(self, category_times, key):
        new_category_times = dict()
        for category in category_times:
            new_category_times[category] = sorted(category_times[category], key=key)
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
        # categories = set(category_times.keys())

        start_key = lambda ktime: ktime.start_time_usec
        end_key = lambda ktime: ktime.end_time_usec

        def min_time(ctimes, key):
            ktimes = []
            for category in ctimes.keys():
                if len(ctimes[category]) > 0:
                    ktimes.append((category, ctimes[category][0]))
            category, ktime = min(ktimes, key=lambda time_ktime: key(time_ktime[1]))
            return category, ktime

        def min_time_start_end(start_ctimes, end_ctimes):
            if not has_times_left(by_end):
                start_category, start_ktime = min_time(start_ctimes, start_key)
                return start_category, start_ktime, 'start'
            elif not has_times_left(by_start):
                end_category, end_ktime = min_time(end_ctimes, end_key)
                return end_category, end_ktime, 'end'

            start_category, start_ktime = min_time(start_ctimes, start_key)
            end_category, end_ktime = min_time(end_ctimes, end_key)
            if start_key(start_ktime) <= end_key(end_ktime):
                return start_category, start_ktime, 'start'
            return end_category, end_ktime, 'end'

        def has_times_left(ctimes):
            return not all(len(ctimes[category]) == 0 for category in ctimes.keys())

        def pop_time(category, ctimes):
            del ctimes[category][0]

        def get_time(ktime, start_or_end):
            if start_or_end == 'start':
                return start_key(ktime)
            return end_key(ktime)

        by_start = self._sort_category_times_by(category_times, key=start_key)
        by_end = self._sort_category_times_by(category_times, key=end_key)

        cur_categories = set()

        times = dict()

        min_category, min_ktime = min_time(by_start, start_key)
        start_or_end = 'start'
        curtime = start_key(min_ktime)
        pop_time(min_category, by_start)
        cur_categories.add(min_category)

        if self.debug:
            print("> Start computing overlap; choose initial start (curtime)")
            pprint.pprint({
                'min_category': min_category,
                'min_ktime': min_ktime,
                'start_or_end': start_or_end,
                'curtime': curtime,
            })
            print()

        while has_times_left(by_start) or has_times_left(by_end):

            min_category, min_ktime, start_or_end = min_time_start_end(by_start, by_end)
            next_time = get_time(min_ktime, start_or_end)
            time_chunk = next_time - curtime

            if len(cur_categories) > 0:
                # Don't bother recording empty gaps between times.
                categories_key = frozenset(cur_categories)
                if categories_key not in times:
                    times[categories_key] = 0.
                times[categories_key] += time_chunk

            if self.debug:
                pprint.pprint({
                    'min_category': min_category,
                    'min_ktime': min_ktime,
                    'start_or_end': start_or_end,
                    'update':"times[{s}] += {t}".format(s=categories_key, t=time_chunk),
                })
                print()

            if start_or_end == 'start':
                if self.debug:
                    pprint.pprint({'cur_categories':cur_categories,
                                   'add': min_category,
                                   'curtime': next_time})
                    print()
                pop_time(min_category, by_start)
                cur_categories.add(min_category)
            else:
                if self.debug:
                    pprint.pprint({'cur_categories':cur_categories,
                                   'remove': min_category,
                                   'curtime': next_time})
                    print()
                pop_time(min_category, by_end)
                cur_categories.remove(min_category)

            curtime = next_time

        assert len(cur_categories) == 0

        for categories_key in list(times.keys()):
            # We may get artificial overlaps even if two categories are synchronous,
            # if the next category starts exactly when the last one ends.
            if times[categories_key] == 0:
                del times[categories_key]

        return times

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

# class TraceEventsParser(ProfilerParserCommonMixin):
class TraceEventsParser:

    # def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
    def __init__(self, directory,
                 # Swallow any excess arguments
                 debug=False,
                 **kwargs):
        # self.is_dqn = 'microbenchmark_json' in src_files.opt_paths
        # self.src_files = src_files

        # When True, include pyprof timestamps; the total-time-sec we end up with is too large
        # (i.e. larger than total time measuring using time.time() around profiled code)
        #
        # When False, use only tfprof timestamps; the total-time-sec "makes sense"
        # (i.e. a bit smaller but similar to time.time() measurement)
        # self.include_pyprof_timestamps = False

        # self.parser = parser
        # self.args = args
        # self.bench_name = bench_name
        # self.data = data
        self.skip = False

        self.directory = directory
        self.debug = debug

        # self.config_path = src_files.get('config_json', bench_name, or_none=True)
        # if self.config_path is not None:
        #     self.config = load_json(self.config_path)
        #     print("> Found optional config_json @ {f}".format(f=self.config_path))
        # else:
        #     self.config = {
        #     }

        self.dummy_times = []

        self.reset()
        self.reproduce_tfprof = False

        self.category_times_readers = []

    def reset(self):
        self._next_pid = 0
        self.category_to_pid = dict()

    def get_source_files(self):
        """
        We want traces.db
        """
        src_files = []
        traces_db = traces_db_path(self.directory)
        if not _e(traces_db):
            raise MissingInputFiles(textwrap.dedent("""
            {klass}: Couldn't find any traces.db at {path}.
            """.format(
                klass=self.__class__.__name__,
                path=traces_db,
            )))
        return src_files

    def parse_dummy_events(self, step):

        self.dummy_times = []

        if self._dummy_events_path is None:
            return

        timestamps = dict()
        with open(self._dummy_events_path) as f:
            cur_step = None
            for line in f:
                line = line.rstrip()

                m = re.search(r'> RECORDING STEP = (?P<step>\d+)', line)
                if m:
                    cur_step = int(m.group('step'))

                if cur_step is None or cur_step != step:
                    continue

                m = re.search(r'> name="(?P<name>[^"]+)", timestamp = (?P<time_usec>{float}) usec'.format(
                    float=float_re),
                    line)
                # print("LINE = {line}".format(line=line))
                if m:
                    assert m.group('name') not in timestamps
                    timestamps[m.group('name')] = int(float(m.group('time_usec')))
                    continue

        for name, time_usec in timestamps.items():
            ktime = KernelTime(start_usec=time_usec, time_usec=1, name=name)
            self.dummy_times.append(ktime)

    @staticmethod
    def required_source_basename_regexes():
        return {
            'tfprof_path': r"^profile{bench}.proto$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def optional_source_basename_regexes():
        return {
            'pyprof_path': r"^Pyprof{bench}.proto$".format(bench=BENCH_SUFFIX_RE),
            'microbenchmark_json':r"^microbenchmark.json$",
            'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE),
            'dummy_events_path': r"^dummy_events{bench}.txt$".format(bench=BENCH_SUFFIX_RE),
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

    # def tfprof_path(self, bench_name):
    #     return self.src_files.get('tfprof_path', self.bench_name)

    # def pyprof_path(self, bench_name):
    #     return self.src_files.get('pyprof_path', self.bench_name)

    # Output

    def _dummy_events_path(self, bench_name):
        path = self.get_dummy_events_path(self.src_files, bench_name)
        if not _e(path):
            return None
        return path

    @classmethod
    def get_dummy_events_path(ParseKlass, src_files, bench_name):
        path = _j(src_files.directory, 'dummy_events{bench}.txt'.format(bench=bench_suffix(bench_name)))
        return path

    def _profile_json_path(self, bench_name):
        path = _j(self.directory, 'traceEvents{bench}.json'.format(bench=bench_suffix(bench_name)))
        return path
        # return self.get_profile_json_path(self.src_files, bench_name)

    # @classmethod
    # def get_profile_json_path(ParseKlass, src_files, bench_name):
    #     path = _j(src_files.directory, 'traceEvents{bench}.json'.format(bench=bench_suffix(bench_name)))
    #     return path

    # @property
    # def _tfprof_txt_path(self):
    #     return self.get_tfprof_txt_path(self.src_files, self.bench_name)
    #
    # @classmethod
    # def get_tfprof_txt_path(ParseKlass, src_files, bench_name):
    #     path = _j(src_files.directory, 'tfprof{bench}.proto.txt'.format(bench=bench_suffix(bench_name)))
    #     return path
    #
    # @property
    # def _pyprof_txt_path(self):
    #     return self.get_pyprof_txt_path(self.src_files, self.bench_name)
    #
    # @classmethod
    # def get_pyprof_txt_path(ParseKlass, src_files, bench_name):
    #     path = _j(src_files.directory, 'pyprof{bench}.proto.txt'.format(bench=bench_suffix(bench_name)))
    #     return path

    def run(self):

        self.sql_reader = SQLiteCategoryTimesReader(traces_db_path(self.directory))
        self.bench_names = self.sql_reader.bench_names
        self.category_times_readers.append(self.sql_reader)

        for bench_name in self.bench_names:

            steps = self.sql_reader.steps(bench_name)
            if self.debug:
                print("> steps = {steps}".format(steps=steps))

            # step = steps[0]
            # Skip the first step, since it includes profiler initialization stuff.
            # In particular, the libcupti NVIDIA library gets loaded on-demand during
            # the first traced step, and this can take 2 seconds to load!
            # (We had this bug before...)
            if len(steps) > 1:
                step = steps[1]
            else:
                step = steps[0]

            print("> Generate traceEvents for step={step}".format(step=step))

            """
            ComputeOverlap ALSO reads the same information; even though it summarizes ACROSS steps, 
            it still prefers to read individual steps at-a-time.
            
            TraceEvents needs to read category times belonging to a single-operation, 
            and for a single-step.
            
            We don't record any notion of "step" in SQLite.
            op_events = 
              Instead, we need to separately query all the 'Operation' events whose event_name == bench_name, 
              sorted by Event.start_us.
            
            
            step = op_events[1]
            
            events = 
                (category_name, event_name, start_us, end_us)
                Query all the event times that fall within (step.start_us, step.end_us).
            
            category_times = rows_as_category_times(events)
            
            return category_times
            """

            category_times = self.read_category_times(step, bench_name)

            self.js_dump(category_times, bench_name)

    def read_category_times(self, step, bench_name):
        # Just default to outputting the first step...
        ignore_cats = list(DEFAULT_ignore_categories)
        if CATEGORY_DUMMY_EVENT in ignore_cats:
            ignore_cats.remove(CATEGORY_DUMMY_EVENT)
        category_times = read_category_times(self.category_times_readers, step, bench_name,
                                             group_by_device=True,
                                             ignore_categories=ignore_cats,
                                             debug=self.debug)

        if len(self.dummy_times) > 0:
            print("> Adding hardcoded times:")
            pprint.pprint(self.dummy_times, indent=2)
            if CATEGORY_DUMMY_EVENT not in category_times:
                category_times[CATEGORY_DUMMY_EVENT] = []
            category_times[CATEGORY_DUMMY_EVENT].extend(self.dummy_times)

        return category_times

    def js_dump(self, category_times, bench_name):
        self.reset()
        self.js_add_category_times(category_times)
        profile_path = self._profile_json_path(bench_name)
        print("> Write traceEvents to: {path}".format(path=profile_path))
        do_dump_json(self.js, profile_path)

    def js_add_category_times(self, category_times):
        """
        self.js = {
            "traceEvents": [
                {
                    "args": {
                        "name": "Op scheduling threads: /job:localhost/replica:0/task:0/device:gpu:0"
                    },
                    "name": "process_name",
                    "ph": "M",
                    "pid": 0
                },
                {
                    "args": {
                        "name": "Op scheduling threads: /job:localhost/replica:0/task:0/device:cpu:0"
                    },
                    "name": "process_name",
                    "ph": "M",
                    "pid": 1
                },
                {
                    "args": {
                        "name": "Op execution threads: /device:gpu:0/stream:all"
                    },
                    "name": "process_name",
                    "ph": "M",
                    "pid": 2
                },
                {
                    "args": {
                        "name": "deepq/target_q_func/action_value/fully_connected_1/biases"
                    },
                    "cat": "Op",
                    "dur": 138,
                    "name": "deepq/target_q_func/action_value/fully_connected_1/biases",
                    "ph": "X",
                    "pid": 0,
                    "tid": 0,
                    "ts": 1546471006818165
                },
            ...
        }
        """
        self.js = {
            "traceEvents": [
            ],
        }

        for category, times in category_times.items():
            if category == CATEGORY_CUDA_API_CPU:
                # category_times[CATEGORY_CUDA_API_CPU] : device -> (tid -> times)
                assert type(category_times[CATEGORY_CUDA_API_CPU]) == dict
                for device, all_times in category_times[CATEGORY_CUDA_API_CPU].items():
                    category_name = "{cat}: {dev}".format(cat=category, dev=device)
                    self.js_add_section(category_name)
                    tid_to_times = self.js_split_into_tids(all_times)
                    for tid, times in tid_to_times.items():
                        for time in times:
                            self.js_add_time(time.name, category_name, time, tid)
            # elif category == CATEGORY_GPU:
            #     ...
            elif self.reproduce_tfprof and category in [CATEGORY_PYTHON, CATEGORY_TF_API]:
                # Ignore pyprof for now.
                # We just want to test that we are properly reproducing the tfprof traceEvents json.
                pass
            else:
                category_name = category
                self.js_add_section(category_name)
                for time in times:
                    if time.name is not None:
                        name = time.name
                    else:
                        name = 'unknown'
                    self.js_add_time(name, category, time, tid=0)
            # else:
            #     raise NotImplementedError("Not sure how to handle category={c}".format(
            #         c=category))

    def js_split_into_tids(self, times):
        times = sorted(times, key=lambda ktime: ktime.start_time_usec)
        # tid -> times
        next_tid = 0
        allocated_tids = []
        tid_times = dict()

        for time in times:
            tid = -1
            # c_tid = candidate tid
            for c_tid in reversed_iter(allocated_tids):
                if time.start_time_usec < tid_times[c_tid][-1].end_time_usec:
                    # Case (1): kernel start-time < lane's last end-time
                    #   Kernel starts earlier in this lane.
                    #   Either try the next lane, or allocate a new lane
                    pass
                elif time.start_time_usec >= tid_times[c_tid][-1].end_time_usec:
                    # Case (2): kernel start-time > lane's last end-time
                    #   Kernel starts later in this lane.
                    #   Place the kernel in this lane.
                    tid = c_tid
                    break
                # else:
                #     # Case (3): lane's last end-time == kernel's start-time.
                #     #   The current kernel starts executing right as the last kernel in this lane ends.
                #     #   Look at the kernel from this lane that execute before the current one.
                #     #   Q: This seems pointless, seems like we're going to fall into case 2.

            if tid == -1:
                tid = next_tid
                allocated_tids.append(tid)
                next_tid += 1
                tid_times[tid] = []

            tid_times[tid].append(time)

        return tid_times

    def js_add_section(self, name):
        if name in self.category_to_pid:
            return
        pid = self._js_allocate_pid()
        section = {
            'args': {
                'name': name,
            },
            'name': 'process_name',
            'ph': 'M',
            'pid': pid,
        }
        assert name not in self.category_to_pid
        self.category_to_pid[name] = pid
        self.js['traceEvents'].append(section)

    def js_add_time(self, name, category, ktime, tid):
        pid = self.category_to_pid[category]
        jtime = {
            'args': {
                'name': name,
            },
            'cat': 'Op',
            'dur': ktime.total_time_usec,
            'name': name,
            'ph': 'X',
            'pid': pid,
            'tid': tid,
            'ts': ktime.start_time_usec,
        }
        self.js['traceEvents'].append(jtime)

    def _js_allocate_pid(self):
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def dump(self, bench_name):
        if self.skip:
            return

class OverlapJSONParser:
    """
    Computes a json file containing the overlap between event categories across different steps,
    on a per-operation/bench_name basis.

    e.g.
    tfprof.q_forward.json:
        For each q_forward operation, tell me the overlap among
        the different categories measure (GPU, Python, CUDA API C, etc)
    """

    def __init__(self, directory,
                 # Swallow any excess arguments
                 debug=False,
                 **kwargs):

        self.directory = directory
        self.debug = debug

    def get_source_files(self):
        """
        We want traces.db
        """
        src_files = []
        traces_db = traces_db_path(self.directory)
        if not _e(traces_db):
            raise MissingInputFiles(textwrap.dedent("""
            {klass}: Couldn't find any traces.db at {path}.
            """.format(
                klass=self.__class__.__name__,
                path=traces_db,
            )))
        return src_files

    def tfprof_path(self, bench_name, or_none=True):
        return self.src_files.get('tfprof_path', self.bench_name, or_none=or_none)

    def pyprof_path(self, bench_name):
        return self.src_files.get('pyprof_path', self.bench_name)

    def get_micro_name(self):
        return self.bench_name

    def run(self):
        # self.tf_proto = None

        # if self.tfprof_path(self.bench_name) is not None:
        #     with open(self.tfprof_path(self.bench_name), 'rb') as f:
        #         self.tf_proto = ProfileProto()
        #         self.tf_proto.ParseFromString(f.read())

        # with open(self.pyprof_path(self.bench_name), 'rb') as f:
        #     self.py_proto = Pyprof()
        #     self.py_proto.ParseFromString(f.read())

        # raise RuntimeError(
        #     "This code has a bug in it; we FORGOT to skip the first step, "
        #     "so these times will include libcupti library-load-time overhead.")

        # if self.tf_proto is not None and len(self.tf_proto.steps) > 0:
        #     steps = list(self.tf_proto.steps)
        #     steps_name = "TF_PROTO"
        # else:
        #     steps = list(self.py_proto.steps)
        #     steps_name = "PY_PROTO"

        # print("OverlapJSONParser > steps={name}".format(name=steps_name))

        # if self.tf_proto is not None:
        #     for tf_step in self.tf_proto.steps:
        #         assert tf_step in self.py_proto.steps

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

        category_times_readers = []

        # tfprof_path = self.tfprof_path(self.bench_name)
        # if tfprof_path is not None:
        #     tfprof_reader = TFProfCategoryTimesReader(tfprof_path)
        #     category_times_readers.append(tfprof_reader)

        # pyprof_path = self.pyprof_path(self.bench_name)
        # pyprof_reader = PyprofCategoryTimesReader(pyprof_path)
        # category_times_readers.append(pyprof_reader)

        sql_reader = SQLiteCategoryTimesReader(traces_db_path(self.directory))
        category_times_readers.append(sql_reader)
        bench_names = sql_reader.bench_names
        for bench_name in bench_names:
            steps = sql_reader.steps(bench_name)
            print("> steps = ")
            pprint.pprint({'len(steps)':len(steps),
                           'steps':steps})

            # Overlap, computed across different "steps".
            overlaps = []

            # Skip the first step (it captures libcupti.so load time).
            keep_steps = steps[1:]
            for step in keep_steps:
                category_times = read_category_times(category_times_readers, step, bench_name,
                                                     # Don't want to compute overlap w/ dummy events; doesn't mean anything.
                                                     ignore_categories=DEFAULT_ignore_categories,
                                                     debug=self.debug)
                compute_overlap = ComputeOverlap(category_times)
                compute_overlap.compute()
                overlaps.append(compute_overlap.get_category_times())

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
            do_dump_json(json_output, self._profile_json_path(bench_name))

    def dump(self, bench_name):
        if self.skip:
            return

    def _profile_json_path(self, bench_name):
        path = _j(self.directory, 'tfprof{bench}.json'.format(bench=bench_suffix(bench_name)))
        return path

    # @classmethod
    # def get_profile_json_path(ParseKlass, src_files, bench_name):
    #     path = _j(src_files.directory, 'tfprof{bench}.json'.format(bench=bench_suffix(bench_name)))
    #     return path

# From TensorFlow code base:
#
# bool CountAsAcceleratorTime(const string& device) {
# return device.find("stream:all") != device.npos;
# }
# bool CountAsCPUTime(const string& device) {
# return RE2::FullMatch(device,
#                       ".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+");
# }

def reversed_iter(xs):
    n = len(xs)
    for i in range(n - 1, 0 - 1, -1):
        yield xs[i]

def test_compute_overlap():
    # Set to true to print info.
    debug = False

    def sec(seconds):
        return seconds*MICROSECONDS_IN_SECOND

    def T(start_sec, end_sec):
        return KernelTime(start_usec=start_sec*MICROSECONDS_IN_SECOND,
                          end_usec=end_sec*MICROSECONDS_IN_SECOND)
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

