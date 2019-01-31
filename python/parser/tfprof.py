import re
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

from parser.db import SQLiteCategoryTimesReader, traces_db_path

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
    def __init__(self, category_times, overlaps_with=None, debug=False):
        self.debug = debug
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

        if self.overlaps_with is not None:
            del_keys = []
            for categories_key in times.keys():
                # if not self.overlaps_with.issubset(categories_key):
                if self.overlaps_with.intersection(categories_key) == 0:
                    del_keys.append(categories_key)

            for categories_key in del_keys:
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

        self.sql_reader = SQLiteCategoryTimesReader(traces_db_path(self.directory))
        self.bench_names = self.sql_reader.bench_names
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
                 # Swallow any excess arguments
                 debug=False,
                 **kwargs):
        self.db_path = db_path
        self.debug = debug

    @property
    def directory(self):
        return _d(self.db_path)

    def compute_process_timeline_overlap(self):
        sql_reader = SQLiteCategoryTimesReader(self.db_path)

        # Overlap, computed across different "steps".
        # overlaps = []

        category_times, categories, operation_types = sql_reader.parse_timeline(debug=self.debug)
        assert len(operation_types) > 0
        assert len(categories) > 0

        def split_combo_key(combo_key):
            op_categories = set()
            non_op_categories = set()
            for category in combo_key:
                if category in operation_types:
                    op_categories.add(category)
                else:
                    non_op_categories.add(category)
            return frozenset(op_categories), frozenset(non_op_categories)

        # We only want to keep CATEGORY_OPERATION times.
        # However, the operation-types have replaced CATEGORY_OPERATION.
        compute_overlap = ComputeOverlap(category_times)
        compute_overlap.compute()
        overlap = compute_overlap.get_category_times()
        assert len(overlap) > 0  #FAILS! ...why?

        for category_key in list(overlap.keys()):
            op_categories, non_op_categories = split_combo_key(category_key)
            # If Overlap involves BOTH execution: {CPU, GPU, CPU/GPU},
            #    and an operation: {q_forward, q_backward, ...}
            #   Keep.
            if len(op_categories) == 0 or len(non_op_categories) == 0:
                del overlap[category_key]

        # This can take a while since the timeline can be large...
        # if self.debug:
        #     print("> DEBUG: write process timeline traceEvents @ {path}".format(
        #         path=self._debug_process_timeline_json_path()))
        #     dump_category_times(category_times, self._debug_process_timeline_json_path(), print_log=False)

        # set(operation categories) -> set(non-operation categories) -> [ CPU, GPU, CPU/GPU ] time
        #  <q_forward, q_backward>       <CPU>, <GPU>, <CPU, GPU>             0.001 sec
        operation_overlap = dict()
        for combo_key, time in overlap.items():
            op_categories, non_op_categories = split_combo_key(combo_key)
            assert len(op_categories) > 0
            assert len(non_op_categories) > 0
            if op_categories not in operation_overlap:
                operation_overlap[op_categories] = dict()
            operation_overlap[op_categories][non_op_categories] = time

        if self.debug:
            self._dump_process_timeline_json(operation_overlap)

        return operation_overlap

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
            if type(obj) == dict and len(obj) > 0 and type(next(iter(obj.keys()))) == frozenset:
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
        path = _j(self.directory, 'process_timeline.debug.json')
        return path

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

        sql_reader = SQLiteCategoryTimesReader(self.db_path)

        # Overlap, computed across different "steps".
        overlaps = []

        sql_reader.parse_debug = self.debug
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
            compute_overlap = ComputeOverlap(category_times, overlaps_with=[CATEGORY_OPERATION])
            compute_overlap.compute()
            overlaps.append(compute_overlap.get_category_times())

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

def test_compute_overlap():
    # Set to true to print info.
    debug = False

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

