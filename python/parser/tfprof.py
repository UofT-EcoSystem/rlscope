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
from proto.tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
from proto.protobuf.pyprof_pb2 import Pyprof

from parser.stats import KernelTime

class TFProfCategoryTimesReader:
    def __init__(self, profile_path):
        self.profile_path = profile_path
        with open(self.profile_path, 'rb') as f:
            self.proto = ProfileProto()
            self.proto.ParseFromString(f.read())

    def parse(self, step):

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

            # self.kernel_times()
            # self.api_times()
            # print("HELLO PROTO 1")
            # print(self.proto)
            # PSEUDOCODE:
            # for step in steps:
            #
            # import ipdb; ipdb.set_trace()
            # print("HELLO PROTO 2")

        # for step in self.proto.steps:
        # Categories:
        # - [CUDA API CPU]
        # - [GPU]
        # - Future: [Framework CPU]
        category_times = dict()

        for node_id, node in self.proto.nodes.items():
            if step not in node.execs.keys():
                continue
            self.add_all_times(category_times, step, node, self.get_accelerator_execs)
            self.add_all_times(category_times, step, node, self.get_cpu_execs)

        return category_times

    def IsGPUTime(self, device):
        return re.search('stream:all', device)

    def IsCPUTime(self, device):
        return re.search(".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+", device)

    def add_time(self, category_times, device, ktime):
        if self.IsGPUTime(device):
            if CATEGORY_GPU not in category_times:
                category_times[CATEGORY_GPU] = []
            category_times[CATEGORY_GPU].append(ktime)
        elif self.IsCPUTime(device):
            if CATEGORY_CUDA_API_CPU not in category_times:
                category_times[CATEGORY_CUDA_API_CPU] = []
            category_times[CATEGORY_CUDA_API_CPU].append(ktime)
        else:
            raise NotImplementedError("Not sure what category device={dev} falls under.".format(dev=device))

    def add_all_times(self, category_times, step, node, get_execs):
        """
        :param get_execs:
            ExecProfile -> map<string, ExecTime>
            Either reteurn accelerator_execs or cpu_execs
        """
        exec_profile = node.execs[step]
        execs = get_execs(exec_profile)
        for device in execs.keys():
            for tupl in execs[device].times:
                start_us, duration_us = tupl.int64_values
                ktime = KernelTime(start_usec=start_us, time_usec=duration_us)
                self.add_time(category_times, device, ktime)

    def get_accelerator_execs(self, exec_profile):
        return exec_profile.accelerator_execs

    def get_cpu_execs(self, exec_profile):
        return exec_profile.cpu_execs

class PyprofCategoryTimesReader:
    def __init__(self, profile_path):
        self.profile_path = profile_path
        with open(self.profile_path, 'rb') as f:
            self.proto = Pyprof()
            self.proto.ParseFromString(f.read())

    def parse(self, step):
        assert step in self.proto.steps

        category_times = dict()

        category_times[CATEGORY_PYTHON] = []
        self.add_event_times_to(category_times[CATEGORY_PYTHON], self.proto.python_events[step].events)

        # clib_times = dict()
        for category, clib_events in self.proto.clibs[step].clibs.items():
            assert category not in category_times
            category_times[category] = []
            self.add_event_times_to(category_times[category], clib_events.events)

        return category_times

    def add_event_times_to(self, ktimes, events):
        for event in events:
            ktime = KernelTime(start_usec=event.start_time_us, time_usec=event.duration_us)
            ktimes.append(ktime)

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

CATEGORY_PYTHON = 'Python'
CATEGORY_CUDA_API_CPU = 'CUDA API CPU'
CATEGORY_GPU = 'GPU'

class TFProfParser(ProfilerParserCommonMixin):
    """
    What do we want to compute?

    - TotalTimeSec
    - CppAndGPUTimeSec

    - CppTimeSec
      Comes from python.

    - FrameworkCppTimeSec

    - CudaCppTimeSec
      Should be able to compute this...but will it include kernel time?
      TODO: plot this in timeline view.

    - PercentTimeInGPU
    - GPUTimeSec
      Should be able to compute this.

    - GPUAndCudaCppTimeSec
    - TheoreticalSpeedup
    - PercentTimeInPython
    - PythonTimeSec
    - PythonOverheadPercent
    """

    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths
        self.src_files = src_files

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
            'tfprof_path': r"^profile_\d+{bench}.proto$".format(bench=BENCH_SUFFIX_RE),
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

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [
            Klass.get_gpu_overhead_path(src_files, bench_name),
            Klass.get_pretty_profile_path(src_files, bench_name),
            Klass.get_variable_path(src_files, bench_name),
        ]

    def tfprof_path(self, bench_name, or_none=True):
        return self.src_files.get('tfprof_path', self.bench_name, or_none=or_none)

    def pyprof_path(self, bench_name):
        return self.src_files.get('pyprof_path', self.bench_name)

    def get_micro_name(self):
        return self.bench_name

    def read_category_times(self, step, category_times_readers):
        category_times = dict()
        for reader in category_times_readers:
            new_times = reader.parse(step)
            same_categories = set(new_times.keys()).intersection(set(category_times.keys()))
            assert len(same_categories) == 0
            category_times.update(new_times)
        return category_times

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

        print("TFProfParser > steps={name}".format(name=steps_name))
        print("> steps = ")
        pprint.pprint({'len(steps)':len(steps),
                       'steps':steps})

        if self.tf_proto is not None:
            for tf_step in self.tf_proto.steps:
                assert tf_step in self.py_proto.steps

        # import ipdb; ipdb.set_trace()

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

        tfprof_path = self.tfprof_path(self.bench_name)
        if tfprof_path is not None:
            tfprof_reader = TFProfCategoryTimesReader(tfprof_path)
            category_times_readers.append(tfprof_reader)

        pyprof_path = self.pyprof_path(self.bench_name)
        pyprof_reader = PyprofCategoryTimesReader(pyprof_path)
        category_times_readers.append(pyprof_reader)

        # Overlap, computed across different "steps".
        overlaps = []

        for step in steps:
            category_times = self.read_category_times(step, category_times_readers)
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
        do_dump_json(json_output, self._profile_json_path)

    # From TensorFlow code base:
    #
    # bool CountAsAcceleratorTime(const string& device) {
    # return device.find("stream:all") != device.npos;
    # }
    # bool CountAsCPUTime(const string& device) {
    # return RE2::FullMatch(device,
    #                       ".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+");
    # }

    def IsGPUTime(self, device):
        return re.search('stream:all', device)

    def IsCPUTime(self, device):
        return re.search(".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+", device)

    def dump(self, bench_name):
        if self.skip:
            return

    @property
    def _profile_json_path(self):
        return self.get_profile_json_path(self.src_files, self.bench_name)

    @classmethod
    def get_profile_json_path(ParseKlass, src_files, bench_name):
        path = _j(src_files.directory, 'tfprof{bench}.json'.format(bench=bench_suffix(bench_name)))
        return path

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

