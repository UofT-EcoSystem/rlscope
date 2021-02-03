"""
Using output of Python profiler to create a flame-graph visualization from call-stack data.
.. deprecated:: 1.0.0
    We don't use the Python profiler anymore.
"""
from rlscope.profiler.rlscope_logging import logger
import re
import numpy as np
import csv
import subprocess
import sys
import time
import os
import csv
import textwrap
import pprint
from io import StringIO
import json
import codecs
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

# pip install progressbar2
import progressbar

from rlscope.parser.common import *
from rlscope.parser import constants
from rlscope.parser.stats import Stats, KernelTime

from rlscope.parser.db import SQLCategoryTimesReader, sql_get_source_files, sql_input_path, process_op_nest_single_thread, each_stack_trace

from rlscope import py_config

FLAME_GRAPH_PERL = _j(py_config.ROOT, 'third_party', 'FlameGraph', 'flamegraph.pl')

class PythonFlameGraphParser:
    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 op_name=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug
        self.op_name = op_name

    def parse_flame_graph(self, op_name, debug_label=None):

        # TODO: we only want samples belonging to <bench_name>
        raise NotImplementedError("Need handle process_category_times[proc], and add process_name to process_events(...) call")
        category_times = self.sql_reader.process_events(
            keep_categories={constants.CATEGORY_PYTHON_PROFILER, constants.CATEGORY_OPERATION},
            op_name=op_name,
            debug=self.debug,
            # fetchall=False,
            fetchall=True,
            debug_label='parse_call_times',
        )

        # category_times[constants.CATEGORY_OPERATION] = process_op_nest_single_thread(
        #     category_times[constants.CATEGORY_OPERATION],
        #     debug=self.debug,
        #     show_progress=True,
        #     debug_label='parse_flame_graph',
        # )

        # use_sample_count = True
        use_sample_count = False

        if use_sample_count:
            sample_count = dict()
        else:
            time_sec_sum = dict()

        for op_stack in each_stack_trace(
            category_times[constants.CATEGORY_PYTHON_PROFILER],
            show_progress=True,
            debug=False,
            debug_label='parse_flame_graph'):

            key = self.flame_graph_key(op_stack)

            if use_sample_count:
                if key not in sample_count:
                    sample_count[key] = 0
                sample_count[key] += 1
            else:
                # Add the time spent in the "leaf" function of this stacktrace.
                if key not in time_sec_sum:
                    time_sec_sum[key] = 0.
                # time_sec_sum[key] += sec_to_ms(op_stack.time_sec)
                time_sec_sum[key] += sec_to_us(op_stack.time_sec)

        if use_sample_count:
            count_data = sample_count
        else:
            count_data = time_sec_sum

        print("> Output python flame-graph input file @ {path}".format(
            path=self._flamegraph_input_path(op_name)))
        with open(self._flamegraph_input_path(op_name), 'w') as f:
            # writer = csv.writer(f, delimiter=' ', lineterminator='\n', quoting=csv.QUOTE_NONE, strict=True)
            for key in sorted(count_data.keys()):
                count = count_data[key]
                f.write(' '.join(map(str, [key, count])))
                f.write("\n")
                # writer.writerow([key, count])

        print("> Output python flame-graph svg file @ {path}".format(
            path=self._flamegraph_svg_path(op_name)))
        with \
            open(self._flamegraph_input_path(op_name), 'r') as input_f, \
            open(self._flamegraph_svg_path(op_name), 'wb') as png_f:

            if use_sample_count:
                argv = []
            else:
                # argv = ['--countname', 'ms']
                argv = ['--countname', 'us']

            subprocess.run(
                ['perl', FLAME_GRAPH_PERL] + argv,
                stdin=input_f,
                stdout=png_f)

    def _flamegraph_input_path(self, op_name):
        return _j(self.directory, 'flamegraph{bench}.txt'.format(
            bench=bench_suffix(op_name),
        ))

    def _flamegraph_svg_path(self, op_name):
        return _j(self.directory, 'flamegraph{bench}.svg'.format(
            bench=bench_suffix(op_name),
        ))

    def flame_graph_time_sec(self, op_stack):
        """
        Return total time spent in the
        :param op_stack:
        :return:
        """

    def flame_graph_key(self, op_stack):
        """
        Return the stacktrace "key" in FlameGraph 'stackcollapse'd format.
        i.e. the format that flamegraph.pl accepts.

        e.g. From https://github.com/brendangregg/FlameGraph @ "2. Fold stacks"

            unix`_sys_sysenter_post_swapgs 1401
            unix`_sys_sysenter_post_swapgs;genunix`close 5
            unix`_sys_sysenter_post_swapgs;genunix`close;genunix`closeandsetf 85
            unix`_sys_sysenter_post_swapgs;genunix`close;genunix`closeandsetf;c2audit`audit_closef 26
            unix`_sys_sysenter_post_swapgs;genunix`close;genunix`closeandsetf;c2audit`audit_setf 5
            unix`_sys_sysenter_post_swapgs;genunix`close;genunix`closeandsetf;genunix`audit_getstate 6
            unix`_sys_sysenter_post_swapgs;genunix`close;genunix`closeandsetf;genunix`audit_unfalloc 2
            unix`_sys_sysenter_post_swapgs;genunix`close;genunix`closeandsetf;genunix`closef 48
            [...]

        e.g. from https://github.com/brendangregg/FlameGraph/blob/1b1c6deede9c33c5134c920bdb7a44cc5528e9a7/stackcollapse.pl
            Example input:

             unix`i86_mwait+0xd
             unix`cpu_idle_mwait+0xf1
             unix`idle+0x114
             unix`thread_start+0x8
             1641

            Example output:

             unix`thread_start;unix`idle;unix`cpu_idle_mwait;unix`i86_mwait 1641

        Documentation from https://github.com/brendangregg/FlameGraph/blob/1b1c6deede9c33c5134c920bdb7a44cc5528e9a7/flamegraph.pl

            The input is stack frames and sample counts formatted as single lines.  Each
            frame in the stack is semicolon separated, with a space and count at the end
            of the line.  These can be generated for Linux perf script output using
            stackcollapse-perf.pl, for DTrace using stackcollapse.pl, and for other tools
            using the other stackcollapse programs.  Example input:

             swapper;start_kernel;rest_init;cpu_idle;default_idle;native_safe_halt 1

            An optional extra column of counts can be provided to generate a differential
            flame graph of the counts, colored red for more, and blue for less.  This
            can be useful when using flame graphs for non-regression testing.
            See the header comment in the difffolded.pl program for instructions.

            The input functions can optionally have annotations at the end of each
            function name, following a precedent by some tools (Linux perf's _[k]):
                _[k] for kernel
                _[i] for inlined
                _[j] for jit
                _[w] for waker
            Some of the stackcollapse programs support adding these annotations, eg,
            stackcollapse-perf.pl --kernel --jit. They are used merely for colors by
            some palettes, eg, flamegraph.pl --color=java.

        :return:
        """
        trace = op_stack.stacktrace()
        for frame in trace:
            assert frame.name is not None
            assert ';' not in frame.name
        key = ';'.join(self.py_name(frame) for frame in trace)
        return key

    def py_name(self, op_stack):
        name = op_stack.name

        m = re.search(r'^<built-in method (?P<func>.*)>$', name)
        if m:
            name = m.group('func')


        # <method 'startswith' of 'str' objects> 315.6939999999996
        # <method 'sub' of '_sre.SRE_Pattern' objects> 4318.467999999995
        m = re.search(r"^<method '(?P<func>.*)' of '(?P<klass>.*)' objects>$", name)
        if m:
            name = "{klass}.{func}".format(
                klass=m.group('klass'),
                func=m.group('func'))

        m = re.search(r'^_pywrap_tensorflow_internal\.(?P<func>.*)', name)
        if m:
            name = m.group('func')

        m = re.search(r'<(?P<func>listcomp|genexpr|lambda)>', name)
        if m:
            name = "<{func}@{file}:{line}>".format(
                func=m.group('func'),
                file=op_stack.ktime.pyprof_filename,
                line=op_stack.ktime.pyprof_line_no)
            # '/home/james/clone/clone/baselines/baselines/common/tf_util.py:165(<lambda>)'
            # I like having the <lambda> at the front instead.
            # name = op_stack.ktime.pyprof_line_description

        return name


    def run(self):
        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

        if self.op_name is not None:
            op_names = [self.op_name]
        else:
            op_names = self.sql_reader.op_names()

        for op_name in op_names:
            self.parse_flame_graph(op_name)

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    @property
    def db_path(self):
        return sql_input_path(self.directory)

class PythonProfileParser:
    """
    Given raw (start_us, end_us) timestamps for python-function calls,
    dump python profiling information in CSV format with these columns:

    - Type
    - Time(%)
    - Time
    - Calls
    - Avg
    - Std
    - Std/Avg(%)
    - Min
    - Max
    - Call#
    - Name

    Input:
    - NOTE: We want to output a python profile for each operation-type
    - List of KernelTime/Event's with pyprof information included in them (NOT NULL)
      - i.e.
        events = Query all the events that are subsumed by (op-nested) constants.CATEGORY_PYTHON_PROFILING
        (NOTE: this is what UtilizationPlot does)

    PROBLEM: How do we select ONLY python call-times specific to a particular operation?
    We need to use the same "overlap after processing-op-nest" queries we've already been doing to construct our plots.
    For nested operations, some python times will span 2 operations;
    for e.g., tree_search python function time will span both the 'tree_search' and 'tree_search_loop' operations.
    To match the plots, we only want to show python-times that are FULLY subsumed by an operation.
    So, we should NOT show tree_search python function time in the profile...
    slightly unintuitive behaviour in this particular scenario, but it is actually the correct behaviour for handling
    other scenarios.
    If you wanted to include the tree_search function start/end time, you need to
    set_operation('tree_search') PRIOR to calling the tree_search function.

    PythonProfileParser:

    PSEUDOCODE:
    # Generator
    def write_python_profile():
        For each op_name:
            pyprof_call_times = query_pyprof_call_times(op_name)
            # NOTE: We must group pyprof events by <file, line, func> in order to report a "row" of the csv
            Report "python_profile.{op_name}.csv" from pyprof_interceptions

    def query_pyprof_call_times(op_name):
        # For each "step"/call to op:
        events = Query constants.CATEGORY_PYTHON_PROFILING/constants.CATEGORY_OPERATION events
                       across ALL steps
                       across ALL processes
                 # ORDER BY <pyprof_filename, pyprof_line_no, pyprof_function>
                 # NOTE: I don't think there's a nice way to preserve this ORDER BY across process_op_nest...
        events[constants.CATEGORY_OPERATION] = process_op_nest(events[constants.CATEGORY_OPERATION])
        pyprof_interceptions = events.filter {
            Event.category == constants.CATEGORY_PYTHON_PROFILING and
              Event is SUBSUMED by an event with op_type == op.type              # <-- NOTE: we can us OpStack to compute this.
        }
        pyprof_call_times = {
            'call_times':
              # NOTE: bins should be sorted by start_time_us (guaranteed since pyprof_interceptions is already sorted)
              Bin pyprof_interceptions into a dict where key = <pyprof_filename, pyprof_line_no, pyprof_function>,
        }
        return pyprof_call_times

    """
    def __init__(self, directory,
                 host=None,
                 user=None,
                 password=None,
                 debug=False,
                 # Swallow any excess arguments
                 **kwargs):
        self.directory = directory
        self.host = host
        self.user = user
        self.password = password
        self.debug = debug

        self.num_calls = None
        self.discard_first_sample = False

        self.time_fields = ['tottime', 'cumtime', 'tottime_percall', 'cumtime_percall']
        self.total_time_fields = ['tottime_seconds']
        self.sortby = ('tottime_seconds', 'filename:lineno(function)')

        # self.pyfunc_stats = Stats(self.discard_first_sample, debug=self.args.debug)
        # self.pyfunc_stats = Stats(discard_first_sample=False, debug=self.debug)
        # self.convert_to_seconds = convert_to_seconds

    def get_source_files(self):
        return sql_get_source_files(self.__class__, self.directory)

    # def _config_path(self, bench_name):
    #     return _j(self.directory(bench_name), "config.json")

    # def _parse_num_calls(self, bench_name):
    #     call_times_path = self._call_times_path(bench_name)
    #     logger.info("> Parsing call times: {f}".format(f=call_times_path))
    #     # self.call_times = self.load_call_times(bench_name)
    #     # if self.is_dqn:
    #     #     micro_data = self.load_microbench(bench_name)
    #     #     micro_name = self.get_micro_name()
    #     #     bench_data = micro_data[micro_name]
    #     #     # This is the number of times (for e.g.) Forward was called.
    #     #     # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
    #     #     # num_calls = iterations*repetitions
    #     #     self.num_calls = compute_num_calls_dqn(bench_data)
    #     # else:
    #     #     # Already read it from self.config in __init__
    #     #     assert self.num_calls is not None
    #     logger.info("> num_calls = {num_calls}".format(
    #         num_calls=self.num_calls))

    def run(self):
        self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password)

        op_names = self.sql_reader.op_names()
        for op_name in op_names:
            self.parse_call_times(op_name)

        # # from UtilizationPlot:
        # self.sql_reader = SQLCategoryTimesReader(self.db_path, host=self.host, user=self.user, password=self.password, debug_ops=self.debug_ops)
        # # self.bench_names = self.sql_reader.bench_names(self.debug_ops) + [NO_BENCH_NAME]
        # # assert len(self.bench_names) == len(unique(self.bench_names))
        # # self.categories = self.sql_reader.categories
        #
        # overlap_computer = OverlapComputer(self.db_path, host=self.host, user=self.user, password=self.password, debug=self.debug, debug_ops=self.debug_ops)
        #
        # operation_overlap, proc_stats = overlap_computer.compute_process_timeline_overlap(
        #     debug_memoize=self.debug_memoize)
        # assert len(operation_overlap) > 0

        # NOTE: we just want to keep RAW events that overlap with constants.CATEGORY_PYTHON_PROFILING events...
        # This isn't QUITE what ComputeOverlap does.
        # ComputeOverlap is incrementally computing the total-sum of time-overlap between category types,
        # but it DISCARDS the raw events when doing this.
        # that's NOT what this does.


    def _call_times_summary_path(self, bench_name):
        path = _j(self.directory, "pyprof_call_times{bench}.txt".format(bench=bench_suffix(bench_name)))
        return path

    @property
    def db_path(self):
        return sql_input_path(self.directory)

    def parse_call_times(self, op_name, debug_label=None):

        pyfunc_stats = Stats(discard_first_sample=False, debug=self.debug)

        # call_times_path = self._call_times_path(op_name)
        # # if not _e(call_times_path):
        # #     return
        # logger.info("> Parsing call times: {f}".format(f=call_times_path))
        # call_times = self.load_call_times(op_name)
        # micro_data = self.load_microbench(op_name)
        # micro_name = self.get_micro_name()
        # bench_data = micro_data[micro_name]
        # # This is the number of times (for e.g.) Forward was called.
        # # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
        # # num_calls = iterations*repetitions
        # self.num_calls = compute_num_calls_dqn(bench_data)
        # logger.info("> num_calls = {num_calls}".format(
        #     num_calls=self.num_calls))
        # start_parse_call_times = time.time()
        # self._parse_num_calls(op_name)

        raise NotImplementedError("Need handle process_category_times[proc], and add process_name to process_events(...) call")
        category_times = self.sql_reader.process_events(
            keep_categories={constants.CATEGORY_PYTHON_PROFILER, constants.CATEGORY_OPERATION},
            op_name=op_name,
            debug=self.debug,
            # fetchall=False,
            fetchall=True,
            debug_label='parse_call_times',
        )

        if len(category_times) == 0:
            logger.info("> WARNING: cannot generate python profile for op={op}; no events found for any process.".format(
                op=op_name))
            return

        # JAMES TODO: process_op_nest(category_times[CATGEORY_OPERATION])
        # replace call_times below with category_times[constants.CATEGORY_PYTHON_PROFILING]; handle usec
        # call_times = ...

        # self.num_calls = min(len(times) for times in category_times.values())
        # for times in category_times.values():
        #     assert self.num_calls == len(times)

        self.num_calls = len(category_times[constants.CATEGORY_OPERATION])

        # end_parse_call_times = time.time()
        # logger.info("> Parsing call times took: {sec} seconds".format(sec=end_parse_call_times - start_parse_call_times)) # 45 seconds, ~ 6-7 GB

        # logger.info("> Adding call times to pyfunc_stats:") # 249 seconds / 4min 9sec; this takes up a LOT of memory: 34.5 GIG.
        start_add_call_times = time.time()
        progress_label = as_progress_label('parse_call_times', debug_label)
        for i, ktime in enumerate(progress(category_times[constants.CATEGORY_PYTHON_PROFILER],
                                           desc=progress_label,
                                           show_progress=self.debug)):
            # if self.config['clock'] == 'monotonic_clock':
            #     time_secs = call_times
            # else:
            #     raise NotImplementedError
            # time_sec = ktime.total_time_sec
            # pyfunc_stats.add_time_sec(
            #     ktime.pyprof_function,
            #     time_sec)
            # Q: Why dont we add the KernelTime entireyl?
            assert type(ktime) == KernelTime
            pyfunc_stats.add_ktime(ktime)
        end_add_call_times = time.time()
        logger.info("> Adding call times to pyfunc_stats took: {sec} seconds".format(sec=end_add_call_times - start_add_call_times))

        # TODO: convert cycles to usec!

        # Q: I don't think it makes sense to call 'split'...?
        logger.info("> Split pyfunc stats:") # 20 seconds
        start_split_call_times = time.time()
        pyfunc_stats.split(self.num_calls)
        end_split_call_times = time.time()
        logger.info("> Split pyfunc stats took: {sec} seconds".format(sec=end_split_call_times - start_split_call_times))

        # This takes a while and we don't use it; skip.
        logger.info("> Dumping pyfunc stats:") # 467 seconds
        start_dump_call_times = time.time()
        with open(self._call_times_summary_path(op_name), 'w') as f:
            # with open(self._call_times_pyprof_path(op_name), 'w') as f_pyprof:
            pyfunc_stats.dump_separate_calls(f, 'Python')
            # Dump profile similar to cProfile/pstats output so
            # we can make sure we're doing things correctly.
            # pyfunc_stats.dump_nvprof(f_pyprof, 'Python')
        end_dump_call_times = time.time()
        logger.info("> Dumping pyfunc stats took: {sec} seconds".format(sec=end_dump_call_times - start_dump_call_times))
        logger.info("> Parsed call times into: {f}".format(f=self._call_times_summary_path(op_name)))

    def total_iterations(self, bench_name):
        if self.num_calls is None:
            self._parse_num_calls(bench_name)
            assert self.num_calls is not None

        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls


class PythonProfileTotalParser(ProfilerParser):
    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME):

        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths

        super().__init__(parser, args, src_files,
                         bench_name=bench_name)

    @staticmethod
    def required_source_basename_regexes():
        return {
            "python_overhead_path":"^python_profile_python_overhead_pyprof{bench}.json$".format(bench=BENCH_SUFFIX_RE),
        }

    @property
    def _python_pyprof_total_path(self):
        return self.get_python_pyprof_total_path(self.src_files)
    @classmethod
    def get_python_pyprof_total_path(Klass, src_files):
        return PythonProfileParser.get_python_overhead_path(src_files, bench_name='total')

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [
            Klass.get_python_pyprof_total_path(src_files),
        ]

    @staticmethod
    def optional_source_basename_regexes():
        return {'microbenchmark_json':r"^microbenchmark.json$",
                'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE)}

    @staticmethod
    def allow_multiple_src_matches():
        return True

    @staticmethod
    def uses_all_benches():
        return True

    @staticmethod
    def uses_multiple_dirs():
        return False

    def run(self, bench_name=NO_BENCH_NAME):
        """
        NOTE: This ought to be its own parser...
        :return:
        """
        parser = self.parser
        args = self.args
        directory = self.src_files.directory
        bench_names = [bench_name for bench_name in self.src_files.bench_names \
                       if bench_name != 'total']
        # bench_names = DQN_BENCH_NAMES
        # profile_paths = [_j(directory, 'python_profile.{b}.txt'.format(b=bench_name))
        #                  for bench_name in bench_names]
        # if not all(_e(path) for path in profile_paths):
        #     return

        path = self._python_pyprof_total_path
        # self.get_python_overhead_path()
        # self.get_python_overhead_path(self.src_files, bench_name='total')
        # path = _j(directory, _pyprof_total_base())

        # if _e(path):
        #     if os.path.getsize(path) == 0:
        #         os.remove(path)
        #     else:
        #         return

        cpp_and_gpu_times = None
        python_times = None
        smallest_length = None
        raw_datas = []
        profile_paths = []
        for bench_name in bench_names:
            # prof = PythonProfileParser(parser, args, profile_path, bench_name)
            # prof = PythonProfileParser(parser, args, self.src_files,
            #                            bench_name=bench_name)
            # raw_data = prof.parse_python_overhead(bench_name, dump_json=False, get_raw_data=True)
            python_overhead_path = PythonProfileParser.get_python_overhead_path(self.src_files, bench_name)
            raw_data = load_json(python_overhead_path)
            if smallest_length is None or smallest_length > len(raw_data['CppAndGPUTimeSec']):
                smallest_length = len(raw_data['CppAndGPUTimeSec'])
            raw_datas.append(raw_data)
            profile_paths.append(python_overhead_path)

        logger.info("> Using the first {n} samples from each DQN stage to compute Total...".format(
            n=smallest_length))

        cpp_and_gpu_times = np.zeros(smallest_length)
        python_times = np.zeros(smallest_length)
        for bench_name, raw_data in zip(bench_names, raw_datas):
            cpp_and_gpu_times += raw_data['CppAndGPUTimeSec'][0:smallest_length]
            python_times += raw_data['PythonTimeSec'][0:smallest_length]

        data = compute_plot_data(cpp_and_gpu_times, python_times)
        do_dump_json(data, path)
        logger.info("> Output pyprof Total: {f}".format(f=path))

        # parse_pyprof_total(self.directory(bench_name), src_files.bench_names, self.parser, self.args)
        # self.parse_pyprof_total()

class _FreqParser:
    """
    > Running on CPU: 14
    > cycles[0] = 12021055605 cycles
    > seconds[0] = 5 sec
    > cpu_freq[0] = 2.40421 GHz
    > cycles[1] = 12021049029 cycles
    > seconds[1] = 5 sec
    > cpu_freq[1] = 2.40421 GHz
    > cycles[2] = 12021048798 cycles
    > seconds[2] = 5 sec
    > cpu_freq[2] = 2.40421 GHz
    > cycles[3] = 12021049617 cycles
    > seconds[3] = 5 sec
    > cpu_freq[3] = 2.40421 GHz
    > cycles[4] = 12021049059 cycles
    > seconds[4] = 5 sec
    > cpu_freq[4] = 2.40421 GHz
    > cycles[5] = 12021051003 cycles
    > seconds[5] = 5 sec
    > cpu_freq[5] = 2.40421 GHz
    > cycles[6] = 12021049767 cycles
    > seconds[6] = 5 sec
    > cpu_freq[6] = 2.40421 GHz
    > cycles[7] = 12021049392 cycles
    > seconds[7] = 5 sec
    > cpu_freq[7] = 2.40421 GHz
    > cycles[8] = 12021050112 cycles
    > seconds[8] = 5 sec
    > cpu_freq[8] = 2.40421 GHz
    > cycles[9] = 12021050535 cycles
    > seconds[9] = 5 sec
    > cpu_freq[9] = 2.40421 GHz
    > Mean CPU frequency: 2.40421 GHz
    > Std CPU frequency: 3.77961e-07 GHz
    > Num measurements: 10

    {
    'repetitions':10,
    'cpu_freq_ghz':[ ... ]
    'cpu_freq_ghz_mean':...,
    'cpu_freq_ghz_std':...,
    'cpu_id':...,
    }
    """
    def __init__(self, parser, args,
                 rep_array_name_regex=r"cpu_freq",
                 stat_name_regex=r"CPU frequency",
                 field_name='cpu_freq_ghz'):
        self.parser = parser
        self.args = args
        self.rep_array_name_regex = rep_array_name_regex
        self.stat_name_regex = stat_name_regex
        self.field_name = field_name
        self.mean_name = '{f}_mean'.format(f=self.field_name)
        self.std_name = '{f}_std'.format(f=self.field_name)

    def parse(self, it, all_results):

        # self.results = dict()
        def store(*args, **kwargs):
            store_group(all_results, *args, **kwargs)

        for line in line_iter(it):
            m = re.search(r'> Running on CPU: (?P<cpu_id>\d+)', line)
            if m:
                store(m)
                continue

            m = re.search(r'> Num measurements: (?P<repetitions>\d+)', line)
            if m:
                store(m)
                continue

            regex = r'> {array_name}\[\d+\]\s*[=:]\s*(?P<freq_ghz>{float}) GHz'.format(
                float=float_re,
                array_name=self.rep_array_name_regex)
            m = re.search(regex, line)
            if m:
                store_as(all_results,
                         self.field_name,
                         float(m.group('freq_ghz')),
                         store_type='list')
                continue

            m = re.search(r'> Mean {stat_name}\s*[=:]\s*(?P<freq_ghz_mean>{float}) GHz'.format(
                float=float_re,
                stat_name=self.stat_name_regex), line)
            if m:
                store_as(all_results,
                         self.mean_name,
                         float(m.group('freq_ghz_mean')))
                assert type(all_results[self.mean_name]) == float
                continue

            m = re.search(r'> Std {stat_name}\s*[=:]\s*(?P<freq_ghz_std>{float}) GHz'.format(
                float=float_re,
                stat_name=self.stat_name_regex), line)
            if m:
                store_as(all_results,
                         self.std_name,
                         float(m.group('freq_ghz_std')))
                continue

        expected_keys = set([
            'repetitions',
            self.field_name,
            self.mean_name,
            self.std_name,
            'cpu_id',
        ])
        missing_keys = expected_keys.difference(set(all_results.keys()))
        assert len(missing_keys) == 0

        assert type(all_results[self.mean_name]) == float
        return all_results

def compute_plot_data(cpp_and_gpu_times, python_times):
    total_times = compute_total_times(cpp_and_gpu_times, python_times)
    theoretical_speedup = compute_theoretical_speedup(cpp_and_gpu_times, python_times)
    percent_time_in_python = compute_percent_time_in_python(cpp_and_gpu_times, python_times)
    python_overhead_percent = compute_python_overhead_percent(cpp_and_gpu_times, python_times)

    def _no_nans(xs):
        assert not np.isnan(xs).any()

    # assert not (cpp_times == 0.).any()
    _no_nans(cpp_and_gpu_times)
    _no_nans(python_times)
    _no_nans(percent_time_in_python)

    data = {
        "TotalTimeSec":total_times,
        "CppAndGPUTimeSec":cpp_and_gpu_times,
        "TheoreticalSpeedup":theoretical_speedup,
        "PercentTimeInPython":percent_time_in_python,
        "PythonTimeSec":python_times,
        "PythonOverheadPercent":python_overhead_percent,
    }
    data = make_json_serializable(data)

    for field in data:
        assert field in PLOT_SUMMMARY_FIELDS

    return data

