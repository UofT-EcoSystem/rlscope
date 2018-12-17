import re
import numpy as np
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

from parser.common import *
from parser.stats import Stats

class PythonProfileParser(ProfilerParser):
    def __init__(self, parser, args, src_files,
                 convert_to_seconds=True, data=None, bench_name=NO_BENCH_NAME):

        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths

        super().__init__(parser, args, src_files,
                         data=data,
                         bench_name=bench_name)

        self.config_path = src_files.get('config_json', bench_name, or_none=True)
        if self.config_path is not None:
            self.config = load_json(self.config_path)
            print("> Found optional config_json @ {f}".format(f=self.config_path))
        else:
            self.config = {
                'clock':'monotonic_clock',
            }

        if not self.is_dqn or 'num_calls' in self.config:
            self.num_calls = compute_num_calls(self.config)
        else:
            # Compute later?
            self.num_calls = None

        if 'discard_first_sample' in self.config:
            self.discard_first_sample = self.config['discard_first_sample']
        else:
            self.discard_first_sample = self.args.discard_first_sample

        self.time_fields = ['tottime', 'cumtime', 'tottime_percall', 'cumtime_percall']
        self.total_time_fields = ['tottime_seconds']
        self.sortby = ('tottime_seconds', 'filename:lineno(function)')

        self.pyfunc_stats = Stats(self.discard_first_sample, debug=self.args.debug)
        self.convert_to_seconds = convert_to_seconds

        # config_path = self._config_path(bench_name)
        # if _e(config_path):
        # self.config = load_json(self._config_path(bench_name))

        # if self.is_dqn:
        #     self.config = load_json(self._config_path(bench_name))
        # else:
        #     self.config = {
        #         'clock':'monotonic_clock',
        #     }

    @property
    def debug(self):
        return self.args.debug

    @staticmethod
    def required_source_basename_regexes():
        return {
            'profile_path': r"^python_profile{bench}\.txt$".format(bench=BENCH_SUFFIX_RE),
            'python_call_times':"^python_profile{bench}\.call_times.json$".format(bench=BENCH_SUFFIX_RE),
            'config_json':r"^config{bench}\.json$".format(bench=BENCH_SUFFIX_RE),
        }

    @staticmethod
    def optional_source_basename_regexes():
        return {'microbenchmark_json':r"^microbenchmark.json$",
                }

    @classmethod
    def test_targets(ParserKlass):
        paths = [
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.q_backward.pretty.csv',
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.q_forward.pretty.csv',
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.q_update_target_network.pretty.csv',
            'checkpoints/PongNoFrameskip-v4/glue/gpu/quadro_p4000.new/microbenchmark/python_profile.step.pretty.csv',
        ]
        src_files = ParserKlass.glob_target_files(paths, debug=True)
        matching_srcs = src_files.all_sources(all_bench_names=True)
        assert set(paths).issubset(set(matching_srcs))

    @staticmethod
    def target_basename_regexes():
        return {
            'pretty_profile_path': r"^python_profile{bench}\.pretty\.csv$".format(bench=BENCH_SUFFIX_RE),
            'python_overhead_path': r"^python_profile{bench}\.python_overhead\.pyprof\.json".format(bench=BENCH_SUFFIX_RE),
            # Klass.get_call_times_path(src_files, bench_name),
            # 'total_path':r"^python_profile.total.python_overhead.pyprof.json$",
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

    def pre_parse(self, bench_name):
        pass
        # We only use monotonic_clock, no longer use cycle counter since we
        # had issues getting it to work.
        # if self.convert_to_seconds:
        #     self.tsc_freq = BenchmarkTSCFreq(self.parser, self.args)
        #     self.tsc_freq.run()

    @classmethod
    def get_targets(Klass, src_files, bench_name):
        return [Klass.get_pretty_profile_path(src_files, bench_name),
                Klass.get_python_overhead_path(src_files, bench_name),
                # Klass.get_python_pyprof_total_path(src_files),

                # Klass.get_call_times_path(src_files, bench_name),
                Klass.get_call_times_summary_path(src_files, bench_name),
                Klass.get_call_times_pyprof_path(src_files, bench_name),
                ]

    def _config_path(self, bench_name):
        return _j(self.directory(bench_name), "config.json")


    @classmethod
    def get_call_times_path(self, src_files, bench_name):
        # ret = _j(src_files.directory, "python_profile_call_times{bench}.json".format(bench=bench_suffix(bench_name)))
        ret = _j(src_files.directory, "python_profile{bench}.call_times.json".format(bench=bench_suffix(bench_name)))
        # call_times_path = re.sub(r'.txt$', '.call_times.json', self.profile_path(bench_name))
        # return call_times_path
        return ret
    def _call_times_path(self, bench_name):
        return self.get_call_times_path(self.src_files, bench_name)
        # call_times_path = re.sub(r'.txt$', '.call_times.json', self.profile_path(bench_name))
        # return call_times_path

    @classmethod
    def get_call_times_summary_path(ParserKlass, src_files, bench_name):
        # profile_path = src_files.get('profile_path', bench_name)
        ret = _j(src_files.directory, "python_profile_call_times{bench}.txt".format(bench=bench_suffix(bench_name)))
        # ret = re.sub(r'.txt$', '.call_times.txt', profile_path)
        # assert ret != profile_path
        return ret
    def _call_times_summary_path(self, bench_name):
        return self.get_call_times_summary_path(self.src_files, bench_name)

    @classmethod
    def get_call_times_pyprof_path(ParserKlass, src_files, bench_name):
        # profile_path = src_files.get('profile_path', bench_name)
        # ret = re.sub(r'.txt$', '.call_times.pyprof.txt', profile_path)
        ret = _j(src_files.directory, "python_profile_call_times_pyprof{bench}.txt".format(bench=bench_suffix(bench_name)))
        # assert ret != profile_path
        return ret
    def _call_times_pyprof_path(self, bench_name):
        return self.get_call_times_pyprof_path(self.src_files, bench_name)

    def _python_overhead_path(self, bench_name):
        return self.get_python_overhead_path(self.src_files, bench_name)

    @classmethod
    def get_python_overhead_path(Klass, src_files, bench_name):
        ret = _j(src_files.directory, "python_profile_python_overhead_pyprof{bench}.json".format(bench=bench_suffix(bench_name)))
        # ret = re.sub(r'.txt$', '.python_overhead.pyprof.json', src_files.get('profile_path', bench_name))
        # assert ret != src_files.get('profile_path', bench_name)
        return ret

    def _microbench_path(self, bench_name):
        return get_microbench_path(_d(self.profile_path(bench_name)))

    def load_microbench(self, bench_name):
        microbench_path = self._microbench_path(bench_name)
        assert _e(microbench_path)

        with codecs.open(microbench_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            data = fixup_json(data)
            return data

    def get_micro_name(self):
        # if self.args.c_only:
        #     return get_c_only_name(self.bench_name)
        return self.bench_name

    def _parse_num_calls(self, bench_name):
        call_times_path = self._call_times_path(bench_name)
        print("> Parsing call times: {f}".format(f=call_times_path))
        self.call_times = self.load_call_times(bench_name)
        if self.is_dqn:
            micro_data = self.load_microbench(bench_name)
            micro_name = self.get_micro_name()
            bench_data = micro_data[micro_name]
            # This is the number of times (for e.g.) Forward was called.
            # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
            # num_calls = iterations*repetitions
            self.num_calls = compute_num_calls_dqn(bench_data)
        else:
            # Already read it from self.config in __init__
            assert self.num_calls is not None
        print("> num_calls = {num_calls}".format(
            num_calls=self.num_calls))

    def parse_call_times(self, bench_name):

        # call_times_path = self._call_times_path(bench_name)
        # # if not _e(call_times_path):
        # #     return
        # print("> Parsing call times: {f}".format(f=call_times_path))
        # call_times = self.load_call_times(bench_name)
        # micro_data = self.load_microbench(bench_name)
        # micro_name = self.get_micro_name()
        # bench_data = micro_data[micro_name]
        # # This is the number of times (for e.g.) Forward was called.
        # # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
        # # num_calls = iterations*repetitions
        # self.num_calls = compute_num_calls_dqn(bench_data)
        # print("> num_calls = {num_calls}".format(
        #     num_calls=self.num_calls))
        start_parse_call_times = time.time()
        self._parse_num_calls(bench_name)
        end_parse_call_times = time.time()
        print("> Parsing call times took: {sec} seconds".format(sec=end_parse_call_times - start_parse_call_times)) # 45 seconds, ~ 6-7 GB

        print("> Adding call times to pyfunc_stats:") # 249 seconds / 4min 9sec; this takes up a LOT of memory: 34.5 GIG.
        start_add_call_times = time.time()
        with progressbar.ProgressBar(max_value=len(self.call_times.items())) as bar:
            for i, (func_name, call_times) in enumerate(self.call_times.items()):
                if self.config['clock'] == 'monotonic_clock':
                    time_secs = call_times
                else:
                    raise NotImplementedError

                self.pyfunc_stats.add_times_sec(func_name, time_secs)

                bar.update(i)
        end_add_call_times = time.time()
        print("> Adding call times to pyfunc_stats took: {sec} seconds".format(sec=end_add_call_times - start_add_call_times))

        # TODO: convert cycles to usec!

        print("> Split pyfunc stats:") # 20 seconds
        start_split_call_times = time.time()
        self.pyfunc_stats.split(self.num_calls)
        end_split_call_times = time.time()
        print("> Split pyfunc stats took: {sec} seconds".format(sec=end_split_call_times - start_split_call_times))

        # This takes a while and we don't use it; skip.
        print("> Dumping pyfunc stats:") # 467 seconds
        start_dump_call_times = time.time()
        with open(self._call_times_summary_path(bench_name), 'w') as f:
            # with open(self._call_times_pyprof_path(bench_name), 'w') as f_pyprof:
            self.pyfunc_stats.dump_separate_calls(f, 'Python')
            # Dump profile similar to cProfile/pstats output so
            # we can make sure we're doing things correctly.
            # self.pyfunc_stats.dump_nvprof(f_pyprof, 'Python')
        end_dump_call_times = time.time()
        print("> Dumping pyfunc stats took: {sec} seconds".format(sec=end_dump_call_times - start_dump_call_times))
        print("> Parsed call times into: {f}".format(f=self._call_times_summary_path(bench_name)))

    def total_iterations(self, bench_name):
        if self.num_calls is None:
            self._parse_num_calls(bench_name)
            assert self.num_calls is not None

        if self.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    # TODO: make this unit-testable (in particular, when we gather python_times/PythonTimeSec
    def parse_python_overhead(self, bench_name, dump_json=True, get_raw_data=False):
        self.parse_call_times(bench_name)

        print("> Parsing python overhead: {f}".format(f=self._python_overhead_path(bench_name)))

        # TODO: This is tensorflow specific, we need to try wrapping end-to-end test
        # and making this work for all C++ libraries
        if self.is_dqn:
            default_cpp_func_re = r'(?:built-in.*pywrap_tensorflow)'
        else:
            default_cpp_func_re = r"CLIB__.*"
        cpp_func_re = self.config.get('c_lib_func_pyprof_pattern', default_cpp_func_re)

        def is_cpp_func(func_name):
            m = re.search(cpp_func_re, func_name)
            return bool(m)
        cpp_and_gpu_times = np.zeros(self.total_iterations(bench_name))
        python_times = np.zeros(self.total_iterations(bench_name))
        # 1000 iterations
        # 22000 calls
        # 22 different calls to this function in each iteration

        print("> Compute python overhead")
        start_compute_python_overhead = time.time()
        if self.debug:
            print("> cpp_func_re = {regex}".format(regex=cpp_func_re))
        for stat in self.pyfunc_stats.stats:
            # Returns an array of size self.num_calls, one entry for each time
            # an "iteration" of the expierment was run.
            times_sec = stat.iteration_times_sec(self.num_calls)
            if self.debug:
                print("> stat.name = {name}".format(name=stat.name))
                if stat.name == "/mnt/data/james/clone/dnn_tensorflow_cpp/python/test/py_interface.py:78(CLIB__run_cpp)":
                    import ipdb; ipdb.set_trace()
            if is_cpp_func(stat.name):
                cpp_and_gpu_times += times_sec
            else:
                python_times += times_sec
        end_compute_python_overhead = time.time()
        print("> Compute python overhead took: {sec} seconds".format(sec=end_compute_python_overhead - start_compute_python_overhead))

        total_times = cpp_and_gpu_times + python_times

        # NOTE: The idea behind this calculation is that in an ideal case,
        # we only need to execute the C code, and the python code doesn't need to be executed.
        # So, overhead = time(python)/time(C)
        # NOT:
        #   overhead = time(python)/time(python + C)
        #   This is what the profiler tells us.
        python_overhead_percent = 100.*python_times/cpp_and_gpu_times

        raw_data = {
            "TotalTimeSec":total_times,
            "CppAndGPUTimeSec":cpp_and_gpu_times,
            "PythonTimeSec":python_times,
            "PythonOverheadPercent":python_overhead_percent,
        }
        if get_raw_data:
            return raw_data

        data = compute_plot_data(cpp_and_gpu_times, python_times)
        if dump_json:
            do_dump_json(data, self._python_overhead_path(bench_name))


        return data

    def directory(self, bench_name):
        return _d(self.profile_path(bench_name))

    def load_call_times(self, bench_name):
        # JAMES TODO: put this into the parser class for the python profiling data; produce output similar to what we did for CUDA stuff.
        call_times_path = self._call_times_path(bench_name)
        assert _e(call_times_path)

        with codecs.open(call_times_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            # data = fixup_json(data)
            return data

    def sec_as_usec(self, sec):
        return sec * 1e6

    def parse_header(self, line, it):
        assert self.header is None
        if re.search(r'^\s*ncalls\s*tottime', line):
            self.header = re.split(r'\s+', line.strip())
            for i in range(len(self.header)):
                if self.header[i] == 'percall':
                    self.header[i] = "{last_col}_percall".format(last_col=self.header[i-1])
            return True
        return False

    def parse_other(self, line, it):
        m = re.search(r'\s*(?P<total_function_calls>{float}) function calls (?:\(\d+ primitive calls\) )?in (?P<total_time>{float}) (?P<profiling_unit>\w+)'.format(float=float_re), line)
        if m:
            store_group(self.results, m)
            return True
        return False

    def parse_columns(self, line, it):
        fields = re.split(r'\s+', line.strip())
        last_field_i = len(self.header) - 1
        for i, name in enumerate(self.header[0:len(self.header) - 1]):
            field = fields[i]
            store_as(self.results, name, field, store_type='list')
        field = " ".join(fields[last_field_i:])
        store_as(self.results, self.header[-1], field, store_type='list')

    def post_parse(self, bench_name):
        if self.convert_to_seconds and self.results['profiling_unit'] == 'cycles':
            for key in self.time_fields:
                time_sec = self.results[key]
                self.results[key] = self.cycles_to_seconds(time_sec)

        self.parse_python_overhead(bench_name)

    def each_line(self):
        num_lines = len(self.results[self.header[0]])
        for i in range(num_lines):
            row = []
            for k in self.header:
                # NOTE: must match order of append inside dump_header.
                value = self.results[k][i]
                if k in self.time_fields:
                    row.append(pretty_time(value))
                row.append(value)
                if k in self.total_time_fields:
                    # per_iter_time_sec = self.results[self.per_iter_field(k)][i]
                    time_per_iter = self.time_per_call(value)
                    row.append(pretty_time(time_per_iter))
            yield row

    def cycles_to_seconds(self, cycles):
        # cpu_freq_ghz = self.cpu_freq.results['cpu_freq_ghz_mean']
        cpu_freq_ghz = self.tsc_freq.results['tsc_freq_ghz_mean']
        HZ_IN_GHZ = 1e9
        cpu_freq_hz = cpu_freq_ghz*HZ_IN_GHZ
        return [x/cpu_freq_hz for x in cycles]

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

        print("> Using the first {n} samples from each DQN stage to compute Total...".format(
            n=smallest_length))

        cpp_and_gpu_times = np.zeros(smallest_length)
        python_times = np.zeros(smallest_length)
        for bench_name, raw_data in zip(bench_names, raw_datas):
            cpp_and_gpu_times += raw_data['CppAndGPUTimeSec'][0:smallest_length]
            python_times += raw_data['PythonTimeSec'][0:smallest_length]

        data = compute_plot_data(cpp_and_gpu_times, python_times)
        do_dump_json(data, path)
        print("> Output pyprof Total: {f}".format(f=path))

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

