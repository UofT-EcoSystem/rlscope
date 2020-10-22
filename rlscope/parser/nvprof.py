"""
Collecting profiling data using ``nvprof``.

.. deprecated:: 1.0.0
    We use CUPTI to collect GPU profiling data.
"""
from rlscope.profiler.rlscope_logging import logger
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

from rlscope.parser.common import *
from rlscope.parser.stats import Stats

cxxfilt = None
try:
    import cxxfilt
except ImportError:
    pass

sqlite3 = None
try:
    import sqlite3
except ImportError:
    pass

class CUDAProfileParser(ProfilerParser):
    """
    NOTE: This is for q_forward with:
    - iterations = 1000
    - repetitions = 10
    - Q: Why is "GPU activities" Calls not a multiple of 10000?
      Looks more like it's only a multiple of iterations=1000 ...
      but cuEventQuery isn't quite a multiple of 1000 (WEIRD)

      - NOTE: this is just b/c of an outdated file...
      GPU activities 'Time' total:
      65.78635
      API calls 'Time' total:
      149.37072

      Q: Why's there a discrepancy?

    ==18229== NVPROF is profiling process 18229, command: /home/jgleeson/envs/cycle_counter_prod/bin/python3 /home/jgleeson/clone/baselines/baselines/deepq/experiments/benchmark_dqn.py --profile-cuda --directory /home/jgleeson/clone/baselines/checkpoints/PongNoFrameskip-v4/glue/gpu/cycle_counter/02/microbenchmar
    ==18229== Profiling application: /home/jgleeson/envs/cycle_counter_prod/bin/python3 /home/jgleeson/clone/baselines/baselines/deepq/experiments/benchmark_dqn.py --profile-cuda --directory /home/jgleeson/clone/baselines/checkpoints/PongNoFrameskip-v4/glue/gpu/cycle_counter/02/microbenchmar
    ==18229== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
     GPU activities:   68.45%  45.0280s    200000  225.14us  2.0160us  525.09us  void gemv2N_kernel_val<float, float, float, int=128, int=32, int=4, int=4, int=1>(float, float, cublasGemv2Params_v2<float, float, float>)
                        8.85%  5.82178s    100000  58.217us  56.641us  58.913us  maxwell_scudnn_128x64_relu_small_nn
                        3.81%  2.50656s    100000  25.065us  23.072us  28.545us  maxwell_scudnn_128x32_relu_medium_nn
                        3.12%  2.05037s    100000  20.503us  18.048us  22.400us  maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148n_nt
                        2.11%  1.38482s    600000  2.3080us     320ns  27.457us  [CUDA memcpy DtoH]
                        1.74%  1.14494s    300000  3.8160us     416ns  27.457us  [CUDA memcpy HtoD]
                        1.43%  939.29ms    300000  3.1300us  1.9200us  4.4160us  void tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>(int, float const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>*)
                        1.37%  902.15ms    100000  9.0210us  8.5120us  10.976us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>(unsigned char const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>*)
                        1.20%  790.26ms    300000  2.6340us  2.1120us  4.6090us  void tensorflow::BiasNCHWKernel<float>(int, float const *, float const , tensorflow::BiasNCHWKernel<float>*, int, int)
                        1.18%  774.10ms    100000  7.7410us  6.8480us  9.0570us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>)
                        0.96%  632.73ms    100000  6.3270us  5.8880us  6.6240us  void tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>(int, float const *, tensorflow::functor::Dimension<int=4>, tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>*, tensorflow::functor::Dimension, float const *)
                        0.88%  580.17ms    400000  1.4500us  1.0240us  3.4240us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const , float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const , Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(float, int=1)
                        0.81%  534.80ms    100000  5.3480us  3.6480us  7.0720us  void cudnn::winograd::generateWinogradTilesKernel<int=0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)
                        0.66%  431.47ms    100000  4.3140us  4.0000us  4.8640us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<__int64, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float>>, Eigen::array<long, unsigned long=1> const , Eigen::TensorMap<Eigen::Tensor<float const , int=2, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(__int64, int=1)
                        0.61%  401.79ms    100000  4.0170us  3.2000us  5.1840us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>)
                        0.60%  393.82ms    200000  1.9690us  1.6640us  4.1280us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(float, int=1)
                        0.56%  366.66ms    100000  3.6660us  3.0400us  5.0560us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>(unsigned int const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>*)
                        0.42%  276.90ms    200000  1.3840us  1.0240us  3.8080us  void tensorflow::BiasNHWCKernel<float>(int, float const *, float const , tensorflow::BiasNHWCKernel<float>*, int)
                        0.42%  274.60ms    200000  1.3730us  1.1840us  3.0080us  cudnn::maxwell::gemm::computeOffsetsKernel(cudnn::maxwell::gemm::ComputeOffsetsParams)
                        0.24%  156.99ms    100000  1.5690us  1.2800us  4.3530us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(__int64, int=1)
                        0.21%  137.27ms    100000  1.3720us  1.0240us  2.5280us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
                        0.21%  137.20ms    100000  1.3720us  1.0560us  3.0400us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
                        0.18%  119.68ms    100000  1.1960us     928ns  2.4640us  [CUDA memcpy DtoD]
          API calls:   41.62%  62.1707s   3100000  20.055us  6.4550us  2.47057s  cudaLaunch
                       21.09%  31.5067s    100000  315.07us  3.9630us  2.4058ms  cuCtxSynchronize
                       14.75%  22.0380s   1800000  12.243us     328ns  2.47052s  cuEventRecord
                       10.86%  16.2258s    600000  27.042us  6.0910us  2.47050s  cuMemcpyDtoHAsync
                        4.12%  6.15277s    300000  20.509us  6.2390us  1.2364ms  cuMemcpyHtoDAsync
                        1.99%  2.97150s   2416996  1.2290us     476ns  389.70us  cuEventQuery
                        1.73%  2.59132s    100000  25.913us  16.561us  1.2549ms  cudaMemcpyAsync
                        1.09%  1.62282s   9200000     176ns     111ns  387.56us  cudaSetupArgument
                        1.03%  1.53828s    300000  5.1270us  1.3230us  87.835ms  cudaEventRecord
                        0.86%  1.27935s   3100000     412ns     152ns  416.18us  cudaConfigureCall
                        0.71%  1.06470s    900000  1.1830us     549ns  392.19us  cuStreamWaitEvent
                        0.14%  208.78ms   1000000     208ns     108ns  386.07us  cudaGetLastError

    {
        'gpu_activities': {
            'Time(%)': [68.45, ...],
            'Time': [45.0280, ...],
            'Calls': [20000, ...],
            # Q: If I divide a floating point number by a really big number, will I lose precision at all...?
            # Or is it only once I start combining 2 small floating points?
            'Avg': [225.14, ...],
            'Min': [2.0160, ...],
            'Max': [525.09, ...],
            'Name': [...],
        }
        'api_calls': {
            ...
        }
    }
    """
    def __init__(self, parser, args, profile_path, bench_name, data=None):
        super().__init__(parser, args, profile_path, bench_name, data)
        self.time_fields = ['Time', 'Avg', 'Min', 'Max']
        self.total_time_fields = ['Time_seconds']
        self.sortby = ('Type', 'Time_seconds')

        self.parsing = None
        self.no_cuda_calls_expected = False
        self.has_error = False

    CUDA_TIME_UNIT_REGEX = r'(?:s|ms|us|ns)'
    def as_seconds(self, time_as_unit, unit):
        if unit == 's':
            return time_as_unit
        elif unit == 'ms':
            return time_as_unit/constants.MILLISECONDS_IN_SECOND
        elif unit == 'us':
            return time_as_unit/constants.MICROSECONDS_IN_SECOND
        elif unit == 'ns':
            return time_as_unit/constants.NANOSECONDS_IN_SECOND
        else:
            raise NotImplementedError()

    def as_value(self, x):
        if re.search(r'%$', x):
            # Convert to percent in [0..1]
            x = re.sub(r'%', '', x)
            x = as_value(x) / 100.
            return x

        m = re.search(r'(?P<float>{float})(?P<unit>{unit})$'.format(
            float=float_re,
            unit=CUDAProfileParser.CUDA_TIME_UNIT_REGEX),
            x)
        if m:
            time_as_unit = float(m.group('float'))
            unit = m.group('unit')
            time_sec = self.as_seconds(time_as_unit, unit)
            return time_sec

        return as_value(x)

    def parse_other(self, line, it):
        self.no_cuda_calls_expected |= bool(re.search(r"|".join([r'^==.*== Generated result file',
                                                                 r'No kernels were profiled']),
                                                      line))
        self.has_error |= bool(re.search(r'^=+\s*Error:.*Application', line))
        return False

    def parse_header(self, line, it):
        assert self.header is None
        if re.search(r'^\s*Type\s*Time', line):
            self.header = re.split(r'\s+', line.strip())
            # for i in range(len(self.header)):
            #     if self.header[i] == 'percall':
            #         self.header[i] = "{last_col}_percall".format(last_col=self.header[i-1])
            return True
        return False

    def parse_columns(self, line, it):
        fields = re.split(r'\s+', line.strip())
        first_two_fields = " ".join(fields[0:2])
        remaining_fields = fields[2:]
        if re.search(r'GPU activities', first_two_fields):
            self.parsing = 'gpu_activities'
            fields = remaining_fields
        elif re.search(r'API calls', first_two_fields):
            self.parsing = 'api_calls'
            fields = remaining_fields

        assert self.parsing is not None

        last_field_i = len(self.header) - 1
        field = " ".join(fields[last_field_i:])
        new_fields = [self.parsing] + fields[0:last_field_i] + [field]
        fields = new_fields
        fields = [self.as_value(x) for x in fields]

        put_key(self.results, self.parsing, dict())

        for i, name in enumerate(self.header):
            field = fields[i]
            store_as(self.results[self.parsing], name, field, store_type='list')

        return True

    def post_parse(self, bench_name):
        assert ( self.has_error or self.no_cuda_calls_expected ) or self.header is not None
        if self.no_cuda_calls_expected:
            logger.info("> Skip pretty cuda profile; didn't see any CUDA calls in {path}".format(path=self.profile_path(bench_name)))
            self.skip = True

        if self.has_error:
            logger.info("> Skip pretty cuda profile; WARNING: saw an ERROR in {path}".format(path=self.profile_path(bench_name)))
            self.skip = True

    def each_line(self):
        assert 'api_calls' in self.results
        assert 'gpu_activities' in self.results
        assert sorted(self.results['api_calls']) == sorted(self.results['gpu_activities'])
        num_lines = len(self.results['api_calls'][self.header[0]])
        for parsing in ['gpu_activities', 'api_calls']:
            for i in range(num_lines):
                row = []
                for k in self.header:
                    value = self.results[parsing][k][i]
                    if k in self.time_fields:
                        pty_time = pretty_time(value)
                    row.append(value)
                    if k in self.total_time_fields:
                        time_per_iter = self.time_per_call(value)
                        row.append(pretty_time(time_per_iter))
                yield row

class CUDASQLiteParser(ProfilerParserCommonMixin):
    def __init__(self, parser, args, src_files, bench_name=NO_BENCH_NAME, data=None):
        self.is_dqn = 'microbenchmark_json' in src_files.opt_paths
        self.src_files = src_files

        assert cxxfilt is not None
        self.parser = parser
        self.args = args
        self.bench_name = bench_name
        self.data = data
        self.skip = False
        self.conn = None

        self.config_path = src_files.get('config_json', bench_name, or_none=True)
        if self.config_path is not None:
            self.config = load_json(self.config_path)
            logger.info("> Found optional config_json @ {f}".format(f=self.config_path))
        else:
            self.config = {
            }

        self.num_calls = self._parse_num_calls(bench_name)

        self.discard_first_sample = self.config.get('discard_first_sample', self.args.discard_first_sample)

        self.kernel_stats = Stats(self.discard_first_sample, debug=self.args.debug, name="CUDA_Kernel_stats",
                                  has_overlap=True)
        self.api_stats = Stats(self.discard_first_sample, debug=self.args.debug, name="CUDA_API_stats",
                               # I don't expect CUDA API calls to overlap (at least, for the single-machine,
                               # single-GPU workloads I'm looking at).
                               has_overlap=False)

    @staticmethod
    def required_source_basename_regexes():
        return {'profile_path': r"^nvidia{bench}\.nvprof$".format(bench=BENCH_SUFFIX_RE)}

    @staticmethod
    def target_basename_regexes():
        return {
            'gpu_overhead_json': r"^nvidia{bench}\.gpu_overhead\.nvprof\.json$".format(bench=BENCH_SUFFIX_RE),
            'pretty_profile_path': r"^nvidia{bench}\.nvprof\.pretty\.txt$".format(bench=BENCH_SUFFIX_RE),
            'variable_path': r"^nvidia{bench}\.variable\.pretty\.txt$".format(bench=BENCH_SUFFIX_RE),
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

    def profile_path(self, bench_name):
        return self.src_files.get('profile_path', bench_name)

    def get_micro_name(self):
        return self.bench_name

    def _parse_num_calls(self, bench_name):
        if self.is_dqn:
            data = self.load_microbench(bench_name)
            # bench_data = data[get_nvprof_name(self.bench_name)]
            bench_data = data[self.bench_name]
            num_calls = compute_num_calls_dqn(bench_data)
        elif 'num_calls' in self.config:
            num_calls = compute_num_calls(self.config)
        else:
            num_calls = 1

        logger.info("> num_calls = {num_calls}".format(
            num_calls=num_calls))
        return num_calls

    def _pr_rows(self, sql_rows):
        if len(sql_rows) == 0 or type(sql_rows[0]) not in set([sqlite3.Row]):
            pprint.pprint(sql_rows)
            return

        xs = []
        for row in sql_rows:
            d = dict()
            for k in row.keys():
                d[k] = sql_rows[k]
        pprint.pprint(xs)

    def gpu_count(self):
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) as c FROM CUPTI_ACTIVITY_KIND_DEVICE")
        return c.fetchone()["c"]

    def api_times(self):
        """
        Select all CUDA API call times: (start_nsec, end_nsec)
        :return:
        """
        c = self.conn.cursor()
        # The times are in nanoseconds; lets convert them to microseconds as our base unit
        def fetch_cbid(table):
            c.execute(textwrap.dedent("""
                SELECT 
                    A.start AS start_nsec, 
                    A.end AS end_nsec,
                    A.cbid as name 
                FROM 
                    {table} AS A 
                ORDER BY A.start
            """.format(table=table)))
            results = c.fetchall()
            return results
        results = fetch_cbid('CUPTI_ACTIVITY_KIND_DRIVER') + fetch_cbid('CUPTI_ACTIVITY_KIND_RUNTIME')
        results.sort(key=lambda r: r['start_nsec'])

        for r in results:
            r['name'] = CBID_TO_CUDA_FUNC.get(r['name'], r['name'])

        for r in results:
            self.api_stats.add(r['name'],
                               start_end_nsec_to_usec(r['start_nsec'], r['end_nsec']),
                               nsec_to_usec(r['start_nsec']),
                               nsec_to_usec(r['end_nsec']))

    def kernel_times(self):
        """
        Select all CUDA kernel call times: (start_nsec, end_nsec)
        :return:
        """
        c = self.conn.cursor()
        # The times are in nanoseconds; lets convert them to microseconds as our base unit
        c.execute(textwrap.dedent("""
            SELECT 
                A.start AS start_nsec, 
                A.end AS end_nsec,
                S.value as name 
            FROM 
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS A 
                join StringTable AS S 
                    ON A.name == S._id_ 
            ORDER BY A.start
        """))
        kernel_results = c.fetchall()

        for r in kernel_results:
            r['name'] = cxxfilt.demangle(r['name'])
        for r in kernel_results:
            self.kernel_stats.add(r['name'],
                                  start_end_nsec_to_usec(r['start_nsec'], r['end_nsec']),
                                  nsec_to_usec(r['start_nsec']),
                                  nsec_to_usec(r['end_nsec']))

        def fetch_memcpy(table):
            c.execute(textwrap.dedent("""
                SELECT 
                    A.start AS start_nsec, 
                    A.end AS end_nsec,
                    copyKind as name 
                FROM 
                    {table} AS A 
                ORDER BY A.start
            """.format(table=table)))
            results = c.fetchall()
            return results

        memcpy_results = fetch_memcpy('CUPTI_ACTIVITY_KIND_MEMCPY') + fetch_memcpy('CUPTI_ACTIVITY_KIND_MEMCPY2')
        for r in memcpy_results:
            r['name'] = COPYKIND_TO_MEMCPY.get(r['name'], r['name'])
            self.kernel_stats.add(r['name'],
                                  start_end_nsec_to_usec(r['start_nsec'], r['end_nsec']),
                                  nsec_to_usec(r['start_nsec']),
                                  nsec_to_usec(r['end_nsec']))

    def parse(self, bench_name):
        # with open(self.profile_path(bench_name)) as f:
        self.conn = sqlite3.connect(self.profile_path(bench_name))
        # https://stackoverflow.com/questions/3300464/how-can-i-get-dict-from-sqlite-query
        # self.conn.row_factory = sqlite3.Row
        self.conn.row_factory = dict_factory
        with self.conn:
            # PSEUDOCODE:
            # select func_name,
            #   ordered by order of occurrence when executing 1000 repetitions.
            # num_gpus = self.gpu_count()
            # logger.info("> num_gpus = {num_gpus}".format(num_gpus=num_gpus))

            self.kernel_times()
            self.api_times()

    def _microbench_path(self, bench_name):
        return get_microbench_path(_d(self._pretty_profile_path(bench_name)))

    def load_microbench(self, bench_name):
        microbench_path = self._microbench_path(bench_name)
        assert _e(microbench_path)

        with codecs.open(microbench_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)
            data = fixup_json(data)
            return data

    def dump(self, bench_name):
        if self.skip:
            return

        summary_type = 'separate_calls'

        self.kernel_stats.split(self.num_calls)
        self.api_stats.split(self.num_calls)

        self.parse_gpu_overhead(bench_name)

        with open(self._pretty_profile_path(bench_name), 'w') as f:
            with open(self._variable_path(bench_name), 'w') as f_variable:

                def dump_data(stats, profile_data_type, skip_header=False):
                    logger.info("> {t}".format(t=profile_data_type))
                    stats.dump(f, profile_data_type, skip_header=skip_header, summary_type=summary_type)
                    stats.dump_variable(f_variable, profile_data_type, skip_header=skip_header)
                    logger.info()

                dump_data(self.kernel_stats, profile_data_type='GPU activities')
                dump_data(self.api_stats, profile_data_type='API calls', skip_header=True)

    def total_iterations(self, bench_name):
        if self.num_calls is None:
            self.num_calls = self._parse_num_calls(bench_name)
            assert self.num_calls is not None

        if self.args.discard_first_sample:
            return self.num_calls - 1
        return self.num_calls

    def parse_gpu_overhead(self, bench_name, dump_json=True):
        # stats = self.api_stats

        if self.num_calls is None:
            self.num_calls = self._parse_num_calls(bench_name)
        # gpu_times = np.zeros(self.total_iterations(bench_name))
        # for stat in stats.stats:
        #     times_sec = stat.iteration_times_sec(self.num_calls)
        #     gpu_times += times_sec

        # {
        #     "CudaCppTimeSec": [
        #         0.20482133099999977
        #     ],
        #     "FrameworkCppTimeSec": [
        #         0.0
        #     ],
        #     "GPUAndCudaCppTimeSec": [
        #         5.3178501549999995
        #     ],
        #     "GPUTimeSec": [
        #         5.113028824
        #     ]
        # }

        gpu_times = self.kernel_stats.sum_calls_sec()
        gpu_and_cuda_cpp_times = self.api_stats.sum_calls_sec()
        assert gpu_times.shape == gpu_and_cuda_cpp_times.shape
        cuda_cpp_times = gpu_and_cuda_cpp_times - gpu_times
        # WRONG:
        # framework_cpp_times = gpu_and_cuda_cpp_times - gpu_times - cuda_cpp_times
        # WANT:
        # framework_cpp_times = pyprof(gpu_and_cpp_times) - gpu_times - cuda_cpp_times

        # 'CUDACppAndGpuTimeSec' = sum("API calls" from nvprof)
        # 'CUDACppTimeSec' = 'CUDACppAndGpuTimeSec' - 'GPUTimeSec'
        # 'CppTimeSec' = 'CppAndGpuTimeSec' - 'GPUTimeSec'

        raw_data = {
            "GPUTimeSec":list(gpu_times),
            "GPUAndCudaCppTimeSec":list(gpu_and_cuda_cpp_times),
            "CudaCppTimeSec":list(cuda_cpp_times),
            # "FrameworkCppTimeSec":list(framework_cpp_times),
        }
        if dump_json:
            json_data = make_json_serializable(raw_data)
            do_dump_json(json_data, self._gpu_overhead_path(bench_name))
        return raw_data

    def _gpu_overhead_path(self, bench_name):
        return self.get_gpu_overhead_path(self.src_files, bench_name)

    @classmethod
    def get_gpu_overhead_path(ParseKlass, src_files, bench_name):
        ret = re.sub(r'.nvprof$', '.gpu_overhead.nvprof.json', src_files.get('profile_path', bench_name))
        assert ret != src_files.get('profile_path', bench_name)
        return ret

    def dump_path(self, bench_name):
        return self._pretty_profile_path(bench_name)

    @classmethod
    def get_pretty_profile_path(ParserKlass, src_files, bench_name):
        pretty_base = "{base}.pretty.txt".format(base=_b(src_files.get('profile_path', bench_name)))
        return _j(_d(src_files.get('profile_path', bench_name)), pretty_base)

    def _pretty_profile_path(self, bench_name):
        return CUDASQLiteParser.get_pretty_profile_path(self.src_files, bench_name)

    @classmethod
    def get_variable_path(ParserKlass, src_files, bench_name):
        pretty_base = "{base}.variable.pretty.txt".format(base=_b(src_files.get('profile_path', bench_name)))
        return _j(_d(src_files.get('profile_path', bench_name)), pretty_base)

    def _variable_path(self, bench_name):
        return self.get_variable_path(self.src_files, bench_name)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

