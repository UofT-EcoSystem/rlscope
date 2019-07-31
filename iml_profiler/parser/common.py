import platform
import codecs
import sys
import shlex
import csv
import decimal
import json
import logging
import os
import pickle
import pprint
import re
import textwrap
import time
import traceback
from io import StringIO
from os.path import join as _j, dirname as _d, exists as _e, basename as _b

import numpy as np
import pandas as pd
import psutil
import pwd
from iml_profiler.protobuf.pyprof_pb2 import Pyprof, MachineUtilization, ProcessMetadata, IncrementalTrainingProgress
from iml_profiler.protobuf.unit_test_pb2 import IMLUnitTestOnce, IMLUnitTestMultiple
from tqdm import tqdm as tqdm_progress

USEC_IN_SEC = 1e6

OPERATION_PYTHON_PROFILING_OVERHEAD = "Python profiling overhead"

CATEGORY_TF_API = "Framework API C"
CATEGORY_PYTHON = 'Python'
CATEGORY_PYTHON_PROFILER = 'Python profiler'
CATEGORY_CUDA_API_CPU = 'CUDA API CPU'
CATEGORY_UNKNOWN = 'Unknown'
CATEGORY_GPU = 'GPU'
CATEGORY_DUMMY_EVENT = 'Dummy event'
# Category captures when operations of a given type start/end.
# That way, if we have a profile with multiple operations in it,
# we can reduce the scope to just an operation of interest (e.g. Q-forward).
#
# NOTE: This operation category should NOT show up in compute_overlap.
CATEGORY_OPERATION = 'Operation'
# Category captures when we are executing a TRACE/WARMUP/DUMP phase of profiling.
# Can be useful for ignoring parts of the execution (e.g. DUMP).
# CATEGORY_PHASE = 'Phase'
CATEGORY_SIMULATOR_CPP = "Simulator C"
CATEGORY_ATARI = CATEGORY_SIMULATOR_CPP

# Category captures special profiling events. Currently includes:
# - event.name = PROFILING_DUMP_TRACE
#   - tfprof/pyprof files are being dumped; we should NOT use any
#     events recorded during this time to avoid accidentally measuring
#     the profiler itself.
CATEGORY_PROFILING = 'Profiling'

CATEGORIES_ALL = [
    CATEGORY_TF_API,
    CATEGORY_PYTHON,
    CATEGORY_CUDA_API_CPU,
    CATEGORY_UNKNOWN,
    CATEGORY_GPU,
    CATEGORY_DUMMY_EVENT,
    CATEGORY_OPERATION,
    CATEGORY_SIMULATOR_CPP,
    CATEGORY_PROFILING,
    CATEGORY_PYTHON_PROFILER,
]

DEFAULT_PHASE = 'default_phase'

# Record a "special" operation event-name (category=CATEGORY_OPERATION)
# when the prof.start()/stop() is called.
#
# These events are to help debug whether all the code is properly marked.
# Generally, they aren't mean to show up in any of the final plots.
# record_event(name="[PROC:run_atari_1]", start_us=prof.start() time, end_us=prof.end() time)
def op_process_event_name(process_name):
    assert process_name is not None
    return "[{proc}]".format(
        proc=process_name)
# def match_special_event_name(event_name):
def match_debug_event_name(event_name):
    return re.search(r'^\[(?P<event_name>.*)\]$', event_name)

CATEGORIES_CPU = set([
    CATEGORY_TF_API,
    CATEGORY_PYTHON,
    CATEGORY_CUDA_API_CPU,
    CATEGORY_SIMULATOR_CPP,
    CATEGORY_PYTHON_PROFILER,
])

CATEGORIES_GPU = set([
    CATEGORY_GPU,
])

# Not a category used during tracing;
# represents a group of categories.
CATEGORY_CPU = 'CPU'

CATEGORY_TOTAL = 'Total'

PROFILING_DUMP_TRACE = "Dump trace"

PLOT_SUMMMARY_FIELDS = [
    "TotalTimeSec",
    "CppAndGPUTimeSec",
    "CppTimeSec",
    "FrameworkCppTimeSec",
    "CudaCppTimeSec",
    "PercentTimeInGPU",
    "GPUTimeSec",
    "GPUAndCudaCppTimeSec",
    "TheoreticalSpeedup",
    "PercentTimeInPython",
    "PythonTimeSec",
    "PythonOverheadPercent",
]
PLOT_SUMMARY_FIELDS_TIME_SEC = [field for field in PLOT_SUMMMARY_FIELDS if re.search(r'TimeSec$', field)]


BENCH_NAME_TO_PRETTY = {
    "checkpoint":"Checkpoint",
    "q_backward":"Q-backward",
    "q_forward":"Q-forward",
    "q_update_target_network":"Q update \ntarget-network",
    "restore":"Restore",
    "training_iteration":"Training iteration",
    "step":"Step",
    "total":"Total",
    # "compress":_compress_name("Compress"),
    # "decompress":_compress_name("Decompress"),
}

DQN_BENCH_NAMES = ['q_forward', 'q_backward', 'q_update_target_network']

CUDA_MICROBENCH_NAMES = [
    'cuda_launch',
    'd2h',
    'h2d',
]
CUDA_MICROBENCH_NAME_TO_PRETTY = {
    'cuda_launch':'Launch Kernel',
    'd2h':'Device-to-Host',
    'h2d':'Host-to-Device',
}
BENCH_NAME_REGEX = r"(?:[^\.]+)"
BENCH_NAME_ORDER = [
    'step',
    'q_forward',
    'q_backward',
    'q_update_target_network',
    # 'training_iteration',
    # 'checkpoint',
    # 'restore',
    # 'compress',
    # 'decompress',
    # 'total',
]
BENCH_TYPES = [
    # Microbenchmarks for the CUDA API
    # (e.g. cudaLaunch latency).
    'cuda',
    # Microbenchmarks for individual operations from the DQN training loop.
    'dqn',
    'cpufreq',
    'nvprof',
    # 'python_profile',
]

BENCH_SUFFIX_RE = r"(:?\.op_(?P<bench_name>{bench}))?".format(bench=BENCH_NAME_REGEX)
BENCH_PREFIX_RE = r"(:?op_(?P<bench_name>{bench})\.)?".format(bench=BENCH_NAME_REGEX)

# trace_id is not optional.
# WARNING: To ensure we can load files created in older commits,
# we allow trace_id and session_id suffixes to be OPTIONAL.
TRACE_SUFFIX_RE = r"(?:\.trace_(?P<trace_id>\d+))?"
SESSION_SUFFIX_RE = r"(?:\.session_(?P<session_id>\d+))?"
CONFIG_RE = r"(?:config_(?P<config>[^\.]+))"

float_re = r'(?:[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'

# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_PROCESS_NAME = "NoProcessName"
NO_PHASE_NAME = "NoProcessName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

MICROSECONDS_IN_SECOND = float(1e6)
MILLISECONDS_IN_SECOND = float(1e3)
NANOSECONDS_IN_SECOND = float(1e9)

# Here's the "API calls" summary output from "nvprof -i file.nvprof" for Forward.
#
# 76.69%  2.60481s     31000  84.026us  5.7450us  711.15ms  cudaLaunch
# 9.50%  322.74ms      1000  322.74us  3.9820us  425.32us  cuCtxSynchronize
# 4.92%  167.25ms     18000  9.2910us     368ns  350.38us  cuEventRecord
# 3.77%  128.08ms      6000  21.346us  7.2610us  378.31us  cuMemcpyDtoHAsync
# 1.74%  59.123ms      3000  19.707us  6.4430us  74.321us  cuMemcpyHtoDAsync
# 0.92%  31.369ms     23532  1.3330us     485ns  344.94us  cuEventQuery
# 0.75%  25.609ms      1000  25.608us  16.958us  355.11us  cudaMemcpyAsync
# 0.49%  16.504ms     92000     179ns     113ns  333.72us  cudaSetupArgument
# 0.47%  15.835ms      3000  5.2780us  1.3280us  62.487us  cudaEventRecord
# 0.38%  13.013ms     31000     419ns     143ns  337.91us  cudaConfigureCall
# 0.29%  9.7108ms      9000  1.0780us     577ns  333.85us  cuStreamWaitEvent
# 0.08%  2.6487ms     12000     220ns     108ns  326.60us  cudaGetLastError

# Here's my manually generated "API calls" summary output for Forward, by querying the SQLite3 database file.
#
# Time(%)|Time|Calls|Avg|Min|Max|Name
# 76.69|2.6048135660000136sec|31000|8.40262440645161e-05|5.745e-06|0.71114529|13
# 9.50|322.7421669999999ms|1000|0.0003227421669999999|3.982e-06|0.000425317|17
# 4.92|167.2546419999997ms|18000|9.291924555555556e-06|3.68e-07|0.000350382|119
# 3.77|128.0785700000002ms|6000|2.1346428333333336e-05|7.2610000000000004e-06|0.000378309|279
# 1.74|59.123038000000015ms|3000|1.9707679333333333e-05|6.443e-06|7.4321e-05|277
# 0.92|31.369078999999882ms|23532|1.3330392231854497e-06|4.85e-07|0.000344937|120
# 0.75|25.608579000000034ms|1000|2.5608579e-05|1.6958e-05|0.00035511|41
# 0.49|16.50449599999818ms|92000|1.7939669565217395e-07|1.13e-07|0.000333715|9
# 0.47|15.835319999999976ms|3000|5.2784399999999996e-06|1.328e-06|6.2487e-05|135
# 0.38|13.013477000000066ms|31000|4.197895806451613e-07|1.43e-07|0.000337914|8
# 0.29|9.71083400000003ms|9000|1.0789815555555554e-06|5.769999999999999e-07|0.00033384699999999996|295
# 0.08|2.6487239999999996ms|12000|2.2072700000000002e-07|1.08e-07|0.0003266|10

# From the above outputs, we can disambiguate which cbid maps to which CUDA function call:
CBID_TO_CUDA_FUNC = {
    13:'cudaLaunch',
    17:'cuCtxSynchronize',
    119:'cuEventRecord',
    279:'cuMemcpyDtoHAsync',
    277:'cuMemcpyHtoDAsync',
    120:'cuEventQuery',
    41:'cudaMemcpyAsync',
    9:'cudaSetupArgument',
    135:'cudaEventRecord',
    8:'cudaConfigureCall',
    295:'cuStreamWaitEvent',
    10:'cudaGetLastError',
}

# Here's the "GPU activities" part of the output from nvprof:
#
# Type  Time(%)      Time     Calls       Avg       Min       Max  Name
# 68.88%  446.48ms      2000  223.24us  2.0160us  537.38us  void gemv2N_kernel_val<float, float, float, int=128, int=32, int=4, int=4, int=1>(float, float, cublasGemv2Params_v2<float, float, float>)
# 8.82%  57.172ms      1000  57.171us  56.128us  58.432us  maxwell_scudnn_128x64_relu_small_nn
# 3.99%  25.871ms      1000  25.871us  24.672us  26.656us  maxwell_scudnn_128x32_relu_medium_nn
# 3.17%  20.540ms      1000  20.540us  18.657us  21.952us  maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148n_nt
# 1.85%  11.977ms      3000  3.9920us     832ns  11.456us  [CUDA memcpy HtoD]
# 1.46%  9.4930ms      3000  3.1640us  2.0160us  4.0640us  void tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>(int, float const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension0And2InTensor3Simple<float, bool=0>*)
# 1.39%  8.9873ms      1000  8.9870us  8.7360us  10.208us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>(unsigned char const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, int=1024, int=1024, int=2, bool=0>*)
# 1.24%  8.0238ms      3000  2.6740us  2.2080us  3.7440us  void tensorflow::BiasNCHWKernel<float>(int, float const *, float const , tensorflow::BiasNCHWKernel<float>*, int, int)
# 1.20%  7.8063ms      1000  7.8060us  7.5200us  8.0640us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, __int64>>)
# 0.98%  6.3426ms      1000  6.3420us  6.2080us  6.4960us  void tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>(int, float const *, tensorflow::functor::Dimension<int=4>, tensorflow::functor::PadInputCustomKernelNCHW<float, int=4>*, tensorflow::functor::Dimension, float const *)
# 0.95%  6.1683ms      6000  1.0280us     416ns  2.0160us  [CUDA memcpy DtoH]
# 0.92%  5.9620ms      4000  1.4900us  1.1200us  2.0160us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const , float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const , Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const >, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(float, int=1)
# 0.76%  4.8959ms      1000  4.8950us  3.6480us  5.5680us  void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>(unsigned int const *, tensorflow::functor::Dimension<int=3>, tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, int=256, int=32, int=32, bool=0>*)
# 0.74%  4.7984ms      1000  4.7980us  4.0960us  6.9760us  void cudnn::winograd::generateWinogradTilesKernel<int=0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)
# 0.67%  4.3490ms      1000  4.3490us  4.0960us  4.7680us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<__int64, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float>>, Eigen::array<long, unsigned long=1> const , Eigen::TensorMap<Eigen::Tensor<float const , int=2, int=1, long>, int=16, Eigen::MakePointer> const > const > const > const , Eigen::GpuDevice>, long>(__int64, int=1)
# 0.64%  4.1744ms      1000  4.1740us  3.7440us  4.9280us  void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>(tensorflow::random::PhiloxRandom, tensorflow::random::PhiloxRandomResultElementType*, __int64, tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>>)
# 0.61%  3.9572ms      2000  1.9780us  1.7600us  3.4880us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, int=1, int=1, long>, int=16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const , int=1, int=1, long>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, long>(float, int=1)
# 0.48%  3.1422ms      2000  1.5710us  1.3440us  1.9840us  cudnn::maxwell::gemm::computeOffsetsKernel(cudnn::maxwell::gemm::ComputeOffsetsParams)
# 0.40%  2.6120ms      2000  1.3050us     992ns  1.6640us  void tensorflow::BiasNHWCKernel<float>(int, float const *, float const , tensorflow::BiasNHWCKernel<float>*, int)
# 0.25%  1.5927ms      1000  1.5920us  1.3760us  1.7920us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<__int64, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const , Eigen::TensorMap<Eigen::Tensor<__int64 const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(__int64, int=1)
# 0.21%  1.3723ms      1000  1.3720us  1.3120us  1.5040us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
# 0.21%  1.3680ms      1000  1.3670us  1.2160us  1.5040us  void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, int=1, int=1, int>, int=16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float>>, Eigen::TensorMap<Eigen::Tensor<float const , int=1, int=1, int>, int=16, Eigen::MakePointer> const > const > const , Eigen::GpuDevice>, int>(bool, int=1)
# 0.18%  1.1568ms      1000  1.1560us     928ns  1.4400us  [CUDA memcpy DtoD]

# Here's my manaully generated "GPU activities":
# Time(%)|Time|Calls|Avg|Min|Max|Name
# 68.88|446.4804500000032ms|2000|0.00022324022499999998|2.016e-06|0.000537379|void gemv2N_kernel_val<float, float, float, 128, 32, 4, 4, 1>(float, float, cublasGemv2Params_v2<float, float, float>)
# 8.82|57.171979ms|1000|5.717197900000001e-05|5.6128e-05|5.8432e-05|maxwell_scudnn_128x64_relu_small_nn
# 3.99|25.871324000000005ms|1000|2.5871324e-05|2.4672e-05|2.6656e-05|maxwell_scudnn_128x32_relu_medium_nn
# 3.17|20.54018699999997ms|1000|2.0540186999999998e-05|1.8657e-05|2.1952000000000003e-05|maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile148n_nt
# 1.85|11.976681999999892ms|3000|3.992227333333333e-06|8.319999999999999e-07|1.1456e-05|1
# 1.46|9.492982000000076ms|3000|3.1643273333333336e-06|2.016e-06|4.064e-06|void tensorflow::functor::SwapDimension0And2InTensor3Simple<float, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)
# 1.39|8.987316999999981ms|1000|8.987317000000001e-06|8.736e-06|1.0208e-05|void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned char, 1024, 1024, 2, false>(unsigned char const*, tensorflow::functor::Dimension<3>, unsigned char*)
# 1.24|8.02377599999999ms|3000|2.674591999999999e-06|2.2080000000000003e-06|3.744e-06|void tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)
# 1.20|7.806344000000012ms|1000|7.806344e-06|7.519999999999999e-06|8.064e-06|void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, long long> >(tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, long long>::ResultElementType*, long long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, long long>)
# 0.98|6.342567000000009ms|1000|6.342567000000001e-06|6.2080000000000005e-06|6.496e-06|void tensorflow::functor::PadInputCustomKernelNCHW<float, 4>(int, float const*, tensorflow::functor::Dimension<4>, float*, tensorflow::functor::Dimension<4>, tensorflow::functor::Dimension<(4)-(2)>)
# 0.95|6.16826100000015ms|6000|1.0280435000000004e-06|4.1599999999999997e-07|2.016e-06|2
# 0.92|5.961957999999928ms|4000|1.4904895000000004e-06|1.12e-06|2.016e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)
# 0.76|4.895866000000007ms|1000|4.8958659999999995e-06|3.648e-06|5.568e-06|void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)
# 0.74|4.798367999999998ms|1000|4.798367999999999e-06|4.096e-06|6.976e-06|void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)
# 0.67|4.349043000000004ms|1000|4.349043e-06|4.096e-06|4.768e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<long long, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)
# 0.64|4.1744210000000015ms|1000|4.174421000000001e-06|3.744e-06|4.928e-06|void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)
# 0.61|3.957243000000013ms|2000|1.9786215e-06|1.76e-06|3.488e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<unsigned char const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)
# 0.48|3.14222300000002ms|2000|1.5711115000000003e-06|1.3440000000000002e-06|1.984e-06|cudnn::maxwell::gemm::computeOffsetsKernel(cudnn::maxwell::gemm::ComputeOffsetsParams)
# 0.40|2.6119829999999906ms|2000|1.3059915000000002e-06|9.92e-07|1.6639999999999999e-06|void tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)
# 0.25|1.592743999999982ms|1000|1.5927440000000002e-06|1.3759999999999998e-06|1.792e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)
# 0.21|1.372327000000001ms|1000|1.372327e-06|1.312e-06|1.504e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::greater_equal<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)
# 0.21|1.3679750000000017ms|1000|1.3679750000000002e-06|1.2159999999999999e-06|1.504e-06|void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<bool, float, Eigen::internal::less<float> >, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)
# 0.18|1.1568079999999958ms|1000|1.156808e-06|9.28e-07|1.44e-06|8

# enum cudaMemcpyKind:
#   cudaMemcpyHostToHost       Host -> Host
#   cudaMemcpyHostToDevice     Host -> Device
#   cudaMemcpyDeviceToHost     Device -> Host
#   cudaMemcpyDeviceToDevice   Device -> Device
#   cudaMemcpyDefault          Default based unified virtual address space
COPYKIND_TO_MEMCPY = {
    1:'[CUDA memcpy HtoD]',
    2:'[CUDA memcpy DtoH]',
    8:'[CUDA memcpy DtoD]',
}

TXT_INDENT = "    "
def get_indent(indents):
    return TXT_INDENT*indents

def maybe_indent(txt, indents):
    if indents is None:
        return txt
    indent = get_indent(indents)
    txt = textwrap.indent(txt, prefix=indent)
    # DON'T indent the first line (typically manually indented in code.
    txt = re.sub('^\s*', '', txt, count=1)
    return txt

def maybe_clause(op_clause, keep, indents=None):
    if not keep:
        return maybe_indent("TRUE", indents)
    return maybe_indent(op_clause, indents)


class ProfilerParserCommonMixin:

    def run(self, bench_name):
        self.parse(bench_name)
        self.dump(bench_name)

    @classmethod
    def _match_regexes(Klass, regexes, paths, allow_multiple,
                       # prefix_path=None,
                       # If the bench_name matches this pattern, consider the pattern a failed match.
                       ignore_bench_re=r"call_times",
                       debug=False):
        """
        Return all paths matching some regex in <regexes>.
        If a single regex matches multiple paths,
        only one of those paths will be added for that regex.

        :param regexes:
          {'<regex_name>':python_regex,
           .... }
        :param paths:
        :param allow_multiple:
          Allow multiple files to match a single regex.
          If that's the case, we store results as
          m = re.search(regex, path)
          matching_paths[regex][m.group('bench_name')].append(path)
        :return:
            # PSEUDOCODE:
            # if uses bench_name:
            #     src_files.get('profile_path', bench_name)[bench_name][0...N]
            # else:
            #     src_files.get('profile_path', bench_name)[None][0]
          matching_paths['<regex_name>'][bench_name/None][0...N] =

            {
              '<regex_name>': {
                <bench_name[0]/None>: [
                  matching_path[0],
                  matching_path[1],
                  ...,
                  ]
                  ...,
              }
            }
        """
        regexes_left = list(regexes.items())
        paths_left = list(paths)
        # matching_paths = []
        matching_paths = dict()
        j = 0
        if debug:
            logging.info("> allow_multiple={v}".format(v=allow_multiple))
        while j < len(regexes):
            regex_name, regex = regexes_left[j]

            i = 0
            while i < len(paths_left):
                path = paths_left[i]
                m = re.search(regex, _b(path))

                # if m and ignore_bench_re is not None and re.search(ignore_bench_re, _b(path)) and 'bench_name' in m.groupdict() and m.group('bench_name'):
                if m and ignore_bench_re is not None \
                    and m.groupdict().get('bench_name', None) is not None \
                    and re.search(ignore_bench_re, m.group('bench_name')):
                    if debug:
                        logging.info("> regex={regex}, ignore_bench_re={ignore_regex} matches _b(path)={path}".format(
                            regex=regex,
                            ignore_regex=ignore_bench_re,
                            path=_b(path)))
                    del paths_left[i]
                elif m:
                    if debug:
                        logging.info("> regex={regex} matches _b(path)={path}".format(regex=regex, path=_b(path)))
                    # PSEUDOCODE:
                    # if uses bench_name:
                    #     src_files.get('profile_path', bench_name)[bench_name]
                    # else:
                    #     src_files.get('profile_path', bench_name)
                    mdict = m.groupdict()
                    # if prefix_path is not None:
                    #     full_path = _j(prefix_path, path)
                    # else:
                    full_path = path
                    if 'bench_name' in mdict:
                        _mk(matching_paths, regex_name, dict())
                        bench_name = mdict['bench_name']
                        if bench_name is None:
                            bench_name = NO_BENCH_NAME
                        assert allow_multiple or bench_name not in matching_paths[regex_name]
                        matching_paths[regex_name][bench_name] = full_path
                    else:
                        _mk(matching_paths, regex_name, dict())
                        assert allow_multiple or NO_BENCH_NAME not in matching_paths[regex_name]
                        matching_paths[regex_name][NO_BENCH_NAME] = full_path

                    del paths_left[i]
                    if not allow_multiple:
                        break
                else:
                    if debug:
                        logging.info("> regex={regex} DOES NOT MATCH _b(path)={path}".format(regex=regex, path=_b(path)))
                    i += 1

            if len(paths_left) == 0:
                break

            # if not allow_multiple:
            j += 1
        return matching_paths

    @classmethod
    def list_files(Klass, direc,
                   # keep_direc=True
                   ):
        def _list_files(direc):
            def _path(path):
                return _j(direc, path)
            try:
                pass  # Cmd
                return [_path(path) for path in os.listdir(direc)]
            except Exception as e:
                import ipdb;
                ipdb.set_trace()
                raise e


        if type(direc) == list:
            all_files = []
            for d in direc:
                all_files.extend(_list_files(d))
            return all_files

        return _list_files(direc)

    @classmethod
    def all_directories_rooted_at(Klass, root_dir):
        direcs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            direcs.append(_j(root_dir, dirpath))
        return direcs

    @classmethod
    def find_source_directories(Klass, root_dir):
        src_dirs = dict()
        all_dirs = Klass.all_directories_rooted_at(root_dir)
        for direc in all_dirs:
            src_files = Klass.get_source_files(direc)
            if src_files.has_all_required_paths:
                assert direc not in src_dirs
                src_dirs[direc] = src_files
        return src_dirs

    @classmethod
    def as_source_files(Klass, srcs, debug=False):
        assert type(srcs) == list
        return Klass._get_source_files(srcs, debug=debug)

    @classmethod
    def get_source_files(Klass, direc, debug=False):
        basenames = Klass.list_files(direc,
                                     # keep_direc=False
                                     )
        return Klass._get_source_files(basenames, debug=debug)


    @classmethod
    def _get_source_files(Klass, paths, debug=False):
        return Klass._as_src_files(paths,
                                   req_regexes=Klass.required_source_basename_regexes(),
                                   opt_regexes=Klass.optional_source_basename_regexes(),
                                   allow_multiple_src_matches=Klass.allow_multiple_src_matches(),
                                   debug=debug)

    @classmethod
    def glob_target_files(Klass, paths, debug=False):
        return Klass._as_src_files(paths,
                                   req_regexes=Klass.target_basename_regexes(),
                                   # opt_regexes=Klass.optional_source_basename_regexes(),
                                   # allow_multiple_src_matches=Klass.allow_multiple_src_matches(),
                                   debug=debug)

    @classmethod
    def _as_src_files(Klass, paths, req_regexes, opt_regexes=None, allow_multiple_src_matches=True, debug=False):
        # basenames = Klass.list_files(direc, keep_direc=False)

        def _mk_src_files(src_paths):

            # req_regexes = Klass.required_source_basename_regexes()
            req_paths = Klass._match_regexes(req_regexes, src_paths, allow_multiple_src_matches,
                                             debug=debug)

            # opt_regexes = Klass.optional_source_basename_regexes()
            if opt_regexes is not None:
                opt_paths = Klass._match_regexes(opt_regexes, src_paths, allow_multiple_src_matches)
            else:
                opt_paths = None

            directory = common_dir(src_paths)
            src_files = SrcFiles(directory, req_paths, opt_paths,
                                 has_all_required_paths=len(req_paths) == len(req_regexes),
                                 allow_multiple_src_matches=allow_multiple_src_matches)
            return src_files

        # directory = common_dir(paths)

        direc_to_srcs = dict()
        for path in paths:
            direc = _d(path)
            _mk(direc_to_srcs, direc, [])
            direc_to_srcs[direc].append(path)

        if len(direc_to_srcs) == 1:
            direc = list(direc_to_srcs.keys())[0]
            return _mk_src_files(direc_to_srcs[direc])

        src_files_list = []
        for direc, src_paths in direc_to_srcs.items():
            src_files_list.append(_mk_src_files(src_paths))
        src_files = SrcFilesGroup(src_files_list)
        return src_files

    @classmethod
    def config_get(Klass, src_files, attr, default):
        config_path = src_files.get('config_json', or_none=True)
        if config_path is None:
            value = default
        else:
            config_json = load_json(config_path)
            if attr in config_json and config_json[attr] is None:
                config_json[attr] = default
            value = config_json[attr]
        return value




class ProfilerParser(ProfilerParserCommonMixin):
    """
         13 function calls in 0.000 seconds

       Ordered by: call count, file name, function name, line number

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/atari-py/atari_py/ale_python_interface.py:305(decodeState)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/atari-py/atari_py/ale_python_interface.py:291(deleteState)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/atari-py/atari_py/ale_python_interface.py:287(restoreSystemState)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/baselines/baselines/deepq/simple_refactor.py:2753(__exit__)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/baselines/baselines/deepq/simple_refactor.py:853(iter)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/core.py:138(unwrapped)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/core.py:291(unwrapped)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/envs/atari/atari_env.py:177(restore)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/clone/gym/gym/envs/atari/atari_env.py:167(restore_full_state)
            1    0.000    0.000    0.000    0.000 /home/jgleeson/envs/benchmark_tf/lib/python3.5/site-packages/numpy/ctypeslib.py:438(as_ctypes)
            1    0.000    0.000    0.000    0.000 {built-in method builtins.len}
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
            1    0.000    0.000    0.000    0.000 {method 'from_address' of '_ctypes.PyCArrayType' objects}

    {
    'ncalls', [1, 1, 1, ...],
    'tottime', [0.0, ...],
    ...
    'filename:lineno(function)', [...],
    }
    """
    def __init__(self, parser, args, src_files,
                 # profile_path,
                 # bench_name,
                 data=None,
                 bench_name=NO_BENCH_NAME):
        self.parser = parser
        self.args = args
        self.bench_name = bench_name
        self.src_files = src_files
        self.data = data
        # assert data is not None
        self.time_fields = ['tottime', 'cumtime', 'tottime_percall', 'cumtime_percall']
        self.skip = False

    def parse_columns(self, line, it):
        raise NotImplementedError()

    def profile_path(self, bench_name):
        return self.src_files.get('profile_path', bench_name)

    def dump_path(self, bench_name):
        return self._pretty_profile_path(bench_name)

    @classmethod
    def get_pretty_profile_path(Klass, src_files, bench_name):
        assert re.search(r'\.txt$', src_files.get('profile_path', bench_name))

        pretty_base = re.sub(r'\.txt$', '.pretty.csv', _b(src_files.get('profile_path', bench_name)))
        return _j(_d(src_files.get('profile_path', bench_name)), pretty_base)

    def _pretty_profile_path(self, bench_name):
        return self.get_pretty_profile_path(self.src_files, bench_name)

    def store(self, *args, **kwargs):
        store_group(self.results, *args, **kwargs)

    def pre_parse(self, bench_name):
        pass

    def parse(self, bench_name):
        assert self.time_fields is not None
        assert self.total_time_fields is not None
        assert self.sortby is not None

        self.results = dict()

        self.pre_parse(bench_name)

        self.header = None
        with open(self.profile_path(bench_name)) as f:
            it = line_iter(f, lstrip=True)
            for line in it:

                if self.args.debug:
                    logging.info("> {klass}, line :: {line}".format(
                        klass=self.__class__.__name__,
                        line=line))

                if self.parse_other(line, it):
                    continue

                if self.header is None and self.parse_header(line, it):
                    continue

                if re.search(r'^\s*$', line):
                    continue

                if self.header is not None:
                    if self.parse_columns(line, it):
                        continue

        self.post_parse(bench_name)

        # if not self.skip:
        #     repetitions = self.data[self.bench_name]['repetitions']
        #     iterations = self.data[self.bench_name]['iterations']
        #     # NOTE: We since we measure like this:
        #     # profiler.start()
        #     # for r in repetitions:
        #     #   for i in iterations:
        #     #     do_iteration()
        #     # profiler.end()
        #     #
        #     # We the number of python calls to EVERY function to be a multiple of (repetitions * iterations)
        #
        #     # TODO: generalize ncalls for CUDA profile
        #     # assert (np.array(self.results[self.bench_name]['ncalls']) % (repetitions * iterations) == 0).all()
        #
        #     for key in self.total_time_fields:
        #         time_sec = self.results[key]
        #         new_key = self.per_iter_field(key)
        #         time_sec_all_calls = self.results[key]
        #         self.results[new_key] = list(np.array(time_sec_all_calls)/float(repetitions * iterations))

    def per_iter_field(self, name):
        assert name in self.total_time_fields
        return "{name}_per_iter".format(name=name)

    def seconds_field(self, name):
        assert name in self.time_fields
        return "{name}_seconds".format(name=name)

    def dump_header(self):
        header = []
        for k in self.header:
            # NOTE: must match order of append inside each_line.
            header.append(k)
            if k in self.time_fields:
                header.append(self.seconds_field(k))
            if k in self.total_time_fields:
                header.append(self.per_iter_field(k))
        return header

    def time_per_call(self, time_sec_all_calls):
        repetitions = self.data[self.bench_name]['repetitions']
        iterations = self.data[self.bench_name]['iterations']
        return time_sec_all_calls/float(repetitions * iterations)

        # for key in self.total_time_fields:
        #     time_sec = self.results[key]
        #     new_key = self.per_iter_field(key)
        #     time_sec_all_calls = self.results[key]
        #     self.results[new_key] = list(np.array(time_sec_all_calls)/float(repetitions * iterations))

    def dump(self, bench_name):
        if self.skip:
            return

        dump_header = self.dump_header()
        rows = []

        sortby_indices = [dump_header.index(h) for h in self.sortby]
        def key_func(row):
            return [row[i] for i in sortby_indices]

        with open(self._pretty_profile_path(bench_name), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(dump_header)
            for line in self.each_line():
                rows.append(line)
            rows.sort(key=key_func, reverse=True)
            for row in rows:
                writer.writerow(row)

    def each_line(self):
        raise NotImplementedError()

class SrcFilesMixin:
    def get_bench_name(self, ParserKlass, allow_none=False):
        if not ParserKlass.uses_all_benches():
            bench_names = self.bench_names
            if len(bench_names) == 0 and allow_none:
                return None
            try:
                pass  # Cmd
                assert len(bench_names) == 1
            except Exception as e:
                import pdb;
                pdb.set_trace()
                raise e

            bench_name = list(bench_names)[0]
            return bench_name
        return NO_BENCH_NAME

    def all_targets(self, ParserKlass):
        targets = []
        if ParserKlass.uses_all_benches():
            bench_name = NO_BENCH_NAME
            targets.extend(ParserKlass.get_targets(self, bench_name))
        else:
            for bench_name in self.bench_names:
                targets.extend(ParserKlass.get_targets(self, bench_name))
        return targets

    def check_has_all_required_paths(self, ParserKlass):
        if not self.has_all_required_paths:
            logging.info(
                textwrap.dedent("""
ERROR: Didn't find all required source files in directory={dir} for parser={parser}
  src_files =
{src_files}
  required_files = 
{required_files}
                """.format(
                    dir=self.directory,
                    parser=ParserKlass.__name__,
                    # src_files=str(src_files),
                    src_files=textwrap.indent(str(self), prefix="  "*2),
                    required_files=as_str(ParserKlass.required_source_basename_regexes(), indent=2),
                )))

class SrcFiles(SrcFilesMixin):
    """
    A bunch of files belongining to the same directory matching certain patterns.

    Files have been grouped together based on regex's that they match.
    Each regex has a name associated with it.
    Also, each regex will have a 'bench_name' capturing group.
    """
    def __init__(self, directory, req_paths, opt_paths, has_all_required_paths, allow_multiple_src_matches):
        # PSEUDOCODE:
        # if uses bench_name:
        #     req_paths['profile_path'][bench_name] = ...
        # else:
        #     req_paths['profile_path'][None] = ...
        self.req_paths = req_paths
        self.opt_paths = opt_paths
        self.has_all_required_paths = has_all_required_paths
        self.allow_multiple_src_matches = allow_multiple_src_matches
        self.directory = directory

    @property
    def bench_names(self):
        # bench_names = set()
        def _bench_names(matches):
            bench_names = set()
            for match_name, bench_matches in matches.items():
                for k in bench_matches.keys():
                    bench_names.add(k)
            return bench_names
        req_bench_names = _bench_names(self.req_paths)
        # opt_bench_names = _bench_names(self.opt_paths)
        # assert opt_bench_names.issubset(req_bench_names)
        # For each bench_name, there should be a full set of required/optional files.
        # assert req_bench_names == opt_bench_names

        return list(req_bench_names)

    def __str__(self):
        return as_str(self)

    def _all_sources_all_bench_names(self):
        bench_names = self.bench_names
        srcs = []
        for bench_name in bench_names:
            srcs.extend(self.all_sources(bench_name))
        return srcs

    def all_sources(self, bench_name=NO_BENCH_NAME, all_bench_names=False):
        if all_bench_names:
            return self._all_sources_all_bench_names()

        def srcs_for(paths):
            if paths is None:
                return []
            sources = []
            for regex_name in paths.keys():
                if bench_name in paths[regex_name]:
                    sources.append(paths[regex_name][bench_name])
            return sources

        return srcs_for(self.req_paths) + \
               srcs_for(self.opt_paths)

    def get(self, src_file_name, bench_name=NO_BENCH_NAME, or_none=False):
        def _get(paths):
            if paths is None:
                return None
            if src_file_name in paths:
                if bench_name in paths[src_file_name]:
                    return paths[src_file_name][bench_name]
            return None

        path = _get(self.req_paths)
        if path is not None:
            return path

        path = _get(self.opt_paths)
        if path is not None:
            return path

        if or_none:
            return None

        raise KeyError((bench_name, src_file_name))


    @property
    def is_group(self):
        return False

    @property
    def directories(self):
        return [self.directory]

    def get_src_files(self, directory):
        assert directory == self.directory
        return self

class SrcFilesGroup(SrcFilesMixin):
    """
    A list of directories, with each directory having a bunch of files matching certain patterns.
    A list of SrcFiles objects.

    Implements a lot of the same interface as SrcFiles (needed by SConstruct).
    """
    def __init__(self, src_files_list):
        # req_paths, opt_paths, has_all_required_paths, allow_multiple_src_matches):
        self.src_files_list = src_files_list
        self._direc_to_src_files = dict(
            (src_files.directory, src_files) for src_files in self.src_files_list)

    @property
    def directory(self):
        srcs = self.all_sources(all_bench_names=True)
        return common_dir(srcs)

    @property
    def bench_names(self):
        benches = set()
        for src_files in self.src_files_list:
            for bench in src_files.bench_names:
                benches.add(bench)
        return list(benches)

    @property
    def is_group(self):
        return True

    @property
    def directories(self):
        return [src_files.directory for src_files in self.src_files_list]

    def get_src_files(self, directory):
        return self._direc_to_src_files[directory]

    @property
    def has_all_required_paths(self):
        return all(src_files.has_all_required_paths \
                   for src_files in self.src_files_list)

    def all_sources(self, bench_name=NO_BENCH_NAME, all_bench_names=False):
        """
        If all_bench_names=True, get source files spanning all bench_names.

        :return:
        """
        srcs = []
        for src_files in self.src_files_list:
            srcs.extend(src_files.all_sources(bench_name=bench_name, all_bench_names=all_bench_names))
        return srcs


def sort_xs_by_ys(xs, ys):
    """
    Sort xs by the elements in ys.

    >>> xs = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    >>> ys = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
    >>> sort_xs_by_ys(xs, ys)
    ["a", "d", "h", "b", "c", "e", "i", "f", "g"]
    """
    assert len(xs) == len(ys)
    return [x for x, y in sorted(zip(xs, ys), key=lambda x_y: x_y[1])]

def pretty_time(time_sec, use_space=True):
    MS_IN_SEC = 1e3
    US_IN_SEC = 1e6
    NS_IN_SEC = 1e9

    unit_names = ['ms', 'us', 'ns']
    unit_in_sec = [MS_IN_SEC, US_IN_SEC, NS_IN_SEC]

    def format_str(time_as_unit, unit):
        if use_space:
            return "{time} {unit}".format(time=time_as_unit, unit=unit)
        return "{time}{unit}".format(time=time_as_unit, unit=unit)

    if time_sec == 0 or time_sec > 1:
        return format_str(time_sec, 'sec')
    for i, (time_unit, sec_as_unit) in enumerate(zip(unit_names, unit_in_sec)):
        time_as_unit = time_sec*sec_as_unit
        if time_as_unit > 1 or i == len(unit_names) - 1:
            return format_str(time_as_unit, time_unit)
    assert False

def us_to_sec(usec):
    return usec/1e6

def sec_to_us(sec):
    return sec*1e6

def sec_to_ms(sec):
    return sec*1e3

def unique(xs):
    return list(set(xs))

def remap_keys(dic, new_key_func):
    new_dic = dict()
    for k, v in dic.items():
        new_key = new_key_func(k)
        new_dic[new_key] = v
    return new_dic

def _mk(dic, key, default):
    if key not in dic:
        dic[key] = default
    return dic[key]

def common_dir(paths):
    assert len(paths) > 0
    directory = os.path.commonprefix(paths)
    if not os.path.isdir(directory):
        directory = _d(directory)
    try:
        pass  # Cmd
        assert os.path.isdir(directory)
    except Exception as e:
        import pdb;
        pdb.set_trace()
        raise e

    return directory

def as_str(obj, indent=None):
    ss = StringIO()
    if type(obj) == dict:
        pprint.pprint(obj, stream=ss)
    else:
        pprint.pprint(obj.__dict__, stream=ss)
    string = ss.getvalue()
    if indent is not None:
        string = textwrap.indent(string, "  "*indent)
    return string

def bench_suffix(bench):
    if bench != NO_BENCH_NAME and bench is not None:
        return ".op_{bench}".format(bench=bench)
    return ""

def trace_suffix(trace_id, allow_none=False):
    if trace_id is None and not allow_none:
        raise RuntimeError("trace_id must be >= 0, got None")

    if trace_id is not None:
        return ".trace_{id}".format(id=trace_id)
    return ""

def sess_suffix(session_id, allow_none=False):
    if session_id is None and not allow_none:
        raise RuntimeError("session_id must be >= 0, got None")

    if session_id is not None:
        return ".session_{id}".format(id=session_id)
    return ""

def debug_suffix(debug):
    if debug:
        return ".debug"
    return ""

def process_suffix(process_name):
    # assert process_name is not None
    if process_name is not None:
        return ".process_{proc}".format(proc=process_name)
    return ""

def machine_suffix(machine_name):
    if machine_name is not None:
        return ".machine_{mach}".format(mach=machine_name)
    return ""

def phase_suffix(phase_name):
    # assert phase_name is not None
    if phase_name is not None:
        return ".phase_{phase}".format(phase=phase_name)
    return ""

def overlap_type_suffix(overlap_type):
    # assert overlap_type is not None
    if overlap_type is not None:
        return ".overlap_{overlap}".format(overlap=overlap_type)
    return ""

def ops_suffix(ops):
    if ops is None:
        return ""

    if type(ops) == str:
        ops = [ops]

    # NOTE: operation names use underscores; we cannot different ops...
    return ".ops_{ops}".format(ops="_".join(sorted(ops)))

def resources_suffix(resources):
    if resources is not None:
        return ".resources_{resources}".format(
            resources="_".join(sorted(resources)))
    return ""

def event_suffix(event_id):
    if event_id is not None:
        return ".event_{id}".format(id=event_id)
    return ""

def step_suffix(step, allow_none=False):
    assert allow_none or step is not None
    if step is None:
        return ""
    return ".step_{id}".format(id=step)

def device_id_suffix(device_id, device_name=None, allow_none=False):
    if device_id is None and not allow_none:
        raise RuntimeError("device_id must be >= 0, got None")

    if device_id is not None and device_name is not None:
        name = device_name
        name = re.sub(r' ', '_', name)
        return ".device_{id}_{name}".format(
            id=device_id,
            name=name,
        )
    elif device_id is not None:
        return ".device_{id}".format(
            id=device_id,
        )

    return ""


def load_json(path):
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        data = fixup_json(data)
        return data

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            # TODO: Should we convert it to float(...)?
            # str will retain precision... but makes it annoying "reading it back in"...
            return float(o)
            # return str(o)
        return super(DecimalEncoder, self).default(o)

def do_dump_json(data, path, cls=DecimalEncoder):
    os.makedirs(_d(path), exist_ok=True)
    json.dump(data,
              codecs.open(path, mode='w', encoding='utf-8'),
              sort_keys=True, indent=4,
              skipkeys=False,
              cls=cls)

def store_group(dic, m, types=dict(), store_as=None, replace=False):
    groupdict = m.groupdict()
    for k, v in groupdict.items():
        if k in types:
            value = types[k](v)
        else:
            value = as_value(v)

        if store_as is None:
            assert replace or k not in dic
            dic[k] = value
        elif store_as == 'list':
            if k not in dic:
                dic[k] = []
            dic[k].append(value)

def store_as(dic, k, v, types=dict(), store_type=None, replace=False):
    if k in types:
        value = types[k](v)
    else:
        value = as_value(v)

    if store_type is None:
        assert replace or k not in dic
        dic[k] = value
    elif store_type == 'list':
        if k not in dic:
            dic[k] = []
        dic[k].append(value)
    else:
        raise NotImplementedError

def fixup_json(obj):
    def fixup_scalar(scalar):
        if type(scalar) != str:
            ret = scalar
            return ret

        try:
            ret = int(scalar)
            return ret
        except ValueError:
            pass

        try:
            ret = float(scalar)
            return ret
        except ValueError:
            pass

        ret = scalar
        return ret

    def fixup_list(xs):
        return [fixup_json(x) for x in xs]

    def fixup_dic(dic):
        items = list(dic.items())
        keys = [k for k, v in items]
        values = [v for k, v in items]
        keys = fixup_json(keys)
        values = fixup_json(values)
        new_dic = dict()
        for k, v in zip(keys, values):
            new_dic[k] = v
        return new_dic

    if type(obj) == dict:
        return fixup_dic(obj)
    elif type(obj) == list:
        return fixup_list(obj)
    return fixup_scalar(obj)

def line_iter(f, lstrip=False):
    for line in f:
        line = line.rstrip()
        if lstrip:
            line = line.lstrip()
        yield line

def as_value(x):
    if type(x) in [int, float, list, set, dict]:
        return x

    assert type(x) == str

    try:
        val = int(x)
        return val
    except ValueError:
        pass

    try:
        val = float(x)
        return val
    except ValueError:
        pass

    return x

def put_key(d, key, value):
    if key not in d:
        d[key] = value

def compute_num_calls_dqn(bench_data):
    # This is the number of times (for e.g.) Forward was called.
    # We expect the number of times a particular CUDA-function/CUDA-API is called to be a multiple of
    # num_calls = iterations*repetitions
    #
    # NOTE: +1 for the extra initial call we do to account for profiler weirdness during warmup (i.e. nvprof).

    num_calls = bench_data['iterations']*bench_data['repetitions'] + 1
    # num_calls = bench_data['iterations']*bench_data['repetitions']

    return num_calls

def compute_num_calls(config):
    assert ( 'iterations' in config and 'repetitions' in config ) \
           or 'num_calls' in config

    if 'num_calls' in config:
        num_calls = config['num_calls']
    else:
        num_calls = config['iterations'] * config['repetitions']

    return num_calls

def start_end_nsec_to_usec(start_nsec, end_nsec):
    return nsec_to_usec(end_nsec - start_nsec)

def nsec_to_usec(nsec):
    return nsec/1e3

def get_microbench_basename():
    return 'microbenchmark.json'

def get_microbench_path(direc):
    return _j(direc, 'microbenchmark.json')

def get_unit_test_once_path(directory, trace_id, bench_name):
    ret = _j(directory, "unit_test_once{bench}{trace}.proto".format(
        bench=bench_suffix(bench_name),
        trace=trace_suffix(trace_id),
    ))
    return ret

def get_unit_test_multiple_path(directory, trace_id, bench_name):
    ret = _j(directory, "unit_test_multiple{bench}{trace}.proto".format(
        bench=bench_suffix(bench_name),
        trace=trace_suffix(trace_id),
    ))
    return ret

def is_microbench_path(path):
    return re.search(r"^microbenchmark\.json$", _b(path))

def make_json_serializable(data):
    new_data = dict()
    for k in data.keys():
        new_data[k] = list(data[k])
    return new_data

def make_json_ndarray(data):
    new_data = dict()
    for k in data.keys():
        if type(data[k]) == list:
            new_data[k] = np.array(data[k])
        else:
            new_data[k] = data[k]
    return new_data

def compute_total_times(cpp_and_gpu_times, python_times):
    total_times = cpp_and_gpu_times + python_times
    return total_times

def compute_theoretical_speedup(cpp_and_gpu_times, python_times):
    total_times = compute_total_times(cpp_and_gpu_times, python_times)
    theoretical_speedup = total_times / cpp_and_gpu_times
    return theoretical_speedup

def compute_percent_time_in_python(cpp_and_gpu_times, python_times):
    total_times = compute_total_times(cpp_and_gpu_times, python_times)
    percent_time_in_python = 100. * python_times / total_times
    return percent_time_in_python

def compute_percent_time_in_gpu(gpu_times, total_times):
    percent_time_in_gpu = 100. * gpu_times / total_times
    return percent_time_in_gpu

def compute_python_overhead_percent(cpp_and_gpu_times, python_times):
    python_overhead_percent = 100. * python_times / cpp_and_gpu_times
    return python_overhead_percent

def as_order_map(xs):
    order_map = dict()
    for i, x in enumerate(xs):
        order_map[x] = i
    return order_map

def each_order(order_map, rev_order_map):
    for x_order in range(max(order_map.values()) + 1):
        x = rev_order_map[x_order]
        yield x, x_order

def reverse_dict(dic):
    r_dic = dict()
    for k, v in dic.items():
        r_dic[v] = k
    return r_dic

def get_pretty_bench(bench_name):
    return BENCH_NAME_TO_PRETTY.get(bench_name, bench_name)

class DataFrame:
    """
    Useful functions for manipulating pandas dataframes.
    """

    @staticmethod
    def get_mean_std(df, value_field):
        # NOTE: Pandas throws an error if you attempt to use decimal.Decimal objects for groupby.
        # This will convert to a float64.
        df[value_field] = df[value_field].astype(float)
        groupby_cols = DataFrame.get_groupby_cols(df, value_field)
        mean = df.groupby(groupby_cols).mean().rename(columns={value_field: 'mean'}).reset_index()
        std = df.groupby(groupby_cols).std().rename(columns={value_field: 'std'}).reset_index()
        mean['std'] = std['std']
        return mean

    @staticmethod
    def print_df(df, **kwargs):
        pd.options.display.max_rows = None
        pd.options.display.max_columns = None
        pd.options.display.width = 9999999
        print(df, **kwargs)
        pd.reset_option('all')


    @staticmethod
    def get_groupby_cols(df, value_field):
        groupby_cols = [field for field in list(df.keys()) if field != value_field]
        return groupby_cols

class ParserException(Exception):
    pass

class MissingInputFiles(ParserException):
    pass

def IsGPUTime(device):
    return re.search('stream:all', device)

def IsCPUTime(device):
    # NOTE: do NOT use re.search or re.match here!
    # We must match the whole device string to match tfprof behaviour.
    return re.fullmatch(".*/(device:gpu|gpu|device:cpu|cpu|device:sycl):\\d+", device)

def read_proto(ProtoKlass, path):
    with open(path, 'rb') as f:
        proto = ProtoKlass()
        proto.ParseFromString(f.read())
    return proto

def read_tfprof_file(path):
    from tensorflow.core.profiler.tfprof_log_pb2 import ProfileProto
    return read_proto(ProfileProto, path)

def read_machine_util_file(path):
    return read_proto(MachineUtilization, path)

def read_training_progress_file(path):
    return read_proto(IncrementalTrainingProgress, path)

def read_pyprof_file(path):
    return read_proto(Pyprof, path)

def read_process_metadata_file(path):
    return read_proto(ProcessMetadata, path)

def read_unit_test_once_file(path):
    return read_proto(IMLUnitTestOnce, path)

def read_unit_test_multiple_file(path):
    return read_proto(IMLUnitTestMultiple, path)

def read_pyprof_call_times_file(path):
    with open(path, 'rb') as f:
        call_times_data = pickle.load(f)
    return call_times_data

def is_tfprof_file(path):
    base = _b(path)
    m = re.search(r'profile{bench}{trace}{sess}\.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
        sess=SESSION_SUFFIX_RE,
    ), base)
    return m

def is_unit_test_once_file(path):
    base = _b(path)
    m = re.search(r'unit_test_once{bench}{trace}.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_unit_test_multiple_file(path):
    base = _b(path)
    m = re.search(r'unit_test_multiple{bench}{trace}\.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_machine_util_file(path):
    base = _b(path)
    m = re.search(r'machine_util{trace}\.proto'.format(
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_training_progress_file(path):
    base = _b(path)
    m = re.search(r'training_progress{trace}\.proto'.format(
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_pyprof_call_times_file(path):
    base = _b(path)
    m = re.search(r'pyprof_call_times{bench}{trace}\.pickle'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_pyprof_file(path):
    base = _b(path)
    m = re.search(r'pyprof{bench}{trace}\.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_process_metadata_file(path):
    base = _b(path)
    m = re.search(r'process_metadata{bench}{trace}\.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_dump_event_file(path):
    base = _b(path)
    m = re.search(r'dump_event{bench}{trace}\.proto'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def is_insertable_file(path):
    """
    Does this file contain event data that can be inserted using SQLParser?
    """
    return is_tfprof_file(path) or \
           is_pyprof_file(path) or \
           is_dump_event_file(path) or \
           is_pyprof_call_times_file(path) or \
           is_machine_util_file(path) or \
           is_training_progress_file(path)

def is_trace_file(path):
    """
    Does this file contain data that should NOT exist if we re-run IML?
    (this is to prevent keeping data from old runs and interpreting it as being from the same run).
    """
    return is_tfprof_file(path) or \
           is_pyprof_file(path) or \
           is_dump_event_file(path) or \
           is_pyprof_call_times_file(path) or \
           is_machine_util_file(path) or \
           is_unit_test_once_file(path) or \
           is_unit_test_multiple_file(path) or \
           is_config_file(path) or \
           is_process_metadata_file(path) or \
           is_training_progress_file(path)

def is_process_trace_file(path):
    """
    Return true if this file contains a proto that has start/end events for a specific process.
    """
    return is_tfprof_file(path) or \
           is_pyprof_file(path) or is_dump_event_file(path) or \
           is_pyprof_call_times_file(path)

def is_config_dir(path):
    return os.path.isdir(path) and re.search(CONFIG_RE, _b(path))

def config_is_uninstrumented(config):
    return re.search(r'uninstrumented', config)

def config_is_no_tfprof(config):
    return re.search(r'no_tfprof', config)

def config_is_no_pyprof(config):
    return re.search(r'no_pyprof', config)

def config_is_no_pytrace(config):
    return re.search(r'no_pytrace', config)

def config_is_no_pydump(config):
    return re.search(r'no_pydump', config)

def config_is_no_tfdump(config):
    return re.search(r'no_tfdump', config)

def config_is_instrumented(config):
    return not config_is_uninstrumented(config) and re.search(r'instrumented', config)

def config_is_full(config):
    # Does this config run for the full training run (i.e. all timesteps)?
    return re.search(r'_full', config)

def is_config_file(path):
    base = _b(path)
    m = re.search(r'config{bench}{trace}.json'.format(
        bench=BENCH_SUFFIX_RE,
        trace=TRACE_SUFFIX_RE,
    ), base)
    return m

def now_us():
    return time.time()*MICROSECONDS_IN_SECOND

def get_stacktrace(n_skip=1, indent=None):
    """
    Return a stacktrace ready for printing; usage:

    # Dump the stack of the current location of this line.
    '\n'.join(get_stacktrace(0))
    logging.info("First line before stacktrace\n{stack}".format(
        stack=get_stacktrace())

    # Outputs to stdout:
    ID=14751/MainProcess @ finish, profilers.py:1658 :: 2019-07-09 15:52:26,015 INFO: First line before stacktrace
      File "*.py", line 375, in <module>
          ...
      ...
      File "*.py", line 1658, in finish
        logging.info("First line before stacktrace\n{stack}".format(

    :param n_skip:
        Number of stack-levels to skip in the caller.
        By default we skip the first one, since it's the call to traceback.extrace_stack()
        inside the get_stacktrace() function.

        If you make n_skip=2, then it will skip you function-call to get_stacktrace() as well.

    :return:
    """
    stack = traceback.extract_stack()
    stack_list = traceback.format_list(stack)
    stack_list_keep = stack_list[0:len(stack_list)-n_skip]
    stack_str = ''.join(stack_list_keep)
    if indent is not None:
        stack_str = textwrap.indent(stack_str, prefix="  "*indent)
    return stack_str

def get_category_from_device(device):
    if IsGPUTime(device):
        return CATEGORY_GPU
    elif IsCPUTime(device):
        return CATEGORY_CUDA_API_CPU
    return CATEGORY_UNKNOWN

def reversed_iter(xs):
    n = len(xs)
    for i in range(n - 1, 0 - 1, -1):
        yield xs[i]

def proc_as_category(process_name):
    return "PROC:{proc}".format(proc=process_name)

def match_proc_category(category):
    m = re.search(r'^PROC:(?P<process_name>.*)', category)
    return m

def match_cpu_gpu_category(category):
    regex = r'^(?:{regex})$'.format(
        regex='|'.join([CATEGORY_CPU, CATEGORY_GPU]))
    m = re.search(regex, category)
    return m

def update_dict(dic1, dic2, overwrite=False):
    if not overwrite:
        common_keys = set(dic1.keys()).intersection(set(dic2.keys()))
        assert len(common_keys) == 0
    dic1.update(dic2)

# Pickle

def should_load_memo(debug_memoize, path):
    return debug_memoize and _e(path) and os.stat(path).st_size > 0

def load_memo(debug_memoize, path):
    with open(path, 'rb') as f:
        # -1 specifies highest binary protocol
        ret = pickle.load(f)
    return ret

def maybe_memoize(debug_memoize, ret, path):
    if debug_memoize:
        with open(path, 'wb') as f:
            # -1 specifies highest binary protocol
            pickle.dump(ret, f, -1)

def progress(xs=None, desc=None, total=None, show_progress=False):
    if xs is None:
        assert total is not None

    if not show_progress:
        return tqdm_progress(xs, desc=desc, disable=True)

    if total is None:
        if hasattr(xs, '__len__'):
            total = len(xs)
        elif hasattr(xs, 'rowcount') and getattr(xs, 'rowcount') != -1:
            total = xs.rowcount
        else:
            logging.info("> WARNING: not sure what total to use for progress(desc={desc})".format(
                desc=desc))
    return tqdm_progress(xs, desc=desc, total=total)

def as_progress_label(func_name, debug_label):
    if debug_label is None:
        progress_label = func_name
    else:
        progress_label = "{func}: {debug_label}".format(
            func=func_name,
            debug_label=debug_label)
    return progress_label

def split_list(xs, n):
    assert n > 0
    chunk_size = max(int(len(xs)/float(n)), 1)
    n_chunks = n
    for i in range(n_chunks-1):
        start = i*chunk_size
        stop = (i + 1)*chunk_size
        yield xs[start:stop]
    start = (n_chunks - 1)*chunk_size
    yield xs[start:]

# Taken from cProfile's Lib/pstats.py;
# func_name is 3-tuple of (file-path, line#, function_name)
# e.g.
#     ('Lib/test/my_test_profile.py', 259, '__getattr__'):
def pyprof_func_std_string(func_tuple): # match what old profile produced
    if func_tuple[:2] == ('~', 0):
        # special case for built-in functions
        name = func_tuple[2]
        if name.startswith('<') and name.endswith('>'):
            return '{%s}' % name[1:-1]
        else:
            return name
    else:
        path, lineno, func_name = func_tuple
        new_path = path
        # Full path is useful for vim when visiting lines; keep it.
        # new_path = re.sub(r'.*/site-packages/', '', new_path)
        # new_path = re.sub(r'.*/clone/', '', new_path)
        # new_path = re.sub(r'.*/lib/python[^/]*/', '', new_path)
        return "{path}:{lineno}({func_name})".format(
            path=new_path,
            lineno=lineno,
            func_name=func_name,
        )

def as_type(value, klass):
    return klass(value)

def num_dump_cpus():
    """
    To minimize interference from profiling,
    reserve the last num_dump_cores() cpus for doing profiling operations
    (i.e. dumping data from DRAM to disk).

    Current huuristic:
    We aim to use <= 1/8-th of the CPU-cores

    NOTE: we should probably consider physical cores, not logical ones.
    """
    num_cpus = psutil.cpu_count()
    if num_cpus <= 8:
        return 1
    if num_cpus <= 16:
        return 2
    # if num_cpus <= 32:
    return 4

def get_dump_cpus():
    num_cpus = psutil.cpu_count()
    dump_cpus = num_dump_cpus()
    return list(range(num_cpus - dump_cpus, num_cpus))

def get_runnable_cpus():
    num_cpus = psutil.cpu_count()
    dump_cpus = num_dump_cpus()
    return list(range(0, num_cpus - dump_cpus))

def get_username():
    return pwd.getpwuid(os.getuid())[0]

def pprint_msg(dic, prefix='  '):
    """
    Give logging.info a string for neatly printing a dictionary.

    Usage:
    logging.info(pprint_msg(arbitrary_object))
    """
    return "\n" + textwrap.indent(pprint.pformat(dic), prefix=prefix)

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
        keys = sorted(obj.keys())
        for key in keys:
            value = obj[key]
            key_values_pairs.append((
                js_friendly(key),
                js_friendly(value)))
        return key_values_pairs
    elif type(obj) == dict and len(obj) > 0:
        new_obj = dict()
        for k in obj.keys():
            new_obj[k] = js_friendly(obj[k])
        return new_obj
    elif type(obj) == frozenset:
        return sorted([js_friendly(x) for x in obj])
    return obj

class pythunk:
    """
    Create a 'thunk' when evaluating func(*args, *kwargs).
    i.e. don't run func as normal when it's called.
    Instead, keep track of the arguments that were passed.
    @pythunk
    def f(x, y):
        return x + y
    thunk = func(1, 2)
    # >>> type(thunk) == pythunk
    thunk.evaluate()
    When thunk.evalulate()

    """
    def __init__(self, func):
        self.func = func
        self._evaluated = False

    def eval(self):
        if not self._evaluated:
            self._ret = self.func(*self.args, **self.kwargs)
            # Don't need these anymore.
            self.args = None
            self.kwargs = None
            self._evaluated = True
        return self._ret

    def maybe_eval(self, should_eval):
        if should_eval or self._evaluated:
            return self.eval()
        return self

    def __call__(self, *args, **kwargs):
        """
        Store the arguments used for the call to func(*args, **kwargs).
        However, don't actually run the function.
        (we run the function only once pythunk.eval() is called).

        :param args:
        :param kwargs:
        :return:
        """
        self.args = args
        self.kwargs = kwargs
        return self


def get_machine_name():
    """
    Portable way of calling hostname shell-command.

    Regarding docker containers:

      NOTE: If we are running from inside the docker-dev environment, then $(hostname) will return
      the container-id by default.

      For now we leave that behaviour.
      We can override it in the future by passing --hostname to the container.

      Documentation:
      https://docs.docker.com/config/containers/container-networking/#ip-address-and-hostname

    :return:
        Unique name for a node in the cluster
    """
    machine_name = platform.node()
    return machine_name

def cmd_debug_msg(cmd, env=None, dry_run=False):
    if type(cmd) == list:
        cmd_str = " ".join([shlex.quote(str(x)) for x in cmd])
    else:
        cmd_str = cmd

    lines = []
    if dry_run:
        lines.append("> CMD [dry-run]:")
    else:
        lines.append("> CMD:")
    lines.extend([
        "  $ {cmd}".format(cmd=cmd_str),
        "  PWD={pwd}".format(pwd=os.getcwd()),
    ])

    if env is not None and len(env) > 0:
        env_vars = sorted(env.keys())
        lines.append("  Environment:")
        for var in env_vars:
            lines.append("    {var}={val}".format(
                var=var,
                val=env[var]))
    string = '\n'.join(lines)

    return string

def log_cmd(cmd, env=None, dry_run=False):
    string = cmd_debug_msg(cmd, env=env, dry_run=dry_run)

    logging.info(string)

def print_cmd(cmd, files=sys.stdout, env=None, dry_run=False):
    string = cmd_debug_msg(cmd, env=env, dry_run=dry_run)

    if type(files) not in [set, list]:
        if type(files) in [list]:
            files = set(files)
        else:
            files = set([files])

    for f in files:
        print(string, file=f)
        f.flush()

def list_files(direc):
    paths = [_j(direc, filename) for filename in os.listdir(direc)]
    return paths
