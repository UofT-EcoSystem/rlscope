import cProfile, pstats, io
import codecs
import sys
import json
import pprint
import subprocess
import textwrap
import os
import time
import re
import math
import numpy as np

# global tf
# global tf_device_lib
# import tensorflow as tf
from tensorflow.python.client import device_lib as tf_device_lib

# pip install py-cpuinfo
import cpuinfo

from os import environ as ENV

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from profiler import cudaprofile

# Avoid using None for no bench_name; doesn't play nice with pandas/numpy
# (None == NaN in that context).
NO_BENCH_NAME = "NoBenchName"
NO_DEVICE_NAME = "NoDeviceName"
NO_IMPL_NAME = "NoImplName"

class Profiler:
    """
    Generic profiler that uses BOTH CUDAProfiler and PythonProfiler.

    Intended use case:

    profiler = Profiler(...)

    for epoch in range(epochs):
        for i in range(steps_per_epoch):

            #
            # Only collect profiling information during inner training loop operations.
            # Don't both collecting "noise" for one-time initialization.
            #

            profiler.enable_profiling()
            # Some part of your inner-training loop.
            # For e.g. if its MNIST, this will be both the Forward/Backward passes.
            sess.run(train_op, ...)
            profiler.disable_profiling()


    Members:

    self.profile_time_sec:
        Total time spent profiling, in seconds.
        i.e. the time spent in between enable/disable profiling calls:

        self.enable_profiling()
        ... # This time.
        self.disable_profiling()

        This time can be compared/validated against the total time reported
        by the profiler (nvprof/pyprof).

    :param exit_early
        if True, exit ML script immediately after benchmarking bench_name.
        Othwerwise, continue executing ML script until it finishes.
    """
    def __init__(self, directory,
                 bench_names=[NO_BENCH_NAME], bench_name=NO_BENCH_NAME, num_calls=None, start_measuring_call=None,
                 no_idempotent=False,
                 repetition_time_limit_sec=10.,
                 debug=False, idempotent_bench_names=[NO_BENCH_NAME],
                 exit_early=True):
        self.directory = directory
        self.exit_early = exit_early
        self.repetition_time_limit_sec = repetition_time_limit_sec
        self.num_calls = num_calls
        self.bench_names = set(bench_names if bench_names is not None else [])
        self.bench_name = bench_name
        self.start_measuring_call = start_measuring_call
        self.no_idempotent = no_idempotent
        self.debug = debug
        self.python_profiler = PythonProfiler(directory, bench_name=self.bench_name)
        self.cuda_profiler = CUDAProfiler()
        def init_bench_dict(default_value):
            return dict((bench_name, default_value) for bench_name in self.bench_names)
        self.idempotent_bench_names = set(idempotent_bench_names) if idempotent_bench_names is not None else set()
        self.start_t = init_bench_dict(None)
        self.end_t = init_bench_dict(None)
        self.time_sec = init_bench_dict(0.)
        # How many times has a block of code that we are intending to profile been run?
        # We expect to run that block of code at least
        # (self.start_measuring_call + self.num_calls) times.
        self.code_count = init_bench_dict(0)
        self.steps = 0

        # assert ( self.num_calls is None and self.start_measuring_call is None ) or \
        #        ( self.num_calls is not None and self.start_measuring_call is not None )
        # assert self.start_measuring_call is not None

    def is_idempotent(self, bench_name):
        return bench_name in self.idempotent_bench_names

    def _init_num_calls(self, bench_name, func, *args, **kwargs):
        """
        PSEUDOCODE:

        We may want to run additional iterations because:

        - We want (stdev / mean) to be below a percentage
          NOTE: it may be the case that what we are measuring is highly variable,
          and this threshold is never crossed!

        - We only want to run the benchmark for a certain time limit (e.g. 1 minute).

        - We need the # of iterations of what we are measuring to be large enough so that:
            time(# of iterations) < clock_precision

        Q: Is there something (statisically) wrong with running some functions MORE than others,
        then combining their results to get a standard deviation result?

          If something that's VERY short has high (relatively speaking) stdev, it will
          contribute LITTLE when adding it to a LONG function.

          This is fine, since we care about the function that dominates the runtime.


        total_time_sec = None
        iterations = 100
        # Total time to run a single repetition
        time_limit_sec = 1 minute
        do:
          start_t = time.time()
          for i in iterations:
            func(...)
          end_t = time.time()
          total_time_sec = end_t - start_t
        while (stdev / mean) > 1% and total_time_sec < time_limit_sec:
        """
        if self.num_calls is not None or self.no_idempotent or not self.is_idempotent(bench_name):
            return

        # Dynamically calculate number of iterations to run for; start with 10 at least.
        iterations = 10
        # max_stdev_percent = 1.
        # time_limit_sec = 60
        # time_limit_sec = 10
        # time_limit_sec = 100

        min_guess_sec = 1.

        def report_decision(total_time_sec, iterations):
            print("> Dynamic iterations for bench_name={b} decided on: {it} iterations".format(
                it=iterations,
                b=bench_name))
            print("  1 repetition takes ~ {sec} seconds".format(sec=total_time_sec))

        # repetition_time_sec = np.zeros(self.repetitions, dtype=np.float32)
        total_time_sec = None
        while True:
            # for r in range(self.repetitions):
            start_t = time.time()
            for i in range(iterations):
                # self._iter(r=-1, i=-1)
                func(*args, **kwargs)
            end_t = time.time()
            total_time_sec = end_t - start_t
            # repetition_time_sec[r] = total_time_sec
            if total_time_sec > self.repetition_time_limit_sec:
                # Took longer than 1 minute to run a single repetition;
                # just run it for that time and the stdev we get is what we get.
                report_decision(total_time_sec, iterations)
                # self.iterations = iterations
                self.num_calls = iterations
                return

            # stdev = np.std(repetition_time_sec)
            # mean = np.mean(repetition_time_sec)
            # if stdev/mean <= max_stdev_percent/100.:
            #     # After 3 repetitions, the standard-deviation as a percentage of the mean was <= 1%;
            #     # good enough!
            #     break

            if total_time_sec > min_guess_sec:
                # Use current iteration time to guess the minimum number of iterations needed to be >= time_limit_sec.
                #
                # ( 2^p ) * total_time_sec >= time_limit_sec
                # p >= log_2 [ time_limit_sec / total_time_sec ]
                # p = ceil [ log_2 [ time_limit_sec / total_time_sec ] ]
                next_iterations = iterations * 2**math.ceil(np.log2(self.repetition_time_limit_sec / total_time_sec))
                assert iterations < next_iterations
                iterations = next_iterations
            else:
                iterations *= 2

        # report_decision(total_time_sec, iterations)
        # self.iterations = iterations

    @property
    def python_profile_path(self):
        return self.python_profiler._prof_path

    def _start(self):
        if self.debug:
            print("    > Start CUDA profiler")
        self.cuda_profiler.start()

        if self.debug:
            print("> Start python profiler")
        self.python_profiler.start()

    def _end(self):
        self.python_profiler.stop()
        if self.debug:
            print("> Stop python profiler")

        self.cuda_profiler.stop()
        if self.debug:
            print("> Stop CUDA profiler")

    def _should_measure_call(self, bench_name=NO_BENCH_NAME):
        # return self.start_measuring_call is None or self.bench_name == bench_name
        return self.bench_name == bench_name and (
                self.start_measuring_call is None or \
                # (self.code_count[bench_name] - 1) >= self.start_measuring_call
                self.steps + 1 >= self.start_measuring_call
                # (self.code_count[bench_name] - 1) >= self.start_measuring_call
        )

    def enable_profiling(self, bench_name=NO_BENCH_NAME):
        if not self._should_measure_call(bench_name):
            return

        self.start_t[bench_name] = time.time()
        self._start()

    def profile_time_sec(self, bench_name):
        return self.time_sec[bench_name]

    def disable_profiling(self, bench_name=NO_BENCH_NAME, num_calls=1):
        if not self._should_measure_call(bench_name):
            self.code_count[bench_name] += 1
            return

        self._end()
        end_time_sec = time.time()
        self.end_t[bench_name] = end_time_sec
        self.time_sec[bench_name] += end_time_sec - self.start_t[bench_name]
        self.code_count[bench_name] += num_calls

    def profile(self, bench_name, func, *args, **kwargs):
        """
        PSEUDOCODE:

        if idempotent:
          # func(...) is idempotent, so just run all the iterations
          # at once to reduce total runtime.

          for i in range(warmup):
            func(...)
          start_t = time.time()
          profiler.start()
          for i in range(iterations):
            func(...)
          profiler.stop()

        else:
          # func(...) is not idempotent, so we must collect iterations
          # one-at-a-time.

          func(...)

        :param self:
        :param func:
        :param args:
        :param kwargs:
        :return:
        """
        should_measure = self._should_measure_call(bench_name)
        if should_measure and not self.no_idempotent and self.is_idempotent(bench_name):
            # idempotent.
            # for i in range(self.start_measuring_call):
            #     func(*args, **kwargs)

            if hasattr(func, 'init'):
                # Do any kind of initialization needed before profiling
                func.init(*args, **kwargs)

            if self.num_calls is None:
                # Dynamically decide # of iterations to run, such that time to
                # run bench_name experiment is <= 10 seconds.
                self._init_num_calls(bench_name, func, *args, **kwargs)
                assert self.num_calls is not None

            self.enable_profiling(bench_name)
            for i in range(self.num_calls):
                ret = func(*args, **kwargs)
            self.disable_profiling(bench_name, num_calls=self.num_calls)

            if hasattr(func, 'reset'):
                # Cleanup anything we did specific to profiling so we can resume
                # running the training loop.
                func.reset(*args, **kwargs)

            self._maybe_finish()

        else:
            # not idempotent.
            self.enable_profiling(bench_name)
            ret = func(*args, **kwargs)
            self.disable_profiling(bench_name)
        return ret

    def finish(self):
        ProfileGlobals.files_after = ls_files(self.directory)

        # Put this here to test that cleanup_files doesn't delete nvprof/pyprof files
        self.dump()
        ProfileGlobals.cleanup_files()

    def dump(self, config_kwargs=dict()):

        # Q: Should we be calling this again...?  We'd like to update num_calls if it was computed dynamically...
        config_path = _j(self.directory, "config{bench}.json".format(
            bench=bench_suffix(self.bench_name)))
        dump_config(config_path,
                    num_calls=self.num_calls,
                    start_measuring_call=self.start_measuring_call,
                    **config_kwargs)

        self.cuda_profiler.dump()
        self.python_profiler.dump()

    def _maybe_finish(self):
        if self.done_measuring():
            self.finish()
        if self.should_stop():
            print("> IML: Stopping training early now that profiler is done")
            sys.exit(0)

    def next_step(self):
        self._maybe_finish()
        self.steps += 1

    def done_measuring(self):
        return self.num_calls is not None and \
               self.code_count[self.bench_name] >= self.total_calls_to_run

    @property
    def total_calls_to_run(self):
        if self.start_measuring_call is None:
            return self.num_calls
        return self.num_calls + self.start_measuring_call

    def should_stop(self):
        # If your using the profiler this way, you need to provide self.num_calls!
        # assert self.num_calls is not None
        # Can only measure one bench_name at a time.
        return self.exit_early and \
               self.done_measuring()
        # return all(self.code_count[bench_name] >= self.num_calls \
        #            for bench_name in self.bench_names)

class CUDAProfiler:
    def __init__(self):
        # NOTE: CUDA profiling output has already been specified when this script was launched.
        # self.profile_basename = profile_basename
        self.already_enabled = False

    def start(self):
        # NOTE: we assume the CUDA
        self.already_enabled = cudaprofile.is_profiler_enabled()
        if not self.already_enabled:
            cudaprofile.start()

    def stop(self):
        if not self.already_enabled:
            cudaprofile.stop()

    def dump(self):
        # Dumping is performed externally by nvprof once the program terminates.
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class PythonProfiler:
    """
    Run profiler on python code block.

    with profiler(python_profile_basename="some/profile"):
            # profile this code
    """
    def __init__(self, directory, bench_name=NO_BENCH_NAME, record_call_times=True, clock='monotonic_clock'):
        self.profile = cProfile.Profile()
        self.directory = directory
        self.bench_name = bench_name
        self.clock = clock
        assert self.clock in ['monotonic_clock']
        self.use_cycle_counter = (self.clock == 'cycle_counter')
        self.use_monotonic_clock = (self.clock == 'monotonic_clock')
        self.record_call_times = record_call_times

        assert not ( self.use_cycle_counter and self.use_monotonic_clock )

        if self.use_cycle_counter:
            self.profile.make_use_cycle_counter()

        if self.use_monotonic_clock:
            self.profile.make_use_monotonic_clock()

        if self.record_call_times:
            self.profile.make_record_call_times()

    def __enter__(self):
        self.start()

    def start(self):
        self.profile.enable()

    def stop(self):
        self.profile.disable()
        # self.dump()

    def dump(self):
        # sortby = ('calls', 'filename', 'name', 'line')
        sortby = ('tottime', 'filename', 'line', 'name')

        os.makedirs(os.path.dirname(self._stats_path), exist_ok=True)
        with open(self._stats_path, mode='w') as f:
            ps = pstats.Stats(self.profile, stream=f).sort_stats(*sortby)
            if self.record_call_times:
                call_times = ps.call_times
            ps.print_stats()

        if self.record_call_times:

            # Tuple keys are not OK; convert to strings.
            new_call_times = dict()
            for func_tuple, times in call_times.items():
                func = func_std_string(func_tuple)
                new_call_times[func] = times
            json.dump(new_call_times,
                                codecs.open(self._call_times_path, mode='w', encoding='utf-8'),
                                sort_keys=True, indent=4)

        os.makedirs(os.path.dirname(self._prof_path), exist_ok=True)
        ps.dump_stats(self._prof_path)

    @property
    def _prof_path(self):
        ret = _j(self.directory, "python_profile{bench}.prof".format(
            bench=bench_suffix(self.bench_name)))
        return ret

    @property
    def _call_times_path(self):
        return _j(self.directory, "python_profile{bench}.call_times.json".format(
            bench=bench_suffix(self.bench_name)))

    @property
    def _stats_path(self):
        return _j(self.directory, "python_profile{bench}.txt".format(
            bench=bench_suffix(self.bench_name)))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# Taken from cProfile's Lib/pstats.py;
# func_name is 3-tuple of (file-path, line#, function_name)
# e.g.
#     ('Lib/test/my_test_profile.py', 259, '__getattr__'):
def func_std_string(func_tuple): # match what old profile produced
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

def add_iml_arguments(parser):
    parser.add_argument('--iml-nvprof-enabled', action='store_true', help=textwrap.dedent("""
        IML: is nvprof running?
        
        Internal use only; 
        used to determine whether this python script has been invoked using nvprof.
        If it hasn't, the script will re-invoke itself with nvprof.
    """))
    parser.add_argument('--iml-no-idempotent', action='store_true', help=textwrap.dedent("""
        IML: don't measure bench_name's iterations all-at-once; measure 1 iteration at each loop-step only.
    """))
    parser.add_argument('--iml-num-calls', type=int, help="IML: how many calls should be measured?")
    parser.add_argument('--iml-start-measuring-call', type=int, help="IML: when should measuring begin?")
    parser.add_argument('--iml-bench-name',
                        default=NO_BENCH_NAME,
                        help=textwrap.dedent("""
    IML: which code block should we measure?
    i.e. --iml-bench-name=some_bench
        # Just measure "some_bench", nothing else.
        profiler.profile('some_bench', do_some_bench)
    """))
    parser.add_argument('--iml-bench-names',
                        default=[NO_BENCH_NAME],
                        help=textwrap.dedent("""
    IML: which code blocks should we measure?
    i.e. --iml-bench-names bench1 bench3
        profiler.profile('bench1', do_bench1) # Measure this
        profiler.profile('bench2', do_bench2) # SKIP
        profiler.profile('bench3', do_bench3) # Measure this
    """), nargs='+')

# Match input/output to PythonProfilerParser
PYPROF_REGEX = r'(?:python_profile.*|microbenchmark\.json|config.*\.json)'
# Match input/output to CUDASQLiteParser
NVPROF_REGEX = r'(?:nvidia.*\.nvprof|microbenchmark\.json|config.*\.json|nvidia.*\.pretty\.txt)'
def is_iml_file(path):
    base = _b(path)
    return re.search(r'{pyprof}|{nvprof}'.format(
        pyprof=PYPROF_REGEX,
        nvprof=NVPROF_REGEX),
        base)

class _ProfileGlobals:
    def __init__(self):
        self.files_before = None
        self.files_after = None

    def cleanup_files(self):
        """
        PROBLEM: script might output result files, and not be written to handle
        re-running itself once those files exist.

        - Modify script to overwrite/delete old output files:
          Can probably handle this with IML wrapper.
          files_before = [ files seen before run ]
          files_after  = [ files seen after run ]
          iml_files    = [ files output by iml ]
          files_to_rm  = files_after - files_before - iml_files
        """
        # self.iml_files = [path for path in self.files_after if is_iml_file(path)]
        self.files_to_rm = set(self.files_after).difference(set(self.files_before))
        self.files_to_rm = [path for path in self.files_to_rm if not is_iml_file(path)]
        for path in self.files_to_rm:
            opts = ""
            if os.path.isdir(path):
                opts = "-r "
            print("> RM {opts}{f}".format(
                opts=opts, f=path))

ProfileGlobals = _ProfileGlobals()

def handle_iml_args(output_directory, parser, args, no_bench_name=False):
    ProfileGlobals.files_before = ls_files(output_directory)

    if args.iml_bench_names is None:
        if no_bench_name:
            args.iml_bench_names = [NO_BENCH_NAME]
        else:
            parser.error("--iml-bench-names must contain the names of all the code-blocks this ML script contains.")

    if not args.iml_nvprof_enabled:
        if args.iml_bench_name is not None:
            # User wants to run specific bench_name.
            run_with_nvprof(output_directory, parser, args,
                            bench_name=args.iml_bench_name)
        else:
            # Run each bench_name.
            for bench_name in args.iml_bench_names:
                run_with_nvprof(output_directory, parser, args,
                                bench_name=bench_name)
        sys.exit(0)

def run_with_nvprof(directory, parser, args,
                    bench_name=NO_BENCH_NAME):
    print("> Reinvoking script with nvprof; bench_name={b}".format(
        b=bench_name))

    nvprof_logfile = _j(directory, "nvidia{bench}.nvprof_logfile.txt".format(
        bench=bench_suffix(bench_name)))
    if _e(nvprof_logfile):
        os.remove(nvprof_logfile)
    nvprof_sqlite_file = _j(directory, "nvidia{bench}.nvprof".format(
        bench=bench_suffix(bench_name)))
    if _e(nvprof_sqlite_file):
        # Nvprof fails if the output file already exists.
        os.remove(nvprof_sqlite_file)
    os.makedirs(_d(nvprof_logfile), exist_ok=True)
    nvprof_args = ["nvprof",
                   "-o", nvprof_sqlite_file,
                   "--log-file", nvprof_logfile,
                   "--profile-from-start", "off"]
    cmdline = args_to_cmdline(parser, args)
    argv_exec = nvprof_args + cmdline + [
        "--iml-nvprof-enabled",
    ]
    if bench_name != NO_BENCH_NAME:
        argv_exec.extend(["--iml-bench-name", bench_name])

    print_cmd(argv_exec)
    subprocess.run(argv_exec, stdout=sys.stdout, stderr=sys.stderr, check=True)

def args_to_cmdline(parser, args):
    """
    # To convert args namespace into cmdline:

    if args.option == True and option in parser:
            cmdline.append(--option)
    elif args.option == False and option in parser:
            pass
    elif type(args.option) in [int, str, float]:
            cmdline.append(--option value)
    elif type(args.open) in [list]:
            cmdline.append(--option elem[0] ... elem[n])
    else:
            raise NotImplementedError
    """

    def option_in_parser(parser, option):
        return parser.get_default(option) is not None

    def optname(option):
        return "--{s}".format(s=re.sub(r'_', '-', option))

    py_script_idx = 0
    while not re.search(r'\.py$', sys.argv[py_script_idx]):
        py_script_idx += 1
    extra_opts = []
    if hasattr(args, 'debug') and args.debug:
        extra_opts.extend(["-m", "ipdb"])
    cmdline = [sys.executable] + extra_opts + sys.argv[0:py_script_idx+1]
    for option, value in args.__dict__.items():
        opt = optname(option)
        if value is None:
            continue

        if type(value) == bool:
            if value and option_in_parser(parser, option):
                cmdline.extend([opt])
            else:
                pass
        elif type(value) in [int, str, float]:
            cmdline.extend([opt, value])
        elif type(value) in [list] and len(value) > 0:
            cmdline.extend([opt])
            cmdline.extend(value)
        else:
            raise NotImplemented

    return [str(x) for x in cmdline]

# If you search for function names matching this pattern in pyprof output, they will match TensorFlow C++ API calls.
CLIB_TENSORFLOW_REGEX = r'(?:built-in.*pywrap_tensorflow)'
# We can manually wrap a c-library in order to record C API call times.  See test_call_c.py for how to do this.
CLIB_WRAPPER_REGEX = r'CLIB__.*'
def dump_config(path, **kwargs):
    config = dict()

    avail_gpus = get_available_gpus()
    avail_cpus = get_available_cpus()
    # We want to be CERTAIN about which device TensorFlow is using.
    # If no GPUs are available, TF will use the CPU.
    # If a GPU is available, make sure only 1 is available so we are certain it's using that one.
    if not( (len(avail_gpus) == 1) or
            (len(avail_gpus) == 0 and len(avail_cpus) == 1) ):
        CUDA_VISIBLE_DEVICES = ENV.get('CUDA_VISIBLE_DEVICES', None)
        print(textwrap.dedent("""
        ERROR: Multiple GPUs were found; IML benchmark requires only one GPU to be visible to TensorFlow via (for example) "export CUDA_VISIBLE_DEVICES=0".
        Use one of the below available GPUs:
        """))
        pprint.pprint({
            'avail_gpus':avail_gpus,
            'avail_cpus':avail_cpus,
            'CUDA_VISIBLE_DEVICES':CUDA_VISIBLE_DEVICES,
        }, indent=2)
        sys.exit(1)
    if len(avail_gpus) == 1:
        device_dict = avail_gpus[0]
    else:
        device_dict = avail_cpus[0]

    config.update(device_dict)

    c_lib_func_pyprof_pattern = kwargs.get('c_lib_func_pyprof_pattern', CLIB_WRAPPER_REGEX)
    defaults = {
        'clock': "monotonic_clock",
        'device_name': None,
        'impl_name': None,
        # For tensorflow: r'(?:built-in.*pywrap_tensorflow)'
        'c_lib_func_pyprof_pattern':c_lib_func_pyprof_pattern,
        # Discard the first nvprof sample since it's typically 700ms
        # (not sure why, presumably some initialization time).
        'discard_first_sample':True,

        # 'bench_name_labels': {
        #     ...
        # },
    }
    config = dict(defaults)
    config.update(kwargs)
    assert 'num_calls' in config or (
        'iterations' in config and \
        'repetitions' in config
    )
    dump_json(config, path)

def print_cmd(cmd):
    print(textwrap.dedent("""
    RUN:
        cwd = {cwd}
        cmd = {cmd}
    """.format(
        cwd=os.getcwd(),
        cmd=" ".join([str(x) for x in cmd]),
    )))

def load_json(path):
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def dump_json(data, path):
    os.makedirs(_d(path), exist_ok=True)
    json.dump(data,
              codecs.open(path, mode='w', encoding='utf-8'),
              sort_keys=True, indent=4,
              skipkeys=False)

def bench_suffix(bench):
    if bench != NO_BENCH_NAME:
        return ".{bench}".format(bench=bench)
    return ""

def list_files(direc):
    def _list_files(direc):
        def _path(path):
            return _j(direc, path)
        return [_path(path) for path in os.listdir(direc)]

    if type(direc) == list:
        all_files = []
        for d in direc:
            all_files.extend(_list_files(d))
        return all_files

    return _list_files(direc)

def get_available_cpus():
    local_device_protos = tf_device_lib.list_local_devices()
    device_protos = [x for x in local_device_protos if x.device_type != 'GPU']
    assert len(device_protos) == 1
    cpu = cpuinfo.get_cpu_info()
    # 'brand': 'Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz',
    device_dict = {
        'device_name':cpu['brand'],
        'device_number':0,
    }
    return [device_dict]
    # device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    # return device_dicts

def get_available_gpus():
    local_device_protos = tf_device_lib.list_local_devices()
    device_protos = [x for x in local_device_protos if x.device_type == 'GPU']
    device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    return device_dicts

def _device_proto_as_dict(device_proto):
    # For GPU's
    # ipdb> device_proto.physical_device_desc
    # 'device: 0, name: Quadro P4000, pci bus id: 0000:04:00.0, compute capability: 6.1'

    # For CPU's
    # ipdb> device_proto
    # name: "/device:CPU:0"
    # device_type: "CPU"
    # memory_limit: 268435456
    # locality {
    # }
    # incarnation: 11976653475402273625

    m = re.search(r'device: (?P<device>\d+), name: (?P<name>.*), pci bus id: (?P<pci_bus_id>[^,]+), compute capability: (?P<compute_capability>.*)',
                  device_proto.physical_device_desc)
    device = int(m.group('device'))
    name = m.group('name')
    return {"device_number":device, "device_name":name}

def ls_files(directory):
    """
    Same as list_files but allow directory to not exist.
    """
    if not os.path.isdir(directory):
        return []
    return list_files(directory)
