import cProfile, pstats, io
import codecs
import sys
import json
import subprocess
import textwrap
import os
import time
import re

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e

from profiler import cudaprofile

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
    """
    def __init__(self, python_profile_basename, debug=False):
        self.debug = debug
        self.python_profiler = PythonProfiler(python_profile_basename)
        self.cuda_profiler = CUDAProfiler()
        self._profile_start_sec = 0
        self.profile_time_sec = 0

    def enable_profiling(self):
        # if self.profile_cuda:
        # nvprof reports the average time over all the repetitions.
        # All the ncalls for the functions called should be divisible by 10 (=repetitions).
        if self.debug:
            print("    > Start CUDA profiler")
        self.cuda_profiler.start()

        # if self.profile_python:
        if self.debug:
            print("> Start python profiler")
        self.python_profiler.start()

        self._profile_start_sec = time.time()

    def disable_profiling(self):
        self._profile_end_sec = time.time()
        self.profile_time_sec += self._profile_end_sec - self._profile_start_sec

        # if self.profile_python:
        self.python_profiler.stop()
        if self.debug:
            print("> Stop python profiler")

        # if self.profile_cuda:
        self.cuda_profiler.stop()
        if self.debug:
            print("> Stop CUDA profiler")

    def dump(self):
        self.cuda_profiler.dump()
        self.python_profiler.dump()

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
    def __init__(self, python_profile_basename, record_call_times=True, clock='monotonic_clock'):
        self.profile = cProfile.Profile()
        self.python_profile_basename = python_profile_basename
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
        return "{base}.prof".format(base=self.python_profile_basename)

    @property
    def _call_times_path(self):
        return "{base}.call_times.json".format(base=self.python_profile_basename)

    @property
    def _stats_path(self):
        return "{base}.txt".format(base=self.python_profile_basename)

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

def run_with_nvprof(parser, args):
    print("> Reinvoking script with nvprof.")

    nvprof_logfile = _j(args.directory, "nvidia.nvprof_logfile.txt")
    if _e(nvprof_logfile):
        os.remove(nvprof_logfile)

    nvprof_sqlite_file = _j(args.directory, "nvidia.nvprof")
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
        "--nvprof-enabled",
        "--nvprof-logfile", nvprof_logfile
    ]

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

def dump_config(path, **kwargs):
    config = dict()
    defaults = {
        'clock': "monotonic_clock",
        'device_name': None,
        'impl_name': None,
        # For tensorflow: r'(?:built-in.*pywrap_tensorflow)'
        'c_lib_func_pyprof_pattern':"CLIB__.*",
        'discard_first_sample':False,
        'num_calls':kwargs['num_calls'],
        'iterations':kwargs['iterations'],
        'repetitions':kwargs['repetitions'],
        # 'bench_name_labels': {
        #     ...
        # },
    }
    config = dict(defaults)
    config.update(kwargs)
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

