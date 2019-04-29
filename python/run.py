#!/usr/bin/env python3
import argparse
import textwrap
import json
import codecs
import re
# pip install py-cpuinfo
from io import StringIO
import pprint
import os
import subprocess
import sys
from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, isdir

from scripts import run_benchmark

MEASURE_DEBUG_OPTS = ["--repetitions", 2]

def reverse_dict(dic):
    r_dic = dict()
    for k, v in dic.items():
        r_dic[v] = k
    return r_dic

# $ROOT/python
SCRIPT_DIR = _d(_a(__file__))
# $ROOT
ROOT = _d(SCRIPT_DIR)
BENCH_PY = _j(ROOT, "python/profiler/run_benchmark.py")
CHECKPOINTS_PONG = _j(ROOT, "checkpoints/PongNoFrameskip-v4")
GPU_DATA = _j(ROOT, "gpu_data.json")

device_to_pretty = {
    'quadro_p4000':'Quadro P4000',
    'quadro_k4000':'Quadro K4000',
    'gtx_1080':'GTX 1080',
}
pretty_to_device = reverse_dict(device_to_pretty)

class Run:
    def __init__(self, args, argv, parser):
        self.args = args
        self.argv = argv
        self.parser = parser

        self._func_to_method = {
            'process': self.process,
            'measure': self.measure,
            'plot': self.plot,
            'scons': self.scons,
        }

        self._init()

    def _init(self):
        if self.args.gpu_data is not None:
            self.load_gpu_data()

    def load_gpu_data(self):
        self.gpu_data = load_json(self.args.gpu_data)

    def generate_gpu_data(self):
        self.gpu_data = dict()

        self.gpu_data['gpus'] = get_available_gpus()
        self.gpu_data['device_name_to_number'] = dict()
        self.gpu_data['devices'] = []
        for gpu in self.gpu_data['gpus']:
            device_name = pretty_to_device[gpu['device_name']]
            self.gpu_data['device_name_to_number'][device_name] = gpu['device_number']
            self.gpu_data['devices'].append(device_name)
        # cpus = run_benchmark.get_available_cpus()

        do_dump_json(self.gpu_data, GPU_DATA)
    @property
    def device_name_to_number(self):
        return self.gpu_data['device_name_to_number']
    @property
    def devices(self):
        return self.gpu_data['devices']
    @property
    def gpus(self):
        return self.gpu_data['gpus']

    def reinvoke_with(self, new_args, remove=[], debug=None, check=False):
        pyfile = sys.argv[0]
        assert is_pyfile(pyfile)
        new_cmd = self._python_cmd(pyfile, debug=debug) + new_args + sys.argv[1:]
        new_cmd = [arg for arg in new_cmd if arg not in remove]

        print("> Reinvoke with cmd: \n  {cmd}".format(cmd=" ".join(new_cmd)))
        proc = subprocess.run(new_cmd, stderr=sys.stderr, stdout=sys.stdout, check=check)
        return proc

    def _get_subdir(self, device):
        # <debug/ if --measure-debug>/<--extra-subdir>/<--subdir>/<device>
        args = self.args
        components = []
        if args.measure_debug:
            components.append("debug")
        if args.extra_subdir is not None:
            components.append(args.extra_subdir)
        if args.subdir is not None:
            components.append(args.subdir)
        components.append(device)

        return _j(*components)

    def process(self):
        args = self.args
        argv = self.argv
        parser = self.parser

        extra_opts = []
        # if args.measure_debug:
        #     extra_opts.extend(MEASURE_DEBUG_OPTS)

        def get_directory(device):
            subdir = self._get_subdir(device)
            subdir_full_path = self._get_dir(subdir)
            directory = subdir_full_path
            return directory

        def get_cmd(rule, directories):
            cmd = self._python_cmd() + ["--directories"] + directories + ["--rule", rule] + extra_opts + argv
            return cmd

        BUILD_FAILED = 'build failed'
        BUILD_SUCCESS = 'build success'
        BUILD_SKIP = 'build skipped'
        def build(ParserKlass, directories, rebuild):
            for direc in directories:
                if not isdir(direc):
                    print("ERROR: Couldn't find directory={d} when attempting to build parser={parser}".format(
                        d=direc,
                        parser=ParserKlass.__name__,
                    ))
                    return BUILD_FAILED
            cmd = get_cmd(ParserKlass.__name__, directories)
            src_files = ParserKlass.get_source_files(directories)

            src_files.check_has_all_required_paths(ParserKlass)
            if not src_files.has_all_required_paths:
                # Q: Should we just skip this one?
                return BUILD_FAILED

            targets = src_files.all_targets(ParserKlass)

            existing_targets = [t for t in targets if _e(t)]
            if len(existing_targets) > 0 and not rebuild:
                print("> Skip rule={rule}; some targets already exist:".format(
                    rule=ParserKlass.__name__,
                    # trgs=existing_targets,
                ))
                pprint.pprint(as_short_list(existing_targets), indent=2)
                return BUILD_SKIP

            env, added_env = self._cuda_env()
            self.do_cmd(cmd, env=env, added_env=added_env)
            return BUILD_SUCCESS

        def should_rebuild(ParserKlass=None, device=None):
            if args.rebuild:
                return True
            if ParserKlass is not None and ParserKlass.__name__ in args.rebuild_rule:
                return True
            if device is not None and device in args.rebuild_device:
                return True
            return False

        def handle_build_ret(ret):
            if ret == BUILD_FAILED:
                sys.exit(1)

        if args.directory is not None:
            """
            Specify a specific directory to run from.
            (not --extra-subdir, e.g. --directory checkpoints/test_call_c)
            """
            for ParserKlass in run_benchmark.PARSER_KLASSES:
                rebuild = should_rebuild(ParserKlass)
                ret = build(ParserKlass, [args.directory], rebuild)
                handle_build_ret(ret)
        else:
            all_device_dirs = [get_directory(device) for device in args.devices]
            for ParserKlass in run_benchmark.PARSER_KLASSES:
                if ParserKlass.uses_multiple_dirs():
                    """
                    TimeBreakdownPlot takes multiple directories as input.
                    """
                    rebuild = should_rebuild(ParserKlass)
                    ret = build(ParserKlass, all_device_dirs, rebuild)
                    handle_build_ret(ret)
                else:
                    """
                    GPUTimeSec.summary.png takes a single directory.
                    """
                    for device in args.devices:
                        directory = get_directory(device)
                        if not _d(directory):
                            print("> Skip processing directory={d}; doesn't exist".format(d=directory))
                            continue
                        rebuild = should_rebuild(ParserKlass, device)
                        ret = build(ParserKlass, [directory], rebuild)
                        handle_build_ret(ret)

    def _get_dir(self, subdir):
        return _j(CHECKPOINTS_PONG, subdir, "microbenchmark")

    def measure(self):
        """
        RUN:
          cwd: /home/james/clone/baselines
          cmd: python3 -u run_benchmark.py
                 --repetitions 2 --subdir debug/glue/gpu/quadro_k4000 --no-profile-cuda
        """
        args = self.args
        argv = self.argv
        parser = self.parser

        extra_opts = []
        if args.measure_debug:
            extra_opts.extend(MEASURE_DEBUG_OPTS)

        def get_cmd(device):
            cmd = self._python_cmd() + ["--subdir", self._get_subdir(device)] + extra_opts + argv
            return cmd

        for device in args.devices:
            cmd = get_cmd(device)
            env, added_env = self._cuda_env(device=device)
            self.do_cmd(cmd, env=env, added_env=added_env)

    def _cuda_env(self, device=None):
        """
        env['CUDA_VISIBLE_DEVICES'] = ...
            Make it so TensorFlow will definitely only use 1 GPU.

        env['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = ...
            We need this, otherwise TensorFlow will ignore the Quadro K4000 GPU.

        :param env:
        :param device:
        :return:
        """
        env = dict(os.environ)
        added_env = dict()

        def _add(k, v):
            env[k] = str(v)
            added_env[k] = str(v)

        if device is not None:
            _add('CUDA_VISIBLE_DEVICES', self.device_name_to_number[device])

        # We need this, otherwise TensorFlow will ignore the Quadro K4000 GPU with the following message:
        #
        #   2018-10-31 10:35:54.318150: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1469] Ignoring visible gpu device (device: 1, name: Quadro K4000,
        #   pci bus id: 0000:08:00.0, compute capability: 3.0) with Cuda multiprocessor count: 4.
        #   The minimum required count is 8. You can adjust this requirement
        #   with the env var TF_MIN_GPU_MULTIPROCESSOR_COUNT.
        #
        _add('TF_MIN_GPU_MULTIPROCESSOR_COUNT', 4)

        return env, added_env

    def _cmd_str(self, cmd):
        return [str(x) for x in cmd]

    def _python_cmd(self, pyfile=None, debug=None):
        args = self.args
        if pyfile is None:
            pyfile = BENCH_PY
        if debug is None:
            debug = args.debug
        if not debug:
            cmd = ["python3", "-u", pyfile]
        else:
            cmd = ["python3", "-m", "ipdb", pyfile]
        return cmd

    def _run_cmd(self, cmd, env=None, skip_error=False):
        args = self.args
        if args.dry_run:
            return None
        # capture_output=True,
        # proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        proc = check_call(cmd, cat_output=True, print_cmd=False, env=env)
        # return proc
        # proc = subprocess.run(cmd, stderr=sys.stderr, stdout=sys.stdout, env=env)
        # if not skip_error:
        #     proc.check_returncode()
        return proc

    def _print_cmd(self, cmd, added_env=None):
        if added_env is not None:
            ss = StringIO()
            pprint.pprint(added_env, stream=ss, indent=2)
            env_str = ss.getvalue()
        else:
            env_str = "{}"
        print(textwrap.dedent("""
            RUN:
              cwd: {cwd}
              cmd: {cmd}
              env: {env}
            """.format(
            cwd=os.getcwd(),
            cmd=" ".join(cmd),
            env=env_str,
        )))

    def scons(self):
        args = self.args
        argv = self.argv
        parser = self.parser

        args.until_complete = args.rule is None and not args.list_targets and not args.list_sources

        def get_cmd():
            # ["--plot"] +
            # ["--directory", directory] +
            # ["--directories"] + directories +
            extra_opt = []
            if args.debug:
                extra_opt.extend(['--print'])
            if args.rule is not None:
                extra_opt.extend(['--rule', args.rule])
            if args.list_targets:
                extra_opt.extend(['--list-targets'])
            if args.list_sources:
                extra_opt.extend(['--list-sources'])
            # if args.until_complete:
            #     extra_opt.extend(['--until-complete'])
            cmd = ["scons"] + \
                  extra_opt + \
                  argv
            return cmd

        def run_scons():
            cmd = get_cmd()
            proc = self.do_cmd(cmd, skip_error=True)
            if proc.returncode != 0:
                sys.exit(proc.returncode)
            return proc

        def scons_complete(proc):
            if type(proc.stdout) == bytes:
                out = proc.stdout.decode('utf-8')
            else:
                out = proc.stdout
            return re.search(r'> SCONS: NO MORE TARGETS', out)

        if args.until_complete:
            while True:
                proc = run_scons()
                if scons_complete(proc):
                    break
        else:
            run_scons()

    def plot(self):
        """
        RUN:
          cwd: /home/james/clone/baselines
          cmd: python3 -u run_benchmark.py --plot
                --directory ...
                --directories ...

        """
        args = self.args
        argv = self.argv
        parser = self.parser

        directories = []
        for device in args.devices:
            directories.append(_j(CHECKPOINTS_PONG, self._get_subdir(device)))
        directories = [_a(d) for d in directories]

        # Output plot to path that is a common prefix of all these directories.
        directory = self._common_dir_prefix(directories)

        def get_cmd():
            cmd = self._python_cmd() + \
                  ["--plot"] + \
                  ["--directory", directory] + \
                  ["--directories"] + directories + \
                  argv
            return cmd

        cmd = get_cmd()
        self.do_cmd(cmd)

    def _common_dir_prefix(self, directories):
        prefix = os.path.commonprefix(directories)
        if os.path.isdir(prefix):
            direc = prefix
        else:
            direc = _d(prefix)

        # assert os.path.isdir(direc)
        return direc

    def do_cmd(self, cmd, env=None, added_env=None, **kwargs):
        if env is None:
            env, added_env = self._cuda_env()
        cmd = self._cmd_str(cmd)
        _print_cmd(cmd, added_env=added_env)
        return self._run_cmd(cmd, env=env, **kwargs)

    @property
    def verbose(self):
        return self.args.verbose

    def run(self):
        args = self.args
        argv = self.argv
        parser = self.parser

        if args.devices == []:
            args.devices = list(self.devices)

        for device in args.devices:
            if device not in self.devices:
                parser.error("Couldn't find device={dev}".format(dev=device))

        if args.func not in self._func_to_method:
            parser.error("Unknown sub-command {cmd}".format(cmd=args.func))

        # if self.verbose:
        pprint.pprint({'args':args.__dict__})

        os.chdir(ROOT)

        func = self._func_to_method[args.func]
        func()

def main():
    parser = argparse.ArgumentParser("run stuff")
    parser.add_argument('--dry-run', help="Don't run commands; just print them",
                        action='store_true')
    parser.add_argument('--debug', help="pdb",
                        action='store_true')
    parser.add_argument('--gpu-data', help="(internal use) JSON file with gpu data read from tensorflow")
    parser.add_argument('--generate-gpu-data', help="(internal use)",
                        action='store_true')
    parser.add_argument('--skip-error', help="If a command fails, ignore it",
                        action='store_true')
    parser.add_argument('--devices',
                        nargs="+", default=[],
                        # action='append', default=[],
                        help=textwrap.dedent("""
                        Devices to run with.
                        """))
    parser.add_argument('--verbose', help="print extra info",
                              action='store_true')

    def _add_plot_and_measure_and_process_args(parser):
        parser.add_argument('--measure-debug', help="Run quickly (few repetitions)",
                            action='store_true')
        parser.add_argument('--subdir',
                            help="<debug/ if --measure-debug>/<--extra-subdir>/<--subdir>/<device>")
        parser.add_argument('--extra-subdir',
                            help="(see --subdir)")
        # parser.add_argument('--devices', help="Devices to measure",
        #                             default=["quadro_k4000", "quadro_p4000"])
        # parser.add_argument('--plot', help="Just plot results, don't run any benchmarks.",
        #                     action='store_true')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_measure = subparsers.add_parser('measure', help='run experiments / measure')
    _add_plot_and_measure_and_process_args(parser_measure)
    parser_measure.set_defaults(func='measure')

    parser_plot = subparsers.add_parser('plot', help='plot experiments')
    _add_plot_and_measure_and_process_args(parser_plot)
    parser_plot.set_defaults(func='plot')

    parser_process = subparsers.add_parser('process', help='process experiments')
    _add_plot_and_measure_and_process_args(parser_process)
    parser_process.add_argument('--directory',
                                help="directory to process (default: see --subdir)")
    parser_process.add_argument('--rebuild',
                                help="Rebuild even if target files exist",
                                action='store_true')
    parser_process.add_argument('--rebuild-device',
                                help="Rebuild even if target files exist",
                                nargs="+", default=[])
    parser_process.add_argument('--rebuild-rule',
                                help="Rebuild even if target files exist",
                                nargs="+", default=[])
    parser_process.set_defaults(func='process')

    parser_scons = subparsers.add_parser('scons', help='process experiments using scons SConstruct file.')
    parser_scons.add_argument(
        '--list-targets',
        help=textwrap.dedent("""
            List targets.
        """),
        action='store_true')
    parser_scons.add_argument(
        '--list-sources',
        help=textwrap.dedent("""
            List sources.
        """),
        action='store_true')
    parser_scons.add_argument(
        '--rule',
        help=textwrap.dedent("""
            What to build?
        """),
        )
    parser_scons.set_defaults(func='scons')

    args, argv = parser.parse_known_args()

    # Initial cmd runs
    # - --gpu-data is unset
    # - Invoke new cmd with --generate-data set
    # -

    run_obj = Run(args, argv, parser)

    if not args.generate_gpu_data and args.gpu_data is None:
        remove = ["--debug"]
        run_obj.reinvoke_with(["--generate-gpu-data"], remove=remove, debug=False, check=True)
        proc = run_obj.reinvoke_with(["--gpu-data", GPU_DATA], debug=False, check=False)
        sys.exit(proc.returncode)

    if args.generate_gpu_data and args.gpu_data is None:
        run_obj.generate_gpu_data()
        return

    assert not args.generate_gpu_data
    assert args.gpu_data is not None
    run_obj.run()

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

# def get_available_cpus():
#     import tensorflow as tf
#     from tensorflow.python.client import device_lib as tf_device_lib
#     local_device_protos = tf_device_lib.list_local_devices()
#     device_protos = [x for x in local_device_protos if x.device_type != 'GPU']
#     assert len(device_protos) == 1
#     cpu = cpuinfo.get_cpu_info()
#     # 'brand': 'Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz',
#     device_dict = {
#         'device_name':cpu['brand'],
#         'device_number':0,
#     }
#     return [device_dict]
#     # device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
#     # return device_dicts

# def get_available_gpus():
#     import tensorflow as tf
#     from tensorflow.python.client import device_lib as tf_device_lib
#     local_device_protos = tf_device_lib.list_local_devices()
#     device_protos = [x for x in local_device_protos if x.device_type == 'GPU']
#     device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
#     return device_dicts

def load_json(path):
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        data = fixup_json(data)
        return data

def do_dump_json(data, path):
    os.makedirs(_d(path), exist_ok=True)
    json.dump(data,
              codecs.open(path, mode='w', encoding='utf-8'),
              sort_keys=True, indent=4,
              skipkeys=False)

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

def is_pyfile(path):
    return re.search('.py$', path)

def check_call(cmd, cat_output=True, print_cmd=False, **kwargs):
    """
    :param cmd:
    :param cat_output:
    If True, cat output to
    :param kwargs:
    :return:
    """
    if print_cmd:
        _print_cmd(cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    # for c in iter(lambda: process.stdout.read(1), ''):    # replace '' with b'' for Python 3
    ss = StringIO()
    for line in iter(process.stdout.readline, b''):    # replace '' with b'' for Python 3
        if line == b'':
            break
        line = line.decode('utf-8')
        if cat_output:
            sys.stdout.write(line)
        ss.write(line)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    proc = subprocess.CompletedProcess(cmd, process.returncode, stdout=ss.getvalue())
    return proc
    # return subprocess.run(cmd, stdout=sys.stdout, stder=sys.stderr, check=True, **kwargs)

def _print_cmd(cmd, added_env=None):
    if added_env is not None:
        ss = StringIO()
        pprint.pprint(added_env, stream=ss, indent=2)
        env_str = ss.getvalue()
    else:
        env_str = "{}"
    print(textwrap.dedent("""
        RUN:
          cwd: {cwd}
          cmd: {cmd}
          env: {env}
        """.format(
        cwd=os.getcwd(),
        cmd=" ".join(cmd),
        env=env_str,
    )))

def as_short_list(xs):
    if len(xs) > 3:
        return xs[0:3] + ["..."]
    return xs

if __name__ == '__main__':
    main()
