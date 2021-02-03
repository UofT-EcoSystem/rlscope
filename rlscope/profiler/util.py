"""
Miscellaneous command-line utility functions for parsing arguments,
checking available CPUs/GPUs, and logging.
"""
import re
import sys
import traceback
import subprocess
import shutil
import shlex
import os
import textwrap
import pprint
from os import environ as ENV

import cpuinfo

from rlscope.profiler.rlscope_logging import logger

def args_to_cmdline(parser, args,
                    argv=None,
                    subparser=None,
                    subparser_argname=None,
                    keep_executable=True,
                    keep_py=False,
                    use_pdb=True,
                    ignore_argnames=None,
                    ignore_unhandled_types=False,
                    debug=False):
    """
    NOTE: This WON'T keep arguments from sys.argv that AREN'T captured by parser.

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

    if ignore_argnames is None:
        ignore_argnames = set()
    else:
        ignore_argnames = set(ignore_argnames)

    def option_in_parser(parser, option):
        return parser.get_default(option) is not None

    def optname(option):
        return "--{s}".format(s=re.sub(r'_', '-', option))

    def get_parser_argnames(parser, argv):
        parser_args, extra_argv = parser.parse_known_args(argv)
        parser_argnames = list(vars(parser_args).keys())
        return parser_argnames

    def is_py_file(path):
        return re.search(r'\.py$', path)
    def find_py_file_idx(argv):
        for i in range(len(argv)):
            if is_py_file(argv[i]):
                return i
        return None

    def _args_to_cmdline(parser, args,
                         argv=None,
                         # subparser=None,
                         # subparser_argname=None,

                         keep_executable=True,
                         keep_py=False,
                         # use_pdb=True,
                         # ignore_unhandled_types=False,

                         ignore_argnames=None,

                         ):

        if ignore_argnames is None:
            ignore_argnames = set()
        else:
            ignore_argnames = set(ignore_argnames)

        py_script_idx = find_py_file_idx(argv)
        if py_script_idx is None:
            # No python executable; looks more like this:
            #
            # ['rlscope-test',
            #      '--train-script',
            #      'run_baselines.sh',
            #      '--test-name',
            #      'PongNoFrameskip-v4/docker',
            #      '--rlscope-directory',
            #      '/home/james/clone/baselines/output']
            py_script_idx = 0

        extra_opts = []
        if use_pdb and hasattr(args, 'debug') and args.debug:
            extra_opts.extend(["-m", "ipdb"])
        cmdline = []
        if keep_executable:
            cmdline.append(sys.executable)
        cmdline.extend(extra_opts)
        if keep_executable or keep_py:
            # Include python script path
            cmdline.extend(argv[0:py_script_idx+1])
        else:
            # Don't include python script path
            cmdline.extend(argv[0:py_script_idx])
        for option, value in args.__dict__.items():
            if ignore_argnames is not None and option in ignore_argnames:
                continue

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
            elif not ignore_unhandled_types:
                raise NotImplementedError("args_to_cmdline: not sure how to add {opt}={val} with type={type}".format(
                    opt=opt,
                    val=value,
                    type=type(value),
                ))

        return [str(x) for x in cmdline]

    if subparser is not None:
        assert subparser_argname is not None
        assert hasattr(args, subparser_argname)

    if argv is None:
        argv = sys.argv

    if subparser is None:
        return _args_to_cmdline(parser, args, argv,
                                keep_py=keep_py,
                                keep_executable=keep_executable)

    # Handle argparse parser.add_subparsers(...).
    #
    # Python subparser syntax is like:
    # $ your/script.py --parent1 --parent2 subcommand --subparser1 --subparser2
    #
    # Where --parent1 and --parent2 are arguments specified on the PARENT parser.
    # --subparser1 and --subparser2 are arguments specific on one of the CHILD subparsers:
    #
    # Basically, argparse subcommands are annoying because:
    # 1) --parent1 and --parent2 MUST precede subcommand
    # 2) The argparse args Namespace object does NOT tell us whether
    #    arguments came from the subparser or the parent parser.
    #
    # So, the code below takes care of challenges 1) and 2)
    # by using the subparser object (from subparsers.add_parser)
    # to figure out what options it parsed.

    subparser_args, _ = subparser.parse_known_args()
    subparser_cmd_opts = _args_to_cmdline(
        subparser, subparser_args, argv,
        keep_py=False,
        keep_executable=False,
        ignore_argnames= \
            set([subparser_argname]) \
                .union(ignore_argnames))

    # Ignore arg-names that are present in subparser.
    # NOTE: we assume subparser and parser don't have any of the same --argname's...
    # (argparse allows this, but the behaviour is for the subparser to "override"
    # the parent parser argument)
    subparser_argnames = get_parser_argnames(subparser, argv)
    parser_cmd_opts = _args_to_cmdline(
        parser, args, argv,
        keep_py=keep_py,
        keep_executable=keep_executable,
        ignore_argnames= \
            set([subparser_argname]) \
                .union(set(subparser_argnames)) \
                .union(ignore_argnames))

    if debug:
        logger.info(pprint_msg({
            'subparser_cmd_opts': subparser_cmd_opts,
            'parser_cmd_opts': parser_cmd_opts,
        }))

    subcommand = getattr(args, subparser_argname)

    opts = parser_cmd_opts + [subcommand] + subparser_cmd_opts
    return opts


def run_with_pdb(args, func, handle_exception=True):
    try:
        return func()
    except Exception as e:
        if not args.pdb and not handle_exception:
            raise
        logger.error("Saw unhandled exception:\n{exception}".format(
            exception=textwrap.indent(traceback.format_exc(), prefix='  ').rstrip(),
        ))
        if args.pdb:
            logger.debug("> Entering pdb:")
            # Fails sometimes, not sure why.
            import pdb
            pdb.post_mortem()
            # raise
        sys.exit(1)


def gather_argv(argv, sep='--', ignore_opts=None):
    """

    $ rls-prof [options]         cmd_exec ...
               ---------         ------------
               rlscope_prof_argv     cmd_argv

    Split sys.argv into:
    - rlscope_prof_argv: Arguments that rls-prof should handle.
    - cmd_argv: Arguments that the profiled script should handle.

    :param argv:
        sys.argv
    :return:
    """
    rlscope_prof_argv = []
    i = 0
    def is_executable(opt):
        return shutil.which(opt) is not None
    has_dashes = any(opt == sep for opt in argv)
    while i < len(argv):

        if ignore_opts is not None and argv[i] in ignore_opts:
            pass
        elif has_dashes:
            if argv[i] == sep:
                i += 1
                break
        elif is_executable(argv[i]):
            break

        rlscope_prof_argv.append(argv[i])
        i += 1
    cmd_argv = argv[i:]
    return rlscope_prof_argv, cmd_argv

def error(msg, parser=None):
    if parser is not None:
        parser.print_usage()
    logger.error(msg)
    sys.exit(1)


GET_AVAILABLE_CPUS_CPU_INFO = None
def get_available_cpus():
    """
    Report a single [0..1] value representing current system-wide CPU utilization.

    psutil.cpu_percent() returns EXACTLY this.
    From psutil.cpu_percent docstring:
        "
        Return a float representing the current system-wide CPU
        utilization as a percentage.
        "

    NOTE: It's also possible to get INDIVIDUAL utilization for each CPU,
    if we choose to do that in the future.
    """
    global GET_AVAILABLE_CPUS_CPU_INFO
    if GET_AVAILABLE_CPUS_CPU_INFO is None:
        # Do this lazily to avoid long import times.
        GET_AVAILABLE_CPUS_CPU_INFO = cpuinfo.get_cpu_info()
    device_name = get_cpu_brand(GET_AVAILABLE_CPUS_CPU_INFO)
    return {
        'device_name':device_name,
        'device_number':0,
    }

def get_cpu_brand(cpu_info):
    if 'brand' in cpu_info:
        return cpu_info['brand']
    elif 'brand_raw' in cpu_info:
        return cpu_info['brand_raw']
    raise RuntimeError((
        "Cannot determine name of CPU from cpuinfo.get_cpu_info(); "
        "tried 'brand' and 'brand_raw', choices are {choices}").format(
        choices=cpu_info.keys()))

def get_available_gpus():
    # $ tensorflow_cuda9 git:(opt-tfprof) ✗ nvidia-smi -L
    # GPU 0: GeForce RTX 2070 (UUID: GPU-e9c6b1d8-2b80-fee2-b750-08c5adcaac3f)
    # GPU 1: Quadro K4000 (UUID: GPU-6a547b6a-ae88-2aac-feb9-ae6b7095baaf)
    proc = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = proc.stdout.decode('utf-8').splitlines()
    device_dicts = []
    for line in lines:
        # if re.search(r'^\s*$', line):
        #     continue
        m = re.search(r'^GPU (?P<gpu_id>\d+):\s*(?P<gpu_name>.*)\s+\(UUID: (?P<gpu_uuid>.*)\)\s*', line)
        if m:
            device_dicts.append({
                "device_number":int(m.group('gpu_id')),
                "device_name":m.group('gpu_name'),
            })
    visible_gpu_ids = get_visible_gpu_ids()
    keep_devices = [gpu for gpu in device_dicts
                    if visible_gpu_ids is None or gpu['device_number'] in visible_gpu_ids]
    return keep_devices

    # Don't user TensorFlow to do this since it allocates the GPU when it runs...
    #
    # config = tf.ConfigProto()
    # # Allow multiple users to use the TensorFlow API.
    # config.gpu_options.allow_growth = True  # <--- even with this, it still user 645 MB!
    #
    # logger.info("Before list_local_devices")
    # local_device_protos = tf_device_lib.list_local_devices(config) # <-- this trigger GPU allocation
    # device_protos = [x for x in local_device_protos if x.device_type == 'GPU']
    # device_dicts = [_device_proto_as_dict(device_proto) for device_proto in device_protos]
    # return device_dicts

def get_visible_gpu_ids():
    if 'CUDA_VISIBLE_DEVICES' not in ENV:
        return None
    gpu_ids = sorted(int(gpu_id) for gpu_id in re.split(r'\s*,\s*', ENV['CUDA_VISIBLE_DEVICES']))
    return gpu_ids


def print_cmd(cmd, files=sys.stdout, env=None, dry_run=False, only_show_env=None):
    string = cmd_debug_msg(cmd, env=env, dry_run=dry_run, only_show_env=only_show_env)

    if type(files) not in [set, list]:
        if type(files) in [list]:
            files = set(files)
        else:
            files = set([files])

    for f in files:
        print(string, file=f)
        f.flush()


def cmd_debug_msg(cmd, env=None, dry_run=False, only_show_env=None):
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
        show_keys = set(env.keys())
        if only_show_env is not None:
            show_keys = show_keys.intersection(set(only_show_env))
        env_vars = sorted(show_keys)
        if len(env_vars) > 0:
            lines.append("  Environment:")
            for var in env_vars:
                lines.append("    {var}={val}".format(
                    var=var,
                    val=env[var]))
    string = '\n'.join(lines)

    return string

def log_cmd(cmd, env=None, dry_run=False):
    string = cmd_debug_msg(cmd, env=env, dry_run=dry_run)

    logger.info(string)

def pprint_msg(dic, prefix='  '):
    """
    Give logger.info a string for neatly printing a dictionary.

    Usage:
    logger.info(pprint_msg(arbitrary_object))
    """
    return "\n" + textwrap.indent(pprint.pformat(dic), prefix=prefix)

def get_stacktrace(n_skip=1, indent=None):
    """
    Return a stacktrace ready for printing; usage:

    # Dump the stack of the current location of this line.
    '\n'.join(get_stacktrace(0))
    logger.info("First line before stacktrace\n{stack}".format(
        stack=get_stacktrace())

    # Outputs to stdout:
    ID=14751/MainProcess @ finish, profilers.py:1658 :: 2019-07-09 15:52:26,015 INFO: First line before stacktrace
      File "*.py", line 375, in <module>
          ...
      ...
      File "*.py", line 1658, in finish
        logger.info("First line before stacktrace\n{stack}".format(

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
