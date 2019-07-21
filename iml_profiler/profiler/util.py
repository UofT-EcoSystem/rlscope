import re
import logging
import sys

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
            # ['iml-test',
            #      '--train-script',
            #      'run_baselines.sh',
            #      '--test-name',
            #      'PongNoFrameskip-v4/docker',
            #      '--iml-directory',
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
        logging.info(pprint_msg({
            'subparser_cmd_opts': subparser_cmd_opts,
            'parser_cmd_opts': parser_cmd_opts,
        }))

    subcommand = getattr(args, subparser_argname)

    opts = parser_cmd_opts + [subcommand] + subparser_cmd_opts
    return opts

