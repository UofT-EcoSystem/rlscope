"""
Manage a singleton instance to a Profiler object.
"""

from profiler import profilers

prof = None
def get_profiler():
    global prof
    # if prof is None:
    #     prof = profilers.Profiler(*args, **kwargs)
    return prof

def init_profiler(*args, **kwargs):
    global prof
    assert prof is None
    prof = profilers.Profiler(*args, **kwargs)
    return prof

def handle_iml_args(parser, args, directory=None):
    """
    Build an argument parser,
    :return:

    :param directory
        The directory used by the ML-script for saving its own files.
        If the user doesn't provide --iml-directory (i.e. a separate directory for storing profiling data),
        we fall back on this.
    """
    if args.iml_directory is not None:
        iml_directory = args.iml_dir
    elif directory is not None:
        iml_directory = directory

    profilers.handle_iml_args(output_directory=iml_directory,
                              parser=parser,
                              args=args)
    init_profiler(
        directory=iml_directory,
        args=args,
    )
