"""
Manage a singleton instance to a Profiler object.
"""

from iml_profiler.profiler import profilers
import tensorflow as tf

import iml_profiler

# prof = None
session = None

# def get_profiler():
#     # global prof
#     # if prof is None:
#     #     prof = profilers.Profiler(*args, **kwargs)
#     return iml_profiler.api.prof

def init_profiler(*args, **kwargs):
    # global prof
    assert iml_profiler.api.prof is None
    iml_profiler.api.prof = profilers.Profiler(*args, **kwargs)
    return iml_profiler.api.prof

def init_session(**kwargs):
    global session
    assert session is None

    if 'config' not in kwargs:
        config = tf.ConfigProto()
    else:
        config = kwargs['config']
        del kwargs['config']

    if 'graph' not in kwargs:
        graph = tf.Graph()
    else:
        graph = kwargs['graph']
        del kwargs['graph']

    # Allow multiple users to use the TensorFlow API.
    config.gpu_options.allow_growth = True

    sess = tf.Session(graph=graph, config=config, **kwargs)
    sess.__enter__()

    return session

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
        iml_directory = args.iml_directory
    else:
        iml_directory = directory

    if iml_directory is None:
        raise RuntimeError("IML: you must provide a location to store trace files: --iml-directory <dir>")

    profilers.handle_iml_args(output_directory=iml_directory,
                              parser=parser,
                              args=args)
    init_profiler(
        directory=iml_directory,
        args=args,
    )
