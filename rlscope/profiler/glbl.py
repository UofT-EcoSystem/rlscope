"""
Functions for initializing a global singleton RL-Scope Profiler
object from ``--rlscope-*`` command-line arguments.

We provide command-line argument handling for popular argument
parsing libraries (i.e., argparse, click, gflags).

NOTE: the singleton :py:class:`rlscope.api.Profiler` object is stored in :py:obj:`rlscope.api.prof`.
"""

from rlscope.profiler.rlscope_logging import logger
import sys
import os
import re
import platform

from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler import profilers
from rlscope.profiler import nvidia_gpu_query
from rlscope.clib import rlscope_api
from rlscope.parser.exceptions import RLScopeAPIError, RLScopeConfigurationError

from rlscope.parser.common import *


from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import rlscope
from rlscope.profiler.rlscope_logging import logger

session = None

def init_profiler(*args, **kwargs):
    # global prof
    assert rlscope.api.prof is None
    rlscope.api.prof = profilers.Profiler(*args, **kwargs)
    return rlscope.api.prof

def init_session(**kwargs):
    global session
    assert session is None

    import tensorflow as tf

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
    # config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(graph=graph, config=config, **kwargs)
    sess.__enter__()

    return session

def patch_environ():
    if 'LD_PRELOAD' not in os.environ:
        return

    if platform.system() != 'Linux':
        raise RLScopeConfigurationError("Currently, RL-Scope only support Linux for handling LD_PRELOAD hackery (need to add support for Mac/Windows)")

    # We need to compute this BEFORE we modify LD_PRELOAD.
    is_used = rlscope_api.is_used()

    ld_preloads = re.split(r':', os.environ['LD_PRELOAD'])
    keep_ld_preloads = []

    for ld_preload in ld_preloads:
        if not is_rlscope_api_lib(ld_preload):
            keep_ld_preloads.append(ld_preload)

    os.environ['LD_PRELOAD'] = ':'.join(keep_ld_preloads)
    logger.info((
        "Remove librlscope.so from LD_PRELOAD:\n"
        "  LD_PRELOAD={LD_PRELOAD}").format(
        LD_PRELOAD=os.environ['LD_PRELOAD']))

def _add_rlscope_args(args):
    if not rlscope_api.is_used() and not get_arg(args, 'rlscope_disable'):
        logger.warning(
            textwrap.dedent("""\
            Skipping RL-Scope profiling; to run with RL-Scope prefix your command with:
              $ rls-prof ...
                --------
            """).rstrip())
        set_arg(args, 'rlscope_disable', True)

def get_arg(args, attr):
    if hasattr(args, attr):
        return getattr(args, attr)
    return args[attr]

def set_arg(args, attr, value):
    if hasattr(args, attr):
        setattr(args, attr, value)
    else:
        args[attr] = value

def handle_rlscope_args(parser=None, args=None, directory=None, reports_progress=False, delay=False, delay_register_libs=False):
    """
    Initialize the RL-Scope profiler (:py:obj:`rlscope.api.prof`) using :py:obj:`sys.argv`.

    Arguments
    ---------
    parser : argparse.ArgumentParser
        The training script's argparse parser, *with* RL-Scope specific arguments added
        (i.e., you should call :py:func:`rlscope.api.add_rlscope_args` on parser before calling this function).

    directory : str
        The directory used by the training script for saving its own files (e.g., weight checkpoints).
        If the user doesn't provide ``--rlscope-directory``
        (i.e. a separate directory for storing profiling data), we fall back on this.
    """
    # if args.rlscope_directory is not None:
    #     rlscope_directory = args.rlscope_directory
    # else:
    #     rlscope_directory = directory

    if args is None:
        args = dict()

    _add_rlscope_args(args)

    if directory is not None:
        rlscope_directory = directory
    else:
        rlscope_directory = get_arg(args, 'rlscope_directory')

    if delay_register_libs:
        from rlscope.profiler import clib_wrap as rlscope_clib_wrap
        rlscope_clib_wrap.delay_register_libs()

    # TODO: train.py apparently like to launch separate process all willy-nilly.
    # I'm not sure what it's doing this for, but it's certainly true that python-side RL-Scope stuff will do it too.
    # We do NOT want to wrap those separate processes with rls-prof.
    # We only want to wrap the "main" training script processes;
    # i.e. the processes we already explicitly handle via the python-api.
    #
    # TLDR: remove librlscope.so from env['LD_PRELOAD'] if it's there, re-add it when we use python-RL-Scope to
    # launch new training scripts.
    patch_environ()

    rlscope_enabled = not get_arg(args, 'rlscope_disable')
    if rlscope_directory is None and rlscope_enabled:
        raise RLScopeAPIError("You must provide a location to store trace files: --rlscope-directory <dir>")

    if rlscope_enabled:
        success = profilers.check_avail_gpus()
        if not success:
            sys.exit(1)

        nvidia_gpu_query.check_nvidia_smi()

    init_profiler(
        directory=rlscope_directory,
        reports_progress=reports_progress,
        delay=delay,
        args=args,
    )

def handle_gflags_rlscope_args(FLAGS, directory=None, reports_progress=False, **kwargs):
    """
    Build an argument parser,
    :return:

    :param directory
        The directory used by the ML-script for saving its own files.
        If the user doesn't provide --rlscope-directory (i.e. a separate directory for storing profiling data),
        we fall back on this.
    """
    return handle_rlscope_args(parser=None, args=FLAGS, directory=directory, reports_progress=reports_progress, **kwargs)

def handle_click_rlscope_args(rlscope_kwargs, directory=None, reports_progress=False, **kwargs):
    """
    Build an argument parser,
    :return:

    :param directory
        The directory used by the ML-script for saving its own files.
        If the user doesn't provide --rlscope-directory (i.e. a separate directory for storing profiling data),
        we fall back on this.
    """
    return handle_rlscope_args(parser=None, args=rlscope_kwargs, directory=directory, reports_progress=reports_progress, **kwargs)
