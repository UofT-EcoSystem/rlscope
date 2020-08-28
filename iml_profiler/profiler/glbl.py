"""
Manage a singleton instance to a Profiler object.
"""

from iml_profiler.profiler.iml_logging import logger
import sys
import os
import re
import platform

from iml_profiler.profiler.iml_logging import logger
from iml_profiler.profiler import profilers
from iml_profiler.profiler import nvidia_gpu_query
from iml_profiler.clib import sample_cuda_api

from iml_profiler.parser.common import *

import tensorflow as tf

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import iml_profiler
from iml_profiler.profiler.iml_logging import logger

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
    # config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(graph=graph, config=config, **kwargs)
    sess.__enter__()

    return session

def patch_environ():
    if 'LD_PRELOAD' not in os.environ:
        return

    if platform.system() != 'Linux':
        raise RuntimeError("IML: currently, IML only support Linux for handling LD_PRELOAD hackery (need to add support for Mac/Windows)")

    # We need to compute this BEFORE we modify LD_PRELOAD.
    is_used = sample_cuda_api.is_used()

    ld_preloads = re.split(r':', os.environ['LD_PRELOAD'])
    keep_ld_preloads = []

    for ld_preload in ld_preloads:
        if not is_sample_cuda_api_lib(ld_preload):
            keep_ld_preloads.append(ld_preload)

    os.environ['LD_PRELOAD'] = ':'.join(keep_ld_preloads)
    logger.info((
        "Remove librlscope.so from LD_PRELOAD:\n"
        "  LD_PRELOAD={LD_PRELOAD}").format(
        LD_PRELOAD=os.environ['LD_PRELOAD']))

def handle_iml_args(parser, args, directory=None, reports_progress=False):
    """
    Build an argument parser,
    :return:

    :param directory
        The directory used by the ML-script for saving its own files.
        If the user doesn't provide --iml-directory (i.e. a separate directory for storing profiling data),
        we fall back on this.
    """
    # if args.iml_directory is not None:
    #     iml_directory = args.iml_directory
    # else:
    #     iml_directory = directory

    if directory is not None:
        iml_directory = directory
    else:
        iml_directory = args.iml_directory

    # TODO: train.py apparently like to launch separate process all willy-nilly.
    # I'm not sure what it's doing this for, but it's certainly true that python-side IML stuff will do it too.
    # We do NOT want to wrap those separate processes with iml-prof.
    # We only want to wrap the "main" training script processes;
    # i.e. the processes we already explicitly handle via the python-api.
    #
    # TLDR: remove librlscope.so from env['LD_PRELOAD'] if it's there, re-add it when we use python-IML to
    # launch new training scripts.
    patch_environ()

    if iml_directory is None:
        raise RuntimeError("IML: you must provide a location to store trace files: --iml-directory <dir>")

    success = profilers.check_avail_gpus()
    if not success:
        sys.exit(1)

    nvidia_gpu_query.check_nvidia_smi()

    init_profiler(
        directory=iml_directory,
        reports_progress=reports_progress,
        args=args,
    )

def handle_gflags_iml_args(FLAGS, directory=None, reports_progress=False):
    """
    Build an argument parser,
    :return:

    :param directory
        The directory used by the ML-script for saving its own files.
        If the user doesn't provide --iml-directory (i.e. a separate directory for storing profiling data),
        we fall back on this.
    """
    return handle_iml_args(parser=None, args=FLAGS, directory=directory, reports_progress=reports_progress)
