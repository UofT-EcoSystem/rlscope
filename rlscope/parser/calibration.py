"""
``rls-calibrate`` command that automates running training script multiple
times to perform profiling overhead calibration.

This script gets run internally when you invoke ``rls-prof python train.py ...``
on your training script.  RL-Scope calibrates for the average duration of book-keeping 
code paths and substracts this time at the precise point when it occurs.  RL-Scope 
calibrates for several sources of overhead:

1. Python :math:`\leftrightarrow` C interception.
2. CUDA API interception.
3. User's operation annotations.
4. Inflation of cudaLaunchKernel and cudaMemcpyAsync CUDA API calls due to enabling CUPTI.

For (1) and (2), the overhead of RL-Scope
book-keeping code only depends on the type of intercepted event. For example, the extra 
CPU time incurred in (1) by RL-Scope intercepting Python :math:`\leftrightarrow` C/C++ 
transitions is the same regardless of which part of the code it occurs in. Similarly, the 
overhead in (2) of our interception of CUDA API calls does not depend on which CUDA API 
was used, and (3) tracking algorithmic annotations does not depend on which operation was 
annotated. For these cases, we find RL-Scope’s overhead by dividing the increase in total 
runtime when enabling profiling by the number of times the book-keeping code was called.

For (4), this overhead is incurred by internal closed-source profiling code inside the 
CUDA library. This code inflates runtime by different amounts, depending on which CUDA API 
is called. Since profiling cannot be enabled separately for different APIs, accounting for 
it requires tracking the number and duration of each individual API call separately.
"""
from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler import rlscope_logging
import argparse
import pprint
from glob import glob
import subprocess
import multiprocessing
import textwrap
import os
import sys
import numpy as np
import pandas as pd
from os import environ as ENV
import json
import functools

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

import progressbar

from rlscope.profiler.util import print_cmd, get_stacktrace, pprint_msg
from rlscope.profiler.util import run_with_pdb, pprint_msg
from rlscope.parser.common import *
from rlscope.experiment.util import tee, expr_run_cmd, expr_already_ran
from rlscope.profiler.concurrent import ForkedProcessPool, ProcessPoolExecutorWrapper
from rlscope.experiment import expr_config
from rlscope.parser.dataframe import RLScopeConfig
from rlscope.parser.profiling_overhead import \
    parse_microbench_overhead_js, \
    DataframeMapper, \
    PyprofDataframeReader, \
    PyprofOverheadParser, \
    MicrobenchmarkOverheadJSON, \
    CalibrationJSONs
from rlscope.parser import check_host
from rlscope.parser.exceptions import RLScopeConfigurationError, RLScopeRunError, RLScopeAnalysisError

SENTINEL = object()

class Calibration:
    """
    Coordinates running training script multiple times (each with multiple repetitions),
    computing average book-keeping duration ("calibration"),
    running ``rls-analyze`` to analyze event overlap with (and without) overhead correction,
    and creating a time breakdown plot.
    Takes care of only re-running configurations if necessary
    (e.g., didn't complete in prior runs).
    Allows parallelizing multiple training script runs across available GPUs (assumes 1 GPU per training script).
    Also parallelizes offline plotting and analysis.

    See ``rls-calibrate --help`` for documentation on class attributes.
    """
    def __init__(self,
                 mode=None,
                 repetitions=1,
                 replace=False,
                 re_calibrate=False,
                 re_plot=False,
                 dry_run=False,
                 retry=None,
                 skip_plot=False,
                 skip_error=False,
                 max_workers=None,
                 parallel_runs=False,
                 plots=None,
                 gpu_hw=False,
                 gpus=None,
                 debug=False,
                 line_numbers=False,
                 verbosity='progress',
                 debug_single_thread=False,
                 pdb=False,
                 # Ignore extra stuff
                 **kwargs,
                 ):
        self.mode = mode
        self.repetitions = repetitions
        self.replace = replace
        self.re_calibrate = re_calibrate
        self.re_plot = re_plot
        self.dry_run = dry_run
        self.retry = retry
        self.skip_plot = skip_plot
        self.skip_error = skip_error
        self.max_workers = max_workers
        self.parallel_runs = parallel_runs
        self.gpus = gpus
        self.plots = plots
        self.gpu_hw = gpu_hw
        self.debug = debug
        self.line_numbers = line_numbers
        self.verbosity = verbosity
        self.debug_single_thread = debug_single_thread
        self.pdb = pdb
        self._pool = ProcessPoolExecutorWrapper(name='{klass}.pool'.format(
            klass=self.__class__.__name__),
            max_workers=self.max_workers)
        self._has_run_calibration = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_pool']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def cupti_scaling_overhead_dir(self, output_directory):
        return _j(
            output_directory,
            "cupti_scaling_overhead")

    def cupti_scaling_overhead_logfile(self, output_directory):
        task = "CUPTIScalingOverheadTask"
        logfile = _j(
            self.cupti_scaling_overhead_dir(output_directory),
            self._logfile_basename(task),
        )
        return logfile

    def compute_time_breakdown_plot_logfile(self, output_directory):
        task = "OverlapStackedBarTask"
        logfile = _j(
            output_directory,
            self._logfile_basename(task),
        )
        return logfile

    def compute_gpu_hw_plot_logfile(self, output_directory):
        task = "GpuHwPlotTask"
        logfile = _j(
            output_directory,
            self._logfile_basename(task),
        )
        return logfile

    def compute_category_transition_plot_logfile(self, output_directory):
        task = "CategoryTransitionPlotTask"
        logfile = _j(
            output_directory,
            self._logfile_basename(task),
        )
        return logfile

    def cupti_overhead_dir(self, output_directory):
        return _j(
            output_directory,
            "cupti_overhead")

    def cupti_overhead_logfile(self, output_directory):
        task = "CUPTIOverheadTask"
        logfile = _j(
            self.cupti_overhead_dir(output_directory),
            self._logfile_basename(task),
        )
        return logfile

    def LD_PRELOAD_overhead_dir(self, output_directory):
        return _j(
            output_directory,
            "LD_PRELOAD_overhead")

    def LD_PRELOAD_overhead_logfile(self, output_directory):
        task = "CallInterceptionOverheadTask"
        logfile = _j(
            self.LD_PRELOAD_overhead_dir(output_directory),
            self._logfile_basename(task),
        )
        return logfile

    def pyprof_overhead_dir(self, output_directory):
        return _j(
            output_directory,
            "pyprof_overhead")

    def pyprof_overhead_logfile(self, output_directory):
        task = "PyprofOverheadTask"
        logfile = _j(
            self.pyprof_overhead_dir(output_directory),
            self._logfile_basename(task),
        )
        return logfile


    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def _glob_json_files(self, direc):
        json_paths = glob("{direc}/*.json".format(
            direc=direc))
        return json_paths

    def _check_directories_opt(self, task, opt, directories):
        if not self.dry_run and len(directories) == 0:
            raise RuntimeError(textwrap.dedent("""\
            {opt} was empty for \"rls-run --task {task} {opt} ...\"; did you forget to run experiments for this configuration? 
            """).format(
                opt=opt,
                task=task,
                directories=directories,
            ).rstrip())

    def _expr_run_cmd(self, *args, **kwargs):
        tee_output = kwargs.get('tee_output', None)
        if tee_output is None:
            tee_output = (self.verbosity == 'output')
        tee_cmd = (self.verbosity == 'commands')
        return expr_run_cmd(
            *args,
            raise_exception=True,
            tee_output=tee_output,
            tee_cmd=tee_cmd,
            **kwargs)

    def compute_category_transition_plot(self, directories, output_directory, correct_overhead, extra_argv=None,
                            # xtick_expression=None
                            **kwargs,
                            ):
        """
        Plot language transition plot showing the number of transitions between:

        * Python :math:`\rightarrow` ML Backend
        * Python :math:`\rightarrow` Simulator
        * ML Backend :math:`\rightarrow` CUDA API calls

        Plotted by running ``rls-run --task CategoryTransitionPlotTask ...``.
        """
        task = "CategoryTransitionPlotTask"

        repetitions = None
        # repetitions = [1]

        for rlscope_directory in directories:
            if self.dry_run and (
                self.conf(rlscope_directory, 'time_breakdown', calibration=True, dflt=None) is None
            ):
                return

        time_breakdown_dirs = []
        raw_rlscope_dirs = []
        for time_breakdown_directory in directories:
            time_breakdown_dirs.extend(self.conf(time_breakdown_directory, 'time_breakdown', calibration=False, debug=True).rlscope_directories(time_breakdown_directory, repetitions=repetitions, correct_overhead=correct_overhead, debug=True))
            raw_rlscope_dirs.extend(self.conf(time_breakdown_directory, 'time_breakdown', calibration=False, debug=True).rlscope_directories(time_breakdown_directory, repetitions=repetitions, correct_overhead=True, debug=True))

        # Stick plots in root directory.
        if not self.dry_run:
            os.makedirs(output_directory, exist_ok=True)

        self._check_directories_opt(task, '--time-breakdown-directories', time_breakdown_dirs)
        self._check_directories_opt(task, '--rlscope-directories', raw_rlscope_dirs)
        cmd = ['rls-run',
               '--directory', output_directory,
               '--task', task,
               '--time-breakdown-directories', json.dumps(time_breakdown_dirs),
               '--rlscope-directories', json.dumps(raw_rlscope_dirs),
               ]
        if extra_argv is not None:
            cmd.extend(extra_argv)
        # if xtick_expression is not None:
        #     cmd.extend(['--xtick-expression', xtick_expression])
        # self._add_calibration_opts(output_directory, cmd)
        add_rlscope_analyze_flags(cmd, self)

        logfile = self.compute_category_transition_plot_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def compute_gpu_hw_plot(self, directories, output_directory, extra_argv=None,
                            # xtick_expression=None
                            **kwargs,
                            ):
        """
        Plot GPU hardware metrics (e.g., SM occupancy, SM efficiency) scoped to user operations.
        Plotted by running ``rls-run --task GpuHwPlotTask ...``.
        """
        task = "GpuHwPlotTask"

        repetitions = None
        # repetitions = [1]

        for rlscope_directory in directories:
            if self.conf(rlscope_directory, 'gpu_hw', calibration=True, dflt=None) is None or (
                    self.dry_run and (
                        self.conf(rlscope_directory, 'gpu_hw', calibration=True, dflt=None) is None or
                        self.conf(rlscope_directory, 'time_breakdown', calibration=True, dflt=None) is None
            )):
                return

        rlscope_dirs = []
        time_breakdown_dirs = []
        for rlscope_directory in directories:
            rlscope_dirs.extend(self.conf(rlscope_directory, 'gpu_hw', calibration=False).rlscope_directories(rlscope_directory, repetitions=repetitions))
        for time_breakdown_directory in directories:
            time_breakdown_dirs.extend(self.conf(time_breakdown_directory, 'time_breakdown', calibration=False).rlscope_directories(time_breakdown_directory, repetitions=repetitions))

        # Stick plots in root directory.
        if not self.dry_run:
            os.makedirs(output_directory, exist_ok=True)

        self._check_directories_opt(task, '--gpu-hw-directories', rlscope_dirs)
        self._check_directories_opt(task, '--time-breakdown-directories', time_breakdown_dirs)
        cmd = ['rls-run',
               '--rlscope-directory', output_directory,
               '--task', task,
               '--gpu-hw-directories', json.dumps(rlscope_dirs),
               '--time-breakdown-directories', json.dumps(time_breakdown_dirs),
               ]
        if extra_argv is not None:
            cmd.extend(extra_argv)
        # if xtick_expression is not None:
        #     cmd.extend(['--xtick-expression', xtick_expression])
        # self._add_calibration_opts(output_directory, cmd)
        add_rlscope_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.compute_gpu_hw_plot_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def compute_time_breakdown_plot(self, directories, output_directory, correct_overhead, extra_argv,
                                    # xtick_expression=None,
                                    # Ignore
                                    **kwargs):
        """
        Plot CPU/GPU training time breakdown plot with (or without) profilng overhead correction.
        Plotted by running ``rls-run --task OverlapStackedBarTask ...``.
        """
        task = "OverlapStackedBarTask"

        # cmd = [
        #     'rls-run',
        # ]
        # cmd.extend([
        #     '--task', 'OverlapStackedBarTask',
        # ])

        # TODO: add support for multiple repetitions for time breakdown plot
        repetitions = None
        # repetitions = [1]

        for rlscope_directory in directories:
            if self.dry_run and (
                self.conf(rlscope_directory, 'uninstrumented', calibration=True, dflt=None) is None or
                self.conf(rlscope_directory, 'time_breakdown', calibration=False, dflt=None) ):
                return

        unins_rlscope_dirs = []
        rlscope_dirs = []
        rlscope_config_dirs = []
        for rlscope_directory in directories:
            if correct_overhead:
                # Use total training time from uninstrumented run.
                training_time_config_suffix = 'uninstrumented'
                unins_rlscope_dirs.extend(self.conf(rlscope_directory, training_time_config_suffix, calibration=True).rlscope_directories(rlscope_directory, repetitions=repetitions))
            else:
                # Use total training time from instrumented run.
                training_time_config_suffix = 'time_breakdown'
                unins_rlscope_dirs.extend(self.conf(rlscope_directory, training_time_config_suffix, calibration=False).rlscope_directories(rlscope_directory, repetitions=repetitions))
            rlscope_dirs.extend(self.conf(rlscope_directory, 'time_breakdown', calibration=False).rlscope_directories(rlscope_directory, repetitions=repetitions, correct_overhead=correct_overhead))
            if not correct_overhead:
                rlscope_config_dirs.extend(self.conf(rlscope_directory, 'time_breakdown', calibration=False).rlscope_directories(rlscope_directory, repetitions=repetitions, correct_overhead=True))
            if len(rlscope_dirs) != len(unins_rlscope_dirs) or \
                (not correct_overhead and len(rlscope_dirs) != len(rlscope_config_dirs)):
                missing_files = {
                    'rlscope_dirs': rlscope_dirs,
                    'unins_rlscope_dirs': unins_rlscope_dirs,
                }
                if not correct_overhead:
                    missing_files['rlscope_config_dirs'] = rlscope_config_dirs
                log_missing_files(self, task=task, files=missing_files)
                return

        # Stick plots in root directory.
        if not self.dry_run:
            os.makedirs(output_directory, exist_ok=True)
        overlap_type = 'CategoryOverlap'
        self._check_directories_opt(task, '--rlscope-directories', rlscope_dirs)
        self._check_directories_opt(task, '--unins-rlscope-directories', unins_rlscope_dirs)
        cmd = ['rls-run',
               '--directory', output_directory,
               '--task', task,
               '--overlap-type', overlap_type,

               '--y-type', 'percent',
               '--x-type', 'rl-comparison',
               '--detailed',

               # How to handle remap-df?

               # Should we include this...?
               '--training-time',
               '--extrapolated-training-time',

               '--rlscope-directories', json.dumps(rlscope_dirs),
               '--unins-rlscope-directories', json.dumps(unins_rlscope_dirs),
               ]
        if not correct_overhead:
            cmd.extend([
                '--rlscope-config-directories', json.dumps(rlscope_config_dirs),
            ])
        if extra_argv is not None:
            cmd.extend(extra_argv)
        # if xtick_expression is not None:
        #     cmd.extend(['--xtick-expression', xtick_expression])
        self._add_calibration_opts(output_directory, cmd)
        add_rlscope_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.compute_time_breakdown_plot_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def _add_calibration_opts(self, output_directory, cmd):
        cmd.extend([
            '--cupti-overhead-json', self.cupti_overhead_json(output_directory),
            '--LD-PRELOAD-overhead-json', self.LD_PRELOAD_overhead_json(output_directory),
            '--python-annotation-json', self.python_annotation_json(output_directory),
            '--python-clib-interception-tensorflow-json', self.python_clib_interception_tensorflow_json(output_directory),
            '--python-clib-interception-simulator-json', self.python_clib_interception_simulator_json(output_directory),
        ])

    def _calibration_paths(self, output_directory):
        return set([
            self.cupti_scaling_overhead_json(output_directory),
            self.cupti_overhead_json(output_directory),
            self.LD_PRELOAD_overhead_json(output_directory),
            self.python_annotation_json(output_directory),
            self.python_clib_interception_tensorflow_json(output_directory),
            self.python_clib_interception_simulator_json(output_directory),
        ])

    def _calibration_directories(self, output_directory):
        return set([
            self.cupti_scaling_overhead_dir(output_directory),
            self.cupti_overhead_dir(output_directory),
            self.LD_PRELOAD_overhead_dir(output_directory),
            self.pyprof_overhead_dir(output_directory),
        ])

    def _calibration_logfiles(self, output_directory):
        logfiles = set()
        calibration_dirs = self._calibration_directories(output_directory)
        for calibration_dir in calibration_dirs:
            for logfile in glob("{dir}/*.logfile.out".format(dir=calibration_dir)):
                logfiles.add(logfile)
        return logfiles

    def cupti_scaling_overhead_json(self, output_directory):
        return _j(self.cupti_scaling_overhead_dir(output_directory), 'cupti_scaling_overhead.json')
    def cupti_overhead_json(self, output_directory):
        return _j(self.cupti_overhead_dir(output_directory), 'cupti_overhead.json')
    def LD_PRELOAD_overhead_json(self, output_directory):
        return _j(self.LD_PRELOAD_overhead_dir(output_directory), 'LD_PRELOAD_overhead.json')
    def python_annotation_json(self, output_directory):
        return _j(self.pyprof_overhead_dir(output_directory), 'category_events.python_annotation.json')
    def python_clib_interception_tensorflow_json(self, output_directory):
        return _j(self.pyprof_overhead_dir(output_directory), 'category_events.python_clib_interception.json')
    def python_clib_interception_simulator_json(self, output_directory):
        return _j(self.pyprof_overhead_dir(output_directory), 'category_events.python_clib_interception.json')

    def rls_analyze_logfile(self, rlscope_directory, conf, rep, correct_overhead):
        task = "RLSAnalyze"
        directory = conf.out_dir(rlscope_directory, rep, correct_overhead)
        logfile = _j(
            directory,
            self._logfile_basename(task),
        )
        return logfile

    def rls_analyze_output_paths(self, rlscope_directory, conf, rep, correct_overhead):
        directory = conf.out_dir(rlscope_directory, rep, correct_overhead)
        paths = set()
        # ${directory}/*.venn_js.json
        # ${directory}/OverlapResult.*
        # ${directory}/RLSAnalyze.*
        # ${directory}/rlscope_plot_index*
        # ${directory}/__pycache__
        def _add_paths(glob_pattern):
            for path in glob("{dir}/{glob}".format(dir=directory, glob=glob_pattern)):
                paths.add(path)
        # --config time-breakdown
        _add_paths("*.venn_js.json")
        _add_paths("OverlapResult.*")
        _add_paths("RLSAnalyze.*")
        _add_paths("rlscope_plot_index*")
        _add_paths("__pycache__")
        # --config gpu-hw
        _add_paths("GPUHwCounterSampler.csv")
        return paths

    def time_breakdown_plot_paths(self, rlscope_directory, correct_overhead):
        paths = set()
        def _add_paths(glob_pattern):
            if correct_overhead:
                glob_expr = "{dir}/corrected_no/{glob}".format(dir=rlscope_directory, glob=glob_pattern)
            else:
                glob_expr = "{dir}/{glob}".format(dir=rlscope_directory, glob=glob_pattern)
            for path in glob(glob_expr):
                paths.add(path)
        # --config time-breakdown
        _add_paths("OverlapStackedBar*")
        _add_paths("CategoryTransition*")
        return paths

    def gpu_hw_plot_paths(self, rlscope_directory, correct_overhead):
        paths = set()
        def _add_paths(glob_pattern):
            if correct_overhead:
                glob_expr = "{dir}/corrected_no/{glob}".format(dir=rlscope_directory, glob=glob_pattern)
            else:
                glob_expr = "{dir}/{glob}".format(dir=rlscope_directory, glob=glob_pattern)
            for path in glob(glob_expr):
                paths.add(path)
        # --config gpu-hw
        _add_paths("GpuHwPlot*")
        _add_paths("rlscope_*dataframe*")
        _add_paths("rlscope_*csv")
        _add_paths("rlscope_*svg")
        _add_paths("rlscope_*png")
        _add_paths("rlscope_*pdf")
        return paths

    def compute_rls_analyze(self, rlscope_directory, output_directory, conf, rep, correct_overhead):
        """
        Compute cross-stack event overlap with (or without) profiling overhead correction.
        Plotted by running ``rls-run --task RLSAnalyze ...``.
        """
        task = "RLSAnalyze"

        assert conf.rls_analyze_mode is not None

        out_dir = conf.out_dir(rlscope_directory, rep, correct_overhead)
        rlscope_dir = conf.rlscope_dir(rlscope_directory, rep)

        cmd = ['rls-run',
               '--task', task,
               '--output-directory', out_dir,
               '--rlscope-directory', rlscope_dir,
               '--mode', conf.rls_analyze_mode,
               ]
        if correct_overhead:
            self._add_calibration_opts(rlscope_directory, cmd)
        add_rlscope_analyze_flags(cmd, self)

        logfile = self.rls_analyze_logfile(rlscope_directory, conf, rep, correct_overhead)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def only_show_env(self):
        if self.debug:
            # Show all enviroment variables
            return None
        # Show no environments variables
        return {'CUDA_VISIBLE_DEVICES'}

    def compute_cupti_scaling_overhead(self, output_directory):
        """
        Plot CUPTI per-api-call time varies as we scale the number of traced training-loop iterations.
        Plotted by running ``rls-run --task CUPTIScalingOverheadTask ...``.
        """
        task = "CUPTIScalingOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'gpu_activities_api_time', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'interception', calibration=True, dflt=None) ):
            return

        all_gpu_activities_api_time_directories = []
        all_interception_directories = []

        gpu_activities_api_time_directories = self.conf(output_directory, 'gpu_activities_api_time', calibration=True).rlscope_directories(output_directory)
        interception_directories = self.conf(output_directory, 'interception', calibration=True).rlscope_directories(output_directory)
        if len(gpu_activities_api_time_directories) != len(interception_directories):
            log_missing_files(self, task=task, files={
                'gpu_activities_api_time_directories': gpu_activities_api_time_directories,
                'interception_directories': interception_directories,
            })
            return
        all_gpu_activities_api_time_directories.extend(gpu_activities_api_time_directories)
        all_interception_directories.extend(interception_directories)

        directory = self.cupti_scaling_overhead_dir(output_directory)
        if not self.dry_run:
            os.makedirs(output_directory, exist_ok=True)
        self._check_directories_opt(task, '--gpu-activities-api-time-directory', all_gpu_activities_api_time_directories)
        self._check_directories_opt(task, '--interception-directory', all_interception_directories)
        cmd = ['rls-run',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-api-time-directory', json.dumps(all_gpu_activities_api_time_directories),
               '--interception-directory', json.dumps(all_interception_directories),
               ]
        add_rlscope_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.cupti_scaling_overhead_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def compute_cupti_overhead(self, output_directory):
        """
        Compute "4. Inflation of cudaLaunchKernel and cudaMemcpyAsync CUDA API calls due to enabling CUPTI"
        by running ``rls-run --task CUPTIOverheadTask ...``.
        """
        task = "CUPTIOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'gpu_activities', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'no_gpu_activities', calibration=True, dflt=None) ):
            return

        gpu_activities_directories = self.conf(output_directory, 'gpu_activities', calibration=True).rlscope_directories(output_directory)
        no_gpu_activities_directories = self.conf(output_directory, 'no_gpu_activities', calibration=True).rlscope_directories(output_directory)
        if self.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'gpu_activities_directories': gpu_activities_directories,
                    'no_gpu_activities_directories': no_gpu_activities_directories,
                })))

        if len(gpu_activities_directories) != len(no_gpu_activities_directories):
            log_missing_files(self, task=task, files={
                'gpu_activities_directories': gpu_activities_directories,
                'no_gpu_activities_directories': no_gpu_activities_directories,
            })
            return

        directory = self.cupti_overhead_dir(output_directory)
        if not self.dry_run:
            os.makedirs(directory, exist_ok=True)
        self._check_directories_opt(task, '--gpu-activities-directory', gpu_activities_directories)
        self._check_directories_opt(task, '--no-gpu-activities-directory', no_gpu_activities_directories)
        cmd = ['rls-run',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-directory', json.dumps(gpu_activities_directories),
               '--no-gpu-activities-directory', json.dumps(no_gpu_activities_directories),
               ]
        add_rlscope_analyze_flags(cmd, self)

        logfile = self.cupti_overhead_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def compute_LD_PRELOAD_overhead(self, output_directory):
        """
        Compute "(2) CUDA API interception" by running ``rls-run --task CallInterceptionOverheadTask ...``.
        """
        task = "CallInterceptionOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'interception', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'uninstrumented', calibration=True, dflt=None) ):
            return

        interception_directories = self.conf(output_directory, 'interception', calibration=True).rlscope_directories(output_directory)
        uninstrumented_directories = self.conf(output_directory, 'uninstrumented', calibration=True).rlscope_directories(output_directory)
        if self.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'interception_directories': interception_directories,
                    'uninstrumented_directories': uninstrumented_directories,
                })))
        if self.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'interception_directories': interception_directories,
                    'uninstrumented_directories': uninstrumented_directories,
                })))
        if len(interception_directories) != len(uninstrumented_directories):
            log_missing_files(self, task=task, files={
                'interception_directories': interception_directories,
                'uninstrumented_directories': uninstrumented_directories,
            })
            return

        directory = self.LD_PRELOAD_overhead_dir(output_directory)
        if not self.dry_run:
            os.makedirs(directory, exist_ok=True)
        self._check_directories_opt(task, '--interception-directory', interception_directories)
        self._check_directories_opt(task, '--uninstrumented-directory', uninstrumented_directories)
        cmd = ['rls-run',
               '--directory', directory,
               '--task', task,
               '--interception-directory', json.dumps(interception_directories),
               '--uninstrumented-directory', json.dumps(uninstrumented_directories),
               ]
        add_rlscope_analyze_flags(cmd, self)

        logfile = self.LD_PRELOAD_overhead_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def compute_pyprof_overhead(self, output_directory):
        """
        Compute "(3) User's operation annotations" by running ``rls-run --task PyprofOverheadTask ...``.
        """
        task = "PyprofOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'uninstrumented', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'just_pyprof_interceptions', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'just_pyprof_annotations', calibration=True, dflt=None) ):
            return

        uninstrumented_directories = self.conf(output_directory, 'uninstrumented', calibration=True).rlscope_directories(output_directory)
        pyprof_annotations_directories = self.conf(output_directory, 'just_pyprof_annotations', calibration=True).rlscope_directories(output_directory)
        pyprof_interceptions_directories = self.conf(output_directory, 'just_pyprof_interceptions', calibration=True).rlscope_directories(output_directory)
        if self.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'uninstrumented_directories': uninstrumented_directories,
                    'pyprof_annotations_directories': pyprof_annotations_directories,
                    'pyprof_interceptions_directories': pyprof_interceptions_directories,
                })))

        if len({
            len(uninstrumented_directories),
            len(pyprof_annotations_directories),
            len(pyprof_interceptions_directories),
        }) != 1:
            log_missing_files(self, task=task, files={
                'uninstrumented_directories': uninstrumented_directories,
                'pyprof_annotations_directories': pyprof_annotations_directories,
                'pyprof_interceptions_directories': pyprof_interceptions_directories,
            })
            return

        directory = self.pyprof_overhead_dir(output_directory)
        if not self.dry_run:
            os.makedirs(directory, exist_ok=True)
        self._check_directories_opt(task, '--uninstrumented-directory', uninstrumented_directories)
        self._check_directories_opt(task, '--pyprof-annotations-directory', pyprof_annotations_directories)
        self._check_directories_opt(task, '--pyprof-interceptions-directory', pyprof_interceptions_directories)
        cmd = ['rls-run',
               '--directory', directory,
               '--task', task,
               '--uninstrumented-directory', json.dumps(uninstrumented_directories),
               '--pyprof-annotations-directory', json.dumps(pyprof_annotations_directories),
               '--pyprof-interceptions-directory', json.dumps(pyprof_interceptions_directories),
               ]
        add_rlscope_analyze_flags(cmd, self)

        logfile = self.pyprof_overhead_logfile(output_directory)
        self._expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            exception_class=RLScopeAnalysisError,
            only_show_env=self.only_show_env())

    def config_repetitions(self):
        config_repetitions = []
        for rep in range(1, self.repetitions+1):
            for config in self.configs:
                config_repetitions.append((rep, config))
        return config_repetitions

    @property
    def should_show_progress(self):
        return self.verbosity == 'progress'

    def run_configs(self, cmd, output_directory):
        if not self.parallel_runs:
            # Run configurations serially on which GPU(s) are visible.
            config_repetitions = self.config_repetitions()
            bar = None
            if self.should_show_progress:
                bar = progressbar.ProgressBar(max_value=len(config_repetitions))
            for i, (rep, config) in enumerate(config_repetitions):
                config.run(rep, cmd, output_directory)
                if self.should_show_progress:
                    bar.update(i)
            if self.should_show_progress:
                bar.finish()
            logger.info("Trace files have been recorded @ {direc}/*".format(
                direc=output_directory,
            ))
            return

        # Parallelize running configurations across GPUs on this machine (assume no CPU interference).
        # Record commands in run_expr.sh, and run:
        # $ rls-run-expr --run-sh --sh run_expr.sh
        run_expr_sh = self._run_expr_sh(output_directory)
        logger.debug(f"Writing run configuration shell commands to {run_expr_sh}")
        os.makedirs(_d(run_expr_sh), exist_ok=True)
        with open(run_expr_sh, 'w') as f:
            for rep, config in self.config_repetitions():
                run_cmd = config.run_cmd(rep, cmd, output_directory)
                full_cmd = run_cmd.cmd + ['--rlscope-logfile', run_cmd.logfile]
                quoted_cmd = [shlex.quote(opt) for opt in full_cmd]
                f.write(' '.join(quoted_cmd))
                f.write('\n')
        run_expr_cmd = ['rls-run-expr', '--verbosity', self.verbosity, '--skip-final-error-message', '--run-sh', '--sh', run_expr_sh]
        if self.dry_run:
            run_expr_cmd.append('--dry-run')
        if self.debug:
            run_expr_cmd.append('--debug')
        if self.line_numbers:
            run_expr_cmd.append('--line-numbers')
        if self.retry is not None:
            run_expr_cmd.extend(['--retry', str(self.retry)])
        # NOTE: don't forward pdb since we cannot interact with parallel processes.
        logger.info(f"Running configurations...")
        print_cmd(run_expr_cmd)
        proc = subprocess.run(run_expr_cmd, check=False)
        retcode = proc.returncode
        if retcode != 0:
            logger.error("At least one run configuration failed; see their logfiles for details.")
            sys.exit(1)
        logger.info("Trace files have been recorded @ {direc}/*".format(
            direc=output_directory,
        ))


    def _run_expr_sh(self, output_directory):
        return _j(output_directory, 'run_expr.sh')

    def do_run(self, cmd, output_directory):

        self.run_configs(cmd, output_directory)

        # Lets save calibration to happen during rls-plot to make it easier to split up "analysis" from
        # "calibration" time.
        # unless they give --rls-plot
        if not self.skip_plot:
            self.do_calibration(output_directory)
        else:
            logger.info("--rlscope-skip-plot: SKIP processing results into plots; run rls-plot to do this later.")

    def _rm_path(self, path, opt):
        if _e(path):
            logger.info("{opt}: RM {path}".format(
                path=path,
                opt=opt))
            if not self.dry_run:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    def clean_plots(self, output_directory):
        opt = '--re-plot'

        def should_clean(consider):
            return self.mode == 'run' or (
                self.mode == 'plot' and ( self.plots == 'all' or self.plots == consider )
            )

        for correct_overhead in [True, False]:

            if should_clean('time-breakdown'):
                for path in self.time_breakdown_plot_paths(output_directory, correct_overhead=correct_overhead):
                    self._rm_path(path, opt)

            if should_clean('gpu-hw'):
                for path in self.gpu_hw_plot_paths(output_directory, correct_overhead=correct_overhead):
                    self._rm_path(path, opt)

    def clean_analysis(self, output_directory):
        opt = '--re-calibrate'

        for path in self._calibration_paths(output_directory):
            self._rm_path(path, opt)
        for path in self._calibration_logfiles(output_directory):
            self._rm_path(path, opt)

        # Need to re-run rls-analyze using new calibration files.
        for rep in range(1, self.repetitions+1):
            for config in self.configs:
                if config.rls_analyze_mode is not None:
                    for correct_overhead in [True, False]:
                        for path in self.rls_analyze_output_paths(output_directory, config, rep, correct_overhead):
                            self._rm_path(path, opt)


    def do_calibration(self, output_directory, sync=True, skip_log=False):
        if output_directory in self._has_run_calibration:
            logger.debug("SKIP re-running calibration for {path}".format(path=output_directory))
            return False
        if not skip_log:
            logger.info("Computing profiling overhead calibration...")
        self._pool.submit(
            get_func_name(self, 'compute_cupti_scaling_overhead'),
            self.compute_cupti_scaling_overhead,
            output_directory,
            sync=self.debug_single_thread,
        )
        self._pool.submit(
            get_func_name(self, 'compute_cupti_overhead'),
            self.compute_cupti_overhead,
            output_directory,
            sync=self.debug_single_thread,
        )
        self._pool.submit(
            get_func_name(self, 'compute_LD_PRELOAD_overhead'),
            self.compute_LD_PRELOAD_overhead,
            output_directory,
            sync=self.debug_single_thread,
        )
        self._pool.submit(
            get_func_name(self, 'compute_pyprof_overhead'),
            self.compute_pyprof_overhead,
            output_directory,
            sync=self.debug_single_thread,
        )
        if sync:
            self._pool.shutdown(show_progress=self.should_show_progress)
        # logger.info("Calibration files have been generated @ {direc}/*_overhead:\n{stack}".format(
        if not skip_log:
            logger.info("Calibration files have been generated @ {direc}/*_overhead".format(
                direc=output_directory,
            ))
        self._has_run_calibration.add(output_directory)
        return True

    def _get_plot_dir(self, output_directory, correct_overhead):
        if correct_overhead:
            plot_dir = output_directory
        else:
            plot_dir = _j(output_directory, corrected_suffix(correct_overhead, skip_dot=True))
        return plot_dir

    def needs_calibration(self, directories):
        return [rlscope_directory for rlscope_directory in directories \
                if rlscope_directory not in self._has_run_calibration]

    def do_plot(self, directories, output_directory, extra_argv=None, **kwargs):
        needs_calibration = self.needs_calibration(directories)
        if len(needs_calibration) > 0:
            lines = []
            for rlscope_directory in needs_calibration:
                lines.append("  {dir}".format(dir=rlscope_directory))
            logger.info("Computing profiling overhead calibration for:\n{lines}".format(
                lines='\n'.join(lines),
            ))
        calibration_running = []
        for rlscope_directory in directories:
            ran_calibration = self.do_calibration(rlscope_directory, sync=False, skip_log=True)
            if ran_calibration:
                calibration_running.append(rlscope_directory)
        self._pool.shutdown(show_progress=self.should_show_progress)

        # rls-analyze needs the JSON calibration files; wait.
        logger.info("Analyzing trace files...")
        for rlscope_directory in directories:
            for rep in self.each_repetition():
                for config in self.configs:
                    if config.rls_analyze_mode is not None:
                        for correct_overhead in [True, False]:
                            self._pool.submit(
                                get_func_name(self, 'compute_rls_analyze'),
                                self.compute_rls_analyze,
                                rlscope_directory, output_directory,
                                config, rep, correct_overhead,
                                sync=self.debug_single_thread,
                            )
        self._pool.shutdown(show_progress=self.should_show_progress)
        # Plotting requires running rls-analyze

        logger.info("Generating plots...")
        def should_plot(consider):
            return self.mode == 'run' or (
                self.mode == 'plot' and ( self.plots == 'all' or self.plots == consider )
            )

        for correct_overhead in [True, False]:
            if should_plot('time-breakdown'):
                plot_dir = self._get_plot_dir(output_directory, correct_overhead)
                self._pool.submit(
                    get_func_name(self, 'compute_time_breakdown_plot'),
                    self.compute_time_breakdown_plot,
                    directories, plot_dir, correct_overhead, extra_argv,
                    sync=self.debug_single_thread,
                    **kwargs,
                )
        if should_plot('gpu-hw'):
            self._pool.submit(
                get_func_name(self, 'compute_gpu_hw_plot'),
                self.compute_gpu_hw_plot,
                directories, output_directory, extra_argv,
                sync=self.debug_single_thread,
                **kwargs,
            )
        # NOTE: there's no such thing as a "uncorrected transition graph"
        for correct_overhead in [False]:
            if should_plot('category-transition'):
                self._pool.submit(
                    get_func_name(self, 'compute_category_transition_plot'),
                    self.compute_category_transition_plot,
                    directories, output_directory, correct_overhead, extra_argv,
                    sync=self.debug_single_thread,
                    **kwargs,
                )
        self._pool.shutdown(show_progress=self.should_show_progress)
        logger.info("Plots have been generated @ {direc}/*.{{pdf,png}}".format(
            direc=output_directory,
        ))

    def each_repetition(self):
        return range(1, self.repetitions+1)

    def conf(self, output_dir, config_suffix, calibration=False, dflt=SENTINEL, debug=False):
        # if debug:
        #     import pdb; pdb.set_trace()
        calib_config_suffix = self.calibrated_name(config_suffix, calibration=calibration)
        if dflt is SENTINEL:
            return self.config_map[output_dir][calib_config_suffix]
        return self.config_map[output_dir].get(calib_config_suffix, dflt)

    def calibrated_name(self, config_suffix, calibration=False):
        if not calibration:
            return config_suffix
        return "calibration_{suffix}".format(
            suffix=config_suffix,
        )

    def init_configs(self, directories, output_directory):
        self.configs = []
        self.config_map = dict()
        for directory in directories:
            self._add_configs(directory)
        ss = StringIO()
        ss.write("Run configurations:")
        ss.write("\n")
        for config in self.configs:
            ss.write("  ")
            ss.write(config.pretty_output_dir(output_directory))
            ss.write("\n")
        logger.info(ss.getvalue().rstrip())


    def _add_configs(self, output_dir):

        def add_calibration_config(calibration=True, **common_kwargs):
            config_kwargs = dict(common_kwargs)
            base_config = RLScopeRunConfig(**config_kwargs)
            if not calibration:
                config = base_config
            else:
                config_suffix = self.calibrated_name(base_config.config_suffix, calibration=calibration)
                calibration_config_kwargs = dict(common_kwargs)
                calibration_config_kwargs.update(dict(
                    # Disable tfprof: CUPTI and LD_PRELOAD.
                    config_suffix=config_suffix,
                ))
                config = RLScopeRunConfig(**calibration_config_kwargs)
                assert config.is_calibration

            self.configs.append(config)
            if output_dir not in self.config_map:
                self.config_map[output_dir] = dict()
            assert config.config_suffix not in self.config_map[output_dir]
            self.config_map[output_dir][config.config_suffix] = config

        add_calibration_config(
            expr=self,
            rlscope_prof_config='time-breakdown',
            config_suffix='time_breakdown',
            script_args=[],
            calibration=False,
            rls_analyze_mode='overlap',
        )

        if self.gpu_hw:
            add_calibration_config(
                expr=self,
                rlscope_prof_config='gpu-hw',
                config_suffix='gpu_hw',
                script_args=[],
                calibration=False,
                rls_analyze_mode='gpu_hw',
            )

        # Entirely uninstrumented configuration; we use this in many of the overhead calculations to determine
        # how much training time is attributable to the enabled "feature" (e.g. CUPTI activities).
        add_calibration_config(
            expr=self,
            rlscope_prof_config='uninstrumented',
            config_suffix='uninstrumented',
            # Disable ALL pyprof/tfprof stuff.
            script_args=['--rlscope-disable'],
        )
        # Entirely uninstrumented configuration (CALIBRATION runs);
        # NOTE: we need to re-run the uninstrumented configuration for the calibration runs, so that we can
        # see how well our calibration generalizes.
        # self.config_uninstrumented = config

        add_calibration_config(
            expr=self,
            rlscope_prof_config='interception',
            config_suffix='interception',
            script_args=['--rlscope-disable-pyprof'],
        )

        # CUPTIOverheadTask: CUPTI, and CUDA API stat-tracking overhead correction.
        add_calibration_config(
            expr=self,
            rlscope_prof_config='gpu-activities',
            config_suffix='gpu_activities',
            script_args=['--rlscope-disable-pyprof'],
        )
        add_calibration_config(
            expr=self,
            rlscope_prof_config='no-gpu-activities',
            config_suffix='no_gpu_activities',
            script_args=['--rlscope-disable-pyprof'],
        )
        # # CUPTIScalingOverheadTask:
        # config = RLScopeRunConfig(
        #     expr=self,
        #     rlscope_prof_config='gpu-activities-api-time',
        #     config_suffix='gpu_activities_api_time_calibration',
        #     script_args=['--rlscope-disable-pyprof'],
        # )
        # self.configs.append(config)
        add_calibration_config(
            expr=self,
            rlscope_prof_config='gpu-activities-api-time',
            config_suffix='gpu_activities_api_time',
            script_args=['--rlscope-disable-pyprof'],
        )

        # if self.calibration_mode == 'validation':
        #     # Evaluate: combined tfprof/pyprof overhead correction.
        #     # (i.e. full RL-Scope trace-collection).
        #     config = RLScopeRunConfig(
        #         expr=self,
        #         rlscope_prof_config='full',
        #         # Enable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix='full',
        #         # Enable pyprof.
        #         script_args=[],
        #         long_run=True,
        #     )
        #     self.configs.append(config)

        # if self.calibration_mode == 'validation':
        #     # Evaluate: tfprof overhead correction in isolation.
        #     config = RLScopeRunConfig(
        #         expr=self,
        #         rlscope_prof_config='full',
        #         # Enable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix='just_tfprof',
        #         # DON'T enable pyprof.
        #         script_args=['--rlscope-disable-pyprof'],
        #         long_run=True,
        #     )
        #     self.configs.append(config)

        # if self.calibration_mode == 'validation':
        #     # Evaluate: pyprof overhead correction in isolation.
        #     config = RLScopeRunConfig(
        #         expr=self,
        #         rlscope_prof_config='uninstrumented',
        #         # Disable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix='just_pyprof',
        #         # Enable pyprof.
        #         script_args=['--rlscope-disable-tfprof'],
        #     )
        #     self.configs.append(config)

        # PyprofOverheadTask: Python->C-lib event tracing, and operation annotation overhead correction.
        add_calibration_config(
            expr=self,
            rlscope_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_annotations',
            # Only enable GPU/C-lib event collection, not operation annotations.
            script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-interceptions'],
        )
        add_calibration_config(
            expr=self,
            rlscope_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_interceptions',
            # Only enable operation annotations, not GPU/C-lib event collection.
            script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-annotations'],
        )

    def maybe_clean(self, directory, skip_analysis=False):
        # PROBLEM: is this going to run twice in the rls-plot call?
        if not skip_analysis and self.re_calibrate:
            self.clean_analysis(directory)
        if self.re_calibrate or self.re_plot:
            self.clean_plots(directory)

    def _mode_run(self, cmd, directory):
        self.init_configs([directory], output_directory=directory)

        self.maybe_clean(directory)

        self.do_run(cmd, directory)
        self.do_plot(directories=[directory], output_directory=directory)

    def mode_run(self, *args, **kwargs):
        """
        Run each configuration needed to calibrate for overhead correction.
        Output results to:
        --rlscope_directory/
          config_calibration_{short_name}/
            Various calibration runs for computing overhead corrections for
            the config_time_breakdown run.
          config_time_breakdown/
            Time breakdown run for collecting the CPU/GPU time.
          config_gpu_hw/
            GPU HW run for collecting GPU HW counters.

        --rlscope-directory: directory to output results of calibration runs, AND regular run.
        (1) First, run each configuration sequentially
        (2) Then, run "rls-analyze" in parallel for each configuration.
            Run "--mode=overlap" for all configuration except config_gpu_hw.
            Run "--mode=gpu_hw" for config_gpu_hw.
        (3) Run the rls-run program that computes configuration json files;
            output them to the root directory.
        """
        self._with_exception_handler(lambda: self._mode_run(*args, **kwargs))


    def mode_plot(self, *args, **kwargs):
        self._with_exception_handler(lambda: self._mode_plot(*args, **kwargs))

    def _with_exception_handler(self, func):
        try:
            func()
        except Exception as e:
            # DON'T wait for pool... it may cause more exceptions.
            self._pool.shutdown(ignore_exceptions=True, show_progress=False)
            if isinstance(e, RLScopeRunError) or isinstance(e, RLScopeAnalysisError):
                logger.error("RL-Scope saw one or more errors; look for \"ERROR\" lines above for details.")
                sys.exit(1)
            if self.debug:
                logger.error("Unhandled exception in calibration.py")
            raise

        self._pool.shutdown(show_progress=False)

    def _mode_plot(self, directories, output_directory, extra_argv=None, **kwargs):
        self.init_configs(directories, output_directory=output_directory)

        # Output directory just contains plots; no need to delete analysis files.
        self.maybe_clean(output_directory, skip_analysis=True)
        if self.re_calibrate:
            # Only delete plots from --rlscope-directories if we're re-calibrating everything.
            for rlscope_directory in directories:
                self.maybe_clean(rlscope_directory)

        # self.do_run(cmd, directory)
        self.do_plot(directories=directories, output_directory=output_directory, extra_argv=extra_argv, **kwargs)
        logger.info("Success! Plots output @ {direc}".format(
            direc=output_directory,
        ))


class RLScopeRunConfig:
    def __init__(self, expr, rlscope_prof_config, config_suffix, rls_analyze_mode=None, script_args=[]):
        self.expr = expr
        # $ rls-prof --config ${rlscope_prof_config}
        self.rlscope_prof_config = rlscope_prof_config
        # $ python train.py --rlscope-directory config_${config_suffix}
        self.config_suffix = config_suffix
        self.script_args = script_args
        self.rls_analyze_mode = rls_analyze_mode

    @property
    def is_calibration(self):
        return re.search(r'calibration', self.config_suffix)

    def out_dir(self, output_directory, rep, correct_overhead=True):
        return _j(
            output_directory,
            "config_{config_suffix}{rep}{corrected}".format(
                config_suffix=self.config_suffix,
                rep=rep_suffix(rep),
                corrected=corrected_suffix(correct_overhead),
            ))

    def rlscope_dir(self, output_directory, rep):
        return _j(
            output_directory,
            "config_{config_suffix}{rep}".format(
                config_suffix=self.config_suffix,
                rep=rep_suffix(rep),
            ))

    def pretty_output_dir(self, output_directory):
        return self.out_dir(output_directory, rep='*')

    def pretty_to_string(self, output_directory):
        return ("{klass}("
                "output={pretty_output_dir}"
                ")").format(
            klass=self.__class__.__name__,
            pretty_output_dir=self.pretty_output_dir(output_directory)
        )

    def to_string(self):
        return ("{klass}("
                "rlscope_prof_config='{rlscope_prof_config}'"
                ", config_suffix='{config_suffix}'"
                ")").format(
            klass=self.__class__.__name__,
            rlscope_prof_config=self.rlscope_prof_config,
            config_suffix=self.config_suffix,
        )

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def logfile(self, output_directory, rep, correct_overhead=True):
        # logfile = _j(self.out_dir(output_directory, rep, correct_overhead), "logfile.out")
        # NOTE: ignore correct_overhead for logfile location of running a configuration.
        logfile = _j(self.out_dir(output_directory, rep, correct_overhead=True), "logfile.out")
        return logfile

    def run_cmd(self, rep, cmd, output_directory):
        rlscope_prof_cmd = [
            'rls-prof',
            '--no-calibrate',
            '--config', self.rlscope_prof_config,
        ]
        rlscope_prof_cmd.extend(cmd)
        rlscope_prof_cmd.extend([
            # IMPORTANT: When we run with "--config uninstrumented" during calibrations runs, we STILL want to
            # keep "python interceptions" and "python annotations" enabled, so we can measure their overhead in
            # in isolation!
            '--rlscope-calibration',
            '--rlscope-directory', _a(self.out_dir(output_directory, rep)),
        ])

        rlscope_prof_cmd.extend(self.script_args)
        logfile = self.logfile(output_directory, rep)
        run_cmd = RunCmd(cmd=rlscope_prof_cmd, output_directory=output_directory, logfile=logfile)
        return run_cmd

    def run(self, rep, cmd, output_directory):
        run_cmd = self.run_cmd(rep, cmd, output_directory)
        self.expr._expr_run_cmd(
            cmd=run_cmd.cmd,
            to_file=run_cmd.logfile,
            # cwd=ENV['RL_BASELINES_ZOO_DIR'],
            tee_output=False,
            replace=self.expr.replace,
            dry_run=self.expr.dry_run,
            skip_error=self.expr.skip_error,
            debug=self.expr.debug,
            exception_class=RLScopeRunError,
            only_show_env=self.expr.only_show_env())

    def already_ran(self, output_directory, rep, correct_overhead):
        logfile = self.logfile(output_directory, rep, correct_overhead)
        return expr_already_ran(logfile, debug=self.expr.debug)

    def rlscope_directories(self, output_directory, repetitions=None, correct_overhead=True, debug=False):
        """
        Return all --rlscope-directories whose runs are completed.
        """
        # if debug:
        #     import pdb; pdb.set_trace()
        rlscope_directories = []
        # for rep in range(1, self.expr.repetitions+1):
        if repetitions is None:
            repetitions = self.expr.each_repetition()
        for rep in repetitions:
            if not self.already_ran(output_directory, rep, correct_overhead):
                continue
            rlscope_directory = self.out_dir(output_directory, rep, correct_overhead)
            rlscope_directories.append(rlscope_directory)
        return rlscope_directories


def error(msg, parser=None):
    if parser is not None:
        parser.print_usage()
    logger.error(msg)
    sys.exit(1)

def main_plot():
    return _main(['plot'] + sys.argv[1:])

def main_run():
    return _main(['run'] + sys.argv[1:])

def _main(argv):

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
        formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(
        title="Subcommands",
        description="Run RLScope in different modes (run configurations, plot results).",
    )
    def add_common_arguments(parser):
        # parser.add_argument("--output-directory",
        #                     required=True,
        #                     help=textwrap.dedent("""\
        #                     Root directory for output
        #                     """))
        parser.add_argument("--pdb",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Debug
                            """))
        parser.add_argument("--verbosity",
                            choices=['progress', 'commands', 'output'],
                            default='progress',
                            help=textwrap.dedent("""\
                            Output information about running commands.
                            --verbosity progress (Default)
                                Only show high-level progress bar information.
                              
                            --verbosity commands
                                Show the command-line of commands that are being run.
                                
                            --verbosity output
                                Show the output of each analysis (not configuration) command on sys.stdout.
                                NOTE: This may cause interleaving of lines.
                            """))
        parser.add_argument("--debug",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Debug
                            """))
        parser.add_argument('--line-numbers', action='store_true', help=textwrap.dedent("""\
        Show line numbers and timestamps in RL-Scope logging messages.
        """))
        parser.add_argument("--max-workers",
                            default=multiprocessing.cpu_count(),
                            help=textwrap.dedent("""\
                            Number of parallel rls-analysis jobs to run at one time.
                            Default: number of CPU cores.
                            """))
        parser.add_argument("--debug-single-thread",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Debug
                            """))
        # parser.add_argument("--sh",
        #                     help=textwrap.dedent("""\
        #                     Shell file to append commands to (see --append).
        #                     """))
        parser.add_argument('--rlscope-repetitions',
                            type=int,
                            default=1,
                            help=textwrap.dedent("""\
                            Repetitions
                            """))
        parser.add_argument("--replace",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Replace
                            """))
        parser.add_argument("--dry-run",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Dry run
                            """))
        parser.add_argument("--re-calibrate",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Remove existing profiling overhead calibration files, and recompute them.
                            """))
        parser.add_argument("--re-plot",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Remove existing plots and remake them (NOTE: doesn't recompute analysis; see --re-calibrate).
                            """))
        parser.add_argument("--skip-error",
                            action='store_true',
                            help=textwrap.dedent("""\
                            Skip errors 
                            """))
        parser.add_argument("--gpu-hw",
                            action='store_true',
                            help=textwrap.dedent("""\
                                Collect GPU hardware counters.
                                """))

    run_parser = subparsers.add_parser('run', description="Run <cmd> with profiling overhead calibration")
    add_common_arguments(run_parser)
    run_parser.add_argument("--rlscope-directory",
                            required=True,
                            help=textwrap.dedent("""\
                        Root directory for output
                        """))
    run_parser.add_argument("--rlscope-skip-plot",
                        action='store_true',
                        help=textwrap.dedent("""\
                            After running configurations, DON'T run analysis to output plots for individual workload.
                            """))
    run_parser.add_argument("--gpus",
                        help=textwrap.dedent("""\
                            GPUs to run with for --parallel-runs
                            """))

    parallel_runs_help = textwrap.dedent("""\
                            Parallelize running configurations across GPUs on this machine (assume no CPU interference). 
                            See --gpus.
                            """)
    run_parser.add_argument("--parallel-runs",
                            dest='parallel_runs',
                            action='store_true',
                            default=True,
                            help=parallel_runs_help)
    run_parser.add_argument("--no-parallel-runs",
                            dest='parallel_runs',
                            action='store_false',
                            help=parallel_runs_help)

    run_parser.add_argument("--retry",
                            type=int,
                            help=textwrap.dedent("""\
                            If a command fails, retry it up to --retry times.
                            Default: don't retry.
                            """))
    run_parser.set_defaults(**{'mode': 'run'})

    plot_parser = subparsers.add_parser('plot', description="Plot multiple workloads")
    add_common_arguments(plot_parser)
    plot_parser.add_argument("--rlscope-directories",
                            required=True,
                            nargs='+',
                            # default=[],
                            help=textwrap.dedent("""\
                        Directories to plot results from.
                        """))
    plot_parser.add_argument("--output-directory",
                            required=True,
                            help=textwrap.dedent("""\
                        Where to output plots.
                        """))
    plot_parser.add_argument("--plots",
                             choices=['gpu-hw', 'time-breakdown', 'all'],
                             default='all',
                             help=textwrap.dedent("""\
                        Where to output plots.
                        """))
    plot_parser.set_defaults(**{'mode': 'plot'})

    # parser.add_argument("--mode",
    #                     choices=['run', 'plot'],
    #                     default=default_mode,
    #                     help=textwrap.dedent("""\
    #                         Debug
    #                         """))

    args, extra_argv = parser.parse_known_args(argv)
    cmd = extra_argv

    rlscope_logging.setup_logger(
        debug=args.debug,
        line_numbers=args.debug or args.line_numbers or py_config.is_development_mode(),
    )

    args_dict = dict(vars(args))
    repetitions = args_dict.pop('rlscope_repetitions')
    # dry_run = args_dict.pop('rlscope_dry_run')
    # re_calibrate = args_dict.pop('rlscope_re_calibrate')
    # re_plot = args_dict.pop('rlscope_re_plot')
    skip_plot = args_dict.pop('rlscope_skip_plot', False)
    # parallel_runs = args_dict.pop('rlscope_parallel_runs')
    # gpus = args_dict.pop('rlscope_gpus')
    obj = Calibration(
        repetitions=repetitions,
        # re_calibrate=re_calibrate,
        # re_plot=re_plot,
        # dry_run=dry_run,
        skip_plot=skip_plot,
        # parallel_runs=parallel_runs,
        # gpus=gpus,
        **args_dict,
    )

    if args.mode == 'run':
        assert cmd[0] in ['run', 'plot']
        cmd = cmd[1:]

        if len(cmd) == 0:
            error("Expected cmd to run with rls-prof for calibration, but non was provided",
                  parser=parser)

        if shutil.which(cmd[0]) is None:
            error("Couldn't find {exec} on PATH".format(
                exec=cmd[0]), parser=parser)

        def _run():
            directory = args_dict['rlscope_directory']
            obj.mode_run(cmd, directory)
        run_with_pdb(args, _run)
    elif args.mode == 'plot':
        # if len(cmd) != 0:
        #     error(
        #         textwrap.dedent("""\
        #         Not sure how to parse extra arguments for "rls-calibrate plot":
        #           {cmd}
        #         Did you intend to run "rls-calibrate run" instead?
        #         """).format(
        #             cmd=' '.join(cmd),
        #         ).rstrip(), parser=parser)

        def _plot():
            directories = args_dict.pop('rlscope_directories')
            output_directory = args_dict.pop('output_directory')
            obj.mode_plot(
                directories=directories,
                output_directory=output_directory,
                extra_argv=cmd,
                **args_dict,
            )
        run_with_pdb(args, _plot)
    else:
        raise NotImplementedError()


def rep_suffix(rep):
    assert rep is not None
    try:
        rep_int = int(rep)
        rep_str = "{rep:02}".format(rep=rep)
    except ValueError:
        rep_str = rep
    return "_repetition_{rep}".format(rep=rep_str)

def corrected_suffix(correct_overhead, skip_dot=False):
    ss = StringIO()
    if not correct_overhead:
        if not skip_dot:
            ss.write('.')
        ss.write('corrected_no')
    return ss.getvalue()

def add_rlscope_analyze_flags(cmd, args):
    if args.line_numbers:
        cmd.append('--line-numbers')
    if args.debug:
        cmd.append('--debug')
    if args.pdb:
        cmd.append('--pdb')
    # if args.debug_memoize:
    #     cmd.append('--debug-memoize')
    if args.debug_single_thread:
        cmd.append('--debug-single-thread')

def log_missing_files(self, task, files):
    logger.info(textwrap.dedent("""
            {klass}: SKIP rls-run --task={task}; still need you to collect 
            some additional runs using "rls-quick-expr".
            Files present so far:
            {files}
            """).format(
        klass=self.__class__.__name__,
        task=task,
        files=textwrap.indent(pprint.pformat(files), prefix='  '),
    ))

def get_func_name(obj, func):
    name = "{klass}.{func}".format(
        klass=obj.__class__.__name__,
        func=func)
    return name


class RunCmd:
    def __init__(self, cmd, output_directory, logfile):
        self.cmd = cmd
        self.output_directory = output_directory
        self.logfile = logfile

    def __repr__(self):
        return "{klass}(cmd=\"{cmd}\", output_directory={output_directory}, logfile={logfile})".format(
            klass=self.__class__.__name__,
            cmd=' '.join(self.cmd),
            output_directory=self.output_directory,
            logfile=self.logfile
        )


if __name__ == '__main__':
    main_run()
