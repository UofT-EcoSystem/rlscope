from iml_profiler.profiler.iml_logging import logger
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

from iml_profiler.profiler.util import print_cmd
from iml_profiler.profiler.util import run_with_pdb, pprint_msg
from iml_profiler.parser.common import *
from iml_profiler.experiment.util import tee, expr_run_cmd, expr_already_ran
from iml_profiler.profiler.concurrent import ForkedProcessPool
from iml_profiler.scripts import bench
from iml_profiler.experiment import expr_config
from iml_profiler.parser.dataframe import IMLConfig
from iml_profiler.parser.profiling_overhead import \
    parse_microbench_overhead_js, \
    DataframeMapper, \
    PyprofDataframeReader, \
    PyprofOverheadParser, \
    MicrobenchmarkOverheadJSON, \
    CalibrationJSONs

"""
cmd: command to run
iml_directory: directory to output results of calibration runs, AND regular run.

Run each configuration needed to calibrate for overhead correction.
Output results to:
--iml_directory/
  config_calibration_{short_name}/
    Various calibration runs for computing overhead corrections for 
    the config_time_breakdown run. 
  config_time_breakdown/
    Time breakdown run for collecting the CPU/GPU time.
  config_gpu_hw/
    GPU HW run for collecting GPU HW counters.
    
NOTE: 
(1) first, run each configuration sequentially
(2) then, run "rls-analyze" in parallel for each configuration.
    Run "--mode=overlap" for all configuration except config_gpu_hw.
    Run "--mode=gpu_hw" for config_gpu_hw.
(3) Run the iml-analyze program that computes configuration json files; 
    output them to the root directory.

Q: What's the best way to detect algo and env...?
A: Add --iml-algo and --iml-env and record it in a file somewhere (iml_config.json).
"""

SENTINEL = object()

class Calibration:
    def __init__(self,
                 repetitions=1,
                 replace=False,
                 re_calibrate=False,
                 re_plot=False,
                 dry_run=False,
                 skip_plot=False,
                 skip_error=False,
                 max_workers=None,
                 parallel_runs=False,
                 gpus=None,
                 debug=False,
                 debug_single_thread=False,
                 pdb=False,
                 # Ignore extra stuff
                 **kwargs,
                 ):
        self.repetitions = repetitions
        self.replace = replace
        self.re_calibrate = re_calibrate
        self.re_plot = re_plot
        self.dry_run = dry_run
        self.skip_plot = skip_plot
        self.skip_error = skip_error
        self.max_workers = max_workers
        self.parallel_runs = parallel_runs
        self.gpus = gpus
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self.pdb = pdb
        self._pool = ForkedProcessPool(name='{klass}.pool'.format(
            klass=self.__class__.__name__),
            max_workers=self.max_workers)

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

    def compute_gpu_hw_plot(self, directories, output_directory, extra_argv=None,
                            # xtick_expression=None
                            **kwargs,
                            ):
        task = "GpuHwPlotTask"

        repetitions = None
        # repetitions = [1]

        for iml_directory in directories:
            if self.dry_run and (
                self.conf(iml_directory, 'gpu_hw', calibration=True, dflt=None) is None
            ):
                return

        iml_dirs = []
        for iml_directory in directories:
            iml_dirs.extend(self.conf(iml_directory, 'gpu_hw', calibration=False).iml_directories(iml_directory, repetitions=repetitions))

        # Stick plots in root directory.
        if not self.dry_run:
            os.makedirs(output_directory, exist_ok=True)
        cmd = ['iml-analyze',
               '--iml-directory', output_directory,
               '--task', task,
               '--iml-directories', json.dumps(iml_dirs),
               ]
        if extra_argv is not None:
            cmd.extend(extra_argv)
        # if xtick_expression is not None:
        #     cmd.extend(['--xtick-expression', xtick_expression])
        # self._add_calibration_opts(output_directory, cmd)
        add_iml_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.compute_gpu_hw_plot_logfile(output_directory)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            only_show_env=self.only_show_env())

    def compute_time_breakdown_plot(self, directories, output_directory, extra_argv,
                                    # xtick_expression=None,
                                    # Ignore
                                    **kwargs):
        task = "OverlapStackedBarTask"

        # cmd = [
        #     'iml-analyze',
        # ]
        # cmd.extend([
        #     '--task', 'OverlapStackedBarTask',
        # ])

        # TODO: add support for multiple repetitions for time breakdown plot
        repetitions = None
        # repetitions = [1]

        for iml_directory in directories:
            if self.dry_run and (
                self.conf(iml_directory, 'uninstrumented', calibration=True, dflt=None) is None or
                self.conf(iml_directory, 'time_breakdown', calibration=False, dflt=None) ):
                return

        unins_iml_dirs = []
        iml_dirs = []
        for iml_directory in directories:
            unins_iml_dirs.extend(self.conf(iml_directory, 'uninstrumented', calibration=True).iml_directories(iml_directory, repetitions=repetitions))
            iml_dirs.extend(self.conf(iml_directory, 'time_breakdown', calibration=False).iml_directories(iml_directory, repetitions=repetitions))
            if len(iml_dirs) != len(unins_iml_dirs):
                log_missing_files(self, task=task, files={
                    'iml_dirs': iml_dirs,
                    'unins_iml_dirs': unins_iml_dirs,
                })
                return

        # Stick plots in root directory.
        if not self.dry_run:
            os.makedirs(output_directory, exist_ok=True)
        overlap_type = 'CategoryOverlap'
        cmd = ['iml-analyze',
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

               '--iml-directories', json.dumps(iml_dirs),
               '--unins-iml-directories', json.dumps(unins_iml_dirs),
               ]
        if extra_argv is not None:
            cmd.extend(extra_argv)
        # if xtick_expression is not None:
        #     cmd.extend(['--xtick-expression', xtick_expression])
        self._add_calibration_opts(output_directory, cmd)
        add_iml_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.compute_time_breakdown_plot_logfile(output_directory)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
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
            self.cupti_overhead_json(output_directory),
            self.LD_PRELOAD_overhead_json(output_directory),
            self.python_annotation_json(output_directory),
            self.python_clib_interception_tensorflow_json(output_directory),
            self.python_clib_interception_simulator_json(output_directory),
        ])

    def _calibration_logfiles(self, output_directory):
        logfiles = set()
        json_paths = self._calibration_paths(output_directory)
        for json_path in json_paths:
            calibration_dir = _d(json_path)
            for logfile in glob("{dir}/*.logfile.out".format(dir=calibration_dir)):
                logfiles.add(logfile)
        return logfiles

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

    def rls_analyze_logfile(self, iml_directory, conf, rep):
        task = "RLSAnalyze"
        directory = conf.out_dir(iml_directory, rep)
        logfile = _j(
            directory,
            self._logfile_basename(task),
        )
        return logfile

    def rls_analyze_output_paths(self, iml_directory, conf, rep):
        directory = conf.out_dir(iml_directory, rep)
        paths = set()
        # ${directory}/*.venn_js.json
        # ${directory}/OverlapResult.*
        # ${directory}/RLSAnalyze.*
        # ${directory}/iml_profiler_plot_index*
        # ${directory}/__pycache__
        def _add_paths(glob_pattern):
            for path in glob("{dir}/{glob}".format(dir=directory, glob=glob_pattern)):
                paths.add(path)
        # --config time-breakdown
        _add_paths("*.venn_js.json")
        _add_paths("OverlapResult.*")
        _add_paths("RLSAnalyze.*")
        _add_paths("iml_profiler_plot_index*")
        _add_paths("__pycache__")
        # --config gpu-hw
        _add_paths("GPUHwCounterSampler.csv")
        return paths

    def time_breakdown_plot_paths(self, iml_directory):
        paths = set()
        def _add_paths(glob_pattern):
            for path in glob("{dir}/{glob}".format(dir=iml_directory, glob=glob_pattern)):
                paths.add(path)
        # --config time-breakdown
        _add_paths("OverlapStackedBar*")
        return paths

    def gpu_hw_plot_paths(self, iml_directory):
        paths = set()
        def _add_paths(glob_pattern):
            for path in glob("{dir}/{glob}".format(dir=iml_directory, glob=glob_pattern)):
                paths.add(path)
        # --config gpu-hw
        _add_paths("GpuHwPlot*")
        _add_paths("rlscope_*dataframe*")
        _add_paths("rlscope_*csv")
        _add_paths("rlscope_*svg")
        _add_paths("rlscope_*png")
        _add_paths("rlscope_*pdf")
        return paths

    def compute_rls_analyze(self, iml_directory, output_directory, conf, rep):
        task = "RLSAnalyze"

        assert conf.rls_analyze_mode is not None

        directory = conf.out_dir(iml_directory, rep)

        cmd = ['iml-analyze',
               '--task', task,
               '--iml-directory', directory,
               '--mode', conf.rls_analyze_mode,
               ]
        self._add_calibration_opts(iml_directory, cmd)
        add_iml_analyze_flags(cmd, self)

        logfile = self.rls_analyze_logfile(iml_directory, conf, rep)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            only_show_env=self.only_show_env())

    def only_show_env(self):
        if self.debug:
            # Show all enviroment variables
            return None
        # Show no environments variables
        return {'CUDA_VISIBLE_DEVICES'}

    def compute_cupti_scaling_overhead(self, output_directory):
        task = "CUPTIScalingOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'gpu_activities_api_time', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'interception', calibration=True, dflt=None) ):
            return

        all_gpu_activities_api_time_directories = []
        all_interception_directories = []

        gpu_activities_api_time_directories = self.conf(output_directory, 'gpu_activities_api_time', calibration=True).iml_directories(output_directory)
        interception_directories = self.conf(output_directory, 'interception', calibration=True).iml_directories(output_directory)
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
        cmd = ['iml-analyze',
               '--directory', output_directory,
               '--task', task,
               '--gpu-activities-api-time-directory', json.dumps(all_gpu_activities_api_time_directories),
               '--interception-directory', json.dumps(all_interception_directories),
               ]
        add_iml_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.cupti_scaling_overhead_logfile(output_directory)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            only_show_env=self.only_show_env())

    def compute_cupti_overhead(self, output_directory):
        task = "CUPTIOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'gpu_activities', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'no_gpu_activities', calibration=True, dflt=None) ):
            return

        gpu_activities_directories = self.conf(output_directory, 'gpu_activities', calibration=True).iml_directories(output_directory)
        no_gpu_activities_directories = self.conf(output_directory, 'no_gpu_activities', calibration=True).iml_directories(output_directory)
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
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-directory', json.dumps(gpu_activities_directories),
               '--no-gpu-activities-directory', json.dumps(no_gpu_activities_directories),
               ]
        add_iml_analyze_flags(cmd, self)

        logfile = self.cupti_overhead_logfile(output_directory)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            only_show_env=self.only_show_env())

    def compute_LD_PRELOAD_overhead(self, output_directory):
        task = "CallInterceptionOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'interception', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'uninstrumented', calibration=True, dflt=None) ):
            return

        interception_directories = self.conf(output_directory, 'interception', calibration=True).iml_directories(output_directory)
        uninstrumented_directories = self.conf(output_directory, 'uninstrumented', calibration=True).iml_directories(output_directory)
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
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--interception-directory', json.dumps(interception_directories),
               '--uninstrumented-directory', json.dumps(uninstrumented_directories),
               ]
        add_iml_analyze_flags(cmd, self)

        logfile = self.LD_PRELOAD_overhead_logfile(output_directory)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            only_show_env=self.only_show_env())

    def compute_pyprof_overhead(self, output_directory):
        task = "PyprofOverheadTask"

        if self.dry_run and (
            self.conf(output_directory, 'uninstrumented', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'just_pyprof_interceptions', calibration=True, dflt=None) is None or
            self.conf(output_directory, 'just_pyprof_annotations', calibration=True, dflt=None) ):
            return

        uninstrumented_directories = self.conf(output_directory, 'uninstrumented', calibration=True).iml_directories(output_directory)
        pyprof_annotations_directories = self.conf(output_directory, 'just_pyprof_annotations', calibration=True).iml_directories(output_directory)
        pyprof_interceptions_directories = self.conf(output_directory, 'just_pyprof_interceptions', calibration=True).iml_directories(output_directory)
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
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--uninstrumented-directory', json.dumps(uninstrumented_directories),
               '--pyprof-annotations-directory', json.dumps(pyprof_annotations_directories),
               '--pyprof-interceptions-directory', json.dumps(pyprof_interceptions_directories),
               ]
        add_iml_analyze_flags(cmd, self)

        logfile = self.pyprof_overhead_logfile(output_directory)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug,
            only_show_env=self.only_show_env())

    def each_config_repetition(self):
        for rep in range(1, self.repetitions+1):
            for config in self.configs:
                yield rep, config

    def run_configs(self, cmd, output_directory):
        # import pdb; pdb.set_trace()
        if not self.parallel_runs:
            # Run configurations serially on which GPU(s) are visible.
            for rep, config in self.each_config_repetition():
                config.run(rep, cmd, output_directory)
            return

        # Parallelize running configurations across GPUs on this machine (assume no CPU interference).
        # Record commands in run_expr.sh, and run:
        # $ iml-run-expr --run-sh --sh run_expr.sh
        run_expr_sh = self._run_expr_sh(output_directory)
        logger.info(f"Writing configuration shell commands to {run_expr_sh}")
        os.makedirs(_d(run_expr_sh), exist_ok=True)
        with open(run_expr_sh, 'w') as f:
            for rep, config in self.each_config_repetition():
                run_cmd = config.run_cmd(rep, cmd, output_directory)
                full_cmd = run_cmd.cmd + ['--iml-logfile', run_cmd.logfile]
                quoted_cmd = [shlex.quote(opt) for opt in full_cmd]
                f.write(' '.join(quoted_cmd))
                f.write('\n')
        run_expr_cmd = ['iml-run-expr', '--run-sh', '--sh', run_expr_sh]
        if self.dry_run:
            run_expr_cmd.append('--dry-run')
        if self.debug:
            run_expr_cmd.append('--debug')
        # NOTE: don't forward pdb since we cannot interact with parallel processes.
        print_cmd(run_expr_cmd)
        proc = subprocess.run(run_expr_cmd, check=False)
        retcode = proc.returncode
        if retcode != 0:
            logger.error("Failed to run configurations in parallel; at least one configuration exited with non-zero exit status.")
            sys.exit(1)


    def _run_expr_sh(self, output_directory):
        return _j(output_directory, 'run_expr.sh')

    def do_run(self, cmd, output_directory):

        self.run_configs(cmd, output_directory)

        # Lets save calibration to happen during iml-plot to make it easier to split up "analysis" from
        # "calibration" time.
        # unless they give --iml-plot
        if not self.skip_plot:
            self.do_calibration(output_directory)
        else:
            logger.info("--iml-skip-plot: SKIP processing results into plots; run iml-plot to do this later.")

    def _rm_path(self, path):
        if _e(path):
            logger.info("--re-calibrate: RM {path}".format(path=path))
            if not self.dry_run:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    def clean_plots(self, output_directory):

        for path in self.time_breakdown_plot_paths(output_directory):
            self._rm_path(path)

        for path in self.gpu_hw_plot_paths(output_directory):
            self._rm_path(path)

    def clean_analysis(self, output_directory):
        for path in self._calibration_paths(output_directory):
            self._rm_path(path)
        for path in self._calibration_logfiles(output_directory):
            self._rm_path(path)

        # Need to re-run rls-analyze using new calibration files.
        for rep in range(1, self.repetitions+1):
            for config in self.configs:
                if config.rls_analyze_mode is not None:
                    for path in self.rls_analyze_output_paths(output_directory, config, rep):
                        self._rm_path(path)


    def do_calibration(self, output_directory, sync=True):
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
            self._pool.shutdown()

    def do_plot(self, directories, output_directory, extra_argv=None, **kwargs):
        for iml_directory in directories:
            self.do_calibration(iml_directory, sync=False)
        self._pool.shutdown()
        # rls-analyze needs the JSON calibration files; wait.
        for iml_directory in directories:
            for rep in self.each_repetition():
                for config in self.configs:
                    if config.rls_analyze_mode is not None:
                        self._pool.submit(
                            get_func_name(self, 'compute_rls_analyze'),
                            self.compute_rls_analyze,
                            iml_directory, output_directory,
                            config, rep,
                            sync=self.debug_single_thread,
                        )
        self._pool.shutdown()
        # Plotting requires running rls-analyze
        self._pool.submit(
            get_func_name(self, 'compute_time_breakdown_plot'),
            self.compute_time_breakdown_plot,
            directories, output_directory, extra_argv,
            sync=self.debug_single_thread,
            **kwargs,
        )
        self._pool.submit(
            get_func_name(self, 'compute_gpu_hw_plot'),
            self.compute_gpu_hw_plot,
            directories, output_directory, extra_argv,
            sync=self.debug_single_thread,
            **kwargs,
        )
        self._pool.shutdown()

    def each_repetition(self):
        return range(1, self.repetitions+1)

    def conf(self, output_dir, config_suffix, calibration=False, dflt=SENTINEL):
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

    def init_configs(self, directories):
        self.configs = []
        self.config_map = dict()
        for directory in directories:
            self._add_configs(directory)
        logger.info("Run configurations: {msg}".format(msg=pprint_msg({
            'configs': self.configs,
        })))


    def _add_configs(self, output_dir):

        def add_calibration_config(calibration=True, **common_kwargs):
            config_kwargs = dict(common_kwargs)
            base_config = RLScopeConfig(**config_kwargs)
            if not calibration:
                config = base_config
            else:
                config_suffix = self.calibrated_name(base_config.config_suffix, calibration=calibration)
                calibration_config_kwargs = dict(common_kwargs)
                calibration_config_kwargs.update(dict(
                    # Disable tfprof: CUPTI and LD_PRELOAD.
                    config_suffix=config_suffix,
                ))
                config = RLScopeConfig(**calibration_config_kwargs)
                assert config.is_calibration

            self.configs.append(config)
            if output_dir not in self.config_map:
                self.config_map[output_dir] = dict()
            assert config.config_suffix not in self.config_map[output_dir]
            self.config_map[output_dir][config.config_suffix] = config

        add_calibration_config(
            expr=self,
            iml_prof_config='time-breakdown',
            config_suffix='time_breakdown',
            script_args=[],
            calibration=False,
            rls_analyze_mode='overlap',
        )

        add_calibration_config(
            expr=self,
            iml_prof_config='gpu-hw',
            config_suffix='gpu_hw',
            script_args=[],
            calibration=False,
            rls_analyze_mode='gpu_hw',
        )

        # Entirely uninstrumented configuration; we use this in many of the overhead calculations to determine
        # how much training time is attributable to the enabled "feature" (e.g. CUPTI activities).
        add_calibration_config(
            expr=self,
            iml_prof_config='uninstrumented',
            config_suffix='uninstrumented',
            # Disable ALL pyprof/tfprof stuff.
            script_args=['--iml-disable'],
        )
        # Entirely uninstrumented configuration (CALIBRATION runs);
        # NOTE: we need to re-run the uninstrumented configuration for the calibration runs, so that we can
        # see how well our calibration generalizes.
        # self.config_uninstrumented = config

        add_calibration_config(
            expr=self,
            iml_prof_config='interception',
            config_suffix='interception',
            script_args=['--iml-disable-pyprof'],
        )

        # CUPTIOverheadTask: CUPTI, and CUDA API stat-tracking overhead correction.
        add_calibration_config(
            expr=self,
            iml_prof_config='gpu-activities',
            config_suffix='gpu_activities',
            script_args=['--iml-disable-pyprof'],
        )
        add_calibration_config(
            expr=self,
            iml_prof_config='no-gpu-activities',
            config_suffix='no_gpu_activities',
            script_args=['--iml-disable-pyprof'],
        )
        # # CUPTIScalingOverheadTask:
        # config = RLScopeConfig(
        #     expr=self,
        #     iml_prof_config='gpu-activities-api-time',
        #     config_suffix='gpu_activities_api_time_calibration',
        #     script_args=['--iml-disable-pyprof'],
        # )
        # self.configs.append(config)
        add_calibration_config(
            expr=self,
            iml_prof_config='gpu-activities-api-time',
            config_suffix='gpu_activities_api_time',
            script_args=['--iml-disable-pyprof'],
        )

        # if self.calibration_mode == 'validation':
        #     # Evaluate: combined tfprof/pyprof overhead correction.
        #     # (i.e. full IML trace-collection).
        #     config = RLScopeConfig(
        #         expr=self,
        #         iml_prof_config='full',
        #         # Enable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix='full',
        #         # Enable pyprof.
        #         script_args=[],
        #         long_run=True,
        #     )
        #     self.configs.append(config)

        # if self.calibration_mode == 'validation':
        #     # Evaluate: tfprof overhead correction in isolation.
        #     config = RLScopeConfig(
        #         expr=self,
        #         iml_prof_config='full',
        #         # Enable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix='just_tfprof',
        #         # DON'T enable pyprof.
        #         script_args=['--iml-disable-pyprof'],
        #         long_run=True,
        #     )
        #     self.configs.append(config)

        # if self.calibration_mode == 'validation':
        #     # Evaluate: pyprof overhead correction in isolation.
        #     config = RLScopeConfig(
        #         expr=self,
        #         iml_prof_config='uninstrumented',
        #         # Disable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix='just_pyprof',
        #         # Enable pyprof.
        #         script_args=['--iml-disable-tfprof'],
        #     )
        #     self.configs.append(config)

        # PyprofOverheadTask: Python->C-lib event tracing, and operation annotation overhead correction.
        add_calibration_config(
            expr=self,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_annotations',
            # Only enable GPU/C-lib event collection, not operation annotations.
            script_args=['--iml-disable-tfprof', '--iml-disable-pyprof-interceptions'],
        )
        add_calibration_config(
            expr=self,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_interceptions',
            # Only enable operation annotations, not GPU/C-lib event collection.
            script_args=['--iml-disable-tfprof', '--iml-disable-pyprof-annotations'],
        )

    def maybe_clean(self, directory, skip_analysis=False):
        # PROBLEM: is this going to run twice in the iml-plot call?
        if not skip_analysis and self.re_calibrate:
            self.clean_analysis(directory)
        if self.re_calibrate or self.re_plot:
            self.clean_plots(directory)

    def mode_run(self, cmd, directory):
        self.init_configs([directory])

        self.maybe_clean(directory)

        self.do_run(cmd, directory)
        self.do_plot(directories=[directory], output_directory=directory)
        logger.info("Success! Calibration files have been generated @ {direc}/*_overhead".format(
            direc=directory,
        ))

    def mode_plot(self, directories, output_directory, extra_argv=None, **kwargs):
        self.init_configs(directories)

        # Output directory just contains plots; no need to delete analysis files.
        self.maybe_clean(output_directory, skip_analysis=True)
        if self.re_calibrate:
            # Only delete plots from --iml-directories if we're re-calibrating everything.
            for iml_directory in directories:
                self.maybe_clean(iml_directory)

        # self.do_run(cmd, directory)
        self.do_plot(directories=directories, output_directory=output_directory, extra_argv=extra_argv, **kwargs)
        logger.info("Success! Plots output @ {direc}".format(
            direc=output_directory,
        ))

    # def run(self):
    #     self.init_configs()
    #
    #     self.do_run()
    #     self.do_plot()


class RLScopeConfig:
    def __init__(self, expr, iml_prof_config, config_suffix, rls_analyze_mode=None, script_args=[]):
        self.expr = expr
        # $ iml-prof --config ${iml_prof_config}
        self.iml_prof_config = iml_prof_config
        # $ python train.py --iml-directory config_${config_suffix}
        self.config_suffix = config_suffix
        self.script_args = script_args
        self.rls_analyze_mode = rls_analyze_mode

    @property
    def is_calibration(self):
        return re.search(r'calibration', self.config_suffix)

    def out_dir(self, output_directory, rep):
        return _j(
            output_directory,
            "config_{config_suffix}{rep}".format(
                config_suffix=self.config_suffix,
                rep=rep_suffix(rep),
            ))

    def to_string(self):
        return ("{klass}("
                "iml_prof_config='{iml_prof_config}'"
                ", config_suffix='{config_suffix}'"
                ")").format(
            klass=self.__class__.__name__,
            iml_prof_config=self.iml_prof_config,
            config_suffix=self.config_suffix,
        )

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def logfile(self, output_directory, rep):
        logfile = _j(self.out_dir(output_directory, rep), "logfile.out")
        return logfile

    def run_cmd(self, rep, cmd, output_directory):
        iml_prof_cmd = [
            'iml-prof',
            '--config', self.iml_prof_config,
        ]
        iml_prof_cmd.extend(cmd)
        iml_prof_cmd.extend([
            # IMPORTANT: When we run with "--config uninstrumented" during calibrations runs, we STILL want to
            # keep "python interceptions" and "python annotations" enabled, so we can measure their overhead in
            # in isolation!
            '--iml-calibration',
            '--iml-directory', _a(self.out_dir(output_directory, rep)),
            '--iml-training-progress',
            '--iml-delay',
        ])

        iml_prof_cmd.extend(self.script_args)
        logfile = self.logfile(output_directory, rep)
        run_cmd = RunCmd(cmd=iml_prof_cmd, output_directory=output_directory, logfile=logfile)
        return run_cmd

    def run(self, rep, cmd, output_directory):
        run_cmd = self.run_cmd(rep, cmd, output_directory)
        expr_run_cmd(
            cmd=run_cmd.cmd,
            to_file=run_cmd.logfile,
            # cwd=ENV['RL_BASELINES_ZOO_DIR'],
            replace=self.expr.replace,
            dry_run=self.expr.dry_run,
            skip_error=self.expr.skip_error,
            debug=self.expr.debug,
            only_show_env=self.expr.only_show_env())

    def already_ran(self, output_directory, rep):
        logfile = self.logfile(output_directory, rep)
        return expr_already_ran(logfile, debug=self.expr.debug)

    def iml_directories(self, output_directory, repetitions=None):
        """
        Return all --iml-directories whose runs are completed.
        """
        iml_directories = []
        # for rep in range(1, self.expr.repetitions+1):
        if repetitions is None:
            repetitions = self.expr.each_repetition()
        for rep in repetitions:
            if not self.already_ran(output_directory, rep):
                continue
            iml_directory = self.out_dir(output_directory, rep)
            iml_directories.append(iml_directory)
        return iml_directories


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
    parser = argparse.ArgumentParser(
        description="Run RLScope calibrated for profiling overhead, and create plots from multiple workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(
        title="Subcommands",
        description="Run RLScope in different modes (run configurations, plot results).",
    )
    def add_common_arguments(parser):
        # parser.add_argument("--output-directory",
        #                     required=True,
        #                     help=textwrap.dedent("""
        #                     Root directory for output
        #                     """))
        parser.add_argument("--pdb",
                            action='store_true',
                            help=textwrap.dedent("""
                            Debug
                            """))
        parser.add_argument("--debug",
                            action='store_true',
                            help=textwrap.dedent("""
                            Debug
                            """))
        parser.add_argument("--max-workers",
                            default=multiprocessing.cpu_count(),
                            help=textwrap.dedent("""
                            Number of parallel rls-analysis jobs to run at one time.
                            Default: number of CPU cores.
                            """))
        parser.add_argument("--debug-single-thread",
                            action='store_true',
                            help=textwrap.dedent("""
                            Debug
                            """))
        # parser.add_argument("--sh",
        #                     help=textwrap.dedent("""
        #                     Shell file to append commands to (see --append).
        #                     """))
        parser.add_argument('--iml-repetitions',
                            type=int,
                            default=1,
                            help=textwrap.dedent("""
                            Repetitions
                            """))
        parser.add_argument("--replace",
                            action='store_true',
                            help=textwrap.dedent("""
                            Replace
                            """))
        parser.add_argument("--dry-run",
                            action='store_true',
                            help=textwrap.dedent("""
                            Dry run
                            """))
        parser.add_argument("--re-calibrate",
                            action='store_true',
                            help=textwrap.dedent("""
                            Remove existing profiling overhead calibration files, and recompute them.
                            """))
        parser.add_argument("--re-plot",
                            action='store_true',
                            help=textwrap.dedent("""
                            Remove existing plots and remake them (NOTE: doesn't recompute analysis; see --re-calibrate).
                            """))
        parser.add_argument("--skip-error",
                            action='store_true',
                            help=textwrap.dedent("""
                            Skip errors 
                            """))

    run_parser = subparsers.add_parser('run', description="Run <cmd> with profiling overhead calibration")
    add_common_arguments(run_parser)
    run_parser.add_argument("--iml-directory",
                            required=True,
                            help=textwrap.dedent("""
                        Root directory for output
                        """))
    run_parser.add_argument("--iml-skip-plot",
                        action='store_true',
                        help=textwrap.dedent("""
                            After running configurations, DON'T run analysis to output plots for individual workload.
                            """))
    run_parser.add_argument("--gpus",
                        help=textwrap.dedent("""
                            GPUs to run with for --parallel-runs
                            """))
    run_parser.add_argument("--parallel-runs",
                        action='store_true',
                        help=textwrap.dedent("""
                            Parallelize running configurations across GPUs on this machine (assume no CPU inteference). See --iml-gpus
                            """))
    run_parser.set_defaults(**{'mode': 'run'})

    plot_parser = subparsers.add_parser('plot', description="Plot multiple workloads")
    add_common_arguments(plot_parser)
    plot_parser.add_argument("--iml-directories",
                            required=True,
                            nargs='+',
                            # default=[],
                            help=textwrap.dedent("""
                        Directories to plot results from.
                        """))
    plot_parser.add_argument("--output-directory",
                            required=True,
                            help=textwrap.dedent("""
                        Where to output plots.
                        """))
    plot_parser.set_defaults(**{'mode': 'plot'})

    # parser.add_argument("--mode",
    #                     choices=['run', 'plot'],
    #                     default=default_mode,
    #                     help=textwrap.dedent("""
    #                         Debug
    #                         """))

    args, extra_argv = parser.parse_known_args(argv)
    cmd = extra_argv

    args_dict = dict(vars(args))
    repetitions = args_dict.pop('iml_repetitions')
    # dry_run = args_dict.pop('iml_dry_run')
    # re_calibrate = args_dict.pop('iml_re_calibrate')
    # re_plot = args_dict.pop('iml_re_plot')
    skip_plot = args_dict.pop('iml_skip_plot', False)
    # parallel_runs = args_dict.pop('iml_parallel_runs')
    # gpus = args_dict.pop('iml_gpus')
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
            error("Expected cmd to run with iml-prof for calibration, but non was provided",
                  parser=parser)

        if shutil.which(cmd[0]) is None:
            error("Couldn't find {exec} on PATH".format(
                exec=cmd[0]), parser=parser)

        def _run():
            directory = args_dict['iml_directory']
            obj.mode_run(cmd, directory)
        run_with_pdb(args, _run)
    elif args.mode == 'plot':
        # if len(cmd) != 0:
        #     error(
        #         textwrap.dedent("""\
        #         Not sure how to parse extra arguments for "iml-calibrate plot":
        #           {cmd}
        #         Did you intend to run "iml-calibrate run" instead?
        #         """).format(
        #             cmd=' '.join(cmd),
        #         ).rstrip(), parser=parser)

        def _plot():
            directories = args_dict.pop('iml_directories')
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
    return "_repetition_{rep:02}".format(rep=rep)

def add_iml_analyze_flags(cmd, args):
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
            {klass}: SKIP iml-analyze --task={task}; still need you to collect 
            some additional runs using "iml-quick-expr".
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
