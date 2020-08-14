from iml_profiler.profiler.iml_logging import logger
import argparse
import pprint
from glob import glob
import textwrap
import os
import sys
import numpy as np
import pandas as pd
from os import environ as ENV
import json
import functools

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

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
    def __init__(self, directory, cmd,
                 repetitions=1,
                 replace=False,
                 dry_run=False,
                 skip_error=False,
                 debug=False,
                 debug_single_thread=False,
                 # Ignore extra stuff
                 **kwargs,
                 ):
        self.directory = directory
        self.cmd = cmd
        self.repetitions = repetitions
        self.replace = replace
        self.dry_run = dry_run
        self.skip_error = skip_error
        self.debug = debug
        self.debug_single_thread = debug_single_thread
        self._pool = ForkedProcessPool(name='{klass}.pool'.format(
            klass=self.__class__.__name__))
        # 1 min, 2 min, 4 min

    def out_dir(self):
        return self.directory

    def cupti_scaling_overhead_dir(self):
        return _j(
            self.out_dir(),
            "cupti_scaling_overhead")

    def cupti_scaling_overhead_logfile(self):
        task = "CUPTIScalingOverheadTask"
        logfile = _j(
            self.cupti_scaling_overhead_dir(),
            self._logfile_basename(task),
        )
        return logfile

    def cupti_overhead_dir(self):
        return _j(
            self.out_dir(),
            "cupti_overhead")

    def cupti_overhead_logfile(self):
        task = "CUPTIOverheadTask"
        logfile = _j(
            self.cupti_overhead_dir(),
            self._logfile_basename(task),
        )
        return logfile

    def LD_PRELOAD_overhead_dir(self):
        return _j(
            self.out_dir(),
            "LD_PRELOAD_overhead")

    def LD_PRELOAD_overhead_logfile(self):
        task = "CallInterceptionOverheadTask"
        logfile = _j(
            self.LD_PRELOAD_overhead_dir(),
            self._logfile_basename(task),
        )
        return logfile

    def pyprof_overhead_dir(self):
        return _j(
            self.out_dir(),
            "pyprof_overhead")

    def pyprof_overhead_logfile(self):
        task = "PyprofOverheadTask"
        logfile = _j(
            self.pyprof_overhead_dir(),
            self._logfile_basename(task),
        )
        return logfile


    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def _glob_json_files(self, direc):
        json_paths = glob("{direc}/*.json".format(
            direc=direc))
        return json_paths

    def compute_rls_analyze(self, conf, rep):
        task = "RLSAnalyze"

        assert conf.rls_analyze_mode is not None

        # if self.dry_run and (
        #     self.conf(config_suffix, calibration=calibration, dflt=None) is None
        # ):
        #     return
        # conf = self.conf(config_suffix, calibration=calibration)

        directory = conf.out_dir(rep)

        # all_gpu_activities_api_time_directories = []
        # all_interception_directories = []

        # gpu_activities_api_time_directories = self.conf('gpu_activities_api_time', calibration=True).iml_directories()
        # interception_directories = self.conf('interception', calibration=True).iml_directories()
        # if len(gpu_activities_api_time_directories) != len(interception_directories):
        #     log_missing_files(self, task=task, files={
        #         'gpu_activities_api_time_directories': gpu_activities_api_time_directories,
        #         'interception_directories': interception_directories,
        #     })
        #     return
        # all_gpu_activities_api_time_directories.extend(gpu_activities_api_time_directories)
        # all_interception_directories.extend(interception_directories)

        # directory = self.cupti_scaling_overhead_dir()
        # if not self.dry_run:
        #     os.makedirs(directory, exist_ok=True)
        cmd = ['iml-analyze',
               '--task', task,
               '--iml-directory', directory,
               '--mode', conf.rls_analyze_mode,

               '--cupti-overhead-json', _j(self.cupti_overhead_dir(), 'cupti_overhead.json'),
               '--LD-PRELOAD-overhead-json', _j(self.LD_PRELOAD_overhead_dir(), 'LD_PRELOAD_overhead.json'),
               '--python-annotation-json', _j(self.pyprof_overhead_dir(), 'category_events.python_annotation.json'),
               '--python-clib-interception-tensorflow-json', _j(self.pyprof_overhead_dir(), 'category_events.python_clib_interception.json'),
               '--python-clib-interception-simulator-json', _j(self.pyprof_overhead_dir(), 'category_events.python_clib_interception.json'),
               ]
        add_iml_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        # logfile = self.cupti_scaling_overhead_logfile()
        logfile = _j(
            directory,
            self._logfile_basename(task),
        )
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug)


    def compute_cupti_scaling_overhead(self):
        task = "CUPTIScalingOverheadTask"

        if self.dry_run and (
            self.conf('gpu_activities_api_time', calibration=True, dflt=None) is None or
            self.conf('interception', calibration=True, dflt=None) ):
            return

        all_gpu_activities_api_time_directories = []
        all_interception_directories = []

        gpu_activities_api_time_directories = self.conf('gpu_activities_api_time', calibration=True).iml_directories()
        interception_directories = self.conf('interception', calibration=True).iml_directories()
        if len(gpu_activities_api_time_directories) != len(interception_directories):
            log_missing_files(self, task=task, files={
                'gpu_activities_api_time_directories': gpu_activities_api_time_directories,
                'interception_directories': interception_directories,
            })
            return
        all_gpu_activities_api_time_directories.extend(gpu_activities_api_time_directories)
        all_interception_directories.extend(interception_directories)

        directory = self.cupti_scaling_overhead_dir()
        if not self.dry_run:
            os.makedirs(directory, exist_ok=True)
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-api-time-directory', json.dumps(all_gpu_activities_api_time_directories),
               '--interception-directory', json.dumps(all_interception_directories),
               ]
        add_iml_analyze_flags(cmd, self)
        # cmd.extend(self.extra_argv)

        logfile = self.cupti_scaling_overhead_logfile()
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug)

    def compute_cupti_overhead(self):
        task = "CUPTIOverheadTask"

        if self.dry_run and (
            self.conf('gpu_activities', calibration=True, dflt=None) is None or
            self.conf('no_gpu_activities', calibration=True, dflt=None) ):
            return

        gpu_activities_directories = self.conf('gpu_activities', calibration=True).iml_directories()
        no_gpu_activities_directories = self.conf('no_gpu_activities', calibration=True).iml_directories()
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

        directory = self.cupti_overhead_dir()
        if not self.dry_run:
            os.makedirs(directory, exist_ok=True)
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-directory', json.dumps(gpu_activities_directories),
               '--no-gpu-activities-directory', json.dumps(no_gpu_activities_directories),
               ]
        add_iml_analyze_flags(cmd, self)

        logfile = self.cupti_overhead_logfile()
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug)

    def compute_LD_PRELOAD_overhead(self):
        task = "CallInterceptionOverheadTask"

        if self.dry_run and (
            self.conf('interception', calibration=True, dflt=None) is None or
            self.conf('uninstrumented', calibration=True, dflt=None) ):
            return

        interception_directories = self.conf('interception', calibration=True).iml_directories()
        uninstrumented_directories = self.conf('uninstrumented', calibration=True).iml_directories()
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

        directory = self.LD_PRELOAD_overhead_dir()
        if not self.dry_run:
            os.makedirs(directory, exist_ok=True)
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--interception-directory', json.dumps(interception_directories),
               '--uninstrumented-directory', json.dumps(uninstrumented_directories),
               ]
        add_iml_analyze_flags(cmd, self)

        logfile = self.LD_PRELOAD_overhead_logfile()
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug)

    def compute_pyprof_overhead(self):
        task = "PyprofOverheadTask"

        if self.dry_run and (
            self.conf('uninstrumented', calibration=True, dflt=None) is None or
            self.conf('just_pyprof_interceptions', calibration=True, dflt=None) is None or
            self.conf('just_pyprof_annotations', calibration=True, dflt=None) ):
            return

        uninstrumented_directories = self.conf('uninstrumented', calibration=True).iml_directories()
        pyprof_annotations_directories = self.conf('just_pyprof_annotations', calibration=True).iml_directories()
        pyprof_interceptions_directories = self.conf('just_pyprof_interceptions', calibration=True).iml_directories()
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

        directory = self.pyprof_overhead_dir()
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

        logfile = self.pyprof_overhead_logfile()
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.dry_run,
            skip_error=self.skip_error,
            debug=self.debug)

    def do_run(self):

        for rep in range(1, self.repetitions+1):
            for config in self.configs:
                config.run(rep, self.cmd)

        # self.compute_cupti_scaling_overhead()
        # self.compute_cupti_overhead()
        # self.compute_LD_PRELOAD_overhead()
        # self.compute_pyprof_overhead()
        self._pool.submit(
            get_func_name(self, 'compute_cupti_scaling_overhead'),
            self.compute_cupti_scaling_overhead,
            sync=self.debug_single_thread,
        )
        self._pool.submit(
            get_func_name(self, 'compute_cupti_overhead'),
            self.compute_cupti_overhead,
            sync=self.debug_single_thread,
        )
        self._pool.submit(
            get_func_name(self, 'compute_LD_PRELOAD_overhead'),
            self.compute_LD_PRELOAD_overhead,
            sync=self.debug_single_thread,
        )
        self._pool.submit(
            get_func_name(self, 'compute_pyprof_overhead'),
            self.compute_pyprof_overhead,
            sync=self.debug_single_thread,
        )
        self._pool.shutdown()
        # rls-analyze needs the JSON calibration files; wait.
        for rep in self.each_repetition():
            for config in self.configs:
                config.run(rep, self.cmd)
                if config.rls_analyze_mode is not None:
                    self._pool.submit(
                        get_func_name(self, 'compute_rls_analyze'),
                        self.compute_rls_analyze,
                        config, rep,
                        sync=self.debug_single_thread,
                    )
        self._pool.shutdown()

    def each_repetition(self):
        return range(1, self.repetitions+1)

    def conf(self, config_suffix, calibration=False, dflt=SENTINEL):
        calib_config_suffix = self.calibrated_name(config_suffix, calibration=calibration)
        if dflt is SENTINEL:
            return self.config_suffix_to_obj[calib_config_suffix]
        return self.config_suffix_to_obj.get(calib_config_suffix, dflt)

    def calibrated_name(self, config_suffix, calibration=False):
        if not calibration:
            return config_suffix
        return "calibration_{suffix}".format(
            suffix=config_suffix,
        )

    def init_configs(self):
        self.configs = []
        self.config_suffix_to_obj = dict()
        self._add_configs()
        logger.info("Run configuration: {msg}".format(msg=pprint_msg({
            'configs': self.configs,
        })))


    def _add_configs(self):

        def add_calibration_config(calibration=True, **common_kwargs):
            config_kwargs = dict(common_kwargs)
            config = ExprSubtractionValidationConfig(**config_kwargs)
            if not calibration:
                self.configs.append(config)
                return

            config_suffix = self.calibrated_name(config.config_suffix, calibration=calibration)
            calibration_config_kwargs = dict(common_kwargs)
            calibration_config_kwargs.update(dict(
                # Disable tfprof: CUPTI and LD_PRELOAD.
                config_suffix=config_suffix,
            ))
            calibration_config = ExprSubtractionValidationConfig(**calibration_config_kwargs)
            assert calibration_config.is_calibration
            self.configs.append(calibration_config)

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
        # config = ExprSubtractionValidationConfig(
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
        #     config = ExprSubtractionValidationConfig(
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
        #     config = ExprSubtractionValidationConfig(
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
        #     config = ExprSubtractionValidationConfig(
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

        for config in self.configs:
            assert config.config_suffix not in self.config_suffix_to_obj
            self.config_suffix_to_obj[config.config_suffix] = config


    def run(self):
        # parser.add_argument(
        #     '--calibration-mode',
        #     default='calibration',
        #     choices=['calibration', 'validation'],
        #     help=textwrap.dedent("""
        #     calibration:
        #         Only run configurations needed to subtract overhead; i.e.
        #
        #         PyprofOverheadTask
        #             uninstrumented
        #             just_pyprof_annotations
        #             just_pyprof_interceptions
        #
        #         CUPTIOverheadTask
        #             gpu_activities
        #             no_gpu_activities
        #
        #         CUPTIScalingOverheadTask
        #             gpu_activities_api_time
        #             interception
        #
        #     validation:
        #         Run additional configurations for "isolating" bugs in overhead correction.
        #         For example, run with just python annotations enabled so we can see if correcting for python annotations in isolation works.
        #     """),
        # )

        self.init_configs()

        self.do_run()
        logger.info("Success! Calibration files have been generated @ {direc}/*_overhead".format(
            direc=self.directory,
        ))


class ExprSubtractionValidationConfig:
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

    def out_dir(self, rep):
        return _j(
            self.expr.out_dir(),
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

    def logfile(self, rep):
        logfile = _j(self.out_dir(rep), "logfile.out")
        return logfile

    def run(self, rep, cmd):
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
            '--iml-directory', _a(self.out_dir(rep)),
            '--iml-training-progress',
            '--iml-delay',
        ])

        # [
        #     # '--iml-training-progress',
        #     # '--iml-max-timesteps', iters,
        #     'python', 'train.py',
        #     '--algo', self.algo,
        #     '--env', self.env,
        #     '--log-folder', _j(ENV['RL_BASELINES_ZOO_DIR'], 'output'),
        #     '--log-interval', '1',
        # ]
        iml_prof_cmd.extend(self.script_args)
        logfile = self.logfile(rep)
        # logger.info("Logging to file {path}".format(
        #     path=logfile))
        expr_run_cmd(
            cmd=iml_prof_cmd,
            to_file=logfile,
            cwd=ENV['RL_BASELINES_ZOO_DIR'],
            replace=self.expr.replace,
            dry_run=self.expr.dry_run,
            skip_error=self.expr.skip_error,
            debug=self.expr.debug)

    def already_ran(self, rep):
        logfile = self.logfile(rep)
        return expr_already_ran(logfile, debug=self.expr.debug)

    def iml_directories(self):
        """
        Return all --iml-directories whose runs are completed.
        """
        iml_directories = []
        # for rep in range(1, self.expr.repetitions+1):
        for rep in self.expr.each_repetition():
            if not self.already_ran(rep):
                continue
            iml_directory = self.out_dir(rep)
            iml_directories.append(iml_directory)
        return iml_directories


def error(msg, parser=None):
    if parser is not None:
        parser.print_usage()
    logger.error(msg)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser("Calibrate for profiling overhead")
    parser.add_argument("--iml-directory",
                        required=True,
                        help=textwrap.dedent("""
                        Root directory for output
                        """))
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
    parser.add_argument("--debug-single-thread",
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug
                        """))
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
    parser.add_argument("--skip-error",
                        action='store_true',
                        help=textwrap.dedent("""
                        Skip errors 
                        """))


    args, extra_argv = parser.parse_known_args()
    cmd = extra_argv

    if len(cmd) == 0:
        error("Expected cmd to run with iml-prof for calibration, but non was provided",
              parser=parser)

    if shutil.which(cmd[0]) is None:
        error("Couldn't find {exec} on PATH".format(
            exec=cmd[0]), parser=parser)

    try:
        args_dict = dict(vars(args))
        directory = args_dict.pop('iml_directory')
        repetitions = args_dict.pop('iml_repetitions')
        obj = Calibration(
            directory=directory,
            cmd=cmd,
            repetitions=repetitions,
            **args_dict,
            # iml_directory=args.iml_directory,
            # debug=args.debug,
            # debug_single_thread=args.debug_single_thread,
            # cmd=cmd,
        )
        obj.run()
    except Exception as e:
        if not args.pdb:
            raise
        logger.debug("> IML: Detected exception:")
        logger.error("{Klass}: {msg}".format(
            Klass=type(e).__name__,
            msg=str(e),
        ))
        logger.debug("> Entering pdb:")
        # Fails sometimes, not sure why.
        # import ipdb
        # ipdb.post_mortem()
        import pdb
        pdb.post_mortem()
        raise

def rep_suffix(rep):
    assert rep is not None
    return "_repetition_{rep:02}".format(rep=rep)

def add_iml_analyze_flags(cmd, args):
    if args.debug:
        cmd.append('--debug')
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

if __name__ == '__main__':
    main()
