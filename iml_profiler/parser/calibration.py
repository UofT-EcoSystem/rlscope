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
"""

class Calibration:
    def __init__(self, iml_directory, cmd, debug=False):
        self.iml_directory = iml_directory
        self.cmd = cmd
        self.debug = debug
        self._pool = ForkedProcessPool(name='{klass}.pool'.format(
            klass=self.__class__.__name__))
        # 1 min, 2 min, 4 min

    def out_dir(self, algo, env):
        return _j(self.iml_directory, algo, env)

    def plot_dir(self, config, iters):
        return _j(
            self.out_dir(config.algo, config.env),
            "config_{conf}_iters_{iters}".format(
                conf=config.config_suffix,
                iters=iters,
            ))

    def plot_logfile(self, config, iters):
        logfile = _j(self.plot_dir(config, iters), "logfile.out")
        return logfile

    def do_plot(self):
        # from concurrent.futures import ProcessPoolExecutor
        # import multiprocessing
        # ncpus = multiprocessing.cpu_count()
        # with ProcessPoolExecutor(max_workers=ncpus) as pool:
        for config in self.configs:
            if config.config_suffix != 'uninstrumented' and \
                not config.is_calibration and \
                self.args.calibration_mode == 'validation':
                # self.plot_config(config)
                name = "{klass}.{func}".format(
                    klass=self.__class__.__name__,
                    func='plot_config')
                self._pool.submit(
                    name,
                    self.plot_config,
                    self.args.algo, self.args.env, config,
                    sync=self.quick_expr.args.debug_single_thread,
                )
        self._pool.shutdown()

    def cupti_scaling_overhead_dir(self, algo, env):
        return _j(
            self.out_dir(algo, env),
            "cupti_scaling_overhead")

    def cupti_scaling_overhead_logfile(self, algo, env):
        task = "CUPTIScalingOverheadTask"
        logfile = _j(
            self.cupti_scaling_overhead_dir(algo, env),
            self._logfile_basename(task),
        )
        return logfile

    def cupti_overhead_dir(self, algo, env, iters):
        return _j(
            self.out_dir(algo, env),
            "cupti_overhead_iters_{iters}".format(
                iters=iters,
            ))

    def cupti_overhead_logfile(self, algo, env, iters):
        task = "CUPTIOverheadTask"
        logfile = _j(
            self.cupti_overhead_dir(algo, env, iters),
            self._logfile_basename(task),
        )
        return logfile

    def LD_PRELOAD_overhead_dir(self, algo, env, iters):
        return _j(
            self.out_dir(algo, env),
            "LD_PRELOAD_overhead_iters_{iters}".format(
                iters=iters,
            ))

    def LD_PRELOAD_overhead_logfile(self, algo, env, iters):
        task = "CallInterceptionOverheadTask"
        logfile = _j(
            self.LD_PRELOAD_overhead_dir(algo, env, iters),
            self._logfile_basename(task),
        )
        return logfile

    def pyprof_overhead_dir(self, algo, env, iters):
        return _j(
            self.out_dir(algo, env),
            "pyprof_overhead_iters_{iters}".format(
                iters=iters,
            ))

    def pyprof_overhead_logfile(self, algo, env, iters):
        task = "PyprofOverheadTask"
        logfile = _j(
            self.pyprof_overhead_dir(algo, env, iters),
            self._logfile_basename(task),
        )
        return logfile


    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def _glob_json_files(self, direc):
        json_paths = glob("{direc}/*.json".format(
            direc=direc))
        return json_paths


    def compute_cupti_scaling_overhead(self, algo, env):
        task = "CUPTIScalingOverheadTask"
        all_gpu_activities_api_time_directories = []
        all_interception_directories = []
        iters_01 = self.get_iterations(self.conf(algo, env, 'gpu_activities_api_time_calibration'))
        iters_02 = self.get_iterations(self.conf(algo, env, 'interception_calibration'))
        if iters_01 != iters_02:
            logger.info("data: {msg}".format(msg=pprint_msg({
                'iters_01': iters_01,
                'iters_02': iters_02,
            })))
            assert iters_01 == iters_02
        iterations = self.get_iterations(self.conf(algo, env, 'gpu_activities_api_time_calibration'))

        if self.quick_expr.args.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'iterations': iterations,
                })))

        for iters in iterations:
            gpu_activities_api_time_directories = self.conf(algo, env, 'gpu_activities_api_time_calibration').iml_directories(iters)
            interception_directories = self.conf(algo, env, 'interception_calibration').iml_directories(iters)
            if len(gpu_activities_api_time_directories) != len(interception_directories):
                log_missing_files(self, task=task, files={
                    'gpu_activities_api_time_directories': gpu_activities_api_time_directories,
                    'interception_directories': interception_directories,
                })
                return
            all_gpu_activities_api_time_directories.extend(gpu_activities_api_time_directories)
            all_interception_directories.extend(interception_directories)

        directory = self.cupti_scaling_overhead_dir(algo, env)
        if not self.quick_expr.args.dry_run:
            os.makedirs(directory, exist_ok=True)
        cmd = ['iml-analyze',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-api-time-directory', json.dumps(all_gpu_activities_api_time_directories),
               '--interception-directory', json.dumps(all_interception_directories),
               ]
        add_iml_analyze_flags(cmd, self.quick_expr.args)
        cmd.extend(self.extra_argv)

        logfile = self.cupti_scaling_overhead_logfile(algo, env)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.quick_expr.args.dry_run,
            skip_error=self.quick_expr.args.skip_error,
            debug=self.quick_expr.args.debug)

    def compute_cupti_overhead(self, algo, env):
        task = "CUPTIOverheadTask"
        assert self.get_iterations(self.conf(algo, env, 'gpu_activities_calibration')) == self.get_iterations(self.conf(algo, env, 'no_gpu_activities_calibration'))
        iterations = self.get_iterations(self.conf(algo, env, 'gpu_activities_calibration'))

        if self.quick_expr.args.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'iterations': iterations,
                })))

        for iters in iterations:

            gpu_activities_directories = self.conf(algo, env, 'gpu_activities_calibration').iml_directories(iters)
            no_gpu_activities_directories = self.conf(algo, env, 'no_gpu_activities_calibration').iml_directories(iters)
            if self.quick_expr.args.debug:
                logger.info("log = {msg}".format(
                    msg=pprint_msg({
                        'iters': iters,
                        'gpu_activities_directories': gpu_activities_directories,
                        'no_gpu_activities_directories': no_gpu_activities_directories,
                    })))

            if len(gpu_activities_directories) != len(no_gpu_activities_directories):
                log_missing_files(self, task=task, files={
                    'gpu_activities_directories': gpu_activities_directories,
                    'no_gpu_activities_directories': no_gpu_activities_directories,
                })
                continue

            directory = self.cupti_overhead_dir(algo, env, iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(directory, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', directory,
                   '--task', task,
                   '--gpu-activities-directory', json.dumps(gpu_activities_directories),
                   '--no-gpu-activities-directory', json.dumps(no_gpu_activities_directories),
                   ]
            add_iml_analyze_flags(cmd, self.quick_expr.args)
            cmd.extend(self.extra_argv)

            logfile = self.cupti_overhead_logfile(algo, env, iters)
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def compute_LD_PRELOAD_overhead(self, algo, env):
        task = "CallInterceptionOverheadTask"

        if self.quick_expr.args.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'iterations': self.iterations,
                })))

        for iters in self.iterations:
            interception_directories = self.conf(algo, env, 'interception_calibration').iml_directories(iters)
            uninstrumented_directories = self.conf(algo, env, 'uninstrumented_calibration').iml_directories(iters)
            if self.quick_expr.args.debug:
                logger.info("log = {msg}".format(
                    msg=pprint_msg({
                        'iters': iters,
                        'interception_directories': interception_directories,
                        'uninstrumented_directories': uninstrumented_directories,
                    })))
            if self.quick_expr.args.debug:
                logger.info("log = {msg}".format(
                    msg=pprint_msg({
                        'iters': iters,
                        'interception_directories': interception_directories,
                        'uninstrumented_directories': uninstrumented_directories,
                    })))
            if len(interception_directories) != len(uninstrumented_directories):
                log_missing_files(self, task=task, files={
                    'interception_directories': interception_directories,
                    'uninstrumented_directories': uninstrumented_directories,
                })
                continue

            directory = self.LD_PRELOAD_overhead_dir(algo, env, iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(directory, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', directory,
                   '--task', task,
                   '--interception-directory', json.dumps(interception_directories),
                   '--uninstrumented-directory', json.dumps(uninstrumented_directories),
                   ]
            add_iml_analyze_flags(cmd, self.quick_expr.args)
            cmd.extend(self.extra_argv)

            logfile = self.LD_PRELOAD_overhead_logfile(algo, env, iters)
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def compute_pyprof_overhead(self, algo, env):
        task = "PyprofOverheadTask"

        if self.quick_expr.args.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'iterations': self.iterations,
                })))

        for iters in self.iterations:
            uninstrumented_directories = self.conf(algo, env, 'uninstrumented_calibration').iml_directories(iters)
            pyprof_annotations_directories = self.conf(algo, env, 'just_pyprof_annotations_calibration').iml_directories(iters)
            pyprof_interceptions_directories = self.conf(algo, env, 'just_pyprof_interceptions_calibration').iml_directories(iters)
            if self.quick_expr.args.debug:
                logger.info("log = {msg}".format(
                    msg=pprint_msg({
                        'iters': iters,
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
                continue

            directory = self.pyprof_overhead_dir(algo, env, iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(directory, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', directory,
                   '--task', task,
                   '--uninstrumented-directory', json.dumps(uninstrumented_directories),
                   '--pyprof-annotations-directory', json.dumps(pyprof_annotations_directories),
                   '--pyprof-interceptions-directory', json.dumps(pyprof_interceptions_directories),
                   ]
            add_iml_analyze_flags(cmd, self.quick_expr.args)
            cmd.extend(self.extra_argv)

            logfile = self.pyprof_overhead_logfile(algo, env, iters)
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def plot_config(self, algo, env, config):
        # PROBLEM: we only want to call plot if there are "enough" files to plot a "configuration";
        # - skip plot if 0 --iml-directories
        # - skip plot if 0 --uninstrumented-directories
        assert config != self.conf(algo, env, 'uninstrumented')
        task = 'CorrectedTrainingTimeTask'
        # iterations = self.iterations
        iterations = self.get_iterations(config)
        iters = iterations[0]

        #
        # Use the pyprof/CUPTI/LD_PRELOAD overhead we calibrated using one_minute_iterations iterations.
        #

        raise NotImplementedError("Use cpp code, not old python implementation... (not maintained anymore)")

        pyprof_overhead_jsons = self._glob_json_files(self.pyprof_overhead_dir(algo, env, iters))
        assert len(pyprof_overhead_jsons) <= 1

        cupti_overhead_jsons = self._glob_json_files(self.cupti_overhead_dir(algo, env, iters))
        assert len(cupti_overhead_jsons) <= 1

        LD_PRELOAD_overhead_jsons = self._glob_json_files(self.LD_PRELOAD_overhead_dir(algo, env, iters))
        assert len(LD_PRELOAD_overhead_jsons) <= 1

        config_uninstrumented = self.conf(algo, env, 'uninstrumented')

        logger.info("log = {msg}".format(
            msg=pprint_msg({
                'iterations': iterations,
                'iters': iters,
                'pyprof_overhead_jsons': pyprof_overhead_jsons,
                'cupti_overhead_jsons': cupti_overhead_jsons,
                'LD_PRELOAD_overhead_jsons': LD_PRELOAD_overhead_jsons,
            })))

        for iters in iterations:
            iml_directories = config.iml_directories(iters)
            uninstrumented_directories = config_uninstrumented.iml_directories(iters)

            if ( len({len(iml_directories), len(uninstrumented_directories)}) != 1 ) or \
                len(pyprof_overhead_jsons) == 0 or \
                len(cupti_overhead_jsons) == 0 or \
                len(LD_PRELOAD_overhead_jsons) == 0:
                log_missing_files(self, task, files={
                    'iml_directories': iml_directories,
                    'uninstrumented_directories': uninstrumented_directories,
                    'pyprof_overhead_jsons': pyprof_overhead_jsons,
                    'cupti_overhead_jsons': cupti_overhead_jsons,
                    'LD_PRELOAD_overhead_jsons': LD_PRELOAD_overhead_jsons,
                })
                logger.info((
                    "{klass}: SKIP plotting iterations={iters}, config={config}, "
                    "since --iml-directories and --uninstrumented-directories haven't been generated yet.").format(
                    iters=iters,
                    config=config.to_string(),
                    klass=self.__class__.__name__
                ))
                # import ipdb; ipdb.set_trace()
                continue

            # iml-analyze
            # --task CorrectedTrainingTimeTask
            # --directory $direc
            # --iml-directories "$(js_list.py output/iml_bench/debug_prof_overhead/config_${config_dir}_*/ppo2/HalfCheetahBulletEnv-v0)"
            # --uninstrumented-directories "$(js_list.py output/iml_bench/debug_prof_overhead/config_no_interception_*/ppo2/HalfCheetahBulletEnv-v0)"
            # --cupti-overhead-json output/iml_bench/debug_prof_overhead/results.config_full/cupti_overhead.json
            # --call-interception-overhead-json output/iml_bench/debug_prof_overhead/results.config_full/LD_PRELOAD_overhead.json
            # --debug
            # --pdb
            # --debug-memoize

            plot_dir = self.plot_dir(config, iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(plot_dir, exist_ok=True)
            # calibration_jsons = CalibrationJSONs(
            #     cupti_overhead_json=cupti_overhead_jsons[0],
            #     LD_PRELOAD_overhead_json=LD_PRELOAD_overhead_jsons[0],
            #     # python_annotation_json=None,
            #     # python_clib_interception_tensorflow_json=None,
            #     # python_clib_interception_simulator_json=None,
            # )
            cmd = ['iml-analyze',
                   '--directory', plot_dir,
                   '--task', task,

                   '--pyprof-overhead-json', pyprof_overhead_jsons[0],
                   '--cupti-overhead-json', cupti_overhead_jsons[0],
                   '--LD-PRELOAD-overhead-json', LD_PRELOAD_overhead_jsons[0],

                   '--iml-directories', json.dumps(iml_directories),
                   '--uninstrumented-directories', json.dumps(uninstrumented_directories),

                   '--iml-prof-config', config.iml_prof_config,
                   ]
            # cmd.extend(calibration_jsons.argv())
            add_iml_analyze_flags(cmd, self.quick_expr.args)
            cmd.extend(self.extra_argv)

            # get these args from forwarding extra_argv
            # --cupti-overhead-json
            # output/iml_bench/debug_prof_overhead/results.config_full/cupti_overhead.json
            # --call-interception-overhead-json
            # output/iml_bench/debug_prof_overhead/results.config_full/LD_PRELOAD_overhead.json
            # --directory
            # $direc
            # --debug
            # --pdb
            # --debug-memoize

            logfile = self.plot_logfile(config, iters)
            # logger.info("Logging to file {path}".format(
            #     path=logfile))
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def do_run(self):

        for rep in range(1, self.args.repetitions+1):
            for config in self.configs:
                iterations = self.get_iterations(config)
                for iters in iterations:
                    config.run(rep, iters)

        for algo, env in self.stable_baselines_algo_env:
            # self.compute_cupti_scaling_overhead(algo, env)
            # self.compute_cupti_overhead(algo, env)
            # self.compute_LD_PRELOAD_overhead(algo, env)
            # self.compute_pyprof_overhead(algo, env)
            self._pool.submit(
                get_func_name(self, 'compute_cupti_scaling_overhead'),
                self.compute_cupti_scaling_overhead,
                algo, env,
                sync=self.quick_expr.args.debug_single_thread,
            )
            self._pool.submit(
                get_func_name(self, 'compute_cupti_overhead'),
                self.compute_cupti_overhead,
                algo, env,
                sync=self.quick_expr.args.debug_single_thread,
            )
            self._pool.submit(
                get_func_name(self, 'compute_LD_PRELOAD_overhead'),
                self.compute_LD_PRELOAD_overhead,
                algo, env,
                sync=self.quick_expr.args.debug_single_thread,
            )
            self._pool.submit(
                get_func_name(self, 'compute_pyprof_overhead'),
                self.compute_pyprof_overhead,
                algo, env,
                sync=self.quick_expr.args.debug_single_thread,
            )
        self._pool.shutdown()

    @property
    def iterations(self):

        def iters_from_runs(runs):
            iterations = [self.args.one_minute_iterations*(2**i) for i in runs]
            return iterations

        if self.args.only_runs is not None and len(self.args.only_runs) > 0:
            runs = [run for run in range(self.args.num_runs) if run in self.args.only_runs]
        else:
            runs = list(range(self.args.num_runs))

        # logger.info("Runs: {msg}".format(
        #     msg=pprint_msg(runs)))

        iterations = iters_from_runs(runs)
        return iterations

    def get_iterations(self, config=None):
        iterations = self.iterations

        if config is not None and not config.is_calibration and ( config.long_run and self.args.long_run ):
            long_run_iterations = self.args.one_minute_iterations*(2**self.args.long_run_exponent)
            if long_run_iterations not in iterations:
                iterations.append(long_run_iterations)

        return iterations

    def conf(self, algo, env, config_suffix):
        return self.config_suffix_to_obj[algo][env][config_suffix]

    def init_configs(self):
        self.configs = []
        self.config_suffix_to_obj = dict()
        self.stable_baselines_algo_env = expr_config.stable_baselines_gather_algo_env_pairs(
            algo=self.args.algo,
            env_id=self.args.env,
            bullet=self.args.bullet,
            atari=self.args.atari,
            lunar=self.args.lunar,
            algo_env_group=self.args.algo_env_group,
            debug=self.quick_expr.args.debug,
        )
        for algo, env in self.stable_baselines_algo_env:
            self._add_configs(algo, env)


        logger.info("Run configuration: {msg}".format(msg=pprint_msg({
            '(algo, env)': self.stable_baselines_algo_env,
            'configs': self.configs,
        })))


    def _add_configs(self, algo, env):

        algo_env_configs = []
        def add_calibration_config(**common_kwargs):
            config_kwargs = dict(common_kwargs)
            config_kwargs['long_run'] = True
            config = ExprSubtractionValidationConfig(**config_kwargs)
            assert not config.is_calibration
            if self.args.calibration_mode == 'validation':
                # *_calibration folders are used for computing average overhead.
                # Folders without the *_calibration are the runs we subtract the averages from.
                algo_env_configs.append(config)

            config_suffix = "{suffix}_calibration".format(
                suffix=config.config_suffix,
            )
            calibration_config_kwargs = dict(common_kwargs)
            calibration_config_kwargs.update(dict(
                # Disable tfprof: CUPTI and LD_PRELOAD.
                config_suffix=config_suffix,
                long_run=False,
            ))
            calibration_config = ExprSubtractionValidationConfig(**calibration_config_kwargs)
            assert calibration_config.is_calibration
            algo_env_configs.append(calibration_config)

        # Entirely uninstrumented configuration; we use this in many of the overhead calculations to determine
        # how much training time is attributable to the enabled "feature" (e.g. CUPTI activities).
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
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
            algo=algo,
            env=env,
            iml_prof_config='interception',
            config_suffix='interception',
            script_args=['--iml-disable-pyprof'],
        )

        # CUPTIOverheadTask: CUPTI, and CUDA API stat-tracking overhead correction.
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            iml_prof_config='gpu-activities',
            config_suffix='gpu_activities',
            script_args=['--iml-disable-pyprof'],
        )
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            iml_prof_config='no-gpu-activities',
            config_suffix='no_gpu_activities',
            script_args=['--iml-disable-pyprof'],
        )
        # CUPTIScalingOverheadTask:
        config = ExprSubtractionValidationConfig(
            expr=self,
            algo=algo,
            env=env,
            iml_prof_config='gpu-activities-api-time',
            config_suffix='gpu_activities_api_time_calibration',
            script_args=['--iml-disable-pyprof'],
            long_run=True,
        )
        algo_env_configs.append(config)

        if self.args.calibration_mode == 'validation':
            # Evaluate: combined tfprof/pyprof overhead correction.
            # (i.e. full IML trace-collection).
            config = ExprSubtractionValidationConfig(
                expr=self,
                algo=algo,
                env=env,
                iml_prof_config='full',
                # Enable tfprof: CUPTI and LD_PRELOAD.
                config_suffix='full',
                # Enable pyprof.
                script_args=[],
                long_run=True,
            )
            algo_env_configs.append(config)

        if self.args.calibration_mode == 'validation':
            # Evaluate: tfprof overhead correction in isolation.
            config = ExprSubtractionValidationConfig(
                expr=self,
                algo=algo,
                env=env,
                iml_prof_config='full',
                # Enable tfprof: CUPTI and LD_PRELOAD.
                config_suffix='just_tfprof',
                # DON'T enable pyprof.
                script_args=['--iml-disable-pyprof'],
                long_run=True,
            )
            algo_env_configs.append(config)

        if self.args.calibration_mode == 'validation':
            # Evaluate: pyprof overhead correction in isolation.
            config = ExprSubtractionValidationConfig(
                expr=self,
                algo=algo,
                env=env,
                iml_prof_config='uninstrumented',
                # Disable tfprof: CUPTI and LD_PRELOAD.
                config_suffix='just_pyprof',
                # Enable pyprof.
                script_args=['--iml-disable-tfprof'],
            )
            algo_env_configs.append(config)

        # PyprofOverheadTask: Python->C-lib event tracing, and operation annotation overhead correction.
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_annotations',
            # Only enable GPU/C-lib event collection, not operation annotations.
            script_args=['--iml-disable-tfprof', '--iml-disable-pyprof-interceptions'],
        )
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_interceptions',
            # Only enable operation annotations, not GPU/C-lib event collection.
            script_args=['--iml-disable-tfprof', '--iml-disable-pyprof-annotations'],
        )

        # for config in algo_env_configs:
        #     if algo not in self.config_suffix_to_obj:
        #         self.config_suffix_to_obj[algo] = dict()
        #     if env not in self.config_suffix_to_obj[algo]:
        #         self.config_suffix_to_obj[algo][env] = dict()
        #     assert config.config_suffix not in self.config_suffix_to_obj[algo][env]
        #     self.config_suffix_to_obj[algo][env][config.config_suffix] = config

        for config in self.configs:
            mk_dict_tree(self.config_suffix_to_obj, [algo, env])
            assert config.config_suffix not in self.config_suffix_to_obj[algo][env]
            self.config_suffix_to_obj[algo][env][config.config_suffix] = config

        self.configs.extend(algo_env_configs)

    def run(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--num-runs',
            type=int,
            default=3)
        parser.add_argument(
            '--calibration-mode',
            default='calibration',
            choices=['calibration', 'validation'],
            help=textwrap.dedent("""
            calibration:
                Only run configurations needed to subtract overhead; i.e.
                
                PyprofOverheadTask
                    uninstrumented
                    just_pyprof_annotations
                    just_pyprof_interceptions

                CUPTIOverheadTask
                    gpu_activities
                    no_gpu_activities

                CUPTIScalingOverheadTask
                    gpu_activities_api_time
                    interception
            
            validation:
                Run additional configurations for "isolating" bugs in overhead correction. 
                For example, run with just python annotations enabled so we can see if correcting for python annotations in isolation works.
            """),
        )
        parser.add_argument(
            '--only-runs',
            type=int,
            nargs='*',
            # action='append',
            # default=[],
            help=textwrap.dedent("""
            For each configuration, we will run it for:
              one_minute_iterations * 2**(i)
              for i in --only-runs
        """))
        parser.add_argument(
            '--long-run-exponent',
            type=int,
            default=5,
            help=textwrap.dedent("""
            For each configuration, we will run it for:
              one_minute_iterations * 2**(long_run_exponent)
               
              --num-runs=3
                  one_minute_iterations
                  one_minute_iterations*2
                  one_minute_iterations*4
                  
              --long-run-exponent=5
                  one_minute_iterations*(2**5) == one_minute_iterations*32
        """))
        parser.add_argument(
            '--one-minute-iterations',
            type=int,
            help="Number of iterations of (algo, env) that take 1 minute to run",
            default=22528)
        parser.add_argument(
            '--repetitions',
            type=int,
            default=3)
        parser.add_argument(
            '--bullet',
            action='store_true',
            help='Limit environments to physics-based Bullet environments')
        parser.add_argument(
            '--atari',
            action='store_true',
            help='Limit environments to Atari Pong environment')
        parser.add_argument(
            '--lunar',
            action='store_true',
            help='Limit environments to LunarLander environments (i.e. LunarLanderContinuous-v2, LunarLander-v2)')
        parser.add_argument(
            '--long-run',
            action='store_true',
            help='For each calibration, do an extra long 30 minute calibration to make sure it works')
        parser.add_argument(
            '--env',
        )
        parser.add_argument(
            '--algo',
        )
        parser.add_argument('--algo-env-group',
                            choices=expr_config.ALGO_ENV_GROUP_CHOICES,
                            help=textwrap.dedent("""
            Only run a specific "experiment".
            i.e. only run (algo, env) combinations needed for a specific graph.
            
            Default: run all (algo, env) combinations for all experiments.
            """))
        parser.add_argument(
            '--plot',
            action='store_true')
        # parser.add_argument(
        #     '--iml-prof-config',
        #     choices=['instrumented', 'full'],
        #     default='full')
        self.args, self.extra_argv = parser.parse_known_args(self.argv)

        self.init_configs()

        if self.args.plot:
            self.do_plot()
        else:
            self.do_run()

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
    parser.add_argument("--iml-debug",
                        action='store_true',
                        help=textwrap.dedent("""
                        Debug
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
        obj = Calibration(
            iml_directory=args.iml_directory,
            debug=args.iml_debug,
            cmd=cmd,
        )
        obj.run()
    except Exception as e:
        if not args.pdb:
            raise
        logger.debug("> IML: Detected exception:")
        print(e)
        logger.debug("> Entering pdb:")
        import ipdb
        ipdb.post_mortem()
        raise

if __name__ == '__main__':
    main()
