from iml_profiler.profiler import iml_logging
import argparse
import pprint
from glob import glob
import textwrap
import os
from os import environ as ENV
import json

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.parser.common import *
from iml_profiler.experiment.util import tee, expr_run_cmd, expr_already_ran
from iml_profiler.profiler.concurrent import ForkedProcessPool
from iml_profiler.scripts import bench
from iml_profiler.experiment import expr_config

# ( set -e; set -x; iml-quick-expr --expr total_training_time --repetitions 3 --bullet; )
# ( set -e; set -x; iml-quick-expr --expr total_training_time --repetitions 3 --bullet --plot; )

# ( set -e; set -x; iml-quick-expr --expr total_training_time --repetitions 3 ---bullet --instrumented; )
# ( set -e; set -x; iml-quick-expr --expr total_training_time --repetitions 3 --bullet --instrumented --plot; )


# algo_envs = [
#     {'algo': 'Walker2DBulletEnv-v0', 'env': 'ddpg'}
#     {'algo': 'HopperBulletEnv-v0', 'env': 'a2c'},
#     {'algo': 'PongNoFrameskip-v4', 'env': 'dqn'},
# ]
# ( set -e; set -x; iml-quick-expr --expr subtraction_validation --repetitions 3 --env Walker2DBulletEnv-v0 --algo ddpg; )
# ( set -e; set -x; iml-quick-expr --expr subtraction_validation --repetitions 3 --env HopperBulletEnv-v0 --algo a2c; )
# ( set -e; set -x; iml-quick-expr --expr subtraction_validation --repetitions 3 --env PongNoFrameskip-v4 --algo dqn; )
# ( set -e; set -x; iml-quick-expr --expr subtraction_validation --repetitions 3 --bullet; )

# ( set -e; set -x; iml-quick-expr --expr subtraction_validation --repetitions 3 --bullet --plot; )

# ( set -e; set -x; timesteps=20000; iml-quick-expr --expr total_training_time --repetitions 3 --env HalfCheetahBulletEnv-v0 --subdir debug_n_timesteps_${timesteps} --plot; )

# Experiments to run:
# - Generalization of calibration:
#   - Atari Pong, dqn
#   - Minigo
#   - Walker2D, ddpg
#   - Hopper, a2c
#   - Ideally, all other OpenAI workloads
# - Full training time:
#   - Minigo
#   - All the OpenAI workloads

class QuickExpr:
    """
    To create a new experiment my_expr, define a method that looks like:

        def expr_my_expr(self):
            ...

    Then you can run it with:

        $ iml-quick-expr --expr my_expr
    """
    def __init__(self, args, extra_argv):
        self.args = args
        self.extra_argv = extra_argv

    @property
    def out_dir(self):
        components = [x for x in [
            self.args.dir,
            "expr_{expr}".format(expr=self.args.expr),
            self.args.subdir,
        ] if x is not None]
        return _j(*components)

    def expr_subtraction_validation(self, extra_argv):
        expr_subtraction_validation = ExprSubtractionValidation(quick_expr=self, argv=extra_argv)
        expr_subtraction_validation.run()

    def expr_total_training_time(self, extra_argv):
        expr_total_training_time = ExprTotalTrainingTime(quick_expr=self, argv=extra_argv)
        expr_total_training_time.run()

    def run(self):
        expr_func_name = "expr_{expr}".format(expr=self.args.expr)
        if not hasattr(self, expr_func_name):
            raise NotImplementedError((
                "Not sure how to run --expr={expr}; "
                "please create a method {klass}.{func}").format(
                expr=self.args.expr,
                func=expr_func_name,
                klass=self.__class__.__name__,
            ))
        if not self.args.dry_run:
            os.makedirs(self.out_dir, exist_ok=True)
        func = getattr(self, expr_func_name)
        func(self.extra_argv)

def main():
    iml_logging.setup_logging()
    parser = argparse.ArgumentParser(
        textwrap.dedent("""\
        Make-once run-once experiments; 
        maintainability not a priority, just run a quick experiment!
        """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--debug',
        action='store_true')
    parser.add_argument(
        '--expr',
        choices=[
            'subtraction_validation',
            'total_training_time',
        ],
        required=True,
        help=textwrap.dedent("""
        --expr my_expr will run experiment QuickExpr.expr_my_expr().
        
        subtraction_validation:
            See how much unaccounted for overhead percent varies as we increase 
            the number of training loop iterations.
        total_training_time:
            Run the entire ML training script to measure how long the total training time is.
        """))
    # Don't support --pdb for iml-bench since I haven't figured out how to
    # both (1) log stdout/stderr of a command, (2) allow pdb debugger prompt
    # to be fully functional.
    # Even if we could...you probably don't want to log your pdb session
    # and all its color-codes anyway.
    # parser.add_argument(
    #     '--pdb',
    #     action='store_true')
    parser.add_argument('--debug-single-thread',
                        action='store_true',
                        help=textwrap.dedent("""
    Debug with single thread.
    """))
    parser.add_argument('--debug-memoize',
                        action='store_true',
                        help=textwrap.dedent("""
    Development: speedup runs my memoizing intermediate results; you need to delete the 
    *.pickle files it generates manually otherwise you may accidentally re-use 
    stale files when code changes.
    """))
    parser.add_argument(
        '--replace',
        action='store_true')
    parser.add_argument(
        '--dry-run',
        action='store_true')
    parser.add_argument(
        '--skip-error',
        action='store_true',
        help=textwrap.dedent("""
    If a script fails and there's more left to run, ignore it and continue.
    (NOTE: don't worry, stderr error will still get captured to a logfile).
    """))
    parser.add_argument(
        '--dir',
        default='./output',
        help=textwrap.dedent("""\
        Directory to store stuff in for the subcommand.
        
        all subcommands: 
          log files
        train_stable_baselines.sh / stable-baselines:
          trace-files @ <algo>/<env>
        plot-stable-baselines:
          plots
        """.rstrip()))
    parser.add_argument(
        '--subdir',
        help=textwrap.dedent("""
        Store files root at:
            <--dir>/expr_<--expr>/<--subdir>
        """))

    args, extra_argv = parser.parse_known_args()
    quick_expr = QuickExpr(args, extra_argv)
    quick_expr.run()


class ExprSubtractionValidationConfig:
    def __init__(self, expr, algo, env, iml_prof_config, config_suffix, script_args=[], long_run=False):
        self.expr = expr
        self.algo = algo
        self.env = env
        self.quick_expr = self.expr.quick_expr
        # $ iml-prof --config ${iml_prof_config}
        self.iml_prof_config = iml_prof_config
        # $ python train.py --iml-directory config_${config_suffix}
        self.config_suffix = config_suffix
        self.script_args = script_args
        self.long_run = long_run

    @property
    def is_calibration(self):
        return re.search(r'_calibration', self.config_suffix)

    def out_dir(self, rep, iters):
        return _j(
            self.expr.out_dir(self.algo, self.env),
            "config_{config_suffix}_iters_{iters}{rep}".format(
                config_suffix=self.config_suffix,
                iters=iters,
                rep=rep_suffix(rep),
            ))

    def to_string(self):
        return ("ExprSubtractionValidationConfig("
                "iml_prof_config='{iml_prof_config}'"
                ", config_suffix='{config_suffix}'"
                ")").format(
            iml_prof_config=self.iml_prof_config,
            config_suffix=self.config_suffix,
        )

    def logfile(self, rep, iters):
        logfile = _j(self.out_dir(rep, iters), "logfile.out")
        return logfile

    def long_run_iters(self):
        return 2**self.expr.args.long_run_exponent

    def run(self, rep, iters):
        cmd = ['iml-prof',
               '--config', self.iml_prof_config,
               'python', 'train.py',

               '--iml-directory', _a(self.out_dir(rep, iters)),
               '--iml-max-timesteps', iters,
               '--iml-training-progress',

               '--algo', self.algo,
               '--env', self.env,

               '--log-folder', _j(ENV['RL_BASELINES_ZOO_DIR'], 'output'),
               '--log-interval', '1',
               '--iml-start-measuring-call', '1',
               '--iml-delay',
               ]
        cmd.extend(self.script_args)
        logfile = self.logfile(rep, iters)
        # logging.info("Logging to file {path}".format(
        #     path=logfile))
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            cwd=ENV['RL_BASELINES_ZOO_DIR'],
            replace=self.quick_expr.args.replace,
            dry_run=self.quick_expr.args.dry_run,
            skip_error=self.quick_expr.args.skip_error,
            debug=self.quick_expr.args.debug)

    def already_ran(self, rep, iters):
        logfile = self.logfile(rep, iters)
        return expr_already_ran(logfile, debug=self.quick_expr.args.debug)

    def iml_directories(self, iters):
        """
        Return all --iml-directories whose runs are completed.
        """
        iml_directories = []
        for rep in range(1, self.expr.args.repetitions+1):
            if not self.already_ran(rep, iters):
                continue
            iml_directory = self.out_dir(rep, iters)
            iml_directories.append(iml_directory)
        return iml_directories

class ExprTotalTrainingTimeConfig:
    def __init__(self, expr, algo, env, iml_prof_config='uninstrumented', script_args=[]):
        self.expr = expr
        self.quick_expr = self.expr.quick_expr
        # $ iml-prof --config ${iml_prof_config}
        # NOTE: we want to run with IML disabled; we just want to know the total training time WITHOUT IML.
        self.iml_prof_config = iml_prof_config
        # $ python train.py --iml-directory config_${config_suffix}
        self.config_suffix = self.iml_prof_config
        self.script_args = script_args
        self.algo = algo
        self.env = env

    def out_dir(self, rep):
        return _j(
            self.expr.out_dir, self.algo, self.env,
            "config_{config_suffix}{rep}".format(
                config_suffix=self.config_suffix,
                rep=rep_suffix(rep),
            ))

    def to_string(self):
        return ("{klass}("
                "config_suffix='{config_suffix}'"
                ", algo='{algo}'"
                ", env='{env}'"
                ")").format(
            klass=self.__class__.__name__,
            config_suffix=self.config_suffix,
            algo=self.algo,
            env=self.env,
        )

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def logfile(self, rep):
        logfile = _j(self.out_dir(rep), "logfile.out")
        return logfile

    def _get_cmd(self, rep, extra_argv=[]):
        cmd = ['iml-prof',
               '--config', self.iml_prof_config,
               'python', 'train.py',

               '--iml-directory', _a(self.out_dir(rep)),
               # '--iml-max-timesteps', iters,
               '--iml-training-progress',

               '--algo', self.algo,
               '--env', self.env,

               '--log-folder', _j(ENV['RL_BASELINES_ZOO_DIR'], 'output'),
               '--log-interval', '1',
               '--iml-start-measuring-call', '1',
               '--iml-delay',
               ]
        if self.iml_prof_config == 'uninstrumented':
            cmd.extend([
                # NOTE: we want to run with IML disabled; we just want to know the total training time WITHOUT IML.
                '--iml-disable',
            ])
        cmd.extend(self.script_args)
        cmd.extend(extra_argv)
        return cmd

    def run(self, rep, extra_argv=[]):
        # TODO: this is OpenAI specific; make it work for minigo.

        cmd = self._get_cmd(rep, extra_argv)

        logfile = self.logfile(rep)
        # logging.info("Logging to file {path}".format(
        #     path=logfile))
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            cwd=ENV['RL_BASELINES_ZOO_DIR'],
            replace=self.quick_expr.args.replace,
            dry_run=self.quick_expr.args.dry_run,
            skip_error=self.quick_expr.args.skip_error,
            debug=self.quick_expr.args.debug)

    def already_ran(self, rep):
        logfile = self.logfile(rep)
        return expr_already_ran(logfile, debug=self.quick_expr.args.debug)

    def iml_directories(self):
        """
        Return all --iml-directories whose runs are completed.
        """
        iml_directories = []
        for rep in range(1, self.expr.args.repetitions+1):
            if not self.already_ran(rep):
                continue
            iml_directory = self.out_dir(rep)
            iml_directories.append(iml_directory)
        return iml_directories

class ExprTotalTrainingTime:
    def __init__(self, quick_expr, argv):
        self.quick_expr = quick_expr
        self.argv = argv
        self._pool = ForkedProcessPool(name='{klass}.pool'.format(
            klass=self.__class__.__name__))

    @property
    def out_dir(self):
        return _j(self.quick_expr.out_dir)

    @property
    def plot_dir(self):
        return _j(
            self.out_dir,
            "plot")

    @property
    def plot_logfile(self):
        logfile = _j(self.plot_dir, "logfile.out")
        return logfile

    def run(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--bullet',
            action='store_true',
            help='Limit environments to physics-based Bullet environments')
        parser.add_argument(
            '--pong',
            action='store_true',
            help='Limit environments to Atari Pong environment')
        parser.add_argument(
            '--repetitions',
            type=int,
            default=3)
        parser.add_argument(
            '--env',
            )
        parser.add_argument(
            '--algo',
            )
        parser.add_argument(
            '--plot',
            action='store_true')
        parser.add_argument(
            '--instrumented',
            help="Run in fully instrumented mode (needed for creating \"Overhead correction\" figure)",
            action='store_true')
        self.args, self.extra_argv = parser.parse_known_args(self.argv)

        self.init_configs()

        if self.args.plot:
            self.do_plot()
        else:
            self.do_run()

    def do_plot(self):
        """
        Create bar graph of total training time for each experiment we run:
        iml_dirs = find_iml_dirs(self.root_dir)
        $ iml-analyze --task TotalTrainingTimePlot --iml-directories iml-dirs
        Read data-frame like:
          algo, env,         x_field, total_training_time_sec
          ppo2, HalfCheetah, ...,     ...
          ppo2, HalfCheetah, ...,     ...
        """

        task = 'TotalTrainingTimeTask'

        iml_directories = []
        for config in self.configs:
            config_dirs = config.iml_directories()
            iml_directories.extend(config_dirs)

        if len(iml_directories) == 0:
            log_missing_files(self, task, files={
                'iml_directories': iml_directories,
            })
            return

        plot_dir = self.plot_dir
        if not self.quick_expr.args.dry_run:
            os.makedirs(plot_dir, exist_ok=True)
        cmd = ['iml-analyze',
               '--directory', plot_dir,
               '--task', task,
               '--uninstrumented-directory', json.dumps(iml_directories),
               ]
        add_iml_analyze_flags(cmd, self.quick_expr.args)
        cmd.extend(self.extra_argv)

        logfile = self.plot_logfile
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.quick_expr.args.dry_run,
            skip_error=self.quick_expr.args.skip_error,
            debug=self.quick_expr.args.debug)

    def init_configs(self):
        self.configs = []
        self.stable_baselines_algo_env = expr_config.stable_baselines_gather_algo_env_pairs(
            algo=self.args.algo,
            env_id=self.args.env,
            all=True,
            bullet=self.args.bullet,
            pong=self.args.pong,
            debug=self.quick_expr.args.debug,
        )
        for algo, env in self.stable_baselines_algo_env:
            if self.args.instrumented:
                iml_prof_config = 'full'
            else:
                iml_prof_config = 'uninstrumented'
            config = ExprTotalTrainingTimeConfig(
                expr=self,
                algo=algo,
                env=env,
                iml_prof_config=iml_prof_config,
            )
            self.configs.append(config)
        logging.info("configs: " + pprint_msg(self.configs))
        # TODO: add config for minigo
        # Q: should we subclass ExprTotalTrainingTimeConfig to specialize running minigo experiment?

    def do_run(self):
        # Run the repetition 01 of every environment,
        # Run the repetition 02 of every environment,
        # ...
        # (allows us to incrementally plot stuff more easily)
        for rep in range(1, self.args.repetitions+1):
            for config in self.configs:
                # For debugging, allow overwriting --n-timesteps in train.py with whatever we want.
                config.run(rep, extra_argv=self.extra_argv)

class ExprSubtractionValidation:
    def __init__(self, quick_expr, argv):
        self.quick_expr = quick_expr
        self.argv = argv
        self._pool = ForkedProcessPool(name='{klass}.pool'.format(
            klass=self.__class__.__name__))
        # 1 min, 2 min, 4 min

    def out_dir(self, algo, env):
        return _j(self.quick_expr.out_dir, algo, env)

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
                    config,
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
        assert self.get_iterations(self.conf(algo, env, 'gpu_activities_api_time_calibration')) == self.get_iterations(self.conf(algo, env, 'interception_calibration'))
        iterations = self.get_iterations(self.conf(algo, env, 'gpu_activities_api_time_calibration'))
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
        for iters in iterations:
            gpu_activities_directories = self.conf(algo, env, 'gpu_activities_calibration').iml_directories(iters)
            no_gpu_activities_directories = self.conf(algo, env, 'no_gpu_activities_calibration').iml_directories(iters)
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
        for iters in self.iterations:
            interception_directories = self.conf(algo, env, 'interception_calibration').iml_directories(iters)
            uninstrumented_directories = self.conf(algo, env, 'uninstrumented_calibration').iml_directories(iters)
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
        for iters in self.iterations:
            uninstrumented_directories = self.conf(algo, env, 'uninstrumented_calibration').iml_directories(iters)
            pyprof_annotations_directories = self.conf(algo, env, 'just_pyprof_annotations_calibration').iml_directories(iters)
            pyprof_interceptions_directories = self.conf(algo, env, 'just_pyprof_interceptions_calibration').iml_directories(iters)

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

        pyprof_overhead_jsons = self._glob_json_files(self.pyprof_overhead_dir(algo, env, iters))
        assert len(pyprof_overhead_jsons) <= 1

        cupti_overhead_jsons = self._glob_json_files(self.cupti_overhead_dir(algo, env, iters))
        assert len(cupti_overhead_jsons) <= 1

        LD_PRELOAD_overhead_jsons = self._glob_json_files(self.LD_PRELOAD_overhead_dir(algo, env, iters))
        assert len(LD_PRELOAD_overhead_jsons) <= 1

        config_uninstrumented = self.conf(algo, env, 'uninstrumented')

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
                logging.info((
                    "{klass}: SKIP plotting iterations={iters}, config={config}, "
                    "since --iml-directories and --uninstrumented-directories haven't been generated yet.").format(
                    iters=iters,
                    config=config.to_string(),
                    klass=self.__class__.__name__
                ))
                import ipdb; ipdb.set_trace()
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
            # logging.info("Logging to file {path}".format(
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
        elif self.args.calibration_mode == 'calibration':
            # one_minute_iterations*(2**i) for i in [2] => 4 minute runs
            runs = [2]
        else:
            runs = list(range(self.args.num_runs))

        # logging.info("Runs: {msg}".format(
        #     msg=pprint_msg(runs)))

        iterations = iters_from_runs(runs)
        return iterations

    def get_iterations(self, config=None):
        iterations = self.iterations

        if config is not None and ( config.long_run and self.args.long_run ):
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
            all=True,
            bullet=self.args.bullet,
            pong=self.args.pong,
            debug=self.quick_expr.args.debug,
        )
        for algo, env in self.stable_baselines_algo_env:
            self._add_configs(algo, env)

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

        for config in algo_env_configs:
            if algo not in self.config_suffix_to_obj:
                self.config_suffix_to_obj[algo] = dict()
            if env not in self.config_suffix_to_obj[algo]:
                self.config_suffix_to_obj[algo][env] = dict()
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
            '--pong',
            action='store_true',
            help='Limit environments to Atari Pong environment')
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

def rep_suffix(rep):
    assert rep is not None
    return "_repetition_{rep:02}".format(rep=rep)

def add_iml_analyze_flags(cmd, args):
    if args.debug:
        cmd.append('--debug')
    if args.debug_memoize:
        cmd.append('--debug-memoize')
    if args.debug_single_thread:
        cmd.append('--debug-single-thread')

def log_missing_files(self, task, files):
    logging.info(textwrap.dedent("""
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
