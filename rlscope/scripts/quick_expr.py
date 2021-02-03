"""
``rls-quick-expr`` script for running lots of different experiments/benchmarks.
.. deprecated:: 1.0.0
    Replaced by :py:mod:`rlscope.scripts.calibration` (``rls-run --calibrate``).
"""
from rlscope.profiler.rlscope_logging import logger
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

from rlscope.profiler.util import pprint_msg
from rlscope.parser.common import *
from rlscope.experiment.util import tee, expr_run_cmd, expr_already_ran
from rlscope.profiler.concurrent import ForkedProcessPool
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
from rlscope.parser.exceptions import RLScopeConfigurationError

# ( set -e; set -x; rls-quick-expr --expr total_training_time --repetitions 3 --bullet; )
# ( set -e; set -x; rls-quick-expr --expr total_training_time --repetitions 3 --bullet --plot; )

# ( set -e; set -x; rls-quick-expr --expr total_training_time --repetitions 3 ---bullet --instrumented; )
# ( set -e; set -x; rls-quick-expr --expr total_training_time --repetitions 3 --bullet --instrumented --plot; )


# algo_envs = [
#     {'algo': 'Walker2DBulletEnv-v0', 'env': 'ddpg'}
#     {'algo': 'HopperBulletEnv-v0', 'env': 'a2c'},
#     {'algo': 'PongNoFrameskip-v4', 'env': 'dqn'},
# ]
# ( set -e; set -x; rls-quick-expr --expr subtraction_validation --repetitions 3 --env Walker2DBulletEnv-v0 --algo ddpg; )
# ( set -e; set -x; rls-quick-expr --expr subtraction_validation --repetitions 3 --env HopperBulletEnv-v0 --algo a2c; )
# ( set -e; set -x; rls-quick-expr --expr subtraction_validation --repetitions 3 --env PongNoFrameskip-v4 --algo dqn; )
# ( set -e; set -x; rls-quick-expr --expr subtraction_validation --repetitions 3 --bullet; )

# ( set -e; set -x; rls-quick-expr --expr subtraction_validation --repetitions 3 --bullet --plot; )

# ( set -e; set -x; timesteps=20000; rls-quick-expr --expr total_training_time --repetitions 3 --env HalfCheetahBulletEnv-v0 --subdir debug_n_timesteps_${timesteps} --plot; )

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

        $ rls-quick-expr --expr my_expr
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

    def expr_plot_fig(self, extra_argv):
        expr_plot_fig = ExprPlotFig(quick_expr=self, argv=extra_argv)
        expr_plot_fig.run()

    def expr_subtraction_validation(self, extra_argv):
        expr_subtraction_validation = ExprSubtractionValidation(quick_expr=self, argv=extra_argv)
        expr_subtraction_validation.run()

    def expr_total_training_time(self, extra_argv):
        expr_total_training_time = ExprTotalTrainingTime(quick_expr=self, argv=extra_argv)
        expr_total_training_time.run()

    def expr_microbenchmark(self, extra_argv):
        expr_microbenchmark = ExprMicrobenchmark(quick_expr=self, argv=extra_argv)
        expr_microbenchmark.run()

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

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--debug',
        action='store_true')
    parser.add_argument(
        '--expr',
        choices=[
            'subtraction_validation',
            'total_training_time',
            'plot_fig',
            'microbenchmark',
        ],
        required=True,
        help=textwrap.dedent("""\
        --expr my_expr will run experiment QuickExpr.expr_my_expr().
        
        subtraction_validation:
            See how much unaccounted for overhead percent varies as we increase 
            the number of training loop iterations.
        total_training_time:
            Run the entire ML training script to measure how long the total training time is.
        """))
    # Don't support --pdb for rls-bench since I haven't figured out how to
    # both (1) log stdout/stderr of a command, (2) allow pdb debugger prompt
    # to be fully functional.
    # Even if we could...you probably don't want to log your pdb session
    # and all its color-codes anyway.
    parser.add_argument(
        '--pdb',
        action='store_true')
    parser.add_argument('--debug-single-thread',
                        action='store_true',
                        help=textwrap.dedent("""\
    Debug with single thread.
    """))
    parser.add_argument('--debug-memoize',
                        action='store_true',
                        help=textwrap.dedent("""\
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
        help=textwrap.dedent("""\
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
        help=textwrap.dedent("""\
        Store files root at:
            <--dir>/expr_<--expr>/<--subdir>
        """))

    args, extra_argv = parser.parse_known_args()
    if args.pdb:
        args.debug_single_thread = True

    quick_expr = QuickExpr(args, extra_argv)

    try:
        quick_expr.run()
    except Exception as e:
        if not args.pdb:
            raise
        print("> RL-Scope: Detected exception:")
        print(e)
        print("> Entering pdb:")
        import pdb
        pdb.post_mortem()
        raise

class ExprSubtractionValidationConfig:
    def __init__(self, expr, algo, env, rlscope_prof_config, config_suffix, script_args=[], long_run=False):
        self.expr = expr
        self.algo = algo
        self.env = env
        self.quick_expr = self.expr.quick_expr
        # $ rls-prof --config ${rlscope_prof_config}
        self.rlscope_prof_config = rlscope_prof_config
        # $ python train.py --rlscope-directory config_${config_suffix}
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
        return ("{klass}("
                "rlscope_prof_config='{rlscope_prof_config}'"
                "config_suffix='{config_suffix}'"
                ", algo='{algo}'"
                ", env='{env}'"
                ")").format(
            klass=self.__class__.__name__,
            rlscope_prof_config=self.rlscope_prof_config,
            config_suffix=self.config_suffix,
            algo=self.algo,
            env=self.env,
        )

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def logfile(self, rep, iters):
        logfile = _j(self.out_dir(rep, iters), "logfile.out")
        return logfile

    def long_run_iters(self):
        return 2**self.expr.args.long_run_exponent

    def run(self, rep, iters):
        cmd = ['rls-prof',
               '--config', self.rlscope_prof_config,
               'python', 'train.py',

               # IMPORTANT: When we run with "--config uninstrumented" during calibrations runs, we STILL want to
               # keep "python interceptions" and "python annotations" enabled, so we can measure their overhead in
               # in isolation!
               '--rlscope-calibration',

               '--rlscope-directory', _a(self.out_dir(rep, iters)),
               '--rlscope-max-passes', iters,

               '--algo', self.algo,
               '--env', self.env,

               '--log-folder', _j(ENV['RL_BASELINES_ZOO_DIR'], 'output'),
               '--log-interval', '1',
               ]
        cmd.extend(self.script_args)
        logfile = self.logfile(rep, iters)
        # logger.info("Logging to file {path}".format(
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

    def rlscope_directories(self, iters):
        """
        Return all --rlscope-directories whose runs are completed.
        """
        rlscope_directories = []
        for rep in range(1, self.expr.args.repetitions+1):
            if not self.already_ran(rep, iters):
                continue
            rlscope_directory = self.out_dir(rep, iters)
            rlscope_directories.append(rlscope_directory)
        return rlscope_directories

class ExprTotalTrainingTimeConfig:
    def __init__(self, expr, algo, env, rlscope_prof_config='uninstrumented', script_args=[]):
        self.expr = expr
        self.quick_expr = self.expr.quick_expr
        # $ rls-prof --config ${rlscope_prof_config}
        # NOTE: we want to run with RL-Scope disabled; we just want to know the total training time WITHOUT IML.
        self.rlscope_prof_config = rlscope_prof_config
        # $ python train.py --rlscope-directory config_${config_suffix}
        self.config_suffix = self.rlscope_prof_config
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
        cmd = ['rls-prof',
               '--config', self.rlscope_prof_config,
               'python', 'train.py',

               '--rlscope-directory', _a(self.out_dir(rep)),

               '--algo', self.algo,
               '--env', self.env,

               '--log-folder', _j(ENV['RL_BASELINES_ZOO_DIR'], 'output'),
               '--log-interval', '1',
               ]
        if self.rlscope_prof_config == 'uninstrumented':
            cmd.extend([
                # NOTE: we want to run with RL-Scope disabled; we just want to know the total training time WITHOUT IML.
                '--rlscope-disable',
            ])
        cmd.extend(self.script_args)
        cmd.extend(extra_argv)
        return cmd

    def run(self, rep, extra_argv=[]):
        # TODO: this is OpenAI specific; make it work for minigo.

        cmd = self._get_cmd(rep, extra_argv)

        logfile = self.logfile(rep)
        # logger.info("Logging to file {path}".format(
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

    def rlscope_directories(self):
        """
        Return all --rlscope-directories whose runs are completed.
        """
        rlscope_directories = []
        for rep in range(1, self.expr.args.repetitions+1):
            if not self.already_ran(rep):
                continue
            rlscope_directory = self.out_dir(rep)
            rlscope_directories.append(rlscope_directory)
        return rlscope_directories

class ExprMicrobenchmarkConfig:
    def __init__(self, expr, algo, env, mode, config, rlscope_prof_config='uninstrumented', config_suffix=None, script_args=[]):
        self.mode = mode
        self.expr = expr
        self.config = config
        self.quick_expr = self.expr.quick_expr
        # $ rls-prof --config ${rlscope_prof_config}
        # NOTE: we want to run with RL-Scope disabled; we just want to know the total training time WITHOUT IML.
        self.rlscope_prof_config = rlscope_prof_config
        assert config_suffix is not None
        self.config_suffix = config_suffix
        # $ python train.py --rlscope-directory config_${config_suffix}
        # self.config_suffix = self.rlscope_prof_config
        self.script_args = script_args
        self.algo = algo
        self.env = env

    def out_dir(self):
        return _j(
            self.expr.out_dir,
            self.algo,
            self.env,
            self.config_suffix,
            "iterations_{i}".format(i=self.expr.args.iterations),
        )

    def to_string(self):
        return ("{klass}("
                "config_suffix='{config_suffix}'"
                ", algo='{algo}'"
                ", env='{env}'"
                ", mode='{mode}'"
                ")").format(
            klass=self.__class__.__name__,
            config_suffix=self.config_suffix,
            algo=self.algo,
            env=self.env,
            mode=self.mode,
        )

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def logfile(self, rep):
        logfile = _j(self.out_dir(), self.mode, "logfile.repetition_{r:02}.out".format(
            r=rep))
        return logfile

    def _get_cmd(self, rep, extra_argv=[]):
        cmd = ['rls-prof',
               '--config', self.rlscope_prof_config,
               'python', 'enjoy.py',

               # IMPORTANT: When we run with "--config uninstrumented" during calibrations runs, we STILL want to
               # keep "python interceptions" and "python annotations" enabled, so we can measure their overhead in
               # in isolation!
               '--rlscope-calibration',
               # '--rlscope-directory', _a(self.out_dir(rep)),
               '--rlscope-directory', _a(self.out_dir()),

               '--mode', self.mode,

               '--repetition', rep,
               '--iterations', self.expr.args.iterations,

               '--algo', self.algo,
               '--env', self.env,

               # '--log-folder', _j(ENV['RL_BASELINES_ZOO_DIR'], 'output'),
               # '--log-interval', '1',
               # '--rlscope-delay',
               ]
        # if self.config == 'uninstrumented':
        #     cmd.extend([
        #         # NOTE: we want to run with RL-Scope disabled; we just want to know the total training time WITHOUT IML.
        #         '--rlscope-disable',
        #     ])
        cmd.extend(self.script_args)
        cmd.extend(extra_argv)
        return cmd

    def run(self, rep, extra_argv=[]):
        # TODO: this is OpenAI specific; make it work for minigo.

        cmd = self._get_cmd(rep, extra_argv)

        logfile = self.logfile(rep)
        # logger.info("Logging to file {path}".format(
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

    def repetition_out_dir(self, rep):
        # output/
        # expr_microbenchmark/
        # ppo2/
        # HalfCheetahBulletEnv-v0/
        # config_mode_microbench_rlscope_clib_interception_simulator.config_just_pyprof_interceptions/
        # microbench_rlscope_clib_interception_simulator/
        # repetition_01

        # self.quick_expr.out_dir,
        # self.algo,
        # self.env,
        # self.config_suffix,
        return _j(
            self.out_dir(),
            # Specific to directory structure of enjoy.py...
            self.mode,
            "repetition_{r:02}".format(r=rep),
        )

    def microbench_json_path(self):
        # output/
        # expr_microbenchmark/
        # ppo2/
        # HalfCheetahBulletEnv-v0/
        # mode_microbench_rlscope_python_annotation.uninstrumented/
        # microbench_rlscope_python_annotation/
        # microbench_rlscope_python_annotation.json

        # self.quick_expr.out_dir,
        # self.algo,
        # self.env,
        # self.config_suffix,
        return _j(
            self.out_dir(),
            self.mode,
            "{mode}.json".format(mode=self.mode),
        )

    def rlscope_directories(self):
        """
        Return all --rlscope-directories whose runs are completed.
        """
        rlscope_directories = []
        for rep in range(1, self.expr.args.repetitions+1):
            if not self.already_ran(rep):
                continue
            rlscope_directory = self.repetition_out_dir(rep)
            rlscope_directories.append(rlscope_directory)
        return rlscope_directories

class ExprMicrobenchmark:
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
        # parser.add_argument(
        #     '--bullet',
        #     action='store_true',
        #     help='Limit environments to physics-based Bullet environments')
        # parser.add_argument(
        #     '--atari',
        #     action='store_true',
        #     help='Limit environments to Atari Pong environment')
        # parser.add_argument(
        #     '--lunar',
        #     action='store_true',
        #     help='Limit environments to LunarLander environments (i.e. LunarLanderContinuous-v2, LunarLander-v2)')

        # parser.add_argument('--mode', help='mode of execution for rl-baselines-zoo enjoy.py microbenchmark mode',
        #                     choices=[
        #                         # 'default',
        #                         'microbench_rlscope_python_annotation',
        #                         'microbench_rlscope_clib_interception_simulator',
        #                         'microbench_rlscope_clib_interception_tensorflow',
        #                     ])
        parser.add_argument(
            '--repetitions',
            type=int,
            default=5)
        parser.add_argument(
            '--iterations',
            type=int,
            # (HalfCheetah, ppo2):
            # - iterations=10**4:        Python->TensorFlow: 27 +/- 1.5
            # - iterations=$((2*10**4)):
            default=10**4)
        parser.add_argument(
            '--env',
            # required=True,
        )
        parser.add_argument(
            '--algo',
            # required=True,
        )
        parser.add_argument('--algo-env-group',
                            choices=expr_config.ALGO_ENV_GROUP_CHOICES,
                            help=textwrap.dedent("""
            Only run a specific "experiment".
            i.e. only run (algo, env) combinations needed for a specific graph.
            
            Default: run all (algo, env) combinations for all experiments.
            """))

        # parser.add_argument(
        #     '--compute-microbench',
        #     action='store_true',
        # )

        # parser.add_argument(
        #     '--plot',
        #     action='store_true')
        # parser.add_argument(
        #     '--instrumented',
        #     help="Run in fully instrumented mode (needed for creating \"Overhead correction\" figure)",
        #     action='store_true')

        self.args, self.extra_argv = parser.parse_known_args(self.argv)

        self.init_configs()

        # if self.args.plot:
        #     self.do_plot()
        # else:
        self.do_run()

    def do_plot(self):
        """
        Create bar graph of total training time for each experiment we run:
        rlscope_dirs = find_rlscope_dirs(self.root_dir)
        $ rls-run --task TotalTrainingTimePlot --rlscope-directories rlscope-dirs
        Read data-frame like:
          algo, env,         x_field, total_training_time_sec
          ppo2, HalfCheetah, ...,     ...
          ppo2, HalfCheetah, ...,     ...
        """

        # TODO:
        #
        # PyprofOverheadTask:
        # - just_pyprof_annotations
        # - just_pyprof_interceptions
        # - uninstrumented

        task = 'TotalTrainingTimeTask'

        rlscope_directories = []
        for config in self.configs:
            config_dirs = config.rlscope_directories()
            rlscope_directories.extend(config_dirs)

        if len(rlscope_directories) == 0:
            log_missing_files(self, task, files={
                'rlscope_directories': rlscope_directories,
            })
            return

        plot_dir = self.plot_dir
        if not self.quick_expr.args.dry_run:
            os.makedirs(plot_dir, exist_ok=True)
        cmd = ['rls-run',
               '--directory', plot_dir,
               '--task', task,
               '--uninstrumented-directory', json.dumps(rlscope_directories),
               ]
        add_rlscope_analyze_flags(cmd, self.quick_expr.args)
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

    def conf(self, algo, env, mode, config):
        config_suffix = self.get_config_suffix(mode, config)
        return self.config_suffix_to_obj[algo][env][mode][config_suffix]
        # return self.config_suffix_to_obj[algo][env][mode][config]

    def pyprof_overhead_dir(self, algo, env, mode):
        return _j(
            self.out_dir,
            algo,
            env,
            mode,
            "iterations_{i}".format(i=self.args.iterations),
            "pyprof_overhead")

    def pyprof_overhead_logfile(self, algo, env, mode):
        task = "PyprofOverheadTask"
        logfile = _j(
            self.pyprof_overhead_dir(algo, env, mode),
            self._logfile_basename(task),
        )
        return logfile

    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def is_interception_config(self, config):
        return re.search(r'interception', config)

    def is_annotation_config(self, config):
        return re.search(r'annotation', config)

    def microbench_add_per_call_us(self, algo, env, mode, config):
        assert config != 'uninstrumented'

        # Python annotations: read the total number of operations
        # Python->CLIB: read the total number of Python->C interceptions.

        """
        PSEUDOCODE:
        if 'pyprof_annotation_overhead_per_call_us' in js:
            return
        js['pyprof_annotation_overhead_per_call_us'] = js['iterations_total_sec'] / ins_df['total_num_calls'}
        """

        if self.quick_expr.args.dry_run:
            return

        instrumented_json_path = self.conf(algo, env, mode, config).microbench_json_path()
        js = MicrobenchmarkOverheadJSON(instrumented_json_path)
        if 'iterations_total_num_calls' in js:
            return False

        # uninstrumented_json_path = self.conf(algo, env, mode, 'uninstrumented').microbench_json_path()

        instrumented_rlscope_dirs = self.conf(algo, env, mode, config).rlscope_directories()
        uninstrumented_rlscope_dirs = self.conf(algo, env, mode, 'uninstrumented').rlscope_directories()

        if self.is_interception_config(config):
            compute_overhead_func = PyprofOverheadParser.compute_interception_overhead
        elif self.is_annotation_config(config):
            compute_overhead_func = PyprofOverheadParser.compute_annotation_overhead
        else:
            raise NotImplementedError()
        pyprof_overhead_calc = compute_overhead_func(
            uninstrumented_rlscope_dirs,
            instrumented_rlscope_dirs,
            debug=self.quick_expr.args.debug,
            debug_single_thread=self.quick_expr.args.debug_single_thread,
        )

        js['iterations_total_num_calls'] = pyprof_overhead_calc.num_calls()
        js['overhead_per_call_us'] = (js['iterations_total_sec']*constants.USEC_IN_SEC)/js['iterations_total_num_calls']
        js.dump()
        return True

    def compute_microbench_pyprof_overhead(self, algo, env, mode, config):
        assert config != 'uninstrumented'

        if self.quick_expr.args.dry_run:
            return

        self.microbench_add_per_call_us(algo, env, mode, config)

        uninstrumented_json_path = self.conf(algo, env, mode, 'uninstrumented').microbench_json_path()
        instrumented_json_path = self.conf(algo, env, mode, config).microbench_json_path()

        if self.quick_expr.args.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'uninstrumented_json_path': uninstrumented_json_path,
                    'instrumented_json_path': instrumented_json_path,
                })))

        if not _e(uninstrumented_json_path) or \
           not _e(instrumented_json_path):
            logger.info(textwrap.dedent("""
                    {klass}: SKIP compute_microbench_pyprof_overhead;
                    Files present so far:
                    {files}
                    """).format(
                klass=self.__class__.__name__,
                files=textwrap.indent(pprint.pformat({
                    'uninstrumented_json_path': uninstrumented_json_path,
                    'instrumented_json_path': instrumented_json_path,
                }), prefix='  '),
            ))
            return

        directory = _d(instrumented_json_path)
        base = self.config_to_overhead_basename[config]
        json_path = _j(directory, base)
        field_name = self.config_to_overhead_field_name[config]
        js = parse_microbench_overhead_js(
            field_name=field_name,
            uninstrumented_json_path=uninstrumented_json_path,
            instrumented_json_path=instrumented_json_path)
        logger.info("Output json @ {path}".format(path=json_path))
        do_dump_json(js, json_path)

    def compute_pyprof_overhead(self, algo, env, mode, config):
        task = "PyprofOverheadTask"

        # if self.quick_expr.args.debug:
        #     logger.info("log = {msg}".format(
        #         msg=pprint_msg({
        #             'iterations': self.iterations,
        #         })))

        # for iters in self.iterations:
        # uninstrumented_directories = self.conf(algo, env, mode, 'uninstrumented_calibration').rlscope_directories()
        # pyprof_annotations_directories = self.conf(algo, env, mode, 'just_pyprof_annotations_calibration').rlscope_directories()
        # pyprof_interceptions_directories = self.conf(algo, env, mode, 'just_pyprof_interceptions_calibration').rlscope_directories()
        uninstrumented_directories = self.conf(algo, env, mode, 'uninstrumented').rlscope_directories()
        # pyprof_annotations_directories = self.conf(algo, env, mode, 'just_pyprof_annotations').rlscope_directories()
        # FAIL: cannot find key:
        # 'mode_microbench_rlscope_python_annotation.config_just_pyprof_interceptions'
        # NOTE: we need to make PyprofOverheadTask JUST output annotation overhead WITHOUT requiring interceptions runs for the same config.
        # Q: any reason to output them at the same time...?
        # pyprof_interceptions_directories = self.conf(algo, env, mode, 'just_pyprof_interceptions').rlscope_directories()

        pyprof_overhead_directories = self.conf(algo, env, mode, config).rlscope_directories()
        if self.quick_expr.args.debug:
            logger.info("log = {msg}".format(
                msg=pprint_msg({
                    'uninstrumented_directories': uninstrumented_directories,
                    'pyprof_overhead_directories': pyprof_overhead_directories,
                    # 'pyprof_annotations_directories': pyprof_annotations_directories,
                    # 'pyprof_interceptions_directories': pyprof_interceptions_directories,
                })))

        if len({
            len(uninstrumented_directories),
            len(pyprof_overhead_directories),
            # len(pyprof_interceptions_directories),
        }) != 1:
            log_missing_files(self, task=task, files={
                'uninstrumented_directories': uninstrumented_directories,
                'pyprof_overhead_directories': pyprof_overhead_directories,
                # 'pyprof_annotations_directories': pyprof_annotations_directories,
                # 'pyprof_interceptions_directories': pyprof_interceptions_directories,
            })
            return

        directory = self.pyprof_overhead_dir(algo, env, mode)
        if not self.quick_expr.args.dry_run:
            os.makedirs(directory, exist_ok=True)
        cmd = ['rls-run',
               '--directory', directory,
               '--task', task,
               '--uninstrumented-directory', json.dumps(uninstrumented_directories),
               # '--pyprof-annotations-directory', json.dumps(pyprof_annotations_directories),
               # '--pyprof-interceptions-directory', json.dumps(pyprof_interceptions_directories),
               ]
        if self.is_annotation_config(config):
            cmd.extend([
                '--pyprof-annotations-directory', json.dumps(pyprof_overhead_directories),
            ])
        elif self.is_interception_config(config):
            cmd.extend([
                '--pyprof-interceptions-directory', json.dumps(pyprof_overhead_directories),
            ])
        else:
            raise NotImplementedError()
        add_rlscope_analyze_flags(cmd, self.quick_expr.args)
        cmd.extend(self.extra_argv)

        logfile = self.pyprof_overhead_logfile(algo, env, mode)
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.quick_expr.args.dry_run,
            skip_error=self.quick_expr.args.skip_error,
            debug=self.quick_expr.args.debug)

    def get_config_suffix(self, mode, config):
        config_suffix = "mode_{mode}.{config}".format(
            mode=mode,
            config=config,
        )
        return config_suffix

    def init_configs(self):
        self.configs = []
        self.config_suffix_to_obj = dict()
        self.stable_baselines_algo_env = expr_config.stable_baselines_gather_algo_env_pairs(
            algo=self.args.algo,
            env_id=self.args.env,
            # bullet=self.args.bullet,
            # atari=self.args.atari,
            # lunar=self.args.lunar,
            algo_env_group=self.args.algo_env_group,
            debug=self.quick_expr.args.debug,
        )

        # MODES = [
        #     'default',
        #     'microbench_rlscope_python_annotation',
        #     'microbench_rlscope_clib_interception_simulator',
        #     'microbench_rlscope_clib_interception_tensorflow',
        # ]

        # algo_env_configs = []
        # def add_calibration_config(**common_kwargs):
        #     # config_kwargs = dict(common_kwargs)
        #     # config_kwargs['long_run'] = True
        #     # config = ExprMicrobenchmarkConfig(**config_kwargs)
        #     # assert not config.is_calibration
        #     # if self.args.calibration_mode == 'validation':
        #     #     # *_calibration folders are used for computing average overhead.
        #     #     # Folders without the *_calibration are the runs we subtract the averages from.
        #     #     algo_env_configs.append(config)
        #
        #     config_suffix = "{suffix}_calibration".format(
        #         suffix=common_kwargs['config_suffix'],
        #     )
        #     calibration_config_kwargs = dict(common_kwargs)
        #     calibration_config_kwargs.update(dict(
        #         # Disable tfprof: CUPTI and LD_PRELOAD.
        #         config_suffix=config_suffix,
        #         # long_run=False,
        #     ))
        #     calibration_config = ExprMicrobenchmarkConfig(**calibration_config_kwargs)
        #     # assert calibration_config.is_calibration
        #     algo_env_configs.append(calibration_config)

        self.config_to_overhead_field_name = {
            'just_pyprof_annotations': 'pyprof_annotation_overhead_per_call_us',
            'just_pyprof_interceptions': 'pyprof_interception_overhead_per_call_us',
            # ... : 'cupti_overhead_per_call_us',
            # ... : 'interception_overhead_per_call_us',
        }

        self.config_to_overhead_basename = {

            # category_events.json
            # cupti_overhead.json
            # LD_PRELOAD_overhead.json

            # category_events.json
            # category_events.python_annotation.json
            # category_events.python_clib_interception.json

            # microbench_rlscope_clib_interception_simulator.json
            # microbench_rlscope_clib_interception_tensorflow.json
            # microbench_rlscope_python_annotation.json

            'just_pyprof_annotations': 'category_events.python_annotation.json',
            'just_pyprof_interceptions': 'category_events.python_clib_interception.json',

        }

        def mk_config_uninstrumented(algo, env, mode):
            config = 'uninstrumented'
            return ExprMicrobenchmarkConfig(
                expr=self,
                algo=algo,
                env=env,
                mode=mode,
                config=config,
                rlscope_prof_config='uninstrumented',
                config_suffix=self.get_config_suffix(mode=mode, config=config),
                # Disable ALL pyprof/tfprof stuff.
                script_args=['--rlscope-disable'],
            )
        def mk_config_just_pyprof_annotations(algo, env, mode):

            # PyprofOverheadTask: Python->C-lib event tracing, and operation annotation overhead correction.
            config = 'just_pyprof_annotations'
            return ExprMicrobenchmarkConfig(
                expr=self,
                algo=algo,
                env=env,
                mode=mode,
                config=config,
                rlscope_prof_config='uninstrumented',
                config_suffix=self.get_config_suffix(mode=mode, config=config),
                # Only enable GPU/C-lib event collection, not operation annotations.
                script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-interceptions'],
            )
        def mk_config_just_pyprof_interceptions(algo, env, mode):
            config = 'just_pyprof_interceptions'
            return ExprMicrobenchmarkConfig(
                expr=self,
                algo=algo,
                env=env,
                mode=mode,
                config=config,
                rlscope_prof_config='uninstrumented',
                config_suffix=self.get_config_suffix(mode=mode, config=config),
                # Only enable operation annotations, not GPU/C-lib event collection.
                script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-annotations'],
            )
        # add_calibration_config(
        #     expr=self,
        #     algo=algo,
        #     env=env,
        #     rlscope_prof_config='uninstrumented',
        #     # Disable tfprof: CUPTI and LD_PRELOAD.
        #     config_suffix='just_pyprof_annotations',
        #     # Only enable GPU/C-lib event collection, not operation annotations.
        #     script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-interceptions'],
        # )
        # add_calibration_config(
        #     expr=self,
        #     algo=algo,
        #     env=env,
        #     rlscope_prof_config='uninstrumented',
        #     # Disable tfprof: CUPTI and LD_PRELOAD.
        #     config_suffix='just_pyprof_interceptions',
        #     # Only enable operation annotations, not GPU/C-lib event collection.
        #     script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-annotations'],
        # )
        # # Entirely uninstrumented configuration; we use this in many of the overhead calculations to determine
        # # how much training time is attributable to the enabled "feature" (e.g. CUPTI activities).
        # add_calibration_config(
        #     expr=self,
        #     algo=algo,
        #     env=env,
        #     rlscope_prof_config='uninstrumented',
        #     config_suffix='uninstrumented',
        #     # Disable ALL pyprof/tfprof stuff.
        #     script_args=['--rlscope-disable'],
        # )

        config_suffix_to_mk_config = {
            'just_pyprof_interceptions': mk_config_just_pyprof_interceptions,
            'just_pyprof_annotations': mk_config_just_pyprof_annotations,
            'uninstrumented': mk_config_uninstrumented,
        }
        mode_to_config_suffix = {
            'microbench_rlscope_python_annotation': [
                'just_pyprof_annotations',
                'uninstrumented',
            ],
            'microbench_rlscope_clib_interception_simulator': [
                'just_pyprof_interceptions',
                'uninstrumented',
            ],
            'microbench_rlscope_clib_interception_tensorflow': [
                'just_pyprof_interceptions',
                'uninstrumented',
            ],
        }
        for algo, env in self.stable_baselines_algo_env:
            for mode in mode_to_config_suffix.keys():
                for config_suffix in mode_to_config_suffix[mode]:
                    mk_config = config_suffix_to_mk_config[config_suffix]
                    config = mk_config(algo, env, mode)
                    self.configs.append(config)

        for config in self.configs:
            mk_dict_tree(self.config_suffix_to_obj, [config.algo, config.env, config.mode])
            assert config.config_suffix not in self.config_suffix_to_obj[config.algo][config.env][config.mode]
            self.config_suffix_to_obj[config.algo][config.env][config.mode][config.config_suffix] = config

            # if algo not in self.config_suffix_to_obj:
            #     self.config_suffix_to_obj[algo] = dict()
            # if env not in self.config_suffix_to_obj[algo]:
            #     self.config_suffix_to_obj[algo][env] = dict()
            # if mode not in self.config_suffix_to_obj[algo][env]:
            #     self.config_suffix_to_obj[algo][env][mode] = dict()
            # assert config.config_suffix not in self.config_suffix_to_obj[algo][env][mode]
            # self.config_suffix_to_obj[algo][env][mode][config.config_suffix] = config

        # for mode in MODES:
        #     for algo, env in self.stable_baselines_algo_env:
        #         rlscope_prof_config = 'full'
        #         # if self.args.instrumented:
        #         #     rlscope_prof_config = 'full'
        #         # else:
        #         #     rlscope_prof_config = 'uninstrumented'
        #         config = ExprMicrobenchmarkConfig(
        #             expr=self,
        #             algo=algo,
        #             env=env,
        #             mode=mode,
        #             rlscope_prof_config=rlscope_prof_config,
        #         )
        #         self.configs.append(config)

        logger.info("configs: " + pprint_msg(self.configs))
        logger.info("config_suffix_to_obj: " + pprint_msg(self.config_suffix_to_obj))
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

        # for algo, env in self.stable_baselines_algo_env:
            # self.compute_cupti_scaling_overhead(algo, env)
            # self.compute_cupti_overhead(algo, env)
            # self.compute_LD_PRELOAD_overhead(algo, env)
            # self.compute_pyprof_overhead(algo, env)
            # self._pool.submit(
            #     get_func_name(self, 'compute_cupti_scaling_overhead'),
            #     self.compute_cupti_scaling_overhead,
            #     algo, env,
            #     sync=self.quick_expr.args.debug_single_thread,
            # )
            # self._pool.submit(
            #     get_func_name(self, 'compute_cupti_overhead'),
            #     self.compute_cupti_overhead,
            #     algo, env,
            #     sync=self.quick_expr.args.debug_single_thread,
            # )
            # self._pool.submit(
            #     get_func_name(self, 'compute_LD_PRELOAD_overhead'),
            #     self.compute_LD_PRELOAD_overhead,
            #     algo, env,
            #     sync=self.quick_expr.args.debug_single_thread,
            # )
        for config in self.configs:

            if config.config == 'uninstrumented':
                continue

            self._pool.submit(
                get_func_name(self, 'compute_pyprof_overhead'),
                self.compute_pyprof_overhead,
                config.algo, config.env, config.mode, config.config,
                sync=self.quick_expr.args.debug_single_thread,
            )

            # if self.args.compute_microbench:
            # Q: Is this function making microbench json empty...?
            self._pool.submit(
                get_func_name(self, 'compute_microbench_pyprof_overhead'),
                self.compute_microbench_pyprof_overhead,
                config.algo, config.env, config.mode, config.config,
                sync=self.quick_expr.args.debug_single_thread,
            )

        self._pool.shutdown()

        # Need to wait for all the compute_microbench_pyprof_overhead calls to finish before calling
        # microbench_add_per_call_us.

        # self._pool.submit(
        #     get_func_name(self, 'microbench_add_per_call_us'),
        #     self.microbench_add_per_call_us,
        #     config.algo, config.env, config.mode, config.config,
        #     sync=self.quick_expr.args.debug_single_thread,
        # )


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
            '--atari',
            action='store_true',
            help='Limit environments to Atari Pong environment')
        parser.add_argument(
            '--lunar',
            action='store_true',
            help='Limit environments to LunarLander environments (i.e. LunarLanderContinuous-v2, LunarLander-v2)')
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
        rlscope_dirs = find_rlscope_dirs(self.root_dir)
        $ rls-run --task TotalTrainingTimePlot --rlscope-directories rlscope-dirs
        Read data-frame like:
          algo, env,         x_field, total_training_time_sec
          ppo2, HalfCheetah, ...,     ...
          ppo2, HalfCheetah, ...,     ...
        """

        task = 'TotalTrainingTimeTask'

        rlscope_directories = []
        for config in self.configs:
            config_dirs = config.rlscope_directories()
            rlscope_directories.extend(config_dirs)

        if len(rlscope_directories) == 0:
            log_missing_files(self, task, files={
                'rlscope_directories': rlscope_directories,
            })
            return

        plot_dir = self.plot_dir
        if not self.quick_expr.args.dry_run:
            os.makedirs(plot_dir, exist_ok=True)
        cmd = ['rls-run',
               '--directory', plot_dir,
               '--task', task,
               '--uninstrumented-directory', json.dumps(rlscope_directories),
               ]
        add_rlscope_analyze_flags(cmd, self.quick_expr.args)
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
            bullet=self.args.bullet,
            atari=self.args.atari,
            lunar=self.args.lunar,
            debug=self.quick_expr.args.debug,
        )
        for algo, env in self.stable_baselines_algo_env:
            if self.args.instrumented:
                rlscope_prof_config = 'full'
            else:
                rlscope_prof_config = 'uninstrumented'
            config = ExprTotalTrainingTimeConfig(
                expr=self,
                algo=algo,
                env=env,
                rlscope_prof_config=rlscope_prof_config,
            )
            self.configs.append(config)
        logger.info("configs: " + pprint_msg(self.configs))
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
            gpu_activities_api_time_directories = self.conf(algo, env, 'gpu_activities_api_time_calibration').rlscope_directories(iters)
            interception_directories = self.conf(algo, env, 'interception_calibration').rlscope_directories(iters)
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
        cmd = ['rls-run',
               '--directory', directory,
               '--task', task,
               '--gpu-activities-api-time-directory', json.dumps(all_gpu_activities_api_time_directories),
               '--interception-directory', json.dumps(all_interception_directories),
               ]
        add_rlscope_analyze_flags(cmd, self.quick_expr.args)
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

            gpu_activities_directories = self.conf(algo, env, 'gpu_activities_calibration').rlscope_directories(iters)
            no_gpu_activities_directories = self.conf(algo, env, 'no_gpu_activities_calibration').rlscope_directories(iters)
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
            cmd = ['rls-run',
                   '--directory', directory,
                   '--task', task,
                   '--gpu-activities-directory', json.dumps(gpu_activities_directories),
                   '--no-gpu-activities-directory', json.dumps(no_gpu_activities_directories),
                   ]
            add_rlscope_analyze_flags(cmd, self.quick_expr.args)
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
            interception_directories = self.conf(algo, env, 'interception_calibration').rlscope_directories(iters)
            uninstrumented_directories = self.conf(algo, env, 'uninstrumented_calibration').rlscope_directories(iters)
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
            cmd = ['rls-run',
                   '--directory', directory,
                   '--task', task,
                   '--interception-directory', json.dumps(interception_directories),
                   '--uninstrumented-directory', json.dumps(uninstrumented_directories),
                   ]
            add_rlscope_analyze_flags(cmd, self.quick_expr.args)
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
            uninstrumented_directories = self.conf(algo, env, 'uninstrumented_calibration').rlscope_directories(iters)
            pyprof_annotations_directories = self.conf(algo, env, 'just_pyprof_annotations_calibration').rlscope_directories(iters)
            pyprof_interceptions_directories = self.conf(algo, env, 'just_pyprof_interceptions_calibration').rlscope_directories(iters)
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
            cmd = ['rls-run',
                   '--directory', directory,
                   '--task', task,
                   '--uninstrumented-directory', json.dumps(uninstrumented_directories),
                   '--pyprof-annotations-directory', json.dumps(pyprof_annotations_directories),
                   '--pyprof-interceptions-directory', json.dumps(pyprof_interceptions_directories),
                   ]
            add_rlscope_analyze_flags(cmd, self.quick_expr.args)
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
        # - skip plot if 0 --rlscope-directories
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
            rlscope_directories = config.rlscope_directories(iters)
            uninstrumented_directories = config_uninstrumented.rlscope_directories(iters)

            if ( len({len(rlscope_directories), len(uninstrumented_directories)}) != 1 ) or \
                len(pyprof_overhead_jsons) == 0 or \
                len(cupti_overhead_jsons) == 0 or \
                len(LD_PRELOAD_overhead_jsons) == 0:
                log_missing_files(self, task, files={
                    'rlscope_directories': rlscope_directories,
                    'uninstrumented_directories': uninstrumented_directories,
                    'pyprof_overhead_jsons': pyprof_overhead_jsons,
                    'cupti_overhead_jsons': cupti_overhead_jsons,
                    'LD_PRELOAD_overhead_jsons': LD_PRELOAD_overhead_jsons,
                })
                logger.info((
                    "{klass}: SKIP plotting iterations={iters}, config={config}, "
                    "since --rlscope-directories and --uninstrumented-directories haven't been generated yet.").format(
                    iters=iters,
                    config=config.to_string(),
                    klass=self.__class__.__name__
                ))
                continue

            # rls-run
            # --task CorrectedTrainingTimeTask
            # --directory $direc
            # --rlscope-directories "$(js_list.py output/rlscope_bench/debug_prof_overhead/config_${config_dir}_*/ppo2/HalfCheetahBulletEnv-v0)"
            # --uninstrumented-directories "$(js_list.py output/rlscope_bench/debug_prof_overhead/config_no_interception_*/ppo2/HalfCheetahBulletEnv-v0)"
            # --cupti-overhead-json output/rlscope_bench/debug_prof_overhead/results.config_full/cupti_overhead.json
            # --call-interception-overhead-json output/rlscope_bench/debug_prof_overhead/results.config_full/LD_PRELOAD_overhead.json
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
            cmd = ['rls-run',
                   '--directory', plot_dir,
                   '--task', task,

                   '--pyprof-overhead-json', pyprof_overhead_jsons[0],
                   '--cupti-overhead-json', cupti_overhead_jsons[0],
                   '--LD-PRELOAD-overhead-json', LD_PRELOAD_overhead_jsons[0],

                   '--rlscope-directories', json.dumps(rlscope_directories),
                   '--uninstrumented-directories', json.dumps(uninstrumented_directories),

                   '--rls-prof-config', config.rlscope_prof_config,
                   ]
            # cmd.extend(calibration_jsons.argv())
            add_rlscope_analyze_flags(cmd, self.quick_expr.args)
            cmd.extend(self.extra_argv)

            # get these args from forwarding extra_argv
            # --cupti-overhead-json
            # output/rlscope_bench/debug_prof_overhead/results.config_full/cupti_overhead.json
            # --call-interception-overhead-json
            # output/rlscope_bench/debug_prof_overhead/results.config_full/LD_PRELOAD_overhead.json
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
            algo=algo,
            env=env,
            rlscope_prof_config='interception',
            config_suffix='interception',
            script_args=['--rlscope-disable-pyprof'],
        )

        # CUPTIOverheadTask: CUPTI, and CUDA API stat-tracking overhead correction.
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            rlscope_prof_config='gpu-activities',
            config_suffix='gpu_activities',
            script_args=['--rlscope-disable-pyprof'],
        )
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            rlscope_prof_config='no-gpu-activities',
            config_suffix='no_gpu_activities',
            script_args=['--rlscope-disable-pyprof'],
        )
        # CUPTIScalingOverheadTask:
        config = ExprSubtractionValidationConfig(
            expr=self,
            algo=algo,
            env=env,
            rlscope_prof_config='gpu-activities-api-time',
            config_suffix='gpu_activities_api_time_calibration',
            script_args=['--rlscope-disable-pyprof'],
            long_run=True,
        )
        algo_env_configs.append(config)

        if self.args.calibration_mode == 'validation':
            # Evaluate: combined tfprof/pyprof overhead correction.
            # (i.e. full RL-Scope trace-collection).
            config = ExprSubtractionValidationConfig(
                expr=self,
                algo=algo,
                env=env,
                rlscope_prof_config='full',
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
                rlscope_prof_config='full',
                # Enable tfprof: CUPTI and LD_PRELOAD.
                config_suffix='just_tfprof',
                # DON'T enable pyprof.
                script_args=['--rlscope-disable-pyprof'],
                long_run=True,
            )
            algo_env_configs.append(config)

        if self.args.calibration_mode == 'validation':
            # Evaluate: pyprof overhead correction in isolation.
            config = ExprSubtractionValidationConfig(
                expr=self,
                algo=algo,
                env=env,
                rlscope_prof_config='uninstrumented',
                # Disable tfprof: CUPTI and LD_PRELOAD.
                config_suffix='just_pyprof',
                # Enable pyprof.
                script_args=['--rlscope-disable-tfprof'],
            )
            algo_env_configs.append(config)

        # PyprofOverheadTask: Python->C-lib event tracing, and operation annotation overhead correction.
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            rlscope_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_annotations',
            # Only enable GPU/C-lib event collection, not operation annotations.
            script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-interceptions'],
        )
        add_calibration_config(
            expr=self,
            algo=algo,
            env=env,
            rlscope_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_interceptions',
            # Only enable operation annotations, not GPU/C-lib event collection.
            script_args=['--rlscope-disable-tfprof', '--rlscope-disable-pyprof-annotations'],
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
        #     '--rls-prof-config',
        #     choices=['instrumented', 'full'],
        #     default='full')
        self.args, self.extra_argv = parser.parse_known_args(self.argv)

        self.init_configs()

        if self.args.plot:
            self.do_plot()
        else:
            self.do_run()

class IMLConfigDir:
    def __init__(self, rlscope_config_path):
        self.rlscope_config = RLScopeConfig(rlscope_config_path=rlscope_config_path)
        self.config_dir = self._as_config_dir(rlscope_config_path)

    def _as_config_dir(self, path):
        m = re.search(r'(?P<config_dir>.*/{CONFIG_RE})'.format(
            CONFIG_RE=CONFIG_RE),
            path)
        assert m
        config_dir = m.group('config_dir')
        assert is_config_dir(config_dir)
        return config_dir

    @property
    def algo(self):
        return self.rlscope_config.algo()

    @property
    def env(self):
        return self.rlscope_config.env()

    def repetition(self, allow_none=False, dflt=None):
        return self._repetition_from_config_dir(self.config_dir, allow_none=allow_none, dflt=dflt)

    def _config_component(self, path):
        m = re.search(r'(?P<config_component>{CONFIG_RE})'.format(
            CONFIG_RE=CONFIG_RE),
            path)
        config_component = m.group('config_component')
        return config_component

    @property
    def config(self):
        config = self._config_component(self.config_dir)
        config = re.sub(r'_repetition_\d+', '', config)
        return config

    def _repetition_from_config_dir(self, path, allow_none=False, dflt=None):
        assert is_config_dir(path)
        _check_has_config_dir(path)
        config_component = self._config_component(path)
        m = re.search(r'repetition_(?P<repetition>\d+)', config_component)
        assert m or allow_none
        if not m:
            return dflt
        repetition = int(m.group('repetition'))
        return repetition


class ExperimentDirectoryWalker:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self._init()

    def complete_repetitions(self, configs):
        """
        PSEUDOCODE:
        def complete_repetitions(self):
            r = start-repetition (0 or 1, or smallest we find.)
            end-repetition (largest repetition we find)
            while r <= end-repetition and r in algo_env_pairs and ( (r-1) no in algo_env_pairs or algo_env_pairs[r].issubset(algo_env_pairs[r-1]) ):
              reps.append( r )
              r += 1
            return reps
        """
        def reps_for_config(config):
            r = self.start_repetition(config)
            end_repetition = self.end_repetition(config)
            reps = set()
            while r <= end_repetition and \
                r in self.algo_env_pairs and \
                (
                    (r - 1) not in self.algo_env_pairs or \
                    self.algo_env_pairs[r-1][config].issubset(self.algo_env_pairs[r][config])
                ):
                reps.add(r)
                r += 1
            return reps
        all_reps = [reps_for_config(config) for config in configs]
        reps = functools.reduce(set.intersection, all_reps)
        return reps

    def start_repetition(self, config):
        """
        r = start-repetition (0 or 1, or smallest we find.)
        """
        min_r = None
        for r in self.algo_env_pairs.keys():
            if config in self.algo_env_pairs[r]:
                min_r = min(r, min_r) if min_r is not None else r
        assert min_r is not None
        return min_r

    def end_repetition(self, config):
        """
        end-repetition (largest repetition we find)
        """
        max_r = None
        for r in self.algo_env_pairs.keys():
            if config in self.algo_env_pairs[r]:
                max_r = max(r, max_r) if max_r is not None else r
        assert max_r is not None
        return max_r

    def get_rlscope_config_dir(self, config, algo, env, r):
        return self._rlscope_config_dir_dict[(config, algo, env, r)]

    def _init(self):
        self.rlscope_config_paths = [
            path for path in each_file_recursive(self.root_dir)
            if is_rlscope_config_file(path) and _b(_d(path)) != constants.DEFAULT_PHASE]
        for rlscope_config_path in self.rlscope_config_paths:
            _check_has_config_dir(rlscope_config_path)
            _check_one_config_dir(rlscope_config_path)

        self.algo_env_pairs = dict()
        self._rlscope_config_dir_dict = dict()
        for rlscope_config_path in self.rlscope_config_paths:
            rlscope_config_dir = IMLConfigDir(rlscope_config_path)
            config = rlscope_config_dir.config
            r = rlscope_config_dir.repetition(allow_none=True, dflt=0)
            algo = rlscope_config_dir.algo
            env = rlscope_config_dir.env
            self._rlscope_config_dir_dict[(config, algo, env, r)] = rlscope_config_dir
            algo_env = (algo, env)
            if r not in self.algo_env_pairs:
                self.algo_env_pairs[r] = dict()
            if config not in self.algo_env_pairs[r]:
                self.algo_env_pairs[r][config] = set()
            assert algo_env not in self.algo_env_pairs[r][config]
            self.algo_env_pairs[r][config].add(algo_env)

    def _all_algo_env_pairs(self, configs, reps):
        # for r in self.algo_env_pairs.keys():
        for r in reps:
            for config in configs:
                algo_env_pairs = self.algo_env_pairs[r][config]
                yield algo_env_pairs

    def all_configs(self):
        def _all_configs():
            for r in self.algo_env_pairs.keys():
                yield set(self.algo_env_pairs[r].keys())
        configs = functools.reduce(set.union, _all_configs())
        return configs

    def get_configs(self, configs):
        """
        config_dict = walker.get_algo_env_pairs(
            # Could require repetitions, or default to:
            # - largest number of repetitions with most (algo, env) pairs
            #   i.e. if bumping the repetitions doesn't LOSE any (algo, env) pairs, do it.
            repetitions=args.repetitions,
            configurations=['config_uninstrumented', 'config_full'])
        # Returns something like:
        {
            # Paths appear in (algo, env, repetition) order, and an (algo, env, repetition) MUST appear in ALL lists, or NO lists.
            'config_uninstrumented': ['path/to/AntBullet/config_uninstrumented_repetition_01', …]
            'config_full': ['path/to/AntBullet/config_full_repetition_01', …]
        }

        def get_algo_env_pairs(self):
            # NOTE: we need to take the intersection of (algo, env) pairs across all repetitions:
            #   INTERSECT { algo_env_pairs[r][config] } for r in range(repetitions), config in algo_env_pairs[r].keys()
            algo_env_pairs
            for config_dir in config_dirs:
                Algo, env = get_algo_env(config_dir)
                R = get_repetition(config_dir)
                Algo_env_pairs[r].add((algo, env))

            reps = complete_repetitions()
            algo_env_pairs = INTERSECT { algo_env_pairs[r] } for r in reps
            return algo_env_pairs
        """
        all_configs = self.all_configs()
        for config in configs:
            assert config in all_configs

        # All the "complete repetitions" the configs have in common.
        # For a config, a repetition is complete if all repetitions for that config have the same (algo, env) pairs measured.
        reps = self.complete_repetitions(configs)
        # The (algo, env) pairs these configs' repetitions have in common.
        algo_env_pairs_keep = functools.reduce(set.intersection, self._all_algo_env_pairs(configs, reps))
        config_dict = dict()
        for algo, env in algo_env_pairs_keep:
            for r in reps:
                for config in configs:
                    rlscope_config_dir = self.get_rlscope_config_dir(config, algo, env, r)
                    if config not in config_dict:
                        config_dict[config] = []
                    config_dict[config].append(rlscope_config_dir.config_dir)
        return config_dict

    def is_config_dir(self, path):
        return is_config_dir(path)

    def _as_config_dir(self, path):
        m = re.search(r'(?P<config_dir>.*/{CONFIG_RE})'.format(
            CONFIG_RE=CONFIG_RE),
            path)
        assert m
        config_dir = m.group('config_dir')
        assert self.is_config_dir(config_dir)
        return config_dir

class ExprPlotFig:
    def __init__(self, quick_expr, argv):
        self.quick_expr = quick_expr
        self.argv = argv
        self._pool = ForkedProcessPool(name='{klass}.pool'.format(
            klass=self.__class__.__name__))

    @property
    def out_dir(self):
        return _j(self.quick_expr.out_dir)

    def plot_dir(self, fig):
        return _j(self.out_dir, fig)

    def plot_logfile(self, fig):
        logfile = _j(self.plot_dir(fig), "logfile.out")
        return logfile

    def run(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # parser.add_argument(
        #     '--bullet',
        #     action='store_true',
        #     help='Limit environments to physics-based Bullet environments')
        # parser.add_argument(
        #     '--atari',
        #     action='store_true',
        #     help='Limit environments to Atari Pong environment')
        # parser.add_argument(
        #     '--lunar',
        #     action='store_true',
        #     help='Limit environments to LunarLander environments (i.e. LunarLanderContinuous-v2, LunarLander-v2)')

        # parser.add_argument(
        #     '--env',
        #     )
        # parser.add_argument(
        #     '--algo',
        #     )
        # parser.add_argument(
        #     '--plot',
        #     action='store_true')
        # parser.add_argument(
        #     '--instrumented',
        #     help="Run in fully instrumented mode (needed for creating \"Overhead correction\" figure)",
        #     action='store_true')

        parser.add_argument(
            '--root-dir',
            help="Root directory that contains ALL the rlscope-directories and config-dirs as children; we walk this recursively.",
            required=True)

        parser.add_argument(
            '--fig',
            choices=[
                'fig_13_overhead_correction',
            ],
            help="Which paper figure are we making? (naming matches filenames used in Latex)",
            required=True,
        )

        CalibrationJSONs.add_argparse(parser, required=True)

        self.args, self.extra_argv = parser.parse_known_args(self.argv)

        self.do_run()

    def plot_dir(self, fig):
        return _j(self.out_dir, fig)

    def plot_logfile(self, fig):
        logfile = _j(self.plot_dir(fig), "logfile.out")
        return logfile

    def rlscope_analyze_cmdline(self, task, argv):
        plot_dir = self.plot_dir(self.args.fig)
        calibration_jsons = CalibrationJSONs.from_obj(self.args)
        cmd = ['rls-run',
               '--directory', plot_dir,
               '--task', task,

               # '--pyprof-overhead-json', self.args.pyprof_overhead_json,
               # '--cupti-overhead-json', self.args.cupti_overhead_json,
               # '--LD-PRELOAD-overhead-json', self.args.LD_PRELOAD_overhead_json,

               ]
        cmd.extend(calibration_jsons.argv())
        cmd.extend(argv)
        add_rlscope_analyze_flags(cmd, self.quick_expr.args)
        cmd.extend(self.extra_argv)
        return cmd

    def do_run(self):
        plot_dir = self.plot_dir(self.args.fig)
        if not self.quick_expr.args.dry_run:
            os.makedirs(plot_dir, exist_ok=True)
        walker = ExperimentDirectoryWalker(root_dir=self.args.root_dir)
        argv = None
        task = None
        if self.args.fig == 'fig_13_overhead_correction':
            configs = walker.get_configs(configs=['config_uninstrumented', 'config_full'])
            argv = [
                '--rls-prof-config', 'full',
                '--rlscope-directories', json.dumps(configs['config_full']),
                '--uninstrumented-directories', json.dumps(configs['config_uninstrumented']),
            ]
            task = 'CorrectedTrainingTimeTask'
        else:
            raise NotImplementedError()

        cmd = self.rlscope_analyze_cmdline(task=task, argv=argv)

        logfile = self.plot_logfile(self.args.fig)
        if self.quick_expr.args.debug:
            logger.info("Logging to file {path}".format(
                path=logfile))
        expr_run_cmd(
            cmd=cmd,
            to_file=logfile,
            # Always re-run plotting script?
            # replace=True,
            dry_run=self.quick_expr.args.dry_run,
            skip_error=self.quick_expr.args.skip_error,
            debug=self.quick_expr.args.debug)


def rep_suffix(rep):
    assert rep is not None
    return "_repetition_{rep:02}".format(rep=rep)

def add_rlscope_analyze_flags(cmd, args):
    if args.debug:
        cmd.append('--debug')
    if args.debug_memoize:
        cmd.append('--debug-memoize')
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

def _check_one_config_dir(path):
    config_matches = re.findall(CONFIG_RE, path)
    if len(config_matches) > 1:
        raise RuntimeError("Saw multiple config_* components in {path}; not sure which one to use; choices: \n{choices}".format(
            path=path,
            choices=pprint.pformat(config_matches)))

def _check_has_config_dir(path):
    if not re.search(CONFIG_RE, path):
        raise RuntimeError("Saw no config_* component in {path}".format(
            path=path,
        ))

def mk_dict_tree(dic, keys):
    d = dic
    for key in keys:
        if key not in d:
            d[key] = dict()
        d = d[key]


if __name__ == '__main__':
    main()
