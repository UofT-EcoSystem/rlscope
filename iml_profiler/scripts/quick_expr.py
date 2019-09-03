from iml_profiler.profiler import iml_logging
import argparse
import pprint
import textwrap
import os
from os import environ as ENV
import json

from os.path import join as _j, abspath as _a, exists as _e, dirname as _d, basename as _b

from iml_profiler.parser.common import *
from iml_profiler.experiment.util import tee, expr_run_cmd, expr_already_ran

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
        return _j(self.args.dir, "expr_{expr}".format(expr=self.args.expr))

    def expr_interception_iters(self, extra_argv):
        expr_interception_iters = ExprInterceptionIters(quick_expr=self, argv=extra_argv)
        expr_interception_iters.run()

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
            'interception_iters',
        ],
        required=True,
        help=textwrap.dedent("""
        --expr my_expr will run experiment QuickExpr.expr_my_expr().
        
        interception_iters:
            See how much unaccounted for overhead % varies as we increase 
            the number of training loop iterations.
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

    args, extra_argv = parser.parse_known_args()
    quick_expr = QuickExpr(args, extra_argv)
    quick_expr.run()


class ExprInterceptionItersConfig:
    def __init__(self, expr_interception_iters, iml_prof_config, config_suffix, script_args=[]):
        self.expr_interception_iters = expr_interception_iters
        self.quick_expr = self.expr_interception_iters.quick_expr
        # $ iml-prof --config ${iml_prof_config}
        self.iml_prof_config = iml_prof_config
        # $ python train.py --iml-directory config_${config_suffix}
        self.config_suffix = config_suffix
        self.script_args = script_args

    def out_dir(self, rep, iters):
        return _j(
            self.quick_expr.out_dir,
            "config_{config_suffix}_iters_{iters}{rep}".format(
                config_suffix=self.config_suffix,
                iters=iters,
                rep=rep_suffix(rep),
            ))

    def to_string(self):
        return ("ExprInterceptionItersConfig("
                "iml_prof_config='{iml_prof_config}'"
                ", config_suffix='{config_suffix}'"
                ")").format(
            iml_prof_config=self.iml_prof_config,
            config_suffix=self.config_suffix,
        )

    # @property
    # def args(self):
    #     return self.quick_expr.args

    def logfile(self, rep, iters):
        logfile = _j(self.out_dir(rep, iters), "logfile.out")
        return logfile

    def run(self, rep, iters):
        cmd = ['iml-prof',
               '--config', self.iml_prof_config,
               'python', 'train.py',

               '--iml-directory', _a(self.out_dir(rep, iters)),
               '--iml-max-timesteps', iters,
               '--iml-training-progress',

               '--algo', self.expr_interception_iters.args.algo,
               '--env', self.expr_interception_iters.args.env,

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
        for rep in range(1, self.expr_interception_iters.args.repetitions+1):
            if not self.already_ran(rep, iters):
                continue
            iml_directory = self.out_dir(rep, iters)
            iml_directories.append(iml_directory)
        return iml_directories

class ExprInterceptionIters:
    def __init__(self, quick_expr, argv):
        self.quick_expr = quick_expr
        self.argv = argv
        # 1 min, 2 min, 4 min

    def plot_dir(self, config, iters):
        return _j(
            self.quick_expr.out_dir,
            "config_{conf}_iters_{iters}".format(
                conf=config.config_suffix,
                iters=iters,
            ))

    def plot_logfile(self, config, iters):
        logfile = _j(self.plot_dir(config, iters), "logfile.out")
        return logfile

    def do_plot(self):
        for config in self.configs:
            if config.config_suffix != 'uninstrumented':
                self.plot_config(config)

    def plot_config(self, config):
        # PROBLEM: we only want to call plot if there are "enough" files to plot a "configuration";
        # - skip plot if 0 --iml-directories
        # - skip plot if 0 --uninstrumented-directories
        assert config != self.config_uninstrumented
        for iters in self.iterations:
            iml_directories = config.iml_directories(iters)
            uninstrumented_directories = self.config_uninstrumented.iml_directories(iters)
            if len(iml_directories) == 0 or len(uninstrumented_directories) == 0:
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
            # --call-interception-overhead-json output/iml_bench/debug_prof_overhead/results.config_full/call_interception_overhead.json
            # --debug
            # --pdb
            # --debug-memoize

            plot_dir = self.plot_dir(config, iters)
            os.makedirs(plot_dir, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', plot_dir,
                   '--task', 'CorrectedTrainingTimeTask',
                   '--iml-directories', json.dumps(iml_directories),
                   '--uninstrumented-directories', json.dumps(uninstrumented_directories),
                   '--iml-prof-config', config.iml_prof_config,
                   ]
            if self.quick_expr.args.debug:
                cmd.extend('--debug')
            cmd.extend(self.extra_argv)

            # get these args from forwarding extra_argv
            # --cupti-overhead-json
            # output/iml_bench/debug_prof_overhead/results.config_full/cupti_overhead.json
            # --call-interception-overhead-json
            # output/iml_bench/debug_prof_overhead/results.config_full/call_interception_overhead.json
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
        for iters in self.iterations:
            for config in self.configs:
                for rep in range(1, self.args.repetitions+1):
                    config.run(rep, iters)

    def run(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--num-runs',
            type=int,
            default=3)
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
            '--env',
            default="HalfCheetahBulletEnv-v0")
        parser.add_argument(
            '--algo',
            default="ppo2")
        parser.add_argument(
            '--plot',
            action='store_true')
        # parser.add_argument(
        #     '--iml-prof-config',
        #     choices=['instrumented', 'full'],
        #     default='full')
        self.args, self.extra_argv = parser.parse_known_args(self.argv)

        self.iterations = [self.args.one_minute_iterations*(2**i) for i in range(self.args.num_runs)]

        script_args_disable_pyprof = ['--iml-disable-pyprof']
        self.config_uninstrumented = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            config_suffix='uninstrumented',
            script_args=script_args_disable_pyprof,
        )
        self.config_interception = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='interception',
            # BUG: I am INCORRECTLY subtracting too much here by subtracting libcupti overhead during an "interception" run;
            # we JUST want to subtract the LD_PRELOAD overhead here.
            # config_suffix='just_tfprof',
            config_suffix='interception',
            script_args=script_args_disable_pyprof,
        )
        self.config_full = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='full',
            # Enable CUPTI and LD_PRELOAD, but DON'T enable pyprof.
            config_suffix='just_tfprof',
            script_args=script_args_disable_pyprof,
        )
        self.config_pyprof = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            # Disable CUPTI and LD_PRELOAD, only enable pyprof.
            config_suffix='just_pyprof',
            # Enable pyprof.
            script_args=[],
        )
        self.configs = [
            self.config_uninstrumented,
            self.config_interception,
            self.config_full,
            self.config_pyprof,
        ]

        if self.args.plot:
            self.do_plot()
        else:
            self.do_run()


def rep_suffix(rep):
    assert rep is not None
    return "_repetition_{rep:02}".format(rep=rep)

if __name__ == '__main__':
    main()
