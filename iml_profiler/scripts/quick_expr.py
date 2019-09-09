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

    def cupti_overhead_dir(self, iters):
        return _j(
            self.quick_expr.out_dir,
            "cupti_overhead_iters_{iters}".format(
                iters=iters,
            ))

    def cupti_overhead_logfile(self, iters):
        task = "CUPTIOverheadTask"
        logfile = _j(
            self.cupti_overhead_dir(iters),
            self._logfile_basename(task),
        )
        return logfile

    def LD_PRELOAD_overhead_dir(self, iters):
        return _j(
            self.quick_expr.out_dir,
            "LD_PRELOAD_overhead_iters_{iters}".format(
                iters=iters,
            ))

    def LD_PRELOAD_overhead_logfile(self, iters):
        task = "CallInterceptionOverheadTask"
        logfile = _j(
            self.LD_PRELOAD_overhead_dir(iters),
            self._logfile_basename(task),
        )
        return logfile

    def pyprof_overhead_dir(self, iters):
        return _j(
            self.quick_expr.out_dir,
            "pyprof_overhead_iters_{iters}".format(
                iters=iters,
            ))

    def pyprof_overhead_logfile(self, iters):
        task = "PyprofOverheadTask"
        logfile = _j(
            self.pyprof_overhead_dir(iters),
            self._logfile_basename(task),
        )
        return logfile


    def _logfile_basename(self, task):
        return "{task}.logfile.out".format(task=task)

    def _glob_json_files(self, direc):
        json_paths = glob("{direc}/*.json".format(
            direc=direc))
        return json_paths


    def _log_missing_files(self, task, iters, files):
        logging.info(textwrap.dedent("""
                {klass}: SKIP iml-analyze --task={task} with iterations={iters}; still need you to collect 
                some additional runs using "iml-quick-expr".
                Files present so far:
                {files}
                """).format(
            klass=self.__class__.__name__,
            task=task,
            iters=iters,
            files= textwrap.indent(pprint.pformat(files), prefix='  '),
        ))

    def compute_cupti_overhead(self):
        task = "CUPTIOverheadTask"
        for iters in self.iterations:
            gpu_activities_directories = self.conf('gpu_activities').iml_directories(iters)
            no_gpu_activities_directories = self.conf('no_gpu_activities').iml_directories(iters)
            if len(gpu_activities_directories) != len(no_gpu_activities_directories):
                self._log_missing_files(task=task, iters=iters, files={
                    'gpu_activities_directories': gpu_activities_directories,
                    'no_gpu_activities_directories': no_gpu_activities_directories,
                })
                continue

            directory = self.cupti_overhead_dir(iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(directory, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', directory,
                   '--task', task,
                   '--gpu-activities-directory', json.dumps(gpu_activities_directories),
                   '--no-gpu-activities-directory', json.dumps(no_gpu_activities_directories),
                   ]
            if self.quick_expr.args.debug:
                cmd.extend('--debug')
            cmd.extend(self.extra_argv)

            logfile = self.cupti_overhead_logfile(iters)
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def compute_LD_PRELOAD_overhead(self):
        task = "CallInterceptionOverheadTask"
        for iters in self.iterations:
            interception_directories = self.conf('interception').iml_directories(iters)
            uninstrumented_directories = self.conf('uninstrumented_calibration').iml_directories(iters)
            if len(interception_directories) != len(uninstrumented_directories):
                self._log_missing_files(task=task, iters=iters, files={
                    'interception_directories': interception_directories,
                    'uninstrumented_directories': uninstrumented_directories,
                })
                continue

            directory = self.LD_PRELOAD_overhead_dir(iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(directory, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', directory,
                   '--task', task,
                   '--interception-directory', json.dumps(interception_directories),
                   '--uninstrumented-directory', json.dumps(uninstrumented_directories),
                   ]
            if self.quick_expr.args.debug:
                cmd.extend('--debug')
            cmd.extend(self.extra_argv)

            logfile = self.LD_PRELOAD_overhead_logfile(iters)
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def compute_pyprof_overhead(self):
        task = "PyprofOverheadTask"
        for iters in self.iterations:
            uninstrumented_directories = self.conf('uninstrumented_calibration').iml_directories(iters)
            pyprof_annotations_directories = self.conf('just_pyprof_annotations').iml_directories(iters)
            pyprof_interceptions_directories = self.conf('just_pyprof_interceptions').iml_directories(iters)

            if len({
                len(uninstrumented_directories),
                len(pyprof_annotations_directories),
                len(pyprof_interceptions_directories),
            }) != 1:
                self._log_missing_files(task=task, iters=iters, files={
                    'uninstrumented_directories': uninstrumented_directories,
                    'pyprof_annotations_directories': pyprof_annotations_directories,
                    'pyprof_interceptions_directories': pyprof_interceptions_directories,
                })
                continue

            directory = self.pyprof_overhead_dir(iters)
            if not self.quick_expr.args.dry_run:
                os.makedirs(directory, exist_ok=True)
            cmd = ['iml-analyze',
                   '--directory', directory,
                   '--task', task,
                   '--uninstrumented-directory', json.dumps(uninstrumented_directories),
                   '--pyprof-annotations-directory', json.dumps(pyprof_annotations_directories),
                   '--pyprof-interceptions-directory', json.dumps(pyprof_interceptions_directories),
                   ]
            if self.quick_expr.args.debug:
                cmd.extend('--debug')
            cmd.extend(self.extra_argv)

            logfile = self.pyprof_overhead_logfile(iters)
            expr_run_cmd(
                cmd=cmd,
                to_file=logfile,
                # Always re-run plotting script?
                # replace=True,
                dry_run=self.quick_expr.args.dry_run,
                skip_error=self.quick_expr.args.skip_error,
                debug=self.quick_expr.args.debug)

    def plot_config(self, config):
        # PROBLEM: we only want to call plot if there are "enough" files to plot a "configuration";
        # - skip plot if 0 --iml-directories
        # - skip plot if 0 --uninstrumented-directories
        assert config != self.config_uninstrumented
        task = 'CorrectedTrainingTimeTask'
        for iters in self.iterations:
            iml_directories = config.iml_directories(iters)
            uninstrumented_directories = self.config_uninstrumented.iml_directories(iters)

            pyprof_overhead_jsons = self._glob_json_files(self.pyprof_overhead_dir(iters))
            assert len(pyprof_overhead_jsons) <= 1

            cupti_overhead_jsons = self._glob_json_files(self.cupti_overhead_dir(iters))
            assert len(cupti_overhead_jsons) <= 1

            LD_PRELOAD_overhead_jsons = self._glob_json_files(self.LD_PRELOAD_overhead_dir(iters))
            assert len(LD_PRELOAD_overhead_jsons) <= 1

            if ( len({len(iml_directories), len(uninstrumented_directories)}) != 1 ) or \
                len(pyprof_overhead_jsons) == 0 or \
                len(cupti_overhead_jsons) == 0 or \
                len(LD_PRELOAD_overhead_jsons) == 0:
                self._log_missing_files(task, iters, files={
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
            if self.quick_expr.args.debug:
                cmd.extend('--debug')
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
        for iters in self.iterations:
            for config in self.configs:
                for rep in range(1, self.args.repetitions+1):
                    config.run(rep, iters)

        self.compute_cupti_overhead()
        self.compute_LD_PRELOAD_overhead()
        self.compute_pyprof_overhead()

    def conf(self, config_suffix):
        return self.config_suffix_to_obj[config_suffix]

    def init_configs(self):
        self.configs = []

        # Entirely uninstrumented configuration (CALIBRATION runs);
        # NOTE: we need to re-run the uninstrumented configuration for the calibration runs, so that we can
        # see how well our calibration generalizes.
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            config_suffix='uninstrumented_calibration',
            # Disable ALL pyprof/tfprof stuff.
            script_args=['--iml-disable'],
        )
        self.configs.append(config)
        self.config_uninstrumented_calibration = config

        # Entirely uninstrumented configuration; we use this in many of the overhead calculations to determine
        # how much training time is attributable to the enabled "feature" (e.g. CUPTI activities).
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            config_suffix='uninstrumented',
            # Disable ALL pyprof/tfprof stuff.
            script_args=['--iml-disable'],
        )
        self.configs.append(config)
        self.config_uninstrumented = config

        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='interception',
            config_suffix='interception',
            script_args=['--iml-disable-pyprof'],
        )
        self.configs.append(config)

        # CUPTIOverheadTask: CUPTI, and CUDA API stat-tracking overhead correction.
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='gpu-activities',
            config_suffix='gpu_activities',
            script_args=['--iml-disable-pyprof'],
        )
        self.configs.append(config)
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='no-gpu-activities',
            config_suffix='no_gpu_activities',
            script_args=['--iml-disable-pyprof'],
        )
        self.configs.append(config)

        # Evaluate: combined tfprof/pyprof overhead correction.
        # (i.e. full IML trace-collection).
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='full',
            # Enable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='full',
            # Enable pyprof.
            script_args=[],
        )
        self.configs.append(config)

        # Evaluate: tfprof overhead correction in isolation.
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='full',
            # Enable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_tfprof',
            # DON'T enable pyprof.
            script_args=['--iml-disable-pyprof'],
        )
        self.configs.append(config)

        # Evaluate: pyprof overhead correction in isolation.
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof',
            # Enable pyprof.
            script_args=['--iml-disable-tfprof'],
        )
        self.configs.append(config)

        # PyprofOverheadTask: Python->C-lib event tracing, and operation annotation overhead correction.
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_annotations',
            # Only enable GPU/C-lib event collection, not operation annotations.
            script_args=['--iml-disable-tfprof', '--iml-disable-pyprof-interceptions'],
        )
        self.configs.append(config)
        config = ExprInterceptionItersConfig(
            expr_interception_iters=self,
            iml_prof_config='uninstrumented',
            # Disable tfprof: CUPTI and LD_PRELOAD.
            config_suffix='just_pyprof_interceptions',
            # Only enable operation annotations, not GPU/C-lib event collection.
            script_args=['--iml-disable-tfprof', '--iml-disable-pyprof-annotations'],
        )
        self.configs.append(config)

        self.config_suffix_to_obj = dict()
        for config in self.configs:
            self.config_suffix_to_obj[config.config_suffix] = config

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

        self.init_configs()

        if self.args.plot:
            self.do_plot()
        else:
            self.do_run()

def rep_suffix(rep):
    assert rep is not None
    return "_repetition_{rep:02}".format(rep=rep)

if __name__ == '__main__':
    main()
