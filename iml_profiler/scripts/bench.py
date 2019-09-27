"""
iml-bench script for running lots of different experiments/benchmarks.
"""
import re
import json
import shlex
from glob import glob
import copy
from glob import glob
import logging
import subprocess
import sys
import os
import traceback
import pandas as pd
import ipdb
import argparse
import pprint
import textwrap
import multiprocessing
import importlib
import gym
# TODO: remove hard dependency on pybullet_envs;
# i.e. we need it for:
#   $ iml-bench stable-baselines --mode [run|all]
# but we don't need pybullet for:
#   $ iml-bench stable-baselines --mode plot
# (assuming all the data is there...)
try:
    import pybullet_envs
except ImportError:
    print(textwrap.dedent("""\
    ERROR: You need to use pip to install pybullet to run iml-bench:
    $ pip install pybullet==2.5.1
    """.rstrip()))
    sys.exit(1)
    pybullet_envs = None

try:
    import atari_py
except ImportError:
    print(textwrap.dedent("""\
    ERROR: You need to use pip to install atari-py to run iml-bench:
    $ pip install git+https://github.com/openai/atari-py.git@0.2.0
    """.rstrip()))
    sys.exit(1)

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler.experiment import expr_config
from iml_profiler.profiler import iml_logging
from iml_profiler.profiler.concurrent import ForkedProcessPool, FailedProcessException
from iml_profiler.experiment.util import tee

from iml_profiler.profiler.util import args_to_cmdline

from iml_profiler.parser.common import *

DEFAULT_IML_TRACE_TIME_SEC = 60*2
DEFAULT_DEBUG_IML_TRACE_TIME_SEC = 20

MODES = [
    'train_stable_baselines.sh',
    'inference_stable_baselines.sh',
]

STABLE_BASELINES_AVAIL_ENV_IDS = None
STABLE_BASELINES_UNAVAIL_ENV_IDS = None

def add_config_options(pars):

    pars.add_argument(
        '--config',
        default='instrumented',
        # choices=[
        #     'instrumented',
        #     'instrumented_no_tfprof',
        #     'instrumented_no_pyprof',
        #
        #     # Just measure tfprof trace-collection overhead (e.g. libcupti).
        #     'instrumented_no_pyprof_no_tfdump',
        #     # Just measure pyprof trace-collection overhead (i.e. python timestamps for wrapped TensorFlow API calls).
        #     'instrumented_no_tfprof_no_pydump',
        #     # Just measure small aspects of pyprof trace-collection overhead.
        #     'instrumented_no_tfprof_no_pydump_no_pytrace',
        #
        #     # Sanity check: try --iml-disable-tfprof and --iml-disable-pyprof...performance SHOULD match --iml-disable…
        #     # RESULT: holds true... Really though...pyprof is THAT BAD?
        #     # TODO: disable pyprof dumping, JUST do collection
        #     'instrumented_no_tfprof_no_pyprof',
        #
        #     'uninstrumented',
        #
        #     'instrumented_full',
        #     'instrumented_full_no_tfprof',
        #     'instrumented_full_no_pyprof',
        #     'instrumented_full_no_tfprof_no_pyprof',
        #
        #     'uninstrumented_full'],
        help=
        textwrap.dedent("""
        instrumented:
            Run the script with IML enabled for the entire duration of training (if --iml-trace-time-sec), 
            record start/end timestamps.
            $ train.py --iml-directory ...
            # NOTE: don't use --iml-trace-time-sec here, we want to collect traces for the whole run
            
        uninstrumented:
            Run the script with IML disabled, record start/end timestamp at start/end of training script
            $ train.py --iml-disable
            
        instrumented_full:
            Run for the ENTIRE training duration (don't stop early)
            
        uninstrumented_full:
            Run for the ENTIRE training duration (don't stop early)
        """))

def add_stable_baselines_options(pars):
    pars.add_argument(
        '--algo',
        choices=expr_config.STABLE_BASELINES_ANNOTATED_ALGOS,
        help='algorithm to run')
    pars.add_argument(
        '--env-id',
        choices=expr_config.STABLE_BASELINES_ENV_IDS,
        help='environment to run')
    pars.add_argument(
        '--iml-trace-time-sec',
        help='IML: How long to trace for? (option added by IML to train.py)')
    pars.add_argument(
        '--bullet',
        action='store_true',
        help='Limit environments to physics-based Bullet environments')
    pars.add_argument(
        '--all',
        action='store_true',
        help=textwrap.dedent("""
        Run all (algo, env_id) pairs.
        Make this explicit to avoid running everything by accident.
        """))

def main():
    iml_logging.setup_logging()
    parser = argparse.ArgumentParser(
        textwrap.dedent("""\
        Run a bunch of experiments for a paper.
        """),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # TODO:
    # - add sub-parser for each training script
    #   - options for train_stable_baselines.sh:
    #     - run all the env-ids for ONE algorithm (investigate simulator overhead)
    #     - run all the algo-ids for ONE env-id (investigate influence of RL algorithm)
    #     - If neither of these is provides, just run EVERYTHING.
    #     - If both of these is provided, just run that algo-id / env-id combo.

    # parser.add_argument('--mode',
    #                     choices=MODES,
    #                     required=True,

    parser.add_argument(
        '--debug',
        action='store_true')
    parser.add_argument(
        '--iml-debug',
        action='store_true')
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
        '--iml-prof',
        default='iml-prof',
        help=textwrap.dedent("""
        Run train.py inside of iml-prof (for uninstrumented runs only)
          $ iml-prof python train.py
        This is done by setting IML_PROF=<--iml-prof> inside the train_stable_baselines.sh training script.
        
        If for some reason you didn't want to run with iml-prof, you could set this to --iml-prof="".
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
        '--analyze',
        action='store_true')
    parser.add_argument(
        '--workers',
        type=int,
        help="Number of simultaneous iml-analyze jobs; memory is the limitation here, not CPUs",
        # For some reason, I'm getting this error when selecting sql_reader.processes:
        #
        # psycopg2.OperationalError: server closed the connection unexpectedly
        # This probably means the server terminated abnormally
        # before or while processing the request.
        #
        # When restarting iml-analyze the problem doesn't re-appear.
        # I suspect this is related to running multiple jobs but it's hard to know for sure.
        # Until we figure it out, run 1 job at time.
        #
        # NOTE: Just hacked around it and reconnected when "SELECT 1" fails.
        # Still not sure about root cause.
        # I don't think it's caused by too many connections; I experienced this error after
        # a connection was left for about 8-9 minutes without being used.
        # default=2,
        #
        # I think we have memory errors now that op-events have increased...
        default=1,
    )
    parser.add_argument(
        '--dir',
        default='./output/iml_bench/all',
        help=textwrap.dedent("""\
        Directory to store stuff in for the subcommand.
        
        all subcommands: 
          log files
        train_stable_baselines.sh / stable-baselines:
          trace-files @ <algo>/<env>
        plot-stable-baselines:
          plots
        """.rstrip()))
    add_config_options(parser)

    subparsers = parser.add_subparsers(
        dest='subcommand',
        description='valid sub-commands, identified by bash training script wrapper',
        title='sub-commands',
        help='sub-command help')

    # create the parser for the "a" command

    parser_train_stable_baselines = subparsers.add_parser(
        'train_stable_baselines.sh',
        help='stable-baselines training experiments')
    add_stable_baselines_options(parser_train_stable_baselines)
    parser_train_stable_baselines.set_defaults(
        func=run_stable_baselines,
        subparser=parser_train_stable_baselines)

    parser_inference_stable_baselines = subparsers.add_parser(
        'inference_stable_baselines.sh',
        help='stable-baselines inference experiments')
    add_stable_baselines_options(parser_inference_stable_baselines)
    parser_inference_stable_baselines.set_defaults(
        func=run_stable_baselines,
        subparser=parser_inference_stable_baselines)

    parser_dummy_error = subparsers.add_parser(
        'dummy_error.sh',
        help='test: make sure errors in bench.py are handled correctly')
    parser_dummy_error.add_argument(
        '--error',
        action='store_true',
        help='If set, run a command with non-zero exit status')
    parser_dummy_error.set_defaults(
        func=run_stable_baselines,
        subparser=parser_dummy_error)

    parser_stable_baselines = subparsers.add_parser(
        'stable-baselines',
        help=textwrap.dedent("""
        iml-bench group: 
        
        Run ALL the stable-baselines experiments.
        
        If you just want to plot stuff, you can use: 
        $ iml-bench stable-baselines --mode plot
        
        See --mode for more.
        """))
    parser_stable_baselines.add_argument(
        '--mode',
        choices=['plot', 'run', 'all'],
        default='all',
        help=textwrap.dedent("""
        train_stable_baselines.sh
          Run stable-baselines experiments.
        """))
    parser_stable_baselines.add_argument(
        '--expr',
        choices=['on_vs_off_policy', 'environments', 'algorithms', 'all_rl_workloads',
                 'debug_expr'],
        help=textwrap.dedent("""
        Only run a specific "experiment".
        i.e. only run (algo, env) combinations needed for a specific graph.
        
        Default: run all (algo, env) combinations for all experiments.
        """))
    parser_stable_baselines.set_defaults(
        func=run_group,
        subparser=parser_stable_baselines)

    args, extra_argv = parser.parse_known_args()
    os.makedirs(args.dir, exist_ok=True)

    global STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS
    STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS = expr_config.detect_available_env(expr_config.STABLE_BASELINES_ENV_IDS)
    print("Available env ids:")
    print(textwrap.indent("\n".join(STABLE_BASELINES_AVAIL_ENV_IDS), prefix='  '))
    print("Unavailable env ids:")
    print(textwrap.indent("\n".join(sorted(STABLE_BASELINES_UNAVAIL_ENV_IDS.keys())), prefix='  '))

    if args.debug:
        logging.info(pprint_msg({'args': args.__dict__}))

    args.func(parser, args, extra_argv)

def run_stable_baselines(parser, args, extra_argv):
    obj = StableBaselines(args, extra_argv)
    assert args.subcommand is not None
    obj.run(parser)

class Experiment:

    @property
    def _sub_cmd(self):
        args = self.args

        sub_cmd = args.subcommand

        m = re.search(r'(?P<sh_name>.*)\.sh$', args.subcommand)
        if m:
            sub_cmd = m.group('sh_name')

        return sub_cmd

    def already_ran(self, to_file):
        if not _e(to_file):
            return False
        with open(to_file) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.rstrip()
                if re.search(r'IML BENCH DONE', line):
                    if self.args.debug:
                        logging.info("Saw \"IML BENCH DONE\" in {path} @ line {lineno}; skipping.".format(
                            lineno=lineno,
                            path=to_file))
                    return True
        return False

    def is_supported(self, algo, env_id):
        for expr in expr_config.STABLE_BASELINES_EXPRS:
            if expr.algo == algo and expr.env_id == env_id:
                return expr
        return None

    def _gather_algo_env_pairs(self, algo=None, env_id=None, all=False, bullet=False, debug=False):
        return expr_config.stable_baselines_gather_algo_env_pairs(
            algo=algo,
            env_id=env_id,
            all=all,
            bullet=bullet,
            debug=debug)

    def _run_cmd(self, cmd, to_file, env=None, replace=False, debug=False):
        args = self.args

        if env is None:
            # Make sure iml-analyze get IML_POSTGRES_HOST
            env = dict(os.environ)

        proc = None
        if ( replace or args.replace ) or not self.already_ran(to_file):

            try:
                proc = tee(
                    cmd=cmd + self.extra_argv,
                    to_file=to_file,
                    env=env,
                    dry_run=args.dry_run,
                )
                failed = False
            except subprocess.CalledProcessError as e:
                if not args.skip_error:
                    logging.info((
                                     "> Command failed: see {path}; exiting early "
                                     "(use --skip-error to ignore individual experiment errors)"
                                 ).format(path=to_file))
                    ret = 1
                    if debug:
                        logging.info("Exiting with ret={ret}\n{stack}".format(
                            ret=ret,
                            stack=get_stacktrace(),
                        ))
                    sys.exit(ret)
                logging.info(
                    "> Command failed; see {path}; continuing (--skip-error was set)".format(
                        path=to_file,
                    ))
                failed = True

            if not failed:
                if proc is not None and proc.returncode != 0:
                    logging.info("BUG: saw returncode = {ret}, expected 0".format(
                        ret=proc.returncode))
                    assert proc.returncode == 0
                if not args.dry_run:
                    with open(to_file, 'a') as f:
                        f.write("IML BENCH DONE\n")
                if not args.dry_run:
                    assert self.already_ran(to_file)

        return proc


class ExperimentGroup(Experiment):
    def __init__(self, args, extra_argv):
        self.args = args
        self.extra_argv = extra_argv
        # self.pool = ForkedProcessPool(name="iml_analyze_pool", max_workers=args.workers, debug=self.args.debug)

    def should_run_expr(self, expr):
        return self.args.expr is None or expr == self.args.expr

    def should_skip_env(self, env):
        # For some reason, the simulation time for Humanoid is huge when trained with sac.
        # Until we figure out WHY, let's just leave it out of all the plots.
        return re.search('Humanoid', env)

    def should_skip_algo(self, algo):
        # For some reason, AirLearningEnv is missing annotations from the sac algorthim.
        # I suspect sac is the "cuplrit" for various issues.
        return re.search('sac', algo)

    def stable_baselines(self, parser):
        """
        To avoid running everything more than once, we will stick ALL (algo, env_id) pairs inside of $IML_DIR/output/iml_bench/all.

        (1) On vs off policy:
        Run Atari Pong on all environments that support it (that we have annotated):
        - Ppo2 [sort of on-policy]
        - A2c [on-policy]
        - Dqn [off-policy]
        We can use this to compare on-policy vs off-policy

        (2) Compare environments:
        Run ALL bullet environments for at least one algorithm (ppo2).

        (3) Compare algorithms:
        Run ALL algorithms on 1 bullet environment (Walker)

        (4) Compare all RL workloads:
        Run ALL algorithms on ALL bullet environments
        """

        args = self.args

        os.makedirs(self.root_iml_directory, exist_ok=True)

        subparser = self.args.subparser
        # Not multiprocessing friendly (cannot pickle)
        del self.args.subparser

        def bench_log(expr):
            if args.analyze:
                return "{expr}.analyze.log".format(expr=expr)
            return "{expr}.log".format(expr=expr)

        def plot_log(expr, overlap_type=None, operation=None):
            """
            {expr}.{overlap_type}.{operation}.plot
            ------
            if present.
            """
            def _add(suffix, string):
                if string is not None:
                    suffix = "{suffix}.{string}".format(suffix=suffix, string=string)
                return suffix

            suffix = expr
            suffix = _add(suffix, overlap_type)
            suffix = _add(suffix, operation)
            suffix = _add(suffix, 'plot')

            return suffix

        self.will_run = False
        # self.will_analyze = False
        self.will_plot = False

        if args.mode in ['all', 'run']:
            self.will_run = True

        if args.mode in ['plot'] and args.config == 'instrumented':
            self.will_plot = True

        # def _run(expr, train_stable_baselines_opts, stacked_args):
        #     self.iml_bench(parser, subparser, 'train_stable_baselines.sh', train_stable_baselines_opts, suffix=bench_log(expr))
        #     self.stacked_plot(stacked_args, train_stable_baselines_opts, suffix=plot_log(expr))


        # TODO: Run all the same experiments, BUT, under a different configuration.
        # We only want to plot stuff during the "extrapolated" configuration.

        # - Debug experiment:
        #   Choose a particular (algo, env) to use to debug stuff on.
        expr = 'debug_expr'
        opts = ['--env-id', 'HalfCheetahBulletEnv-v0', '--algo', 'ppo2']
        # print("HELLO 1")
        # sys.exit(1)
        if self.should_run_expr(expr):
            self.iml_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr), debug=True)
            overlap_type = 'ResourceOverlap'
            # print("HELLO 2")
            # sys.exit(1)
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--y-type', 'percent',
                '--x-type', 'algo-comparison',
                '--training-time',
            ], suffix=plot_log(expr, overlap_type), train_stable_baselines_opts=opts)

        # (1) On vs off policy:
        expr = 'on_vs_off_policy'
        opts = ['--env-id', 'PongNoFrameskip-v4']
        if self.should_run_expr(expr):
            self.iml_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            overlap_type = 'ResourceOverlap'
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--y-type', 'percent',
                '--x-type', 'algo-comparison',
                '--training-time',
            ], suffix=plot_log(expr, overlap_type), train_stable_baselines_opts=opts)

        # (2) Compare environments:
        expr = 'environments'
        opts = ['--bullet', '--algo', 'ppo2']
        if self.should_run_expr(expr):
            self.iml_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            overlap_type = 'OperationOverlap'
            algo_env_pairs = self.pairs_by_algo('ppo2')
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--resource-overlap', json.dumps(['CPU']),
                '--y-type', 'percent',
                '--x-type', 'env-comparison',
                '--training-time',
                '--y2-logscale',
            ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        # (3) Compare algorithms:
        # Want to show on-policy vs off-policy.
        # e.g. compare DDPG against PPO
        # We want to show that:
        # - 'step' time of DDPG vs PPO; 'step' of DDPG should be bigger for SAME environment
        # - Q: this info comes from OperationOverlap CPU-breakdown; each algo definitely has a step annotation;
        #   however currently the algorithms have other annotations that are inconsistent;
        # - So, this feature depends on region-merging.
        # - HOWEVER, we DON'T need to re-run for this one.
        # - region-map:
        #     step = step
        #     other = sum(r for r in regions if r != step)
        expr = 'algorithms'
        opts = ['--env-id', 'Walker2DBulletEnv-v0']
        if self.should_run_expr(expr):
            self.iml_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            overlap_type = 'OperationOverlap'
            algo_env_pairs = self.pairs_by_env('Walker2DBulletEnv-v0')
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--resource-overlap', json.dumps(['CPU']),
                '--training-time',
                '--remap-df', json.dumps([textwrap.dedent("""
                # Keep 'step' region
                new_df[('step',)] = df[('step',)]

                # Sum up regions besides 'step'
                new_df[('other',)] = 0.
                for r in regions:
                    if r == ('step',):
                        continue
                    new_df[('other',)] = new_df[('other',)] + df[r]
                """)]),
                '--y-type', 'percent',
                '--x-type', 'algo-comparison',
            ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        # (4) Compare all RL workloads:
        expr = 'all_rl_workloads'
        opts = ['--all', '--bullet']
        common_dims = [
            '--width', '16',
            '--height', '6',
            '--rotation', '45',
        ]
        rl_workload_dims = [
            '--show-legend', 'False',
        ]
        if self.should_run_expr(expr):
            self.iml_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            # - Statement: GPU utilization is low across all workloads.
            # - Plot: ResourceOverlap that compares [CPU] vs [CPU + GPU] breakdown
            # Need 2 plots
            # - 1st plot shows GPU time spent is tiny
            #   TODO: we want categories to be 'CPU', 'CPU + GPU', 'GPU' (ResourceOverlap)
            overlap_type = 'ResourceOverlap'
            algo_env_pairs = self.algo_env_pairs()
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--y-type', 'percent',
                '--training-time',
                '--y2-logscale',
            ] + rl_workload_dims + common_dims, suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

            self.util_plot([
                '--algo-env-from-dir',
            ], plot_args=[
                '--y-title', 'GPU utilization (%)',
                         ] + common_dims, suffix=plot_log(expr), algo_env_pairs=algo_env_pairs)

            # - 2nd plot shows where CPU time is going
            #   TODO: we want categories to be 'C++ framework', 'CUDA API C', 'Python' (CategoryOverlap)
            #   TODO: For WHAT GPU operation...? We need to merge everyone into an 'Inference' category first.
            overlap_type = 'CategoryOverlap'
            # PROBLEM: sample_action is q_forward for DQN... but that means it appears in an entirely separate file.
            # What we'd like to do:
            # - read the q_forward CategoryOverlap file.
            # - IF operation == q_forward, THEN remap it to sample_action.
            # - SOLUTION:
            #   - allow specifying multiple selector's; currently we can only specify 1 selector.
            #     e.g. --selectors "[{selector-1}, {selector-2}]"
            # - For Sri meeting:
            #   - remove DQN for now.
            #   - fix janky plot x-axis alignment
            # TODO: keep dqn and use --selectors to select different files and remap-df to remap dqn[q_forward] to dqn[sample_action]
            algo_env_pairs = [(algo, env) for algo, env in algo_env_pairs if not re.search(r'dqn', algo)]

            gpu_operations = ['sample_action']
            for gpu_operation in gpu_operations:
                self.stacked_plot([
                    '--overlap-type', overlap_type,
                    '--resource-overlap', json.dumps(['CPU']),
                    '--operation', gpu_operation,
                    # '--selectors', gpu_operation,
                    '--training-time',
                    '--remap-df', json.dumps([textwrap.dedent("""
                        keep_regions = [
                            ('Python',),
                            ('Framework API C',),
                        ]
                        for r in keep_regions:
                            new_df[r] = df[r]
                        new_df[('CUDA API CPU',)] = df[('CUDA API CPU', 'Framework API C',)]
                    """)]),


                    # '--remap-df', json.dumps([textwrap.dedent("""
                    # # Inference:
                    # #   - operations we would use when deploying the trained model in production
                    # # TODO:
                    # # Backward-pass:
                    # #   - additional operations required when training, but NOT from a fully trained model.
                    # # Weight-update:
                    # #   - after gradients are computed, update the weights
                    # inference_ops = set([('step',), ('sample_action',)])
                    # other_ops = set([op for op in regions if op not in inference_ops])
                    # import ipdb; ipdb.set_trace()
                    # new_df[('inference',)] = np.sum(df[inference_ops])
                    # new_df[('other',)] = np.sum(df[other_ops])
                    # """)]),

                    '--y-type', 'percent',
                ] + rl_workload_dims + common_dims, suffix=plot_log(expr, overlap_type, gpu_operation), algo_env_pairs=algo_env_pairs)

        # - Statement: GPU operations are dominated by CPU time. In particular, CUDA API C
        #   time is a dominant contributor.
        # - Plot: There are multiple GPU operations, and for each one we would have a separate plot...?
        #   Well, we COULD stick them all in one plot, so long as the operations don't overlap.
        #   I need to be careful about inference is in the case of minigo.
        #   Inference - Forward-pass: CategoryOverlap, [CPU, GPU]
        #   Training - Backward-pass: CategoryOverlap, [CPU, GPU]
        #   Q: Should we include weight-update here?
        # - IDEALLY: we need the generic breakdown; for now we can do "Inference", since
        #   every algorithm I collected at least has "sample_action"
        # - NOTE: it would be nice to have GPU-time in this plot as well...
        #   currently I don't think that's easy to do since it would involve "merging"
        #   two *.venn_js.json files, one from ['CPU', 'GPU'] and one from ['CPU']

        # # (5) Different RL algorithms on the industrial-scale simulator: AirLearning environment:
        # expr = 'beefy_simulator'
        # # opts = ['--bullet', '--algo', 'ppo2']
        # # self.iml_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
        # overlap_type = 'OperationOverlap'
        # iml_dirs  = self.iml_dirs_air_learning()
        # self.stacked_plot([
        #     '--overlap-type', overlap_type,
        #     '--resource-overlap', json.dumps(['CPU']),
        #     '--y-type', 'percent',
        #     '--x-type', 'algo-comparison',
        # ], suffix=plot_log(expr, overlap_type), iml_dirs=iml_dirs)

    def algos(self):
        return set(algo for algo, env in self.algo_env_pairs())

    def envs(self):
        return set(env for algo, env in self.algo_env_pairs())

    def pairs_by_algo(self, algo):
        return set((a, e) for a, e in self.algo_env_pairs() if a == algo)

    def pairs_by_env(self, env):
        return set((a, e) for a, e in self.algo_env_pairs() if e == env)

    @property
    def root_iml_directory(self):
        args = self.args
        root_iml_directory = get_root_iml_directory(args.config, args.dir)
        return root_iml_directory

    def iml_directory(self, algo, env_id):
        args = self.args
        iml_directory = get_iml_directory(args.config, args.dir, algo, env_id)
        return iml_directory

    def _is_env_dir(self, path):
        return os.path.isdir(path) and re.search('Env', path)

    def _is_algo_dir(self, path):
        return os.path.isdir(path)

    def machine_util_files(self, algo, env):
        iml_directory = self.iml_directory(algo, env)
        return [path for path in list_files(iml_directory) if is_machine_util_file(path)]

    def has_machine_util(self, algo, env):
        machine_util_files = self.machine_util_files(algo, env)
        return len(machine_util_files) > 0

    def algo_env_pairs(self, has_machine_util=False):
        args = self.args
        algo_env_pairs = []
        for algo_path in glob(_j(self.root_iml_directory, '*')):
            if is_config_dir(algo_path):
                continue
            if not self._is_algo_dir(algo_path):
                logging.info("Skip algo_path={dir}".format(dir=algo_path))
                continue
            algo = _b(algo_path)
            if self.should_skip_algo(algo):
                continue
            for env_path in glob(_j(algo_path, '*')):
                if not self._is_env_dir(env_path):
                    logging.info("Skip env_path={dir}".format(dir=env_path))
                    continue
                env = _b(env_path)
                if self.should_skip_env(env):
                    continue

                if has_machine_util and not self.has_machine_util(algo, env):
                    continue

                algo_env_pairs.append((algo, env))
        return algo_env_pairs

    def _is_air_learning_env(self, env):
        return re.search(r'AirLearning', env)

    def iml_dirs_air_learning(self, debug=False):
        algo_env_pairs = [(algo, env) for algo, env in self.algo_env_pairs() \
                          if self._is_air_learning_env(env)]
        logging.info(pprint_msg({
            'self.algo_env_pairs()': self.algo_env_pairs(),
            'algo_env_pairs': algo_env_pairs}))
        iml_dirs = [self.iml_directory(algo, env) for algo, env in algo_env_pairs]
        return iml_dirs

    def algo_env_pairs_train_stable_baselines(self, train_stable_baselines_opts, debug=False):
        args = self.args

        parser_train_stable_baselines = argparse.ArgumentParser()
        add_stable_baselines_options(parser_train_stable_baselines)
        train_stable_baselines_args = parser_train_stable_baselines.parse_args(train_stable_baselines_opts)

        keep_argnames = {
            'algo',
            'env_id',
            'all',
            'bullet',
            'debug',
        }
        gather_algo_env_dict = vars(train_stable_baselines_args)
        for k in list(gather_algo_env_dict.keys()):
            if k not in keep_argnames:
                del gather_algo_env_dict[k]
        algo_env_pairs = self._gather_algo_env_pairs(debug=debug, **gather_algo_env_dict)
        assert algo_env_pairs is not None
        if len(algo_env_pairs) == 0:
            logging.info(
                textwrap.dedent("""\
                Not sure what to use for --iml-directories.
                Didn't see any (algo, env) pairs for
                  $ train_stable_baselines.sh {opts}
                """.rstrip().format(opts=' '.join(train_stable_baselines_opts))))
            if args.skip_error:
                logging.info("Skipping plot (--skip-error)")
                return
            else:
                logging.info("Exiting due to failed plot; use --skip-error to ignore")
                sys.exit(1)
        return algo_env_pairs
        # iml_dirs = [self.iml_directory(algo, env_id) for algo, env_id in algo_env_pairs]
        # return iml_dirs

    def stacked_plot(self, stacked_args, suffix, algo_env_pairs=None, train_stable_baselines_opts=None, debug=False):
        if not self.will_plot:
            return
        args = self.args
        cmd = [
            'iml-analyze',
        ]
        cmd.extend([
            '--task', 'OverlapStackedBarTask',
        ])

        if args.debug:
            cmd.append('--debug')
        # if args.pdb:
        #     cmd.append('--pdb')

        if algo_env_pairs is None:
            algo_env_pairs = []
        algo_env_pairs = list(algo_env_pairs)

        if train_stable_baselines_opts is not None:
            algo_env_pairs.extend(self.algo_env_pairs_train_stable_baselines(train_stable_baselines_opts, debug=debug))
        # if algo_env_pairs is None:
        #     raise NotImplementedError("Not sure what to use for --iml-directories")
        if len(algo_env_pairs) == 0:
            raise NotImplementedError("Need at least one directory for --iml-directories but saw 0.")
        def sort_key(algo_env):
            """
            Show bar graphs order by algo first, then environment.
            """
            algo, env = algo_env
            return (algo, env)
        # Remove duplicates.
        algo_env_pairs = list(set(algo_env_pairs))
        algo_env_pairs.sort(key=sort_key)
        iml_dirs = [self.iml_directory(algo, env_id) for algo, env_id in algo_env_pairs]
        cmd.extend([
            '--iml-directories', json.dumps(iml_dirs),
        ])

        cmd.extend([
            # Output directory for the png plots.
            '--directory', self.root_iml_directory,
            # Add expr-name to png.
            '--suffix', suffix,
        ])

        cmd.extend(stacked_args)

        cmd.extend(self.extra_argv)

        to_file = self._get_logfile(suffix="{suffix}.log".format(suffix=suffix))

        self._run_cmd(cmd=cmd, to_file=to_file, replace=True)

    def util_plot(self, stacked_args, suffix, plot_args=None, algo_env_pairs=None, debug=False):
        if not self.will_plot:
            return

        args = self.args

        def _util_csv(algo_env_pairs):
            util_task_cmd = [
                'iml-analyze',
            ]
            util_task_cmd.extend([
                '--task', 'UtilTask',
            ])

            if args.debug:
                util_task_cmd.append('--debug')

            if algo_env_pairs is None:
                algo_env_pairs = []
            algo_env_pairs = list(algo_env_pairs)

            if len(algo_env_pairs) == 0:
                raise NotImplementedError("Need at least one directory for --iml-directories but saw 0.")
            def sort_key(algo_env):
                """
                Show bar graphs order by algo first, then environment.
                """
                algo, env = algo_env
                return (algo, env)
            # Remove duplicates.
            algo_env_pairs = list(set(algo_env_pairs))
            algo_env_pairs.sort(key=sort_key)
            iml_dirs = [self.iml_directory(algo, env_id) for algo, env_id in algo_env_pairs]
            util_task_cmd.extend([
                '--iml-directories', json.dumps(iml_dirs),
            ])

            util_task_cmd.extend([
                # Output directory for the png plots.
                '--directory', self.root_iml_directory,
                # Add expr-name to png.
                '--suffix', suffix,
            ])

            util_task_cmd.extend(stacked_args)

            util_task_cmd.extend(self.extra_argv)

            to_file = self._get_logfile(suffix="{suffix}.log".format(suffix=suffix))

            self._run_cmd(cmd=util_task_cmd, to_file=to_file, replace=True)

        def _util_plot():
            util_plot_cmd = [
                'iml-analyze',
                '--task', 'UtilPlotTask',
            ]
            if args.debug:
                util_plot_cmd.append('--debug')
            util_csv = _j(self.root_iml_directory, "overall_machine_util.raw.csv")
            util_plot_cmd.extend([
                '--csv', util_csv,
                # Output directory for the png plots.
                '--directory', self.root_iml_directory,
                # Add expr-name to png.
                '--suffix', suffix,
            ])
            if plot_args is not None:
                util_plot_cmd.extend(plot_args)
            if not _e(util_csv):
                logging.info("SKIP UtilTaskPlot; {path} doesn't exist")
                return
            to_file = self._get_logfile(suffix="UtilPlot.{suffix}.log".format(suffix=suffix))
            self._run_cmd(cmd=util_plot_cmd, to_file=to_file, replace=True)

        _util_csv(algo_env_pairs)
        _util_plot()

    def iml_bench(self, parser, subparser, subcommand, subcmd_args, suffix='log', env=None, debug=False):
        args = self.args
        if not self.will_run:
            return
        main_cmd = self._get_main_cmd(parser, subparser, subcommand)
        cmd = main_cmd + subcmd_args
        to_file = self._get_logfile(suffix=suffix)
        logging.info("Logging iml-bench to file {path}".format(path=to_file))
        self._run_cmd(cmd=cmd, to_file=to_file, env=env, debug=debug)

    def _get_main_cmd(self, parser, subparser, subcommand):
        args = self.args
        cmd_args = copy.copy(args)
        cmd_args.subcommand = subcommand
        main_cmd = args_to_cmdline(parser, cmd_args,
                                   subparser=subparser,
                                   subparser_argname='subcommand',
                                   use_pdb=False,
                                   ignore_unhandled_types=True,
                                   ignore_argnames=['func', 'subparser', 'mode', 'expr'],
                                   debug=args.debug)
        return main_cmd

    def run(self, parser):
        args = self.args
        if args.subcommand == 'stable-baselines':
            self.stable_baselines(parser)
        else:
            raise NotImplementedError

    def _get_logfile(self, suffix='log'):
        args = self.args

        to_file = _j(self.root_iml_directory, '{sub}.{suffix}'.format(
            sub=self._sub_cmd,
            suffix=suffix,
        ))
        return to_file

def run_group(parser, args, extra_argv):
    obj = ExperimentGroup(args, extra_argv)
    assert args.subcommand is not None
    obj.run(parser)


class StableBaselines(Experiment):
    def __init__(self, args, extra_argv):
        # self.parser = parser
        self.args = args
        self.extra_argv = extra_argv
        self.pool = ForkedProcessPool(name="iml_analyze_pool", max_workers=args.workers,
                                      # debug=self.args.debug,
                                      )

    def _analyze(self, algo, env_id):
        args = self.args

        iml_directory = self.iml_directory(algo, env_id)
        cmd = ['iml-analyze', "--iml-directory", iml_directory]

        to_file = self._get_logfile(algo, env_id, suffix='analyze.log')
        logging.info("Analyze logfile = {path}".format(path=to_file))

        self._run_cmd(cmd=cmd, to_file=to_file)

    def _error(self):
        args = self.args

        if args.error:
            cmd = ['ls', "-z"]
        else:
            cmd = ['ls', "-l"]

        to_file = _j(self.root_iml_directory, 'error.txt')

        self._run_cmd(cmd=cmd, to_file=to_file)

    @property
    def iml_trace_time_sec(self):
        args = self.args
        if args.iml_trace_time_sec is not None:
            return args.iml_trace_time_sec
        if args.debug:
            return DEFAULT_DEBUG_IML_TRACE_TIME_SEC
        return DEFAULT_IML_TRACE_TIME_SEC

    def _config_opts(self):
        args = self.args

        opts = []

        if not config_is_full(args.config):
            # If we DON'T want to run for the full training duration add --iml-trace-time-sec
            opts.extend(['--iml-trace-time-sec', self.iml_trace_time_sec])

        # "Instrumented, no tfprof"
        # "Instrumented, no pyprof"

        # TODO: I suspect this config-dir names will get overloaded fast...need to use iml_config.json file that stores
        # Profiler.attrs instead.  Technically, we should store this in the process directory...
        # {
        #   'disable_tfprof'
        # }
        # 'config_instrumented_no_tfprof'
        # 'config_instrumented_no_pyprof'

        if config_is_uninstrumented(args.config):
            # If we want to run uninstrumented, add --iml-disable, but still record training progress
            opts.extend(['--iml-disable', '--iml-training-progress'])

        if config_is_no_tfprof(args.config):
            opts.extend(['--iml-disable-tfprof'])

        if config_is_no_pyprof(args.config):
            opts.extend(['--iml-disable-pyprof'])

        if config_is_no_pydump(args.config):
            opts.extend(['--iml-disable-pyprof-dump'])

        if config_is_no_pytrace(args.config):
            opts.extend(['--iml-disable-pyprof-trace'])

        if config_is_no_tfdump(args.config):
            opts.extend(['--iml-disable-tfprof-dump'])

        return opts

    def _get_logfile(self, algo, env_id, suffix='log'):
        args = self.args

        to_file = _j(self.root_iml_directory, '{sub}.algo_{algo}.env_id_{env_id}.{suffix}'.format(
            sub=self._sub_cmd,
            algo=algo,
            env_id=env_id,
            suffix=suffix,
        ))
        return to_file

    @property
    def root_iml_directory(self):
        args = self.args
        root_iml_directory = get_root_iml_directory(args.config, args.dir)
        return root_iml_directory

    def iml_directory(self, algo, env_id):
        args = self.args
        iml_directory = get_iml_directory(args.config, args.dir, algo, env_id)
        return iml_directory

    def _sh_env(self, algo, env_id):
        args = self.args

        env = dict(os.environ)
        env['ENV_ID'] = env_id
        env['ALGO'] = algo
        if args.debug:
            env['DEBUG'] = 'yes'
            env['IML_DEBUG'] = 'yes'

        if args.iml_prof:
            env['IML_PROF'] = args.iml_prof

        if args.iml_trace_time_sec is not None:
            env['IML_TRACE_TIME_SEC'] = str(args.iml_trace_time_sec)

        return env

    def _run(self, algo, env_id):
        args = self.args

        # NOTE: We use absolute path of iml-directory
        # since some training scripts change cd to a
        # different directory before they run.
        iml_directory = _a(self.iml_directory(algo, env_id))
        cmd = [args.subcommand, "--iml-directory", iml_directory]
        if args.iml_debug:
            cmd.append('--iml-debug')
        config_opts = self._config_opts()
        cmd.extend(config_opts)

        env = self._sh_env(algo, env_id)

        to_file = self._get_logfile(algo, env_id, suffix='log')

        self._run_cmd(cmd=cmd, to_file=to_file, env=env)


    def run(self, parser):
        args = self.args

        os.makedirs(self.root_iml_directory, exist_ok=True)

        subparser = self.args.subparser
        # Not multiprocessing friendly (cannot pickle)
        del self.args.subparser

        # if args.debug:
        # logging.info(pprint_msg({'StableBaselines.args': args.__dict__}))

        if args.subcommand == 'dummy_error.sh':
            self._error()
            return

        if args.env_id is not None and args.env_id in STABLE_BASELINES_UNAVAIL_ENV_IDS:
            print("ERROR: env_id={env} is not available since gym.make('{env}') failed.".format(
                env=args.env_id))
            sys.exit(1)

        if args.analyze and config_is_uninstrumented(args.config):
            logging.info(("Cannot run iml-analyze on --config={config}; config must be instrumented "
                          "(e.g. --config instrumented), otherwise there are no IML traces to process.").format(
                config=args.config,
            ))
            sys.exit(1)

        algo_env_pairs = self._gather_algo_env_pairs(
            algo=args.algo,
            env_id=args.env_id,
            all=args.all,
            bullet=args.bullet)
        if algo_env_pairs is None:
            logging.info('Please provide either --env-id or --algo')
            sys.exit(1)

        if args.debug:
            logging.info({'algo_env_pairs': algo_env_pairs, 'args.analyze': args.analyze})

        if args.analyze:
            if args.debug:
                logging.info("Run --analyze")
            for algo, env_id in algo_env_pairs:
                # self._analyze(algo, env_id)
                self.pool.submit(
                    'iml-analyze --iml-directory {iml}'.format(iml=self.iml_directory(algo, env_id)),
                    self._analyze,
                    algo, env_id,
                    sync=self.args.debug_single_thread)
            # Wait for processes to finish.
            # If we forget to do this, even if a child exits with nonzero status, we won't see it!
            try:
                self.pool.shutdown()
            except FailedProcessException as e:
                # Child has already printed its error message; just forward the exit code.
                assert e.exitcode != 0
                sys.exit(e.exitcode)
        else:
            for algo, env_id in algo_env_pairs:
                self._run(algo, env_id)


def get_root_iml_directory(config, direc):
    # if config == 'instrumented':
    #     # The default run --config.
    #     # Run IML for 2 minutes and collect full traces.
    #     iml_directory = direc
    # else:
    #     # Either 'instrumented' or 'uninstrumented'.
    #     # Add a --config sub-directory.
    #     config_dir = "config_{config}".format(config=config)
    #     iml_directory = _j(direc, config_dir)
    config_dir = "config_{config}".format(config=config)
    iml_directory = _j(direc, config_dir)
    return iml_directory

def get_iml_directory(config, direc, algo, env_id):
    root_iml_directory = get_root_iml_directory(config, direc)
    iml_directory = _j(root_iml_directory, algo, env_id)
    return iml_directory

if __name__ == '__main__':
    main()
