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

from iml_profiler.profiler import iml_logging
from iml_profiler.profiler.concurrent import ForkedProcessPool

from iml_profiler.profiler.util import args_to_cmdline

from iml_profiler.parser.common import *

MODES = [
    'train_stable_baselines.sh',
    'inference_stable_baselines.sh',
]

STABLE_BASELINES_ANNOTATED_ALGOS = [
    'ppo2',
    'dqn',
    'sac',
    'ddpg',
    # Should we include this?
    # It's not A3C, and that's what people will expect.
    # This is a single A3C worker (I think)?
    # I don't think they implement it correctly either...
    # They claim to be using SubprocEnv (run steps in parallel), but they're
    # doing DummyEnv (steps in serial).
    'a2c',
]
STABLE_BASELINES_AVAIL_ENV_IDS = None
STABLE_BASELINES_UNAVAIL_ENV_IDS = None

def add_stable_baselines_options(pars):
    pars.add_argument(
        '--algo',
        choices=STABLE_BASELINES_ANNOTATED_ALGOS,
        help='algorithm to run')
    pars.add_argument(
        '--env-id',
        choices=STABLE_BASELINES_ENV_IDS,
        help='environment to run')
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
        choices=['on_vs_off_policy', 'environments', 'algorithms', 'all_rl_workloads'],
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
    STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS = detect_available_env(STABLE_BASELINES_ENV_IDS)
    print("Available env ids:")
    print(textwrap.indent("\n".join(STABLE_BASELINES_AVAIL_ENV_IDS), prefix='  '))
    print("Unavailable env ids:")
    print(textwrap.indent("\n".join(sorted(STABLE_BASELINES_UNAVAIL_ENV_IDS.keys())), prefix='  '))

    if args.debug:
        logging.info(pprint_msg({'args': args.__dict__}))

    args.func(parser, args, extra_argv)

def detect_available_env(env_ids):
    # Going through custom gym packages to let them register in the global registory
    # for env_module in args.gym_packages:
    #     importlib.import_module(env_module)
    avail_env = []
    unavail_env = dict()
    for env_id in env_ids:
        try:
            env = gym.make(env_id)
            avail_env.append(env_id)
        except Exception as e:
            tb = traceback.format_exc()
            unavail_env[env_id] = tb
    return avail_env, unavail_env

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
        for expr in STABLE_BASELINES_EXPRS:
            if expr.algo == algo and expr.env_id == env_id:
                return expr
        return None

    def should_run(self, algo, env_id, bullet):
        if not self.is_supported(algo, env_id):
            return False
        if bullet and not is_bullet_env(env_id):
            return False
        return True

    def _gather_algo_env_pairs(self, algo=None, env_id=None, all=False, bullet=False, debug=False):

        if env_id is not None and algo is not None:
            algo_env_pairs = [(algo, env_id)]
            return algo_env_pairs

        if env_id is not None:
            algo_env_pairs = []
            for algo in STABLE_BASELINES_ANNOTATED_ALGOS:
                if not self.should_run(algo, env_id, bullet):
                    continue
                algo_env_pairs.append((algo, env_id))
            return algo_env_pairs

        if algo is not None:
            algo_env_pairs = []
            for env_id in STABLE_BASELINES_AVAIL_ENV_IDS:
                if not self.should_run(algo, env_id, bullet):
                    continue
                algo_env_pairs.append((algo, env_id))
            return algo_env_pairs

        if all:
            algo_env_pairs = []
            for algo in STABLE_BASELINES_ANNOTATED_ALGOS:
                for env_id in STABLE_BASELINES_AVAIL_ENV_IDS:
                    if not self.should_run(algo, env_id, bullet):
                        continue
                    algo_env_pairs.append((algo, env_id))
            return algo_env_pairs

        return None

    def _run_cmd(self, cmd, to_file, env=None, replace=False):
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
                    sys.exit(1)
                logging.info(
                    "> Command failed; see {path}; continuing (--skip-error was set)".format(
                        path=to_file,
                    ))
                failed = True

            if not failed:
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

        if args.mode in ['plot']:
            self.will_plot = True

        # def _run(expr, train_stable_baselines_opts, stacked_args):
        #     self.iml_bench(parser, subparser, 'train_stable_baselines.sh', train_stable_baselines_opts, suffix=bench_log(expr))
        #     self.stacked_plot(stacked_args, train_stable_baselines_opts, suffix=plot_log(expr))

        # (1) On vs off policy:
        expr = 'on_vs_off_policy'
        opts = ['--env-id', 'PongNoFrameskip-v4']
        if self.should_run_expr(expr):
            logging.info("Running expr = {expr}".format(expr=expr))
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

    def iml_directory(self, algo, env_id):
        iml_directory = _j(self.args.dir, algo, env_id)
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
        for algo_path in glob(_j(args.dir, '*')):
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

        algo_env_pairs = self._gather_algo_env_pairs(debug=debug, **vars(train_stable_baselines_args))
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
            '--directory', args.dir,
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
                '--directory', args.dir,
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
            util_csv = _j(args.dir, "overall_machine_util.raw.csv")
            util_plot_cmd.extend([
                '--csv', util_csv,
                # Output directory for the png plots.
                '--directory', args.dir,
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

    def iml_bench(self, parser, subparser, subcommand, subcmd_args, suffix='log'):
        args = self.args
        if not self.will_run:
            return
        main_cmd = self._get_main_cmd(parser, subparser, subcommand)
        cmd = main_cmd + subcmd_args
        to_file = self._get_logfile(suffix=suffix)
        logging.info("Logging iml-bench to file {path}".format(path=to_file))
        self._run_cmd(cmd=cmd, to_file=to_file)

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

        to_file = _j(args.dir, '{sub}.{suffix}'.format(
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

        to_file = _j(args.dir, 'error.txt')

        self._run_cmd(cmd=cmd, to_file=to_file)

    def _get_logfile(self, algo, env_id, suffix='log'):
        args = self.args

        to_file = _j(args.dir, '{sub}.algo_{algo}.env_id_{env_id}.{suffix}'.format(
            sub=self._sub_cmd,
            algo=algo,
            env_id=env_id,
            suffix=suffix,
        ))
        return to_file

    def iml_directory(self, algo, env_id):
        iml_directory = _j(self.args.dir, algo, env_id)
        return iml_directory

    def _sh_env(self, algo, env_id):
        args = self.args

        env = dict(os.environ)
        env['ENV_ID'] = env_id
        env['ALGO'] = algo
        if args.debug:
            env['DEBUG'] = 'yes'

        return env

    def _run(self, algo, env_id):
        args = self.args

        iml_directory = self.iml_directory(algo, env_id)
        cmd = [args.subcommand, "--iml-directory", iml_directory]

        env = self._sh_env(algo, env_id)

        to_file = self._get_logfile(algo, env_id, suffix='log')

        self._run_cmd(cmd=cmd, to_file=to_file, env=env)


    def run(self, parser):
        args = self.args

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
        else:
            for algo, env_id in algo_env_pairs:
                self._run(algo, env_id)

def tee(cmd, to_file, append=False, check=True, dry_run=False, **kwargs):
    if dry_run:
        print_cmd(cmd, files=[sys.stdout], env=kwargs.get('env', None), dry_run=dry_run)
        return

    with ScopedLogFile(to_file, append) as f:
        print_cmd(cmd, files=[sys.stdout, f], env=kwargs.get('env', None))

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, **kwargs)
        with p:
            # NOTE: if this blocks it may be because there's a zombie utilization_sampler.py still running
            # (that was created by the training script) that hasn't been terminated!
            for line in p.stdout:
                # b'\n'-separated lines
                line = line.decode("utf-8")
                sys.stdout.write(line)
                f.write(line)
        f.flush()
        if check and p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, p.args)
        return p


class ScopedLogFile:
    def __init__(self, file, append=False):
        self.file = file
        self.append = append

    def __enter__(self):
        if self._is_path:
            # if self.append:
            #         self.mode = 'ab'
            # else:
            #         self.mode = 'wb'

            if self.append:
                self.mode = 'a'
            else:
                self.mode = 'w'
            self.f = open(self.file, self.mode)
            return self.f
        else:
            # We can only append to a stream.
            self.f = self.file
            return self.f

    @property
    def _is_path(self):
        return type(self.file) == str

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.flush()
        if self._is_path:
            self.f.close()

class StableBaselinesExpr:
    def __init__(self, algo, env_id, mean_reward, std_reward, n_timesteps, n_episodes):
        self.algo = algo
        self.env_id = env_id
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.n_timesteps = n_timesteps
        self.n_episodes = n_episodes

STABLE_BASELINES_EXPRS = []
def add_stable_baselines_expr(*args, **kwargs):
    expr = StableBaselinesExpr(*args, **kwargs)
    STABLE_BASELINES_EXPRS.append(expr)
ENV_TO_STABLE_BASELINES_EXPRS = dict((expr.env_id, expr) for expr in STABLE_BASELINES_EXPRS)
ALGO_TO_STABLE_BASELINES_EXPRS = dict((expr.algo, expr) for expr in STABLE_BASELINES_EXPRS)

add_stable_baselines_expr('ppo2', 'Acrobot-v1', -85.137, 26.272, 149963, 1741)
add_stable_baselines_expr('a2c', 'Acrobot-v1', -86.616, 25.097, 149997, 1712)
add_stable_baselines_expr('dqn', 'Acrobot-v1', -88.103, 33.037, 149954, 1683)
add_stable_baselines_expr('acer', 'Acrobot-v1', -90.85, 32.797, 149989, 1633)
add_stable_baselines_expr('acktr', 'Acrobot-v1', -91.284, 32.515, 149959, 1625)
add_stable_baselines_expr('sac', 'AntBulletEnv-v0', 3485.228, 29.964, 150000, 150)
add_stable_baselines_expr('a2c', 'AntBulletEnv-v0', 2271.322, 160.233, 150000, 150)
add_stable_baselines_expr('ppo2', 'AntBulletEnv-v0', 2170.104, 250.575, 150000, 150)
add_stable_baselines_expr('ddpg', 'AntBulletEnv-v0', 2070.79, 74.253, 150000, 150)
add_stable_baselines_expr('acktr', 'BeamRiderNoFrameskip-v4', 3760.976, 1826.059, 147414, 41)
add_stable_baselines_expr('a2c', 'BeamRiderNoFrameskip-v4', 2809.115, 1298.573, 150181, 52)
add_stable_baselines_expr('acer', 'BeamRiderNoFrameskip-v4', 2440.692, 1357.964, 149127, 52)
add_stable_baselines_expr('ppo2', 'BeamRiderNoFrameskip-v4', 1691.072, 904.484, 149975, 69)
add_stable_baselines_expr('dqn', 'BeamRiderNoFrameskip-v4', 888.741, 248.487, 149395, 81)
add_stable_baselines_expr('ppo2', 'BipedalWalkerHardcore-v2', 166.481, 119.3, 149509, 154)
add_stable_baselines_expr('a2c', 'BipedalWalkerHardcore-v2', 102.754, 136.304, 149643, 137)
add_stable_baselines_expr('sac', 'BipedalWalkerHardcore-v2', 100.802, 117.769, 148974, 84)
add_stable_baselines_expr('sac', 'BipedalWalker-v2', 307.198, 1.055, 149794, 175)
add_stable_baselines_expr('ppo2', 'BipedalWalker-v2', 265.939, 80.994, 149968, 159)
add_stable_baselines_expr('a2c', 'BipedalWalker-v2', 255.012, 71.426, 149890, 169)
add_stable_baselines_expr('ddpg', 'BipedalWalker-v2', 94.202, 142.679, 149647, 240)
add_stable_baselines_expr('acktr', 'BreakoutNoFrameskip-v4', 448.514, 88.882, 143118, 37)
add_stable_baselines_expr('a2c', 'BreakoutNoFrameskip-v4', 384.865, 51.231, 146703, 52)
add_stable_baselines_expr('ppo2', 'BreakoutNoFrameskip-v4', 228.594, 141.964, 150921, 101)
add_stable_baselines_expr('dqn', 'BreakoutNoFrameskip-v4', 191.165, 97.795, 149817, 97)
add_stable_baselines_expr('dqn', 'CartPole-v1', 500, 0, 150000, 300)
add_stable_baselines_expr('ppo2', 'CartPole-v1', 500, 0, 150000, 300)
add_stable_baselines_expr('a2c', 'CartPole-v1', 499.903, 1.672, 149971, 300)
add_stable_baselines_expr('acer', 'CartPole-v1', 498.62, 23.862, 149586, 300)
add_stable_baselines_expr('acktr', 'CartPole-v1', 487.573, 63.866, 149685, 307)
add_stable_baselines_expr('trpo', 'CartPole-v1', 485.392, 70.505, 149986, 309)
add_stable_baselines_expr('dqn', 'EnduroNoFrameskip-v4', 699.8, 214.231, 146363, 15)
add_stable_baselines_expr('ppo2', 'EnduroNoFrameskip-v4', 643.824, 205.988, 149683, 17)
add_stable_baselines_expr('a2c', 'EnduroNoFrameskip-v4', 0, 0, 149574, 45)
add_stable_baselines_expr('acer', 'EnduroNoFrameskip-v4', 0, 0, 149574, 45)
add_stable_baselines_expr('acktr', 'EnduroNoFrameskip-v4', 0, 0, 149574, 45)
add_stable_baselines_expr('sac', 'HalfCheetahBulletEnv-v0', 3330.911, 95.575, 150000, 150)
add_stable_baselines_expr('ppo2', 'HalfCheetahBulletEnv-v0', 3195.326, 115.73, 150000, 150)
add_stable_baselines_expr('ddpg', 'HalfCheetahBulletEnv-v0', 2549.452, 37.652, 150000, 150)
add_stable_baselines_expr('a2c', 'HalfCheetahBulletEnv-v0', 2069.627, 103.895, 150000, 150)
add_stable_baselines_expr('trpo', 'HalfCheetahBulletEnv-v0', 1850.967, 282.093, 150000, 150)
add_stable_baselines_expr('sac', 'HopperBulletEnv-v0', 2438.152, 335.284, 149232, 155)
add_stable_baselines_expr('ppo2', 'HopperBulletEnv-v0', 1944.588, 612.994, 149157, 176)
add_stable_baselines_expr('a2c', 'HopperBulletEnv-v0', 1575.871, 669.267, 149075, 189)
add_stable_baselines_expr('sac', 'HumanoidBulletEnv-v0', 2048.187, 829.776, 149886, 172)
add_stable_baselines_expr('ppo2', 'HumanoidBulletEnv-v0', 1285.814, 918.715, 149544, 244)
add_stable_baselines_expr('sac', 'InvertedDoublePendulumBulletEnv-v0', 9357.406, 0.504, 150000, 150)
add_stable_baselines_expr('ppo2', 'InvertedDoublePendulumBulletEnv-v0', 7702.75, 2888.815, 149089, 181)
add_stable_baselines_expr('sac', 'InvertedPendulumSwingupBulletEnv-v0', 891.508, 0.963, 150000, 150)
add_stable_baselines_expr('ppo2', 'InvertedPendulumSwingupBulletEnv-v0', 866.989, 27.134, 150000, 150)
add_stable_baselines_expr('sac', 'LunarLanderContinuous-v2', 269.783, 57.077, 149852, 709)
add_stable_baselines_expr('ddpg', 'LunarLanderContinuous-v2', 244.566, 75.617, 149531, 660)
add_stable_baselines_expr('a2c', 'LunarLanderContinuous-v2', 203.912, 59.265, 149938, 253)
add_stable_baselines_expr('ppo2', 'LunarLanderContinuous-v2', 128.124, 44.384, 149971, 164)
add_stable_baselines_expr('trpo', 'LunarLanderContinuous-v2', 64.619, 94.377, 149127, 181)
add_stable_baselines_expr('dqn', 'LunarLander-v2', 269.048, 41.056, 149827, 624)
add_stable_baselines_expr('acer', 'LunarLander-v2', 185.21, 64.829, 149415, 248)
add_stable_baselines_expr('trpo', 'LunarLander-v2', 149.313, 108.546, 149893, 320)
add_stable_baselines_expr('ppo2', 'LunarLander-v2', 99.676, 62.033, 149512, 174)
add_stable_baselines_expr('acktr', 'LunarLander-v2', 96.822, 64.02, 149905, 176)
add_stable_baselines_expr('a2c', 'LunarLander-v2', 36.321, 135.294, 149696, 463)
add_stable_baselines_expr('ppo2', 'MinitaurBulletDuckEnv-v0', 5.78, 3.372, 149873, 416)
add_stable_baselines_expr('ppo2', 'MinitaurBulletEnv-v0', 11.334, 3.562, 150000, 252)
add_stable_baselines_expr('a2c', 'MountainCarContinuous-v0', 93.659, 0.199, 149985, 2187)
add_stable_baselines_expr('trpo', 'MountainCarContinuous-v0', 93.428, 1.509, 149998, 1067)
add_stable_baselines_expr('ddpg', 'MountainCarContinuous-v0', 91.858, 1.35, 149945, 1818)
add_stable_baselines_expr('ppo2', 'MountainCarContinuous-v0', 91.705, 1.706, 149985, 1082)
add_stable_baselines_expr('sac', 'MountainCarContinuous-v0', 90.421, 0.997, 149989, 1311)
add_stable_baselines_expr('acktr', 'MountainCar-v0', -111.917, 21.422, 149969, 1340)
add_stable_baselines_expr('a2c', 'MountainCar-v0', -130.921, 32.188, 149904, 1145)
add_stable_baselines_expr('acer', 'MountainCar-v0', -131.213, 32.541, 149976, 1143)
add_stable_baselines_expr('dqn', 'MountainCar-v0', -134.507, 24.748, 149975, 1115)
add_stable_baselines_expr('ppo2', 'MountainCar-v0', -143.501, 22.928, 149959, 1045)
add_stable_baselines_expr('trpo', 'MountainCar-v0', -144.537, 33.584, 149885, 1037)
add_stable_baselines_expr('acer', 'MsPacmanNoFrameskip-v4', 3908.105, 585.407, 148924, 95)
add_stable_baselines_expr('ppo2', 'MsPacmanNoFrameskip-v4', 2255.09, 706.412, 150040, 167)
add_stable_baselines_expr('dqn', 'MsPacmanNoFrameskip-v4', 1781.818, 605.289, 149783, 176)
add_stable_baselines_expr('acktr', 'MsPacmanNoFrameskip-v4', 1598.776, 264.338, 149588, 147)
add_stable_baselines_expr('a2c', 'MsPacmanNoFrameskip-v4', 1581.111, 499.757, 150229, 189)
add_stable_baselines_expr('sac', 'Pendulum-v0', -159.669, 86.665, 150000, 750)
add_stable_baselines_expr('a2c', 'Pendulum-v0', -162.24, 99.351, 150000, 750)
add_stable_baselines_expr('ppo2', 'Pendulum-v0', -168.285, 107.164, 150000, 750)
add_stable_baselines_expr('ddpg', 'Pendulum-v0', -169.829, 93.303, 150000, 750)
add_stable_baselines_expr('trpo', 'Pendulum-v0', -176.951, 97.098, 150000, 750)
add_stable_baselines_expr('dqn', 'PongNoFrameskip-v4', 21, 0, 148764, 93)
add_stable_baselines_expr('acer', 'PongNoFrameskip-v4', 20.667, 0.507, 148275, 57)
add_stable_baselines_expr('ppo2', 'PongNoFrameskip-v4', 20.507, 0.694, 149402, 69)
add_stable_baselines_expr('acktr', 'PongNoFrameskip-v4', 19.224, 3.697, 147753, 67)
add_stable_baselines_expr('a2c', 'PongNoFrameskip-v4', 18.973, 2.135, 148288, 75)
add_stable_baselines_expr('acer', 'QbertNoFrameskip-v4', 18880.469, 1648.937, 148617, 64)
add_stable_baselines_expr('ppo2', 'QbertNoFrameskip-v4', 14510, 2847.445, 150251, 90)
add_stable_baselines_expr('acktr', 'QbertNoFrameskip-v4', 9569.575, 3980.468, 150896, 106)
add_stable_baselines_expr('a2c', 'QbertNoFrameskip-v4', 5742.333, 2033.074, 151311, 150)
add_stable_baselines_expr('dqn', 'QbertNoFrameskip-v4', 644.345, 66.854, 152286, 252)
add_stable_baselines_expr('ppo2', 'ReacherBulletEnv-v0', 17.879, 9.78, 150000, 1000)
add_stable_baselines_expr('sac', 'ReacherBulletEnv-v0', 17.529, 9.86, 150000, 1000)
add_stable_baselines_expr('dqn', 'SeaquestNoFrameskip-v4', 1948.571, 234.328, 148547, 70)
add_stable_baselines_expr('ppo2', 'SeaquestNoFrameskip-v4', 1782.687, 80.883, 150535, 67)
add_stable_baselines_expr('acktr', 'SeaquestNoFrameskip-v4', 1672.239, 105.092, 149148, 67)
add_stable_baselines_expr('acer', 'SeaquestNoFrameskip-v4', 872.121, 25.555, 149650, 66)
add_stable_baselines_expr('a2c', 'SeaquestNoFrameskip-v4', 746.42, 111.37, 149749, 81)
add_stable_baselines_expr('acktr', 'SpaceInvadersNoFrameskip-v4', 738.045, 306.756, 149714, 156)
add_stable_baselines_expr('ppo2', 'SpaceInvadersNoFrameskip-v4', 689.631, 202.143, 150081, 176)
add_stable_baselines_expr('a2c', 'SpaceInvadersNoFrameskip-v4', 658.907, 197.833, 149846, 151)
add_stable_baselines_expr('dqn', 'SpaceInvadersNoFrameskip-v4', 636.618, 146.066, 150041, 136)
add_stable_baselines_expr('acer', 'SpaceInvadersNoFrameskip-v4', 542.556, 172.332, 150374, 133)
add_stable_baselines_expr('sac', 'Walker2DBulletEnv-v0', 2052.646, 13.631, 150000, 150)
add_stable_baselines_expr('ddpg', 'Walker2DBulletEnv-v0', 1954.753, 368.613, 149152, 155)
add_stable_baselines_expr('ppo2', 'Walker2DBulletEnv-v0', 1276.848, 504.586, 149959, 179)
add_stable_baselines_expr('a2c', 'Walker2DBulletEnv-v0', 618.318, 291.293, 149234, 187)
STABLE_BASELINES_ALGOS = sorted(set(expr.algo for expr in STABLE_BASELINES_EXPRS))
STABLE_BASELINES_ENV_IDS = sorted(set(expr.env_id for expr in STABLE_BASELINES_EXPRS))

def is_bullet_env(env_id):
    return re.search(r'BulletEnv', env_id)

if __name__ == '__main__':
    main()
