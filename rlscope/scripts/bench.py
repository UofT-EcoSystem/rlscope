"""
``rls-bench`` script for running lots of different experiments/benchmarks.
.. deprecated:: 1.0.0
    Replaced by :py:mod:`rlscope.scripts.calibration` (``rls-run --calibrate``).
"""
import re
import json
import shlex
from glob import glob
import copy
from glob import glob
from rlscope.profiler.rlscope_logging import logger
import subprocess
import sys
import os
import traceback
import pandas as pd
import argparse
import pprint
import textwrap
import multiprocessing
import importlib
try:
    # $ pip install 'gym >= 0.13.0'
    import gym
except ImportError:
    pass

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from rlscope.profiler.util import get_stacktrace
from rlscope.profiler.util import pprint_msg
from rlscope.experiment import expr_config
from rlscope.profiler.rlscope_logging import logger
from rlscope.profiler.concurrent import ForkedProcessPool, FailedProcessException
from rlscope.experiment.util import tee

from rlscope.profiler.util import args_to_cmdline

from rlscope.parser import check_host
from rlscope.parser.exceptions import RLScopeConfigurationError

from rlscope.parser.common import *

# DEFAULT_RLSCOPE_TRACE_TIME_SEC = 60*2
# DEFAULT_DEBUG_RLSCOPE_TRACE_TIME_SEC = 20

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
        #     # Sanity check: try --rlscope-disable-tfprof and --rlscope-disable-pyprof...performance SHOULD match --rlscope-disable…
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
        help=textwrap.dedent("""\
        instrumented:
            Run the script with RL-Scope enabled for the entire duration of training (if --rlscope-trace-time-sec), 
            record start/end timestamps.
            $ train.py --rlscope-directory ...
            # NOTE: don't use --rlscope-trace-time-sec here, we want to collect traces for the whole run
            
        uninstrumented:
            Run the script with RL-Scope disabled, record start/end timestamp at start/end of training script
            $ train.py --rlscope-disable
            
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
        '--bullet',
        action='store_true',
        help='Limit environments to physics-based Bullet environments')
    pars.add_argument(
        '--atari',
        action='store_true',
        help='Limit environments to Atari Pong environment')
    pars.add_argument(
        '--lunar',
        action='store_true',
        help='Limit environments to LunarLander environments (i.e. LunarLanderContinuous-v2, LunarLander-v2)')

def main():

    try:
        check_host.check_config()
    except RLScopeConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    # TODO: remove hard dependency on pybullet_envs;
    # i.e. we need it for:
    #   $ rls-bench stable-baselines --mode [run|all]
    # but we don't need pybullet for:
    #   $ rls-bench stable-baselines --mode plot
    # (assuming all the data is there...)
    try:
        import pybullet_envs
    except ImportError:
        print(textwrap.dedent("""\
        ERROR: You need to use pip to install pybullet to run rls-bench:
        $ pip install pybullet==2.5.1
        """.rstrip()))
        pybullet_envs = None
        sys.exit(1)

    try:
        import atari_py
    except ImportError:
        print(textwrap.dedent("""\
        ERROR: You need to use pip to install atari-py to run rls-bench:
        $ pip install git+https://github.com/openai/atari-py.git@0.2.0
        """.rstrip()))
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__.lstrip().rstrip()),
        formatter_class=argparse.RawTextHelpFormatter)

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
        '--rlscope-debug',
        action='store_true')
    # Don't support --pdb for rls-bench since I haven't figured out how to
    # both (1) log stdout/stderr of a command, (2) allow pdb debugger prompt
    # to be fully functional.
    # Even if we could...you probably don't want to log your pdb session
    # and all its color-codes anyway.
    # parser.add_argument(
    #     '--pdb',
    #     action='store_true')
    parser.add_argument('--debug-single-thread',
                        action='store_true',
                        help=textwrap.dedent("""\
    Debug with single thread.
    """))
    parser.add_argument(
        '--rls-prof',
        default='rls-prof',
        help=textwrap.dedent("""\
        Run train.py inside of rls-prof (for uninstrumented runs only)
          $ rls-prof python train.py
        This is done by setting RLSCOPE_PROF=<--rls-prof> inside the train_stable_baselines.sh training script.
        
        If for some reason you didn't want to run with rls-prof, you could set this to --rls-prof="".
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
        '--analyze',
        action='store_true')
    parser.add_argument(
        '--workers',
        type=int,
        help="Number of simultaneous rls-run jobs; memory is the limitation here, not CPUs",
        # For some reason, I'm getting this error when selecting sql_reader.processes:
        #
        # psycopg2.OperationalError: server closed the connection unexpectedly
        # This probably means the server terminated abnormally
        # before or while processing the request.
        #
        # When restarting rls-run the problem doesn't re-appear.
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
        default='./output/rlscope_bench/all',
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
        '--unins-dir',
        default='./output/expr_total_training_time',
        help=textwrap.dedent("""\
        Root directory containing full uninstrumented training runs 
        """.rstrip()))
    parser.add_argument(
        '--repetition',
        type=int,
        help=textwrap.dedent("""\
        e.g. for --repetition 1, --config instrumented
        $RLSCOPE_DIR/output/rlscope_bench/all/config_instrumented_repetition_01
        """.rstrip()))

    parser.add_argument('--algo-env-group',
                        choices=expr_config.ALGO_ENV_GROUP_CHOICES,
                        help=textwrap.dedent("""\
        Only run a specific "experiment".
        i.e. only run (algo, env) combinations needed for a specific graph.
        
        Default: run all (algo, env) combinations for all experiments.
        """))
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
        help=textwrap.dedent("""\
        rls-bench group: 
        
        Run ALL the stable-baselines experiments.
        
        If you just want to plot stuff, you can use: 
        $ rls-bench stable-baselines --mode plot
        
        See --mode for more.
        """))
    parser_stable_baselines.add_argument(
        '--mode',
        choices=['plot', 'run', 'all'],
        default='all',
        help=textwrap.dedent("""\
        train_stable_baselines.sh
          Run stable-baselines experiments.
        """))
    parser_stable_baselines.set_defaults(
        func=run_group,
        subparser=parser_stable_baselines)


    args, extra_argv = parser.parse_known_args()
    # logger.info("parsed arguments: {msg}".format(
    #     msg=pprint_msg({
    #         'args': args,
    #         'extra_argv': extra_argv,
    #     })))
    os.makedirs(args.dir, exist_ok=True)

    global STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS
    STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS = expr_config.detect_available_env(expr_config.STABLE_BASELINES_ENV_IDS)
    print("Available env ids:")
    print(textwrap.indent("\n".join(STABLE_BASELINES_AVAIL_ENV_IDS), prefix='  '))
    print("Unavailable env ids:")
    print(textwrap.indent("\n".join(sorted(STABLE_BASELINES_UNAVAIL_ENV_IDS.keys())), prefix='  '))

    if args.debug:
        logger.info(pprint_msg({'args': args.__dict__}))

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
                        logger.info("Saw \"IML BENCH DONE\" in {path} @ line {lineno}; skipping.".format(
                            lineno=lineno,
                            path=to_file))
                    return True
        return False

    def is_supported(self, algo, env_id):
        for expr in expr_config.STABLE_BASELINES_EXPRS:
            if expr.algo == algo and expr.env_id == env_id:
                return expr
        return None

    def _gather_algo_env_pairs(self, algo=None, env_id=None, bullet=False, atari=False, lunar=False, algo_env_group=None, debug=False):
        return expr_config.stable_baselines_gather_algo_env_pairs(
            algo=algo,
            env_id=env_id,
            bullet=bullet,
            atari=atari,
            lunar=lunar,
            algo_env_group=algo_env_group,
            debug=debug)

    def _run_cmd(self, cmd, to_file, env=None, replace=False, debug=False):
        args = self.args

        if env is None:
            # Make sure rls-run get RLSCOPE_POSTGRES_HOST
            env = dict(os.environ)

        proc = None
        if ( replace or args.replace ) or not self.already_ran(to_file):

            try:
                logger.info("> Calling tee: cmd =\n  $ {cmd}".format(cmd=cmd_as_string(cmd + self.extra_argv)))
                proc = tee(
                    cmd=cmd + self.extra_argv,
                    to_file=to_file,
                    env=env,
                    dry_run=args.dry_run,
                )
                logger.info("> Done calling tee: cmd =\n  $ {cmd}".format(cmd=cmd_as_string(cmd + self.extra_argv)))
                failed = False
            except subprocess.CalledProcessError as e:
                if not args.skip_error:
                    logger.info((
                                     "Command failed: see {path}; exiting early "
                                 ).format(path=to_file))
                    ret = 1
                    if debug:
                        logger.info("Exiting with ret={ret}\n{stack}".format(
                            ret=ret,
                            stack=get_stacktrace(),
                        ))
                    sys.exit(ret)
                logger.info(
                    "Command failed; see {path}; continuing (--skip-error was set)".format(
                        path=to_file,
                    ))
                failed = True

            if not failed:
                if proc is not None and proc.returncode != 0:
                    logger.info("BUG: saw returncode = {ret}, expected 0".format(
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
        # self.pool = ForkedProcessPool(name="rlscope_analyze_pool", max_workers=args.workers, debug=self.args.debug)

    def should_run_algo_env_group(self, algo_env_group):
        return self.args.algo_env_group is None or algo_env_group == self.args.algo_env_group

    @property
    def root_rlscope_directory(self):
        args = self.args
        root_rlscope_directory = get_root_rlscope_directory(args.config, args.dir, args.repetition)
        return root_rlscope_directory

    def stable_baselines(self, parser):
        """
        To avoid running everything more than once, we will stick ALL (algo, env_id) pairs inside of $RLSCOPE_DIR/output/rlscope_bench/all.

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

        os.makedirs(self.root_rlscope_directory, exist_ok=True)

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

        def rl_workloads_msg(expr, algo_env_pairs):
            all_pairs = gather.algo_env_pairs()
            return "RL workloads: algo_env_group={algo_env_group}: {msg}".format(
                algo_env_group=expr,
                msg=pprint_msg({
                    'all_pairs': all_pairs,
                    'algo_env_pairs': algo_env_pairs,
                }),
            )

        self.will_run = False
        # self.will_analyze = False
        self.will_plot = False

        if args.mode in ['all', 'run']:
            self.will_run = True

        if args.mode in ['plot'] and args.config == 'instrumented':
            self.will_plot = True

        # def _run(expr, train_stable_baselines_opts, stacked_args):
        #     self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', train_stable_baselines_opts, suffix=bench_log(expr))
        #     self.stacked_plot(stacked_args, train_stable_baselines_opts, suffix=plot_log(expr))


        # TODO: Run all the same experiments, BUT, under a different configuration.
        # We only want to plot stuff during the "extrapolated" configuration.

        gather = GatherAlgoEnv(args)

        # - Debug experiment:
        #   Choose a particular (algo, env) to use to debug stuff on.
        expr = 'debug_expr'
        algo = 'ppo2'
        env = 'HalfCheetahBulletEnv-v0'
        algo_env_pairs = [(algo, env)]
        opts = ['--env-id', env, '--algo', algo]
        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr), debug=True)
            overlap_type = 'ResourceOverlap'
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--y-type', 'percent',
                '--x-type', 'algo-comparison',
                '--training-time',
            ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        # (1) On vs off policy:
        expr = 'on_vs_off_policy'
        env = 'PongNoFrameskip-v4'
        algo_env_pairs = gather.pairs_by_env(env)
        opts = ['--env-id', env]
        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            overlap_type = 'ResourceOverlap'
            self.stacked_plot([
                '--overlap-type', overlap_type,
                '--y-type', 'percent',
                '--x-type', 'algo-comparison',
                '--training-time',
            ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        # (2) Environment choice:
        expr = 'environment_choice'
        algo_env_pairs = gather.pairs_by_algo_env_group(expr)
        # NOTE: --algo-env-group is forwarded automatically since it's an option that belongs to the "main" parser, not the subparser
        opts = []
        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            # overlap_type = 'OperationOverlap'
            overlap_type = 'CategoryOverlap'
            # NOTE: gather across ALL ppo directories, including environment we cannot run like AirLearning.
            algo_env_pairs = gather.pairs_by_algo('ppo2')
            # algo_env_pairs = [(algo, env) for algo, env in algo_env_pairs if re.search(r'AirLearning', env)]
            self.stacked_plot([
                '--overlap-type', overlap_type,

                # '--resource-overlap', json.dumps(['CPU']),
                # '--remap-df', textwrap.dedent("""
                #         # Replace ppo2 operations with simple (Inference, Simulation, Backpropagation) labels
                #         # established in Background discussion of typical RL workloads.
                #         # Sanity check to make sure we don't forget any operations.
                #         assert regions.issubset({
                #             ('step',),
                #             ('sample_action',),
                #             ('compute_advantage_estimates',),
                #             ('optimize_surrogate',),
                #             ('training_loop',),
                #         })
                #         new_df[('Simulation',)] = df[('step',)]
                #         new_df[('Inference',)] = df[('sample_action',)]
                #         new_df[('Backpropagation',)] = df[('compute_advantage_estimates',)] + df[('optimize_surrogate',)] + df[('training_loop',)]
                #         """),
                # '--training-time',
                # # '--extrapolated-training-time',

                '--y-lim-scale-factor', 1.10,
                '--detailed',
                '--rotation', 15,
                '--remap-df', textwrap.dedent("""
                        # Replace ppo2 operations with simple (Inference, Simulation, Backpropagation) labels
                        # established in Background discussion of typical RL workloads.
                        def ppo_pretty_operation(op_name):
                            if op_name in {'compute_advantage_estimates', 'optimize_surrogate', 'training_loop'}:
                                return "Backpropagation"
                            elif op_name == 'sample_action':
                                return "Inference"
                            elif op_name == 'step':
                                return "Simulation"
                            else:
                                raise NotImplementedError("Not sure what pretty-name to use for op_name={op_name}".format(
                                    op_name=op_name))
                        new_df['operation'] = new_df['operation'].apply(ppo_pretty_operation)
                        """),

                '--y-type', 'percent',
                '--x-type', 'env-comparison',
                # '--y2-logscale',
            ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        # (3) Algorithm choice:
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

        # (4) DQN detailed category/operation breakdown plot:
        expr = 'dqn_detailed'
        # algo_env_pairs = [('dqn', 'PongNoFrameskip-v4')]
        algo_env_pairs = gather.pairs_by_algo_env_group(expr)
        # NOTE: --algo-env-group is forwarded automatically since it's an option that belongs to the "main" parser, not the subparser
        opts = []

        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            overlap_type = 'CategoryOverlap'
            self.stacked_plot([
                '--detailed',
                '--overlap-type', overlap_type,
                '--remap-df', textwrap.dedent("""\
                    # Replace operations with simple (Inference, Simulation, Backpropagation) labels
                    # established in Background discussion of typical RL workloads.
                    def dqn_pretty_operation(op_name):
                        if op_name in {'q_backward', 'q_update_target_network', 'training_loop'}:
                            return "Backpropagation"
                        elif op_name == 'q_forward':
                            return "Inference"
                        elif op_name == 'step':
                            return "Simulation"
                        else:
                            raise NotImplementedError("Not sure what pretty-name to use for op_name={op_name}".format(
                                op_name=op_name))
                    new_df['operation'] = new_df['operation'].apply(dqn_pretty_operation)
                        """),
                '--y-type', 'percent',
                '--x-type', 'rl-comparison',
                # '--training-time',
                # '--y2-logscale',
            ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        def plot_expr_algorithm_choice(expr, algo_env_pairs):
            """
            Map all algorithm operations to uniform, simplified labels:
            - Inference: choosing an action to perform by consulting a neural network
            - Simulation: interacting with a simulator
            - Backpropagation: forward/backward pass, update network, maintain target network

            A2C:
                Inference = sample_action
                Simulation = step
                Backpropagation = train_step + training_loop

            DDPG:
                Inference = sample_action
                Simulation = step
                Backpropagation = evaluate + train_step + training_loop + update_target_network
                ...seems like evaluate may belong to inference...not sure though...
                IDEA: just delete / ignore it... these are RAW times so we CAN do that....HOWEVER training time still includes it so its WRONG to show that.

            PPO2:
                Inference = sample_action
                Simulation = step
                Backpropagation = compute_advantage_estimates + optimize_surrogate + training_loop

            SAC:
                Inference = sample_action
                Simulation = step
                Backpropgation = training_loop + update_actor_and_critic + update_target_network
            """
            if self.should_run_algo_env_group(expr):
                # overlap_type = 'OperationOverlap'
                overlap_type = 'CategoryOverlap'
                self.stacked_plot([
                    '--overlap-type', overlap_type,

                    '--detailed',
                    '--remap-df', textwrap.dedent("""
                    def pretty_operation(algo, op_name):
                        if algo == 'ppo2':
                            if op_name in {'compute_advantage_estimates', 'optimize_surrogate', 'training_loop'}:
                                return "Backpropagation"
                            elif op_name == 'sample_action':
                                return "Inference"
                            elif op_name == 'step':
                                return "Simulation"
                        elif algo == 'ddpg':
                            if op_name in {'evaluate', 'train_step', 'training_loop', 'update_target_network'}:
                                return "Backpropagation"
                            elif op_name == 'sample_action':
                                return "Inference"
                            elif op_name == 'step':
                                return "Simulation"
                        elif algo == 'a2c':
                            if op_name in {'train_step', 'training_loop'}:
                                return "Backpropagation"
                            elif op_name == 'sample_action':
                                return "Inference"
                            elif op_name == 'step':
                                return "Simulation"
                        elif algo == 'sac':
                            if op_name in {'training_loop', 'update_actor_and_critic', 'update_target_network'}:
                                return "Backpropagation"
                            elif op_name == 'sample_action':
                                return "Inference"
                            elif op_name == 'step':
                                return "Simulation"
                        raise NotImplementedError("Not sure what pretty-name to use for algo={algo}, op_name={op_name}".format(
                            algo=algo,
                            op_name=op_name))
                    new_df['operation'] = np.vectorize(pretty_operation, otypes=[str])(new_df['algo'], new_df['operation'])
                        """),

                    # '--resource-overlap', json.dumps(['CPU']),
                    # '--training-time',
                    # '--remap-df', textwrap.dedent("""
                    #     # Categorize ALL RL algorithms into simplified categories:
                    #     #   Inference, Simulation, Backpropgation
                    #     import pprint
                    #     pprint.pprint({
                    #         'algorithm_choice.remap_df.original_df': df,
                    #         'regions': regions,
                    #         '(algo, env)': "({algo}, {env})".format(algo=algo, env=env),
                    #     })
                    #     # Keep 'step' region
                    #     if algo == 'ppo2':
                    #         # Inference = sample_action
                    #         # Simulation = step
                    #         # Backpropagation = compute_advantage_estimates + optimize_surrogate + training_loop
                    #         assert regions.issubset({
                    #             ('step',),
                    #             ('sample_action',),
                    #             ('compute_advantage_estimates',),
                    #             ('optimize_surrogate',),
                    #             ('training_loop',),
                    #         })
                    #         new_df[('Simulation',)] = df[('step',)]
                    #         new_df[('Inference',)] = df[('sample_action',)]
                    #         new_df[('Backpropagation',)] = df[('compute_advantage_estimates',)] + df[('optimize_surrogate',)] + df[('training_loop',)]
                    #     elif algo == 'ddpg':
                    #         # Inference = sample_action
                    #         # Simulation = step
                    #         # Backpropagation = evaluate + train_step + training_loop + update_target_network
                    #         assert regions.issubset({
                    #             ('step',),
                    #             ('sample_action',),
                    #             ('evaluate',),
                    #             ('train_step',),
                    #             ('training_loop',),
                    #             ('update_target_network',),
                    #         })
                    #         new_df[('Simulation',)] = df[('step',)]
                    #         new_df[('Inference',)] = df[('sample_action',)]
                    #         new_df[('Backpropagation',)] = df[('evaluate',)] + df[('train_step',)] + df[('training_loop',)] + df[('update_target_network',)]
                    #     elif algo == 'a2c':
                    #         # Inference = sample_action
                    #         # Simulation = step
                    #         # Backpropagation = train_step + training_loop
                    #         assert regions.issubset({
                    #             ('step',),
                    #             ('sample_action',),
                    #             ('train_step',),
                    #             ('training_loop',),
                    #         })
                    #         new_df[('Simulation',)] = df[('step',)]
                    #         new_df[('Inference',)] = df[('sample_action',)]
                    #         new_df[('Backpropagation',)] = df[('train_step',)] + df[('training_loop',)]
                    #     elif algo == 'sac':
                    #         # Inference = sample_action
                    #         # Simulation = step
                    #         # Backpropgation = training_loop + update_actor_and_critic + update_target_network
                    #         assert regions.issubset({
                    #             ('step',),
                    #             ('sample_action',),
                    #             ('update_actor_and_critic',),
                    #             ('update_target_network',),
                    #             ('training_loop',),
                    #         })
                    #         new_df[('Simulation',)] = df[('step',)]
                    #         new_df[('Inference',)] = df[('sample_action',)]
                    #         new_df[('Backpropagation',)] = df[('training_loop',)] + df[('update_actor_and_critic',)] + df[('update_target_network',)]
                    #     else:
                    #         raise NotImplementedError("Not sure how to remap operation labels {labels} into simplified labels=[Inference, Simulation, Backpropgation] for algo={algo}".format(labels=regions, algo=algo))
                    #     """),

                    '--y-type', 'percent',
                    '--x-type', 'algo-comparison',
                ], suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

        # env_id = 'Walker2DBulletEnv-v0'
        expr = 'algorithm_choice_1a_med_complexity'
        algo_env_pairs = gather.pairs_by_algo_env_group(expr)
        opts = []
        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
        plot_expr_algorithm_choice(expr=expr, algo_env_pairs=algo_env_pairs)

        expr = 'algorithm_choice_1b_low_complexity'
        algo_env_pairs = gather.pairs_by_algo_env_group(expr)
        opts = []
        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
        plot_expr_algorithm_choice(expr=expr, algo_env_pairs=algo_env_pairs)

        # (4) Compare all RL workloads:
        expr = 'all_rl_workloads'
        algo_env_pairs = gather.pairs_by_algo_env_group(expr)
        opts = []
        common_dims = [
            '--width', '16',
            '--height', '6',
            '--rotation', '45',
        ]
        rl_workload_dims = [
            '--show-legend', 'False',
        ]
        if self.should_run_algo_env_group(expr):
            logger.info(rl_workloads_msg(expr, algo_env_pairs))
            # self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
            # # - Statement: GPU utilization is low across all workloads.
            # # - Plot: ResourceOverlap that compares [CPU] vs [CPU + GPU] breakdown
            # # Need 2 plots
            # # - 1st plot shows GPU time spent is tiny
            # #   TODO: we want categories to be 'CPU', 'CPU + GPU', 'GPU' (ResourceOverlap)
            # overlap_type = 'ResourceOverlap'
            algo_env_pairs = self.algo_env_pairs(include_minigo=True, debug=True)
            # self.stacked_plot([
            #     '--overlap-type', overlap_type,
            #     '--y-type', 'percent',
            #     '--training-time',
            #     '--y2-logscale',
            # ] + rl_workload_dims + common_dims, suffix=plot_log(expr, overlap_type), algo_env_pairs=algo_env_pairs)

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

            # algo_env_pairs = [(algo, env) for algo, env in algo_env_pairs if not re.search(r'dqn', algo)]
            # gpu_operations = ['sample_action']
            # for gpu_operation in gpu_operations:
            #     self.stacked_plot([
            #         '--overlap-type', overlap_type,
            #         '--resource-overlap', json.dumps(['CPU']),
            #         '--operation', gpu_operation,
            #         # '--selectors', gpu_operation,
            #         '--training-time',
            #         '--remap-df', textwrap.dedent("""
            #             keep_regions = [
            #                 ('Python',),
            #                 ('Framework API C',),
            #             ]
            #             for r in keep_regions:
            #                 new_df[r] = df[r]
            #             new_df[('CUDA API CPU',)] = df[('CUDA API CPU', 'Framework API C',)]
            #         """),
            #
            #
            #         # '--remap-df', textwrap.dedent("""
            #         # # Inference:
            #         # #   - operations we would use when deploying the trained model in production
            #         # # TODO:
            #         # # Backward-pass:
            #         # #   - additional operations required when training, but NOT from a fully trained model.
            #         # # Weight-update:
            #         # #   - after gradients are computed, update the weights
            #         # inference_ops = set([('step',), ('sample_action',)])
            #         # other_ops = set([op for op in regions if op not in inference_ops])
            #         # new_df[('inference',)] = np.sum(df[inference_ops])
            #         # new_df[('other',)] = np.sum(df[other_ops])
            #         # """),
            #
            #         '--y-type', 'percent',
            #     ] + rl_workload_dims + common_dims, suffix=plot_log(expr, overlap_type, gpu_operation), algo_env_pairs=algo_env_pairs)

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
        # # self.rlscope_bench(parser, subparser, 'train_stable_baselines.sh', opts, suffix=bench_log(expr))
        # overlap_type = 'OperationOverlap'
        # rlscope_dirs  = self.rlscope_dirs_air_learning()
        # self.stacked_plot([
        #     '--overlap-type', overlap_type,
        #     '--resource-overlap', json.dumps(['CPU']),
        #     '--y-type', 'percent',
        #     '--x-type', 'algo-comparison',
        # ], suffix=plot_log(expr, overlap_type), rlscope_dirs=rlscope_dirs)

    def algos(self):
        return set(algo for algo, env in self.algo_env_pairs())

    def envs(self):
        return set(env for algo, env in self.algo_env_pairs())

    def rlscope_directory(self, algo, env_id):
        args = self.args
        rlscope_directory = get_rlscope_directory(args.config, args.dir, algo, env_id, args.repetition)
        return rlscope_directory

    def unins_rlscope_directory(self, algo, env_id):
        args = self.args
        config = 'uninstrumented'
        repetition = 1
        rlscope_directory = _j(args.unins_dir, algo, env_id, 'config_{config}_repetition_{r:02}'.format(
            config=config,
            r=repetition,
        ))
        return rlscope_directory

    def algo_env_pairs(self, include_minigo=False, debug=False):
        gather = GatherAlgoEnv(self.args, include_minigo=include_minigo)
        return gather.algo_env_pairs(debug=debug)

    def _is_air_learning_env(self, env):
        return re.search(r'AirLearning', env)

    def rlscope_dirs_air_learning(self, debug=False):
        algo_env_pairs = [(algo, env) for algo, env in self.algo_env_pairs() \
                          if self._is_air_learning_env(env)]
        logger.info(pprint_msg({
            'self.algo_env_pairs()': self.algo_env_pairs(),
            'algo_env_pairs': algo_env_pairs}))
        rlscope_dirs = [self.rlscope_directory(algo, env) for algo, env in algo_env_pairs]
        return rlscope_dirs

    def stacked_plot(self, stacked_args, suffix, algo_env_pairs=None, debug=False):
        if not self.will_plot:
            return
        args = self.args
        cmd = [
            'rls-run',
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

        # if algo_env_pairs is None:
        #     raise NotImplementedError("Not sure what to use for --rlscope-directories")
        if len(algo_env_pairs) == 0 and not self.args.dry_run:
            raise NotImplementedError("Need at least one directory for --rlscope-directories but saw 0.")
        def sort_key(algo_env):
            """
            Show bar graphs order by algo first, then environment.
            """
            algo, env = algo_env
            return (algo, env)
        # Remove duplicates.
        algo_env_pairs = list(set(algo_env_pairs))
        algo_env_pairs.sort(key=sort_key)
        rlscope_dirs = [self.rlscope_directory(algo, env_id) for algo, env_id in algo_env_pairs]
        unins_rlscope_dirs = []
        for algo, env_id in algo_env_pairs:
            unins_rlscope_dir = self.unins_rlscope_directory(algo, env_id)
            if os.path.isdir(unins_rlscope_dir):
                unins_rlscope_dirs.append(unins_rlscope_dir)
            else:
                logger.info("OverlapStackedBarTask.SKIP unins_rlscope_dir = {path}".format(
                    path=unins_rlscope_dir))
        logger.info("OverlapStackedBarTask.args =\n{msg}".format(
            msg=pprint_msg({
                'algo_env_pairs': algo_env_pairs,
                'unins_rlscope_dirs': unins_rlscope_dirs,
            })))
        cmd.extend([
            '--rlscope-directories', json.dumps(rlscope_dirs),
            '--unins-rlscope-directories', json.dumps(unins_rlscope_dirs),
        ])

        cmd.extend([
            # Output directory for the png plots.
            '--directory', self.root_rlscope_directory,
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
                'rls-run',
            ]
            util_task_cmd.extend([
                '--task', 'UtilTask',
            ])

            if args.debug:
                util_task_cmd.append('--debug')

            if algo_env_pairs is None:
                algo_env_pairs = []
            algo_env_pairs = list(algo_env_pairs)

            if len(algo_env_pairs) == 0 and not self.args.dry_run:
                raise NotImplementedError("Need at least one directory for --rlscope-directories but saw 0.")
            def sort_key(algo_env):
                """
                Show bar graphs order by algo first, then environment.
                """
                algo, env = algo_env
                return (algo, env)
            # Remove duplicates.
            algo_env_pairs = list(set(algo_env_pairs))
            algo_env_pairs.sort(key=sort_key)
            rlscope_dirs = [self.rlscope_directory(algo, env_id) for algo, env_id in algo_env_pairs]
            util_task_cmd.extend([
                '--rlscope-directories', json.dumps(rlscope_dirs),
            ])

            util_task_cmd.extend([
                # Output directory for the png plots.
                '--directory', self.root_rlscope_directory,
                # Add expr-name to png.
                '--suffix', suffix,
            ])

            util_task_cmd.extend(stacked_args)

            util_task_cmd.extend(self.extra_argv)

            to_file = self._get_logfile(suffix="{suffix}.log".format(suffix=suffix))

            self._run_cmd(cmd=util_task_cmd, to_file=to_file, replace=True)

        def _util_plot():
            util_plot_cmd = [
                'rls-run',
                '--task', 'UtilPlotTask',
            ]
            if args.debug:
                util_plot_cmd.append('--debug')
            util_csv = _j(self.root_rlscope_directory, "overall_machine_util.raw.csv")
            util_plot_cmd.extend([
                '--csv', util_csv,
                # Output directory for the png plots.
                '--directory', self.root_rlscope_directory,
                # Add expr-name to png.
                '--suffix', suffix,
            ])
            if plot_args is not None:
                util_plot_cmd.extend(plot_args)
            if not _e(util_csv):
                logger.info("SKIP UtilTaskPlot; {path} doesn't exist")
                return
            to_file = self._get_logfile(suffix="UtilPlot.{suffix}.log".format(suffix=suffix))
            self._run_cmd(cmd=util_plot_cmd, to_file=to_file, replace=True)

        _util_csv(algo_env_pairs)
        _util_plot()

    def rlscope_bench(self, parser, subparser, subcommand, subcmd_args, suffix='log', env=None, debug=False):
        args = self.args
        if not self.will_run:
            return
        main_cmd = self._get_main_cmd(parser, subparser, subcommand)
        cmd = main_cmd + subcmd_args
        to_file = self._get_logfile(suffix=suffix)
        logger.info("Logging rls-bench to file {path}".format(path=to_file))
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

        to_file = _j(self.root_rlscope_directory, '{sub}.{suffix}'.format(
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
        self.pool = ForkedProcessPool(name="rlscope_analyze_pool", max_workers=args.workers,
                                      # debug=self.args.debug,
                                      )

    def _analyze(self, algo, env_id):
        args = self.args

        rlscope_directory = self.rlscope_directory(algo, env_id)
        cmd = ['rls-run', "--rlscope-directory", rlscope_directory]

        to_file = self._get_logfile(algo, env_id, suffix='analyze.log')
        logger.info("Analyze logfile = {path}".format(path=to_file))

        self._run_cmd(cmd=cmd, to_file=to_file)

    def _error(self):
        args = self.args

        if args.error:
            cmd = ['ls', "-z"]
        else:
            cmd = ['ls', "-l"]

        to_file = _j(self.root_rlscope_directory, 'error.txt')

        self._run_cmd(cmd=cmd, to_file=to_file)

    def _config_opts(self):
        args = self.args

        opts = []

        # if not config_is_full(args.config):
        #     # If we DON'T want to run for the full training duration add --rlscope-trace-time-sec
        #     pass

        # "Instrumented, no tfprof"
        # "Instrumented, no pyprof"

        # TODO: I suspect this config-dir names will get overloaded fast...need to use rlscope_config.json file that stores
        # Profiler.attrs instead.  Technically, we should store this in the process directory...
        # {
        #   'disable_tfprof'
        # }
        # 'config_instrumented_no_tfprof'
        # 'config_instrumented_no_pyprof'

        if config_is_uninstrumented(args.config):
            # If we want to run uninstrumented, add --rlscope-disable, but still record training progress
            opts.extend(['--rlscope-disable',
                         ])

        if config_is_no_tfprof(args.config):
            opts.extend(['--rlscope-disable-tfprof'])

        if config_is_no_pyprof(args.config):
            opts.extend(['--rlscope-disable-pyprof'])

        return opts

    def _get_logfile(self, algo, env_id, suffix='log'):
        args = self.args

        to_file = _j(self.root_rlscope_directory, '{sub}.algo_{algo}.env_id_{env_id}.{suffix}'.format(
            sub=self._sub_cmd,
            algo=algo,
            env_id=env_id,
            suffix=suffix,
        ))
        return to_file

    @property
    def root_rlscope_directory(self):
        args = self.args
        root_rlscope_directory = get_root_rlscope_directory(args.config, args.dir, args.repetition)
        return root_rlscope_directory

    def rlscope_directory(self, algo, env_id):
        args = self.args
        rlscope_directory = get_rlscope_directory(args.config, args.dir, algo, env_id, args.repetition)
        return rlscope_directory

    def _sh_env(self, algo, env_id):
        args = self.args

        env = dict(os.environ)
        env['ENV_ID'] = env_id
        env['ALGO'] = algo
        if args.debug:
            env['DEBUG'] = 'yes'
            env['RLSCOPE_DEBUG'] = 'yes'

        if args.rlscope_prof:
            env['RLSCOPE_PROF'] = args.rlscope_prof

        return env

    def _run(self, algo, env_id):
        args = self.args

        # NOTE: We use absolute path of rlscope-directory
        # since some training scripts change cd to a
        # different directory before they run.
        rlscope_directory = _a(self.rlscope_directory(algo, env_id))
        cmd = [args.subcommand, "--rlscope-directory", rlscope_directory]
        if args.rlscope_debug:
            cmd.append('--rlscope-debug')
        config_opts = self._config_opts()
        cmd.extend(config_opts)

        env = self._sh_env(algo, env_id)

        to_file = self._get_logfile(algo, env_id, suffix='log')

        self._run_cmd(cmd=cmd, to_file=to_file, env=env)


    def run(self, parser):
        args = self.args

        os.makedirs(self.root_rlscope_directory, exist_ok=True)

        subparser = self.args.subparser
        # Not multiprocessing friendly (cannot pickle)
        del self.args.subparser

        # if args.debug:
        # logger.info(pprint_msg({'StableBaselines.args': args.__dict__}))

        if args.subcommand == 'dummy_error.sh':
            self._error()
            return

        if args.env_id is not None and args.env_id in STABLE_BASELINES_UNAVAIL_ENV_IDS:
            print("ERROR: env_id={env} is not available since gym.make('{env}') failed.".format(
                env=args.env_id))
            sys.exit(1)

        if args.analyze and config_is_uninstrumented(args.config):
            logger.info(("Cannot run rls-run on --config={config}; config must be instrumented "
                          "(e.g. --config instrumented), otherwise there are no RL-Scope traces to process.").format(
                config=args.config,
            ))
            sys.exit(1)


        if args.analyze:
            gather = GatherAlgoEnv(args)
            algo_env_pairs = gather.pairs_by_args()
            # if args.debug:
            logger.info("Run --analyze over (algo, env) pairs: {msg}".format(
                msg=pprint_msg({
                    'algo_env_pairs': algo_env_pairs,
                })
            ))
            for algo, env_id in algo_env_pairs:
                # self._analyze(algo, env_id)
                self.pool.submit(
                    'rls-run --rlscope-directory {rlscope}'.format(rlscope=self.rlscope_directory(algo, env_id)),
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
            # Gather (algo, env) pairs whose environments we can run.
            # (GatherAlgoEnv only looks at existing runs in output/all/rlscope_bench)
            algo_env_pairs = self._gather_algo_env_pairs(
                algo=args.algo,
                env_id=args.env_id,
                bullet=args.bullet,
                atari=args.atari,
                lunar=args.lunar,
                algo_env_group=args.algo_env_group,
            )
            if algo_env_pairs is None:
                logger.info('Please provide either --env-id or --algo')
                sys.exit(1)

            if args.debug:
                logger.info({'algo_env_pairs': algo_env_pairs, 'args.analyze': args.analyze})

            logger.info("Collect trace-data over (algo, env) pairs: {msg}".format(
                msg=pprint_msg({
                    'algo_env_pairs': algo_env_pairs,
                })
            ))
            for algo, env_id in algo_env_pairs:
                self._run(algo, env_id)


def get_root_rlscope_directory(config, direc, repetition):
    # if config == 'instrumented':
    #     # The default run --config.
    #     # Run RL-Scope for 2 minutes and collect full traces.
    #     rlscope_directory = direc
    # else:
    #     # Either 'instrumented' or 'uninstrumented'.
    #     # Add a --config sub-directory.
    #     config_dir = "config_{config}".format(config=config)
    #     rlscope_directory = _j(direc, config_dir)
    if repetition is not None:
        config_dir = "config_{config}_repetition_{r:02}".format(
            config=config,
            r=repetition)
    else:
        config_dir = "config_{config}".format(
            config=config)
    rlscope_directory = _j(direc, config_dir)
    return rlscope_directory

def get_rlscope_directory(config, direc, algo, env_id, repetition):
    root_rlscope_directory = get_root_rlscope_directory(config, direc, repetition)
    rlscope_directory = _j(root_rlscope_directory, algo, env_id)
    return rlscope_directory

class GatherAlgoEnv:
    def __init__(self, args, include_minigo=False):
        self.args = args
        self.include_minigo = include_minigo

    def should_skip_env(self, env):
        # For some reason, the simulation time for Humanoid is huge when trained with sac.
        # Until we figure out WHY, let's just leave it out of all the plots.
        # return re.search(r'Humanoid', env)
        return False

    def should_skip_algo_env(self, algo, env):
        return not expr_config.is_paper_env(algo, env)

    def should_skip_algo(self, algo):
        # For some reason, AirLearningEnv is missing annotations from the sac algorthim.
        # I suspect sac is the "cuplrit" for various issues.
        # return re.search(r'sac', algo)
        return False

    def _is_env_dir(self, path):
        return os.path.isdir(path) and re.search(r'Env|-v\d+$', path)

    def _is_algo_dir(self, path):
        return os.path.isdir(path)

    def algo_env_pairs(self, has_machine_util=False, debug=False):
        args = self.args
        algo_env_pairs = []
        algo_paths = glob(_j(self.root_rlscope_directory, '*'))
        for algo_path in algo_paths:
            if is_config_dir(algo_path):
                continue
            if not self._is_algo_dir(algo_path):
                if args.debug:
                    logger.info("Skip algo_path={dir}".format(dir=algo_path))
                continue
            algo = _b(algo_path)
            if self.should_skip_algo(algo):
                continue
            env_paths = glob(_j(algo_path, '*'))
            for env_path in env_paths:
                if not self._is_env_dir(env_path):
                    if args.debug:
                        logger.info("Skip env_path={dir}".format(dir=env_path))
                    continue
                env = _b(env_path)
                if self.should_skip_env(env):
                    continue
                if self.should_skip_algo_env(algo, env):
                    if args.debug:
                        logger.info("Skip (algo={algo}, env={env}) @ {dir}".format(
                            algo=algo,
                            env=env,
                            dir=env_path))
                    continue

                if has_machine_util and not self.has_machine_util(algo, env):
                    continue

                algo_env_pairs.append((algo, env))
        return algo_env_pairs

    def pairs_by_algo(self, algo):
        return set((a, e) for a, e in self.algo_env_pairs() if a == algo)

    def pairs_by_args(self):
        args = self.args
        def should_keep(algo, env):
            # NOTE: If they don't provide ANY flags for selecting algo/env to run, we run NOTHING.
            # We no longer default to running everything (since it's a confusing bug)
            ret = ( args.atari and expr_config.is_atari_env(env) ) or \
                  ( args.lunar and expr_config.is_lunar_combo(algo, env) ) or \
                  ( args.bullet and expr_config.is_bullet_env(env) ) or \
                  ( args.algo_env_group is not None and expr_config.is_algo_env_group_combo(args.algo_env_group, algo, env) )
            return ret
        return self.pairs_by_func(should_keep)

    def pairs_by_env(self, env):
        return set((a, e) for a, e in self.algo_env_pairs() if e == env)

    def pairs_by_func(self, func):
        return set((a, e) for a, e in self.algo_env_pairs() if func(a, e))

    def pairs_by_lunar(self):
        return self.pairs_by_func(expr_config.is_lunar_combo)

    def pairs_by_algo_env_group(self, algo_env_group):
        def is_algo_env(algo, env):
            return expr_config.is_algo_env_group_combo(algo_env_group, algo, env)
        return self.pairs_by_func(is_algo_env)

    # def pairs_by_is_fig_algo_comparison_low_complexity(self):
    #     return self.pairs_by_func(expr_config.is_fig_algo_comparison_low_complexity)
    #
    # def pairs_by_is_fig_algo_comparison_med_complexity(self):
    #     return self.pairs_by_func(expr_config.is_fig_algo_comparison_med_complexity)
    #
    # def pairs_by_is_fig_env_comparison(self):
    #     return self.pairs_by_func(expr_config.is_fig_env_comparison)
    #
    # def pairs_by_is_paper_env(self):
    #     return self.pairs_by_func(expr_config.is_paper_env)

    @property
    def root_rlscope_directory(self):
        args = self.args
        root_rlscope_directory = get_root_rlscope_directory(args.config, args.dir, args.repetition)
        return root_rlscope_directory

    def machine_util_files(self, algo, env):
        rlscope_directory = self.rlscope_directory(algo, env)
        return [path for path in list_files(rlscope_directory) if is_machine_util_file(path)]

    def has_machine_util(self, algo, env):
        machine_util_files = self.machine_util_files(algo, env)
        return len(machine_util_files) > 0

if __name__ == '__main__':
    main()
