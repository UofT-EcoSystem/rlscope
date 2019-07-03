"""
iml-bench script for running lots of different experiments/benchmarks.
"""
import re
import logging
import subprocess
import sys
import os
import traceback
import ipdb
import argparse
import pprint
import textwrap
import multiprocessing
import importlib
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

from os.path import join as _j, abspath as _a, dirname as _d, exists as _e, basename as _b

from iml_profiler.profiler import glbl
from iml_profiler.profiler.profilers import MyProcess, ForkedProcessPool

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

def main():
    glbl.setup_logging()
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
    #                     help=textwrap.dedent("""
    #                     train_stable_baselines.sh
    #                       Run stable-baselines experiments.
    #                     """))

    parser.add_argument(
        '--debug',
        action='store_true')
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
        '--analyze',
        action='store_true')
    parser.add_argument(
        '--workers',
        type=int,
        help="Number of simultaneous iml-analyze jobs; memory is the limitation here, not CPUs",
        default=2)
    parser.add_argument(
        '--dir',
        default='.',
        help='directory to store log files in')

    subparsers = parser.add_subparsers(
        description='valid sub-commands, identified by bash training script wrapper',
        title='sub-commands',
        help='sub-command help')

    # create the parser for the "a" command

    def add_stable_baselines_options(parser_stable_baselines):
        parser_stable_baselines.add_argument(
            '--algo',
            choices=STABLE_BASELINES_ANNOTATED_ALGOS,
            help='algorithm to run')
        parser_stable_baselines.add_argument(
            '--env-id',
            choices=STABLE_BASELINES_ENV_IDS,
            help='environment to run')
        parser_stable_baselines.add_argument(
            '--bullet',
            action='store_true',
            help='Limit environments to physics-based Bullet environments')

    parser_train_stable_baselines = subparsers.add_parser(
        'train_stable_baselines.sh',
        help='stable-baselines training experiments')
    add_stable_baselines_options(parser_train_stable_baselines)
    parser_train_stable_baselines.set_defaults(
        sh_script='train_stable_baselines.sh',
        func=run_stable_baselines)

    parser_inference_stable_baselines = subparsers.add_parser(
        'inference_stable_baselines.sh',
        help='stable-baselines inference experiments')
    add_stable_baselines_options(parser_inference_stable_baselines)
    parser_inference_stable_baselines.set_defaults(
        sh_script='inference_stable_baselines.sh',
        func=run_stable_baselines)

    args, extra_argv = parser.parse_known_args()
    os.makedirs(args.dir, exist_ok=True)

    global STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS
    STABLE_BASELINES_AVAIL_ENV_IDS, STABLE_BASELINES_UNAVAIL_ENV_IDS = detect_available_env(STABLE_BASELINES_ENV_IDS)
    print("Available env ids:")
    print(textwrap.indent("\n".join(STABLE_BASELINES_AVAIL_ENV_IDS), prefix='  '))
    print("Unavailable env ids:")
    print(textwrap.indent("\n".join(sorted(STABLE_BASELINES_UNAVAIL_ENV_IDS.keys())), prefix='  '))

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
    assert args.sh_script is not None
    obj.run()


class StableBaselines:
    def __init__(self, args, extra_argv):
        # self.parser = parser
        self.args = args
        self.extra_argv = extra_argv
        self.pool = ForkedProcessPool(name="iml_analyze_pool", max_workers=args.workers, debug=self.args.debug)

    def already_ran(self, to_file):
        if not _e(to_file):
            return False
        with open(to_file) as f:
            for line in f:
                line = line.rstrip()
                if re.search(r'IML BENCH DONE', line):
                    return True
        return False

    def _run_cmd(self, cmd, to_file, env=None):
        args = self.args

        if env is None:
            # Make sure iml-analyze get IML_POSTGRES_HOST
            env = dict(os.environ)

        if args.replace or not self.already_ran(to_file):
            tee(
                cmd=cmd + self.extra_argv,
                to_file=to_file,
                env=env,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                with open(to_file, 'a') as f:
                    f.write("IML BENCH DONE")
            if not args.dry_run or _e(to_file):
                assert self.already_ran(to_file)

    def _analyze(self, algo, env_id):
        args = self.args

        iml_directory = self.iml_directory(algo, env_id)
        cmd = ['iml-analyze', "--iml-directory", iml_directory]

        to_file = self._get_logfile(algo, env_id, suffix='analyze.log')

        self._run_cmd(cmd=cmd, to_file=to_file)

    def _get_logfile(self, algo, env_id, suffix='log'):
        args = self.args

        m = re.search(r'(?P<sh_name>.*)\.sh$', args.sh_script)
        sh_name = m.group('sh_name')

        to_file = _j(args.dir, '{sh_name}.algo_{algo}.env_id_{env_id}.{suffix}'.format(
            sh_name=sh_name,
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
        cmd = [args.sh_script, "--iml-directory", iml_directory]

        env = self._sh_env(algo, env_id)

        to_file = self._get_logfile(algo, env_id, suffix='log')

        self._run_cmd(cmd=cmd, to_file=to_file, env=env)

    def is_supported(self, algo, env_id):
        for expr in STABLE_BASELINES_EXPRS:
            if expr.algo == algo and expr.env_id == env_id:
                return expr
        return None

    def should_run(self, algo, env_id):
        if not self.is_supported(algo, env_id):
            return False
        if self.args.bullet and not is_bullet_env(env_id):
            return False
        return True

    def run(self):
        args = self.args

        if args.env_id is not None and args.env_id in STABLE_BASELINES_UNAVAIL_ENV_IDS:
            print("ERROR: env_id={env} is not available since gym.make('{env}') failed.".format(
                env=args.env_id))
            sys.exit(1)

        if args.env_id is not None and args.algo is not None:
            algo_env_pairs = [(args.algo, args.env_id)]
        elif args.env_id is not None:
            algo_env_pairs = []
            for algo in STABLE_BASELINES_ANNOTATED_ALGOS:
                if not self.should_run(algo, args.env_id):
                    continue
                algo_env_pairs.append((algo, args.env_id))
        elif args.algo is not None:
            algo_env_pairs = []
            for env_id in STABLE_BASELINES_AVAIL_ENV_IDS:
                if not self.should_run(args.algo, env_id):
                    continue
                algo_env_pairs.append((args.algo, env_id))
        else:
            print('Please provide either --env-id or --algo', file=sys.stderr)
            sys.exit(1)

        for algo, env_id in algo_env_pairs:
            self._run(algo, env_id)

        if args.analyze:
            for algo, env_id in algo_env_pairs:
                # self._analyze(algo, env_id)
                self.pool.submit(
                    'iml-analyze --iml-directory {iml}'.format(iml=self.iml_directory(algo, env_id)),
                    self._analyze,
                    algo, env_id,
                    sync=self.args.debug_single_thread)

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

def print_cmd(cmd, files=sys.stdout, env=None, dry_run=False):
    if type(cmd) == list:
        cmd_str = " ".join([str(x) for x in cmd])
    else:
        cmd_str = cmd

    lines = []
    if dry_run:
        lines.append("> CMD [dry-run]:")
    else:
        lines.append("> CMD:")
    lines.extend([
        "  $ {cmd}".format(cmd=cmd_str),
        "  PWD={pwd}".format(pwd=os.getcwd()),
    ])

    if env is not None and len(env) > 0:
        env_vars = sorted(env.keys())
        lines.append("  Environment:")
        for var in env_vars:
            lines.append("    {var}={val}".format(
                var=var,
                val=env[var]))
    string = '\n'.join(lines)

    if type(files) not in [set, list]:
        if type(files) in [list]:
            files = set(files)
        else:
            files = set([files])

    for f in files:
        print(string, file=f)
        f.flush()

def is_bullet_env(env_id):
    return re.search(r'BulletEnv', env_id)

if __name__ == '__main__':
    main()
