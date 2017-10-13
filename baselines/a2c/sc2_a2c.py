#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import gflags as flags
import util.helpers as helpers

FLAGS = flags.FLAGS

flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'lnlstm'], 'Policy architecture')
flags.DEFINE_enum('lrschedule', 'constant',['linear', 'constant'], 'Learning rate schedule')
flags.DEFINE_integer('seed', 0, 'RNG seed')


def train(num_frames, seed, policy, lrschedule, num_cpu):

    def make_env(rank):
        def _thunk():
            env = helpers.get_env_wrapper()
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and 
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, total_timesteps=num_frames, lrschedule=lrschedule)
    env.close()


class A2c:
    def __init__(self, run_time, a_space, s_space):
        pass

    def run(self):
        self.train(num_frames=FLAGS.run_time, seed=FLAGS.seed,
              policy=FLAGS.policy, lrschedule=FLAGS.lrschedule, num_cpu=FLAGS.threads)

    def train(self, num_frames, seed, policy, lrschedule, num_cpu):

        def make_env(rank):
            def _thunk():
                env = helpers.get_env_wrapper()
                env.seed(seed + rank)
                env = bench.Monitor(env, logger.get_dir() and
                                    os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
                gym.logger.setLevel(logging.WARN)
                return env

            return _thunk

        set_global_seeds(seed)
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        if policy == 'cnn':
            policy_fn = CnnPolicy
        elif policy == 'lstm':
            policy_fn = LstmPolicy
        elif policy == 'lnlstm':
            policy_fn = LnLstmPolicy
        learn(policy_fn, env, seed, total_timesteps=num_frames, lrschedule=lrschedule)
        env.close()
