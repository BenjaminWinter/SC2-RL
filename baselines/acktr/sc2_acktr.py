#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.acktr.policies import CnnPolicy
import gflags as flags
import util.helpers as helpers

FLAGS = flags.FLAGS

class Acktr:
    def __init__(self, run_time, a_space, s_space):
        pass

    def run(self):
        self.train(num_frames=FLAGS.run_time, seed=FLAGS.seed, num_cpu=FLAGS.threads)

    def train(self, num_frames, seed, num_cpu):

        def make_env(rank):
            def _thunk():
                env = helpers.get_env_wrapper(render=FLAGS.render)
                env.seed(seed + rank)
                if logger.get_dir():
                    env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
                gym.logger.setLevel(logging.WARN)
                return env

            return _thunk

        set_global_seeds(seed)
        if FLAGS.validate:
            from util.environments.sim_proc_env import SimProcEnv
            env = SimProcEnv(render=True)
        else:
            env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        policy_fn = CnnPolicy
        learn(policy_fn, env, seed, total_timesteps=num_frames, nprocs=num_cpu)
        env.close()
