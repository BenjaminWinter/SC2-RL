import logging
import time, json
import multiprocessing as mp
from multiprocessing.managers import BaseManager

import matplotlib as mpl
import pickle
import sys
mpl.use('Agg')

import numpy as np
from absl import flags
import matplotlib.pyplot as plt

from .optimizer import Optimizer
import a3c.common.shared as shared


FLAGS = flags.FLAGS
FLAGS(sys.argv)


if FLAGS.action_args:
    from a3c.action_args.environment import Environment
    from a3c.action_args.brain import Brain
else:
    from a3c.standard.environment import Environment
    from a3c.standard.brain import Brain

flags.DEFINE_float('gamma', 0.99, 'Discount Value Gamma')
flags.DEFINE_integer('n_step_return', 8, 'N Step Return Value')
flags.DEFINE_integer('optimizers', 2, 'Number of Optimizer Threads')

class A3CManager(BaseManager):
    pass

A3CManager.register('Brain', Brain)

class A3c:
    def __init__(self, run_time, a_space, s_space):
        """
        :type _env: BaseEnv
        :type episodes Integer
        """

        shared.gamma_n = FLAGS.gamma ** FLAGS.n_step_return
        self.none_state = np.zeros(s_space)
        self.none_state = self.none_state.reshape(s_space)

        self.replay_data = None
        if FLAGS.replay_file:
            with open(FLAGS.replay_file, 'rb') as f:
                self.replay_data = pickle.load(f)

        print(len(self.replay_data))
        self.manager = A3CManager()
        self.manager.start()
        self.q_manager = mp.Manager()
        self.queue = self.q_manager.Queue()

        self.shared_brain = self.manager.Brain(s_space, a_space, self.none_state, t_queue=self.queue, replay_data=self.replay_data)
        self.stop_signal = mp.Value('i', 0)

        if not FLAGS.validate:
            self.envs = [Environment(brain=self.shared_brain, stop=self.stop_signal, t_queue=self.queue, thread_num=i, log_data=True, none_state=self.none_state) for i in range(FLAGS.threads)]
            self.opts = [Optimizer(brain=self.shared_brain, stop=self.stop_signal, thread_num=i) for i in range(FLAGS.optimizers)]


        #shared.brain = Brain(s_space, a_space, none_state, saved_model=FLAGS.load_model)

        self.runtime = run_time

        self.logger = logging.getLogger('sc2rl.' + __name__ )
        self.logger.info('Starting Up A3C Algorithm')

    def run(self):
        if FLAGS.validate:
            self.logger.info('starting validation')
            run_env = Environment(brain=self.shared_brain, stop=self.stop_signal, t_queue=self.queue, log_data=True, none_state=self.none_state)
            run_env.start()
            time.sleep(FLAGS.run_time)
            run_env.stop()
            run_env.join()
            plt.plot(run_env.rewards, 'r')
            plt.show()
            return

        self.logger.info('starting Training')
        # tracemalloc.start()
        # yappi.start()
        for o in self.opts:
            o.start()

        for e in self.envs:
            e.start()

        for i in range(20):
            time.sleep(FLAGS.run_time/20)
            self.logger.info("Progress:" + str((i+1)*5) + "%")
            self.shared_brain.save_model(FLAGS.save_model + ".checkpoint." + str(i))

        self.stop_signal.value = 1

        for e in self.envs:
            e.join()

        for o in self.opts:
            o.join()

        rewards = self.shared_brain.get_rewards()
        self.logger.info('Rewards:')
        self.logger.info(rewards)
        self.logger.info('Episodes:' + str(self.shared_brain.get_episodes()))
        steps = sum(sum(x) for x in self.shared_brain.get_steps())
        self.logger.info('Steps:' + str(steps))
        self.logger.info('Steps per Second: ' + str(steps / FLAGS.run_time))
        for x in rewards:
            plt.plot(x)
        plt.savefig('logs/plot.png')

        print("Training finished")

        self.shared_brain.save_model(FLAGS.save_model + ".final")
