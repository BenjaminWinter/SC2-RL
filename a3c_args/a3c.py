import time
from absl import flags
import numpy as np
import logging
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .environment import Environment
from .optimizer import Optimizer
from .brain import Brain
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import a3c.shared as shared
import random

FLAGS = flags.FLAGS
flags.DEFINE_float('e_start', 0.4, 'Starting Epsilon')
flags.DEFINE_float('e_end', 0.15, 'End Epsilon')
flags.DEFINE_float('e_steps', 80000, 'Number of steps over which to decay Epsilon')
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
        e_starts = [random.uniform(0.2, 0.4) for x in range(FLAGS.threads)]
        e_ends = [random.uniform(0.05, 0.2) for x in range(FLAGS.threads)]

        shared.gamma_n = FLAGS.gamma ** FLAGS.n_step_return
        self.none_state = np.zeros(s_space)
        self.none_state = self.none_state.reshape(s_space)

        self.manager = A3CManager()
        self.manager.start()
        self.q_manager = mp.Manager()
        self.queue = self.q_manager.Queue()

        self.shared_brain = self.manager.Brain(s_space, a_space, self.none_state, t_queue=self.queue)
        self.stop_signal = mp.Value('i', 0)

        if not FLAGS.validate:
            self.envs = [Environment(brain=self.shared_brain, stop=self.stop_signal, t_queue=self.queue, thread_num=i, log_data=True, e_start=e_starts[i], e_end=e_ends[i], none_state=self.none_state) for i in range(FLAGS.threads)]
            self.opts = [Optimizer(brain=self.shared_brain, stop=self.stop_signal, thread_num=i) for i in range(FLAGS.optimizers)]


        #shared.brain = Brain(s_space, a_space, none_state, saved_model=FLAGS.load_model)

        self.runtime = run_time

        self.logger = logging.getLogger('sc2rl.' + __name__ )
        self.logger.info('Starting Up A3C Algorithm')

    def run(self):
        if FLAGS.validate:
            self.logger.info('starting validation')
            run_env = Environment(brain=self.shared_brain, stop=self.stop_signal, t_queue=self.queue, e_start=0., e_end=0., log_data=True, none_state=self.none_state)
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
