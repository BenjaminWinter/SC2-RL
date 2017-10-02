import time
import datetime
import gflags as flags
import numpy as np
import logging
import yappi

from .environment import Environment
from .optimizer import Optimizer
from .brain import Brain
import a3c.shared as shared
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_float('e_start', 0.4, 'Starting Epsilon')
flags.DEFINE_float('e_end', 0.15, 'End Epsilon')
flags.DEFINE_float('e_steps', 80000, 'Number of steps over which to decay Epsilon')
flags.DEFINE_float('gamma', 0.99, 'Discount Value Gamma')
flags.DEFINE_integer('n_step_return', 8, 'N Step Return Value')
flags.DEFINE_integer('threads', 8, 'Number of Parallel Agents')
flags.DEFINE_integer('optimizers', 1, 'Number of Optimizer Threads')
flags.DEFINE_integer('run_time', 300, 'Number of Seconds to train')


class A3c:
    def __init__(self, episodes, a_space, s_space):
        """
        :type _env: BaseEnv
        :type episodes Integer
        """
        shared.gamma_n = FLAGS.gamma ** FLAGS.n_step_return

        self.envs = [Environment(thread_num=i) for i in range(FLAGS.threads)]
        self.opts = [Optimizer(thread_num=i) for i in range(FLAGS.optimizers)]

        none_state = np.zeros(s_space)
        none_state = none_state.reshape((FLAGS.screen_resolution, FLAGS.screen_resolution, 1))
        shared.brain = Brain(s_space, a_space, none_state, saved_model="models/training_model")

        self.runtime = FLAGS.run_time

        self.logger = logging.getLogger('sc2rl.' + __name__ )
        self.logger.info('Starting Up A3C Algorithm')

    def run(self):
        # tracemalloc.start()
        # yappi.start()
        for o in self.opts:
            o.start()

        for e in self.envs:
            e.start()

        time.sleep(FLAGS.run_time)

        episodes = 0
        sps=0
        for e in self.envs:
            e.stop()
        for e in self.envs:
            self.logger.info('Rewards:')
            self.logger.info(np.array(e.rewards))
            self.logger.info('Steps: ' + str(e.steps))
            sps += sum(e.steps)
            episodes += e.episodes
            e.join()
        self.logger.info('Episodes: ' + str(episodes))
        self.logger.info('Steps per Second: ' + str(sps / FLAGS.run_time))

        for o in self.opts:
            o.stop()
        for o in self.opts:
            o.join()
        # yappi.stop()
        #
        # OUT_FILE = 'logs/profiling'
        # print('[YAPPI WRITE]')
        #
        # stats = yappi.get_func_stats()
        #
        # for stat_type in ['pstat', 'callgrind', 'ystat']:
        #     print('writing {}.{}'.format(OUT_FILE, stat_type))
        #     stats.save('{}.{}'.format(OUT_FILE, stat_type), type=stat_type)
        #
        # print('\n[YAPPI FUNC_STATS]')
        #
        # print('writing {}.func_stats'.format(OUT_FILE))
        # with open('{}.func_stats'.format(OUT_FILE), 'wb') as fh:
        #     stats.print_all(out=fh)
        #
        # print('\n[YAPPI THREAD_STATS]')
        #
        # print('writing {}.thread_stats'.format(OUT_FILE))
        # tstats = yappi.get_thread_stats()
        # with open('{}.thread_stats'.format(OUT_FILE), 'wb') as fh:
        #     tstats.print_all(out=fh)
        #
        # print('[YAPPI OUT]')

        print("Training finished")

        shared.brain.model.save('models/training_model')
        # run_env = Environment(e_start=0., e_end=0., log_data=True)
        # run_env.run()
        # time.sleep(300)
        # run_env.stop()
        # plt.plot(e.rewards, 'r')
        # plt.plot(e.steps, 'g')
        # plt.show()
