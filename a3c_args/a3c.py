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
import a3c.shared as shared
import random

FLAGS = flags.FLAGS
flags.DEFINE_float('e_start', 0.4, 'Starting Epsilon')
flags.DEFINE_float('e_end', 0.15, 'End Epsilon')
flags.DEFINE_float('e_steps', 80000, 'Number of steps over which to decay Epsilon')
flags.DEFINE_float('gamma', 0.99, 'Discount Value Gamma')
flags.DEFINE_integer('n_step_return', 8, 'N Step Return Value')
flags.DEFINE_integer('optimizers', 1, 'Number of Optimizer Threads')
flags.DEFINE_string('load_model', None, 'Keras model to load')
flags.DEFINE_string('save_model', 'models/training_model', 'Where to save Keras model')



class A3c:
    def __init__(self, run_time, a_space, s_space):
        """
        :type _env: BaseEnv
        :type episodes Integer
        """
        e_starts = [random.uniform(0.2, 0.4) for x in range(FLAGS.threads)]
        e_ends = [random.uniform(0.05, 0.2) for x in range(FLAGS.threads)]

        shared.gamma_n = FLAGS.gamma ** FLAGS.n_step_return

        if not FLAGS.validate:
            self.envs = [Environment(thread_num=i, log_data=True, e_start=e_starts[i], e_end=e_ends[i]) for i in range(FLAGS.threads)]
            self.opts = [Optimizer(thread_num=i) for i in range(FLAGS.optimizers)]

        none_state = np.zeros(s_space)
        none_state = none_state.reshape(s_space)
        shared.brain = Brain(s_space, a_space, none_state, saved_model=FLAGS.load_model)

        self.runtime = run_time

        self.logger = logging.getLogger('sc2rl.' + __name__ )
        self.logger.info('Starting Up A3C Algorithm')

    def run(self):
        if FLAGS.validate:
            self.logger.info('starting validation')
            run_env = Environment(e_start=0., e_end=0., log_data=True)
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
            shared.brain.model.save(FLAGS.save_model + ".checkpoint." + str(i))
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
            self.logger.info(e.rewards)
            plt.plot(e.rewards)
            e.join()
        self.logger.info('Episodes: ' + str(episodes))
        self.logger.info('Steps per Second: ' + str(sps / FLAGS.run_time))
        plt.savefig('logs/plot.png')
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

        shared.brain.model.save(FLAGS.save_model + ".final")
        # run_env = Environment(e_start=0., e_end=0., log_data=True)
        # run_env.start()
        # time.sleep(300)
        # run_env.stop()
        # run_env.join()
        # self.logger.info('run_env Rewards:')
        # self.logger.info(run_env.rewards)
        # plt.plot(run_env.rewards, 'r')
        # plt.savefig('logs/plot.png')
