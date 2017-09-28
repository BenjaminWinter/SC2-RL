import time
import gflags as flags
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_float('e_start', 0.4, 'Starting Epsilon')
flags.DEFINE_float('e_end', 0.15, 'End Epsilon')
flags.DEFINE_float('e_steps', 80000, 'Number of steps over which to decay Epsilon')
flags.DEFINE_float('gamma', 0.99, 'Discount Value Gamma')
flags.DEFINE_integer('n_step_return', 8, 'N Step Return Value')
flags.DEFINE_float('gamma_n', FLAGS.gamma ** FLAGS.n_step_return, 'Discount Value N Step')
flags.DEFINE_integer('threads', 8, 'Number of Parallel Agents')
flags.DEFINE_integer('optimizers', 2, 'Number of Optimizer Threads')
flags.DEFINE_integer('run_time', 300, 'Number of Seconds to train')

from a3c.environment import Environment
from a3c.optimizer import Optimizer
from a3c.brain import Brain
import a3c.shared as shared


class A3c:
    def __init__(self, _env, episodes):
        """

        :type _env: BaseEnv
        :type episodes Integer
        """
        none_state = np.zeros(_env.get_state_space())
        shared.brain = Brain(_env.get_state_space(), _env.get_action_space(), none_state)

        self.envs = [Environment() for i in range(FLAGS.threads)]
        self.opts = [Optimizer() for i in range(FLAGS.optimizers)]

    def run(self):
        for o in self.opts:
            o.start()

        for e in self.envs:
            e.start()

        time.sleep(FLAGS.run_time)

        for e in self.envs:
            e.stop()
        for e in self.envs:
            e.join()

        for o in self.opts:
            o.stop()
        for o in self.opts:
            o.join()

        print("Training finished")

        run_env = Environment(render=True, e_start=0., e_end=0.)
        run_env.run()
