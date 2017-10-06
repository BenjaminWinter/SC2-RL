import threading
import gflags as flags
import time
import logging
from .agent import Agent
import util.helpers as helpers
import multiprocessing as mp

FLAGS = flags.FLAGS

flags.DEFINE_float('thread_delay', 0.0001, 'Delay of Workers. used to use more Workers than physical CPUs')


class Environment(mp.Process):
    stop_signal = False

    def __init__(self, e_start=0, e_end=0, e_steps=0, sc2env=None, thread_num=999, log_data=False, brain=None):
        super(Environment, self).__init__()
        self.logger = logging.getLogger('sc2rl.' + __name__ + " | " + str(thread_num))
        self.start_time = time.time()

        self.e_start = e_start
        self.episodes = 0
        self.rewards = []
        self.steps = []
        self.log_data = log_data
        if sc2env is not None:
            self.env = sc2env
        else:
            self.env = helpers.get_env_wrapper()

        self.agent = Agent(self.env.get_action_space(), e_start or FLAGS.e_start, e_end or FLAGS.e_end, e_steps or FLAGS.e_steps, brain=brain)

    def run_episode(self):
        if time.time() - self.start_time > 1800:
            self.env = None
            time.sleep(1)
            self.env = helpers.get_env_wrapper()
            self.start_time = time.time()
        self.episodes += 1
        s = self.env.reset()
        R = 0
        step = 0
        while 1:
            step += 1
            time.sleep(FLAGS.thread_delay)  # yield

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            if (not FLAGS.validate) or self.e_start > 0:
                self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                self.episodes += 1
                if self.log_data:
                    self.rewards.append(R)
                break

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True

