import threading
import gflags as flags
import time
import logging
from a3c.agent import Agent
from util.environments.simple_env import SimpleEnv

FLAGS = flags.FLAGS

flags.DEFINE_float('thread_delay', 0.0005, 'Delay of Workers. used to use more Workers than physical CPUs')


class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, e_start=FLAGS.e_start, e_end=FLAGS.e_end, e_steps=FLAGS.e_steps, sc2env=None, thread_num=999):
        threading.Thread.__init__(self)
        self.logger = logging.getLogger('sc2rl.' + __name__ + " | " + thread_num)

        self.episodes = 0
        self.rewards = []
        self.steps = []

        if sc2env is not None:
            self.env = sc2env
        else:
            self.env = SimpleEnv()

        self.agent = Agent(self.env.get_action_space(), e_start, e_end, e_steps)

    def run_episode(self):
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

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                self.episodes += 1
                self.logger.info('')
                self.rewards.append(R)
                self.steps.append(step)
                break

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True
