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

    def __init__(self, none_state, e_start=.4, sc2env=None, thread_num=999, log_data=False, work_remote=None, t_queue=None):
        super(Environment, self).__init__()
        self.logger = logging.getLogger('sc2rl.' + __name__ + " | " + str(thread_num))
        self.start_time = time.time()
        self.id = thread_num
        self.e_start = e_start
        self.episodes = 0
        self.rewards = []
        self.steps = []
        self.log_data = log_data
        self.work_remote = work_remote
        self.cur_step = 0
        self.episode_r = 0

        if sc2env is not None:
            self.env = sc2env
        else:
            self.env = helpers.get_env_wrapper()

        self.agent = Agent(none_state, self.env.get_action_space(), t_queue=t_queue)

    def close(self, start):
        self.rewards.append(self.episode_r)
        self.steps.append(self.cur_step)
        self.episodes += 1

        self.logger.info('Rewards:')
        self.logger.info(self.rewards)
        self.logger.info('Episodes: ' + str(self.episodes))
        self.logger.info('Steps: ' + str(sum(self.steps)))
        self.logger.info('Seconds: ' + str(time.time() - start))
        self.logger.info('Steps per Second:' + str(sum(self.steps) / (time.time() - start)))
        self.work_remote.close()

    def step(self, action):
        s = self.env.get_state()
        self.cur_step += 1
        s_, r, done, info = self.env.step(action)

        if done:
            s_ = None

        if (not FLAGS.validate) or self.e_start > 0:
            self.agent.train(s, action, r, s_)

        self.episode_r += r

        if done:
            if self.log_data:
                self.rewards.append(self.episode_r)
                self.steps.append(self.cur_step)
                self.episodes += 1

            self.episode_r = 0
            self.cur_step = 0

            s_ = self.env.reset()
            self.work_remote.send(s_)

        self.work_remote.send(s_)

    def run(self):
        start = time.time()
        while 1:
            time.sleep(FLAGS.thread_delay)
            cmd, data = self.work_remote.recv()
            if cmd == "reset":
                self.work_remote.send(self.env.reset())
            if cmd == "close":
                self.close(start)
                break
            if cmd == "step":
                self.step(data)




