import multiprocessing as mp
from absl import flags
import time
import logging
from .agent import Agent
import util.helpers as helpers

FLAGS = flags.FLAGS

flags.DEFINE_float('thread_delay', 0.0001, 'Delay of Workers. used to use more Workers than physical CPUs')


class Environment(mp.Process):
    stop_signal = False

    def __init__(self, e_start=0, e_end=0, e_steps=0, sc2env=None, thread_num=999, log_data=False, brain=None, stop=None, t_queue=None, none_state=None):
        super(Environment, self).__init__()
        self.logger = logging.getLogger('sc2rl.' + __name__ + " | " + str(thread_num))
        self.start_time = time.time()

        self.e_start = e_start
        self.episodes = 0
        self.rewards = []
        self.steps = []
        self.log_data = log_data
        self.brain = brain
        self.stop = stop

        if sc2env is not None:
            self.env = sc2env
        else:
            self.env = helpers.get_env_wrapper(render=FLAGS.render)

        self.agent = Agent(self.env.action_space.n, e_start or FLAGS.e_start, e_end or FLAGS.e_end, e_steps or FLAGS.e_steps, brain=brain, t_queue=t_queue, none_state=none_state)

    def run_episode(self):
        # if time.time() - self.start_time > 3600:
        #     self.env = None
        #     self.env = helpers.get_env_wrapper()
        #     self.start_time = time.time()
        self.episodes += 1
        s = self.env.reset()
        R = 0
        step = 0
        while 1:
            step += 1
            time.sleep(FLAGS.thread_delay)  # yield

            a, x, y = self.agent.act(s)
            s_, r, done, info = self.env.step([a, x, y])

            if done:  # terminal state
                s_ = None

            if (not FLAGS.validate) or self.e_start > 0:
                self.agent.train(s, a, x, y, r, s_)

            s = s_
            R += r

            if done or self.stop.value:
                self.episodes += 1
                if self.log_data:
                    self.rewards.append(R)
                    self.steps.append(step)
                break

    def run(self):
        while not self.stop.value:
            self.run_episode()
        self.brain.add_episodes(self.episodes)
        self.brain.add_rewards(self.rewards)
        self.brain.add_steps(self.steps)

