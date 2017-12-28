import random

import numpy as np
from absl import flags

import a3c.common.shared as shared

FLAGS = flags.FLAGS


class Agent:
    def __init__(self, action_space, brain=None, t_queue=None, none_state=None):
        self.action_space = action_space
        self.brain = brain
        self.queue = t_queue
        self.none_state = none_state
        self.frames = 0
        self.memory = []  # used for n_step return
        self.R = 0.

    def act(self, s):

        self.frames = self.frames + 1

        s = np.array([s])
        p = self.brain.predict_p(s)[0]

        # a = np.argmax(p)
        a = np.random.choice(self.action_space, p=p)

        return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(self.action_space)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * shared.gamma_n) / FLAGS.gamma

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / FLAGS.gamma
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= FLAGS.n_step_return:
            s, a, r, s_ = get_sample(self.memory, FLAGS.n_step_return)
            self.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect

    def train_push(self, s, a, r, s_):
        if s_ is None:
            self.queue.put([s, a, r, self.none_state, 0.])
        else:
            self.queue.put([s, a, r, s_, 1.])
