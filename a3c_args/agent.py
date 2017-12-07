import random
import numpy as np
import gflags as flags
import a3c.shared as shared

FLAGS = flags.FLAGS


class Agent:
    def __init__(self, action_space, e_start=0, e_end=0, e_steps=0):
        self.e_start = e_start or FLAGS.e_start
        self.e_end = e_end or FLAGS.e_end
        self.e_steps = e_steps or FLAGS.e_steps
        self.action_space = action_space

        self.frames = 0
        self.memory = []  # used for n_step return
        self.R = 0.

    def get_epsilon(self):
        if self.frames >= self.e_steps:
            return self.e_end
        else:
            return self.e_start + self.frames * (self.e_end - self.e_start) / self.e_steps  # linearly interpolate

    def act(self, s):
        eps = self.get_epsilon()

        self.frames = self.frames + 1

        if random.random() < eps:
            return random.randint(0, self.action_space - 1), random.randint(0, FLAGS.screen_resolution -1), random.randint(0, FLAGS.screen_resolution -1)

        else:
            s = np.array([s])
            p, xp, yp = shared.brain.predict_p(s)
            p = p[0]
            xp = xp[0]
            yp = yp[0]
            # a = np.argmax(p)
            a = np.random.choice(self.action_space, p=p)
            x = np.random.choice(FLAGS.screen_resolution, p=xp)
            y = np.random.choice(FLAGS.screen_resolution, p=yp)

            return a, x, y

    def train(self, s, a, x, y, r, s_):
        def get_sample(memory, n):
            s, a, x, y, _, _ = memory[0]
            _, _, _, _, _, s_ = memory[n - 1]

            return s, a, x, y, self.R, s_

        a_cats = np.zeros(self.action_space)  # turn action into one-hot representation
        a_cats[a] = 1
        x_cats = np.zeros(FLAGS.screen_resolution)  # turn action into one-hot representation
        x_cats[x] = 1
        y_cats = np.zeros(FLAGS.screen_resolution)  # turn action into one-hot representation
        y_cats[y] = 1

        self.memory.append((s, a_cats, x_cats, y_cats, r, s_))

        self.R = (self.R + r * shared.gamma_n) / FLAGS.gamma

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, x, y, r, s_ = get_sample(self.memory, n)
                shared.brain.train_push(s, a, x, y, r, s_)

                self.R = (self.R - self.memory[0][4]) / FLAGS.gamma
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= FLAGS.n_step_return:
            s, a, x, y, r, s_ = get_sample(self.memory, FLAGS.n_step_return)
            shared.brain.train_push(s, a, x, y, r, s_)

            self.R = self.R - self.memory[0][4]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect
