import random
import numpy as np
import a3c.shared as shared
import gflags as flags

FLAGS = flags.FLAGS


class Runner:
    def __init__(self, remotes=None, action_space=1):
        self.remotes = remotes
        self.frames = 0
        self.action_space = action_space

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        states = [remote.recv() for remote in self.remotes]
        return states

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))

    def get_epsilon(self):
        if self.frames >= FLAGS.e_steps:
            return FLAGS.e_end
        else:
            return FLAGS.e_start + self.frames * (FLAGS.e_end - FLAGS.e_start) / FLAGS.e_steps  # linearly interpolate

    def act(self, s):
        eps = self.get_epsilon()

        self.frames = self.frames + 1

        if random.random() < eps:
            return random.randint(0, self.action_space - 1)

        else:
            s = np.array([s])
            p = shared.brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(self.action_space, p=p)

            return a