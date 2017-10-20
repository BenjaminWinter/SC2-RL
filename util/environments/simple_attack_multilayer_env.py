from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
import gflags as flags
import logging
import util.helpers as helpers
import numpy as np

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_CREEP = features.SCREEN_FEATURES.creep.index
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_NOT_QUEUED = [False]


class SimpleAttackMultilayerEnv(BaseEnv):
    def __init__(self, render):
        BaseEnv.__init__(self, render)
        self.logger = logging.getLogger('sc2rl.' + __name__)
        self.logger.info('starting ' + __name__ + ' environment')
        #
        # SC2 action IDs for the high level  actions:
        #  LEFT
        #  RIGHT
        #  UP
        #  DOWN
        #  SELECT ARMY
        #
        self._actions = [_ATTACK_SCREEN, _ATTACK_SCREEN, _ATTACK_SCREEN, _ATTACK_SCREEN]

    def _step(self, action):
        self._env_timestep = self._env.step([self.get_sc2_action(action)])
        r = self._env_timestep[0].reward
        s_ = self.get_state()

        return s_, r, self._env_timestep[0].last(), {}

    def get_state(self):
        layer1 = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE] * 255 / 4
        layer1 = layer1.reshape(layer1.shape + (1, ))
        layer2 = self._env_timestep[0].observation['screen'][_CREEP] * 255
        layer2 = layer2.reshape(layer2.shape + (1, ))

        return np.concatenate((layer1, layer2), 2)

    def get_sc2_action(self, action):
        args = [_NOT_QUEUED, helpers.get_shifted_position(action, self._env_timestep[0], 10)]
        return actions.FunctionCall(self._actions[action], args)
