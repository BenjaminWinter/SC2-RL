from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
import gflags as flags
import logging
import util.helpers as helpers
import numpy as np

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]


class AttackEnv(BaseEnv):
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
        self._actions = [_ATTACK_SCREEN, _MOVE_SCREEN, _NO_OP, _SELECT_ARMY]

    def get_state(self):
        layer1 = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE] * 255 / 4
        layer1 = layer1.reshape(layer1.shape + (1, ))
        layer2 = self._env_timestep[0].observation['screen'][_HEIGHT_MAP] * 40
        layer2 = layer2.reshape(layer2.shape + (1, ))

        return np.concatenate((layer1, layer2), 2)

    def get_sc2_action(self, action):
        a, x, y = action
        if self._actions[a] in [_ATTACK_SCREEN, _MOVE_SCREEN]:
            args = [_NOT_QUEUED, [x, y]]
        elif self._actions[a] == _NO_OP:
            args = []
        elif self._actions[a] == _SELECT_ARMY:
            args = [_NOT_QUEUED]
        else:
            raise ValueError("Cant find action")
        return actions.FunctionCall(self._actions[a], args)
