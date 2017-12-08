from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
from absl import flags
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


class ReaperZerglings(BaseEnv):
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
        if FLAGS.action_args:
            self._actions = [_ATTACK_SCREEN, _MOVE_SCREEN, _NO_OP, _SELECT_ARMY]
        else:
            self._actions = [_ATTACK_SCREEN, _ATTACK_SCREEN, _ATTACK_SCREEN, _ATTACK_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN]
        self._input_layers = [_PLAYER_RELATIVE, _HEIGHT_MAP]

    def get_sc2_action(self, action):
        if FLAGS.action_args:
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
        else:
            args = [_NOT_QUEUED, helpers.get_shifted_position(action % 4, self._env_timestep[0], 15)]
            return actions.FunctionCall(self._actions[action], args)