from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
from absl import flags
import logging
import numpy as np
import util.helpers as helpers

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_HIT_POINTS_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_BLINK_SCREEN = actions.FUNCTIONS.Effect_Blink_screen.id
_SELECT_POINT_SCREEN = actions.FUNCTIONS.select_point.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]


class StalkersRoaches(BaseEnv):
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
        self._actions = [_BLINK_SCREEN, _SELECT_POINT_SCREEN, _NO_OP,]
        self._input_layers = [_PLAYER_RELATIVE, _HIT_POINTS, _SELECTED, _UNIT_TYPE, _HIT_POINTS_RATIO]

    def get_sc2_action(self, action):
        a, x, y = action
        if self._actions[a] not in self._env_timestep[0].observation['available_actions']:
            return actions.FunctionCall(_NO_OP, [])
        
        if self._actions[a] in [_BLINK_SCREEN, _SELECT_POINT_SCREEN]:
            args = [_NOT_QUEUED, [x, y]]
        elif self._actions[a] == _NO_OP:
            args = []
        else:
            raise ValueError("Cant find action")

        return actions.FunctionCall(self._actions[a], args)
