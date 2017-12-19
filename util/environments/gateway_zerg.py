from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
from absl import flags
import logging
import numpy as np
import util.helpers as helpers

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_FORCE_FIELD_SCREEN = actions.FUNCTIONS.Effect_ForceField_screen.id
_GUARDIAN_SHIELD = actions.FUNCTIONS.Effect_GuardianShield_quick.id

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]


class GatewayZerg(BaseEnv):
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
        self._actions = [_ATTACK_SCREEN, _FORCE_FIELD_SCREEN, _GUARDIAN_SHIELD, _NO_OP]
        self._input_layers = [_PLAYER_RELATIVE, _HIT_POINTS, _UNIT_TYPE]

    def get_sc2_action(self, action):
        a, x, y = action
        if self._actions[a] not in self._env_timestep[0].observation['available_actions']:
            return actions.FunctionCall(_NO_OP, [])
        
        if self._actions[a] in [_ATTACK_SCREEN, _FORCE_FIELD_SCREEN]:
            args = [_NOT_QUEUED, [x, y]]
        elif self._actions[a] == _NO_OP:
            args = []
        elif self._actions[a] in [_SELECT_ARMY, _GUARDIAN_SHIELD]:
            args = [_NOT_QUEUED]
        else:
            raise ValueError("Cant find action")

        return actions.FunctionCall(self._actions[a], args)
