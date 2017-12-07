from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
import gflags as flags
import logging
import util.helpers as helpers

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id


class SimpleVultureEnv(BaseEnv):
    def __init__(self, render):
        BaseEnv.__init__(self, render)
        self.logger = logging.getLogger('sc2rl.' + __name__)
        self.logger.info('starting ' + __name__ + ' environment')
        #
        # SC2 action IDs for the high level  actions:
        #  Retreat
        #  Attack
        #  No Action
        #
        self._actions = [_MOVE_SCREEN, _ATTACK_SCREEN, _NO_OP]

    def get_state(self):
        state = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE]
        return state.reshape(state.shape + (1, ))

    def _get_sc2_action(self, action):
        if action == 0:
            args = [[False], helpers.get_attack_coordinates(self._env_timestep[0])]
        elif action == 1:
            args = [[False], helpers.get_retreat_coordinates(self._env_timestep[0])]
        elif action == 2:
            args = []
        else:
            self.logger.error('Action Not Recognised:' + str(action))
            raise KeyError()

        return actions.FunctionCall(self._actions[action], args)
