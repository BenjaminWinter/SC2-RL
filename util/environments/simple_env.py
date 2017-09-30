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


class SimpleEnv(BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self)
        self.logger = logging.getLogger('sc2rl.' + __name__)
        #
        # SC2 action IDs for the high level  actions:
        #  Retreat
        #  Attack
        #  No Action
        #
        self._actions = [_MOVE_SCREEN, _ATTACK_SCREEN, _NO_OP]

    def step(self, action):
        self._env_timestep = self._env.step([self.get_sc2_action(action)])
        r = self._env_timestep[0].reward
        s_ = self.get_state()

        return s_, r, self._env_timestep[0].last(), []

    def get_state(self):
        state = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE]
        return state.reshape(state.shape + (1, ))

    def get_sc2_action(self, action):
        args = {
            0: [[False], helpers.get_attack_coordinates(self._env_timestep[0])],
            1: [[False], helpers.get_retreat_coordinates(self._env_timestep[0])],
            2: []
        }
        if action not in args.keys():
            self.logger.error('action' + action + 'not recognized')
            raise IndexError()

        return actions.FunctionCall(self._actions[action], args.get(action))
