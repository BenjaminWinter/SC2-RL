from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
import gflags as flags

import util.helpers as helpers

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id


class SimpleEnv(BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self)
        self._env_timestep = self._env.reset()
        #
        # SC2 action IDs for the high level  actions:
        #  Retreat
        #  Attack
        #  No Action
        #
        self._actions = [_MOVE_SCREEN, _ATTACK_SCREEN, _NO_OP]

    def step(self, action):
        s = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE]

        self._env_timestep = self._env.step(self.get_sc2_action(action))
        r = self._env_timestep[0].reward
        s_ = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE]

        return s, action, r, s_

    def reset(self):
        self._env_timestep = self._env.reset()

    def get_available_actions(self):
        return zip(self._actions, ['Retreat', 'Attack', 'None'])

    def get_state(self):
        return self._env_timestep[0]['screen'][_PLAYER_RELATIVE]

    def get_sc2_action(self, action):
        if self._actions[action] == _ATTACK_SCREEN:
            args = [[False], helpers.get_attack_coordinates(self._env[0])]
        elif self._actions[action] == _MOVE_SCREEN:
            args = [[False], helpers.get_retreat_coordinates(self._env(0))]
        elif self._actions[action] == _NO_OP:
            args = []
        else:
            raise ValueError("Action not recognised")

        return actions.FunctionCall(self._actions[action], args)
