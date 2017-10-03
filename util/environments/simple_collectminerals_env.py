from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
import gflags as flags
import logging
import util.helpers as helpers

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.Select_army.id


class SimpleCollectMineralEnv(BaseEnv):
    def __init__(self):
        BaseEnv.__init__(self)
        self.logger = logging.getLogger('sc2rl.' + __name__)
        #
        # SC2 action IDs for the high level  actions:
        #  LEFT
        #  RIGHT
        #  UP
        #  DOWN
        #  SELECT ARMY
        #
        self._actions = [_MOVE_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN]

    def step(self, action):
        self._env_timestep = self._env.step([self.get_sc2_action(action)])
        r = self._env_timestep[0].reward
        s_ = self.get_state()

        return s_, r, self._env_timestep[0].last(), []

    def get_state(self):
        state = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE]
        return state.reshape(state.shape + (1, ))

    def get_sc2_action(self, action):
        args = helpers.get_shifted_position(action, self._env_timestep[0], 10)
        return actions.FunctionCall(self._actions[action], args)
