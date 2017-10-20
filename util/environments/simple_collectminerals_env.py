from util.environments.base_env import BaseEnv
from pysc2.lib import actions, features
import gflags as flags
import logging
import util.helpers as helpers

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [False]
class SimpleCollectMineralEnv(BaseEnv):
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
        self._actions = [_MOVE_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN, _MOVE_SCREEN]

    def _step(self, action):
        self._env_timestep = self._env.step([self.get_sc2_action(action)])
        r = self._env_timestep[0].reward
        s_ = self.get_state()

        return s_, r, self._env_timestep[0].last(), {}

    def get_state(self):
        state = self._env_timestep[0].observation['screen'][_PLAYER_RELATIVE] * 255 /4
        return state.reshape(state.shape + (1, ))

    def get_sc2_action(self, action):
        args = [_NOT_QUEUED, helpers.get_shifted_position(action, self._env_timestep[0], 10)]
        return actions.FunctionCall(self._actions[action], args)
