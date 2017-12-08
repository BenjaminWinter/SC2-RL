from util.environments.find_ultralisk import FindUltralisk
from pysc2.lib import actions, features
from absl import flags
import logging

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_CREEP = features.SCREEN_FEATURES.creep.index

_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]


class FindUltraliskWithCreep(FindUltralisk):
    def __init__(self, render):
        FindUltralisk.__init__(self, render)
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

        self._input_layers = [_PLAYER_RELATIVE, _CREEP]
