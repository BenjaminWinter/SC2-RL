import math
import numpy as np
import gflags as flags

from pysc2.lib import features
from pysc2.lib import actions

FLAGS = flags.FLAGS

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_VISIBILITY = features.SCREEN_FEATURES.visibility_map.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id


def rotate_point(center, point, angle):
    py, px = point
    cy, cx = center

    qx = px + math.cos(angle) * (cx - px) - math.sin(angle) * (cy - py)
    qy = py + math.sin(angle) * (cx - px) + math.cos(angle) * (cy - py)
    return np.array([qy, qx])


def get_attack_coordinates(obs):
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    if not enemy_y.any():
        return [32, 32]
    pos = zip(enemy_y, enemy_x)

    dist = 999
    minX = 0
    minY = 0

    for y, x in pos:
        newDist = math.sqrt((x - player_x[0]) ** 2 + (y - player_y[0]) ** 2)
        if newDist < dist:
            dist = newDist
            minX = x
            minY = y

    return np.array([minX, minY])
# def get_attack_coordinates(obs):
#     player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
#     enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()
#     player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
#     if not enemy_y.any():
#         return [32, 32]
#     return (np.array(enemy_y[0] - enemy_x[0]) - np.array(player_y[0], player_x[0]))*0.1 + np.array(player_y[0], player_x[0])


def get_retreat_coordinates(obs):
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    visibility = obs.observation["screen"][_VISIBILITY]

    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

    player_pos = np.array([player_y[0], player_x[0]])
    target = np.flip(get_attack_coordinates(obs), 0)
    screen_space = player_relative.shape

    retreat_vector = target - player_pos
    retreat_vector = retreat_vector / np.linalg.norm(retreat_vector)
    retreat_vector *= 25

    retreat_point = (player_pos - retreat_vector)
    tried = 1
    dir = 1

    while (
        retreat_point[0] < 0 or
        retreat_point[0] >= screen_space[0] or
        retreat_point[1] < 0 or
        retreat_point[1] >= screen_space[1] or
        visibility[int(retreat_point[0])][int(retreat_point[1])] != 2
    ):
        retreat_point = (rotate_point(retreat_point, player_pos, math.radians(10 * tried * dir)))
        dir *= -1
        tried += 1

    result = np.flip(retreat_point.astype(int), 0)
    return result


def get_available_actions():
    return ['ATTACK', 'RETREAT', '_NO_OP']


def get_sc2_action(next_action, obs):
    if next_action == _ATTACK_SCREEN:
        args = [[False], get_attack_coordinates(obs)]
    elif next_action == _MOVE_SCREEN:
        args = [[False], get_retreat_coordinates(obs)]
    elif next_action == _SELECT_ARMY:
        args = [[False]]
    elif next_action == _NO_OP:
        args = []
    else:
        raise ValueError("Action not recognised")

    return actions.FunctionCall(next_action, args)


def get_env_wrapper():

    if FLAGS.map in ['Vulture_Firebats', 'Marine_Zerglings']:
        from util.environments.simple_vulture_env import SimpleVultureEnv
        return SimpleVultureEnv()
    if FLAGS.map == 'CollectMineralShardsMod' or FLAGS.map == 'MoveToBeaconMod':
        from util.environments.simple_collectminerals_env import SimpleCollectMineralEnv
        return SimpleCollectMineralEnv()
    if FLAGS.map == "FindZergling":
        from util.environments.simple_attack_env import SimpleAttackEnv
        return SimpleAttackEnv()

def get_shifted_position(direction, obs, dist):
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    if not player_x.any():
        return [42, 42]
    player = [player_y[0], player_x[0]]
   # ORDER: left,right,up,down
    if direction == 0:
        player[1] -= dist
    elif direction == 1:
        player[1] += dist
    elif direction == 2:
        player[0] -= dist
    elif direction == 3:
        player[0] += dist
    return[min(max(0, player[1]), 84 - 1), min(max(0, player[0]), 84 - 1)]
