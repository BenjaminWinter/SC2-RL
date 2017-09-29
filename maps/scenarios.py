from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class Scenario(lib.Map):
    directory = "rl_scenarios"
    download = "https://cgmgit.beuth-hochschule.de/s62776/SC2-RL/tree/master/maps"
    players = 1
    score_index = 0
    game_steps_per_episode = 0
    step_mul = 8


scenarios = [
    "Vulture_Firebats",
    "Stalkers_Stalkers",
    "Hydras_Zealots"
]


for name in scenarios:
    globals()[name] = type(name, (Scenario,), dict(filename=name))