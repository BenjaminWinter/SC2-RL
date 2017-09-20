# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define the mini game map configs. These are maps made by Deepmind."""

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
    "Hydras_Zealots",
    "Stalkers_Stalkers"
]


for name in scenarios:
    globals()[name] = type(name, (Scenario,), dict(filename=name))