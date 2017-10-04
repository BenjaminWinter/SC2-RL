"""Runs through the Training Process"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import time
import numpy as np
import os


from pysc2 import maps
from pysc2.env import sc2_env

from pysc2.lib import app
import gflags as flags
import logging

import util.helpers as helpers

import a3c.a3c
import maps.scenarios as scenarios

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("episodes", 1000, "Number of Episodes to run")
flags.DEFINE_integer("training_time", 300, "Number of Seconds to train for")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_string('algorithm', 'test.Test', 'Which Algorithm to run')
flags.DEFINE_bool("validate", False, 'Validation instead of training mode')

flags.mark_flag_as_required("map")


def main(argv):
    logger = logging.getLogger('sc2rl')
    logger.setLevel(logging.INFO)

    if not os.path.isdir('logs'):
        os.mkdir('logs')

    fh = logging.FileHandler('logs/' + time.strftime("%Y%m%d-%H%M%S") + '.log')
    ch = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    np.set_printoptions(threshold=np.nan)

    #stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    #stopwatch.sw.trace = FLAGS.trace
    scenarios.load_scenarios()

    maps.get(FLAGS.map)  # Assert the map exists.

    with helpers.get_env_wrapper() as env:
        a_space = env.get_action_space()
        s_space = env.get_state_space()

    algo_module, algo_name = FLAGS.algorithm.rsplit(".", 1)
    algo_cls = getattr(importlib.import_module(algo_module), algo_name)
    algo = algo_cls(FLAGS.episodes, a_space, s_space)
    algo.run()


if __name__ == "__main__":
  app.run(main)
