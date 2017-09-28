"""Runs through the Training Process"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import datetime
import numpy as np

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from pysc2.lib import app
import gflags as flags
import logging

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
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
flags.mark_flag_as_required("map")


def main(argv):
    logger = logging.getLogger('sc2rl')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('logs/' + str(datetime.datetime.now()) + '.log')
    ch = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    np.set_printoptions(threshold=np.nan)

    stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
    stopwatch.sw.trace = FLAGS.trace

    maps.get(FLAGS.map)  # Assert the map exists.

    with sc2_env.SC2Env(
            FLAGS.map,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=FLAGS.difficulty,
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=FLAGS.render) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        algo_module, algo_name = FLAGS.algorithm.rsplit(".", 1)
        print("debug")
        algo_cls = getattr(importlib.import_module(algo_module), algo_name)

        algo = algo_cls(env, FLAGS.episodes)
        algo.run()


if __name__ == "__main__":
  app.run(main)
