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

from absl import app
from absl import flags
import logging

import util.helpers as helpers

import maps.scenarios as scenarios
import sys

FLAGS = flags.FLAGS
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("validate", False, 'Validation instead of training mode')
flags.DEFINE_bool("action_args", False, 'wether simple environment or envs'
                                        ' with arguments should be used')
flags.DEFINE_integer('history_size', 4, 'Number of Steps for an environment to remember')
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer('run_time', 300, 'Number of Seconds/Steps to train')
flags.DEFINE_integer('threads', 8, 'Number of Parallel Agents')
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_string('algorithm', 'test.Test', 'Which Algorithm to run')
flags.DEFINE_string('load_model', None, 'Keras model to load')
flags.DEFINE_string('save_model', 'models/training_model', 'Where to save Keras model')
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

    with helpers.get_env_wrapper(render=False) as env:
        a_space = env.action_space.n
        s_space = env.observation_space.shape
    algo_module, algo_name = FLAGS.algorithm.rsplit(".", 1)
    algo_cls = getattr(importlib.import_module(algo_module), algo_name)
    algo = algo_cls(FLAGS.run_time, a_space, s_space)
    algo.run()
    print('sys exit')
    sys.exit()

if __name__ == "__main__":
  app.run(main)
