import json, argparse, glob, os, math
import numpy as np

from pysc2 import maps
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import renderer_human
from pysc2.lib import stopwatch
from pysc2.lib import features
from a3c.action_args.agent import Agent
from util import helpers
import multiprocessing as mp
import mpyq
import six
import train

import a3c.common.shared as shared

from absl import app
from absl import flags
from s2clientprotocol import sc2api_pb2 as sc_pb

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', 'logs/replays', 'replaydirectory')


def main(argv):
    import a3c.common.a3c

    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.raw = False
    interface.score = True
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = FLAGS.screen_resolution
    interface.feature_layer.resolution.y = FLAGS.screen_resolution
    interface.feature_layer.minimap_resolution.x = 64
    interface.feature_layer.minimap_resolution.y = 64

    queue = mp.Queue()
    #
    shared.gamma_n = FLAGS.gamma ** FLAGS.n_step_return
    env = helpers.get_env_wrapper(False)
    s_space = env.observation_space.shape

    none_state = np.zeros(s_space)
    none_state = none_state.reshape(s_space)
    replay_agent = Agent(env.action_space.n, t_queue=queue, none_state=none_state)


    for fname in glob.glob(os.path.join(FLAGS.dir, '*.SC2Replay')):
        replay_data = run_config.replay_data(fname)
        start_replay = sc_pb.RequestStartReplay(
            replay_data=replay_data,
            options=interface,
            disable_fog=True,
            observed_player_id=1)
        game_version = get_game_version(replay_data)
        with run_config.start(game_version=game_version,
                              full_screen=False) as controller:
            feat = features.Features(controller.game_info())

            controller.start_replay(start_replay)
            obs = controller.observe()
            s = helpers.get_input_layers(env.input_layers, obs[0])
            while True:
                action = feat.reverse_action(obs.actions[0])
                controller.step(FLAGS.step_mul)
                obs = controller.observe()
                r = obs.reward
                s_ = helpers.get_input_layers(env.input_layers, obs[0])

                if obs.done:
                    s_ = None
                x = 0
                y = 0
                replay_agent.train(s, action, x, y, r, s_)

                if obs.done:
                    controller.step(FLAGS.step_mul)
                    obs = controller.observe()
                    s = helpers.get_input_layers(env.input_layers, obs[0])
                else:
                    s = s_

                if obs.player_result:
                    break
    with open('./replay_info/info.json', 'w+') as outfile:
        obj = []
        while not queue.empty():
            obj.append(queue.get())

        outfile.write(json.dump(obj))


def get_game_version(replay_data):
  replay_io = six.BytesIO()
  replay_io.write(replay_data)
  replay_io.seek(0)
  archive = mpyq.MPQArchive(replay_io).extract()
  metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
  version = metadata["GameVersion"]
  return ".".join(version.split(".")[:-1])

if __name__ == '__main__':
    app.run(main)
