import glob, os, gzip, json
import numpy as np

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import features, actions
from pysc2.lib.protocol import ProtocolError
from a3c.action_args.agent import Agent

from util import helpers
import maps.scenarios as scenarios
import mpyq
import six
import train
import pickle

import a3c.common.shared as shared

from absl import app
from absl import flags
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', 'logs/replays', 'replaydirectory')
flags.DEFINE_bool('raw_rewards', False, 'use environment rewards instead of n-step-rewards')


class FakeQueue():
    def __init__(self):
        self._arr = []

    def put(self, el):
        self._arr.append(el)

    def get(self):
        return self._arr


def main(argv):
    import a3c.common.a3c
    scenarios.load_scenarios()
    run_config = run_configs.get()

    interface = sc_pb.InterfaceOptions()
    interface.raw = False
    interface.score = True
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = FLAGS.screen_resolution
    interface.feature_layer.resolution.y = FLAGS.screen_resolution
    interface.feature_layer.minimap_resolution.x = 64
    interface.feature_layer.minimap_resolution.y = 64

    queue = FakeQueue()
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


            controller.start_replay(start_replay)
            feat = features.Features(controller.game_info())

            obs = controller.observe()
            s = get_obs(env._input_layers, obs)
            results = 0
            last_reward = 0
            while True:
                actions = []
                for a in obs.actions:
                    try:
                        temp = feat.reverse_action(a)
                        x = 0
                        y = 0
                        if temp[0] not in [0, 7]:
                            x = temp.arguments[1][0]
                            y = temp.arguments[1][1]
                        actions.append([env._actions.index(temp[0]), x ,y])
                    except ValueError:
                        pass

                if len(actions) < 1:
                    try:
                        controller.step(FLAGS.step_mul)
                    except ProtocolError:
                        break;

                    obs = controller.observe()
                    s = get_obs(env._input_layers, obs)
                    continue

                r = obs.observation.score.score

                controller.step(FLAGS.step_mul)
                obs = controller.observe()

                s_ = get_obs(env._input_layers, obs)

                if r == 0 and last_reward != 0:
                    s_ = None
                    print('Episode end')

                results += 1

                if not FLAGS.raw_rewards:
                    replay_agent.train(s, actions[0][0], actions[0][1], actions[0][2], r, s_)
                else:
                    queue.put([s, actions[0][0], actions[0][1], actions[0][2], r, s_])

                if obs.player_result:
                    break

                if r == 0 and last_reward != 0:
                    last_reward = 0
                else:
                    s = s_
                    last_reward = r

    with gzip.open('./replay_info/info.gz', 'wb+') as outfile:
        print('pushed: {}'.format(results))
        #json.dump(queue.get(), outfile)
        pickle.dump(queue.get(), outfile)

sf = features.SCREEN_FEATURES
FEATURE_IDS = {
    sf.height_map.index : 'height_map',
    sf.visibility_map.index : 'visibility',
    sf.creep.index : 'creep',
    sf.power.index : 'power',
    sf.player_id.index : 'player_id',
    sf.unit_type.index : 'unit_type',
    sf.selected.index : 'selected',
    sf.unit_hit_points.index : 'unit_hit_points',
    sf.unit_energy.index : 'unit_energy',
    sf.player_relative.index : 'player_relative',
    sf.unit_hit_points_ratio.index : 'unit_hit_points_ratio',
    sf.unit_energy_ratio.index : 'unit_energy_ratio',
    sf.effects.index : 'effects'
}
TYPES = {
    8 : 'i1',
    32 : 'i4'
}


def get_obs(ids, obs):
    layers = []
    for id in ids:
        bytelayer = getattr(obs.observation.feature_layer_data.renders, FEATURE_IDS[id])
        layer = np.frombuffer(bytelayer.data, dtype=TYPES[bytelayer.bits_per_pixel]).reshape((FLAGS.screen_resolution, FLAGS.screen_resolution))
        layers.append(layer.reshape(layer.shape + (1,)))

    if len(ids) < 2:
        return layers[0]
    return np.concatenate(tuple(layers), 2)

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
