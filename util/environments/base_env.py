from pysc2.env import sc2_env
from pysc2.env import available_actions_printer
import gflags as flags

FLAGS = flags.FLAGS

class BaseEnv():
    def __init__(self):
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
            self._env = available_actions_printer.AvailableActionsPrinter(env)
            self._actions = [0]

    def step(self, action):
        pass

    def reset(self):
        pass

    def get_available_actions(self):
        pass

    def get_action_space(self):
        return len(self._actions)

    def get_state_space(self):
        return self.get_state().flatten().shape[0]

    def get_state(self):
        pass
