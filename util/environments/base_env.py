from pysc2.env import sc2_env
from pysc2.env import available_actions_printer
import gflags as flags
import time
FLAGS = flags.FLAGS


class BaseEnv:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __init__(self):
        env = sc2_env.SC2Env(
                FLAGS.map,
                agent_race=FLAGS.agent_race,
                bot_race=FLAGS.bot_race,
                difficulty=FLAGS.difficulty,
                step_mul=FLAGS.step_mul,
                game_steps_per_episode=FLAGS.game_steps_per_episode,
                screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
                visualize=FLAGS.render)
        self._env = env
        self._env_timestep = self._env.reset()
        self._actions = [0]

    def step(self, action):
        pass

    def reset(self):
        self._env_timestep = self._env.reset()
        return self.get_state()

    def get_action_space(self):
        return len(self._actions)

    def get_state_space(self):
        return self.get_state().flatten().shape[0]

    def get_state(self):
        pass
