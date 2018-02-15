from pysc2.env import sc2_env
from absl import flags
import gym
import gym.spaces as spaces
import util.helpers as helpers
FLAGS = flags.FLAGS


class BaseEnv(gym.Env):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __init__(self, render):
        env = sc2_env.SC2Env(
                map_name=FLAGS.map,
                agent_race=FLAGS.agent_race,
                bot_race=FLAGS.bot_race,
                difficulty=FLAGS.difficulty,
                step_mul=FLAGS.step_mul,
                game_steps_per_episode=FLAGS.game_steps_per_episode,
                screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
                visualize=render)
        self.do_render = render
        self.resets = 0
        self._env = env
        self._env_timestep = self._env.reset()
        self.history = [self._env_timestep] * FLAGS.history_size
        self._actions = [0]
        self._input_layers = [0]


    def _step(self, action):
        self._env_timestep = self._env.step([self.get_sc2_action(action)])
        self.history.append(self._env_timestep)
        self.history = self.history[1:]
        r = self._env_timestep[0].reward
        s_ = self.get_state()

        return s_, r, self._env_timestep[0].last(), {}

    def _reset(self):
        self.resets += 1

        if self.resets % 8000 == 0:
            self.rebuild()

        self._env_timestep = self._env.reset()
        self.history = [self._env_timestep] * FLAGS.history_size
        return self.get_state()

    @property
    def action_space(self):
        return spaces.Discrete(len(self._actions))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=self.get_state().shape)

    def get_state(self):
        return helpers.get_input_layers(self._input_layers, self.history)

    def rebuild(self):
        self._env = sc2_env.SC2Env(
                map_name=FLAGS.map,
                agent_race=FLAGS.agent_race,
                bot_race=FLAGS.bot_race,
                difficulty=FLAGS.difficulty,
                step_mul=FLAGS.step_mul,
                game_steps_per_episode=FLAGS.game_steps_per_episode,
                screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
                visualize=FLAGS.render)
        self._env_timestep = self._env.reset()
        return self.get_state()

    def render(self, close=True):
        pass

    def get_sc2_action(self, action):
        pass