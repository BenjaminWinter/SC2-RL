from pysc2.env import sc2_env
import gflags as flags
import gym
import gym.spaces as spaces

FLAGS = flags.FLAGS


class BaseEnv(gym.Env):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __init__(self, render):
        env = sc2_env.SC2Env(
                FLAGS.map,
                agent_race=FLAGS.agent_race,
                bot_race=FLAGS.bot_race,
                difficulty=FLAGS.difficulty,
                step_mul=FLAGS.step_mul,
                game_steps_per_episode=FLAGS.game_steps_per_episode,
                screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
                visualize=render)
        self.do_render = render
        self._env = env
        self._env_timestep = self._env.reset()
        self._actions = [0]
        self.resets = 0

    def _step(self, action):
        pass

    def _reset(self):
        self.resets += 1
        if self.resets % 8000 == 0:
            self._env = sc2_env.SC2Env(
                FLAGS.map,
                agent_race=FLAGS.agent_race,
                bot_race=FLAGS.bot_race,
                difficulty=FLAGS.difficulty,
                step_mul=FLAGS.step_mul,
                game_steps_per_episode=FLAGS.game_steps_per_episode,
                screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
                minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
                visualize=self.do_render)
        self._env_timestep = self._env.reset()
        return self.get_state()

    @property
    def action_space(self):
        return spaces.Discrete(len(self._actions))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=self.get_state().shape)

    def get_state(self):
        pass

    def rebuild(self):
        self._env = sc2_env.SC2Env(
                FLAGS.map,
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