import util.helpers as helpers

class SimProcEnv():
    def __init__(self, render=False):
        self._env = helpers.get_env_wrapper(render)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.num_envs = 1
    def step(self, actions):
        obs,r,done, _ = self._env.step(actions[0])
        return [obs], [r], [done], [_]
    def reset(self):
        obs = self._env.reset()
        return [obs]
    def close(self):
        self._env.save_replay()
        return self._env.close()
