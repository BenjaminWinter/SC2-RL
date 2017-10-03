import time

from pysc2.lib import actions
from pysc2.lib import features

import util.helpers as util
from util.environments.simple_vulture_env import SimpleVultureEnv
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_VISIBILITY = features.SCREEN_FEATURES.visibility_map.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_AVAILABLE_ACTIONS = ["ATTACK", "RETREAT", "SELECT_ARMY", "NO_OP"]


class Test:
    def __init__(self, episodes, a_space, s_space):
        self.env = SimpleVultureEnv()
        self.episodes = episodes
        self.total_frames = 0
        self.env_steps = []

    def run(self):
        start_time = time.time()
        for i in range(self.episodes):
            try:
                env_steps = self.env.reset()
                self.reset(i)
                while True:
                    if env_steps[0].last():
                        break
                    self.total_frames += 1

                    obs = env_steps[0]

                    next_action = _MOVE_SCREEN
                    if next_action == _ATTACK_SCREEN:
                        args = [[False], util.get_attack_coordinates(obs)]
                    elif next_action == _MOVE_SCREEN:
                        args = [[False], util.get_retreat_coordinates(obs)]
                    elif next_action == _SELECT_ARMY:
                        args = [[False]]
                    elif next_action == _NO_OP:
                        args = []
                    else:
                        raise ValueError("Action not recognised")

                    action = actions.FunctionCall(next_action, args)
                    env_steps = self.env.step([action])
            
            except KeyboardInterrupt:
                pass
            finally:
                elapsed_time = time.time() - start_time
                print("Took %.3f seconds for %s steps: %.3f fps" % (
                    elapsed_time, self.total_frames, self.total_frames / elapsed_time))

    def reset(self, finished_episode):
        print("Finished Episode %s" % finished_episode)
        return