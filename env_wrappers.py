from stable_baselines3.common.atari_wrappers import AtariWrapper
import numpy as np
from collections import deque
import gymnasium as gym

class FrameStacking_Atari:
    # just for adding frame stacking
    # frameskip & max pooling of frames is done by AtariWrapper
    def __init__(self, env, obs_shape = (84, 84)) -> None:

        self.obs_shape = obs_shape
        self.framestack = deque(
            [np.zeros(obs_shape, dtype=np.float32) for _ in range(4)], maxlen=4
        )
        self.env = env

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        next_state_normalized = next_state.reshape(self.obs_shape) / 255

        self.framestack.append(next_state_normalized)

        return self._state, reward, done, truncated, info

    def reset(self):
        state, info = self.env.reset()
        state_normalized = state.reshape(self.obs_shape) / 255

        self.framestack.append(state_normalized)

        return self._state, info

    @property
    def _state(self):
        return np.stack(tuple(self.framestack), dtype=np.float32)


def make_pong_env(render: bool = False):

    if render:
        pong_raw = gym.make("PongNoFrameskip-v4", render_mode="human")
    else:
        pong_raw = gym.make("PongNoFrameskip-v4")

    pong_frameskip = AtariWrapper(
        env=pong_raw,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        clip_reward=True,
        action_repeat_probability=0.0,
    )

    pong_framestacked = FrameStacking_Atari(pong_frameskip)

    return pong_framestacked
