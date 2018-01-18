
import threading

import gym
import numpy as np

from actionProvider import ActionProvider
from gameInterface import GameInterface
from transition import Transition

import skimage.color
import skimage.transform

class CartPolePixelsGameInterface(GameInterface):

    def __init__(self, num_state_frames=4):
        super().__init__()

        self.env = gym.make("CartPole-v1")
        self.env.reset()

        self._frame_buffer = []
        self._num_state_frames = num_state_frames
        self._state_shape = self.current_state().shape
        self._action_space_length = self.env.action_space.n

    @property
    def action_space_length(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):

        self._should_run = True

        n=0
        while self._should_run:
            self.env.reset()

            max_time = 2000
            for time_t in range(max_time):

                state = self.current_state()

                # Decide action
                action = action_provider.action(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                _, reward, is_final, _ = self.env.step(action)
                next_state = self.current_state()

                if is_final and time_t < 500:
                    reward = -500

                transition = Transition(state, action, reward, next_state, is_final)

                if self.delegate is not None:
                    self.delegate.game_did_receive_update(self, transition)

                # make next_state the new current state for the next frame.
                state = next_state

                if is_final:
                    if display:
                        print("Game finished with score: {}".format(time_t + reward))

                    break

            n += 1

            if num_episodes is not None and n == num_episodes:
                self._should_run = False

    def current_state(self) -> np.ndarray:
        return self.current_state_frame()

    def current_state_frame(self) -> np.ndarray:
        original = self.env.render(mode="rgb_array")
        grayscale = skimage.color.rgb2gray(original)

        resized = skimage.transform.resize(grayscale, (80, 120))
        shape = resized.shape

        return resized.reshape(shape[0], shape[1], 1)
