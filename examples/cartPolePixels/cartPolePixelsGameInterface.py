
import threading

import gym
import numpy as np
from PIL import Image

from actionProvider import ActionProvider
from gameInterface import GameInterface
from markovDecisionProcess import MarkovDecisionProcess
from transition import Transition

import skimage.color
import skimage.transform
import skimage.exposure


class CartPolePixelsGameInterface(GameInterface, MarkovDecisionProcess):

    def __init__(self, num_state_frames=4):
        super().__init__()

        self._env = gym.make("CartPole-v1")
        self._env.reset()

        self._frame_buffer = []
        self._num_state_frames = num_state_frames
        self._state_shape = self.state().shape
        self._action_space_length = self._env.action_space.n

    @property
    def action_space_length(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    def state(self):
        image = self._env.render(mode="rgb_array")
        image = skimage.color.rgb2gray(image)  # Convert to grayscale
        image = image[155:318]  # Crop to content
        image = skimage.transform.rescale(image, 1.0 / 4.0)  # Downscale by a factor of 6
        image[image == 1] = 0  # Make background black
        image[image != 0] = 1  # Make everything else completely white
        # Image.fromarray(np.uint8(image * 255), 'L').show()

        return image.astype(np.float).flatten()

    def reset(self):
        self._env.reset()

    def take_action(self, action: int) -> Transition:

        previous_state = self.state()

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        _, reward, is_final, _ = self._env.step(action)
        next_state = self.state()

        # We redefine the losing reward so it has more relevance in
        # the transitions and training is faster.
        if is_final:
            reward = -500

        transition = Transition(previous_state, action, reward, next_state, is_final)
        return transition

    def display_episode(self, action_provider: ActionProvider) -> float:

        self.reset()

        t = 0
        episode_ended = False
        while not episode_ended:
            self._env.render()

            # Decide action
            action = action_provider.action(self.state())
            transition = self.take_action(action)
            t += 1

            episode_ended = transition.game_ended

        score = t + transition.reward
        print("Game finished with score: {}".format(score))
        return score
