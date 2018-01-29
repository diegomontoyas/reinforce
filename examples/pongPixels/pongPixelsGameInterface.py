from PIL import Image
from collections import deque

import gym
import numpy as np

import skimage.color
import skimage.exposure
import skimage.transform

from source.gameInterface import GameInterface
from source.markovDecisionProcess.actionProvider import ActionProvider
from source.markovDecisionProcess.markovDecisionProcess import MarkovDecisionProcess
from source.markovDecisionProcess.transition import Transition


class PongPixelsGameInterface(GameInterface, MarkovDecisionProcess):

    def __init__(self, state_frames=2):
        super().__init__()

        self._env = gym.make("Pong-v0")
        self._frame_buffer = deque(maxlen=state_frames)

        self.reset()
        self._state_shape = self.state().shape
        self._action_space_length = self._env.action_space.n

    @property
    def num_actions(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    def state(self):
        overlapped_frame = self._frame_buffer[-1]

        # Combine all the frames in the buffer, each with half the intensity
        # of the other
        for i, frame in enumerate(reversed(list(self._frame_buffer)[:-1])):
            overlapped_frame = np.maximum(overlapped_frame, frame/(i+1))

        return overlapped_frame.flatten()

    def _process_frame(self, image) -> np.ndarray:
        image = skimage.color.rgb2gray(image)  # Convert to grayscale
        image = image[34:193]  # Remove score and black bars
        image[image-0.434 < 0.001] = 0
        image[image != 0] = 1

        # Seriously downscale that thing, life's too short
        image = skimage.transform.resize(image, output_shape=(55, 55))
        return image.reshape(image.shape[0], image.shape[1], 1)

    def reset(self):
        self._frame_buffer = deque([self._process_frame(self._env.reset())])

    def take_action(self, action: int) -> Transition:

        previous_state = self.state()
        next_frame, reward, is_final, _ = self._env.step(action)
        self._frame_buffer.append(self._process_frame(next_frame))
        next_state = self.state()

        if is_final:
            self.reset()

        transition = Transition(previous_state, action, reward, next_state, is_final)
        return transition

    def play_episode(self, action_provider: ActionProvider, display: bool=True) -> float:
        self.reset()

        score = 0
        episode_ended = False
        while not episode_ended:
            self._env.render()

            action = action_provider.action(self.state())
            transition = self.take_action(action)
            episode_ended = transition.game_ended
            score += transition.reward

        print("Game finished with score: {}".format(score))
        return score
