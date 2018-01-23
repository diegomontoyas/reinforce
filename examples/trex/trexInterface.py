import base64
import io
import threading

import numpy as np
import time

import os

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from actionProvider import ActionProvider
from gameInterface import GameInterface
from markovDecisionProcess import MarkovDecisionProcess
from transition import Transition

import skimage.color
import skimage.transform
import skimage.exposure


class TrexGameInterface(GameInterface, MarkovDecisionProcess):

    CANVAS_ELEMENT_ID = "runner-canvas"
    SCORE_ELEMENT_ID = "distance-additional-label"
    PLAYING_STATE_ELEMENT_ID = "state-label"

    # The available actions. The index defines the action
    # and the value corresponds to the associated keyboard key
    ACTION_KEYS = [None, Keys.SPACE, Keys.ARROW_DOWN]

    # The duration of a single transition, in seconds
    TRANSITION_DURATION = 0.1

    def __init__(self):
        super().__init__()

        self._driver = webdriver.Firefox(executable_path=os.path.abspath("../../venv/geckodriver"))
        self._driver.get("file:///" + os.path.abspath("./game/trex_container.html"))

        self._canvas = self._driver.find_element_by_id(TrexGameInterface.CANVAS_ELEMENT_ID)
        self._score_label = self._driver.find_element_by_id(TrexGameInterface.SCORE_ELEMENT_ID)
        self._playing_state_label = self._driver.find_element_by_id(TrexGameInterface.PLAYING_STATE_ELEMENT_ID)

        self._state_shape = self.state().shape
        self._action_space_length = len(TrexGameInterface.ACTION_KEYS)

        self._total_actions_taken = 0
        self._avg_seconds_between_actions = 0
        self._last_action_time = None

    @property
    def action_space_length(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    def state(self):
        return self.current_state_frame()

    def current_state_frame(self) -> np.ndarray:
        image = self._canvas.screenshot_as_base64

        def string_to_rgb(base64_string) -> np.ndarray:
            imgdata = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(imgdata))
            return np.array(pil_image)

        image = string_to_rgb(image)  # RGB
        image = skimage.color.rgb2gray(image)  # Grayscale
        image = image # skimage.transform.resize(grayscale, (300, 426))  # Resized
        image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))

        return image.reshape(image.shape[0], image.shape[1], 1)

    def is_game_over(self) -> bool:
        return self._playing_state_label.text == "over"

    def current_score(self) -> int:
        return int(self._score_label.text)

    def _send_key(self, key: Keys):
        if key is not None:
            self._canvas.send_keys(key)

    def _restart_game(self):
        self._driver.execute_script("window.runner.restart()")

    def _pause_game(self):
        self._driver.execute_script("window.runner.stop()")

    def _resume_game(self):
        self._driver.execute_script("window.runner.play()")

    def take_action(self, action: int) -> Transition:
        self._resume_game()

        if self.is_game_over():
            self._restart_game()

        score_before = self.current_score()
        self._send_key(TrexGameInterface.ACTION_KEYS[action])
        print("Choosing: {}".format(action))
        time.sleep(TrexGameInterface.TRANSITION_DURATION)

        is_final = self.is_game_over()
        reward = self.current_score() - score_before
        next_state = self.state()

        if is_final:
            reward = -500

        print("Reward: {}. Lost: {}".format(reward, is_final))

        transition = Transition(self.state(), action, reward, next_state, is_final)
        self._pause_game()

        return transition

    def reset(self):
        pass

    def display(self, action_provider: ActionProvider, num_episodes: int = None):

        n = 0
        finished_episodes = False
        self.reset()

        while not finished_episodes:

            while not self.is_game_over():

                action = action_provider.action(self.state())
                transition = self.take_action(action)

                if transition.game_ended:
                    print("Game finished with score: {}".format(self.current_score()))

            n += 1

            if n == num_episodes:
                finished_episodes = True
