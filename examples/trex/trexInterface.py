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
from transition import Transition

import skimage.color
import skimage.transform


class TrexGameInterface(GameInterface):
    CANVAS_ELEMENT_ID = "runner-canvas"
    SCORE_ELEMENT_ID = "distance-additional-label"
    PLAYING_STATE_ELEMENT_ID = "state-label"
    ACTION_KEYS = [Keys.SPACE, None]

    def __init__(self):
        super().__init__()

        self._driver = webdriver.Safari()
        self._driver.get("file:///" + os.path.abspath("./game/trex_container.html"))

        self._canvas = self._driver.find_element_by_id(TrexGameInterface.CANVAS_ELEMENT_ID)
        self._score_label = self._driver.find_element_by_id(TrexGameInterface.SCORE_ELEMENT_ID)
        self._playing_state_label = self._driver.find_element_by_id(TrexGameInterface.PLAYING_STATE_ELEMENT_ID)
        self._action_chains = ActionChains(self._driver)

        self._should_run = False
        self._state_shape = self.current_state().shape
        self._action_space_length = len(TrexGameInterface.ACTION_KEYS)

    @property
    def action_space_length(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):

        self._should_run = True

        n = 0
        while self._should_run:

            self.send_action(Keys.SPACE)
            time.sleep(0.2)
            state = self.current_state()

            while not self.is_game_over():

                score_before = self.current_score()

                # Decide action
                action = action_provider.action(state)
                action_key = TrexGameInterface.ACTION_KEYS[action]

                if action_key is not None:
                    self.send_action(action_key)

                time.sleep(0.1)

                is_final = self.is_game_over()
                reward = self.current_score() - score_before
                next_state = self.current_state()

                if is_final:
                    reward = -500

                print("Chose: {}. Reward: {}. Lost: {}".format(action, reward, is_final))

                transition = Transition(state, action, reward, next_state, is_final)

                if self.delegate is not None:
                    self.delegate.game_did_receive_update(self, transition)

                # make next_state the new current state for the next frame.
                state = next_state

                if is_final:
                    if display:
                        print("Game finished with score: {}".format(self.current_score()))

                    break

            n += 1

            if num_episodes is not None and n == num_episodes:
                self._should_run = False

    def current_state(self) -> np.ndarray:
        return self.current_state_frame()

    def current_state_frame(self) -> np.ndarray:
        original = self._driver.get_screenshot_as_base64()

        def stringToImage(base64_string):
            imgdata = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(imgdata))

        rgb = np.array(stringToImage(original))
        grayscale = skimage.color.rgb2gray(rgb)

        resized = grayscale # skimage.transform.resize(grayscale, (300, 426))
        shape = resized.shape

        return resized.reshape(shape[0], shape[1], 1)

    def is_game_over(self) -> bool:
        return self._playing_state_label.text == "over"

    def current_score(self) -> int:
        return int(self._score_label.text)

    def send_action(self, key: Keys):
        self._action_chains.send_keys(key)
        self._action_chains.perform()
        self._action_chains.reset_actions()
