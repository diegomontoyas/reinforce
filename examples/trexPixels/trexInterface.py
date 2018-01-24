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

    # The available actions. The index defines the action
    # and the value corresponds to the associated keyboard key
    ACTION_KEYS = [None, Keys.SPACE]

    # The duration of a single transition, in seconds
    TRANSITION_DURATION = 0.6

    def __init__(self):
        super().__init__()

        self._last_frame = None

        self._driver = webdriver.Firefox(executable_path=os.path.abspath("../../venv/geckodriver"))
        self._driver.get("file:///" + os.path.abspath("./game/trex_container.html"))

        self._canvas = self._driver.find_element_by_id(TrexGameInterface.CANVAS_ELEMENT_ID)

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
        # current_frame = self.current_state_frame()
        # last_frame = self._last_frame if self._last_frame is not None else current_frame
        #
        # mixed_frame = (current_frame + last_frame/2)/2
        return self.current_state_frame()

    def current_state_frame(self) -> np.ndarray:
        image = self._canvas.screenshot_as_base64

        def string_to_rgb(base64_string) -> np.ndarray:
            imgdata = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(imgdata))
            return np.array(pil_image)

        image = string_to_rgb(image)  # Covert base 64 to RGB
        image = skimage.color.rgb2gray(image)  # Convert to grayscale
        image = skimage.transform.resize(image, output_shape=(100, 200))  # Downscale by a factor of 6
        image = image[:, :100]

        return image.astype(np.float).reshape(image.shape[0], image.shape[1], 1)

    def is_game_over(self) -> bool:
        activated = self._evaluate_js("window.runner.activated")
        crashed = self._evaluate_js("window.runner.crashed")

        return crashed or not activated

    def current_score(self) -> int:
        return self._evaluate_js("window.runner.distanceMeter.currentDistance")

    def _evaluate_js(self, expression: str):
        return self._driver.execute_script("return {};".format(expression))

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

        if self.is_game_over():
            self._restart_game()

        self._pause_game()

        previous_state = self.state()
        self._last_frame = self.current_state_frame()
        score_before = self.current_score()

        self._resume_game()

        self._send_key(TrexGameInterface.ACTION_KEYS[action])
        print("Choosing: {}".format(action))
        time.sleep(TrexGameInterface.TRANSITION_DURATION)

        is_final = self.is_game_over()
        reward = self.current_score() - score_before

        self._pause_game()

        next_state = self.state()

        if is_final:
            reward = -500

        print("Reward: {}. Lost: {}".format(reward, is_final))

        transition = Transition(previous_state, action, reward, next_state, is_final)
        return transition

    def reset(self):
        pass

    def display(self, action_provider: ActionProvider, num_episodes: int):

        n = 0
        finished_episodes = False
        self.reset()

        while not finished_episodes:

            while not self.is_game_over():

                if self.is_game_over():
                    self._restart_game()

                self._pause_game()
                action = action_provider.action(self.state())
                self._resume_game()

                transition = self.take_action(action)

                if transition.game_ended:
                    print("Game finished with score: {}".format(self.current_score()))

            n += 1

            if n == num_episodes:
                finished_episodes = True
