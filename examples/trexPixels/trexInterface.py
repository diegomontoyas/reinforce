import signal
import time

import base64
import io
import numpy as np
import os
import psutil
import skimage.color
import skimage.exposure
import skimage.transform
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from source.gameInterface import GameInterface
from source.markovDecisionProcess.actionProvider import ActionProvider
from source.markovDecisionProcess.markovDecisionProcess import MarkovDecisionProcess
from source.markovDecisionProcess.transition import Transition


class TrexGameInterface(GameInterface, MarkovDecisionProcess):

    CANVAS_ELEMENT_ID = "runner-canvas"

    # The available actions. The index defines the action
    # and the value corresponds to the associated keyboard key
    ACTION_KEYS = [None, Keys.SPACE]

    # The duration of a single transition, in seconds
    TRANSITION_DURATION = 0

    # The percentage of the width of the screen that is contemplated by the state
    WIDTH_PERCENT_FIELD_OF_VISION = 0.4

    def __init__(self):
        super().__init__()

        self._last_transition = None

        self._is_browser_suspended = False

        self._driver = webdriver.Firefox(executable_path=os.path.abspath("../../venv/geckodriver"))
        self._driver.get("file:///" + os.path.abspath("./game/trex_container.html"))
        self._browser_process = psutil.Process(self._driver.service.process.pid).children()[0]

        self._canvas = self._driver.find_element_by_id(TrexGameInterface.CANVAS_ELEMENT_ID)

        self._state_shape = self.state().shape
        self._action_space_length = len(TrexGameInterface.ACTION_KEYS)

        self._total_actions_taken = 0
        self._avg_seconds_between_actions = 0
        self._last_action_time = None

    @property
    def num_actions(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    def state(self):
        # current_frame = self.current_state_frame()
        # last_frame = self._last_frame if self._last_frame is not None else current_frame
        #
        # mixed_frame = (current_frame + last_frame/2)/2
        return self._current_state_frame()

    def _current_state_frame(self) -> np.ndarray:

        was_browser_suspended = self._is_browser_suspended

        self._resume_browser()
        image = self._canvas.screenshot_as_base64

        if was_browser_suspended:
            self._suspend_browser()

        def string_to_rgb(base64_string) -> np.ndarray:
            imgdata = base64.b64decode(base64_string)
            pil_image = Image.open(io.BytesIO(imgdata))
            return np.array(pil_image)

        image = string_to_rgb(image)  # Covert base 64 to RGB
        image = skimage.color.rgb2gray(image)  # Convert to grayscale
        image = skimage.transform.resize(image, output_shape=(30, 50))  # Downscale by a factor of 6

        slice_end = int(image.shape[1]*TrexGameInterface.WIDTH_PERCENT_FIELD_OF_VISION)
        image = image[:, :slice_end]

        return image.reshape(image.shape[0], image.shape[1], 1)

    def _is_game_over(self) -> bool:
        return not self._evaluate_js("window.runner.playing")

    def _current_score(self) -> int:
        return self._evaluate_js("window.runner.distanceMeter.currentDistance")

    def _evaluate_js(self, expression: str):
        return self._driver.execute_script("return {};".format(expression))

    def _send_key(self, key: Keys):
        if key is not None:
            self._driver.switch_to.active_element.send_keys(key)

    def _restart_game(self):
        if not self._evaluate_js("window.runner.activated"):
            self._send_key(Keys.SPACE)
            time.sleep(0.8)

        self._driver.execute_script("window.runner.restart()")

    def _suspend_browser(self):
        # self._driver.execute_script("window.runner.stop()")
        self._browser_process.send_signal(signal.SIGSTOP)
        self._is_browser_suspended = True

    def _resume_browser(self):
        # self._driver.execute_script("window.runner.play()")
        self._browser_process.send_signal(signal.SIGCONT)
        self._is_browser_suspended = False

    def take_action(self, action: int) -> Transition:

        self._resume_browser()

        while self._is_game_over():

            if self._last_transition is not None and not self._last_transition.game_ended:
                self._last_transition.game_ended = True
                self._last_transition.new_state = self.state()
                self._last_transition.reward = -1

                self._suspend_browser()
                # print("Reward: {}. Lost: {}".format(self._last_transition.reward, True))
                return self._last_transition

            self._restart_game()

        previous_state = self.state()

        self._send_key(TrexGameInterface.ACTION_KEYS[action])
        # print("Choosing: {}".format(action))
        time.sleep(TrexGameInterface.TRANSITION_DURATION)
        is_final = self._is_game_over()
        next_state = self.state()
        reward = -1 if is_final else 1

        self._suspend_browser()
        # print("Reward: {}. Lost: {}".format(reward, is_final))

        transition = Transition(previous_state, action, reward, next_state, is_final)
        self._last_transition = transition
        return transition

    def reset(self):
        pass

    def play_episode(self, action_provider: ActionProvider, display: bool=True) -> float:

        self.reset()
        self._resume_browser()

        while self._is_game_over():
            self._restart_game()

        while not self._is_game_over():
            action = action_provider.action(self.state())
            self.take_action(action)
            self._resume_browser()

        score = self._current_score()
        print("Game finished with score: {}".format(score))
        return score
