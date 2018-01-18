
import threading

import numpy as np
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
    ACTIONS = [Keys.SPACE, Keys.ARROW_DOWN, None]

    def __init__(self):
        super().__init__()

        self._driver = webdriver.Chrome()
        self._driver.get("file:///./game/trex_container.html")

        self._canvas = self._driver.find_element_by_id(TrexGameInterface.CANVAS_ELEMENT_ID)
        self._score_label = self._driver.find_element_by_id(TrexGameInterface.SCORE_ELEMENT_ID)
        self._action_chains = ActionChains(self._driver)

        self._should_run = False
        self._state_shape = self.current_state().shape
        self._action_space_length = len(TrexGameInterface.ACTIONS)

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

            self._action_chains.send_keys(Keys.SPACE)

            max_time = 2000
            for time_t in range(max_time):

                state = self.current_state()

                # Decide action
                action = action_provider.action(state)

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
        original = self._driver.get_screenshot_as_png()
        grayscale = skimage.color.rgb2gray(original)

        resized = skimage.transform.resize(grayscale, (150, 225))
        shape = resized.shape

        return resized.reshape(shape[0], shape[1], 1)

    def is_playing(self) -> bool:
        return True

    def current_score(self) -> int:
        return int(self._score_label.text)
