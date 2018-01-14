
import threading
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from actionProvider import ActionProvider
from gameInterface import GameInterface
from transition import Transition


class CartPoleGameInterface(GameInterface):

    def __init__(self):
        super().__init__()

        self.driver = webdriver.Chrome()
        self.driver.get("file:///Users/diego.montoya/Downloads/trex/trex_container.html")

        elem = driver.find_element_by_name("q")
        elem.clear()
        elem.send_keys("pycon")
        elem.send_keys(Keys.RETURN)

        driver.close()

        self._should_run = False
        self._feature_vector_length = self.env.observation_space.shape[0]
        self._action_space_length = self.env.action_space.n

    @property
    def action_space_length(self) -> int:
        return self._action_space_length

    @property
    def feature_vector_length(self) -> int:
        return self._feature_vector_length

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):

        self._should_run = True

        thread = threading.Thread(target=self._run, args=(action_provider, display, num_episodes))
        thread.run()

    def _run(self, action_provider: ActionProvider, display: bool, num_episodes=None):
        while self._should_run:
            state = self.env.reset()

            max_time = 2000
            for time_t in range(max_time):
                if display:
                    self.env.render()

                # Decide action
                action = action_provider.action(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, is_final, _ = self.env.step(action)

                transition = Transition(state, action, reward, next_state, is_final)
                self.delegate.game_did_receive_update(self, transition)

                # make next_state the new current state for the next frame.
                state = next_state

                if is_final:
                    # print the score and break out of the loop
                    print("Game finished with score: {}".format(time_t))
                    break

    def stop(self):
        self._should_run = False
