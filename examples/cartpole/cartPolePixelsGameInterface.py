
import threading

import gym
import numpy as np

import utils
from actionProvider import ActionProvider
from gameInterface import GameInterface
from transition import Transition


class CartPolePixelsGameInterface(GameInterface):

    def __init__(self):
        super().__init__()

        self.env = gym.make("CartPole-v1")
        self.env.reset()

        self._should_run = False

        self._state_shape = self.current_state().shape
        self._action_space_length = self.env.action_space.n

    def action_space_length(self) -> int:
        return self._action_space_length

    def state_shape(self) -> tuple:
        return self._state_shape

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):

        self._should_run = True
        thread = threading.Thread(target=self._run, args=(action_provider, display, num_episodes))
        thread.run()

    def _run(self, action_provider: ActionProvider, display: bool, num_episodes=None):

        while self._should_run:
            self.env.reset()
            state = self.current_state()

            max_time = 2000
            for time_t in range(max_time):

                if display:
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
                self.delegate.game_did_receive_update(self, transition)

                # make next_state the new current state for the next frame.
                state = next_state

                if is_final:
                    # print the score and break out of the loop
                    print("Game finished with score: {}".format(time_t + reward))
                    break

    def current_state(self) -> np.ndarray:
        state = self.env.render(mode="rgb_array")
        shape = state.shape
        return self.to_grayscale(state).reshape(shape[0], shape[1], 1)

    def to_grayscale(self, rgb_matrix: np.ndarray):
        return np.dot(rgb_matrix[..., :3], [.3, .6, .1])

    def stop(self):
        self._should_run = False
