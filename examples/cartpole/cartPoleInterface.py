
import threading

import gym
import numpy as np

import utils
from actionProvider import ActionProvider
from gameInterface import GameInterface
from transition import Transition


class CartPoleGameInterface(GameInterface):

    def __init__(self):
        super().__init__()

        self.env = gym.make("CartPole-v1")
        self._should_run = False
        self._feature_vector_length = self.env.observation_space.shape[0]
        self._action_space_length = self.env.action_space.n

    def action_space_length(self) -> int:
        return self._action_space_length

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
                    pixel_matrix = self.env.render(mode="rgb_array")

                # Decide action
                action = action_provider.action(state)

                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, is_final, _ = self.env.step(action)

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

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        cart_pos = utils.normalize_0_1(state[0], in_min=-2.4, in_max=2.4)
        cart_vel = utils.normalize_0_1(state[1], in_min=-10000, in_max=10000)
        pole_angle = utils.normalize_0_1(state[2], in_min=-41.8, in_max=41.8)
        pole_vel_tip = utils.normalize_0_1(state[3], in_min=-10000, in_max=10000)

        return np.array([cart_pos, cart_vel, pole_angle, pole_vel_tip])

    def stop(self):
        self._should_run = False
