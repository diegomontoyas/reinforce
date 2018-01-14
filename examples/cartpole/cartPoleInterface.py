
import threading

import asyncio
import gym
import time

from dispatch import Dispatch
from gameInterface import *
from transition import Transition


class CartPoleGameInterface(GameInterface):

    def __init__(self):
        super().__init__()

        self.env = gym.make("CartPole-v1")
        self._feature_vector_length = self.env.observation_space.shape[0]
        self._action_space_length = self.env.action_space.n

    @property
    def action_space_length(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._feature_vector_length,

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):

        n=0
        self._should_run = True

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

                if is_final and time_t < 500:
                    reward = -500

                transition = Transition(state, action, reward, next_state, is_final)

                if self.delegate:
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
