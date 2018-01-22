import multiprocessing
import random
from collections import deque
from multiprocessing import Process
from threading import Thread
from typing import List
import os

import keras
import numpy as np

from actionProvider import ActionProvider
from epsilonFunctions.epsilonChangeFunction import EpsilonChangeFunction
from gameInterface import GameInterface
from markovDecisionProcess import MarkovDecisionProcess
from trainer import Trainer
from transition import Transition


def action(state: np.ndarray, model: keras.Model, epsilon: float, action_space_length: int) -> int:
    if np.random.rand() <= epsilon:
        return random.randrange(action_space_length)

    q_values = model.predict(np.array([state]))[0]
    return int(np.argmax(q_values))


class DeepQLearningTrainer(Trainer):
    def __init__(self,
                 model: keras.Model,
                 game: MarkovDecisionProcess,
                 transitions_per_episode: int,
                 epsilon_function: EpsilonChangeFunction,
                 batch_size: int = 32,
                 discount: float = 0.95,
                 replay_memory_max_size=2000,
                 game_for_preview: GameInterface = None,
                 episodes_between_previews: int = None,
                 preview_num_episodes: int = 1):

        super().__init__()

        self._model = model
        self._game = game
        self._batch_size = batch_size
        self._replay_memory = deque(maxlen=replay_memory_max_size)
        self._discount = discount
        self._epsilon_function = epsilon_function
        self._transitions_per_episode = transitions_per_episode

        self._game_for_preview = game_for_preview
        self._episodes_between_previews = episodes_between_previews
        self._preview_num_episodes = preview_num_episodes

        self._is_training = False

    def train(self, target_episodes: int):
        if self._is_training:
            raise RuntimeError("A training session is already in progress")

        self._game.reset()

        transitions_since_last_training = 0

        for episode in range(target_episodes):

            action_to_take = self._action(self._game.state)
            transition = self._game.take_action(action_to_take)
            self._replay_memory.append(transition)
            transitions_since_last_training += 1

            if transition.game_ended:
                self._game.reset()

            if transitions_since_last_training >= self._transitions_per_episode \
                    and len(self._replay_memory) > self._batch_size:

                mini_batch = random.sample(self._replay_memory, self._batch_size)
                loss = self._train(mini_batch)
                transitions_since_last_training = 0

                self._log(num_episodes=episode, epsilon=self._epsilon_function.epsilon, loss=loss)

            if self._episodes_between_previews is not None and episode % self._episodes_between_previews == 0:
                self.preview()

    def _train(self, transitions_batch: List[Transition]) -> float:
        """
        Update the Q-Values from the given batch of transitions
        :param transitions_batch: List of transitions
        """

        batch_updated_Q_values = []
        batch_previous_states = []

        for transition in transitions_batch:
            predicted_Q_values = self._model.predict(np.array([transition.previous_state]))[0]

            # Update rule
            if transition.game_ended:
                updated_Q_value_prediction = transition.reward

            else:
                predicted_next_actions_Q_values = self._model.predict(np.array([transition.new_state]))[0]
                max_next_action_Q_Value = max(predicted_next_actions_Q_values)

                updated_Q_value_prediction = (transition.reward + self._discount * max_next_action_Q_Value)

            predicted_Q_values[transition.action] = updated_Q_value_prediction

            batch_previous_states.append(transition.previous_state)
            batch_updated_Q_values.append(predicted_Q_values)

        self._epsilon_function.step()

        loss = self._model.train_on_batch(x=np.array(batch_previous_states), y=np.array(batch_updated_Q_values))
        return loss

    def _action(self, state) -> int:
        return action(state, self._model, self._epsilon_function.epsilon, self._game.action_space_length)

    def preview(self):
        action_provider = Epsilon0ActionProvider(self._model, self._game.action_space_length)
        self._game_for_preview.display(action_provider, num_episodes=self._preview_num_episodes)


class Epsilon0ActionProvider(ActionProvider):
    def __init__(self, model: keras.Model, action_space_length):
        self._model = model
        self._action_space_length = action_space_length

    def action(self, state: np.ndarray) -> int:
        return action(state=state, model=self._model, epsilon=0, action_space_length=self._action_space_length)
