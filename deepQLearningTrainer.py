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
from gameDelegate import GameDelegate
from gameInterface import GameInterface
from trainer import Trainer
from transition import Transition


def action(state: np.ndarray, model: keras.Model, epsilon: float, action_space_length: int) -> int:
    if np.random.rand() <= epsilon:
        return random.randrange(action_space_length)

    q_values = model.predict(np.array([state]))[0]
    return int(np.argmax(q_values))


class DeepQLearningTrainer(Trainer, GameDelegate, ActionProvider):
    def __init__(self,
                 model: keras.Model,
                 game: GameInterface,
                 transitions_per_episode: int,
                 epsilon_function: EpsilonChangeFunction,
                 batch_size: int = 32,
                 discount: float = 0.95,
                 replay_memory_max_size=2000):

        super().__init__()

        self._model = model
        self._game = game
        self._batch_size = batch_size
        self._replay_memory = deque(maxlen=replay_memory_max_size)
        self._discount = discount
        self._epsilon_function = epsilon_function
        self._transitions_per_episode = transitions_per_episode

        self._num_transitions_since_last_training = 0
        self._session_num_completed_episodes = 0
        self._session_target_num_episodes = 0

        self._game_for_preview = None
        self._episodes_between_previews = None
        self._preview_num_episodes = 1

        self._game.delegate = self

        self._model_updates_queue = multiprocessing.Queue(maxsize=1)

    def _process_transition(self, transition: Transition):
        self._replay_memory.append(transition)
        self._num_transitions_since_last_training += 1

        if self._num_transitions_since_last_training >= self._transitions_per_episode \
                and len(self._replay_memory) > self._batch_size:
            mini_batch = random.sample(self._replay_memory, self._batch_size)
            self._train(mini_batch)

    def _train(self, transitions_batch: List[Transition]):
        """
        Update the Q-Values from the given batch of transitions
        :param transitions_batch: List of transitions
        """

        batch_updated_Q_values = []
        batch_previous_states = []

        for transition in transitions_batch:
            predicted_Q_values = self._model.predict(np.array([transition.previous_state]))[0]

            # Update rule
            if transition.is_new_state_final:
                updated_Q_value_prediction = transition.reward

            else:
                predicted_next_actions_Q_values = self._model.predict(np.array([transition.new_state]))[0]
                max_next_action_Q_Value = max(predicted_next_actions_Q_values)

                updated_Q_value_prediction = (transition.reward + self._discount * max_next_action_Q_Value)

            predicted_Q_values[transition.action] = updated_Q_value_prediction

            batch_previous_states.append(transition.previous_state)
            batch_updated_Q_values.append(predicted_Q_values)

        self._epsilon_function.step()
        self._session_num_completed_episodes += 1
        self._num_transitions_since_last_training = 0

        loss = self._model.train_on_batch(x=np.array(batch_previous_states),
                                                y=np.array(batch_updated_Q_values))

        self._log(num_episodes=self._session_num_completed_episodes, epsilon=self.epsilon(), loss=loss)

        if self._episodes_between_previews is not None \
                and self._session_num_completed_episodes % self._episodes_between_previews == 0:
            self.preview()

        self.delegate.trainer_did_finish_training_episode(trainer=self, episode=self._session_num_completed_episodes)

        if self._session_num_completed_episodes >= self._session_target_num_episodes:
            self._finish_training()

        return loss

    def _finish_training(self):
        self._session_num_completed_episodes = 0
        self._session_target_num_episodes = 0
        self._game.stop()
        self._game_for_preview = None

        self.delegate.trainer_did_finish_training(trainer=self)

    def action(self, state, epsilon: float = None) -> int:
        """
        Decides an action for the state provided
        :param state: State for which to choose an action
        :param epsilon: (Optional) Epsilon to use. If not provided the current training epsilon is used
        :return: The action to take
        """
        if epsilon is None:
            epsilon = self.epsilon()

        return action(state, self._model, epsilon, self._game.action_space_length)

    def train(self, num_episodes: int, game_for_preview: GameInterface = None,
              episodes_between_previews: int = None, preview_num_episodes: int = 1):

        if self._session_num_completed_episodes != 0:
            return

        self._session_num_completed_episodes = 0
        self._session_target_num_episodes = num_episodes

        self._game_for_preview = game_for_preview
        self._episodes_between_previews = episodes_between_previews
        self._preview_num_episodes = preview_num_episodes

        self._game.run(action_provider=self, display=False)

    def epsilon(self) -> float:
        return self._epsilon_function.epsilon()

    def preview(self):
        action_provider = Epsilon0ActionProvider(self._model, self._game.action_space_length)
        self._game_for_preview.run(action_provider, display=True, num_episodes=self._preview_num_episodes)

    # +-------------------------------------+
    # |      GAME INTERFACE DELEGATE        |
    # +-------------------------------------+

    def game_did_receive_update(self, game: GameInterface, transition: Transition):
        self._process_transition(transition)


class Epsilon0ActionProvider(ActionProvider):
    def __init__(self, model: keras.Model, action_space_length):
        self._model = model
        self._action_space_length = action_space_length

    def action(self, state: np.ndarray) -> int:
        return action(state=state, model=self._model, epsilon=0, action_space_length=self._action_space_length)
