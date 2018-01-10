import random
import threading
from collections import deque
from typing import List

import numpy as np

from actionProvider import ActionProvider
from transition import Transition
from gameInterface import GameInterface
from gameDelegate import GameDelegate
from trainer import Trainer


class DeepQLearningTrainer(Trainer, GameDelegate, ActionProvider):
    def __init__(self,
                 model,
                 game: GameInterface,
                 transitions_per_episode: int,
                 batch_size: int = 32,
                 discount: float = 0.95,
                 initial_epsilon: float = 1,
                 final_epsilon: float = 0.01,
                 epsilon_decay_multiplier: float = 0.995,
                 replay_memory_max_size=2000):

        super().__init__()

        self._model = model
        self._game = game
        self._batch_size = batch_size
        self._replay_memory = deque(maxlen=replay_memory_max_size)
        self._discount = discount
        self._epsilon = initial_epsilon
        self._final_epsilon = final_epsilon
        self._epsilon_decay_multiplier = epsilon_decay_multiplier
        self._transitions_per_episode = transitions_per_episode

        self._num_transitions_since_last_training = 0
        self._session_num_completed_episodes = 0
        self._session_target_num_episodes = 0

        self._training_thread = None
        self._game.delegate = self

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
            predicted_Q_values = self._model.model.predict(np.array([transition.previous_state]))[0]

            # Update rule
            if transition.is_new_state_final:
                updated_Q_value_prediction = transition.reward

            else:
                predicted_next_actions_Q_values = self._model.model.predict(np.array([transition.new_state]))[0]
                max_next_action_Q_Value = max(predicted_next_actions_Q_values)

                updated_Q_value_prediction = (transition.reward + self._discount * max_next_action_Q_Value)

            predicted_Q_values[transition.action] = updated_Q_value_prediction

            batch_previous_states.append(transition.previous_state)
            batch_updated_Q_values.append(predicted_Q_values)

        self._update_epsilon()
        self._session_num_completed_episodes += 1
        self._num_transitions_since_last_training = 0

        loss = self._model.model.train_on_batch(x=np.array(batch_previous_states),
                                                  y=np.array(batch_updated_Q_values))

        self._log(num_episodes=self._session_num_completed_episodes, epsilon=self._epsilon, loss=loss)

        self.delegate.trainer_did_finish_training_episode(trainer=self, episode=self._session_num_completed_episodes)

        if self._session_num_completed_episodes >= self._session_target_num_episodes:
            self._finish_training()

        return loss

    def _update_epsilon(self):
        """
        Decreases the epsilon value by `epsilon_decay_multiplier`
        """

        if self._epsilon > self._final_epsilon:
            self._epsilon *= self._epsilon_decay_multiplier

    def _finish_training(self):
        self._session_num_completed_episodes = 0
        self._session_target_num_episodes = 0
        self._game.stop()
        self.delegate.trainer_did_finish_training(trainer=self)

    def action(self, state, epsilon: float = None):
        """
        Decides an action for the state provided
        :param state: State for which to choose an action
        :param epsilon: (Optional) Epsilon to use. If not provided the current training epsilon is used
        :return: The action to take
        """
        epsilon = epsilon or self._epsilon

        if np.random.rand() <= epsilon:
            return random.randrange(self._game.action_space_length())

        q_values = self._model.predict(np.array([state]))[0]
        return int(np.argmax(q_values))

    def train(self, num_episodes: int, display: bool):
        if self._session_num_completed_episodes != 0:
            return

        self._session_num_completed_episodes = 0
        self._session_target_num_episodes = num_episodes
        self._game.run(action_provider=self, display=display)

    # +-------------------------------------+
    # |      GAME INTERFACE OBSERVER        |
    # +-------------------------------------+

    def game_did_receive_update(self, game: GameInterface, transition: Transition):
        self._training_thread = threading.Thread(target=self._process_transition, args=[transition])
        self._training_thread.run()
