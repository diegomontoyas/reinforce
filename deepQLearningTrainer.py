import time
from typing import List

import keras
import numpy as np
import random
from collections import deque

from actionProvider import ActionProvider, Epsilon0ActionProvider
from epsilonChangeFunctions.epsilonChangeFunction import EpsilonChangeFunction
from epsilonGreedyFunction import e_greedy_action
from gameInterface import GameInterface
from markovDecisionProcess import MarkovDecisionProcess
from tensorboardLogger import TensorboardLogger
from trainer import Trainer
from transition import Transition


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
                 preview_num_episodes: int = 1,
                 log_analytics: bool = True,
                 logging_dir: str = "./analytics"):

        super().__init__()

        if log_analytics:
            path = logging_dir + "/{}".format(time.strftime("%a %b %d, %Y, %I:%M:%S %p"))
            self._logger = TensorboardLogger(log_dir=path)

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

        model.summary()

    def train(self, target_episodes: int):
        if self._is_training:
            raise RuntimeError("A training session is already in progress")

        self._game.reset()

        current_transition = 0
        transitions_since_last_training = 0

        current_episode = 0
        while current_episode < target_episodes:

            action_to_take = self._action(self._game.state())
            transition = self._game.take_action(action_to_take)
            self._replay_memory.append(transition)

            transitions_since_last_training += 1

            if self._logger is not None:
                self._logger.log_transition_data(transition=current_transition,
                                                 training_episode=current_episode,
                                                 reward=transition.reward)

            current_transition += 1

            if transition.game_ended:
                self._game.reset()

            if transitions_since_last_training >= self._transitions_per_episode \
                    and len(self._replay_memory) > self._batch_size:

                mini_batch = random.sample(self._replay_memory, self._batch_size)
                loss = self._train(mini_batch)

                if self._logger is not None:
                    self._logger.log_training_episode_data(episode=current_episode,
                                                           loss=loss,
                                                           epsilon=self._epsilon_function.epsilon)

                self._print(num_episodes=current_episode,
                            epsilon=self._epsilon_function.epsilon,
                            loss=loss)

                if self._episodes_between_previews is not None \
                        and current_episode % self._episodes_between_previews == 0:

                    self.preview(episode=current_episode)

                transitions_since_last_training = 0
                current_episode += 1

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
        return e_greedy_action(state, self._model, self._epsilon_function.epsilon, self._game.action_space_length)

    def preview(self, episode: int):
        action_provider = Epsilon0ActionProvider(self._model, self._game.action_space_length)
        scores = self._game_for_preview.display(action_provider, num_episodes=self._preview_num_episodes)

        if scores is not None and self._logger is not None:
            self._logger.log_epsilon_0_game_summary(training_episode=episode, final_score=np.array(scores).mean())
