import time
from collections import deque
from typing import List

import keras
import numpy as np
import os
import random

from source.epsilonChangeFunctions.epsilonChangeFunction import EpsilonChangeFunction
from source.gameInterface import GameInterface
from source.markovDecisionProcess.markovDecisionProcess import MarkovDecisionProcess
from source.markovDecisionProcess.transition import Transition
from source.trainers.trainer import Trainer
from source.utils.epsilonGreedyFunction import e_greedy_action
from source.utils.learningPreviewHelper import LearningPreviewHelper
from source.utils.tensorboardLogger import TensorboardLogger


class DeepQLearningTrainer(Trainer):
    """
    A trainer which uses Deep Q-Learning as algorithm.
    """

    ANALYTICS_DIR = "./analytics"
    CHECKPOINTS_DIR = "./checkpoints"

    def __init__(self,
                 model: keras.Model,
                 game: MarkovDecisionProcess,
                 epsilon_function: EpsilonChangeFunction,
                 transitions_per_episode: int = 1,
                 batch_size: int = 32,
                 discount: float = 0.95,
                 min_transitions_until_training: int = None,
                 replay_memory_max_size=2000,
                 game_for_preview: GameInterface = None,
                 episodes_between_previews: int = None,
                 preview_num_episodes: int = 1,
                 log_analytics: bool = True,
                 logging_dir: str = ANALYTICS_DIR,
                 episodes_between_checkpoints: int = None,
                 checkpoints_dir: str = CHECKPOINTS_DIR,
                 ):

        """
        :param model: Keras model to use as the Q-Function approximator
        :param game: A `MarkovDecisionProcess` to train with
        :param epsilon_function: Epsilon function that will control how epsilon varies during training
        :param transitions_per_episode: A training episode will be done every `transitions_per_episode` transitions
        :param batch_size: Number of transitions used in each training episode
        :param discount: The Q-Learning discount factor
        :param min_transitions_until_training: Minimum number of transitions that have to be sampled before beginning
            training. In other words, the minimum size of the replay memory before training.
        :param replay_memory_max_size: The maximum size of the replay memory.
        :param game_for_preview: A `GameInterface` to preview games with. If provided the trainer will be able to
             play a game using epsilon=0 and calculate the final score every certain number of episodes.
        :param episodes_between_previews: The number of training episodes between each epsilon 0 preview session.
        :param preview_num_episodes: The number of games played in each epsilon 0 preview. An average of the scores
            is logged in this case.
        :param log_analytics: If true, training analytics will be logged using Tensorboard
        :param logging_dir: The directory for analytics logging
        :param episodes_between_checkpoints: The number of training episodes before saving a checkpoint of the model.
        :param checkpoints_dir: The directory in which model checkpoints are saved.
        """

        super().__init__()

        self._session_id = time.strftime("%a %b %d, %Y, %I:%M:%S %p")

        if log_analytics:
            path = logging_dir + "/{}".format(self._session_id)
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

        self._episodes_between_checkpoints = episodes_between_checkpoints
        self._checkpoints_dir = checkpoints_dir

        if min_transitions_until_training is None:
            self._min_transitions_until_training = self._batch_size
        else:
            self._min_transitions_until_training = min_transitions_until_training

        self._is_training = False
        self._current_episode = 0
        self._target_episodes = 0

        self._preview_helper = LearningPreviewHelper(self._game_for_preview, self._model)

        model.summary()

    def _prepare_for_training(self):
        self._current_episode = 0
        self._game.reset()
        self._is_training = True

    def train(self, target_episodes: int):
        if self._is_training:
            raise RuntimeError("A training session is already in progress")

        self._prepare_for_training()
        self._target_episodes = target_episodes

        current_transition = 0
        transitions_since_last_training = 0

        while self._current_episode < target_episodes:

            action_to_take = self._action(self._game.state())
            transition = self._game.take_action(action_to_take)
            self._replay_memory.append(transition)

            transitions_since_last_training += 1

            if self._logger is not None:
                self._logger.log_transition_data(transition=current_transition,
                                                 training_episode=self._current_episode,
                                                 reward=transition.reward)

            current_transition += 1

            if transition.game_ended:
                self._game.reset()

            if transitions_since_last_training >= self._transitions_per_episode \
                    and len(self._replay_memory) >= self._min_transitions_until_training:

                mini_batch = random.sample(self._replay_memory, self._batch_size)
                loss = self._train(mini_batch)

                if self._logger is not None:
                    self._logger.log_training_episode_data(episode=self._current_episode,
                                                           loss=loss,
                                                           epsilon=self._epsilon_function.epsilon)

                self._print(num_episodes=self._current_episode,
                            epsilon=self._epsilon_function.epsilon,
                            loss=loss)

                self._preview_if_needed()
                self._save_model_if_needed()

                transitions_since_last_training = 0
                self._current_episode += 1

        self._finish_training()

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

    def _finish_training(self):
        self._is_training = False

    def _action(self, state) -> int:
        return e_greedy_action(state, self._model, self._epsilon_function.epsilon, self._game.num_actions)

    def _preview_if_needed(self):
        if self._episodes_between_previews is None or self._current_episode % self._episodes_between_previews != 0:
            return

        scores = self._preview_helper.play(episodes=self._preview_num_episodes, display=True)

        if self._logger is not None:
            self._logger.log_epsilon_0_game_summary(training_episode=self._current_episode,
                                                    final_score=np.array(scores).mean())

    def _save_model_if_needed(self):
        if self._episodes_between_checkpoints is None \
                or self._current_episode % self._episodes_between_checkpoints != 0:
            return

        if not os.path.exists(self._checkpoints_dir):
            os.makedirs(self._checkpoints_dir)

        self._model.save(filepath=self._checkpoints_dir + "/" + self._session_id + ".kerasmodel", overwrite=True)