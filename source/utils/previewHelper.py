import keras
import os
import sys
import importlib
import ntpath

from examples.trexPixels.trexInterface import TrexGameInterface
from source.gameInterface import GameInterface
from source.markovDecisionProcess.actionProvider import ActionProvider, Epsilon0ActionProvider
from source.utils.utils import get_options


class PreviewHelper:

    def __init__(self, game: GameInterface, model: keras.Model):
        self._game = game
        self._model = model
        self._action_provider = Epsilon0ActionProvider(self._model, self._game.num_actions)

    def play(self, episodes: int, display: bool) -> [float]:
        scores = []
        for episode in range(episodes):
            scores.append(self._game.play_episode(self._action_provider))
        return scores
