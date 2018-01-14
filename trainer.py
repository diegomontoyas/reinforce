import numpy as np

from actionProvider import ActionProvider
from gameInterface import GameInterface
from transition import Transition


class Trainer(ActionProvider):

    def __init__(self):
        self.delegate = None

    def _process_transition(self, transition: Transition):
        raise NotImplementedError

    def train(self, num_episodes: int, game_for_preview: GameInterface,
              episodes_between_previews: int = None, preview_num_episodes: int = 1):

        raise NotImplementedError

    def _log(self, num_episodes: int, epsilon: float, loss: float):
        print("Training episode: {} | Epsilon: {} | Loss: {}".format(num_episodes, epsilon, loss))
