import numpy as np

from actionProvider import ActionProvider
from gameInterface import GameInterface
from transition import Transition


class Trainer:

    def train(self, num_episodes: int):
        raise NotImplementedError

    def _print(self, num_episodes: int, epsilon: float, loss: float):
        print("Training episode: {} | Epsilon: {} | Loss: {}".format(num_episodes, epsilon, loss))
