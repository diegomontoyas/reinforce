import numpy as np

from actionProvider import ActionProvider
from transition import Transition


class Trainer(ActionProvider):

    def __init__(self):
        self.delegate = None

    def _process_transition(self, transition: Transition):
        raise NotImplementedError

    def train(self, num_episodes: int, display: bool):
        raise NotImplementedError

    def _log(self, num_episodes: int, epsilon: float, loss: float):
        print("Training episode: {} | Epsilon: {} | Loss: {}".format(num_episodes, epsilon, loss))
