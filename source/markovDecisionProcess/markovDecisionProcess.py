import numpy as np

from source.markovDecisionProcess.transition import Transition


class MarkovDecisionProcess:

    def __init__(self):
        self.delegate = None
        self._should_run = False

    @property
    def num_actions(self) -> int:
        raise NotImplementedError

    @property
    def state_shape(self) -> tuple:
        raise NotImplementedError

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def take_action(self, action: int) -> Transition:
        raise NotImplementedError
