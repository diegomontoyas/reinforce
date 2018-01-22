import numpy as np

from transition import Transition


class MarkovDecisionProcess:

    def __init__(self):
        self.delegate = None
        self._should_run = False

    @property
    def action_space_length(self) -> int:
        raise NotImplementedError

    @property
    def state_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def state(self) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def take_action(self, action: int) -> Transition:
        raise NotImplementedError
