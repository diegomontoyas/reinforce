import keras
import numpy as np

from source.utils.epsilonGreedyFunction import e_greedy_action


class ActionProvider:

    def action(self, state: np.ndarray) -> int:
        raise NotImplementedError


class Epsilon0ActionProvider(ActionProvider):
    def __init__(self, model: keras.Model, action_space_length):
        self._model = model
        self._action_space_length = action_space_length

    def action(self, state: np.ndarray) -> int:
        return e_greedy_action(state=state,
                               model=self._model,
                               epsilon=0,
                               action_space_length=self._action_space_length)