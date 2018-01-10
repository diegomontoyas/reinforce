import numpy as np


class ActionProvider:

    def action(self, state: np.ndarray, epsilon: float = None):
        raise NotImplementedError
