import numpy as np


class ActionProvider:

    def action(self, state: np.ndarray) -> int:
        raise NotImplementedError
