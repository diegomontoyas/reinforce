from typing import List
import numpy as np


class Transition:

    def __init__(self,
                 previous_state: np.ndarray,
                 action: int,
                 reward: float,
                 new_state: np.ndarray,
                 is_new_state_final: bool = None):

        self.previous_state = previous_state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.is_new_state_final = is_new_state_final
