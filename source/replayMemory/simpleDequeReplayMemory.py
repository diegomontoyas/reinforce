import random
from collections import deque

import numpy as np

from source.replayMemory.replayMemory import ReplayMemory


class SimpleDequeReplayMemory(ReplayMemory):

    def __init__(self, max_size: int):
        self._deque = deque(maxlen=max_size)

    def remember(self, transition):
        self._deque.append(transition)

    def sample(self, size: int) -> np.ndarray:
        return random.sample(self._deque, size)

    def __len__(self):
        return len(self._deque)
