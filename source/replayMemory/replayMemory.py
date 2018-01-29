import numpy as np


class ReplayMemory:

    def remember(self, transition):
        raise NotImplementedError

    def sample(self, size: int) -> list:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
