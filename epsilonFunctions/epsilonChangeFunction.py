
class EpsilonChangeFunction:

    def __init__(self, initial_value: float):
        self._epsilon = initial_value

    def step(self) -> float:
        raise NotImplementedError

    def epsilon(self) -> float:
        return self._epsilon
