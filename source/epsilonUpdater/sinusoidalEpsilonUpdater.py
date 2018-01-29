from math import sin

from source.epsilonUpdater.epsilonUpdater import EpsilonUpdater


class SinusoidalEpsilonUpdater(EpsilonUpdater):

    def __init__(self, x_step_value: float):

        super().__init__(initial_value=0)
        self._x = 0
        self._x_step_value = x_step_value

    def step(self) -> float:

        epsilon = self._epsilon

        self._x += self._x_step_value
        self._epsilon = 1-abs(sin(self._x))

        return epsilon
