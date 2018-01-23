from math import sin

from epsilonChangeFunctions.epsilonChangeFunction import EpsilonChangeFunction


class ConstantEpsilonFunction(EpsilonChangeFunction):

    def step(self) -> float:
        pass

    def __init__(self, initial_value: float):
        super().__init__(initial_value)


class ConstMultiplierEpsilonDecayFunction(EpsilonChangeFunction):

    def __init__(self, initial_value: float,
                 final_value: float = 0.01,
                 decay_multiplier: float = 0.995):

        super().__init__(initial_value=initial_value)
        self._final_epsilon = final_value
        self._decay_multiplier = decay_multiplier

    def step(self) -> float:

        epsilon = self._epsilon

        if self._epsilon > self._final_epsilon:
            self._epsilon *= self._decay_multiplier

        return epsilon


class SinusoidalEpsilonChangeFunction(EpsilonChangeFunction):

    def __init__(self, x_step_value: float):

        super().__init__(initial_value=0)
        self._x = 0
        self._x_step_value = x_step_value

    def step(self) -> float:

        epsilon = self._epsilon

        self._x += self._x_step_value
        self._epsilon = 1-abs(sin(self._x))

        return epsilon
