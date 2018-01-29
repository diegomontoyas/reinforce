from source.epsilonUpdater.epsilonUpdater import EpsilonUpdater


class ConstMultiplierEpsilonUpdater(EpsilonUpdater):

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
