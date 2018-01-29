from source.epsilonUpdater.epsilonUpdater import EpsilonUpdater


class ConstantEpsilonUpdater(EpsilonUpdater):

    def __init__(self, initial_value: float):
        super().__init__(initial_value)

    def step(self) -> float:
        pass
