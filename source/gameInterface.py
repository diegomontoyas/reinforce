from source.markovDecisionProcess.actionProvider import ActionProvider


class GameInterface:
    @property
    def num_actions(self) -> int:
        raise NotImplementedError

    def play_episode(self, action_provider: ActionProvider, display: bool=True) -> [float]:
        raise NotImplementedError
