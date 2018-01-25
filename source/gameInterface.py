from source.markovDecisionProcess.actionProvider import ActionProvider


class GameInterface:

    def display_episode(self, action_provider: ActionProvider) -> [float]:
        raise NotImplementedError
