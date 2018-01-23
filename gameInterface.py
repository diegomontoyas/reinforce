from actionProvider import ActionProvider


class GameInterface:

    def display(self, action_provider: ActionProvider, num_episodes: int) -> [float]:
        raise NotImplementedError
