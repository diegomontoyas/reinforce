from actionProvider import ActionProvider


class GameInterface:

    def __init__(self):
        self.delegate = None

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def action_space_length(self) -> int:
        raise NotImplementedError

    def state_shape(self) -> tuple:
        raise NotImplementedError
