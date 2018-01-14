from actionProvider import ActionProvider


class GameInterface:

    def __init__(self):
        self.delegate = None
        self._should_run = False

    @property
    def action_space_length(self) -> int:
        raise NotImplementedError

    @property
    def state_shape(self) -> tuple:
        raise NotImplementedError

    def run(self, action_provider: ActionProvider, display: bool, num_episodes: int = None):
        raise NotImplementedError

    def stop(self):
        self._should_run = False
