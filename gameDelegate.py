import numpy as np

from transition import Transition
from gameInterface import GameInterface


class GameDelegate:

    def game_did_receive_update(self, game: GameInterface, transition: Transition):
        raise NotImplementedError

    def game_did_end_episode(self, game: GameInterface):
        pass

    def new_game_did_start(self, game: GameInterface):
        pass
