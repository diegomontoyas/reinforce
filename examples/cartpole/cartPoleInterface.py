import gym

from source.gameInterface import *
from source.markovDecisionProcess.markovDecisionProcess import MarkovDecisionProcess
from source.markovDecisionProcess.transition import Transition


class CartPoleGameInterface(GameInterface, MarkovDecisionProcess):

    def __init__(self):
        super().__init__()

        self._env = gym.make("CartPole-v1")
        self._feature_vector_length = self._env.observation_space.shape[0]
        self._action_space_length = self._env.action_space.n
        self._state = None

    @property
    def num_actions(self) -> int:
        return self._action_space_length

    @property
    def state_shape(self) -> tuple:
        return self._feature_vector_length,

    def state(self):
        return self._state

    def reset(self):
        self._state = self._env.reset()

    def take_action(self, action: int) -> Transition:

        # Advance the game to the next frame based on the action.
        # Reward is 1 for every frame the pole survived
        next_state, reward, is_final, _ = self._env.step(action)

        # We redefine the losing reward so it has more relevance in
        # the transitions and training is faster.
        if is_final:
            reward = -500

        transition = Transition(self.state(), action, reward, next_state, is_final)
        self._state = transition.new_state
        return transition

    def play_episode(self, action_provider: ActionProvider, display: bool=True) -> float:

        self.reset()

        t = 0
        episode_ended = False
        while not episode_ended:
            self._env.render()

            # Decide action
            action = action_provider.action(self.state())
            transition = self.take_action(action)
            t += 1

            episode_ended = transition.game_ended

        score = t + transition.reward
        print("Game finished with score: {}".format(score))
        return score
