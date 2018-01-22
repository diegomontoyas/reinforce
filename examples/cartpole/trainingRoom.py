import asyncio
from threading import Thread

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

from dispatch import Dispatch
from epsilonFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction, SinusoidalEpsilonChangeFunction, \
    ConstantEpsilonFunction
from examples.cartpole.cartPoleInterface import CartPoleGameInterface
from deepQLearningTrainer import DeepQLearningTrainer


class TrainingRoom:

    def __init__(self):
        super().__init__()
        self.game = CartPoleGameInterface()
        model = self.build_model()

        epsilon_function = ConstantEpsilonFunction(initial_value=0)

        self.trainer = DeepQLearningTrainer(
            model=model,
            game=self.game,
            epsilon_function=epsilon_function,
            transitions_per_episode=2,
            batch_size=32,
            discount=0.95,
            replay_memory_max_size=2000,
            game_for_preview=CartPoleGameInterface(),
            episodes_between_previews=250,
            preview_num_episodes=1
        )

        self.trainer.delegate = self

    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.game.state_shape[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.game.action_space_length, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def start_training(self):
        self.trainer.train(target_episodes=30000)


if __name__ == "__main__":

    def main():
        TrainingRoom().start_training()

    Dispatch.main(main)
