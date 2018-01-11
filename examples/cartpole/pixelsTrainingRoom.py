from keras import Sequential
from keras.layers import Dense, Convolution2D, Activation, Flatten
from keras.optimizers import Adam, SGD

from epsilonFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction
from examples.cartpole.cartPoleInterface import CartPoleGameInterface
from deepQLearningTrainer import DeepQLearningTrainer
from examples.cartpole.cartPolePixelsGameInterface import CartPolePixelsGameInterface
from trainer import Trainer
from trainerDelegate import TrainerDelegate


class TrainingRoom(TrainerDelegate):

    def __init__(self):
        super().__init__()
        self.game = CartPolePixelsGameInterface()
        model = self.build_model()

        epsilon_function = ConstMultiplierEpsilonDecayFunction(
            initial_value=1,
            final_value=0.01,
            decay_multiplier=0.991
        )

        self.trainer = DeepQLearningTrainer(
            model=model,
            game=self.game,
            epsilon_function=epsilon_function,
            transitions_per_episode=2,
            batch_size=32,
            discount=0.95,
            replay_memory_max_size=2000
        )

        self.trainer.delegate = self

    def build_model(self):
        model = Sequential()

        shape = self.game.state_shape()

        model.add(Convolution2D(filters=32, kernel_size=8, strides=8, padding='same',
                                input_shape=(shape[0], shape[1], 1)))

        model.add(Activation('relu'))
        model.add(Convolution2D(filters=64, kernel_size=4, strides=4, padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(filters=64, kernel_size=3, strides=3, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))

        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return model

    def start_training(self):
        self.trainer.train(num_episodes=50000, display=True)

    def trainer_did_finish_training(self, trainer: Trainer):
        pass


if __name__ == "__main__":
    TrainingRoom().start_training()
