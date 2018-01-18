from keras import Sequential
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D
from keras.optimizers import Adam

from deepQLearningTrainer import DeepQLearningTrainer
from epsilonFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction
from examples.cartpole.cartPolePixelsGameInterface import CartPolePixelsGameInterface
from trainer import Trainer
from trainerDelegate import TrainerDelegate
from examples.trex.trexInterface import TrexGameInterface


class TrainingRoom(TrainerDelegate):

    def __init__(self):
        super().__init__()
        self.game = TrexGameInterface()
        model = self.build_model(num_actions=self.game.action_space_length)

        epsilon_function = ConstMultiplierEpsilonDecayFunction(
            initial_value=1,
            final_value=0.01,
            decay_multiplier=0.995
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

    def build_model(self, num_actions):
        model = Sequential()

        shape = self.game.state_shape

        model.add(Convolution2D(filters=32, kernel_size=8, strides=8, input_shape=(shape[0], shape[1], 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

        model.add(Convolution2D(filters=64, kernel_size=4, strides=4))
        model.add(Activation('relu'))

        model.add(Convolution2D(filters=64, kernel_size=3, strides=3))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))

        model.compile(loss='mse', optimizer=Adam(lr=1e-6))
        return model

    def start_training(self):
        self.trainer.train(num_episodes=100000)

    def trainer_did_finish_training(self, trainer: Trainer):
        pass


if __name__ == "__main__":
    TrainingRoom().start_training()
