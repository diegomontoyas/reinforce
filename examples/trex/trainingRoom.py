from keras import Sequential
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D
from keras.optimizers import Adam

from deepQLearningTrainer import DeepQLearningTrainer
from epsilonChangeFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction
from examples.trex.trexInterface import TrexGameInterface
from trainer import Trainer


class TrainingRoom:

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
            transitions_per_episode=1,
            batch_size=32,
            discount=0.95,
            replay_memory_max_size=2000,
            game_for_preview=TrexGameInterface(),
            episodes_between_previews=15,
            preview_num_episodes=1
        )

    def build_model(self, num_actions):
        model = Sequential()

        model.add(Convolution2D(filters=16, kernel_size=8, strides=8,
                                input_shape=self.game.state_shape))

        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=5))

        model.add(Convolution2D(filters=16, kernel_size=2, strides=2))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(num_actions))

        model.compile(loss='mse', optimizer=Adam(lr=1e-6))
        return model

    def start_training(self):
        self.trainer.train(target_episodes=600000)


if __name__ == "__main__":
    TrainingRoom().start_training()
