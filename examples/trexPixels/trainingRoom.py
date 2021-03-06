from keras import Sequential
from keras.layers import Dense, Convolution2D, Flatten, MaxPooling2D
from keras.optimizers import Adam

from examples.trexPixels.trexInterface import TrexGameInterface
from source.epsilonUpdater.constantMultiplierEpsilonUpdater import ConstMultiplierEpsilonUpdater
from source.replayMemory.simpleDequeReplayMemory import SimpleDequeReplayMemory
from source.trainers.deepQLearningTrainer import DeepQLearningTrainer


class TrainingRoom:

    def __init__(self):
        super().__init__()
        self.game = TrexGameInterface()
        model = self.build_model()

        epsilon_function = ConstMultiplierEpsilonUpdater(
            initial_value=1,
            final_value=0.01,
            decay_multiplier=0.99999
        )

        self.trainer = DeepQLearningTrainer(
            model=model,
            game=self.game,
            epsilon_function=epsilon_function,
            replay_memory=SimpleDequeReplayMemory(max_size=100000),
            transitions_per_episode=1,
            batch_size=32,
            discount=0.95,
            game_for_preview=TrexGameInterface(),
            episodes_between_previews=250,
            preview_num_episodes=1,
            episodes_between_checkpoints=50,
            log_analytics=True
        )

    def build_model(self):
        model = Sequential()
        shape = self.game.state_shape

        model.add(Convolution2D(filters=32, kernel_size=3, strides=(2,2), input_shape=shape, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.game.num_actions))

        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def start_training(self):
        self.trainer.train(target_episodes=1000000)


if __name__ == "__main__":
    TrainingRoom().start_training()
