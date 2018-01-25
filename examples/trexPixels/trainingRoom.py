from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from examples.trexPixels.trexInterface import TrexGameInterface
from source.epsilonChangeFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction
from source.trainers.deepQLearningTrainer import DeepQLearningTrainer


class TrainingRoom:

    def __init__(self):
        super().__init__()
        self.game = TrexGameInterface()
        model = self.build_model(num_actions=self.game.action_space_length)

        epsilon_function = ConstMultiplierEpsilonDecayFunction(
            initial_value=1,
            final_value=0.01,
            decay_multiplier=0.9999
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
            episodes_between_previews=100,
            preview_num_episodes=1,
            episodes_between_checkpoints=50
        )

    def build_model(self, num_actions):
        model = Sequential()
        shape = self.game.state_shape

        model.add(Dense(256, input_dim=shape[0], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.game.action_space_length, activation='linear'))

        model.add(Dense(num_actions))
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        return model

    def start_training(self):
        self.trainer.train(target_episodes=1000000)


if __name__ == "__main__":
    TrainingRoom().start_training()
