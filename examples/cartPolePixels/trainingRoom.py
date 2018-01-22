from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from deepQLearningTrainer import DeepQLearningTrainer
from epsilonFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction
from examples.cartPolePixels.cartPolePixelsGameInterface import CartPolePixelsGameInterface
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

    def build_model(self):
        model = Sequential()

        shape = self.game.state_shape

        model.add(Dense(128, input_dim=shape[0], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.game.action_space_length, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        model.summary()
        return model

    def start_training(self):
        self.trainer.train(target_episodes=100000, game_for_preview=CartPolePixelsGameInterface(),
                           episodes_between_previews=15, preview_num_episodes=1)

    def trainer_did_finish_training(self, trainer: Trainer):
        pass


if __name__ == "__main__":
    TrainingRoom().start_training()
