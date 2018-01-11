from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

from epsilonFunctions.epsilonChangeFunctions import ConstMultiplierEpsilonDecayFunction, SinusoidalEpsilonChangeFunction
from examples.cartpole.cartPoleInterface import CartPoleGameInterface
from deepQLearningTrainer import DeepQLearningTrainer
from trainer import Trainer
from trainerDelegate import TrainerDelegate


class TrainingRoom(TrainerDelegate):

    def __init__(self):
        super().__init__()
        self.game = CartPoleGameInterface()
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
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.game.state_shape(), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.game.action_space_length(), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def start_training(self):
        self.trainer.train(num_episodes=50000, display=True)

    def trainer_did_finish_training(self, trainer: Trainer):
        pass


if __name__ == "__main__":
    TrainingRoom().start_training()
