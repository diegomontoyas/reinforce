from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from examples.cartpolePixels.cartPolePixelsGameInterface import CartPolePixelsGameInterface
from examples.pongPixels.pongPixelsGameInterface import PongPixelsGameInterface
from source.epsilonChangeFunctions.epsilonChangeFunctions import ConstantEpsilonFunction, \
    ConstMultiplierEpsilonDecayFunction
from source.trainers.deepQLearningTrainer import DeepQLearningTrainer


class TrainingRoom():
    def __init__(self):
        super().__init__()
        self.game = PongPixelsGameInterface()
        model = self.build_model()

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
            discount=0.98,
            replay_memory_max_size=100000,
            game_for_preview=PongPixelsGameInterface(),
            episodes_between_previews=1500,
            preview_num_episodes=1
        )

    def build_model(self):
        model = Sequential()

        shape = self.game.state_shape

        model.add(Dense(512, input_dim=shape[0], activation='relu'))
        model.add(Dense(self.game.num_actions))

        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        model.summary()
        return model

    def start_training(self):
        self.trainer.train(target_episodes=100000)

if __name__ == "__main__":
    TrainingRoom().start_training()
