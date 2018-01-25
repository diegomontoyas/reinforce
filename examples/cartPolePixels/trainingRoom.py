from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from examples.cartpolePixels.cartPolePixelsGameInterface import CartPolePixelsGameInterface
from source.epsilonChangeFunctions.epsilonChangeFunctions import ConstantEpsilonFunction
from source.trainers.deepQLearningTrainer import DeepQLearningTrainer


class TrainingRoom():
    def __init__(self):
        super().__init__()
        self.game = CartPolePixelsGameInterface()
        model = self.build_model()

        epsilon_function = ConstantEpsilonFunction(initial_value=0.05)

        self.trainer = DeepQLearningTrainer(
            model=model,
            game=self.game,
            epsilon_function=epsilon_function,
            transitions_per_episode=2,
            batch_size=32,
            discount=0.95,
            replay_memory_max_size=2000,
            game_for_preview=CartPolePixelsGameInterface(),
            episodes_between_previews=50,
            preview_num_episodes=1
        )

        self.trainer.delegate = self

    def build_model(self):
        model = Sequential()

        shape = self.game.state_shape

        model.add(Dense(512, input_dim=shape[0], activation='relu'))
        model.add(Dense(self.game.num_actions, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        model.summary()
        return model

    def start_training(self):
        self.trainer.train(target_episodes=100000)

if __name__ == "__main__":
    TrainingRoom().start_training()
