from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from examples.cartpole.cartPoleInterface import CartPoleGameInterface
from source.epsilonUpdater.constantEpsilonUpdater import ConstantEpsilonUpdater
from source.replayMemory.simpleDequeReplayMemory import SimpleDequeReplayMemory
from source.trainers.deepQLearningTrainer import DeepQLearningTrainer


class TrainingRoom:

    def __init__(self):
        super().__init__()
        self.game = CartPoleGameInterface()
        model = self.build_model()

        epsilon_function = ConstantEpsilonUpdater(initial_value=0.05)

        self.trainer = DeepQLearningTrainer(
            model=model,
            game=self.game,
            epsilon_function=epsilon_function,
            replay_memory=SimpleDequeReplayMemory(max_size=2000),
            transitions_per_episode=2,
            batch_size=32,
            discount=0.95,
            game_for_preview=CartPoleGameInterface(),
            episodes_between_previews=250,
            preview_num_episodes=1
        )

    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=self.game.state_shape[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.game.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def start_training(self):
        self.trainer.train(target_episodes=30000)


if __name__ == "__main__":

    TrainingRoom().start_training()