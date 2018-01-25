# Reinforce

<p align="center"><img src ="./assets/trex_gif.gif" /></p>


`Reinforce` is (will be) a Python library to facilitate playing with reinforcement learning. A reinforcement learning playground? 
The idea is to add different algorithms with time. So far there's only an implementation of Deep Q Learning.

# How it works (so far)

In order to run an example we need a **game**, a **trainer**, and a **traning room**. The **trainer** trains a **model** using a **game** in a **traning room**.
The trainer is provided by the library, but a custom implementation of the game must be provided by conforming to `GameInterface`.

## Example usage

```python
# TRAINING ROOM

epsilon_function = ConstantEpsilonFunction(initial_value=0.05)

trainer = DeepQLearningTrainer(
    model=build_model(),  # Keras model to use
    game=CartPoleGameInterface()  # Game wrapper conforming to `MarkovDecisionProcess`
    epsilon_function=epsilon_function,
    transitions_per_episode=2,  # Train every 2 episodes
    batch_size=32,
    discount=0.95,
    replay_memory_max_size=2000,
    game_for_preview=CartPoleGameInterface(),  # The game to use for previews conforming to `GameInterface`
    episodes_between_previews=250,  # Preview the game with epsilon=0 every 250 episodes
    preview_num_episodes=1  # The epsilon 0 preview will last for 1 episode
)

trainer.train(target_episodes=20000)

```

By default, analytics will be logged using `Tensorboard` under `./analytics` and checkpoints of the model will be saved to `./checkpoints`. These checkpoints can then be loaded back when creating a trainer.

# TODOs

- Add more trainers (ex. Double Deep Q Learning)
- Add more examples
- Improve current implementations (ex. implement prioritized experience replay)
- Add a way to visualize trained models
- Improve Trex training
- Buy a time machine to skip training
