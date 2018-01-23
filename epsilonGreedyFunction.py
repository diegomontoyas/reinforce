import keras
import numpy as np
import random


def e_greedy_action(state: np.ndarray, model: keras.Model, epsilon: float, action_space_length: int) -> int:
    if np.random.rand() <= epsilon:
        return random.randrange(action_space_length)

    q_values = model.predict(np.array([state]))[0]
    return int(np.argmax(q_values))
