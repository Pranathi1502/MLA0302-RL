import gymnasium as gym
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Read Dataset
data = pd.read_csv("Experiment_17_MountainCar.csv")

print("MountainCar Dataset\n")
print(data)
print()

# Create Environment
env = gym.make("MountainCar-v0")

# Build Neural Network
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(24, activation="relu"),
    Dense(24, activation="relu"),
    Dense(3, activation="linear")
])

model.compile(optimizer="adam", loss="mse")

gamma = 0.95
episodes = 5

print("Training MountainCar Agent\n")

for episode in range(episodes):

    state, _ = env.reset()
    state = np.array(state).reshape(1, 2)

    total_reward = 0

    for step in range(200):

        q_values = model.predict(state, verbose=0)

        action = np.argmax(q_values[0])

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_state = np.array(next_state).reshape(1, 2)

        target = reward

        if not done:
            future_reward = np.max(model.predict(next_state, verbose=0)[0])
            target = reward + gamma * future_reward

        target_q = model.predict(state, verbose=0)
        target_q[0][action] = target

        model.fit(state, target_q, epochs=1, verbose=0)

        state = next_state
        total_reward += reward

        if done:
            break

    print("Episode:", episode + 1)
    print("Total Reward:", total_reward)
    print()

print("Training Completed\n")

state, _ = env.reset()
state = np.array(state).reshape(1, 2)

q = model.predict(state, verbose=0)

actions = ["LEFT", "NO PUSH", "RIGHT"]

best_action = actions[np.argmax(q)]

print("Best Initial Action:", best_action)

env.close()