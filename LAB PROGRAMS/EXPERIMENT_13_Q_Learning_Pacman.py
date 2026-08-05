import pandas as pd
import random

data = pd.read_csv(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_13_Qlearning_Game.csv")

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

q_table = {}

for state in data["State"]:
    q_table[state] = [0.0, 0.0, 0.0, 0.0]

print("Q-Learning Grid Game\n")

total_reward = 0

for i in range(len(data) - 1):

    state = data.loc[i, "State"]

    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        action = q_table[state].index(max(q_table[state]))

    reward = data.loc[i, "Reward"]

    next_state = data.loc[i + 1, "State"]

    best_next = max(q_table[next_state])

    q_table[state][action] = q_table[state][action] + alpha * (
        reward + gamma * best_next - q_table[state][action]
    )

    total_reward += reward

    print("State:", state)
    print("Food:", data.loc[i, "Food"])
    print("Ghost:", data.loc[i, "Ghost"])
    print("Action:", actions[action])
    print("Reward:", reward)
    print()

print("Training Completed\n")

print("Total Reward:", total_reward)

print("\nLearned Policy\n")

for state in data["State"]:
    best_action = q_table[state].index(max(q_table[state]))
    print(state, "->", actions[best_action])