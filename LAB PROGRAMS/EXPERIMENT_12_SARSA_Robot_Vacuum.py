import pandas as pd
import random

data = pd.read_csv(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_12_Robot_Vacuum.csv")

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

q_table = {}

for state in data["State"]:
    q_table[state] = [0.0, 0.0, 0.0, 0.0]

print("SARSA Robot Vacuum Cleaning\n")

total_reward = 0

for i in range(len(data) - 1):

    state = data.loc[i, "State"]

    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        action = q_table[state].index(max(q_table[state]))

    reward = data.loc[i, "Reward"]
    next_state = data.loc[i + 1, "State"]

    if random.random() < epsilon:
        next_action = random.randint(0, 3)
    else:
        next_action = q_table[next_state].index(max(q_table[next_state]))

    q_table[state][action] = q_table[state][action] + alpha * (
        reward + gamma * q_table[next_state][next_action] - q_table[state][action]
    )

    total_reward += reward

    print("State:", state)
    print("Obstacle:", data.loc[i, "Obstacle"])
    print("Action:", actions[action])
    print("Reward:", reward)
    print()

print("Training Completed\n")

print("Total Reward:", total_reward)

print("\nOptimal Cleaning Policy\n")

for state in data["State"]:
    best_action = q_table[state].index(max(q_table[state]))
    print(state, "->", actions[best_action])