import pandas as pd
import random

data = pd.read_csv(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_11_Stock_Trading.csv")

actions = ["BUY", "SELL", "HOLD"]

q1 = {}
q2 = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for state in data["Day"]:
    q1[state] = [0, 0, 0]
    q2[state] = [0, 0, 0]

print("Double DQN Stock Trading\n")

total_reward = 0

for i in range(len(data)):

    state = data.loc[i, "Day"]
    price = data.loc[i, "Stock_Price"]
    reward = data.loc[i, "Reward"]

    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        values = []
        for j in range(3):
            values.append(q1[state][j] + q2[state][j])
        action = values.index(max(values))

    total_reward += reward

    if i < len(data) - 1:
        next_state = data.loc[i + 1, "Day"]

        if random.random() < 0.5:
            best = q1[next_state].index(max(q1[next_state]))
            target = reward + gamma * q2[next_state][best]
            q1[state][action] += alpha * (target - q1[state][action])
        else:
            best = q2[next_state].index(max(q2[next_state]))
            target = reward + gamma * q1[next_state][best]
            q2[state][action] += alpha * (target - q2[state][action])

    print("Day:", state)
    print("Stock Price:", price)
    print("Action:", actions[action])
    print("Reward:", reward)
    print()

print("Training Completed\n")

print("Total Reward:", total_reward)

print("\nLearned Policy\n")

for state in data["Day"]:
    values = []
    for j in range(3):
        values.append(q1[state][j] + q2[state][j])
    best = values.index(max(values))
    print(state, "->", actions[best])