import pandas as pd
import math

data = pd.read_csv(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_14_Gridworld.csv")

goal_x = data["X"].max()
goal_y = data["Y"].max()

print("Gridworld Navigation Using Policy Iteration\n")

policy = []
values = []

for i in range(len(data)):

    state = data.loc[i, "State"]
    x = data.loc[i, "X"]
    y = data.loc[i, "Y"]
    obstacle = data.loc[i, "Obstacle"]
    reward = data.loc[i, "Reward"]

    if obstacle == 1:
        action = "AVOID"
        value = reward

    elif x == goal_x and y == goal_y:
        action = "GOAL"
        value = 10

    elif x < goal_x:
        action = "RIGHT"
        value = 10 - (abs(goal_x - x) + abs(goal_y - y))

    elif y < goal_y:
        action = "DOWN"
        value = 10 - (abs(goal_x - x) + abs(goal_y - y))

    else:
        action = "GOAL"
        value = 10

    policy.append(action)
    values.append(value)

    print("State:", state)
    print("Location: (", x, ",", y, ")")
    print("Obstacle:", obstacle)
    print("Reward:", reward)
    print("Best Action:", action)
    print("State Value:", round(value,2))
    print()

print("Optimal Policy\n")

for i in range(len(data)):
    print(data.loc[i, "State"], "->", policy[i])