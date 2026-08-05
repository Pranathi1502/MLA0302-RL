import pandas as pd

data = pd.read_csv("Experiment_16_Robot_Grid.csv")

gamma = 0.9

data["Value"] = 0.0
data["Policy"] = ""

goal_x = 4
goal_y = 0

print("Bellman Optimality Robot Navigation\n")

for i in range(len(data)):

    state = data.loc[i, "State"]
    x = data.loc[i, "X"]
    y = data.loc[i, "Y"]
    reward = data.loc[i, "Reward"]

    if x < goal_x:
        action = "RIGHT"
    elif y > goal_y:
        action = "UP"
    else:
        action = "GOAL"

    distance = abs(goal_x - x) + abs(goal_y - y)

    value = reward + gamma * (10 - distance)

    data.loc[i, "Value"] = round(value, 2)
    data.loc[i, "Policy"] = action

    print("State:", state)
    print("Location:", (x, y))
    print("Reward:", reward)
    print("Value:", round(value, 2))
    print("Best Action:", action)
    print()

print("Optimal Path\n")

for i in range(len(data)):
    print(data.loc[i, "State"], "->", data.loc[i, "Policy"])