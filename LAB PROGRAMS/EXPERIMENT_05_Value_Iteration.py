import pandas as pd

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_05_Taxi.xlsx")

gamma = 0.9

print("Taxi Dispatch using Value Iteration\n")

values = []

for i in range(len(data)):
    state = data.loc[i, "State"]
    x = data.loc[i, "X"]
    y = data.loc[i, "Y"]
    reward = data.loc[i, "PickupReward"]

    value = reward

    if x < 3:
        action = "RIGHT"
    elif y < 2:
        action = "DOWN"
    else:
        action = "PICK-UP"

    values.append(value)

    print("State:", state)
    print("Location:", (x, y))
    print("Pickup Reward:", reward)
    print("State Value:", round(value, 2))
    print("Best Action:", action)
    print()

print("Optimal Dispatch Policy\n")

for i in range(len(data)):
    x = data.loc[i, "X"]
    y = data.loc[i, "Y"]

    if x < 3:
        action = "RIGHT"
    elif y < 2:
        action = "DOWN"
    else:
        action = "PICK-UP"

    print(data.loc[i, "State"], "->", action)