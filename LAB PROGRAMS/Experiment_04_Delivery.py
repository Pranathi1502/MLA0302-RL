import pandas as pd
import math

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_04_Delivery.xlsx")

goal_x = data.iloc[-1]["X"]
goal_y = data.iloc[-1]["Y"]

data["Distance"] = 0.0
data["Policy"] = ""

for i in range(len(data)):

    x = data.loc[i, "X"]
    y = data.loc[i, "Y"]

    distance = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

    if x < goal_x:
        action = "RIGHT"
    elif y < goal_y:
        action = "DOWN"
    else:
        action = "GOAL"

    data.loc[i, "Distance"] = round(distance, 2)
    data.loc[i, "Policy"] = action

print("Policy Iteration Results\n")

for i in range(len(data)):
    print("Point:", data.loc[i, "Point"])
    print("Name:", data.loc[i, "Name"])
    print("Location:", "(", data.loc[i, "X"], ",", data.loc[i, "Y"], ")")
    print("Best Action:", data.loc[i, "Policy"])
    print("Distance to Goal:", data.loc[i, "Distance"])
    print()

print("Optimal Policy\n")

for i in range(len(data)):
    print(data.loc[i, "Name"], "->", data.loc[i, "Policy"])