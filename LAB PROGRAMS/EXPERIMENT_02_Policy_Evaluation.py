import pandas as pd

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_02_Warehouse.xlsx")

gamma = 0.9

data["Value"] = 0.0

for i in range(len(data)):
    pick_reward = data.loc[i, "PickReward"]
    goal_reward = data.loc[i, "GoalReward"]
    obstacle_penalty = data.loc[i, "ObstaclePenalty"]

    reward = pick_reward + goal_reward + obstacle_penalty

    value = reward + gamma * 0

    data.loc[i, "Value"] = value

print("\nWarehouse Policy Evaluation\n")

for i in range(len(data)):
    print("State:", data.loc[i, "State"])
    print("Actions:", data.loc[i, "Actions"])
    print("Pick Reward:", data.loc[i, "PickReward"])
    print("Goal Reward:", data.loc[i, "GoalReward"])
    print("Obstacle Penalty:", data.loc[i, "ObstaclePenalty"])
    print("State Value:", round(data.loc[i, "Value"], 2))
    print()

print("Final Value Function")

for i in range(len(data)):
    print(data.loc[i, "State"], ":", round(data.loc[i, "Value"], 2))