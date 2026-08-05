import pandas as pd
import random

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_01_Grid.xlsx")

grid = {}

for index, row in data.iterrows():
    grid[(int(row["Row"]), int(row["Col"]))] = {
        "CellType": row["CellType"],
        "Reward": int(row["Reward"])
    }

ROWS = 5
COLS = 5

current_state = (0, 0)

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

total_reward = 0

for step in range(1, 21):

    action = random.choice(actions)

    row, col = current_state

    if action == "UP":
        row = max(0, row - 1)

    elif action == "DOWN":
        row = min(ROWS - 1, row + 1)

    elif action == "LEFT":
        col = max(0, col - 1)

    elif action == "RIGHT":
        col = min(COLS - 1, col + 1)

    current_state = (row, col)

    cell_type = grid[current_state]["CellType"]
    reward = grid[current_state]["Reward"]

    total_reward += reward

    print("\nStep:", step)
    print("Action:", action)
    print("Current State:", current_state)
    print("Cell Type:", cell_type)
    print("Reward:", reward)

print("\nFinal Position:", current_state)
print("Total Reward:", total_reward)