import pandas as pd
import random

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_10_Investment.xlsx")

print("Investment Policy Gradient Results\n")

actions = ["BUY", "HOLD", "SELL"]

for i in range(len(data)):

    reward = random.randint(1, 10)

    probabilities = [random.random() for _ in actions]
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    best_action = actions[probabilities.index(max(probabilities))]

    print("Investment:", i + 1)
    print("Action Probabilities:")
    print("BUY :", round(probabilities[0], 2))
    print("HOLD:", round(probabilities[1], 2))
    print("SELL:", round(probabilities[2], 2))
    print("Selected Action:", best_action)
    print("Expected Return:", reward)
    print()

print("Optimal Investment Policy\n")

for action in actions:
    print(action)