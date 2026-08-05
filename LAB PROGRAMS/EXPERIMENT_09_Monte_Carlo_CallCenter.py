import pandas as pd
import random

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_09_CallCenter.xlsx")

print("Monte Carlo Call Center Simulation\n")

values = []

for i in range(len(data)):
    call = data.loc[i, "CallID"]
    rep = data.loc[i, "Representative"]
    time = data.loc[i, "HandlingTime"]
    satisfaction = data.loc[i, "Satisfaction"]

    returns = []

    for j in range(20):
        reward = satisfaction - time + random.randint(-3, 3)
        returns.append(reward)

    value = round(sum(returns) / len(returns), 2)
    values.append(value)

    print("Call ID:", call)
    print("Representative:", rep)
    print("Handling Time:", time)
    print("Satisfaction:", satisfaction)
    print("Estimated Value:", value)
    print()

print("Assignment Policy Value Function\n")

for i in range(len(data)):
    print(data.loc[i, "Representative"], ":", values[i])

best = values.index(max(values))

print("\nBest Assignment")
print("Representative:", data.loc[best, "Representative"])
print("Estimated Value:", values[best])