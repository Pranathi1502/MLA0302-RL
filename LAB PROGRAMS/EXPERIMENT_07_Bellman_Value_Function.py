import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_07_Bellman.xlsx")

gamma = 0.9

data["Value"] = 0.0
data["Policy"] = ""

for i in range(len(data)):

    if "Reward" in data.columns:
        reward = data.loc[i, "Reward"]
    elif "DeliveryReward" in data.columns:
        reward = data.loc[i, "DeliveryReward"]
    else:
        reward = 0

    value = reward + gamma * 0

    if reward > 0:
        action = "MOVE TO DELIVERY"
    elif reward < 0:
        action = "AVOID"
    else:
        action = "MOVE"

    data.loc[i, "Value"] = value
    data.loc[i, "Policy"] = action

print("Bellman Value Function Results\n")

for i in range(len(data)):
    print("State:", data.iloc[i,0])
    print("Value:", round(data.loc[i,"Value"],2))
    print("Best Policy:", data.loc[i,"Policy"])
    print()

print("Optimal Delivery Policy\n")

for i in range(len(data)):
    print(data.iloc[i,0],"->",data.loc[i,"Policy"])

plt.figure(figsize=(8,5))
plt.bar(data.iloc[:,0], data["Value"])
plt.xlabel("Delivery Points")
plt.ylabel("State Value")
plt.title("Bellman State Value Function")
plt.xticks(rotation=45)
plt.show()