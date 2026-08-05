import pandas as pd

data = pd.read_csv(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_15_Call_Center.csv")

print("Monte Carlo Policy Control\n")

representatives = data["Representative"].unique()

returns = {}
counts = {}
policy = {}

for rep in representatives:
    returns[rep] = 0
    counts[rep] = 0
    policy[rep] = "Assign"

for i in range(len(data)):

    rep = data.loc[i, "Representative"]
    call_time = data.loc[i, "Handling_Time"]

    # Reward is higher for shorter handling time
    reward = 10 - call_time

    returns[rep] += reward
    counts[rep] += 1

    value = returns[rep] / counts[rep]

    if value >= 5:
        action = "Assign"
    else:
        action = "Reassign"

    policy[rep] = action

    print("Call ID:", data.loc[i, "Call_ID"])
    print("Representative:", rep)
    print("Handling Time:", call_time)
    print("Reward:", reward)
    print("Average Value:", round(value,2))
    print("Policy:", action)
    print()

print("Final Policy\n")

for rep in representatives:
    print(rep, "->", policy[rep])