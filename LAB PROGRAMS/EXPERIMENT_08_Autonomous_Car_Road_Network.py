import pandas as pd

path = r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_08_Road.xlsx"

data = pd.read_excel(path)

print("Autonomous Car Navigation Results\n")

data["Policy"] = ""
data["Safety Score"] = 0


for i in range(len(data)):

    state = data.iloc[i,0]

    reward = 0

    for col in data.columns:
        if "Reward" in col:
            reward = data.loc[i,col]


    if reward > 0:
        policy = "MOVE FORWARD"
        safety = 100

    elif reward < 0:
        policy = "STOP AND AVOID"
        safety = 40

    else:
        policy = "FOLLOW TRAFFIC RULES"
        safety = 80


    data.loc[i,"Policy"] = policy
    data.loc[i,"Safety Score"] = safety


    print("State:", state)
    print("Policy:", policy)
    print("Safety Score:", safety)
    print()


average = data["Safety Score"].mean()

print("Final Evaluation")
print("Average Safety Score:", round(average,2))


if average >= 80:
    print("Result: Safe Navigation")

else:
    print("Result: Needs Improvement")