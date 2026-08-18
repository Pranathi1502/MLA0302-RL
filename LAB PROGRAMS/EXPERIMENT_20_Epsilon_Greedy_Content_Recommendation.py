import pandas as pd
import random

print("Epsilon-Greedy Content Recommendation")
print("--------------------------------------")

# Load dataset
data = pd.read_csv(
    r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_20_Content_Recommendation.csv"
)

print("\nContent Recommendation Dataset")
print(data)

# Content options
contents = data["Content"].unique().tolist()

# Epsilon values
epsilons = [0.1, 0.3, 0.5]

# Number of runs
runs = 5

print("\nEpsilon-Greedy Simulation")
print("-------------------------")

for epsilon in epsilons:

    total_rewards = []

    for run in range(1, runs + 1):

        # Initialize estimated values
        value = {content: 0 for content in contents}

        # Number of selections
        count = {content: 0 for content in contents}

        total_reward = 0

        for step in range(len(data)):

            # Exploration
            if random.random() < epsilon:
                action = random.choice(contents)

            # Exploitation
            else:
                action = max(value, key=value.get)

            # Find a matching content record
            matching_rows = data[data["Content"] == action]

            # Randomly select one user record
            row = matching_rows.sample(1).iloc[0]

            reward = int(row["Clicked"])

            # Update count
            count[action] += 1

            # Incremental average
            value[action] = value[action] + (
                reward - value[action]
            ) / count[action]

            total_reward += reward

        total_rewards.append(total_reward)

        print(
            "Epsilon:",
            epsilon,
            "| Run:",
            run,
            "| Total Reward:",
            total_reward
        )

    # Average performance
    average_reward = sum(total_rewards) / len(total_rewards)

    print(
        "Epsilon:",
        epsilon,
        "| Average Reward:",
        round(average_reward, 2)
    )

    print()

print("Final Analysis")
print("--------------")

print("Epsilon values tested:", epsilons)
print("Number of runs:", runs)

print("\nLower epsilon:")
print("More exploitation of the best-known content.")

print("\nHigher epsilon:")
print("More exploration of different content.")

print("\nResult:")
print("The epsilon-greedy strategy balances exploration and exploitation.")