import pandas as pd
import numpy as np
import random

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_03_Bandit.xlsx")

prices = data["Price"].tolist()
probabilities = data["TrueCTR"].tolist()

n_arms = len(prices)
rounds = 100

epsilon = 0.1
counts = [0] * n_arms
values = [0] * n_arms
epsilon_reward = 0

for i in range(rounds):

    if random.random() < epsilon:
        arm = random.randint(0, n_arms - 1)
    else:
        arm = np.argmax(values)

    reward = prices[arm] if random.random() < probabilities[arm] else 0

    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    epsilon_reward += reward

counts = [0] * n_arms
values = [0] * n_arms
ucb_reward = 0

for i in range(rounds):

    if 0 in counts:
        arm = counts.index(0)
    else:
        ucb = []

        for j in range(n_arms):
            value = values[j] + np.sqrt((2 * np.log(i + 1)) / counts[j])
            ucb.append(value)

        arm = np.argmax(ucb)

    reward = prices[arm] if random.random() < probabilities[arm] else 0

    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    ucb_reward += reward

success = [1] * n_arms
failure = [1] * n_arms
thompson_reward = 0

for i in range(rounds):

    samples = [np.random.beta(success[j], failure[j]) for j in range(n_arms)]

    arm = np.argmax(samples)

    if random.random() < probabilities[arm]:
        reward = prices[arm]
        success[arm] += 1
    else:
        reward = 0
        failure[arm] += 1

    thompson_reward += reward

print("Revenue Comparison\n")
print("Epsilon-Greedy Revenue :", epsilon_reward)
print("UCB Revenue :", ucb_reward)
print("Thompson Sampling Revenue :", thompson_reward)

best = max(epsilon_reward, ucb_reward, thompson_reward)

if best == epsilon_reward:
    print("\nBest Strategy : Epsilon-Greedy")

elif best == ucb_reward:
    print("\nBest Strategy : UCB")

else:
    print("\nBest Strategy : Thompson Sampling")