import pandas as pd
import random
import math

data = pd.read_excel(r"C:\Users\SAMSUNG\Downloads\RL_Experiments_1_to_10_Separate_Excel_Files\Experiment_06_Ads.xlsx")

n = len(data)
rounds = 500

epsilon = 0.1

counts = [0] * n
values = [0] * n
eps_reward = 0

for t in range(rounds):
    if random.random() < epsilon:
        arm = random.randint(0, n - 1)
    else:
        arm = values.index(max(values))

    reward = 1 if random.random() < data.loc[arm, "TrueCTR"] else 0
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    eps_reward += reward

counts = [0] * n
values = [0] * n
ucb_reward = 0

for t in range(rounds):

    if t < n:
        arm = t
    else:
        ucb = []
        for i in range(n):
            score = values[i] + math.sqrt(2 * math.log(t + 1) / counts[i])
            ucb.append(score)
        arm = ucb.index(max(ucb))

    reward = 1 if random.random() < data.loc[arm, "TrueCTR"] else 0
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    ucb_reward += reward

success = [1] * n
failure = [1] * n
ts_reward = 0

for t in range(rounds):

    samples = [random.betavariate(success[i], failure[i]) for i in range(n)]
    arm = samples.index(max(samples))

    reward = 1 if random.random() < data.loc[arm, "TrueCTR"] else 0

    if reward == 1:
        success[arm] += 1
    else:
        failure[arm] += 1

    ts_reward += reward

print("Advertisement Bandit Results\n")

print("Epsilon-Greedy CTR :", round(eps_reward / rounds, 3))
print("UCB CTR :", round(ucb_reward / rounds, 3))
print("Thompson Sampling CTR :", round(ts_reward / rounds, 3))
print()

best = max(eps_reward, ucb_reward, ts_reward)

if best == eps_reward:
    print("Best Algorithm : Epsilon-Greedy")
elif best == ucb_reward:
    print("Best Algorithm : UCB")
else:
    print("Best Algorithm : Thompson Sampling")