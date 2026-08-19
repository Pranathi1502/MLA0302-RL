import random
import math

print("K-Armed Bandit - Marketing Campaign Optimization")
print("------------------------------------------------")

# Number of marketing campaigns
k = 5

campaigns = [
    "Email Campaign",
    "Social Media",
    "Search Ads",
    "SMS Campaign",
    "Video Ads"
]

# Actual conversion probabilities of campaigns
true_probabilities = [0.20, 0.35, 0.25, 0.15, 0.40]

# Number of trials
trials = 1000


# -------------------------------------------------
# Epsilon-Greedy Algorithm
# -------------------------------------------------

def epsilon_greedy(epsilon=0.1):

    counts = [0] * k
    values = [0.0] * k

    total_reward = 0

    for _ in range(trials):

        # Exploration
        if random.random() < epsilon:
            action = random.randint(0, k - 1)

        # Exploitation
        else:
            action = values.index(max(values))

        # Generate reward
        reward = 1 if random.random() < true_probabilities[action] else 0

        counts[action] += 1

        # Update average reward
        values[action] += (
            reward - values[action]
        ) / counts[action]

        total_reward += reward

    return total_reward, values, counts


# -------------------------------------------------
# UCB Algorithm
# -------------------------------------------------

def ucb():

    counts = [0] * k
    values = [0.0] * k

    total_reward = 0

    for step in range(trials):

        # Select every campaign once
        if step < k:
            action = step

        else:
            ucb_values = []

            for i in range(k):

                confidence = math.sqrt(
                    (2 * math.log(step + 1)) / counts[i]
                )

                ucb_value = values[i] + confidence
                ucb_values.append(ucb_value)

            action = ucb_values.index(max(ucb_values))

        # Generate reward
        reward = 1 if random.random() < true_probabilities[action] else 0

        counts[action] += 1

        values[action] += (
            reward - values[action]
        ) / counts[action]

        total_reward += reward

    return total_reward, values, counts


# -------------------------------------------------
# Thompson Sampling Algorithm
# -------------------------------------------------

def thompson_sampling():

    successes = [1] * k
    failures = [1] * k

    total_reward = 0

    for _ in range(trials):

        samples = []

        for i in range(k):

            sample = random.betavariate(
                successes[i],
                failures[i]
            )

            samples.append(sample)

        action = samples.index(max(samples))

        # Generate reward
        reward = 1 if random.random() < true_probabilities[action] else 0

        if reward == 1:
            successes[action] += 1
        else:
            failures[action] += 1

        total_reward += reward

    values = []

    counts = []

    for i in range(k):

        estimated_probability = (
            successes[i] /
            (successes[i] + failures[i])
        )

        values.append(estimated_probability)

        counts.append(
            successes[i] + failures[i] - 2
        )

    return total_reward, values, counts


# -------------------------------------------------
# Run Algorithms
# -------------------------------------------------

epsilon_reward, epsilon_values, epsilon_counts = epsilon_greedy(0.1)

ucb_reward, ucb_values, ucb_counts = ucb()

thompson_reward, thompson_values, thompson_counts = thompson_sampling()


# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\nTrue Campaign Conversion Probabilities")
print("---------------------------------------")

for i in range(k):
    print(
        campaigns[i],
        ":", true_probabilities[i]
    )


print("\nEpsilon-Greedy Results")
print("----------------------")

print("Total Reward:", epsilon_reward)

for i in range(k):
    print(
        campaigns[i],
        "| Estimated Reward:",
        round(epsilon_values[i], 3),
        "| Selections:",
        epsilon_counts[i]
    )


print("\nUCB Results")
print("-----------")

print("Total Reward:", ucb_reward)

for i in range(k):
    print(
        campaigns[i],
        "| Estimated Reward:",
        round(ucb_values[i], 3),
        "| Selections:",
        ucb_counts[i]
    )


print("\nThompson Sampling Results")
print("-------------------------")

print("Total Reward:", thompson_reward)

for i in range(k):
    print(
        campaigns[i],
        "| Estimated Reward:",
        round(thompson_values[i], 3),
        "| Selections:",
        thompson_counts[i]
    )


# -------------------------------------------------
# Compare Algorithms
# -------------------------------------------------

print("\nAlgorithm Performance Comparison")
print("---------------------------------")

print(
    "Epsilon-Greedy   :",
    epsilon_reward
)

print(
    "UCB              :",
    ucb_reward
)

print(
    "Thompson Sampling:",
    thompson_reward
)


# Find best algorithm
results = {
    "Epsilon-Greedy": epsilon_reward,
    "UCB": ucb_reward,
    "Thompson Sampling": thompson_reward
}

best_algorithm = max(
    results,
    key=results.get
)

print("\nBest Algorithm:", best_algorithm)
print(
    "Highest Reward:",
    results[best_algorithm]
)


# -------------------------------------------------
# Best Marketing Campaign
# -------------------------------------------------

best_campaign = campaigns[
    true_probabilities.index(
        max(true_probabilities)
    )
]

print("\nBest Marketing Campaign:")
print(best_campaign)

print(
    "Conversion Probability:",
    max(true_probabilities)
)

print("\nExperiment Completed Successfully!")