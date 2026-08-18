import random

print("Manufacturing Process Optimization using Reinforcement Learning")
print()

# -----------------------------------
# 1. ENVIRONMENT
# -----------------------------------

states = ["Low", "Medium", "High"]

# Actions represent different machine settings
actions = {
    0: "Low Machine Setting",
    1: "Medium Machine Setting",
    2: "High Machine Setting"
}

# Product quality reward table
rewards = {
    "Low": {
        0: 50,
        1: 70,
        2: 60
    },

    "Medium": {
        0: 60,
        1: 90,
        2: 75
    },

    "High": {
        0: 55,
        1: 80,
        2: 95
    }
}

# -----------------------------------
# 2. Q-TABLE
# -----------------------------------

q_table = {}

for state in states:
    q_table[state] = [0.0, 0.0, 0.0]

# -----------------------------------
# 3. PARAMETERS
# -----------------------------------

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2

episodes = 1000

# -----------------------------------
# 4. TRAINING
# -----------------------------------

for episode in range(episodes):

    # Random initial machine state
    state = random.choice(states)

    for step in range(10):

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = q_table[state].index(max(q_table[state]))

        # Get reward
        reward = rewards[state][action]

        # Select next state
        next_state = random.choice(states)

        # Maximum future value
        max_next_q = max(q_table[next_state])

        # Q-learning update
        q_table[state][action] = q_table[state][action] + learning_rate * (
            reward
            + discount_factor * max_next_q
            - q_table[state][action]
        )

        state = next_state

# -----------------------------------
# 5. DISPLAY VALUE FUNCTION
# -----------------------------------

print("Learned Value Function")
print("----------------------")

for state in states:

    value = max(q_table[state])

    print("State:", state)
    print("Value:", round(value, 2))
    print()

# -----------------------------------
# 6. DISPLAY OPTIMAL POLICY
# -----------------------------------

print("Optimal Manufacturing Policy")
print("----------------------------")

total_quality = 0

for state in states:

    best_action = q_table[state].index(max(q_table[state]))

    quality = rewards[state][best_action]

    total_quality += quality

    print("State:", state)
    print("Best Machine Setting:", actions[best_action])
    print("Expected Product Quality:", quality)
    print()

# -----------------------------------
# 7. FINAL RESULT
# -----------------------------------

average_quality = total_quality / len(states)

print("Final Evaluation")
print("----------------")
print("Average Product Quality:", round(average_quality, 2))

if average_quality >= 80:
    print("Result: Optimized Manufacturing Process")
else:
    print("Result: Further Optimization Required")