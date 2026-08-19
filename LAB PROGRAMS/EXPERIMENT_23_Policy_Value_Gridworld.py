import numpy as np
import matplotlib.pyplot as plt

print("Experiment 23 - Policy and Value Functions")
print("-------------------------------------------")

# Gridworld
rows = 4
cols = 4

# Goal state
goal = (3, 3)

# Obstacles
obstacles = [(1, 1), (2, 1)]

# Actions
actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# ------------------------------------------------
# Function to get next state
# ------------------------------------------------

def next_state(state, action):

    r, c = state
    dr, dc = actions[action]

    nr = r + dr
    nc = c + dc

    # Boundary check
    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state

    # Obstacle check
    if (nr, nc) in obstacles:
        return state

    return (nr, nc)


# ------------------------------------------------
# Two different policies
# ------------------------------------------------

policy1 = {}

for r in range(rows):
    for c in range(cols):

        state = (r, c)

        if state == goal or state in obstacles:
            continue

        # Policy 1: Prefer RIGHT, then DOWN
        if c < cols - 1 and (r, c + 1) not in obstacles:
            policy1[state] = "RIGHT"
        else:
            policy1[state] = "DOWN"


policy2 = {}

for r in range(rows):
    for c in range(cols):

        state = (r, c)

        if state == goal or state in obstacles:
            continue

        # Policy 2: Prefer DOWN, then RIGHT
        if r < rows - 1 and (r + 1, c) not in obstacles:
            policy2[state] = "DOWN"
        else:
            policy2[state] = "RIGHT"


# ------------------------------------------------
# Policy Evaluation
# ------------------------------------------------

def evaluate_policy(policy):

    values = np.zeros((rows, cols))

    gamma = 0.9

    for iteration in range(100):

        new_values = values.copy()

        for r in range(rows):
            for c in range(cols):

                state = (r, c)

                if state == goal:
                    new_values[r, c] = 10
                    continue

                if state in obstacles:
                    continue

                action = policy[state]
                next_s = next_state(state, action)

                # Reward
                if next_s == goal:
                    reward = 10
                elif next_s == state:
                    reward = -2
                else:
                    reward = -1

                nr, nc = next_s

                # Bellman expectation equation
                new_values[r, c] = reward + gamma * values[nr, nc]

        values = new_values

    return values


# ------------------------------------------------
# Evaluate both policies
# ------------------------------------------------

value1 = evaluate_policy(policy1)
value2 = evaluate_policy(policy2)


# ------------------------------------------------
# Display Policy 1
# ------------------------------------------------

print("\nPolicy 1")
print("--------")

for state, action in policy1.items():
    print(
        "State",
        state,
        "->",
        action,
        "| Value =",
        round(value1[state], 2)
    )


# ------------------------------------------------
# Display Policy 2
# ------------------------------------------------

print("\nPolicy 2")
print("--------")

for state, action in policy2.items():
    print(
        "State",
        state,
        "->",
        action,
        "| Value =",
        round(value2[state], 2)
    )


# ------------------------------------------------
# Display Value Functions
# ------------------------------------------------

print("\nValue Function - Policy 1")
print(np.round(value1, 2))

print("\nValue Function - Policy 2")
print(np.round(value2, 2))


# ------------------------------------------------
# Visualization - Policy 1
# ------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(value1)

plt.colorbar(label="State Value")

for r in range(rows):
    for c in range(cols):

        if (r, c) in obstacles:
            plt.text(
                c,
                r,
                "X",
                ha="center",
                va="center",
                fontsize=18
            )

        elif (r, c) == goal:
            plt.text(
                c,
                r,
                "G",
                ha="center",
                va="center",
                fontsize=16
            )

        else:
            plt.text(
                c,
                r,
                f"{value1[r, c]:.1f}",
                ha="center",
                va="center"
            )

plt.title("Value Function - Policy 1")
plt.xlabel("Column")
plt.ylabel("Row")
plt.grid()
plt.show()


# ------------------------------------------------
# Visualization - Policy 2
# ------------------------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(value2)

plt.colorbar(label="State Value")

for r in range(rows):
    for c in range(cols):

        if (r, c) in obstacles:
            plt.text(
                c,
                r,
                "X",
                ha="center",
                va="center",
                fontsize=18
            )

        elif (r, c) == goal:
            plt.text(
                c,
                r,
                "G",
                ha="center",
                va="center",
                fontsize=16
            )

        else:
            plt.text(
                c,
                r,
                f"{value2[r, c]:.1f}",
                ha="center",
                va="center"
            )

plt.title("Value Function - Policy 2")
plt.xlabel("Column")
plt.ylabel("Row")
plt.grid()
plt.show()


print("\nExperiment 23 Completed Successfully!")