import numpy as np

print("Experiment 24 - Inventory Management")
print("--------------------------------------")

# Inventory states: 0 to 10 units
max_inventory = 10

# Possible order quantities
actions = [0, 2, 4, 6]

# Demand values
demands = [1, 2, 3]

# Discount factor
gamma = 0.9

# Cost parameters
ordering_cost = 2
holding_cost = 1
shortage_cost = 5

# Initial value function
V = np.zeros(max_inventory + 1)

# Optimal policy
policy = np.zeros(max_inventory + 1, dtype=int)


# ------------------------------------------------
# Calculate cost
# ------------------------------------------------

def calculate_cost(inventory, order, demand):

    # Inventory after ordering
    available = inventory + order

    # Inventory after demand
    ending_inventory = available - demand

    # Ordering cost
    cost = 0

    if order > 0:
        cost += ordering_cost + order

    # Holding cost
    if ending_inventory >= 0:
        cost += holding_cost * ending_inventory

    # Shortage cost
    else:
        cost += shortage_cost * abs(ending_inventory)

    return cost


# ------------------------------------------------
# Bellman Optimality Equation
# ------------------------------------------------

for iteration in range(100):

    new_V = np.zeros(max_inventory + 1)

    for inventory in range(max_inventory + 1):

        action_costs = []

        for order in actions:

            # Do not exceed maximum inventory
            if inventory + order > max_inventory:
                continue

            expected_cost = 0

            for demand in demands:

                cost = calculate_cost(
                    inventory,
                    order,
                    demand
                )

                next_inventory = inventory + order - demand

                # Keep next state inside range
                next_inventory = max(
                    0,
                    min(max_inventory, next_inventory)
                )

                expected_cost += (
                    cost + gamma * V[next_inventory]
                ) / len(demands)

            action_costs.append(
                (expected_cost, order)
            )

        # Select action with minimum cost
        best_cost, best_action = min(
            action_costs,
            key=lambda x: x[0]
        )

        new_V[inventory] = best_cost
        policy[inventory] = best_action

    # Check convergence
    if np.max(np.abs(new_V - V)) < 0.001:
        V = new_V
        break

    V = new_V


# ------------------------------------------------
# Display results
# ------------------------------------------------

print("\nOptimal Inventory Policy")
print("------------------------")

for inventory in range(max_inventory + 1):

    print(
        "Inventory:",
        inventory,
        "| Order:",
        policy[inventory],
        "| Minimum Cost:",
        round(V[inventory], 2)
    )


# ------------------------------------------------
# Demonstrate policy
# ------------------------------------------------

print("\nPolicy Demonstration")
print("--------------------")

inventory = 2
total_cost = 0

for day in range(1, 6):

    order = policy[inventory]

    demand = demands[(day - 1) % len(demands)]

    cost = calculate_cost(
        inventory,
        order,
        demand
    )

    new_inventory = inventory + order - demand

    if new_inventory < 0:
        new_inventory = 0

    print(
        "Day:",
        day,
        "| Starting Stock:",
        inventory,
        "| Order:",
        order,
        "| Demand:",
        demand,
        "| Cost:",
        cost,
        "| Ending Stock:",
        new_inventory
    )

    total_cost += cost
    inventory = new_inventory


print("\nTotal Cost for 5 Days:", total_cost)

print("\nExperiment 24 Completed Successfully!")