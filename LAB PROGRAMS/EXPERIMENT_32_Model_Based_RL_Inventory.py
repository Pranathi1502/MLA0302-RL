import numpy as np
import matplotlib.pyplot as plt

print("==============================================")
print("EXPERIMENT 32 - MODEL-BASED RL")
print("INVENTORY MANAGEMENT")
print("==============================================")


# -------------------------------------------------
# 1. Parameters
# -------------------------------------------------

np.random.seed(42)

DAYS = 100
MAX_INVENTORY = 50

INITIAL_INVENTORY = 25

ORDER_COST = 2
HOLDING_COST = 1
SHORTAGE_COST = 5


# -------------------------------------------------
# 2. Generate Synthetic Customer Demand
# -------------------------------------------------

demand = np.random.poisson(
    lam=10,
    size=DAYS
)

print("\nSynthetic Customer Demand:")
print(demand[:20])


# -------------------------------------------------
# 3. Inventory Simulation
# -------------------------------------------------

def simulate_policy(policy_name):

    inventory = INITIAL_INVENTORY

    total_cost = 0

    inventory_history = []
    order_history = []
    demand_history = []
    cost_history = []

    for day in range(DAYS):

        current_demand = demand[day]

        # -----------------------------------------
        # Select order quantity
        # -----------------------------------------

        if policy_name == "Fixed Order":

            order = 10

        elif policy_name == "Conservative":

            if inventory < 15:
                order = 20
            else:
                order = 5

        elif policy_name == "Model-Based RL":

            # Estimate future demand
            start = max(0, day - 4)

            predicted_demand = np.mean(
                demand[start:day + 1]
            )

            # Reorder when inventory is low
            target_inventory = predicted_demand * 2

            if inventory < target_inventory:

                order = int(
                    max(
                        0,
                        target_inventory - inventory
                    )
                )

            else:

                order = 0

        # Add ordered stock
        inventory += order

        # Ordering cost
        order_cost = order * ORDER_COST

        # -----------------------------------------
        # Satisfy customer demand
        # -----------------------------------------

        if inventory >= current_demand:

            inventory -= current_demand

            shortage = 0

        else:

            shortage = (
                current_demand - inventory
            )

            inventory = 0

        # Holding cost
        holding_cost = (
            inventory * HOLDING_COST
        )

        # Shortage cost
        shortage_cost = (
            shortage * SHORTAGE_COST
        )

        daily_cost = (
            order_cost
            + holding_cost
            + shortage_cost
        )

        total_cost += daily_cost

        # Store data
        inventory_history.append(inventory)

        order_history.append(order)

        demand_history.append(
            current_demand
        )

        cost_history.append(
            daily_cost
        )

    return {
        "inventory": inventory_history,
        "orders": order_history,
        "demand": demand_history,
        "cost": cost_history,
        "total_cost": total_cost
    }


# -------------------------------------------------
# 4. Evaluate Policies
# -------------------------------------------------

fixed_result = simulate_policy(
    "Fixed Order"
)

conservative_result = simulate_policy(
    "Conservative"
)

rl_result = simulate_policy(
    "Model-Based RL"
)


# -------------------------------------------------
# 5. Display Results
# -------------------------------------------------

print("\n==============================================")
print("POLICY EVALUATION")
print("==============================================")

print(
    "\nFixed Order Policy Cost:",
    round(
        fixed_result["total_cost"],
        2
    )
)

print(
    "Conservative Policy Cost:",
    round(
        conservative_result["total_cost"],
        2
    )
)

print(
    "Model-Based RL Policy Cost:",
    round(
        rl_result["total_cost"],
        2
    )
)


# -------------------------------------------------
# 6. Find Best Policy
# -------------------------------------------------

results = {
    "Fixed Order":
        fixed_result["total_cost"],

    "Conservative":
        conservative_result["total_cost"],

    "Model-Based RL":
        rl_result["total_cost"]
}

best_policy = min(
    results,
    key=results.get
)

print(
    "\nBest Inventory Policy:",
    best_policy
)

print(
    "Minimum Total Cost:",
    round(
        results[best_policy],
        2
    )
)


# -------------------------------------------------
# 7. Average Inventory
# -------------------------------------------------

print("\nAverage Inventory:")

print(
    "Fixed Order:",
    round(
        np.mean(
            fixed_result["inventory"]
        ),
        2
    )
)

print(
    "Conservative:",
    round(
        np.mean(
            conservative_result["inventory"]
        ),
        2
    )
)

print(
    "Model-Based RL:",
    round(
        np.mean(
            rl_result["inventory"]
        ),
        2
    )
)


# -------------------------------------------------
# 8. Visualization - Inventory Levels
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    fixed_result["inventory"],
    label="Fixed Order"
)

plt.plot(
    conservative_result["inventory"],
    label="Conservative"
)

plt.plot(
    rl_result["inventory"],
    label="Model-Based RL"
)

plt.xlabel("Day")
plt.ylabel("Inventory")

plt.title(
    "Inventory Levels under Different Policies"
)

plt.legend()
plt.grid()

plt.show()


# -------------------------------------------------
# 9. Visualization - Demand
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    demand,
    label="Customer Demand"
)

plt.xlabel("Day")
plt.ylabel("Demand")

plt.title(
    "Synthetic Customer Demand Pattern"
)

plt.legend()
plt.grid()

plt.show()


# -------------------------------------------------
# 10. Visualization - Total Cost
# -------------------------------------------------

policy_names = list(results.keys())
cost_values = list(results.values())

plt.figure(figsize=(8, 5))

plt.bar(
    policy_names,
    cost_values
)

plt.xlabel("Policy")
plt.ylabel("Total Cost")

plt.title(
    "Inventory Policy Cost Comparison"
)

plt.grid(axis="y")

plt.show()


print("\n==============================================")
print("Experiment 32 Completed Successfully!")
print("==============================================")