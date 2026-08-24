import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("==============================================")
print("EXPERIMENT 34 - MODEL-BASED RL")
print("DYNAMIC PRICING OPTIMIZATION")
print("==============================================")


# -------------------------------------------------
# 1. Generate Historical Sales Data
# -------------------------------------------------

np.random.seed(42)

days = 500

prices = np.random.uniform(50, 150, days)

# Market conditions
market_condition = np.random.uniform(0.8, 1.2, days)

# Customer demand model
base_demand = 120

demand = (
    base_demand
    - 0.65 * prices
    + 30 * market_condition
    + np.random.normal(0, 5, days)
)

# Demand cannot be negative
demand = np.maximum(demand, 1)

print("\nHistorical sales data generated.")
print("Number of samples:", days)


# -------------------------------------------------
# 2. Prepare Machine Learning Dataset
# -------------------------------------------------

X = np.column_stack(
    (
        prices,
        market_condition
    )
)

y = demand


# -------------------------------------------------
# 3. Split Dataset
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------------------------
# 4. Train Demand Prediction Model
# -------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Demand prediction model trained successfully.")


# -------------------------------------------------
# 5. Evaluate Prediction Model
# -------------------------------------------------

predicted_demand = model.predict(X_test)

mse = mean_squared_error(
    y_test,
    predicted_demand
)

print(
    "Mean Squared Error:",
    round(mse, 4)
)


# -------------------------------------------------
# 6. Model-Based Policy Optimization
# -------------------------------------------------

price_options = np.arange(
    50,
    151,
    5
)

policy_prices = []
policy_demands = []
policy_profits = []

for market in np.linspace(0.8, 1.2, days):

    best_price = None
    best_demand = 0
    best_profit = -float("inf")

    for price in price_options:

        state = np.array(
            [[price, market]]
        )

        predicted = model.predict(
            state
        )[0]

        # Profit = price × predicted demand
        profit = price * predicted

        if profit > best_profit:

            best_profit = profit
            best_price = price
            best_demand = predicted

    policy_prices.append(best_price)
    policy_demands.append(best_demand)
    policy_profits.append(best_profit)


# -------------------------------------------------
# 7. Display Policy Results
# -------------------------------------------------

print("\n==============================================")
print("MODEL-BASED PRICING POLICY")
print("==============================================")

average_price = np.mean(
    policy_prices
)

average_demand = np.mean(
    policy_demands
)

average_profit = np.mean(
    policy_profits
)

print(
    "Average Optimal Price:",
    round(average_price, 2)
)

print(
    "Average Predicted Demand:",
    round(average_demand, 2)
)

print(
    "Average Expected Profit:",
    round(average_profit, 2)
)


# -------------------------------------------------
# 8. Compare Fixed Pricing
# -------------------------------------------------

fixed_prices = [80, 100, 120]

print("\n==============================================")
print("PRICE STRATEGY COMPARISON")
print("==============================================")

strategy_profits = {}

for price in fixed_prices:

    total_profit = 0

    for market in np.linspace(
        0.8,
        1.2,
        days
    ):

        state = np.array(
            [[price, market]]
        )

        predicted = model.predict(
            state
        )[0]

        profit = price * predicted

        total_profit += profit

    average_profit_fixed = (
        total_profit / days
    )

    strategy_profits[
        "Fixed " + str(price)
    ] = average_profit_fixed

    print(
        "Fixed Price",
        price,
        "-> Average Profit:",
        round(
            average_profit_fixed,
            2
        )
    )


strategy_profits[
    "Model-Based RL"
] = average_profit


# -------------------------------------------------
# 9. Find Best Strategy
# -------------------------------------------------

best_strategy = max(
    strategy_profits,
    key=strategy_profits.get
)

print(
    "\nBest Pricing Strategy:",
    best_strategy
)

print(
    "Best Average Profit:",
    round(
        strategy_profits[best_strategy],
        2
    )
)


# -------------------------------------------------
# 10. Plot Historical Price vs Demand
# -------------------------------------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    prices,
    demand,
    alpha=0.5
)

plt.xlabel("Price")
plt.ylabel("Customer Demand")

plt.title(
    "Historical Price vs Customer Demand"
)

plt.grid()

plt.show()


# -------------------------------------------------
# 11. Plot Dynamic Prices
# -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    policy_prices
)

plt.xlabel("Time Period")
plt.ylabel("Optimal Price")

plt.title(
    "Dynamic Pricing Selected by Model-Based Policy"
)

plt.grid()

plt.show()


# -------------------------------------------------
# 12. Plot Profit Comparison
# -------------------------------------------------

strategy_names = list(
    strategy_profits.keys()
)

profit_values = list(
    strategy_profits.values()
)

plt.figure(figsize=(9, 5))

plt.bar(
    strategy_names,
    profit_values
)

plt.xlabel("Pricing Strategy")
plt.ylabel("Average Profit")

plt.title(
    "Pricing Strategy Profit Comparison"
)

plt.grid(axis="y")

plt.show()


print("\n==============================================")
print("Experiment 34 Completed Successfully!")
print("==============================================")