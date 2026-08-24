import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("==============================================")
print("EXPERIMENT 33 - VALUE EQUIVALENCE PREDICTION")
print("PORTFOLIO PERFORMANCE ANALYSIS")
print("==============================================")


# -------------------------------------------------
# 1. Generate Historical Financial Data
# -------------------------------------------------

np.random.seed(42)

days = 500

# Simulated daily returns
stock_returns = np.random.normal(0.0008, 0.02, days)
bond_returns = np.random.normal(0.0003, 0.008, days)
gold_returns = np.random.normal(0.0004, 0.012, days)

# Market indicators
market_return = (
    0.5 * stock_returns
    + 0.3 * bond_returns
    + 0.2 * gold_returns
)

# -------------------------------------------------
# 2. Create Machine Learning Dataset
# -------------------------------------------------

X = []
y = []

window = 10

for i in range(window, days):

    # Recent historical returns
    features = [
        np.mean(stock_returns[i-window:i]),
        np.std(stock_returns[i-window:i]),

        np.mean(bond_returns[i-window:i]),
        np.std(bond_returns[i-window:i]),

        np.mean(gold_returns[i-window:i]),
        np.std(gold_returns[i-window:i]),

        market_return[i-1]
    ]

    # Future portfolio return
    future_return = (
        0.5 * stock_returns[i]
        + 0.3 * bond_returns[i]
        + 0.2 * gold_returns[i]
    )

    X.append(features)
    y.append(future_return)

X = np.array(X)
y = np.array(y)

print("\nHistorical dataset created.")
print("Number of samples:", len(X))


# -------------------------------------------------
# 3. Split Data
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------------------------
# 4. Train Value Prediction Model
# -------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nValue prediction model trained successfully.")


# -------------------------------------------------
# 5. Evaluate Prediction Model
# -------------------------------------------------

predictions = model.predict(X_test)

mse = mean_squared_error(
    y_test,
    predictions
)

print(
    "Mean Squared Error:",
    round(mse, 8)
)


# -------------------------------------------------
# 6. Define Portfolio Strategies
# -------------------------------------------------

strategies = {

    "Conservative": {
        "Stock": 0.20,
        "Bond": 0.60,
        "Gold": 0.20
    },

    "Balanced": {
        "Stock": 0.50,
        "Bond": 0.30,
        "Gold": 0.20
    },

    "Aggressive": {
        "Stock": 0.70,
        "Bond": 0.10,
        "Gold": 0.20
    }
}


# -------------------------------------------------
# 7. Calculate Portfolio Performance
# -------------------------------------------------

portfolio_results = {}

initial_value = 100000

for name, weights in strategies.items():

    portfolio_return = (
        weights["Stock"] * stock_returns
        + weights["Bond"] * bond_returns
        + weights["Gold"] * gold_returns
    )

    portfolio_value = (
        initial_value
        * np.cumprod(1 + portfolio_return)
    )

    total_return = (
        portfolio_value[-1]
        - initial_value
    ) / initial_value * 100

    risk = np.std(portfolio_return) * np.sqrt(252) * 100

    predicted_return = np.mean(
        model.predict(X_test)
    ) * 100

    portfolio_results[name] = {
        "values": portfolio_value,
        "return": total_return,
        "risk": risk,
        "predicted": predicted_return
    }


# -------------------------------------------------
# 8. Display Results
# -------------------------------------------------

print("\n==============================================")
print("PORTFOLIO PERFORMANCE")
print("==============================================")

for name, result in portfolio_results.items():

    print("\nStrategy:", name)

    print(
        "Final Portfolio Value: ₹",
        round(result["values"][-1], 2)
    )

    print(
        "Historical Return:",
        round(result["return"], 2),
        "%"
    )

    print(
        "Annualized Risk:",
        round(result["risk"], 2),
        "%"
    )

    print(
        "Predicted Value Equivalence:",
        round(result["predicted"], 4),
        "%"
    )


# -------------------------------------------------
# 9. Find Best Strategy
# -------------------------------------------------

best_strategy = max(
    portfolio_results,
    key=lambda x:
    portfolio_results[x]["return"]
)

print("\n==============================================")
print("BEST INVESTMENT STRATEGY")
print("==============================================")

print(
    "Best Strategy:",
    best_strategy
)

print(
    "Highest Historical Return:",
    round(
        portfolio_results[
            best_strategy
        ]["return"],
        2
    ),
    "%"
)


# -------------------------------------------------
# 10. Portfolio Value Visualization
# -------------------------------------------------

plt.figure(figsize=(10, 6))

for name, result in portfolio_results.items():

    plt.plot(
        result["values"],
        label=name
    )

plt.xlabel("Trading Day")
plt.ylabel("Portfolio Value (₹)")

plt.title(
    "Portfolio Value Comparison"
)

plt.legend()
plt.grid()

plt.show()


# -------------------------------------------------
# 11. Return Comparison
# -------------------------------------------------

names = list(portfolio_results.keys())

returns = [
    portfolio_results[name]["return"]
    for name in names
]

plt.figure(figsize=(8, 5))

plt.bar(
    names,
    returns
)

plt.xlabel("Investment Strategy")
plt.ylabel("Return (%)")

plt.title(
    "Investment Strategy Return Comparison"
)

plt.grid(axis="y")

plt.show()


# -------------------------------------------------
# 12. Risk Comparison
# -------------------------------------------------

risks = [
    portfolio_results[name]["risk"]
    for name in names
]

plt.figure(figsize=(8, 5))

plt.bar(
    names,
    risks
)

plt.xlabel("Investment Strategy")
plt.ylabel("Annualized Risk (%)")

plt.title(
    "Investment Risk Comparison"
)

plt.grid(axis="y")

plt.show()


print("\n==============================================")
print("Experiment 33 Completed Successfully!")
print("==============================================")