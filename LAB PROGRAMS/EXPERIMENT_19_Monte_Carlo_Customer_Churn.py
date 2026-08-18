import pandas as pd
import random

print("Monte Carlo Customer Churn Policy Evaluation")
print("---------------------------------------------")

# Load dataset
data = pd.read_csv(
    r"C:\Users\SAMSUNG\Downloads\RL_Experiments_11_to_20_CSV\Experiment_19_Customer_Churn.csv"
)

print("\nCustomer Churn Dataset")
print(data)
print()

# -----------------------------------------
# Monte Carlo Policy Evaluation
# -----------------------------------------

returns = []

print("Monte Carlo Evaluation Results")
print("--------------------------------")

for i in range(len(data)):

    customer_id = data.loc[i, "Customer_ID"]
    tenure = data.loc[i, "Tenure"]
    churn = data.loc[i, "Churn"]

    # Policy:
    # Customers with low tenure are considered high churn risk
    if tenure <= 3:
        policy = "Retention"
    else:
        policy = "Normal Service"

    # Reward calculation
    if churn == 1:
        if policy == "Retention":
            reward = 10
        else:
            reward = -10
    else:
        if policy == "Retention":
            reward = 5
        else:
            reward = 8

    returns.append(reward)

    print("Customer ID:", customer_id)
    print("Tenure:", tenure)
    print("Churn:", churn)
    print("Policy:", policy)
    print("Reward:", reward)
    print()

# -----------------------------------------
# Monte Carlo Value Estimation
# -----------------------------------------

average_value = sum(returns) / len(returns)

print("Final Policy Evaluation")
print("-----------------------")
print("Number of Customers:", len(data))
print("Average Monte Carlo Value:", round(average_value, 2))

# -----------------------------------------
# Churn Analysis
# -----------------------------------------

churned = data["Churn"].sum()
total = len(data)

churn_rate = (churned / total) * 100

print("Total Churned Customers:", churned)
print("Churn Rate:", round(churn_rate, 2), "%")

# -----------------------------------------
# Final Analysis
# -----------------------------------------

print("\nAnalysis")
print("--------")

if churn_rate > 40:
    print("Result: High customer churn detected.")
    print("Recommendation: Increase retention efforts for high-risk customers.")
else:
    print("Result: Customer churn is under control.")
    print("Recommendation: Continue the current customer service policy.")