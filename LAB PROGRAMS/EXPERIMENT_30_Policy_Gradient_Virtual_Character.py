import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("==============================================")
print("EXPERIMENT 30 - POLICY GRADIENT")
print("VIRTUAL CHARACTER CONTENT CREATION")
print("==============================================")


# =========================================================
# 1. VIRTUAL WORLD ENVIRONMENT
# =========================================================

class VirtualWorld:

    def __init__(self):
        self.max_steps = 10
        self.reset()

    def reset(self):
        self.step_count = 0
        self.engagement = 0
        return np.array([0.0], dtype=np.float32)

    def step(self, action):

        # Actions:
        # 0 = Tell Story
        # 1 = Interactive Question
        # 2 = Funny Content
        # 3 = Exploration

        rewards = [6, 8, 7, 5]

        reward = rewards[action]

        # Small random variation in audience response
        reward += np.random.normal(0, 1)

        self.engagement += reward
        self.step_count += 1

        done = self.step_count >= self.max_steps

        next_state = np.array(
            [self.step_count / self.max_steps],
            dtype=np.float32
        )

        return next_state, reward, done


# =========================================================
# 2. POLICY NETWORK
# =========================================================

class PolicyNetwork(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            32,
            activation="relu"
        )

        self.dense2 = tf.keras.layers.Dense(
            32,
            activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            4,
            activation="softmax"
        )

    def call(self, state):

        x = self.dense1(state)
        x = self.dense2(x)

        return self.output_layer(x)


# =========================================================
# 3. CREATE POLICY
# =========================================================

policy = PolicyNetwork()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.01
)

# Build network

policy(
    tf.zeros(
        (1, 1),
        dtype=tf.float32
    )
)


# =========================================================
# 4. TRAINING PARAMETERS
# =========================================================

episodes = 300
gamma = 0.95

reward_history = []


action_names = [
    "Tell Story",
    "Interactive Question",
    "Funny Content",
    "Exploration"
]


# =========================================================
# 5. POLICY GRADIENT TRAINING
# =========================================================

print("\nTraining Virtual Character...")
print("--------------------------------")

for episode in range(episodes):

    environment = VirtualWorld()

    state = environment.reset()

    states = []
    actions = []
    rewards = []

    done = False

    while not done:

        state_tensor = tf.convert_to_tensor(
            [state],
            dtype=tf.float32
        )

        probabilities = policy(
            state_tensor
        )[0]

        # Select action according to policy

        action = np.random.choice(
            4,
            p=probabilities.numpy()
        )

        next_state, reward, done = (
            environment.step(action)
        )

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    # =====================================================
    # Calculate discounted returns
    # =====================================================

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)

    returns = np.array(
        returns,
        dtype=np.float32
    )

    # Normalize returns

    returns = (
        returns - np.mean(returns)
    ) / (
        np.std(returns) + 1e-8
    )

    # =====================================================
    # Policy Gradient Update
    # =====================================================

    with tf.GradientTape() as tape:

        states_tensor = tf.convert_to_tensor(
            states,
            dtype=tf.float32
        )

        probabilities = policy(
            states_tensor
        )

        loss = 0.0

        for i in range(len(actions)):

            action = actions[i]

            probability = probabilities[
                i, action
            ]

            loss += (
                -tf.math.log(
                    probability + 1e-8
                ) * returns[i]
            )

        loss /= len(actions)

    gradients = tape.gradient(
        loss,
        policy.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            policy.trainable_variables
        )
    )

    total_reward = sum(rewards)

    reward_history.append(
        total_reward
    )

    if (episode + 1) % 25 == 0:

        average_reward = np.mean(
            reward_history[-25:]
        )

        print(
            "Episode:",
            episode + 1,
            "| Average Engagement:",
            round(average_reward, 2)
        )


# =========================================================
# 6. TEST TRAINED CHARACTER
# =========================================================

print("\n==============================================")
print("TRAINED VIRTUAL CHARACTER")
print("==============================================")

environment = VirtualWorld()

state = environment.reset()

total_reward = 0

done = False

while not done:

    state_tensor = tf.convert_to_tensor(
        [state],
        dtype=tf.float32
    )

    probabilities = policy(
        state_tensor
    )[0]

    action = np.argmax(
        probabilities.numpy()
    )

    next_state, reward, done = (
        environment.step(action)
    )

    print(
        "Step:",
        environment.step_count,
        "| Content:",
        action_names[action],
        "| Engagement:",
        round(reward, 2)
    )

    total_reward += reward

    state = next_state


# =========================================================
# 7. FINAL RESULTS
# =========================================================

print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

print(
    "Total Audience Engagement:",
    round(total_reward, 2)
)

final_probabilities = policy(
    tf.convert_to_tensor(
        [[1.0]],
        dtype=tf.float32
    )
)[0].numpy()

best_action = np.argmax(
    final_probabilities
)

print(
    "Preferred Content:",
    action_names[best_action]
)

print(
    "Policy Probabilities:"
)

for i in range(4):

    print(
        action_names[i],
        "->",
        round(
            final_probabilities[i],
            3
        )
    )


# =========================================================
# 8. TRAINING GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    reward_history
)

plt.xlabel("Episode")
plt.ylabel("Audience Engagement")

plt.title(
    "Policy Gradient Training - Virtual Character"
)

plt.grid()

plt.show()


print("\nExperiment 30 Completed Successfully!")