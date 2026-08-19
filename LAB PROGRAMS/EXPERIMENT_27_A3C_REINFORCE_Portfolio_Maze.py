import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("===================================================")
print("EXPERIMENT 27")
print("A3C Portfolio Management + REINFORCE Maze")
print("===================================================")


# =========================================================
# PART 1: ACTOR-CRITIC / A3C PORTFOLIO MANAGEMENT
# =========================================================

print("\nPART 1: A3C Financial Portfolio Management")
print("-------------------------------------------")


class PortfolioEnvironment:

    def __init__(self):

        self.num_stocks = 3

        # Simulated daily returns
        self.returns = np.array([
            [0.02, 0.01, -0.01],
            [0.01, 0.03, 0.00],
            [-0.01, 0.02, 0.01],
            [0.03, -0.01, 0.02],
            [0.02, 0.01, 0.03],
            [-0.02, 0.03, 0.01],
            [0.01, 0.02, 0.02],
            [0.03, 0.01, -0.01],
            [0.02, 0.03, 0.01],
            [0.01, 0.02, 0.03]
        ], dtype=np.float32)

        self.reset()

    def reset(self):

        self.day = 0

        self.portfolio = np.array(
            [1 / 3, 1 / 3, 1 / 3],
            dtype=np.float32
        )

        return self.portfolio.copy()

    def step(self, action):

        action = np.array(action)

        # Normalize portfolio allocation
        action = np.maximum(action, 0)

        if np.sum(action) == 0:
            action = np.ones(3) / 3
        else:
            action = action / np.sum(action)

        daily_return = np.dot(
            action,
            self.returns[self.day]
        )

        # Risk penalty
        risk = np.std(
            self.returns[self.day]
        )

        reward = daily_return - 0.5 * risk

        self.portfolio = action

        self.day += 1

        done = self.day >= len(self.returns)

        return (
            self.portfolio.copy(),
            reward,
            done
        )


# ---------------------------------------------------------
# Actor Network
# ---------------------------------------------------------

class Actor(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.dense2 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            3,
            activation="softmax"
        )

    def call(self, state):

        x = self.dense1(state)
        x = self.dense2(x)

        return self.output_layer(x)


# ---------------------------------------------------------
# Critic Network
# ---------------------------------------------------------

class Critic(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.dense2 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            1
        )

    def call(self, state):

        x = self.dense1(state)
        x = self.dense2(x)

        return self.output_layer(x)


actor = Actor()
critic = Critic()

actor_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.002
)


# Build networks

dummy_state = tf.zeros((1, 3))

actor(dummy_state)
critic(dummy_state)


gamma = 0.95

episodes = 100

portfolio_rewards = []


# ---------------------------------------------------------
# A3C-style Actor-Critic Training
# ---------------------------------------------------------

for episode in range(episodes):

    env = PortfolioEnvironment()

    state = env.reset()

    total_reward = 0

    while True:

        state_tensor = tf.convert_to_tensor(
            [state],
            dtype=tf.float32
        )

        with tf.GradientTape(
            persistent=True
        ) as tape:

            probabilities = actor(
                state_tensor
            )

            value = critic(
                state_tensor
            )

            # Sample action from policy
            action_index = tf.random.categorical(
                tf.math.log(probabilities),
                1
            )[0, 0]

            action = tf.one_hot(
                action_index,
                depth=3
            )

            next_state, reward, done = env.step(
                action.numpy()
            )

            next_state_tensor = tf.convert_to_tensor(
                [next_state],
                dtype=tf.float32
            )

            next_value = critic(
                next_state_tensor
            )

            target = reward

            if not done:
                target += gamma * next_value[0, 0]

            advantage = target - value[0, 0]

            selected_probability = probabilities[
                0,
                action_index
            ]

            actor_loss = (
                -tf.math.log(
                    selected_probability + 1e-8
                )
                * tf.stop_gradient(advantage)
            )

            critic_loss = tf.square(
                advantage
            )

        actor_gradients = tape.gradient(
            actor_loss,
            actor.trainable_variables
        )

        critic_gradients = tape.gradient(
            critic_loss,
            critic.trainable_variables
        )

        actor_optimizer.apply_gradients(
            zip(
                actor_gradients,
                actor.trainable_variables
            )
        )

        critic_optimizer.apply_gradients(
            zip(
                critic_gradients,
                critic.trainable_variables
            )
        )

        del tape

        state = next_state

        total_reward += reward

        if done:
            break

    portfolio_rewards.append(
        total_reward
    )

    if (episode + 1) % 10 == 0:

        print(
            "Episode:",
            episode + 1,
            "Average Reward:",
            round(
                np.mean(
                    portfolio_rewards[-10:]
                ),
                4
            )
        )


# ---------------------------------------------------------
# Display Final Portfolio
# ---------------------------------------------------------

state = np.array(
    [1 / 3, 1 / 3, 1 / 3],
    dtype=np.float32
)

state_tensor = tf.convert_to_tensor(
    [state],
    dtype=tf.float32
)

final_allocation = actor(
    state_tensor
).numpy()[0]

print("\nFinal Portfolio Allocation")

print(
    "Stock 1:",
    round(final_allocation[0] * 100, 2),
    "%"
)

print(
    "Stock 2:",
    round(final_allocation[1] * 100, 2),
    "%"
)

print(
    "Stock 3:",
    round(final_allocation[2] * 100, 2),
    "%"
)


# =========================================================
# PART 2: REINFORCE MAZE NAVIGATION
# =========================================================

print("\n\nPART 2: REINFORCE Maze Navigation")
print("----------------------------------")


# Maze:
#
# S = Start
# 0 = Free cell
# X = Wall
# G = Goal
#
# S 0 0 0
# X X 0 X
# 0 0 0 0
# X X X 0
# 0 0 0 G


maze = np.array([
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]
])


start = (0, 0)
goal = (4, 3)

rows = 5
cols = 4

# Actions:
# 0 = UP
# 1 = DOWN
# 2 = LEFT
# 3 = RIGHT

actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


class MazeEnvironment:

    def __init__(self):

        self.reset()

    def reset(self):

        self.position = start

        return self.position

    def step(self, action):

        row, col = self.position

        dr, dc = actions[action]

        new_row = row + dr
        new_col = col + dc

        # Outside maze
        if (
            new_row < 0
            or new_row >= rows
            or new_col < 0
            or new_col >= cols
        ):

            reward = -5

            return self.position, reward, False

        # Wall
        if maze[new_row, new_col] == 1:

            reward = -5

            return self.position, reward, False

        # Move
        self.position = (
            new_row,
            new_col
        )

        # Goal
        if self.position == goal:

            reward = 20

            return self.position, reward, True

        # Normal movement
        reward = -1

        return self.position, reward, False


# ---------------------------------------------------------
# Policy Network
# ---------------------------------------------------------

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


policy = PolicyNetwork()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.01
)

policy_rewards = []


def state_vector(position):

    vector = np.zeros(
        rows * cols,
        dtype=np.float32
    )

    index = (
        position[0] * cols
        + position[1]
    )

    vector[index] = 1

    return vector


# ---------------------------------------------------------
# REINFORCE Training
# ---------------------------------------------------------

episodes = 500

for episode in range(episodes):

    env = MazeEnvironment()

    state = env.reset()

    episode_states = []
    episode_actions = []
    episode_rewards = []

    for step in range(50):

        state_vec = state_vector(
            state
        )

        state_tensor = tf.convert_to_tensor(
            [state_vec],
            dtype=tf.float32
        )

        probabilities = policy(
            state_tensor
        )[0]

        action = np.random.choice(
            4,
            p=probabilities.numpy()
        )

        next_state, reward, done = env.step(
            action
        )

        episode_states.append(
            state_vec
        )

        episode_actions.append(
            action
        )

        episode_rewards.append(
            reward
        )

        state = next_state

        if done:
            break

    # -----------------------------------------------------
    # Calculate discounted returns
    # -----------------------------------------------------

    returns = []

    G = 0

    for reward in reversed(
        episode_rewards
    ):

        G = reward + gamma * G

        returns.insert(
            0,
            G
        )

    returns = np.array(
        returns,
        dtype=np.float32
    )

    # Normalize returns

    if len(returns) > 1:

        returns = (
            returns - np.mean(returns)
        ) / (
            np.std(returns) + 1e-8
        )

    # -----------------------------------------------------
    # Policy gradient update
    # -----------------------------------------------------

    with tf.GradientTape() as tape:

        states_tensor = tf.convert_to_tensor(
            episode_states,
            dtype=tf.float32
        )

        probabilities = policy(
            states_tensor
        )

        loss = 0

        for i in range(
            len(episode_actions)
        ):

            action = episode_actions[i]

            probability = probabilities[
                i,
                action
            ]

            loss += (
                -tf.math.log(
                    probability + 1e-8
                )
                * returns[i]
            )

        loss /= len(
            episode_actions
        )

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

    total_reward = sum(
        episode_rewards
    )

    policy_rewards.append(
        total_reward
    )

    if (episode + 1) % 50 == 0:

        print(
            "Episode:",
            episode + 1,
            "Average Reward:",
            round(
                np.mean(
                    policy_rewards[-50:]
                ),
                2
            )
        )


# =========================================================
# TEST LEARNED MAZE POLICY
# =========================================================

print("\nLearned Maze Path")
print("-----------------")

env = MazeEnvironment()

state = env.reset()

path = [state]

for step in range(50):

    state_vec = state_vector(
        state
    )

    state_tensor = tf.convert_to_tensor(
        [state_vec],
        dtype=tf.float32
    )

    probabilities = policy(
        state_tensor
    )[0]

    action = np.argmax(
        probabilities.numpy()
    )

    next_state, reward, done = env.step(
        action
    )

    state = next_state

    path.append(state)

    if done:
        break


print("Path:")

for position in path:

    print(position)

if state == goal:

    print("\nGoal reached successfully!")

else:

    print("\nGoal not reached.")


# =========================================================
# PLOTS
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    portfolio_rewards
)

plt.xlabel("Episode")
plt.ylabel("Portfolio Reward")
plt.title("A3C Portfolio Training")
plt.grid()

plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    policy_rewards
)

plt.xlabel("Episode")
plt.ylabel("Maze Reward")
plt.title("REINFORCE Maze Training")
plt.grid()

plt.show()


print("\n===================================================")
print("Experiment 27 Completed Successfully!")
print("===================================================")