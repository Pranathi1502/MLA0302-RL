import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("Experiment 25 - TRPO Smart Grid Energy Management")
print("------------------------------------------------")

# ------------------------------------------------
# Smart Grid Environment
# ------------------------------------------------

class SmartGrid:

    def __init__(self):
        self.days = 24
        self.reset()

    def reset(self):
        self.hour = 0
        self.battery = 50.0
        return self.get_state()

    def get_state(self):

        demand = self.get_demand(self.hour)
        solar = self.get_solar(self.hour)

        return np.array([
            demand / 100,
            solar / 100,
            self.battery / 100
        ], dtype=np.float32)

    def get_demand(self, hour):

        demand = [
            45, 42, 40, 38, 37, 40,
            50, 60, 70, 75, 72, 68,
            65, 63, 66, 70, 78, 85,
            90, 88, 82, 75, 65, 55
        ]

        return demand[hour]

    def get_solar(self, hour):

        solar = [
            0, 0, 0, 0, 0, 0,
            10, 20, 35, 50, 65, 80,
            90, 95, 90, 75, 55, 30,
            10, 0, 0, 0, 0, 0
        ]

        return solar[hour]

    def step(self, action):

        demand = self.get_demand(self.hour)
        solar = self.get_solar(self.hour)

        # Action represents battery charging/discharging
        # -1 = discharge
        #  0 = no battery action
        # +1 = charge

        battery_change = action * 10

        # Available energy from solar
        available_energy = solar + max(0, -battery_change)

        # Battery charging
        if battery_change > 0:
            charge = min(
                battery_change,
                100 - self.battery
            )
            self.battery += charge

        # Battery discharging
        elif battery_change < 0:
            discharge = min(
                -battery_change,
                self.battery
            )
            self.battery -= discharge

        # Energy supplied by solar and battery
        supplied = solar

        if battery_change < 0:
            supplied += -battery_change

        # Grid energy required
        grid_energy = max(
            0,
            demand - supplied
        )

        # Electricity price
        price = 5 if self.hour >= 17 and self.hour <= 21 else 3

        # Cost of grid energy
        cost = grid_energy * price

        # Penalty for energy imbalance
        imbalance = abs(demand - supplied)

        penalty = imbalance * 0.5

        reward = -(cost + penalty)

        self.hour += 1

        done = self.hour >= self.days

        if done:
            next_state = np.zeros(3)
        else:
            next_state = self.get_state()

        return next_state, reward, done


# ------------------------------------------------
# Policy Network
# ------------------------------------------------

class PolicyNetwork(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.dense1 = tf.keras.layers.Dense(
            32,
            activation="tanh"
        )

        self.dense2 = tf.keras.layers.Dense(
            32,
            activation="tanh"
        )

        self.output_layer = tf.keras.layers.Dense(
            3,
            activation="softmax"
        )

    def call(self, state):

        x = self.dense1(state)
        x = self.dense2(x)

        return self.output_layer(x)


policy = PolicyNetwork()


# ------------------------------------------------
# Value Network
# ------------------------------------------------

value_network = tf.keras.Sequential([
    tf.keras.layers.Dense(
        32,
        activation="tanh",
        input_shape=(3,)
    ),

    tf.keras.layers.Dense(
        32,
        activation="tanh"
    ),

    tf.keras.layers.Dense(1)
])


# ------------------------------------------------
# Select Action
# ------------------------------------------------

def select_action(state):

    state = np.expand_dims(
        state,
        axis=0
    )

    probabilities = policy(state).numpy()[0]

    action_index = np.random.choice(
        3,
        p=probabilities
    )

    actions = [-1, 0, 1]

    return actions[action_index], action_index, probabilities


# ------------------------------------------------
# Generate Episode
# ------------------------------------------------

def generate_episode():

    environment = SmartGrid()

    state = environment.reset()

    states = []
    actions = []
    rewards = []

    done = False

    while not done:

        action, action_index, probabilities = select_action(state)

        next_state, reward, done = environment.step(
            action
        )

        states.append(state)
        actions.append(action_index)
        rewards.append(reward)

        state = next_state

    return (
        np.array(states),
        np.array(actions),
        np.array(rewards)
    )


# ------------------------------------------------
# TRPO-style Policy Optimization
# ------------------------------------------------

learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate
)

gamma = 0.95

episodes = 100

reward_history = []


print("\nTraining TRPO-style Agent...")
print("-----------------------------")


for episode in range(episodes):

    states, actions, rewards = generate_episode()

    # Calculate discounted returns
    returns = []

    total = 0

    for reward in reversed(rewards):

        total = reward + gamma * total

        returns.insert(
            0,
            total
        )

    returns = np.array(
        returns,
        dtype=np.float32
    )

    # Normalize returns
    returns = (
        returns - returns.mean()
    ) / (
        returns.std() + 1e-8
    )

    states_tensor = tf.convert_to_tensor(
        states,
        dtype=tf.float32
    )

    actions_tensor = tf.convert_to_tensor(
        actions,
        dtype=tf.int32
    )

    returns_tensor = tf.convert_to_tensor(
        returns,
        dtype=tf.float32
    )

    # Policy gradient update
    with tf.GradientTape() as tape:

        probabilities = policy(
            states_tensor
        )

        action_probabilities = tf.reduce_sum(
            probabilities
            * tf.one_hot(
                actions_tensor,
                depth=3
            ),
            axis=1
        )

        log_probabilities = tf.math.log(
            action_probabilities + 1e-8
        )

        loss = -tf.reduce_mean(
            log_probabilities
            * returns_tensor
        )

    gradients = tape.gradient(
        loss,
        policy.trainable_variables
    )

    # Clip gradients to keep updates stable
    gradients = [
        tf.clip_by_norm(g, 0.5)
        for g in gradients
    ]

    optimizer.apply_gradients(
        zip(
            gradients,
            policy.trainable_variables
        )
    )

    episode_reward = np.sum(rewards)

    reward_history.append(
        episode_reward
    )

    if (episode + 1) % 10 == 0:

        average_reward = np.mean(
            reward_history[-10:]
        )

        print(
            "Episode:",
            episode + 1,
            "| Average Reward:",
            round(average_reward, 2)
        )


# ------------------------------------------------
# Evaluate Trained Policy
# ------------------------------------------------

print("\nEvaluating Trained Policy...")
print("----------------------------")

environment = SmartGrid()

state = environment.reset()

done = False

total_reward = 0

battery_history = []
grid_history = []
demand_history = []
solar_history = []

while not done:

    state_input = np.expand_dims(
        state,
        axis=0
    )

    probabilities = policy(
        state_input
    ).numpy()[0]

    action_index = np.argmax(
        probabilities
    )

    actions = [-1, 0, 1]

    action = actions[action_index]

    demand = environment.get_demand(
        environment.hour
    )

    solar = environment.get_solar(
        environment.hour
    )

    next_state, reward, done = environment.step(
        action
    )

    total_reward += reward

    battery_history.append(
        environment.battery
    )

    demand_history.append(
        demand
    )

    solar_history.append(
        solar
    )

    # Approximate grid requirement
    grid_energy = max(
        0,
        demand - solar
    )

    grid_history.append(
        grid_energy
    )

    state = next_state


print(
    "\nTotal Evaluation Reward:",
    round(total_reward, 2)
)

print(
    "Final Battery Level:",
    round(environment.battery, 2)
)

print(
    "Average Grid Energy:",
    round(
        np.mean(grid_history),
        2
    )
)


# ------------------------------------------------
# Plot Training Reward
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    reward_history
)

plt.title(
    "TRPO Smart Grid Training Performance"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.grid()

plt.show()


# ------------------------------------------------
# Plot Energy Management
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    demand_history,
    label="Demand"
)

plt.plot(
    solar_history,
    label="Solar Production"
)

plt.plot(
    grid_history,
    label="Grid Energy"
)

plt.xlabel(
    "Hour"
)

plt.ylabel(
    "Energy"
)

plt.title(
    "Smart Grid Energy Management"
)

plt.legend()

plt.grid()

plt.show()


print(
    "\nExperiment 25 Completed Successfully!"
)