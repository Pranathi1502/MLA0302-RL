import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("==============================================")
print("EXPERIMENT 29 - A2C AUTONOMOUS VEHICLE RACING")
print("==============================================")


# =========================================================
# 1. RACING ENVIRONMENT
# =========================================================

class RacingEnvironment:

    def __init__(self):

        self.track_length = 100
        self.max_steps = 100

        self.reset()

    def reset(self):

        self.position = 0.0
        self.speed = 0.0
        self.lap_time = 0

        return self.get_state()

    def get_state(self):

        # State:
        # position, speed, remaining distance

        return np.array([
            self.position / self.track_length,
            self.speed / 10.0,
            (self.track_length - self.position)
            / self.track_length
        ], dtype=np.float32)

    def step(self, action):

        # Actions:
        # 0 = Brake
        # 1 = Maintain Speed
        # 2 = Accelerate

        if action == 0:

            self.speed -= 1.0

        elif action == 1:

            self.speed += 0.1

        elif action == 2:

            self.speed += 1.0

        # Speed limits

        self.speed = np.clip(
            self.speed,
            0,
            10
        )

        # Move vehicle

        self.position += self.speed

        self.lap_time += 1

        # Reward encourages high speed
        # but penalizes excessive speed

        reward = self.speed * 0.5

        if self.speed > 8:

            reward -= 1.0

        done = False

        # Finish lap

        if self.position >= self.track_length:

            reward += 100

            done = True

        # Maximum race duration

        if self.lap_time >= self.max_steps:

            done = True

        return (
            self.get_state(),
            reward,
            done
        )


# =========================================================
# 2. A2C NETWORK
# =========================================================

class A2CNetwork(tf.keras.Model):

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

        # Actor

        self.actor = tf.keras.layers.Dense(
            3,
            activation="softmax"
        )

        # Critic

        self.critic = tf.keras.layers.Dense(
            1
        )

    def call(self, state):

        x = self.dense1(state)

        x = self.dense2(x)

        action_probabilities = self.actor(x)

        value = self.critic(x)

        return action_probabilities, value


# =========================================================
# 3. CREATE A2C AGENT
# =========================================================

agent = A2CNetwork()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

dummy_state = tf.zeros(
    (1, 3),
    dtype=tf.float32
)

agent(dummy_state)


# =========================================================
# 4. TRAINING PARAMETERS
# =========================================================

gamma = 0.99

episodes = 300

reward_history = []

lap_time_history = []


# =========================================================
# 5. A2C TRAINING
# =========================================================

print("\nTraining A2C Racing Agent")
print("------------------------")

for episode in range(episodes):

    environment = RacingEnvironment()

    state = environment.reset()

    total_reward = 0

    states = []
    actions = []
    rewards = []

    done = False

    while not done:

        state_tensor = tf.convert_to_tensor(
            [state],
            dtype=tf.float32
        )

        probabilities, value = agent(
            state_tensor
        )

        probabilities = probabilities[0]

        value = value[0, 0]

        # Select action from policy

        action = np.random.choice(
            3,
            p=probabilities.numpy()
        )

        next_state, reward, done = (
            environment.step(action)
        )

        states.append(state)

        actions.append(action)

        rewards.append(reward)

        state = next_state

        total_reward += reward

    # =====================================================
    # Calculate discounted returns
    # =====================================================

    returns = []

    G = 0

    for reward in reversed(rewards):

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

    # =====================================================
    # Update Actor and Critic
    # =====================================================

    with tf.GradientTape() as tape:

        states_tensor = tf.convert_to_tensor(
            states,
            dtype=tf.float32
        )

        probabilities, values = agent(
            states_tensor
        )

        values = tf.squeeze(
            values
        )

        actor_loss = 0.0

        critic_loss = 0.0

        for i in range(
            len(actions)
        ):

            action = actions[i]

            probability = probabilities[
                i,
                action
            ]

            advantage = (
                returns[i]
                - values[i]
            )

            actor_loss += (
                -tf.math.log(
                    probability + 1e-8
                )
                * tf.stop_gradient(
                    advantage
                )
            )

            critic_loss += tf.square(
                advantage
            )

        actor_loss /= len(actions)

        critic_loss /= len(actions)

        total_loss = (
            actor_loss
            + 0.5 * critic_loss
        )

    gradients = tape.gradient(
        total_loss,
        agent.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            agent.trainable_variables
        )
    )

    reward_history.append(
        total_reward
    )

    lap_time_history.append(
        environment.lap_time
    )

    if (episode + 1) % 25 == 0:

        avg_reward = np.mean(
            reward_history[-25:]
        )

        avg_lap_time = np.mean(
            lap_time_history[-25:]
        )

        print(
            "Episode:",
            episode + 1,
            "| Average Reward:",
            round(avg_reward, 2),
            "| Average Lap Time:",
            round(avg_lap_time, 2)
        )


# =========================================================
# 6. TEST TRAINED AGENT
# =========================================================

print("\n==============================================")
print("FINAL RACING PERFORMANCE")
print("==============================================")

environment = RacingEnvironment()

state = environment.reset()

total_reward = 0

action_names = [
    "BRAKE",
    "MAINTAIN SPEED",
    "ACCELERATE"
]

done = False

while not done:

    state_tensor = tf.convert_to_tensor(
        [state],
        dtype=tf.float32
    )

    probabilities, value = agent(
        state_tensor
    )

    action = np.argmax(
        probabilities.numpy()[0]
    )

    next_state, reward, done = (
        environment.step(action)
    )

    print(
        "Time:",
        environment.lap_time,
        "| Position:",
        round(environment.position, 2),
        "| Speed:",
        round(environment.speed, 2),
        "| Action:",
        action_names[action],
        "| Reward:",
        round(reward, 2)
    )

    state = next_state

    total_reward += reward


# =========================================================
# 7. FINAL RESULTS
# =========================================================

print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

print(
    "Lap Time:",
    environment.lap_time
)

print(
    "Final Position:",
    round(
        environment.position,
        2
    )
)

print(
    "Total Reward:",
    round(
        total_reward,
        2
    )
)

if environment.position >= environment.track_length:

    print("Race Result: LAP COMPLETED")

else:

    print("Race Result: LAP NOT COMPLETED")


# =========================================================
# 8. TRAINING GRAPH
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    reward_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "A2C Autonomous Racing Training"
)

plt.grid()

plt.show()


# =========================================================
# 9. LAP TIME GRAPH
# =========================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    lap_time_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Lap Time"
)

plt.title(
    "A2C Racing Lap Time"
)

plt.grid()

plt.show()


print("\nExperiment 29 Completed Successfully!")