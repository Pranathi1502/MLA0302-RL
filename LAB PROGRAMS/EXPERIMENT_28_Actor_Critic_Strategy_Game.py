import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("==============================================")
print("EXPERIMENT 28 - ACTOR-CRITIC STRATEGY GAME")
print("==============================================")


# ==========================================================
# 1. GAME ENVIRONMENT
# ==========================================================

class StrategyGame:

    def __init__(self):
        self.reset()

    def reset(self):

        self.resources = 20
        self.structures = 0
        self.enemy_health = 100
        self.player_health = 100

        self.step_count = 0

        return self.get_state()

    def get_state(self):

        return np.array([
            self.resources / 100,
            self.structures / 5,
            self.enemy_health / 100,
            self.player_health / 100
        ], dtype=np.float32)

    def step(self, action):

        reward = -1

        # ---------------------------------------------
        # Action 0: Gather Resources
        # ---------------------------------------------

        if action == 0:

            self.resources += 15

            if self.resources > 100:
                self.resources = 100

            reward = 3

        # ---------------------------------------------
        # Action 1: Build Structure
        # ---------------------------------------------

        elif action == 1:

            if self.resources >= 20:

                self.resources -= 20

                self.structures += 1

                reward = 8

            else:

                reward = -5

        # ---------------------------------------------
        # Action 2: Attack Enemy
        # ---------------------------------------------

        elif action == 2:

            if self.resources >= 10:

                self.resources -= 10

                damage = np.random.randint(
                    10,
                    26
                )

                self.enemy_health -= damage

                reward = 5

                # Enemy counterattack

                enemy_damage = np.random.randint(
                    3,
                    11
                )

                self.player_health -= enemy_damage

            else:

                reward = -5

        self.step_count += 1

        # ---------------------------------------------
        # Bonus for destroying enemy
        # ---------------------------------------------

        if self.enemy_health <= 0:

            self.enemy_health = 0

            reward += 50

            done = True

        elif self.player_health <= 0:

            self.player_health = 0

            reward -= 50

            done = True

        elif self.step_count >= 50:

            done = True

        else:

            done = False

        return (
            self.get_state(),
            reward,
            done
        )


# ==========================================================
# 2. ACTOR NETWORK
# ==========================================================

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


# ==========================================================
# 3. CRITIC NETWORK
# ==========================================================

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


# ==========================================================
# 4. CREATE ACTOR AND CRITIC
# ==========================================================

actor = Actor()

critic = Critic()

actor_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.002
)

# Build networks

dummy_state = tf.zeros(
    (1, 4),
    dtype=tf.float32
)

actor(dummy_state)

critic(dummy_state)


# ==========================================================
# 5. TRAINING PARAMETERS
# ==========================================================

gamma = 0.95

episodes = 300

reward_history = []


# ==========================================================
# 6. ACTOR-CRITIC TRAINING
# ==========================================================

print("\nTraining Actor-Critic Agent")
print("--------------------------")

for episode in range(episodes):

    environment = StrategyGame()

    state = environment.reset()

    total_reward = 0

    for step in range(50):

        state_tensor = tf.convert_to_tensor(
            [state],
            dtype=tf.float32
        )

        # ---------------------------------------------
        # Actor chooses action
        # ---------------------------------------------

        action_probabilities = actor(
            state_tensor
        )[0]

        action = np.random.choice(
            3,
            p=action_probabilities.numpy()
        )

        # ---------------------------------------------
        # Environment response
        # ---------------------------------------------

        next_state, reward, done = environment.step(
            action
        )

        # ---------------------------------------------
        # Calculate target value
        # ---------------------------------------------

        next_state_tensor = tf.convert_to_tensor(
            [next_state],
            dtype=tf.float32
        )

        current_value = critic(
            state_tensor
        )[0, 0]

        next_value = critic(
            next_state_tensor
        )[0, 0]

        if done:

            target = tf.constant(
                reward,
                dtype=tf.float32
            )

        else:

            target = (
                reward
                + gamma * next_value
            )

        advantage = (
            target - current_value
        )

        # ---------------------------------------------
        # Actor update
        # ---------------------------------------------

        with tf.GradientTape() as actor_tape:

            probabilities = actor(
                state_tensor
            )[0]

            selected_probability = probabilities[
                action
            ]

            actor_loss = (
                -tf.math.log(
                    selected_probability
                    + 1e-8
                )
                * tf.stop_gradient(
                    advantage
                )
            )

        actor_gradients = actor_tape.gradient(
            actor_loss,
            actor.trainable_variables
        )

        actor_optimizer.apply_gradients(
            zip(
                actor_gradients,
                actor.trainable_variables
            )
        )

        # ---------------------------------------------
        # Critic update
        # ---------------------------------------------

        with tf.GradientTape() as critic_tape:

            value = critic(
                state_tensor
            )[0, 0]

            critic_loss = tf.square(
                target - value
            )

        critic_gradients = critic_tape.gradient(
            critic_loss,
            critic.trainable_variables
        )

        critic_optimizer.apply_gradients(
            zip(
                critic_gradients,
                critic.trainable_variables
            )
        )

        state = next_state

        total_reward += reward

        if done:
            break

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
            "| Average Reward:",
            round(
                average_reward,
                2
            )
        )


# ==========================================================
# 7. EVALUATE TRAINED AGENT
# ==========================================================

print("\n==============================================")
print("FINAL STRATEGY EVALUATION")
print("==============================================")

environment = StrategyGame()

state = environment.reset()

action_names = [
    "Gather Resources",
    "Build Structure",
    "Attack Enemy"
]

total_reward = 0

for step in range(20):

    state_tensor = tf.convert_to_tensor(
        [state],
        dtype=tf.float32
    )

    probabilities = actor(
        state_tensor
    )[0]

    action = np.argmax(
        probabilities.numpy()
    )

    next_state, reward, done = environment.step(
        action
    )

    print(
        "Step:",
        step + 1,
        "| Action:",
        action_names[action],
        "| Reward:",
        round(reward, 2),
        "| Resources:",
        environment.resources,
        "| Structures:",
        environment.structures,
        "| Enemy Health:",
        environment.enemy_health
    )

    state = next_state

    total_reward += reward

    if done:
        break


# ==========================================================
# 8. FINAL RESULTS
# ==========================================================

print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

print(
    "Resources:",
    environment.resources
)

print(
    "Structures Built:",
    environment.structures
)

print(
    "Enemy Health:",
    environment.enemy_health
)

print(
    "Player Health:",
    environment.player_health
)

print(
    "Total Reward:",
    round(total_reward, 2)
)

if environment.enemy_health == 0:

    print("Result: Enemy defeated!")

else:

    print("Result: Mission completed without defeating enemy.")


# ==========================================================
# 9. TRAINING GRAPH
# ==========================================================

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
    "Actor-Critic Strategy Game Training"
)

plt.grid()

plt.show()


print("\nExperiment 28 Completed Successfully!")