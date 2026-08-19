import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import random

print("Experiment 26 - DDPG Movie Recommendation System")
print("------------------------------------------------")


# ==========================================================
# 1. MOVIE RECOMMENDATION ENVIRONMENT
# ==========================================================

class MovieEnvironment:

    def __init__(self):
        self.num_movies = 5
        self.reset()

    def reset(self):

        # User preference:
        # Action, Comedy, Drama
        self.user_preference = np.array([
            0.7,
            0.5,
            0.3
        ], dtype=np.float32)

        return self.user_preference.copy()

    def step(self, action):

        # Convert action into movie preference
        action = np.clip(action, 0, 1)

        # Five movies with different characteristics
        movies = np.array([
            [0.9, 0.2, 0.1],   # Movie 1
            [0.2, 0.9, 0.2],   # Movie 2
            [0.3, 0.2, 0.9],   # Movie 3
            [0.8, 0.6, 0.2],   # Movie 4
            [0.4, 0.5, 0.8]    # Movie 5
        ])

        # Find movie closest to selected action
        distances = np.linalg.norm(
            movies - action,
            axis=1
        )

        movie_index = np.argmin(distances)

        selected_movie = movies[movie_index]

        # User feedback based on preference similarity
        similarity = np.dot(
            self.user_preference,
            selected_movie
        )

        reward = similarity * 10

        # Small random feedback
        reward += np.random.normal(0, 0.2)

        # Update user preference
        self.user_preference = (
            0.9 * self.user_preference
            + 0.1 * selected_movie
        )

        self.user_preference = np.clip(
            self.user_preference,
            0,
            1
        )

        done = False

        return (
            self.user_preference.copy(),
            reward,
            done,
            movie_index
        )


# ==========================================================
# 2. ACTOR NETWORK
# ==========================================================

class Actor(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.layer1 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.layer2 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            3,
            activation="sigmoid"
        )

    def call(self, state):

        x = self.layer1(state)
        x = self.layer2(x)

        return self.output_layer(x)


# ==========================================================
# 3. CRITIC NETWORK
# ==========================================================

class Critic(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.layer1 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.layer2 = tf.keras.layers.Dense(
            64,
            activation="relu"
        )

        self.output_layer = tf.keras.layers.Dense(
            1
        )

    def call(self, state, action):

        x = tf.concat(
            [state, action],
            axis=1
        )

        x = self.layer1(x)
        x = self.layer2(x)

        return self.output_layer(x)


# ==========================================================
# 4. REPLAY BUFFER
# ==========================================================

class ReplayBuffer:

    def __init__(self, size=5000):

        self.buffer = deque(
            maxlen=size
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = np.array(
            [x[0] for x in batch],
            dtype=np.float32
        )

        actions = np.array(
            [x[1] for x in batch],
            dtype=np.float32
        )

        rewards = np.array(
            [x[2] for x in batch],
            dtype=np.float32
        )

        next_states = np.array(
            [x[3] for x in batch],
            dtype=np.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states
        )

    def __len__(self):

        return len(self.buffer)


# ==========================================================
# 5. CREATE ENVIRONMENT AND NETWORKS
# ==========================================================

env = MovieEnvironment()

actor = Actor()
critic = Critic()

target_actor = Actor()
target_critic = Critic()


# Build networks
dummy_state = tf.zeros((1, 3))
dummy_action = tf.zeros((1, 3))

actor(dummy_state)
critic(dummy_state, dummy_action)

target_actor(dummy_state)
target_critic(
    dummy_state,
    dummy_action
)


# Copy initial weights
target_actor.set_weights(
    actor.get_weights()
)

target_critic.set_weights(
    critic.get_weights()
)


actor_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.002
)


# ==========================================================
# 6. DDPG PARAMETERS
# ==========================================================

gamma = 0.95
tau = 0.005

episodes = 100
steps_per_episode = 10

batch_size = 32

buffer = ReplayBuffer()

reward_history = []


# ==========================================================
# 7. SOFT UPDATE
# ==========================================================

def soft_update(target, source):

    for target_weight, source_weight in zip(
        target.weights,
        source.weights
    ):

        target_weight.assign(
            tau * source_weight
            + (1 - tau) * target_weight
        )


# ==========================================================
# 8. TRAINING FUNCTION
# ==========================================================

def train_ddpg():

    if len(buffer) < batch_size:
        return

    (
        states,
        actions,
        rewards,
        next_states
    ) = buffer.sample(batch_size)

    states = tf.convert_to_tensor(
        states,
        dtype=tf.float32
    )

    actions = tf.convert_to_tensor(
        actions,
        dtype=tf.float32
    )

    rewards = tf.convert_to_tensor(
        rewards,
        dtype=tf.float32
    )

    next_states = tf.convert_to_tensor(
        next_states,
        dtype=tf.float32
    )

    # ------------------------------------------------------
    # Critic update
    # ------------------------------------------------------

    with tf.GradientTape() as tape:

        next_actions = target_actor(
            next_states
        )

        target_values = target_critic(
            next_states,
            next_actions
        )

        y = rewards + gamma * tf.squeeze(
            target_values,
            axis=1
        )

        current_values = tf.squeeze(
            critic(
                states,
                actions
            ),
            axis=1
        )

        critic_loss = tf.reduce_mean(
            tf.square(
                y - current_values
            )
        )

    critic_gradients = tape.gradient(
        critic_loss,
        critic.trainable_variables
    )

    critic_optimizer.apply_gradients(
        zip(
            critic_gradients,
            critic.trainable_variables
        )
    )

    # ------------------------------------------------------
    # Actor update
    # ------------------------------------------------------

    with tf.GradientTape() as tape:

        predicted_actions = actor(
            states
        )

        actor_loss = -tf.reduce_mean(
            critic(
                states,
                predicted_actions
            )
        )

    actor_gradients = tape.gradient(
        actor_loss,
        actor.trainable_variables
    )

    actor_optimizer.apply_gradients(
        zip(
            actor_gradients,
            actor.trainable_variables
        )
    )

    # ------------------------------------------------------
    # Update target networks
    # ------------------------------------------------------

    soft_update(
        target_actor,
        actor
    )

    soft_update(
        target_critic,
        critic
    )


# ==========================================================
# 9. TRAIN DDPG AGENT
# ==========================================================

print("\nTraining DDPG Agent")
print("-------------------")

for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(
        steps_per_episode
    ):

        state_tensor = tf.convert_to_tensor(
            [state],
            dtype=tf.float32
        )

        action = actor(
            state_tensor
        ).numpy()[0]

        # Exploration noise
        noise = np.random.normal(
            0,
            0.1,
            size=3
        )

        action = np.clip(
            action + noise,
            0,
            1
        )

        (
            next_state,
            reward,
            done,
            movie_index
        ) = env.step(action)

        buffer.add(
            state,
            action,
            reward,
            next_state
        )

        train_ddpg()

        state = next_state

        total_reward += reward

    reward_history.append(
        total_reward
    )

    if (episode + 1) % 10 == 0:

        average_reward = np.mean(
            reward_history[-10:]
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
# 10. EVALUATE RECOMMENDATION POLICY
# ==========================================================

print("\nFinal Movie Recommendations")
print("---------------------------")

state = env.reset()

movie_names = [
    "Action Movie",
    "Comedy Movie",
    "Drama Movie",
    "Action-Comedy Movie",
    "Comedy-Drama Movie"
]

total_reward = 0

for step in range(10):

    state_tensor = tf.convert_to_tensor(
        [state],
        dtype=tf.float32
    )

    action = actor(
        state_tensor
    ).numpy()[0]

    (
        next_state,
        reward,
        done,
        movie_index
    ) = env.step(action)

    print(
        "Step:",
        step + 1,
        "| Recommended:",
        movie_names[movie_index],
        "| User Feedback:",
        round(reward, 2)
    )

    total_reward += reward

    state = next_state


print(
    "\nTotal Recommendation Reward:",
    round(total_reward, 2)
)


# ==========================================================
# 11. PLOT TRAINING PERFORMANCE
# ==========================================================

plt.figure(figsize=(8, 5))

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
    "DDPG Movie Recommendation Training"
)

plt.grid()

plt.show()


print(
    "\nExperiment 26 Completed Successfully!"
)