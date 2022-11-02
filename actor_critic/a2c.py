import gym
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def build_actor_critic(input_shape, action_space):
    input_layer = Input(shape=input_shape)

    x = Dense(128, activation="relu")(input_layer)

    v = Dense(128, activation="relu")(x)
    v = Dense(1, activation="linear")(v)

    pi = Dense(128, activation="relu")(x)
    pi = Dense(action_space, activation="softmax")(pi)

    model = Model(inputs=input_layer, outputs=[v, pi])

    return model


class A2CAgent:
    def __init__(self, env_name):
        self.max_episodes = 1000
        self.gamma = 0.99
        self.lr = 1e-2

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.episode_rewards = np.zeros(self.max_episodes)
        self.played_episode = 0

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = (len(self.env.observation_space.high),)
        self.action_size = self.env.action_space.n

        self.actor_critic = build_actor_critic(input_shape=self.state_size, action_space=self.action_size)
        self.optim = Adam(self.lr)

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(1.0 if done else 0.0)

    def act(self, state):
        _, dist = self.actor_critic.predict(state)
        action = np.random.choice(self.action_size, p=dist[0])

        return action

    def discount_rewards(self, rewards, is_normalized=True):
        discounted_r = np.zeros_like(rewards)

        running_r = 0.0
        for i in np.arange(len(rewards))[::-1]:
            running_r = rewards[i] + self.gamma * running_r
            discounted_r[i] = running_r

        if is_normalized:
            discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r) + 1e-8)

        return discounted_r

    def replay(self):
        states = np.vstack(self.states)
        actions = np.array(self.actions, dtype=int)
        rewards = np.array(self.rewards)
        next_states = np.vstack(self.next_states)
        dones = np.array(self.dones)

        idx = np.stack([np.arange(len(actions)), actions], axis=-1)
        with tf.GradientTape() as tape:
            state_values, probs = self.actor_critic(states)
            state_values = tf.squeeze(state_values)
            next_state_values, _ = self.actor_critic(next_states)
            next_state_values = tf.squeeze(next_state_values)

            # critic
            returns = rewards + self.gamma * next_state_values * (1.0 - dones)
            advantages = returns - state_values
            critic_loss = tf.reduce_mean(tf.square(advantages))

            # actor
            selected_action_prob = tf.gather_nd(probs, idx)
            log_prob = tf.math.log(selected_action_prob)
            actor_loss = -tf.reduce_mean(log_prob * advantages)
            entropy_loss = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs), axis=1))

            total_loss = actor_loss + critic_loss - 1e-3 * entropy_loss

        grads = tape.gradient(total_loss, self.actor_critic.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.actor_critic.trainable_weights))

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def run(self):
        for e in range(self.max_episodes):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            episode_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if done:
                    self.replay()
                    print(f"episode: {e + 1}/{self.max_episodes}, score: {episode_reward:.2f}")
                    self.episode_rewards[e] = episode_reward
                    self.played_episode += 1

            if e >= 5 and np.mean(self.episode_rewards[:e + 1][-3:]) >= 500.0:
                break

        self.env.close()
        self.plot()

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.plot(np.arange(self.played_episode), self.episode_rewards[:self.played_episode])
        ma = np.convolve(self.episode_rewards[:self.played_episode], np.ones(10) / 10.0, mode="valid")
        plt.plot(np.arange(len(ma)) + 10 - 1, ma)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    agent = A2CAgent("CartPole-v1")
    agent.run()
