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


def build_actor(input_shape, action_space, lr):
    input_layer = Input(shape=input_shape)
    x = Dense(128, activation="relu")(input_layer)
    output_layer = Dense(action_space, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def build_critic(input_shape, lr):
    input_layer = Input(shape=input_shape)
    x = Dense(128, activation="relu")(input_layer)
    output_layer = Dense(1, activation="linear")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


class A2CAgent:
    def __init__(self, env_name):
        self.max_episodes = 500
        self.gamma = 0.99
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3

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

        self.actor = build_actor(input_shape=self.state_size, action_space=self.action_size, lr=self.actor_lr)
        self.critic = build_critic(input_shape=self.state_size, lr=self.critic_lr)
        self.actor_optim = Adam(self.actor_lr)
        self.critic_optim = Adam(self.critic_lr)

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(1.0 if done else 0.0)

    def act(self, state):
        prediction = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)

        return action

    def discount_rewards(self, rewards):
        discounted_r = np.zeros_like(rewards)
        running_r = 0.0
        for i in np.arange(len(rewards))[::-1]:
            running_r = rewards[i] + self.gamma * running_r
            discounted_r[i] = running_r

        return discounted_r

    def replay(self):
        states = np.vstack(self.states)
        actions = np.zeros([len(self.actions), self.action_size], dtype=np.float32)
        actions[np.arange(len(actions)), self.actions] = 1.0
        rewards = np.array(self.rewards)
        next_states = np.vstack(self.next_states)
        dones = np.array(self.dones)

        # critic
        with tf.GradientTape() as tape:
            returns = rewards + self.gamma * tf.squeeze(self.critic(next_states)) * (1.0 - dones)
            state_values = self.critic(states)
            advantages = returns - state_values
            loss = tf.reduce_mean(tf.square(advantages))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_optim.apply_gradients(zip(grads, self.critic.trainable_weights))

        # actor
        idx = np.stack([np.arange(len(actions)), np.argmax(actions, axis=1)], axis=-1)
        with tf.GradientTape() as tape:
            action_prob = self.actor(states)
            entropy_loss = -tf.reduce_sum(action_prob * tf.math.log(action_prob), axis=1)
            selected_action_prob = tf.gather_nd(action_prob, idx)
            loss = tf.reduce_mean(-tf.math.log(selected_action_prob) * advantages - entropy_loss * 1e-3)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(grads, self.actor.trainable_weights))

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
                self.replay()

                if done:
                    print(f"episode: {e + 1}/{self.max_episodes}, score: {episode_reward:.2f}")
                    self.episode_rewards[e] = episode_reward
                    self.played_episode += 1

            if e >= 5 and np.mean(self.episode_rewards[:e + 1][-3:]) >= 500.0:
                break

        self.env.close()
        self.plot()

    def plot(self):
        plt.figure(figsize=(10,7))
        plt.plot(np.arange(self.played_episode), self.episode_rewards[:self.played_episode])
        ma = np.convolve(self.episode_rewards[:self.played_episode], np.ones(10) / 10.0, mode="valid")
        plt.plot(np.arange(len(ma)) + 10 - 1, ma)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    agent = A2CAgent("CartPole-v1")
    agent.run()
