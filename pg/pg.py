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


def build_actor(input_shape, action_space):
    input_layer = Input(shape=input_shape)
    x = Dense(64, activation="relu", input_shape=input_shape)(input_layer)
    output_layer = Dense(action_space, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


class PGAgent:
    def __init__(self, env_name):
        self.max_episodes = 1000
        self.gamma = 0.99
        self.actor_lr = 1e-2

        self.states = []
        self.actions = []
        self.rewards = []
        self.episode_rewards = np.zeros(self.max_episodes)
        self.played_episode = 0

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_size = (len(self.env.observation_space.high),)
        self.action_size = self.env.action_space.n

        self.actor = build_actor(input_shape=self.state_size, action_space=self.action_size)
        self.optim = Adam(self.actor_lr)

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def act(self, state):
        prediction = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)

        return action

    def discount_rewards(self, rewards):
        discounted_r = np.zeros_like(rewards).astype(np.float32)
        running_r = 0.0
        for i in np.arange(len(rewards))[::-1]:
            running_r = rewards[i] + self.gamma * running_r
            discounted_r[i] = running_r

        return discounted_r

    def replay(self):
        states = np.vstack(self.states)
        actions = np.array(self.actions, dtype=int)
        rewards = np.array(self.rewards)

        discounted_r = self.discount_rewards(rewards)
        discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r) + 1e-8)

        idx = np.stack([np.arange(len(actions)), actions], axis=-1)
        with tf.GradientTape() as tape:
            action_prob = self.actor(states)
            selected_action_prob = tf.gather_nd(action_prob, idx)
            loss = tf.reduce_mean(-tf.math.log(selected_action_prob) * discounted_r)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.actor.trainable_weights))

        self.states = []
        self.actions = []
        self.rewards = []

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
                self.remember(state, action, reward)
                state = next_state
                episode_reward += reward

                if done:
                    print(f"episode: {e+1}/{self.max_episodes}, score: {episode_reward:.2f}")
                    self.replay()
                    self.episode_rewards[e] = episode_reward
                    self.played_episode += 1

            if e >= 5 and np.mean(self.episode_rewards[:e+1][-5:]) >= 500.0:
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
    agent = PGAgent("CartPole-v1")
    agent.run()
