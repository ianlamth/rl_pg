import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.losses import mean_squared_error
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self.dense1_layer = Dense(128, activation="relu")
        self.dense2_layer = Dense(128, activation="relu")
        self.output_layer = Dense(self.action_dim, activation="softmax")

    def call(self, state):
        x = self.dense1_layer(state)
        x = self.dense2_layer(x)
        dist = self.output_layer(x)

        return dist


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1_layer = Dense(128, activation="relu")
        self.dense2_layer = Dense(128, activation="relu")
        self.output_layer = Dense(1)

    def call(self, state):
        x = self.dense1_layer(state)
        x = self.dense2_layer(x)
        v = self.output_layer(x)

        return v


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    masks = 1.0 - masks
    # values = values + [next_value]
    values = np.append(values, next_value)
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return np.array(returns)


class Agent:
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.n
        self.env = env

        self.actor = Actor(self.action_shape)
        self.critic = Critic()
        self.actor_optim = Adam(1e-3)
        self.critic_optim = Adam(1e-4)
        self.actor(Input(shape=self.obs_shape))
        self.critic(Input(shape=self.obs_shape))

        self.clip_parameter = 0.2
        self.batch_size = 128
        self.n_train_epoch = 4

    def get_action(self, state):
        p = self.actor.predict(state[np.newaxis, ...])[0]
        action = np.random.choice(self.action_shape, p=p)

        return action, p

    def update_model(self, actions, old_probs, states, advantages, returns):
        # critic
        with tf.GradientTape() as tape_v:
            v = tf.squeeze(self.critic(states))
            critic_loss = tf.reduce_mean(tf.square(returns - v))

        # actor
        idx = np.stack([np.arange(len(actions)), actions], axis=-1).astype(int)
        old_probs = tf.cast(tf.gather_nd(old_probs, idx), tf.float32)
        with tf.GradientTape() as tape_a:
            new_dist = self.actor(states)
            new_probs = tf.gather_nd(new_dist, idx)
            ratio = new_probs / old_probs

            # objective loss
            actor_loss = -tf.reduce_mean(tf.minimum(
                ratio * advantages,
                tf.clip_by_value(ratio, 1.0 - self.clip_parameter, 1.0 + self.clip_parameter) * advantages
            ))

            # entropy
            actor_loss += 1e-3 * -tf.reduce_mean(tf.reduce_sum(new_dist * tf.math.log(new_dist), axis=1))

        grads_v = tape_v.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_optim.apply_gradients(zip(grads_v, self.critic.trainable_weights))
        grads_a = tape_a.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(grads_a, self.actor.trainable_weights))

        return actor_loss.numpy(), critic_loss.numpy()

    def train(self):
        eps_rewards = []

        state = self.env.reset()
        eps_rwd = 0
        n_episode = 0

        n_steps = 200
        for step in range(n_steps):
            states = np.empty([self.batch_size, self.obs_shape])
            actions = np.empty([self.batch_size])
            probs = np.empty([self.batch_size, self.action_shape])
            rewards = np.empty([self.batch_size])
            dones = np.empty([self.batch_size])

            for i in range(self.batch_size):
                action, p = self.get_action(state)
                next_state, rwd, done, _ = self.env.step(action)

                states[i] = state
                probs[i] = p
                actions[i] = action
                rewards[i] = rwd
                dones[i] = 1 if done else 0

                state = next_state
                eps_rwd += rwd

                if done:
                    rewards[i] = 0.0
                    state = self.env.reset()
                    print(f"step: {step: >6}, {n_episode+1: >4}: {eps_rwd}")
                    eps_rewards.append(eps_rwd)
                    eps_rwd = 0
                    n_episode += 1

            next_value = self.critic(state[np.newaxis, ...])[0]
            values = self.critic(states)
            returns = compute_gae(next_value, rewards, dones, values)
            advantages = returns - values

            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            for epoch in range(self.n_train_epoch):
                self.update_model(actions, probs, states, advantages, returns)

        env.close()

        # plot
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(eps_rewards)
        axs[0].set_title("eps_rewards")
        axs[0].grid(1)

        plt.tight_layout()
        fig.savefig("ppo_fig.jpg")
        plt.show()


if __name__ == "__main__":
    # env = gym.make("Pendulum-v1")
    env = gym.make("CartPole-v1")
    # env = ActionNormalizer(env)
    agent = Agent(env)

    agent.train()






