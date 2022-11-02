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


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, batch_size=64, max_size=int(1e5)):
        self.batch_size = batch_size
        self.size = 0
        self.max_size = max_size
        self.pointer = 0

        self.current_states = np.empty([self.max_size] + obs_shape, dtype=np.float64)
        self.actions = np.empty([self.max_size] + action_shape, dtype=np.float64)
        self.next_states = np.empty([self.max_size] + obs_shape, dtype=np.float64)
        self.rewards = np.empty([self.max_size], dtype=np.float64)
        self.dones = np.empty([self.max_size], dtype=int)

    def store(self, state, action, next_state, rwd, done):
        self.current_states[self.pointer] = state
        self.actions[self.pointer] = action
        self.next_states[self.pointer] = next_state
        self.rewards[self.pointer] = rwd
        self.dones[self.pointer] = done

        if self.size < self.max_size:
            self.size += 1
        self.pointer = (self.pointer + 1) % self.max_size

    def get_sample(self):
        idx = np.random.choice(self.size, self.batch_size, replace=False)

        return self.current_states[idx], self.actions[idx], self.next_states[idx], self.rewards[idx], self.dones[idx]


class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


class OUNoise(object):
    def __init__(self, action_shape, mu=0.0, theta=0.15, sigma=0.2):
        self.action_shape = action_shape
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_shape) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_shape)
        self.state = x + dx

        return self.state


class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self.dense1_layer = Dense(128, activation="relu")
        self.dense2_layer = Dense(128, activation="relu")
        self.output_layer = Dense(self.action_dim,
                                  kernel_initializer=RandomUniform(-3e-3, 3e-3),
                                  bias_initializer=RandomUniform(-3e-3, 3e-3),
                                  activation="tanh")

    def call(self, state):
        x = self.dense1_layer(state)
        x = self.dense2_layer(x)
        x = self.output_layer(x)

        return x


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1_layer = Dense(128, activation="relu")
        self.dense2_layer = Dense(128, activation="relu")
        self.output_layer = Dense(1,
                                  kernel_initializer=RandomUniform(-3e-3, 3e-3),
                                  bias_initializer=RandomUniform(-3e-3, 3e-3))

    def call(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        x = self.dense1_layer(state_action)
        x = self.dense2_layer(x)
        q = self.output_layer(x)

        return q


class Agent:
    def __init__(
            self,
            env,
            batch_size=128,
            memory_size=int(1e6),
            gamma=0.99,
            tau=5e-3,
    ):
        obs_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]

        self.env = env
        self.batch_size = batch_size
        self.memory = ReplayBuffer([obs_shape], [action_shape], self.batch_size, memory_size)
        self.gamma = gamma
        self.tau = tau
        self.ou_noise = OUNoise(action_shape)

        # actor
        self.actor = Actor(action_shape)
        self.actor_target = Actor(action_shape)

        # q
        self.critic = Critic()
        self.critic_target = Critic()

        input_obs = Input(shape=obs_shape, dtype=tf.float32)
        input_action = Input(shape=action_shape, dtype=tf.float32)
        self.actor(input_obs)
        self.actor_target(input_obs)
        self.critic(input_obs, input_action)
        self.critic_target(input_obs, input_action)
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        # optim
        self.actor_optim = Adam(1e-3)
        self.critic_optim = Adam(1e-3)

    def get_action(self, current_state, is_training=True):
        action = self.actor(current_state[np.newaxis, ...])[0]
        if is_training:
            noise = self.ou_noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def update_model(self, frame):
        states, actions, next_states, rewards, dones = self.memory.get_sample()

        # update q
        next_actions = self.actor_target(next_states)
        next_values = self.critic(next_states, next_actions)
        q_target = rewards + self.gamma * (1 - dones) * tf.squeeze(next_values)

        with tf.GradientTape() as tape_q:
            q = tf.squeeze(self.critic(states, actions))
            q_loss = tf.reduce_mean(tf.square(q - q_target))

        # update actor
        with tf.GradientTape() as tape_actor:
            actor_loss = -tf.reduce_mean(
                self.critic(states, self.actor(states))
            )

        grads_q = tape_q.gradient(q_loss, self.critic.trainable_weights)
        self.critic_optim.apply_gradients(zip(grads_q, self.critic.trainable_weights))
        grads_actor = tape_actor.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(grads_actor, self.actor.trainable_weights))

        self.soft_update()

        return actor_loss.numpy(), q_loss.numpy()

    def soft_update(self):
        w = self.critic.get_weights()
        w_target = self.critic_target.get_weights()

        self.critic_target.set_weights([
            self.tau * layer_w + (1.0 - self.tau) * layer_w_target
            for layer_w, layer_w_target in zip(w, w_target)
        ])

        w = self.actor.get_weights()
        w_target = self.actor_target.get_weights()

        self.actor_target.set_weights([
            self.tau * layer_w + (1.0 - self.tau) * layer_w_target
            for layer_w, layer_w_target in zip(w, w_target)
        ])

    def train(self):
        eps_rewards = []
        actor_losses = []
        q_losses = []

        state = self.env.reset()
        eps_rwd = 0
        n_episode = 0

        n_frames = 100000
        for t in range(1, n_frames + 1):
            action = self.get_action(state)
            next_state, rwd, done, _ = self.env.step(action)
            self.memory.store(state, action, next_state, rwd, done)

            state = next_state
            eps_rwd += rwd

            if done:
                print(f"frame: {t: >5}/{n_frames}, eps: {n_episode+1: >4}: {eps_rwd}")
                eps_rewards.append(eps_rwd)
                eps_rwd = 0
                n_episode += 1

                if n_episode % 50 == 0:
                    test_result = self.test()
                    if test_result >= 200.0:
                        break

                state = self.env.reset()
                self.ou_noise.reset()

            if self.memory.size >= self.batch_size:
                losses = self.update_model(t)
                actor_losses.append(losses[0])
                q_losses.append(losses[1])

        self.test()

        env.close()

        # plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs[0].plot(eps_rewards)
        axs[0].set_title("eps_rewards")
        axs[1].plot(actor_losses)
        axs[1].set_title("actor_losses")
        axs[2].plot(q_losses)
        axs[2].set_title("q_losses")

        plt.tight_layout()
        fig.savefig("fig.jpg")
        plt.show()

    def test(self, n_test_eps=10):
        test_eps_rewards = np.empty(n_test_eps)

        for eps in range(n_test_eps):
            state = self.env.reset()
            done = False
            eps_rwd = 0.0

            while not done:
                action = self.get_action(state, is_training=False)
                next_state, rwd, done, _ = self.env.step(action)

                eps_rwd += rwd
                state = next_state

            test_eps_rewards[eps] = eps_rwd

        avg_rwd = np.mean(test_eps_rewards)
        print(f"test result: {avg_rwd}")

        return avg_rwd


if __name__ == "__main__":
    # env = gym.make("Pendulum-v1")
    env = gym.make("LunarLanderContinuous-v2")
    env = ActionNormalizer(env)
    agent = Agent(env)

    agent.train()






