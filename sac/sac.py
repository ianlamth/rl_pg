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


class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = -20.0
        self.log_std_max = 2.0

        self.dense1_layer = Dense(256, activation="relu")
        self.dense2_layer = Dense(256, activation="relu")
        self.mean_layer = Dense(self.action_dim,
                                kernel_initializer=RandomUniform(-3e-3, 3e-3),
                                bias_initializer=RandomUniform(-3e-3, 3e-3))
        self.log_std_layer = Dense(self.action_dim,
                                   kernel_initializer=RandomUniform(-3e-3, 3e-3),
                                   bias_initializer=RandomUniform(-3e-3, 3e-3))

    def call(self, state):
        x = self.dense1_layer(state)
        x = self.dense2_layer(x)

        # mean
        mu = self.mean_layer(x)

        # std
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)

        # sample action
        dist = tfp.distributions.Normal(mu, std)
        z = dist.sample()

        action = tf.tanh(z)
        log_pi = tf.reduce_sum(
            dist.log_prob(z) - tf.math.log(1 - tf.square(action) + 1e-7), axis=-1, keepdims=True
        )

        return action, log_pi


class CriticQ(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1_layer = Dense(256, activation="relu")
        self.dense2_layer = Dense(256, activation="relu")
        self.output_layer = Dense(1,
                                  kernel_initializer=RandomUniform(-3e-3, 3e-3),
                                  bias_initializer=RandomUniform(-3e-3, 3e-3))

    def call(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        x = self.dense1_layer(state_action)
        x = self.dense2_layer(x)
        q = self.output_layer(x)

        return q


class SoftActorCritic:
    def __init__(
            self,
            env,
            batch_size=128,
            memory_size=int(1e5),
            gamma=0.99,
            tau=5e-3,
            policy_update_frequency=2,
            initial_random_steps=int(1e4)
    ):
        obs_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]

        self.env = env
        self.batch_size = batch_size
        self.memory = ReplayBuffer([obs_shape], [action_shape], self.batch_size, memory_size)
        self.gamma = gamma
        self.tau = tau
        self.policy_update_frequency = policy_update_frequency
        self.policy_update_counter = 0
        self.initial_random_steps = initial_random_steps

        # entropy
        self.target_entropy = -np.prod((action_shape,)).item()
        self.log_alpha = tf.Variable(0.0, trainable=True)

        # actor
        self.actor = Actor(action_shape)

        # q
        self.q1 = CriticQ()
        self.q1_target = CriticQ()
        self.q2 = CriticQ()
        self.q2_target = CriticQ()

        input1 = Input(shape=obs_shape, dtype=tf.float32)
        input2 = Input(shape=action_shape, dtype=tf.float32)
        self.q1(input1, input2)
        self.q1_target(input1, input2)
        self.q2(input1, input2)
        self.q2_target(input1, input2)
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

        # optim
        self.alpha_optim = Adam(3e-4)
        self.actor_optim = Adam(3e-4)
        self.q1_optim = Adam(3e-4)
        self.q2_optim = Adam(3e-4)

    def get_action(self, current_state):
        action, _ = self.actor(current_state[np.newaxis, ...])

        return action[0].numpy()

    def update_model(self, frame):
        states, actions, next_states, rewards, dones = self.memory.get_sample()

        # update alpha
        with tf.GradientTape(watch_accessed_variables=False) as tape_alpha:
            tape_alpha.watch(self.log_alpha)
            action, log_prob = self.actor(states)
            alpha_loss = tf.reduce_mean(
                -tf.math.exp(self.log_alpha) * (log_prob + self.target_entropy)
            )

        grads = tape_alpha.gradient(alpha_loss, self.log_alpha)
        self.alpha_optim.apply_gradients([(grads, self.log_alpha)])

        alpha = tf.exp(self.log_alpha)

        # update q
        next_action, next_log_prob = self.actor(next_states)
        next_q1_target = self.q1_target(next_states, next_action)
        next_q2_target = self.q2_target(next_states, next_action)
        next_q_target = tf.math.minimum(next_q1_target, next_q2_target) - alpha * next_log_prob
        q_target = rewards + self.gamma * (1 - dones) * tf.squeeze(next_q_target)

        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch(self.q1.trainable_weights)
            q1 = tf.squeeze(self.q1(states, actions))
            q1_loss = tf.reduce_mean(tf.square(q1 - q_target))

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(self.q2.trainable_weights)
            q2 = tf.squeeze(self.q2(states, actions))
            q2_loss = tf.reduce_mean(tf.square(q2 - q_target))

        # update actor
        with tf.GradientTape(watch_accessed_variables=False) as tape_actor:
            tape_actor.watch(self.actor.trainable_weights)
            new_actions, new_log_prob = self.actor(states)
            soft_q = tf.math.minimum(self.q1(states, new_actions), self.q2(states, new_actions)) - alpha * new_log_prob
            actor_loss = -tf.reduce_mean(soft_q)

        grads_actor = tape_actor.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optim.apply_gradients(zip(grads_actor, self.actor.trainable_weights))
        grads1 = tape1.gradient(q1_loss, self.q1.trainable_weights)
        self.q1_optim.apply_gradients(zip(grads1, self.q1.trainable_weights))
        grads2 = tape2.gradient(q2_loss, self.q2.trainable_weights)
        self.q2_optim.apply_gradients(zip(grads2, self.q2.trainable_weights))

        self.update_q_target()

        q_loss = q1_loss + q2_loss
        return actor_loss.numpy(), q_loss.numpy(), alpha_loss.numpy()

    def update_q_target(self):
        w = self.q1.get_weights()
        w_target = self.q1_target.get_weights()

        self.q1_target.set_weights([
            self.tau * layer_w + (1.0 - self.tau) * layer_w_target
            for layer_w, layer_w_target in zip(w, w_target)
        ])

        w = self.q2.get_weights()
        w_target = self.q2_target.get_weights()

        self.q2_target.set_weights([
            self.tau * layer_w + (1.0 - self.tau) * layer_w_target
            for layer_w, layer_w_target in zip(w, w_target)
        ])

    def train(self):
        eps_rewards = []
        actor_losses = []
        q_losses = []
        alpha_losses = []

        state = self.env.reset()
        eps_rwd = 0
        n_episode = 0

        n_frames = 100000
        for t in range(1, n_frames + 1):
            if t < self.initial_random_steps:
                action = self.env.action_space.sample()
            else:
                action = self.get_action(state)
            next_state, rwd, done, _ = self.env.step(action)
            self.memory.store(state, action, next_state, rwd, done)

            state = next_state
            eps_rwd += rwd

            if done:
                state = self.env.reset()
                print(f"{t: >5}/{n_frames}, {n_episode+1: >4}: {eps_rwd}")
                eps_rewards.append(eps_rwd)
                eps_rwd = 0
                n_episode += 1

                if n_episode % 50 == 0:
                    self.test()

            if self.memory.size >= self.batch_size and t >= self.initial_random_steps:
                losses = self.update_model(t)
                actor_losses.append(losses[0])
                q_losses.append(losses[1])
                alpha_losses.append(losses[2])

        self.test()
        self.env.close()

        # plot
        fig, axs = plt.subplots(1, 4, figsize=(15, 6))
        axs[0].plot(eps_rewards)
        axs[0].set_title("eps_rewards")
        axs[1].plot(actor_losses)
        axs[1].set_title("actor_losses")
        axs[2].plot(q_losses)
        axs[2].set_title("q_losses")
        axs[3].plot(alpha_losses)
        axs[3].set_title("alpha_losses")

        plt.tight_layout()
        fig.savefig("lander.jpg")
        plt.show()

    def test(self, n_test_eps=10):
        test_eps_rewards = np.empty(n_test_eps)

        for eps in range(n_test_eps):
            state = self.env.reset()
            done = False
            eps_rwd = 0.0

            while not done:
                action = self.get_action(state)
                next_state, rwd, done, _ = self.env.step(action)

                eps_rwd += rwd
                state = next_state

            test_eps_rewards[eps] = eps_rwd

        avg_rwd = np.mean(test_eps_rewards)
        print(f"test result: {avg_rwd}, min: {np.min(test_eps_rewards):.2f}, max: {np.max(test_eps_rewards):.2f}")

        return avg_rwd


if __name__ == "__main__":
    # env = gym.make("Pendulum-v1")
    env = gym.make("LunarLanderContinuous-v2")
    # env = gym.make("BipedalWalker-v3")
    env = ActionNormalizer(env)
    agent = SoftActorCritic(env)

    agent.train()






