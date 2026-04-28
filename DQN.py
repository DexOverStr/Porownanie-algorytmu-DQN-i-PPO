import random
from collections import deque

import numpy as np
import tensorflow as tf


"""
dqn agent do porownania z ppo

q learning plus neural network
input to obserwacja labiryntu
output to q value dla kazdej akcji

replay buffer jako losowe probki starych ruchow
target network dla stabilizacji
double dqn jako mniejsze przeszacowanie q value
epsilon greedy jako exploration
"""


class DQNAgent:
    """
    prosty dqn do porownania z ppo

    replay buffer
    target network
    opcjonalny double dqn
    epsilon greedy exploration
    szybkie wywolania modelu bez model predict
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        gamma=0.95,
        lr=5e-4,
        batch_size=64,
        memory_size=50_000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        use_target=True,
        use_double_dqn=True,
        target_update_freq=500,
        train_every=16,
        min_memory=1000,
        greedy_tie_tol=1e-6,
        hidden=128,
        seed=None,
    ):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.memory = deque(maxlen=int(memory_size))

        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.use_target = bool(use_target)
        self.use_double_dqn = bool(use_double_dqn)
        self.target_update_freq = int(target_update_freq)
        self.train_every = int(train_every)
        self.min_memory = int(min_memory)
        self.greedy_tie_tol = float(greedy_tie_tol)

        self.train_steps = 0
        self.gradient_updates = 0

        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            tf.random.set_seed(int(seed))

        self.model = self._build_model(hidden)
        self.target_model = self._build_model(hidden)
        self.update_target_network()

    def _build_model(self, hidden):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dense(hidden, activation="relu"),
            tf.keras.layers.Dense(self.n_actions, activation="linear"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=tf.keras.losses.Huber(),
        )
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def act(self, state, use_epsilon=True):
        # Strategia epsilon-greedy z malejacym poziomem eksploracji.
        if use_epsilon and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q_values = self.model(s, training=False).numpy()[0]

        max_q = np.max(q_values)
        best = np.flatnonzero(q_values >= max_q - self.greedy_tie_tol)
        if best.size == 1:
            return int(best[0])
        return int(np.random.choice(best))

    def replay(self):
        # Aktualizacja sieci wykonywana co train_every krokow.
        if len(self.memory) < max(self.batch_size, self.min_memory):
            return

        self.train_steps += 1
        if self.train_steps % self.train_every != 0:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.asarray([m[0] for m in minibatch], dtype=np.float32)
        actions = np.asarray([m[1] for m in minibatch], dtype=np.int32)
        rewards = np.asarray([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.asarray([m[3] for m in minibatch], dtype=np.float32)
        dones = np.asarray([m[4] for m in minibatch], dtype=np.float32)

        if self.use_target and self.use_double_dqn:
            # Double DQN: wybor akcji przez model online, wycena przez model docelowy.
            next_q_online = self.model(next_states, training=False).numpy()
            next_actions = np.argmax(next_q_online, axis=1)
            next_q_target = self.target_model(next_states, training=False).numpy()
            max_next_q = next_q_target[np.arange(self.batch_size), next_actions]
        else:
            q_net = self.target_model if self.use_target else self.model
            next_q = q_net(next_states, training=False).numpy()
            max_next_q = np.max(next_q, axis=1)

        target_values = rewards + self.gamma * max_next_q * (1.0 - dones)

        q_values = self.model(states, training=False).numpy()
        q_values[np.arange(self.batch_size), actions] = target_values

        self.model.train_on_batch(states, q_values)
        self.gradient_updates += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.use_target and self.gradient_updates % self.target_update_freq == 0:
            self.update_target_network()
