import numpy as np
import tensorflow as tf


"""
ppo agent do porownania z dqn

policy gradient zamiast klasycznego q value
actor jako wybor akcji
critic jako ocena stanu
gae jako liczenie advantage
clipping jako hamulec na zbyt duzy update 

fajny kontrast do pracy
value based kontra actor critic
"""


class PPOAgent:
    """
    prosta implementacja ppo do porownania z dqn

    actor critic
    gae lambda
    clipped policy objective
    osobne optimizery actor i critic
    tf data do minibatchy
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        gamma=0.99,
        lam=0.95,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        train_epochs=10,
        batch_size=64,
        rollout_size=1024,
        target_kl=0.02,
        max_grad_norm=0.5,
        hidden=128,
        seed=None,
    ):
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)

        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.train_epochs = int(train_epochs)
        self.batch_size = int(batch_size)
        self.rollout_size = int(rollout_size)
        self.target_kl = float(target_kl)
        self.max_grad_norm = float(max_grad_norm)

        if seed is not None:
            np.random.seed(int(seed))
            tf.random.set_seed(int(seed))

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.actor_updates = 0
        self.critic_updates = 0

        self.actor = self._build_actor(hidden)
        self.critic = self._build_critic(hidden)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def _build_actor(self, hidden):
        inp = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(hidden, activation="relu")(inp)
        x = tf.keras.layers.Dense(hidden, activation="relu")(x)
        logits = tf.keras.layers.Dense(self.n_actions)(x)
        return tf.keras.Model(inp, logits)

    def _build_critic(self, hidden):
        inp = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(hidden, activation="relu")(inp)
        x = tf.keras.layers.Dense(hidden, activation="relu")(x)
        value = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inp, value)

    def act_full(self, state, deterministic=False):
        # Aktor zwraca logity, z ktorych wyznaczane sa prawdopodobienstwa akcji.
        s = np.asarray(state, dtype=np.float32)[np.newaxis, :]
        logits = self.actor(s, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.n_actions, p=probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        value = float(self.critic(s, training=False).numpy()[0, 0])
        return action, log_prob, value

    def act(self, state, deterministic=False):
        action, _, _ = self.act_full(state, deterministic=deterministic)
        return int(action)

    def remember(self, state, action, reward, done, log_prob=None, value=None):
        state = np.asarray(state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done_f = float(bool(done))

        if log_prob is None or value is None:
            s = state[np.newaxis, :]
            logits = self.actor(s, training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            log_prob = float(np.log(probs[action] + 1e-8))
            value = float(self.critic(s, training=False).numpy()[0, 0])

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done_f)
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def should_update(self, done=False):
        return bool(done) or len(self.states) >= self.rollout_size

    def replay(self, last_state, done):
        # PPO aktualizuje model na podstawie zebranego rollouta.
        n = len(self.states)
        if n < 2:
            self._clear()
            return

        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.int32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        old_logp = np.asarray(self.log_probs, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        if bool(done):
            last_val = 0.0
        else:
            ls = np.asarray(last_state, dtype=np.float32)[np.newaxis, :]
            last_val = float(self.critic(ls, training=False).numpy()[0, 0])

        adv = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = last_val if t == n - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * nonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch_size = min(self.batch_size, n)
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_logp, adv, returns))
        dataset = dataset.shuffle(buffer_size=n).batch(batch_size, drop_remainder=False)

        stop_early = False
        for _ in range(self.train_epochs):
            for batch in dataset:
                approx_kl = self._train_step(*batch)
                if float(approx_kl.numpy()) > 1.5 * self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        self.actor_updates += 1
        self.critic_updates += 1
        self._clear()

    @tf.function
    def _train_step(self, states, actions, old_logp, adv, returns):
        # Aktualizacja PPO dla jednego minibatcha.
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            logits = self.actor(states, training=True)
            logp_all = tf.nn.log_softmax(logits, axis=-1)
            logp = tf.reduce_sum(logp_all * tf.one_hot(actions, self.n_actions), axis=1)

            ratio = tf.exp(logp - old_logp)
            clipped_ratio = tf.clip_by_value(
                ratio,
                1.0 - self.clip_ratio,
                1.0 + self.clip_ratio,
            )
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv, clipped_ratio * adv))

            probs = tf.nn.softmax(logits, axis=-1)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp_all, axis=1))
            actor_loss = policy_loss - self.entropy_coef * entropy

            values = tf.squeeze(self.critic(states, training=True), axis=1)
            value_loss = tf.reduce_mean(tf.square(returns - values))
            critic_loss = self.value_coef * value_loss

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        approx_kl = tf.reduce_mean(old_logp - logp)
        return approx_kl

    def _clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
