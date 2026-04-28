"""
helpery do ewaluacji agentow

osobno trening osobno test
w treningu exploration
w ewaluacji sprawdzenie co model faktycznie umie

metryki pod prace
success rate
srednia nagroda
srednia liczba krokow
powody konca epizodu
"""

from __future__ import annotations

import numpy as np

def _tiebreak_argmax(x, tol=1e-6, rng=None):
    """Argmax with random tie-breaking within tol."""
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x)
    idx = np.flatnonzero(x >= m - float(tol))
    if idx.size == 0:
        return int(np.argmax(x))
    if idx.size == 1:
        return int(idx[0])
    rng = rng if rng is not None else np.random
    return int(rng.choice(idx))

def _infer_n_actions(env, agent):
    # Liczba akcji z przestrzeni akcji srodowiska.
    if hasattr(env, "action_space"):
        a = env.action_space
        if hasattr(a, "n"):
            return int(a.n)
        if isinstance(a, (int, np.integer)):
            return int(a)
    # Alternatywne odczytanie liczby akcji z atrybutow agenta.
    for attr in ("n_actions", "action_dim", "num_actions"):
        if hasattr(agent, attr):
            return int(getattr(agent, attr))
    raise AttributeError("Nie mogę ustalić liczby akcji (env.action_space.n lub agent.n_actions/action_dim/num_actions).")

def _reset_env(env):
    out = env.reset()
    # Obsluga Gymnasium oraz klasycznego Gym.
    if isinstance(out, tuple) and len(out) == 2:
        obs, _info = out
        return obs
    return out

def _step_env(env, action):
    out = env.step(action)
    # format gymnasium
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    # Format klasycznego Gym.
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    raise RuntimeError("Nieznany format zwrotu z env.step(action).")

def dqn_greedy_action(agent, state, tol=1e-6, rng=None):
    """Wybor akcji DQN z losowym rozstrzyganiem remisow."""
    s = np.asarray(state, dtype=np.float32)[np.newaxis, :]
    q = agent.model(s, training=False).numpy()[0]
    return _tiebreak_argmax(q, tol=tol, rng=rng)

def ppo_greedy_action(agent, state, tol=1e-6, rng=None):
    """
    greedy akcja ppo z losowaniem remisow
    wsparcie dla kilku nazw modelu actora
    """
    actor = getattr(agent, "actor", None) or getattr(agent, "actor_model", None) or getattr(agent, "model_actor", None)
    if actor is None:
        if hasattr(agent, "act"):
            # Deterministyczny wybor zachlanny.
            try:
                return int(agent.act(state, deterministic=True))
            except TypeError:
                return int(agent.act(state))
        raise AttributeError("Nie znalazłem modelu aktora w PPO (actor/actor_model/model_actor) ani metody act().")

    out = actor(state[np.newaxis, :], training=False).numpy()[0]

    # Obsluga prawdopodobienstw lub logitow.
    scores = out
    return _tiebreak_argmax(scores, tol=tol, rng=rng)

def ppo_stochastic_action(agent, state, rng=None):
    """
    losowanie akcji z polityki ppo
    najpierw agent act
    potem actor output
    probs albo logits
    """
    rng = rng if rng is not None else np.random

    if hasattr(agent, "act"):
        try:
            return int(agent.act(state, deterministic=False))
        except TypeError:
            # Obsluga agentow z metoda act.
            try:
                return int(agent.act(state))
            except Exception:
                pass

    actor = getattr(agent, "actor", None) or getattr(agent, "actor_model", None) or getattr(agent, "model_actor", None)
    if actor is None:
        raise AttributeError("Brak obsługi wyboru akcji PPO: brak act() i brak actor/actor_model/model_actor.")

    out = actor(state[np.newaxis, :], training=False).numpy()[0].astype(np.float64)

    # Wybor na podstawie prawdopodobienstw lub logitow.
    if np.all(out >= -1e-6) and 0.9 <= float(np.sum(out)) <= 1.1:
        probs = np.clip(out, 1e-12, 1.0)
        probs = probs / float(np.sum(probs))
    else:
        # Przeksztalcenie logitow funkcja softmax.
        z = out - np.max(out)
        e = np.exp(z)
        probs = e / float(np.sum(e))

    return int(rng.choice(len(probs), p=probs))

def eval_action(agent, state, algo: str, env=None, eps_random: float = 0.0, tol=1e-6, stochastic_policy: bool = False, rng=None):
    """
    wspolny wrapper dla dqn i ppo
    ewaluacja agentow w tym samym protokole
    eps random jako dodatkowy losowy ruch
    stochastic policy glownie dla ppo
    """
    rng = rng if rng is not None else np.random
    algo = algo.lower()

    if eps_random > 0.0 and rng.rand() < eps_random:
        if env is None:
            raise ValueError("eval_action: żeby losować akcję potrzebuję env (do action_space) albo agent.n_actions.")
        n_actions = _infer_n_actions(env, agent)
        return int(rng.randint(n_actions))

    if algo == "dqn":
        return dqn_greedy_action(agent, state, tol=tol, rng=rng)

    if algo == "ppo":
        if stochastic_policy:
            return ppo_stochastic_action(agent, state, rng=rng)
        return ppo_greedy_action(agent, state, tol=tol, rng=rng)

    raise ValueError("algo musi być 'dqn' albo 'ppo'")

def evaluate_agent(env_factory, agent, algo: str, n_episodes: int = 20, eps_random: float = 0.0, tol=1e-6,
                   stochastic_policy: bool = False, seed: int | None = None,
                   eval_episodes: int | None = None, eval_seed: int | None = None):
    """
    uczciwa ewaluacja na nowych labiryntach
    nowy env na epizod
    success rate srednia nagroda srednie kroki
    """
    # Powtarzalna ewaluacja na nowych labiryntach.
    if eval_episodes is not None:
        n_episodes = int(eval_episodes)
    if eval_seed is not None:
        seed = int(eval_seed)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    goals = 0
    rewards = []
    steps_list = []
    reasons = {"goal": 0, "max_steps": 0, "no_lives": 0, "other": 0, "unknown": 0}

    for _ in range(n_episodes):
        env = env_factory()

        # Ustawienie seeda srodowiska dla kazdego epizodu, jesli jest dostepne.
        if seed is not None:
            try:
                s = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
                if isinstance(s, tuple) and len(s) == 2:
                    s, _info = s
            except TypeError:
                s = _reset_env(env)
        else:
            s = _reset_env(env)
        total = 0.0
        done = False
        info = {}
        steps = 0

        max_steps = getattr(env, "max_steps", None)
        if max_steps is None:
            max_steps = 250  # Domyslny limit krokow, gdy srodowisko nie udostepnia max_steps.

        for t in range(int(max_steps)):
            a = eval_action(agent, s, algo=algo, env=env, eps_random=eps_random, tol=tol,
                            stochastic_policy=stochastic_policy, rng=rng)
            s, r, done, info = _step_env(env, a)
            total += float(r)
            steps = t + 1
            if done:
                break

        rewards.append(total)
        steps_list.append(steps)

        # Sprawdzenie osiagniecia celu przez metode srodowiska lub pola info/reason.
        at_goal = False
        if hasattr(env, "is_at_goal"):
            try:
                at_goal = bool(env.is_at_goal())
            except Exception:
                at_goal = False
        if at_goal or info.get("reason") == "goal":
            goals += 1

        reason = info.get("reason", "other")
        if reason not in reasons:
            reason = "other"
        reasons[reason] += 1

    return {
        "success_rate": 100.0 * goals / float(n_episodes),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "reasons": reasons,
        "n_episodes": int(n_episodes),
        "eps_random": float(eps_random),
        "stochastic_policy": bool(stochastic_policy),
        "seed": seed,
    }
