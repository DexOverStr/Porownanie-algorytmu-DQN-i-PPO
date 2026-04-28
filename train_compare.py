"""
runner eksperymentow dqn vs ppo

taki glowny experiment pipeline do pracy
Porownanie DQN i PPO w tym samym protokole eksperymentalnym.
stage trudnosci
ewaluacja
osobne csv pod wykresy i tabelki
"""

from __future__ import annotations

import os
import sys
import csv
import json
import gc
import random
import multiprocessing
import time
from datetime import datetime


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_utf8_stdio()

# Wyciszenie logow TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Import TensorFlow z ograniczonym logowaniem.
class _NullWriter:
    def write(self, _): pass
    def flush(self): pass
    def close(self): pass

_old_stderr = sys.stderr
sys.stderr = _NullWriter()
import tensorflow as tf
sys.stderr = _old_stderr
configure_utf8_stdio()

import numpy as np

from lab_env import LabEnv
from eval_helpers import evaluate_agent
from DQN import DQNAgent
from ppo import PPOAgent


DEFAULT_TOTAL_ENV_STEPS = 360000
DEFAULT_EVAL_EVERY = 20000
DEFAULT_MAX_STEPS = 250
DEFAULT_EVAL_EPISODES = 50
DEFAULT_STEP_PENALTY = -0.01
DEFAULT_EP_PRINT_EVERY = 10


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def stage_for_steps(env_steps_total: int) -> int:
    """
    stage co 60k krokow
    0 od 0
    1 od 60000
    2 od 120000
    3 od 180000
    4 od 240000
    5 od 300000
    """
    if env_steps_total < 60000:
        return 0
    if env_steps_total < 120000:
        return 1
    if env_steps_total < 180000:
        return 2
    if env_steps_total < 240000:
        return 3
    if env_steps_total < 300000:
        return 4
    return 5


def make_env_for_stage(stage: int, base_n: int = 21, seed: int | None = None):
    """
    parametry trudnosci dla stage
    tunnels rosnie ze stage
    pulapki od trudniejszych stage
    max steps brane z ustawien globalnych
    """
    stage = int(stage)

    tunnels_factor = 1.0 + 0.2 * stage
    # Pulapki od trzeciego etapu curriculum.
    p_trap = 0.0 if stage < 3 else min(0.01 * (stage - 2), 0.06)

    n = int(base_n)  # Rozmiar labiryntu.

    max_steps = 800

    max_steps = DEFAULT_MAX_STEPS

    def factory():
        return LabEnv(
            n=n,
            tunnels_factor=tunnels_factor,
            p_trap=p_trap,
            max_tunnel_factor=0.5,
            step_penalty=DEFAULT_STEP_PENALTY,
            seed=seed,
            max_steps=max_steps,
        )

    meta = {
        "stage": stage,
        "n": n,
        "tunnels_factor": float(tunnels_factor),
        "p_trap": float(p_trap),
        "max_steps": int(max_steps),
    }
    return factory, meta


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def init_csv(path: str):
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow([
            "time",
            "algo",
            "env_steps_total",
            "episode",
            "stage",
            "n",
            "tunnels_factor",
            "p_trap",
            "max_steps",
            "episode_steps",
            "episode_reward",
            "episode_done_reason",
            "eval_seed",
            "eval_episodes",
            "eval_success_rate",
            "eval_avg_reward",
            "eval_avg_steps",
        ])
        f.flush()
    return f, w


def train_by_steps(
    algo: str,
    total_env_steps: int = 360000,
    eval_every: int = DEFAULT_EVAL_EVERY,
    base_n: int = 21,
    seed: int = 123,
    eval_seed: int = 999,
    out_dir: str = "logs_out",
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
):
    """
    trening jednego algorytmu przez dana liczbe krokow env

    env steps total wazniejsze niz liczba epizodow
    dqn i ppo moga konczyc epizody w roznym tempie
    porownanie po tej samej liczbie interakcji z labiryntem
    """
    algo = algo.lower().strip()
    assert algo in ("dqn", "ppo")

    set_global_seed(seed)
    ensure_dir(out_dir)

    csv_path = os.path.join(out_dir, f"logs_{algo}.csv")
    fcsv, wcsv = init_csv(csv_path)

    stage = stage_for_steps(0)
    env_factory, meta = make_env_for_stage(stage, base_n=base_n, seed=seed)
    env = env_factory()

    state_dim = env.state_dim
    n_actions = env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)

    if algo == "dqn":
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            seed=seed,
            hidden=64,
            train_every=32,
            min_memory=500,
        )
    else:
        agent = PPOAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            seed=seed,
            hidden=64,
            rollout_size=512,
            train_epochs=4,
        )

    env_steps_total = 0
    next_eval_at = eval_every
    episode = 0

    print(f"START {algo.upper()} | total_env_steps={total_env_steps} | eval_every={eval_every} | max_steps_ep={meta['max_steps']}", flush=True)

    try:
        while env_steps_total < total_env_steps:
            # Aktualizacja etapu curriculum wedlug progow.
            new_stage = stage_for_steps(env_steps_total)
            if new_stage != stage:
                stage = new_stage
                env_factory, meta = make_env_for_stage(stage, base_n=base_n, seed=seed)
                env = env_factory()
                print(f"=== STAGE CHANGE -> {stage} at env_steps_total={env_steps_total} | tunnels_factor={meta['tunnels_factor']} | p_trap={meta['p_trap']} ===", flush=True)

            episode += 1
            s = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            last_done_reason = "unknown"

            while not done and env_steps_total < total_env_steps:
                if algo == "dqn":
                    a = agent.act(s, use_epsilon=True)
                    s2, r, done, info = env.step(a)
                    agent.remember(s, a, r, s2, done)
                    agent.replay()
                else:
                    a, logp, val = agent.act_full(s, deterministic=False)
                    s2, r, done, info = env.step(a)
                    agent.remember(s, a, r, done, log_prob=logp, value=val)
                    if agent.should_update(done=done):
                        agent.replay(last_state=s2, done=done)

                ep_reward += float(r)
                ep_steps += 1
                env_steps_total += 1
                s = s2

                if done:
                    last_done_reason = str(info.get("reason", "unknown"))

                # Ewaluacja co eval_every krokow.
                if env_steps_total >= next_eval_at:
                    # Ewaluacja oddzielona od eksploracji treningowej.
                    env_eval_factory, _ = make_env_for_stage(stage, base_n=base_n, seed=None)
                    eval_result = evaluate_agent(
                        agent=agent,
                        env_factory=env_eval_factory,
                        algo=algo,
                        n_episodes=eval_episodes,
                        seed=eval_seed,
                    )
                    success = eval_result["success_rate"]
                    avg_reward = eval_result["avg_reward"]
                    avg_steps = eval_result["avg_steps"]

                    # Zapis punktu kontrolnego ewaluacji.
                    wcsv.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        algo,
                        env_steps_total,
                        episode,
                        meta["stage"],
                        meta["n"],
                        meta["tunnels_factor"],
                        meta["p_trap"],
                        meta["max_steps"],
                        "", "", "",  # Pusty wpis ep_info dla punktu kontrolnego.
                        eval_seed,
                        eval_episodes,
                        float(success),
                        float(avg_reward),
                        float(avg_steps),
                    ])
                    fcsv.flush()

                    print(f"[EVAL] {algo.upper()} env_steps_total={env_steps_total} stage={meta['stage']} succ={success:.1f}% avg_rew={avg_reward:.2f} avg_steps={avg_steps:.1f}", flush=True)
                    next_eval_at += eval_every

            if not done and env_steps_total >= total_env_steps:
                # Ostatni epizod zakonczony przez limit eksperymentu.

                last_done_reason = "training_limit"

            # Wypisanie postepu co kilka epizodow.
            should_print_episode = (
                episode == 1
                or episode % DEFAULT_EP_PRINT_EVERY == 0
                or last_done_reason in ("goal", "no_lives", "training_limit")
            )
            if should_print_episode:
                print(
                    f"[EP] {algo.upper()} ep={episode} env_steps_total={env_steps_total} stage={meta['stage']} "
                    f"ep_steps={ep_steps}/{meta['max_steps']} ep_reward={ep_reward:.2f} done={last_done_reason} "
                    f"(n={meta['n']} tunnels={meta['tunnels_factor']:.2f} p_trap={meta['p_trap']:.3f})",
                    flush=True
                )

            # Zapis epizodu do pliku CSV.
            wcsv.writerow([
                datetime.now().isoformat(timespec="seconds"),
                algo,
                env_steps_total,
                episode,
                meta["stage"],
                meta["n"],
                meta["tunnels_factor"],
                meta["p_trap"],
                meta["max_steps"],
                ep_steps,
                float(ep_reward),
                last_done_reason,
                "", "", "", "",  # Pusty wpis eval dla wiersza epizodu.
            ])
            fcsv.flush()

    finally:
        fcsv.close()

    print(f"KONIEC {algo.upper()} | zapisano: {csv_path}", flush=True)


def ask_int(prompt: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Podaj liczbę całkowitą.", flush=True)
                continue

        if min_value is not None and value < min_value:
            print(f"Minimum: {min_value}", flush=True)
            continue
        if max_value is not None and value > max_value:
            print(f"Maksimum: {max_value}", flush=True)
            continue
        return value


def ask_seed_list(prompt: str, default: list[int] | None = None) -> list[int]:
    default = default or [111, 222, 333, 444, 555]
    raw_default = ",".join(str(x) for x in default)
    while True:
        raw = input(f"{prompt} [{raw_default}] > ").strip()
        if raw == "":
            return list(default)
        try:
            seeds = [int(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
        except ValueError:
            print("Podaj seedy jako liczby, np. 111,222,333.", flush=True)
            continue
        if not seeds:
            print("Podaj co najmniej jeden seed.", flush=True)
            continue
        return seeds


def _compare_out_dir(seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("logs_out", f"compare_seed_{int(seed)}_{stamp}")


def _write_run_config(out_dir: str, seed: int, parallel: bool):
    config = {
        "seed": int(seed),
        "parallel": bool(parallel),
        "total_env_steps": int(DEFAULT_TOTAL_ENV_STEPS),
        "eval_every": int(DEFAULT_EVAL_EVERY),
        "eval_episodes": int(DEFAULT_EVAL_EPISODES),
        "max_steps_per_episode": int(DEFAULT_MAX_STEPS),
        "step_penalty": float(DEFAULT_STEP_PENALTY),
        "episode_print_every": int(DEFAULT_EP_PRINT_EVERY),
        "algorithms": ["dqn", "ppo"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _drain_log_files(log_paths, positions):
    printed = False
    for log_path in log_paths:
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
            log_file.seek(positions.get(log_path, 0))
            text = log_file.read()
            positions[log_path] = log_file.tell()
        if text:
            print(text, end="", flush=True)
            printed = True
    return printed


def _wait_training_processes(processes, log_paths):
    positions = {log_path: 0 for log_path in log_paths}
    while any(process.is_alive() for process in processes):
        _drain_log_files(log_paths, positions)
        time.sleep(0.2)

    for process in processes:
        process.join()

    while _drain_log_files(log_paths, positions):
        pass


def _train_worker(algo: str, seed: int, out_dir: str, log_path: str | None = None):
    """Uruchomienie treningu w osobnym procesie."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    log_file = None
    if log_path is not None:
        log_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    try:
        configure_utf8_stdio()
        train_by_steps(
            algo,
            total_env_steps=DEFAULT_TOTAL_ENV_STEPS,
            seed=seed,
            eval_seed=seed + 1000,
            out_dir=out_dir,
            eval_episodes=DEFAULT_EVAL_EPISODES,
        )
    finally:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        if log_file is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()


def _run_training_process(algo: str, seed: int, out_dir: str):
    # Osobny proces dla kazdego treningu.
    # Ograniczenie pozostawionego stanu TensorFlow po poprzednim seedzie.
    log_path = os.path.join(out_dir, f"console_{algo}.log")
    process = multiprocessing.Process(target=_train_worker, args=(algo, seed, out_dir, log_path))
    process.start()
    _wait_training_processes([process], [log_path])
    if process.exitcode != 0:
        raise RuntimeError(f"Trening {algo.upper()} dla seed={seed} zakończył się błędem: {process.exitcode}")


def train_compare_run(seed: int, parallel: bool = False):
    """
    para dqn ppo dla jednego seeda

    parallel true raczej szybki test
    do pracy lepiej parallel false
    spokojniejsze warunki dla obu algorytmow
    """
    out_dir = _compare_out_dir(seed)
    ensure_dir(out_dir)
    _write_run_config(out_dir, seed=seed, parallel=parallel)
    print(f"CSV będą zapisane w: {out_dir}", flush=True)

    if not parallel:
        print("=== PORÓWNANIE SEKWENCYJNE: najpierw DQN, potem PPO ===", flush=True)
        _run_training_process("dqn", seed, out_dir)
        _run_training_process("ppo", seed, out_dir)
        return out_dir

    print("=== PORÓWNANIE RÓWNOLEGŁE: DQN i PPO jednocześnie ===", flush=True)
    log_paths = [
        os.path.join(out_dir, "console_dqn.log"),
        os.path.join(out_dir, "console_ppo.log"),
    ]
    processes = [
        multiprocessing.Process(target=_train_worker, args=("dqn", seed, out_dir, log_paths[0])),
        multiprocessing.Process(target=_train_worker, args=("ppo", seed, out_dir, log_paths[1])),
    ]
    for process in processes:
        process.start()
    _wait_training_processes(processes, log_paths)

    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"Co najmniej jeden proces treningu zakończył się błędem: {failed}")
    return out_dir


def train_many_seeds(seeds: list[int], parallel: bool = False):
    """Uruchomienie eksperymentow dla wielu seedow."""
    ensure_dir("logs_out")
    index_path = os.path.join("logs_out", f"runs_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "parallel", "out_dir"])
        for seed in seeds:
            print(f"=== START SEED {seed} ===", flush=True)
            out_dir = train_compare_run(seed=seed, parallel=parallel)
            writer.writerow([seed, bool(parallel), out_dir])
            f.flush()
            print(f"=== KONIEC SEED {seed}: {out_dir} ===", flush=True)
    print(f"Indeks uruchomien: {index_path}", flush=True)
    return index_path


def _maze_image(env):
    maze = env.get_maze().copy()
    pos = env.get_agent_position()
    if pos is not None:
        maze[pos] = 5

    colors = np.array([
        [1.00, 1.00, 1.00],  # Sciezka.
        [0.05, 0.05, 0.05],  # Sciana.
        [0.85, 0.12, 0.12],  # Pulapka.
        [0.20, 0.65, 0.25],  # Start.
        [0.10, 0.30, 0.85],  # Cel.
        [1.00, 0.82, 0.05],  # Agent.
    ], dtype=np.float32)
    maze = np.clip(maze, 0, len(colors) - 1)
    return colors[maze]


def visualize_agents(seed: int, stage: int, base_n: int = 21, delay: float = 0.08):
    """
    szybka wizualizacja live

    nie finalny benchmark
    demo jak agent chodzi
    dobre na prezentacje bo widac zachowanie a nie tylko csv
    """
    import matplotlib.pyplot as plt

    env_factory, meta = make_env_for_stage(stage, base_n=base_n, seed=None)
    env_dqn = env_factory()
    env_ppo = env_factory()

    maze_seed = int(seed) + 10000 * int(stage)
    s_dqn = env_dqn.reset(seed=maze_seed)
    s_ppo = env_ppo.reset(seed=maze_seed)

    dqn = DQNAgent(
        state_dim=env_dqn.state_dim,
        n_actions=env_dqn.action_space.n,
        seed=seed,
        batch_size=32,
        min_memory=32,
        train_every=4,
        hidden=64,
    )
    ppo = PPOAgent(
        state_dim=env_ppo.state_dim,
        n_actions=env_ppo.action_space.n,
        seed=seed,
        batch_size=16,
        rollout_size=64,
        train_epochs=2,
        hidden=64,
    )

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im_dqn = axes[0].imshow(_maze_image(env_dqn), interpolation="nearest")
    im_ppo = axes[1].imshow(_maze_image(env_ppo), interpolation="nearest")
    axes[0].set_title("DQN")
    axes[1].set_title("PPO")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    done_dqn = False
    done_ppo = False
    reward_dqn = 0.0
    reward_ppo = 0.0

    for step in range(meta["max_steps"]):
        if not plt.fignum_exists(fig.number):
            break

        if not done_dqn:
            a = dqn.act(s_dqn, use_epsilon=True)
            s2, r, done_dqn, _info = env_dqn.step(a)
            dqn.remember(s_dqn, a, r, s2, done_dqn)
            dqn.replay()
            s_dqn = s2
            reward_dqn += float(r)

        if not done_ppo:
            a, logp, val = ppo.act_full(s_ppo, deterministic=False)
            s2, r, done_ppo, _info = env_ppo.step(a)
            ppo.remember(s_ppo, a, r, done_ppo, log_prob=logp, value=val)
            if ppo.should_update(done=done_ppo):
                ppo.replay(last_state=s2, done=done_ppo)
            s_ppo = s2
            reward_ppo += float(r)

        im_dqn.set_data(_maze_image(env_dqn))
        im_ppo.set_data(_maze_image(env_ppo))
        fig.suptitle(
            f"stage={meta['stage']} seed={seed} krok={step + 1}/{meta['max_steps']} "
            f"DQN={reward_dqn:.1f} PPO={reward_ppo:.1f}"
        )
        fig.canvas.draw_idle()
        plt.pause(delay)

        if done_dqn and done_ppo:
            break

    plt.ioff()
    plt.show()


def bfs_shortest_path(maze, start, goal):
    from collections import deque

    n = maze.shape[0]

    def passable(i, j):
        v = int(maze[i, j])
        return v in (0, 3, 4)

    q = deque([start])
    parent = {start: None}

    while q:
        i, j = q.popleft()
        if (i, j) == goal:
            break
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in parent and passable(ni, nj):
                parent[(ni, nj)] = (i, j)
                q.append((ni, nj))

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _bfs_image(maze, path):
    draw = maze.copy()
    if path:
        for i, j in path:
            if int(draw[i, j]) == 0:
                draw[i, j] = 6

    colors = np.array([
        [1.00, 1.00, 1.00],  # Pusta sciezka.
        [0.05, 0.05, 0.05],  # Sciana.
        [0.85, 0.12, 0.12],  # Pulapka.
        [0.20, 0.65, 0.25],  # Start.
        [0.10, 0.30, 0.85],  # Cel.
        [1.00, 0.82, 0.05],  # Agent.
        [0.97, 0.80, 0.10],  # Sciezka BFS.
    ], dtype=np.float32)
    draw = np.clip(draw, 0, len(colors) - 1)
    return colors[draw]


def show_bfs_path(seed: int, stage: int, base_n: int = 21):
    import matplotlib.pyplot as plt

    env_factory, meta = make_env_for_stage(stage, base_n=base_n, seed=seed)
    env = env_factory()
    maze = env.get_maze().copy()
    start = tuple(env.start)
    goal = tuple(env.goal)
    path = bfs_shortest_path(maze, start, goal)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(_bfs_image(maze, path), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"BFS | stage={meta['stage']} seed={seed} n={meta['n']} "
        f"max_steps={meta['max_steps']} path_len={len(path) if path else 'NONE'}"
    )

    if path:
        ys = [i for i, _j in path]
        xs = [j for _i, j in path]
        ax.plot(xs, ys, color="black", linewidth=1.4, alpha=0.75)

    plt.tight_layout()
    plt.show()


def main():
    print("Wybierz tryb:", flush=True)
    print("1 = tylko DQN", flush=True)
    print("2 = tylko PPO", flush=True)
    print("3 = porównanie równoległe: DQN i PPO", flush=True)
    print("4 = pokaż drogę BFS", flush=True)
    print("5 = aktywna wizualizacja DQN i PPO", flush=True)
    print("6 = porównanie wielu seedów równolegle", flush=True)
    choice = input("> ").strip()

    if choice == "6":
        seeds = ask_seed_list("Seedy")
        train_many_seeds(seeds=seeds, parallel=True)
        return

    if choice not in ("1", "2", "3", "4", "5"):
        print("Nieznany wybór.", flush=True)
        return

    seed = ask_int("Seed [123] > ", default=123)

    if choice == "1":
        train_by_steps("dqn", seed=seed, eval_seed=seed + 1000)
        return

    if choice == "2":
        train_by_steps("ppo", seed=seed, eval_seed=seed + 1000)
        return

    if choice == "3":
        out_dir = train_compare_run(seed=seed, parallel=True)
        print(f"Gotowe. Porównawcze CSV: {out_dir}", flush=True)
        return

    if choice == "4":
        stage = ask_int("Stage [0-5] > ", default=0, min_value=0, max_value=5)
        show_bfs_path(seed=seed, stage=stage)
        return

    if choice == "5":
        stage = ask_int("Stage [0-5] > ", default=0, min_value=0, max_value=5)
        visualize_agents(seed=seed, stage=stage)
        return

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()