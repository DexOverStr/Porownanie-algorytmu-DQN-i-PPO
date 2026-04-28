"""
agregacja wynikow po eksperymentach

zbieranie katalogow compare seed
summary csv i xlsx
zero recznego klejenia tabelek
mini etl logs na tabelki pod prace
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path("logs_out")


def _seed_from_dir(path: Path):
    parts = path.name.split("_")
    try:
        idx = parts.index("seed")
        return int(parts[idx + 1])
    except Exception:
        return None


def _read_config(path: Path):
    config_path = path / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_evaluations(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """wczytanie tylko wierszy ewaluacyjnych"""
    rows = []
    for run_dir in sorted(base_dir.glob("compare_seed_*")):
        if not run_dir.is_dir():
            continue

        config = _read_config(run_dir)
        seed = config.get("seed", _seed_from_dir(run_dir))

        for algo in ("dqn", "ppo"):
            csv_path = run_dir / f"logs_{algo}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if "eval_success_rate" not in df.columns:
                continue

            evals = df[df["eval_success_rate"].notna()].copy()
            if evals.empty:
                continue

            if "eval_episodes" not in evals.columns:
                evals["eval_episodes"] = config.get("eval_episodes")

            evals["seed"] = seed
            evals["algo"] = algo
            evals["run_dir"] = str(run_dir)
            rows.append(evals)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize(evals: pd.DataFrame):
    """srednie i odchylenia po seedach osobno dla dqn i ppo"""
    metric_cols = ["eval_success_rate", "eval_avg_reward", "eval_avg_steps"]

    by_step = (
        evals
        .groupby(["algo", "env_steps_total", "stage"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            eval_episodes=("eval_episodes", "max"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
        )
    )

    last = (
        evals
        .sort_values(["seed", "algo", "env_steps_total"])
        .groupby(["seed", "algo"], as_index=False)
        .tail(1)
    )

    final = (
        last
        .groupby("algo", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            eval_episodes=("eval_episodes", "max"),
            final_success_rate_mean=("eval_success_rate", "mean"),
            final_success_rate_std=("eval_success_rate", "std"),
            final_avg_reward_mean=("eval_avg_reward", "mean"),
            final_avg_reward_std=("eval_avg_reward", "std"),
            final_avg_steps_mean=("eval_avg_steps", "mean"),
            final_avg_steps_std=("eval_avg_steps", "std"),
        )
    )

    return by_step, final, last


def main():
    evals = load_evaluations()
    if evals.empty:
        print("Brak danych ewaluacyjnych w logs_out/compare_seed_*.")
        return

    by_step, final, last = summarize(evals)

    out_dir = BASE_DIR / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    evals.to_csv(out_dir / "all_evaluations.csv", index=False)
    by_step.to_csv(out_dir / "summary_by_step.csv", index=False)
    final.to_csv(out_dir / "summary_final.csv", index=False)
    last.to_csv(out_dir / "final_per_seed.csv", index=False)

    with pd.ExcelWriter(out_dir / "summary.xlsx") as writer:
        final.to_excel(writer, sheet_name="final_summary", index=False)
        by_step.to_excel(writer, sheet_name="by_step", index=False)
        last.to_excel(writer, sheet_name="final_per_seed", index=False)

    print(f"Zapisano podsumowania w: {out_dir}")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
