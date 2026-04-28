from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_utf8_stdio()


MODES = [
    "DQN train",
    "PPO train",
    "Porównanie równoległe DQN i PPO",
    "Pokaż drogę BFS",
    "Aktywna wizualizacja DQN i PPO",
    "Porównanie wielu seedów równolegle",
]

MODE_IDS = {
    "DQN train": "dqn",
    "PPO train": "ppo",
    "Porównanie równoległe DQN i PPO": "compare_parallel",
    "Pokaż drogę BFS": "bfs",
    "Aktywna wizualizacja DQN i PPO": "visualize",
    "Porównanie wielu seedów równolegle": "many_parallel",
}


class LabiryntApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Labirynt - DQN/PPO")
        self.geometry("920x640")
        self.minsize(760, 520)

        self.process: subprocess.Popen | None = None
        self.reader_thread: threading.Thread | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.mode_var = tk.StringVar(value=MODES[0])
        self.seed_var = tk.StringVar(value="123")
        self.stage_var = tk.StringVar(value="0")
        self.seeds_var = tk.StringVar(value="111,222,333,444,555")
        self.status_var = tk.StringVar(value="Gotowe")

        self._build_ui()
        self.after(100, self._poll_output)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(6, weight=1)

        ttk.Label(root, text="Tryb").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        mode = ttk.Combobox(root, textvariable=self.mode_var, values=MODES, state="readonly")
        mode.grid(row=0, column=1, sticky="ew", pady=4)
        mode.bind("<<ComboboxSelected>>", lambda _event: self._sync_inputs())

        ttk.Label(root, text="Seed").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        self.seed_entry = ttk.Entry(root, textvariable=self.seed_var)
        self.seed_entry.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(root, text="Stage").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        self.stage_spin = ttk.Spinbox(root, from_=0, to=5, textvariable=self.stage_var, width=8)
        self.stage_spin.grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(root, text="Seedy").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=4)
        self.seeds_entry = ttk.Entry(root, textvariable=self.seeds_var)
        self.seeds_entry.grid(row=3, column=1, sticky="ew", pady=4)

        controls = ttk.Frame(root)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        controls.columnconfigure(3, weight=1)

        self.start_btn = ttk.Button(controls, text="Start", command=self.start_process)
        self.start_btn.grid(row=0, column=0, padx=(0, 8))

        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop_process, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))

        ttk.Button(controls, text="Wyczyść log", command=self.clear_log).grid(row=0, column=2, padx=(0, 8))
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=3, sticky="e")

        self.help_label = ttk.Label(root, text="", foreground="#444")
        self.help_label.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        log_frame = ttk.Frame(root)
        log_frame.grid(row=6, column=0, columnspan=2, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, wrap="word", height=20, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

        self._sync_inputs()

    def _sync_inputs(self):
        mode = self.mode_var.get()
        needs_stage = mode in ("Pokaż drogę BFS", "Aktywna wizualizacja DQN i PPO")
        needs_many_seeds = mode == "Porównanie wielu seedów równolegle"

        self.stage_spin.configure(state="normal" if needs_stage else "disabled")
        self.seed_entry.configure(state="disabled" if needs_many_seeds else "normal")
        self.seeds_entry.configure(state="normal" if needs_many_seeds else "disabled")

        descriptions = {
            "DQN train": "Trenuje tylko DQN i zapisuje logs_out/logs_dqn.csv.",
            "PPO train": "Trenuje tylko PPO i zapisuje logs_out/logs_ppo.csv.",
            "Porównanie równoległe DQN i PPO": "Uruchamia DQN i PPO jednocześnie dla jednego seeda.",
            "Pokaż drogę BFS": "Pokazuje najkrótszą drogę BFS dla wybranego seed i stage.",
            "Aktywna wizualizacja DQN i PPO": "Pokazuje live, jak DQN i PPO poruszają się po tej samej mapie.",
            "Porównanie wielu seedów równolegle": "Uruchamia porównania równoległe dla listy seedów.",
        }
        self.help_label.configure(text=descriptions.get(mode, ""))

    def start_process(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("Proces działa", "Najpierw zatrzymaj aktualny proces.")
            return

        try:
            command = self._build_worker_command()
        except ValueError as exc:
            messagebox.showerror("Błędne dane", str(exc))
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._append_log(f"\n=== START: {self.mode_var.get()} ===\n")
        self.status_var.set("Działa")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self.process = subprocess.Popen(
            command,
            cwd=get_app_dir(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=creationflags,
        )

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _build_worker_command(self) -> list[str]:
        mode = self.mode_var.get()
        mode_id = MODE_IDS.get(mode)
        if mode_id is None:
            raise ValueError("Nieznany tryb.")

        if mode_id == "many_parallel":
            seeds = self._parse_seed_list(self.seeds_var.get())
            return make_worker_command(["--worker", mode_id, "--seeds", ",".join(str(s) for s in seeds)])

        seed = self._parse_int(self.seed_var.get(), "Seed")
        args = ["--worker", mode_id, "--seed", str(seed)]

        if mode_id in ("bfs", "visualize"):
            stage = self._parse_int(self.stage_var.get(), "Stage")
            if stage < 0 or stage > 5:
                raise ValueError("Stage musi być w zakresie 0-5.")
            args.extend(["--stage", str(stage)])

        return make_worker_command(args)

    def _parse_int(self, raw: str, name: str) -> int:
        raw = raw.strip()
        if not raw:
            raise ValueError(f"{name} nie może być puste.")
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} musi być liczbą całkowitą.") from exc

    def _parse_seed_list(self, raw: str) -> list[int]:
        raw = raw.strip()
        if not raw:
            raise ValueError("Podaj co najmniej jeden seed.")
        try:
            seeds = [int(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("Seedy podaj jako liczby, np. 111,222,333.") from exc
        if not seeds:
            raise ValueError("Podaj co najmniej jeden seed.")
        return seeds

    def _read_process_output(self):
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            self.output_queue.put(line)
        return_code = process.wait()
        self.output_queue.put(f"\n=== KONIEC procesu: kod {return_code} ===\n")
        self.output_queue.put("__PROCESS_DONE__")

    def _poll_output(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item == "__PROCESS_DONE__":
                    self._process_done()
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def _append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def stop_process(self):
        process = self.process
        if process is None or process.poll() is not None:
            self._process_done()
            return

        self._append_log("\n=== STOP: zatrzymywanie procesu ===\n")
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.send_signal(signal.SIGTERM)
        except Exception as exc:
            self._append_log(f"Nie udało się zatrzymać procesu: {exc}\n")

    def _process_done(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Gotowe")
        self.process = None

    def _on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Zamknąć?", "Proces nadal działa. Zatrzymać go i zamknąć aplikację?"):
                return
            self.stop_process()
        self.destroy()


def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def make_worker_command(args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-u", os.path.abspath(__file__), *args]


def parse_worker_args(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        key = argv[i]
        if key == "--worker":
            if i + 1 >= len(argv):
                raise SystemExit("Brak trybu workera.")
            out["worker"] = argv[i + 1]
            i += 2
            continue
        if key in ("--seed", "--stage", "--seeds"):
            if i + 1 >= len(argv):
                raise SystemExit(f"Brak wartości dla {key}.")
            out[key[2:]] = argv[i + 1]
            i += 2
            continue
        i += 1
    return out


def run_worker(argv: list[str]) -> int:
    opts = parse_worker_args(argv)
    mode = opts.get("worker")
    if not mode:
        raise SystemExit("Brak trybu workera.")

    import train_compare

    if mode == "dqn":
        seed = int(opts.get("seed", "123"))
        train_compare.train_by_steps("dqn", seed=seed, eval_seed=seed + 1000)
        return 0

    if mode == "ppo":
        seed = int(opts.get("seed", "123"))
        train_compare.train_by_steps("ppo", seed=seed, eval_seed=seed + 1000)
        return 0

    if mode == "compare_parallel":
        seed = int(opts.get("seed", "123"))
        train_compare.train_compare_run(seed=seed, parallel=True)
        return 0

    if mode == "many_parallel":
        raw = opts.get("seeds", "111,222,333,444,555")
        seeds = [int(part.strip()) for part in raw.replace(";", ",").split(",") if part.strip()]
        train_compare.train_many_seeds(seeds=seeds, parallel=True)
        return 0

    if mode == "bfs":
        seed = int(opts.get("seed", "123"))
        stage = int(opts.get("stage", "0"))
        train_compare.show_bfs_path(seed=seed, stage=stage)
        return 0

    if mode == "visualize":
        seed = int(opts.get("seed", "123"))
        stage = int(opts.get("stage", "0"))
        train_compare.visualize_agents(seed=seed, stage=stage)
        return 0

    raise SystemExit(f"Nieznany tryb workera: {mode}")


def main() -> int:
    multiprocessing.freeze_support()
    if "--worker" in sys.argv:
        return run_worker(sys.argv[1:])
    app = LabiryntApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())