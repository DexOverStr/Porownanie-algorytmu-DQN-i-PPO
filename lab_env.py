import numpy as np
from Generator_lab import generate_maze_with_path


"""
lab env srodowisko labiryntu

minimalny interfejs
reset jako start epizodu
step jako jeden ruch
action space jako liczba akcji

stan ma 6 liczb
pozycja agenta plus cztery pola obok
lekki input taki sam dla dqn i ppo
porownanie algorytmow
"""


class DiscreteActionSpace:
    """Minimalna przestrzen akcji zgodna z interfejsem ewaluacji."""
    def __init__(self, n):
        self.n = int(n)

    def sample(self):
        return int(np.random.randint(self.n))

AGENT_VALUE = 5  # Dane pomocnicze do wizualizacji.

_SYMBOL_MAP = {
    0: ' ',
    1: '#',
    2: 'X',
    3: 'S',
    4: 'G',
    5: 'A',
}


def _print_maze(maze, agent_pos=None):
    maze_to_print = maze.copy()

    if agent_pos is not None:
        x, y = agent_pos
        if maze_to_print[x, y] not in (3, 4):
            maze_to_print[x, y] = AGENT_VALUE

    for row in maze_to_print:
        print("".join(_SYMBOL_MAP.get(int(cell), '?') for cell in row))


class LabEnv:
    """
    akcje
    0 gora
    1 dol
    2 lewo
    3 prawo

    obserwacja
    x y plus pola gora dol lewo prawo
    razem 6 wartosci
    """

    def __init__(
        self,
        n=40,
        max_steps=None,
        max_lives=5,
        p_trap=0.0,
        tunnels_factor=1.0,
        max_tunnel_factor=0.5,
        step_penalty=-0.01,
        seed=None,
        rng=None,
    ):
        self.n = int(n)

        self.max_steps = int(max_steps) if max_steps is not None else 250
        self.step_penalty = float(step_penalty)

        # Parametry uzywane przez agentow i procedury treningowe.
        self.action_space = DiscreteActionSpace(4)
        self.n_actions = 4
        self.state_dim = 6

        self.maze = None
        self.start = None
        self.goal = None
        self.agent_pos = None
        self.steps = 0
        self.visited = set()

        self.max_lives = int(max_lives)
        self.lives = int(max_lives)

        self._gen_params = dict(
            p_trap=p_trap,
            tunnels_factor=tunnels_factor,
            max_tunnel_factor=max_tunnel_factor,
        )

        # Kontrola deterministycznosci.
        self.base_seed = int(seed) if seed is not None else None
        self.episode_idx = 0
        self.current_maze_seed = None

        # Bazowy generator liczb losowych.
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(self.base_seed)

        self.reset()

    def set_seed(self, seed: int):
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)
        self.episode_idx = 0
        self.current_maze_seed = None

    def get_current_maze_seed(self):
        return self.current_maze_seed

    def reset_same_maze(self):
        self.agent_pos = self.start
        self.steps = 0
        self.lives = self.max_lives
        self.visited = {self.start}
        return self._get_obs()

    def _get_obs(self):
        x, y = self.agent_pos
        n = self.n

        def get_cell(i, j):
            if 0 <= i < n and 0 <= j < n:
                return int(self.maze[i, j])
            return 1

        up = get_cell(x - 1, y)
        down = get_cell(x + 1, y)
        left = get_cell(x, y - 1)
        right = get_cell(x, y + 1)

        pos = np.array([x, y], dtype=np.float32) / (n - 1)
        neigh = np.array([up, down, left, right], dtype=np.float32) / 5.0

        obs = np.concatenate([pos, neigh], dtype=np.float32)
        # Zabezpieczenie pozycji startu.
        if obs.shape[0] != 6:
            obs = obs[:6]
        return obs

    def reset(self, seed=None):
        """
        reset env i nowy labirynt

        seed jako identyczna mapa
        base seed jako deterministyczna sekwencja
        bez seeda losowo
        """
        # Generator losowy dla aktualnego resetu.
        if seed is not None:
            s = int(seed)
            local_rng = np.random.default_rng(s)
            self.current_maze_seed = s

        elif self.base_seed is not None:
            # Sekwencja zaleznosci od indeksu epizodu.
            ss = np.random.SeedSequence([self.base_seed, self.episode_idx])
            local_rng = np.random.default_rng(ss)
            self.current_maze_seed = None

        else:
            local_rng = self.rng
            self.current_maze_seed = None

        # Aktualizacja licznika resetow.
        self.episode_idx += 1

        # Generacja labiryntu.
        self.maze, self.start, self.goal = generate_maze_with_path(
            self.n,
            rng=local_rng,
            **self._gen_params
        )

        self.agent_pos = self.start
        self.steps = 0
        self.visited = {self.start}
        self.lives = self.max_lives

        return self._get_obs()

    def step(self, action):
        # Ksztaltowanie funkcji nagrody.
        # Kara za wykonanie kroku.
        # Kara za sciane i pulapke.
        # Bonus za odwiedzenie nowego pola.
        # Bonus za osiagniecie celu.
        # Bonus za zmniejszenie odleglosci Manhattan.
        self.steps += 1
        x, y = self.agent_pos

        reward = self.step_penalty
        done = False
        info = {}

        gx, gy = self.goal
        dist_before = abs(x - gx) + abs(y - gy)

        # Wykonanie ruchu.
        if action == 0:
            nx, ny = x - 1, y
        elif action == 1:
            nx, ny = x + 1, y
        elif action == 2:
            nx, ny = x, y - 1
        elif action == 3:
            nx, ny = x, y + 1
        else:
            nx, ny = x, y

        new_pos = (x, y)

        # Logika ruchu i nagrody.
        if nx < 0 or nx >= self.n or ny < 0 or ny >= self.n:
            reward -= 0.1
        else:
            cell = int(self.maze[nx, ny])

            if cell == 1:
                reward -= 0.1

            elif cell == 2:
                reward -= 1.0
                self.lives -= 1

                if self.lives <= 0:
                    done = True
                    info["reason"] = "no_lives"
                    new_pos = (nx, ny)
                else:
                    new_pos = self.start
                    self.visited = {self.start}
                    info["trap"] = True
                    info["lives_left"] = self.lives

            elif cell == 4:
                reward += 150.0
                new_pos = (nx, ny)
                done = True
                info["reason"] = "goal"

            else:
                new_pos = (nx, ny)

                if new_pos in self.visited:
                    reward -= 0.05
                else:
                    reward += 1.0
                    self.visited.add(new_pos)

        self.agent_pos = new_pos

        # Ksztaltowanie nagrody na podstawie odleglosci Manhattan.
        x2, y2 = self.agent_pos
        dist_after = abs(x2 - gx) + abs(y2 - gy)
        # Korekta kierunku bez dominowania nagrody za cel.
        reward += 0.3 * (dist_before - dist_after)

        # Limit czasu epizodu.
        if self.steps >= self.max_steps and not done:
            done = True
            info["reason"] = "max_steps"

        next_state = self._get_obs()
        return next_state, reward, done, info

    def render(self):
        _print_maze(self.maze, self.agent_pos)

    def get_agent_position(self):
        return self.agent_pos

    def is_at_goal(self):
        return self.agent_pos == self.goal

    def get_maze(self):
        return self.maze

    def get_size(self):
        return self.n
