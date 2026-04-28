import numpy as np
from collections import deque

"""
generator labiryntow do eksperymentow

dfs jako baza labiryntu
troche petli zeby nie bylo samego drzewa
start i meta
bfs path jako check czy da sie przejsc
pulapki poza najkrotsza sciezka

Generator tworzy trudne, ale wykonalne labirynty.
agent moze przegrac ale nie przez impossible level
"""

# 0 – wolne pole
# 1 – ściana
# 2 – pułapka
# 3 – start
# 4 – meta


def _can_place_trap(maze, i, j):
    n = maze.shape[0]

    # Ograniczenie skupisk pulapek.
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and maze[ni, nj] == 2:
                return False

    # Ograniczenie pulapek w zbyt ciasnych odnogach.
    wall_count = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n and maze[ni, nj] == 1:
            wall_count += 1

    return wall_count < 3


def _bfs_path(maze, start, goal, blocked_values=(1,)):
    """
    bfs jako najkrotsza sciezka
    none jak brak przejscia
    blocked values jako sciany i inne blokady
    """
    n = maze.shape[0]
    si, sj = start
    gi, gj = goal
    blocked = set(blocked_values)

    q = deque([(si, sj)])
    parent = {(si, sj): None}

    while q:
        i, j = q.popleft()

        if (i, j) == (gi, gj):
            path = []
            cur = (i, j)
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if not (0 <= ni < n and 0 <= nj < n):
                continue
            if (ni, nj) in parent:
                continue
            if int(maze[ni, nj]) in blocked:
                continue

            parent[(ni, nj)] = (i, j)
            q.append((ni, nj))

    return None


def _carve_dfs_maze(n, rng, corridor_bias=0.0):
    """
    bazowy dfs maze
    grid zero jeden
    jeden jako sciana
    zero jako przejscie
    corridor bias jako dluzsze korytarze
    """
    maze = np.ones((n, n), dtype=int)

    def make_odd(x):
        if x % 2 == 0:
            return x + 1 if x + 1 < n else x - 1
        return x

    start_i = make_odd(n - 2)
    start_j = make_odd(1)

    maze[start_i, start_j] = 0

    stack = [(start_i, start_j)]
    last_dir = None

    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def inb(i, j):
        return 0 < i < n - 1 and 0 < j < n - 1

    while stack:
        ci, cj = stack[-1]

        neighbors = []
        for di, dj in dirs:
            ni, nj = ci + di, cj + dj
            if inb(ni, nj) and maze[ni, nj] == 1:
                neighbors.append((di, dj, ni, nj))

        if not neighbors:
            stack.pop()
            last_dir = None
            continue

        if last_dir is not None and corridor_bias > 0:
            same_dir = [x for x in neighbors if (x[0], x[1]) == last_dir]
            if same_dir and rng.random() < corridor_bias:
                di, dj, ni, nj = same_dir[0]
            else:
                di, dj, ni, nj = neighbors[rng.integers(len(neighbors))]
        else:
            di, dj, ni, nj = neighbors[rng.integers(len(neighbors))]

        wall_i = ci + di // 2
        wall_j = cj + dj // 2
        maze[wall_i, wall_j] = 0
        maze[ni, nj] = 0

        stack.append((ni, nj))
        last_dir = (di, dj)

    return maze


def _add_loops(maze, rng, loop_factor=0.05):
    """
    dodanie petli przez przebijanie scian
    loop factor jako ile scian ruszamy
    """
    n = maze.shape[0]
    candidates = []

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if maze[i, j] != 1:
                continue

            if maze[i, j - 1] == 0 and maze[i, j + 1] == 0:
                candidates.append((i, j))
                continue

            if maze[i - 1, j] == 0 and maze[i + 1, j] == 0:
                candidates.append((i, j))
                continue

    rng.shuffle(candidates)

    k = int(loop_factor * (n * n))
    k = max(0, min(k, len(candidates)))

    for idx in range(k):
        i, j = candidates[idx]
        maze[i, j] = 0


def _ensure_open_cell(maze, cell):
    """
    gwarancja przejscia w danym punkcie
    """
    n = maze.shape[0]
    i, j = cell
    maze[i, j] = 0

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:
            maze[ni, nj] = 0


def generate_maze_with_path(
    n=21,
    p_trap=0.0,
    tunnels_factor=1.0,
    max_tunnel_factor=0.5,
    rng=None,
):
    """
    Generator DFS labiryntu + pętle + pułapki,
    ale z gwarancją, że pułapki NIE trafią na najkrótszą ścieżkę BFS.

    safety check dla eksperymentu
    agent moze przegrac ale nie przez totalnie nieuczciwa plansze
    """
    if rng is None:
        rng = np.random.default_rng()

    # Mapowanie parametrow na aktualna konfiguracje generatora.
    loop_factor = np.clip(0.02 * tunnels_factor, 0.0, 0.15)
    corridor_bias = np.clip(max_tunnel_factor, 0.0, 0.90)

    # 1) Bazowy labirynt DFS.
    maze = _carve_dfs_maze(n, rng, corridor_bias=corridor_bias)

    # 2) Dodanie petli.
    _add_loops(maze, rng, loop_factor=loop_factor)

    # 3) Wyznaczenie startu i mety.
    start = (n - 2, 1)
    goal = (1, n - 2)

    _ensure_open_cell(maze, start)
    _ensure_open_cell(maze, goal)

    # Najkrotsza sciezka pozostaje przechodnia.
    path = _bfs_path(maze, start, goal, blocked_values=(1,))
    path_set = set(path) if path is not None else set()

    # 5) Pulapki na polach przechodnich poza sciezka BFS.
    for i in range(n):
        for j in range(n):
            if (i, j) == start or (i, j) == goal:
                continue

            # Najkrotsza sciezka bez pulapek.
            if (i, j) in path_set:
                continue

            if maze[i, j] == 0 and rng.random() < p_trap:
                if _can_place_trap(maze, i, j):
                    maze[i, j] = 2

    # 6) Oznaczenie startu i mety.
    maze[start] = 3
    maze[goal] = 4

    return maze, start, goal
