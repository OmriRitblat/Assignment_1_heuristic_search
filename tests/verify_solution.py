"""Black-box verification of the flat-board solution.

For many random small boards it builds the FULL reachable state graph with an
independent re-implementation of the game rules, computes the exact optimal
cost-to-goal h* of every state, and checks through the public API only:

  * get_neighbors()  == the independent simulator (same set of boards)
  * get_state_board() round-trips the input board
  * both heuristics are admissible (h <= h*; h == inf on a solvable state counts as
    inadmissible), and advanced is consistent (h(s) <= 1 + h(s') on every edge)
  * A* with each heuristic returns the optimal cost (or None if unsolvable)

Board families: 'valid' (#cubes == #holes, rectangular) plus boards that break
the assignment's assumptions: more cubes, more holes, ragged rows.

usage:  python3 tests/verify_solution.py [seed] [boards_per_family]
"""
import math
import os
import random
import sys
from collections import deque

# SOLUTION_DIR lets the same checks run against another copy of the solution
sys.path.insert(0, os.environ.get("SOLUTION_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "flat_board_state"))
from flat_board_state import flat_board_state          # noqa: E402
from heuristics import base_heuristic, advanced_heuristic  # noqa: E402
import search as S                                      # noqa: E402

HOLE, FLOOR, WALL, CUBE = 0, 1, 2, 3
DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
INF = math.inf
STATE_CAP = 3000


# ---------- independent simulator, works directly on grids --------------------
def sim_move(grid, d):
    dr, dc = d
    rows = len(grid)
    cubes = {(r, c) for r in range(rows) for c in range(len(grid[r])) if grid[r][c] == CUBE}

    def free_cell(r, c):
        return 0 <= r < rows and 0 <= c < len(grid[r]) and grid[r][c] != WALL

    memo = {}

    def can(p):                      # recursive: can the cube at p move?
        if p not in memo:
            t = (p[0] + dr, p[1] + dc)
            if not free_cell(*t):
                memo[p] = False
            elif t in cubes:
                memo[p] = can(t)
            else:
                memo[p] = True
        return memo[p]

    new = [row[:] for row in grid]
    movers = [p for p in cubes if can(p)]
    for r, c in movers:
        new[r][c] = FLOOR
    for r, c in movers:
        tr, tc = r + dr, c + dc
        new[tr][tc] = FLOOR if grid[tr][tc] == HOLE else CUBE
    return new


def key(grid):
    return tuple(map(tuple, grid))


def has_holes(grid):
    return any(v == HOLE for row in grid for v in row)


def explore(grid):
    start = key(grid)
    edges, q = {}, deque([start])
    seen = {start}
    while q:
        s = q.popleft()
        g = [list(r) for r in s]
        succ = set()
        for d in DIRS:
            n = key(sim_move(g, d))
            if n != s:
                succ.add(n)
                if n not in seen:
                    if len(seen) >= STATE_CAP:
                        return None
                    seen.add(n)
                    q.append(n)
        edges[s] = succ
    rev = {s: [] for s in seen}
    for s, ss in edges.items():
        for n in ss:
            rev[n].append(s)
    hstar = {s: INF for s in seen}
    q = deque()
    for s in seen:
        if not has_holes(s):
            hstar[s] = 0
            q.append(s)
    while q:
        s = q.popleft()
        for p in rev[s]:
            if hstar[p] == INF:
                hstar[p] = hstar[s] + 1
                q.append(p)
    return hstar, edges


# ---------- random boards --------------------------------------------------------
def rand_board(family):
    while True:
        R, C = random.randint(1, 5), random.randint(2, 5)
        wall_p = random.choice([0, 0.1, 0.25])
        grid = [[WALL if random.random() < wall_p else FLOOR for _ in range(C)] for _ in range(R)]
        if family == "ragged":
            grid = [row[:random.randint(1, C)] for row in grid]
        free = [(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] == FLOOR]
        k = random.randint(1, 3)
        nc, nh = {"valid": (k, k), "ragged": (k, k),
                  "more_cubes": (k + random.randint(1, 2), k),
                  "more_holes": (k, k + random.randint(1, 2))}[family]
        if len(free) < nc + nh:
            continue
        random.shuffle(free)
        for r, c in free[:nc]:
            grid[r][c] = CUBE
        for r, c in free[nc:nc + nh]:
            grid[r][c] = HOLE
        return grid


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    per_family = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    random.seed(seed)
    failures = 0
    for family in ("valid", "more_cubes", "more_holes", "ragged"):
        st = dict(boards=0, states=0, unsolvable=0, neighbor_mismatch=0, roundtrip=0,
                  inadmissible_base=0, inadmissible_adv=0, inconsistent_adv=0,
                  nonoptimal_base=0, nonoptimal_adv=0, crashes=0)
        while st["boards"] < per_family:
            grid = rand_board(family)
            res = explore(grid)
            if res is None:
                continue
            hstar, edges = res
            st["boards"] += 1
            try:
                hadv = {}
                for s, hs in hstar.items():
                    g = [list(r) for r in s]
                    state = flat_board_state(g)
                    st["states"] += 1
                    if state.get_state_board() != g:
                        st["roundtrip"] += 1
                    mine = {key(n.get_state_board()) for n, cost in state.get_neighbors()}
                    if mine != edges[s]:
                        st["neighbor_mismatch"] += 1
                    hb, ha = base_heuristic(state), advanced_heuristic(state)
                    hadv[s] = ha
                    if hb > hs:
                        st["inadmissible_base"] += 1
                    if ha > hs:
                        st["inadmissible_adv"] += 1
                for s in hstar:
                    for n in edges[s]:
                        if hadv[s] > 1 + hadv[n]:
                            st["inconsistent_adv"] += 1
                opt = hstar[key(grid)]
                if opt == INF:
                    st["unsolvable"] += 1
                for name, h in (("base", base_heuristic), ("adv", advanced_heuristic)):
                    path = S.search(flat_board_state(grid), h)
                    got = path[-1].g if path else INF
                    if got != opt:
                        st["nonoptimal_" + name] += 1
            except Exception as e:                 # noqa: BLE001
                st["crashes"] += 1
                print("  crash:", type(e).__name__, e, grid)
        bad = sum(v for k, v in st.items() if k not in ("boards", "states", "unsolvable"))
        failures += bad
        print(f"{family:<11} {'OK ' if bad == 0 else 'FAIL'} {st}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
