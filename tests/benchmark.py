"""Runtime benchmark of one copy of the solution on a fixed set of boards.

  python3 tests/benchmark.py make  boards.json          # generate the board set
  SOLUTION_DIR=<dir> python3 tests/benchmark.py run boards.json out.json
  python3 tests/benchmark.py compare before.json after.json
  parallel chunks: SETS=large SLICE=0/2 ... run, then
  python3 tests/benchmark.py merge all.json part1.json part2.json ...

Each board is solved with base and advanced heuristic, through two drivers:
  ours    - search() from the solution's search.py
  foreign - an independent textbook A* that only uses the public API
            (get_neighbors / is_goal_state / __eq__ / __hash__ / heuristic),
            standing in for the course tester's own search.
Each run is capped at TIME_CAP seconds.
"""
import heapq
import itertools
import json
import os
import random
import signal
import sys
import time
from collections import deque

TIME_CAP = 20


# ---------------------------------------------------------------- board set ---
def rand_grid(R, C, nc, nh, wall_p, ragged=False):
    while True:
        g = [[2 if random.random() < wall_p else 1 for _ in range(C)] for _ in range(R)]
        if ragged:
            g = [row[:random.randint(max(1, C // 2), C)] for row in g]
        free = [(r, c) for r in range(len(g)) for c in range(len(g[r])) if g[r][c] == 1]
        if len(free) < nc + nh:
            continue
        random.shuffle(free)
        for r, c in free[:nc]:
            g[r][c] = 3
        for r, c in free[nc:nc + nh]:
            g[r][c] = 0
        return g


def sim_solvable(grid, cap=200000):
    """Exhaustive check with an independent simulator (None if too big)."""
    DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))
    rows = len(grid)

    def ok(r, c):
        return 0 <= r < rows and 0 <= c < len(grid[r]) and grid[r][c] != 2
    start = (frozenset((r, c) for r in range(rows) for c in range(len(grid[r])) if grid[r][c] == 3),
             frozenset((r, c) for r in range(rows) for c in range(len(grid[r])) if grid[r][c] == 0))
    seen, q = {start}, deque([start])
    while q:
        cubes, holes = q.popleft()
        if not holes:
            return True
        for dr, dc in DIRS:
            memo = {}

            def can(p):
                if p not in memo:
                    t = (p[0] + dr, p[1] + dc)
                    memo[p] = ok(*t) and (can(t) if t in cubes else True)
                return memo[p]
            nc, nh = set(), set(holes)
            for p in cubes:
                if can(p):
                    t = (p[0] + dr, p[1] + dc)
                    if t in holes:
                        nh.discard(t)
                    else:
                        nc.add(t)
                else:
                    nc.add(p)
            n = (frozenset(nc), frozenset(nh))
            if n not in seen:
                if len(seen) > cap:
                    return None
                seen.add(n)
                q.append(n)
    return False


def make(path):
    random.seed(2026)
    sets = {}
    sets["assignment_like"] = [rand_grid(R, R, k, k, random.choice([0.05, 0.15, 0.25]))
                               for R, k in ((random.randint(5, 10), random.randint(2, 4))
                                            for _ in range(60))]
    sets["large"] = [rand_grid(R, R, k, k, random.choice([0.1, 0.2]))
                     for R, k in ((random.randint(15, 30), random.randint(3, 6)) for _ in range(30))]
    sets["xlarge_40x40_60x60"] = [rand_grid(R, R, k, k, 0.15)
                                  for R, k in ((40, 3), (40, 4), (50, 3), (60, 3), (60, 4))]
    uns = []
    while len(uns) < 30:
        g = rand_grid(random.randint(4, 7), random.randint(4, 7), *(2 * [random.randint(2, 3)]),
                      random.choice([0.25, 0.35]))
        if sim_solvable(g) is False:
            uns.append(g)
    sets["unsolvable"] = uns
    sets["more_cubes"] = [rand_grid(R, R, k + random.randint(1, 3), k, 0.15)
                          for R, k in ((random.randint(6, 12), random.randint(1, 4)) for _ in range(30))]
    sets["more_holes"] = [rand_grid(R, R, k, k + random.randint(1, 2), 0.15)
                          for R, k in ((random.randint(6, 12), random.randint(1, 4)) for _ in range(30))]
    sets["ragged"] = [rand_grid(R, R, k, k, 0.15, ragged=True)
                      for R, k in ((random.randint(6, 12), random.randint(2, 4)) for _ in range(30))]
    with open(path, "w") as f:
        json.dump(sets, f)
    print({k: len(v) for k, v in sets.items()})


# ---------------------------------------------------------------- drivers -----
def foreign_astar(start, h, is_goal):
    tie = itertools.count()
    h0 = h(start)
    heap = [(h0, h0, next(tie), 0, start)]
    best_g = {start: 0}
    closed = set()
    gen = exp = 0
    while heap:
        f, hv, _, g, s = heapq.heappop(heap)
        if s in closed or best_g.get(s, g) < g:
            continue
        if is_goal(s):
            return g, gen, exp
        closed.add(s)
        exp += 1
        for n, c in s.get_neighbors():
            ng = g + c
            if n in closed or best_g.get(n, float("inf")) <= ng:
                continue
            best_g[n] = ng
            hn = h(n)
            gen += 1
            heapq.heappush(heap, (ng + hn, hn, next(tie), ng, n))
    return None, gen, exp


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def run(boards_path, out_path):
    sys.path.insert(0, os.environ["SOLUTION_DIR"])
    from flat_board_state import flat_board_state
    from heuristics import base_heuristic, advanced_heuristic
    import search as S
    signal.signal(signal.SIGALRM, _alarm)
    sets = json.load(open(boards_path))
    # optional parallel chunking: SETS=name1,name2  SLICE=i/k (every k-th board)
    only = set(filter(None, os.environ.get("SETS", "").split(",")))
    part, parts = map(int, os.environ.get("SLICE", "0/1").split("/"))
    out = {}
    for name, boards in sets.items():
        if only and name not in only:
            continue
        rows = []
        for idx, grid in enumerate(boards):
            if idx % parts != part:
                continue
            row = {}
            for hname, h in (("base", base_heuristic), ("adv", advanced_heuristic)):
                for driver in ("ours", "foreign"):
                    signal.alarm(TIME_CAP)
                    t = time.perf_counter()
                    try:
                        st = flat_board_state([r[:] for r in grid])
                        if driver == "ours":
                            p = S.search(st, h)
                            cost = p[-1].g if p else None
                            gen, exp = S.get_gen_exp_count()
                        else:
                            cost, gen, exp = foreign_astar(st, h, flat_board_state.is_goal_state)
                        signal.alarm(0)
                        r = dict(status="ok", cost=cost, time=time.perf_counter() - t, gen=gen, exp=exp)
                    except Timeout:
                        r = dict(status="timeout", time=TIME_CAP)
                    except Exception as e:                     # noqa: BLE001
                        signal.alarm(0)
                        r = dict(status="crash", error=f"{type(e).__name__}: {e}"[:80],
                                 time=time.perf_counter() - t)
                    row[f"{hname}/{driver}"] = r
            row["idx"] = idx
            rows.append(row)
            print(name, len(rows), {k: (v["status"], round(v["time"], 3), v.get("cost"))
                                    for k, v in row.items() if k != "idx"}, flush=True)
        out[name] = rows
    json.dump(out, open(out_path, "w"))


def merge(out_path, *parts):
    """Merge chunked result files back into one, in original board order."""
    merged = {}
    for path in parts:
        for name, rows in json.load(open(path)).items():
            merged.setdefault(name, []).extend(rows)
    for rows in merged.values():
        rows.sort(key=lambda r: r["idx"])
    json.dump(merged, open(out_path, "w"))


def compare(before_path, after_path):
    """Per board set and heuristic/driver: outcome counts, and the time on
    the boards both versions finished (timeouts counted as TIME_CAP, so a
    speed-up that involves a timeout is a lower bound)."""
    before, after = json.load(open(before_path)), json.load(open(after_path))
    for name in before:
        print(f"\n== {name} ({len(before[name])} boards)")
        for key in ("base/ours", "base/foreign", "adv/ours", "adv/foreign"):
            def outcome(rows):
                c = {"solved": 0, "unsolvable": 0, "timeout": 0, "crash": 0}
                for r in rows:
                    r = r[key]
                    if r["status"] == "ok":
                        c["solved" if r["cost"] is not None else "unsolvable"] += 1
                    else:
                        c[r["status"]] += 1
                return c
            b_rows, a_rows = before[name], after[name]
            pairs = [(b[key], a[key]) for b, a in zip(b_rows, a_rows)]
            both = [(b, a) for b, a in pairs if b["status"] == a["status"] == "ok"]
            mismatch = sum(1 for b, a in both if b["cost"] != a["cost"])
            tb = sum(b["time"] for b, _ in pairs)
            ta = sum(a["time"] for _, a in pairs)
            eb = sum(b["exp"] for b, _ in both)
            ea = sum(a["exp"] for _, a in both)
            tbb = sum(b["time"] for b, _ in both)
            tab = sum(a["time"] for _, a in both)
            print(f"  {key:<13} before {outcome(b_rows)}\n  {'':<13} after  {outcome(a_rows)}")
            print(f"  {'':<13} total time {tb:8.2f}s -> {ta:8.2f}s"
                  f"  | on {len(both)} boards both finished: {tbb:7.3f}s -> {tab:7.3f}s"
                  f" (x{tbb / tab if tab else float('inf'):.1f}), expanded {eb} -> {ea},"
                  f" cost mismatches {mismatch}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    elif sys.argv[1] == "merge":
        merge(sys.argv[2], *sys.argv[3:])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[2], sys.argv[3])
