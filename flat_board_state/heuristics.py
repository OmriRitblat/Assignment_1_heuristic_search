import math

INF = math.inf


def base_heuristic(_flat_board_state):
    # For every hole: the Manhattan distance to its closest cube.
    # The heuristic is the maximum of those distances over all holes.
    holes = _flat_board_state.holes
    if not holes:
        return 0
    cubes = _flat_board_state.cubes
    if len(cubes) < len(holes):
        # every cube fills at most one hole: this state can never be solved
        return INF
    coords = _flat_board_state.board.coords
    cube_rc = [coords[i] for i in cubes]
    max_dist = 0
    for h in holes:
        hr, hc = coords[h]
        nearest = min(abs(hr - cr) + abs(hc - cc) for cr, cc in cube_rc)
        if nearest > max_dist:
            max_dist = nearest
    return max_dist


def advanced_heuristic(_flat_board_state):
    # Lower bound on the number of moves still needed, built from per-pair
    # tables precomputed once per hole (walls respected):
    #
    #   bfs(c, h)  shortest wall-respecting walk from cube c to hole h
    #   D(c, h)    fewest steps in direction D (right/left/up/down) on any
    #              such walk
    #
    # In any solution every hole h is filled by a DIFFERENT cube M(h)
    # (a cube disappears into the hole it fills).  For that assignment M:
    #   * the solution has at least  max_h bfs(M(h), h)  moves, because a cube
    #     advances at most one cell per move, and
    #   * at least  max_h D(M(h), h)  moves in direction D, because only a
    #     D-move gives a cube a D-step.  The four directions are different
    #     moves, so the solution has at least  sum_D max_h D(M(h), h)  moves.
    # We don't know M, so we take the minimum over all one-to-one
    # assignments of  max( max_h bfs , sum_D max_h D ).  This is admissible
    # and consistent, and dominates both the old "nearest cube" terms and
    # the Manhattan base heuristic.  It also returns INF (a dead state) when
    # no assignment exists, e.g. a hole no remaining cube can reach.
    #
    # The exact minimum is a small search over assignments; with many cubes
    # that could explode, so above EXACT_MAX_CUBES cubes we use a weaker
    # polynomial bound (_bottleneck_bound).  The choice depends only on the
    # number of cubes, which never grows along a path, so a path can switch
    # from the weaker bound to the exact one but never back - this keeps the
    # heuristic consistent (weaker <= exact on the same state).
    holes = _flat_board_state.holes
    if not holes:
        return 0
    # A state generated again through another path is a duplicate: its value
    # is already known, so one dictionary lookup replaces the evaluation.
    cache = _flat_board_state.board.h_cache
    cached = cache.get(_flat_board_state)
    if cached is None:
        cached = cache[_flat_board_state] = _advanced_value(_flat_board_state)
    return cached


def _advanced_value(_flat_board_state):
    holes = _flat_board_state.holes
    cubes = _flat_board_state.cubes
    m, n = len(holes), len(cubes)
    if n < m:
        return INF
    dist = _flat_board_state.board.dist

    if m == 1:
        # one hole: the best single cube
        table = dist[next(iter(holes))]
        best = INF
        for c in cubes:
            bv, r, l, u, d = table[c]
            v = r + l + u + d
            if bv > v:
                v = bv
            if v < best:
                best = v
        return best

    cube_list = list(cubes)
    # rows[k] = candidate (cost tuple, cube position in cube_list) for hole k,
    # cheapest first, unreachable cubes dropped
    rows = []
    lower = 0
    for h in holes:
        table = dist[h]
        cand = []
        for j, c in enumerate(cube_list):
            t = table[c]
            if t[0] != INF:
                cand.append((max(t[0], t[1] + t[2] + t[3] + t[4]), t, j))
        if not cand:
            return INF                  # nobody can reach this hole
        cand.sort()
        if cand[0][0] > lower:
            lower = cand[0][0]          # this hole alone needs at least that
        rows.append(cand)
    # most constrained hole first: fewer branches, earlier pruning
    rows.sort(key=lambda cand: (len(cand), -cand[0][0]))

    if n > EXACT_MAX_CUBES:
        return _bottleneck_bound(rows, m, n)

    best = [INF]

    def assign(k, used, b, r, l, u, d):
        # returns True to stop the search (lower bound reached = optimum)
        if k == m:
            v = r + l + u + d
            best[0] = b if b > v else v
            return best[0] <= lower     # cannot do better than the lower bound
        for _, (bv, rv, lv, uv, dv), j in rows[k]:
            if used >> j & 1:
                continue
            nb = b if b > bv else bv
            nr = r if r > rv else rv
            nl = l if l > lv else lv
            nu = u if u > uv else uv
            nd = d if d > dv else dv
            v = nr + nl + nu + nd
            if nb > v:
                v = nb
            if v >= best[0]:            # adding holes never lowers the value
                continue
            if assign(k + 1, used | (1 << j), nb, nr, nl, nu, nd):
                return True
        return False

    assign(0, 0, 0, 0, 0, 0, 0)
    return best[0]


# Above this many cubes the exact assignment search is replaced by the
# polynomial bottleneck bound (see advanced_heuristic).
EXACT_MAX_CUBES = 7


def _bottleneck_bound(rows, m, n):
    """max( B_bfs , B_right + B_left + B_up + B_down ) where B_x is the
    smallest t such that every hole can get a different cube with x <= t.
    Admissible: the real assignment M has  max_h x(M(h), h) >= B_x  for every
    metric x, so the bounds above still hold with B_x in place of the max."""
    def bottleneck(metric):
        values = sorted({t[metric] for cand in rows for _, t, _ in cand})
        # any assignment is at least the worst hole's cheapest option
        floor = max(min(t[metric] for _, t, _ in cand) for cand in rows)
        for limit in values:
            if limit >= floor and _has_assignment(rows, m, n, metric, limit):
                return limit
        return INF

    b = bottleneck(0)
    if b == INF:
        return INF
    s = bottleneck(1) + bottleneck(2) + bottleneck(3) + bottleneck(4)
    return b if b > s else s


def _has_assignment(rows, m, n, metric, limit):
    """Can every hole get a distinct cube whose `metric` is <= limit?
    (Kuhn's augmenting-path bipartite matching.)"""
    adj = [[j for _, t, j in cand if t[metric] <= limit] for cand in rows]
    owner = [-1] * n

    def augment(k, seen):
        for j in adj[k]:
            if not seen[j]:
                seen[j] = True
                if owner[j] < 0 or augment(owner[j], seen):
                    owner[j] = k
                    return True
        return False

    for k in range(m):
        if not augment(k, [False] * n):
            return False
    return True
