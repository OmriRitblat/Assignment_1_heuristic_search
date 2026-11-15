import math


def base_heuristic(_flat_board_state):
    # For every hole: the Manhattan distance to its closest cube.
    # The heuristic is the maximum of those distances over all holes.
    cubes = _flat_board_state.cubes
    holes = _flat_board_state.holes
    if not holes:
        return 0
    max_dist = 0
    for hr, hc in holes:
        nearest = min(abs(hr - cr) + abs(hc - cc) for cr, cc in cubes)
        if nearest > max_dist:
            max_dist = nearest
    return max_dist


def advanced_heuristic(_flat_board_state):
    # h = max( BFS term , directional term )
    #
    # BFS term:  h1 = max over holes of ( min over cubes of BFS_dist(cube, hole) )
    #            BFS distances respect walls, precomputed once per hole.
    #
    # Directional term: a cube advances at most one cell per move in a given
    #            direction, so the solution needs at least D* moves in each
    #            direction D, and the four directions are different moves:
    #            D(cube, hole) = fewest D-moves on any wall-respecting path
    #                            from the cube to the hole (precomputed table)
    #            D_holes = max over holes of ( min over cubes of D(cube, hole) )
    #            D_cubes = max over cubes of ( min over holes of D(cube, hole) )
    #            D*      = max(D_holes, D_cubes)
    #            h2 = right* + left* + up* + down*
    #
    # Both terms are computed in ONE pass over the (hole, cube) pairs, with a
    # single table lookup per pair: (bfs_dist, right, left, up, down).
    holes = _flat_board_state.holes
    if not holes:
        return 0
    # A state generated again through another path is a duplicate: its value
    # is already known, so one dictionary lookup replaces the evaluation.
    # Keyed by (cubes, holes) rather than the state object, so the cache
    # holds no state objects (keeps pickle / deepcopy of a state working).
    cache = _flat_board_state.h_cache
    key = (_flat_board_state.cubes, holes)
    cached = cache.get(key)
    if cached is not None:
        return cached
    cols = _flat_board_state.cols
    dist = _flat_board_state.dist
    inf = math.inf

    # flat cell index of every cube, plus running per-cube minima
    cube_idx = [cr * cols + cc for cr, cc in _flat_board_state.cubes]
    n = len(cube_idx)
    if n == 0:                          # holes left but no cubes: unsolvable
        return inf
    cube_r = [inf] * n
    cube_l = [inf] * n
    cube_u = [inf] * n
    cube_d = [inf] * n

    bfs_term = 0
    hole_r = hole_l = hole_u = hole_d = 0
    for hole in holes:
        table = dist[hole]
        best = inf                      # nearest cube by BFS
        mr = ml = mu = md = inf         # per-hole minima per direction
        for i in range(n):
            dv, need_r, need_l, need_u, need_d = table[cube_idx[i]]
            if dv < best:
                best = dv
            # hole side: min over cubes
            if need_r < mr: mr = need_r
            if need_l < ml: ml = need_l
            if need_u < mu: mu = need_u
            if need_d < md: md = need_d
            # cube side: min over holes
            if need_r < cube_r[i]: cube_r[i] = need_r
            if need_l < cube_l[i]: cube_l[i] = need_l
            if need_u < cube_u[i]: cube_u[i] = need_u
            if need_d < cube_d[i]: cube_d[i] = need_d
        if best > bfs_term: bfs_term = best
        if mr > hole_r: hole_r = mr
        if ml > hole_l: hole_l = ml
        if mu > hole_u: hole_u = mu
        if md > hole_d: hole_d = md

    directional_term = (max(hole_r, max(cube_r)) + max(hole_l, max(cube_l))
                        + max(hole_u, max(cube_u)) + max(hole_d, max(cube_d)))

    result = bfs_term if bfs_term > directional_term else directional_term
    cache[key] = result
    return result
