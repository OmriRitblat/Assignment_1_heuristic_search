# you can add global params
import math
from collections import deque

HOLE = 0
FLOOR = 1
WALL = 2
CUBE = 3

# (row step, col step) for the 4 moves: up, down, right, left
DIRECTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def build_bfs_table(walls, rows, cols, hole):
    """Distance from every cell to `hole`, walking around walls.
    Flat list: table[r * cols + c]. Walls and unreachable cells get math.inf."""
    table = [math.inf] * (rows * cols)
    hr, hc = hole
    table[hr * cols + hc] = 0
    queue = deque([hole])
    while queue:
        r, c = queue.popleft()
        step = table[r * cols + c] + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols \
               and (nr, nc) not in walls and table[nr * cols + nc] == math.inf:
                table[nr * cols + nc] = step
                queue.append((nr, nc))
    return table


def build_direction_table(walls, rows, cols, hole, dr, dc):
    """For every cell: the fewest moves in direction (dr, dc) that any
    wall-respecting path from that cell to `hole` must contain.
    0-1 BFS from the hole: a step in direction (dr, dc) costs 1, others 0."""
    table = [math.inf] * (rows * cols)
    hr, hc = hole
    table[hr * cols + hc] = 0
    queue = deque([hole])
    while queue:
        r, c = queue.popleft()
        cur = table[r * cols + c]
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in walls:
                continue
            # the cube would step from (nr, nc) to (r, c): direction (r-nr, c-nc)
            cost = 1 if (r - nr == dr and c - nc == dc) else 0
            idx = nr * cols + nc
            if cur + cost < table[idx]:
                table[idx] = cur + cost
                if cost == 0:
                    queue.appendleft((nr, nc))
                else:
                    queue.append((nr, nc))
    return table


def build_hole_table(walls, rows, cols, hole):
    """One tuple per cell: (bfs_dist, min_right, min_left, min_up, min_down)."""
    d = build_bfs_table(walls, rows, cols, hole)
    r = build_direction_table(walls, rows, cols, hole, 0, 1)
    l = build_direction_table(walls, rows, cols, hole, 0, -1)
    u = build_direction_table(walls, rows, cols, hole, -1, 0)
    dn = build_direction_table(walls, rows, cols, hole, 1, 0)
    return list(zip(d, r, l, u, dn))


class flat_board_state:
    def __init__(self, instance_map=None, *args, **kwargs):
        # you can use the init function for several purposes
        #
        # Two ways to build a state:
        #   1. From a board (list of lists of 0-3): the initial state.
        #      We scan the grid once, collect walls / cubes / holes, and
        #      precompute one BFS distance table per hole.
        #   2. From a parent state (kwargs: parent, cubes, holes): a neighbor.
        #      Static data (size, walls, BFS tables) is shared by reference,
        #      only the cube and hole sets are new.
        parent = kwargs.get("parent")
        if parent is not None:
            self.rows = parent.rows
            self.cols = parent.cols
            self.walls = parent.walls
            self.dist = parent.dist
            self.h_cache = parent.h_cache
            self.cubes = frozenset(kwargs["cubes"])
            self.holes = frozenset(kwargs["holes"])   # no copy if already a frozenset
        else:
            self.rows = len(instance_map)
            self.cols = len(instance_map[0]) if self.rows else 0
            walls, cubes, holes = [], [], []
            for r, row in enumerate(instance_map):
                for c, value in enumerate(row):
                    if value == WALL:
                        walls.append((r, c))
                    elif value == CUBE:
                        cubes.append((r, c))
                    elif value == HOLE:
                        holes.append((r, c))
            self.walls = frozenset(walls)
            self.cubes = frozenset(cubes)
            self.holes = frozenset(holes)
            # dist[hole][r * cols + c] = (bfs_dist, min_right, min_left, min_up, min_down)
            # from cell (r, c) to that hole, walls respected
            self.dist = {hole: build_hole_table(self.walls, self.rows, self.cols, hole)
                         for hole in self.holes}
            # state -> heuristic value, shared by all states of this board, so a
            # state reached again (duplicate) is not evaluated twice
            self.h_cache = {}

        # cubes + holes fully describe the state (walls never change)
        self._hash = hash((self.cubes, self.holes))

    @staticmethod
    def is_goal_state(_flat_board_state):
        # goal: no holes left on the board
        return not _flat_board_state.holes

    def get_neighbors(self):
        neighbors = []
        for dr, dc in DIRECTIONS:
            new_state = self._apply_move(dr, dc)
            if new_state is not None:          # skip moves that change nothing
                neighbors.append((new_state, 1))
        return neighbors

    def _apply_move(self, dr, dc):
        """Shift every cube one cell in direction (dr, dc).
        Returns the new state, or None if no cube moved."""
        rows, cols, walls = self.rows, self.cols, self.walls

        # cubes in front go first: sort by the coordinate along the move,
        # descending when moving right/down, ascending when moving left/up
        if dr != 0:
            order = sorted(self.cubes, key=lambda p: p[0], reverse=(dr > 0))
        else:
            order = sorted(self.cubes, key=lambda p: p[1], reverse=(dc > 0))

        holes = self.holes
        new_cubes = set()
        filled = []          # holes filled by this move (two cubes never share a target)
        moved = False
        for r, c in order:
            nr, nc = r + dr, c + dc
            if (nr < 0 or nr >= rows or nc < 0 or nc >= cols   # edge
                    or (nr, nc) in walls                       # wall
                    or (nr, nc) in new_cubes):                 # stuck cube ahead
                new_cubes.add((r, c))
            elif (nr, nc) in holes:                            # falls into hole
                filled.append((nr, nc))
                moved = True
            else:                                              # free cell
                new_cubes.add((nr, nc))
                moved = True

        if not moved:
            return None
        # most moves fill nothing: reuse the parent's holes frozenset, no copy
        new_holes = holes.difference(filled) if filled else holes
        return flat_board_state(parent=self, cubes=new_cubes, holes=new_holes)

    # you can change the body of the function if you want
    def __hash__(self):
        return self._hash

    # you can change the body of the function if you want
    def __eq__(self, other):
        # walls are static, so same cubes + same holes == same board
        return self.cubes == other.cubes and self.holes == other.holes

    def get_state_board(self):
        # rebuild the grid: everything is floor except walls, holes and cubes
        board = [[FLOOR] * self.cols for _ in range(self.rows)]
        for r, c in self.walls:
            board[r][c] = WALL
        for r, c in self.holes:
            board[r][c] = HOLE
        for r, c in self.cubes:
            board[r][c] = CUBE
        return board

    def get_state_str(self):
        return "\n".join(" ".join(str(v) for v in row) for row in self.get_state_board())

    # print_path in search.py reads `state.state_str` as an attribute
    @property
    def state_str(self):
        return self.get_state_str()

    # you can add helper functions
