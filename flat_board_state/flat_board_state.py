# you can add global params
import math
from collections import deque

HOLE = 0
FLOOR = 1
WALL = 2
CUBE = 3
VALID_VALUES = (HOLE, FLOOR, WALL, CUBE)

INF = math.inf

# (row step, col step) for the 4 moves: up, down, right, left
DIRECTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))


class _Board:
    """Everything that never changes during the search, built once per puzzle
    and shared by reference between all states of that puzzle.

    Cells are addressed by a flat index  i = r * cols + c.
    """
    __slots__ = ("rows", "cols", "row_lengths", "walls", "padding", "coords",
                 "step", "reverse_order", "dist", "key", "_initial", "h_cache")

    def __init__(self, instance_map):
        rows = len(instance_map)
        row_lengths = [len(row) for row in instance_map]
        cols = max(row_lengths) if rows else 0
        walls, padding = set(), set()
        cubes, holes = [], []
        for r, row in enumerate(instance_map):
            for c in range(cols):
                i = r * cols + c
                if c >= len(row):
                    # ragged row: the missing cells are treated as wall
                    walls.add(i)
                    padding.add(i)
                    continue
                value = row[c]
                if value not in VALID_VALUES:
                    raise ValueError(f"invalid cell value {value!r} at ({r}, {c}); "
                                     f"expected one of {VALID_VALUES}")
                if value == WALL:
                    walls.add(i)
                elif value == CUBE:
                    cubes.append(i)
                elif value == HOLE:
                    holes.append(i)

        self.rows = rows
        self.cols = cols
        self.row_lengths = row_lengths
        self.walls = frozenset(walls)
        self.padding = frozenset(padding)
        self.coords = [divmod(i, cols) for i in range(rows * cols)] if cols else []
        # step[d][i] = index of the cell reached from i in direction d,
        #              or -1 if that is a wall or outside the board
        self.step = []
        for dr, dc in DIRECTIONS:
            table = [-1] * (rows * cols)
            for i, (r, c) in enumerate(self.coords):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    j = nr * cols + nc
                    if j not in self.walls:
                        table[i] = j
            self.step.append(table)
        # cubes in front must move first: process in descending flat index
        # when moving down / right, ascending when moving up / left
        self.reverse_order = tuple(dr > 0 or dc > 0 for dr, dc in DIRECTIONS)
        # dist[hole][cell] = (bfs, need_right, need_left, need_up, need_down)
        self.dist = {h: self._hole_table(h) for h in holes}
        self.key = (rows, cols, self.walls)
        self._initial = (frozenset(cubes), frozenset(holes))
        # (cubes, holes) -> advanced heuristic value, shared by all states of
        # this board, so a state reached again (duplicate) is not evaluated twice
        self.h_cache = {}

    def _neighbours(self, i):
        """Non-wall cells adjacent to i, with the direction (dr, dc) of the
        step that goes FROM that neighbour TO i."""
        out = []
        for d, (dr, dc) in enumerate(DIRECTIONS):
            j = self.step[d][i]
            if j >= 0:
                out.append((j, -dr, -dc))
        return out

    def _hole_table(self, hole):
        n = self.rows * self.cols
        adj = [self._neighbours(i) for i in range(n)]
        bfs = self._bfs(hole, adj)
        right = self._direction_bfs(hole, adj, 0, 1)
        left = self._direction_bfs(hole, adj, 0, -1)
        up = self._direction_bfs(hole, adj, -1, 0)
        down = self._direction_bfs(hole, adj, 1, 0)
        return list(zip(bfs, right, left, up, down))

    def _bfs(self, hole, adj):
        """Shortest wall-respecting walk from every cell to `hole`."""
        table = [INF] * (self.rows * self.cols)
        table[hole] = 0
        queue = deque([hole])
        while queue:
            i = queue.popleft()
            nxt = table[i] + 1
            for j, _, _ in adj[i]:
                if table[j] == INF:
                    table[j] = nxt
                    queue.append(j)
        return table

    def _direction_bfs(self, hole, adj, dr, dc):
        """For every cell: the fewest steps in direction (dr, dc) that any
        wall-respecting walk from that cell to `hole` must contain.
        0-1 BFS from the hole: a step in direction (dr, dc) costs 1, others 0."""
        table = [INF] * (self.rows * self.cols)
        table[hole] = 0
        queue = deque([hole])
        while queue:
            i = queue.popleft()
            cur = table[i]
            for j, sr, sc in adj[i]:
                # the cube would step from j to i, i.e. in direction (sr, sc)
                cost = 1 if (sr == dr and sc == dc) else 0
                if cur + cost < table[j]:
                    table[j] = cur + cost
                    if cost:
                        queue.append(j)
                    else:
                        queue.appendleft(j)
        return table


class flat_board_state:
    __slots__ = ("board", "cubes", "holes", "_hash", "_str")

    def __init__(self, instance_map=None, *args, **kwargs):
        # you can use the init function for several purposes
        #
        # Two ways to build a state:
        #   1. From a board (list of lists of 0-3): the initial state.
        #      The board is scanned once into a shared _Board (walls, move
        #      tables, per-hole distance tables).
        #   2. From a parent state (kwargs: parent, cubes, holes): a neighbour.
        #      The _Board is shared by reference, only cubes/holes are new.
        #      cubes / holes are flat cell indices (r * cols + c).
        parent = kwargs.get("parent")
        if parent is not None:
            self.board = parent.board
            self.cubes = frozenset(kwargs["cubes"])
            self.holes = frozenset(kwargs["holes"])
        else:
            if instance_map is None:
                raise ValueError("flat_board_state needs a board (list of lists)")
            self.board = _Board(instance_map)
            self.cubes, self.holes = self.board._initial
        # walls never change, so cubes + holes fully describe the state
        self._hash = hash((self.cubes, self.holes))
        self._str = None

    # convenience accessors
    @property
    def rows(self):
        return self.board.rows

    @property
    def cols(self):
        return self.board.cols

    @staticmethod
    def is_goal_state(_flat_board_state):
        # goal: no holes left on the board
        return not _flat_board_state.holes

    def get_neighbors(self):
        neighbors = []
        for d in range(4):
            new_state = self._apply_move(d)
            if new_state is not None:          # skip moves that change nothing
                neighbors.append((new_state, 1))
        return neighbors

    def _apply_move(self, d):
        """Shift every cube one cell in direction DIRECTIONS[d].
        Returns the new state, or None if no cube moved."""
        step = self.board.step[d]
        holes = self.holes
        new_cubes = set()
        filled = []
        moved = False
        # cubes in front go first, so a cube behind sees where they ended up
        for i in sorted(self.cubes, reverse=self.board.reverse_order[d]):
            j = step[i]
            if j < 0 or j in new_cubes:        # wall / edge / stuck cube ahead
                new_cubes.add(i)
            elif j in holes:                   # falls into a hole and fills it
                filled.append(j)
                moved = True
            else:                              # free cell
                new_cubes.add(j)
                moved = True
        if not moved:
            return None
        new_holes = holes.difference(filled) if filled else holes
        return flat_board_state(parent=self, cubes=new_cubes, holes=new_holes)

    # you can change the body of the function if you want
    def __hash__(self):
        return self._hash

    # you can change the body of the function if you want
    def __eq__(self, other):
        if not isinstance(other, flat_board_state):
            return NotImplemented
        if self is other:
            return True
        return (self._hash == other._hash
                and self.cubes == other.cubes and self.holes == other.holes
                and (self.board is other.board or self.board.key == other.board.key))

    def get_state_board(self):
        b = self.board
        cols = b.cols
        grid = [[FLOOR] * cols for _ in range(b.rows)]
        for i in b.walls:
            r, c = divmod(i, cols)
            grid[r][c] = WALL
        for i in self.holes:
            r, c = divmod(i, cols)
            grid[r][c] = HOLE
        for i in self.cubes:
            r, c = divmod(i, cols)
            grid[r][c] = CUBE
        if b.padding:                          # restore the original ragged rows
            grid = [row[:n] for row, n in zip(grid, b.row_lengths)]
        return grid

    def get_state_str(self):
        if self._str is None:                  # built once, then cached
            self._str = "\n".join(" ".join(str(v) for v in row)
                                  for row in self.get_state_board())
        return self._str

    # print_path in search.py reads `state.state_str` as an attribute
    @property
    def state_str(self):
        return self.get_state_str()

    # you can add helper functions
