import heapq

from search_node import search_node
from flat_board_state import flat_board_state

count_generated = 0
count_expanded = 0

# open_set = (heap, lookup)
#   heap   : list of search_node, ordered by search_node.__lt__ (f, then h)
#   lookup : dict  state -> the node currently in open for that state
# Removing from the middle of a heap is slow, so a removed node is only
# dropped from `lookup`; its heap entry becomes stale and is skipped by get_best.
def create_open_set():
    return ([], {})


# closed_set : dict  state -> g with which that state was expanded
def create_closed_set():
    return {}


def add_to_open(vn, open_set):
    heap, lookup = open_set
    heapq.heappush(heap, vn)
    lookup[vn.state] = vn


def open_not_empty(open_set):
    # count only live nodes, not stale heap entries
    return len(open_set[1]) > 0


def get_best(open_set):
    heap, lookup = open_set
    while heap:
        node = heapq.heappop(heap)
        if lookup.get(node.state) is node:      # live entry
            del lookup[node.state]
            return node
        # otherwise stale (was replaced by a better node) -> skip
    return None


def add_to_closed(vn, closed_set):
    closed_set[vn.state] = vn.g

#returns False if curr_neighbor state not in open_set or has a lower g from the node in open_set
#remove the node with the higher g from open_set (if exists)
def duplicate_in_open(vn, open_set):
    lookup = open_set[1]
    existing = lookup.get(vn.state)
    if existing is None:
        return False
    if existing.g <= vn.g:
        return True
    del lookup[vn.state]        # heap entry left behind as stale
    return False

#returns False if curr_neighbor state not in closed_set or has a lower g from the node in closed_set
#remove the node with the higher g from closed_set (if exists)
def duplicate_in_closed(vn, closed_set):
    existing_g = closed_set.get(vn.state)
    if existing_g is None:
        return False
    if existing_g <= vn.g:
        return True
    del closed_set[vn.state]    # reopen: the new node will be added to open
    return False


# helps to debug sometimes..
def print_path(path):
    for i in range(len(path)-1):
        print(f"[{path[i].state.get_state_str()}]", end=", ")
    print(path[-1].state.state_str)

def get_gen_exp_count():
    global count_generated
    global count_expanded
    return count_generated, count_expanded

def search(start_state, heuristic):
    global count_generated
    global count_expanded
    count_generated = 0
    count_expanded = 0
    open_set = create_open_set()
    closed_set = create_closed_set()
    start_node = search_node(start_state, 0, heuristic(start_state))
    add_to_open(start_node, open_set)

    while open_not_empty(open_set):

        current = get_best(open_set)

        if flat_board_state.is_goal_state(current.state):
            path = []
            while current:
                path.append(current)
                current = current.prev
            path.reverse()
            return path

        add_to_closed(current, closed_set)
        count_expanded += 1

        for neighbor, edge_cost in current.get_neighbors():
            curr_neighbor = search_node(neighbor, current.g + edge_cost, heuristic(neighbor), current)
            if not duplicate_in_open(curr_neighbor, open_set) and not duplicate_in_closed(curr_neighbor, closed_set):
                add_to_open(curr_neighbor, open_set)
                count_generated += 1

    return None




